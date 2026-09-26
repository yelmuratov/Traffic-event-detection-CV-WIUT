"""Detection + ByteTrack tracking (P1). One pass over the video produces:

  tracks  : float32 array, columns TRACK_COLS, boxes in ORIGINAL pixel coordinates
  signals : {signal_name: (times, states)} traffic-light state per sampled frame

Results can be cached on disk (WIUT_CACHE) so rule development never re-runs the GPU.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .utils import set_seed
from .video import VideoMeta, iter_frames, iter_frames_fast, video_key

log = logging.getLogger("wiut.tracker")
TRACK_COLS = ["frame", "t", "tid", "cls", "conf", "x1", "y1", "x2", "y2"]

_MODELS: dict = {}


def _tracker_yaml(overrides: dict) -> str:
    """Copy ultralytics' own bytetrack.yaml and apply overrides (robust across versions)."""
    import yaml
    from ultralytics.utils import ROOT as UL_ROOT
    base = yaml.safe_load(open(UL_ROOT / "cfg" / "trackers" / "bytetrack.yaml"))
    base.update(overrides)
    key = hashlib.md5(json.dumps(base, sort_keys=True).encode()).hexdigest()[:8]
    path = Path(tempfile.gettempdir()) / f"wiut_bytetrack_{key}.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump(base))
    return str(path)


def load_model(weights: str):
    """Load YOLO weights from weights/ (offline). Cached per process."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO
    local = config.WEIGHTS_DIR / weights
    src = str(local) if local.exists() else weights  # falls back to auto-download in dev only
    if src not in _MODELS:
        _MODELS[src] = YOLO(src)
    return _MODELS[src]


def new_model(weights: str):
    """A separate model instance (own tracker state) - used by Part B."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO
    local = config.WEIGHTS_DIR / weights
    return YOLO(str(local) if local.exists() else weights)


def reset_tracker(model) -> None:
    pred = getattr(model, "predictor", None)
    if pred is not None and getattr(pred, "trackers", None):
        for t in pred.trackers:
            t.reset()


def results_to_rows(res, frame_idx: int, t: float, inv_scale: float) -> list:
    boxes = res.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy() / inv_scale
    ids = boxes.id.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return [[frame_idx, t, ids[k], cls[k], conf[k], *xyxy[k]] for k in range(len(ids))]


def _cache_path(kind: str, vkey: str, cfg: dict) -> Path | None:
    if not config.CACHE_DIR:
        return None
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    d = Path(config.CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{vkey}_{h}.npz"


def analyse_video(meta: VideoMeta, scene=None, want_tracks: bool = True, progress: bool = False):
    """Run detector+tracker (and read signal ROIs) in one decoding pass, with caching."""
    from .scene import signal_state
    det = config.DETECTOR
    signals_cfg = {k: {kk: list(map(int, vv)) for kk, vv in v.items()} for k, v in (scene.signals if scene else {}).items()}
    vkey = video_key(meta.path)
    det_key = {k: v for k, v in det.items() if not (k == "det_every" and v == 1)}  # default keeps old cache keys
    tr_cfg = {"det": det_key, "trk": config.TRACKER, "classes": config.DET_CLASSES, "thumbs": config.THUMBS}
    tr_cache = _cache_path("tracks", vkey, tr_cfg)
    sg_cache = _cache_path("signals", vkey, {"sig": signals_cfg, "stride": det["stride"],
                                             "decoder": det.get("decoder"), "w": det["max_width"]})

    tracks = signals = thumbs = None
    if tr_cache and tr_cache.exists():
        z = np.load(tr_cache)
        tracks = z["tracks"]
        if "thumbs" in z:
            thumbs = (z["thumb_t"], z["thumbs"])
    if sg_cache and sg_cache.exists():
        z = np.load(sg_cache, allow_pickle=True)
        signals = z["signals"].item()
    if not signals_cfg:
        signals = {}
    need_tracks = want_tracks and tracks is None
    need_signals = signals is None
    if not need_tracks and not need_signals:
        return (tracks if tracks is not None else np.zeros((0, len(TRACK_COLS)), np.float32)), \
            _with_thumbs(signals, thumbs)

    set_seed(config.SEED)
    model = load_model(det["weights"]) if need_tracks else None
    if model is not None:
        reset_tracker(model)
    trk_yaml = _tracker_yaml(config.TRACKER) if need_tracks else None
    rows: list = []
    sig_t: list = []
    sig_s: dict = {k: [] for k in signals_cfg}
    batch: list = []
    thumb_t: list = []
    thumb_img: list = []
    tw, th_ = config.THUMBS["width"], config.THUMBS["height"]
    t0 = time.time()

    def flush():
        if not batch:
            return
        res = model.track([b[2] for b in batch], persist=True, tracker=trk_yaml, classes=config.DET_CLASSES,
                          conf=det["conf"], iou=det["iou"], imgsz=det["imgsz"], half=config.HALF,
                          device=config.DEVICE, verbose=False)
        for (fi, t, _, s), r in zip(batch, res):
            rows.extend(results_to_rows(r, fi, t, s))
        batch.clear()

    # signals need full-res crops, so keep full frames and let the model resize
    if det.get("decoder") == "pyav_ref":
        frames = iter_frames_fast(meta.path, max_width=det["max_width"])
    else:
        frames = iter_frames(meta.path, stride=det["stride"], max_width=det["max_width"])
    det_every = max(1, int(det.get("det_every", 1)))
    prof = {"decode": 0.0, "model": 0.0, "signals": 0.0, "thumbs": 0.0}
    frames = iter(frames)
    n_dec = -1
    while True:
        tq = time.perf_counter()
        item = next(frames, None)
        prof["decode"] += time.perf_counter() - tq          # time spent waiting for the decoder
        if item is None:
            break
        fi, t, frame, s = item
        n_dec += 1
        tq = time.perf_counter()
        if need_signals:
            sig_t.append(t)
            for k, sc in scene.signals.items():
                # signal ROIs are in full-resolution pixels; the frame may be downscaled by s
                sig_s[k].append(signal_state(frame, {kk: (np.asarray(vv) * s).round().astype(int) for kk, vv in sc.items()}))
        prof["signals"] += time.perf_counter() - tq
        if need_tracks:
            tq = time.perf_counter()
            if not thumb_t or t - thumb_t[-1] >= config.THUMBS["every_s"]:
                # small grey background samples for the road-obstacle detector
                thumb_t.append(t)
                thumb_img.append(cv2.cvtColor(cv2.resize(frame, (tw, th_), interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY))
            prof["thumbs"] += time.perf_counter() - tq
            if n_dec % det_every == 0:          # detector on every det_every-th decoded frame
                batch.append((fi, t, frame, s))
                if len(batch) >= det["batch"]:
                    tq = time.perf_counter()
                    flush()
                    prof["model"] += time.perf_counter() - tq
        if progress and fi % (det["stride"] * 500) == 0:
            log.info("%s  t=%.0fs  %.1f fps", meta.name, t, (fi + 1) / max(time.time() - t0, 1e-6))
    if need_tracks:
        flush()
        tracks = np.asarray(rows, np.float32).reshape(-1, len(TRACK_COLS))
        thumbs = (np.asarray(thumb_t, np.float32), np.stack(thumb_img) if thumb_img else np.zeros((0, th_, tw), np.uint8))
        if tr_cache:
            np.savez_compressed(tr_cache, tracks=tracks, thumb_t=thumbs[0], thumbs=thumbs[1])
    if need_signals:
        signals = {k: (np.asarray(sig_t, np.float32), np.asarray(v, np.int8)) for k, v in sig_s.items()}
        if sg_cache:
            np.savez_compressed(sg_cache, signals=np.array(signals, dtype=object))
    log.info("%s analysed in %.1fs (%.2fx realtime); %d frames; time split: %s", meta.name, time.time() - t0,
             (time.time() - t0) / max(meta.duration, 1e-6), n_dec + 1,
             ", ".join(f"{k} {v:.0f}s" for k, v in prof.items()))
    if tracks is None:
        tracks = np.zeros((0, len(TRACK_COLS)), np.float32)
    return tracks, _with_thumbs(signals, thumbs)


def _with_thumbs(signals: dict | None, thumbs) -> dict:
    """Signals dict plus the background thumbnails under the reserved key '__thumbs__'."""
    out = dict(signals or {})
    if thumbs is not None:
        out["__thumbs__"] = thumbs
    return out
