"""Video probing and fast strided frame reading with a background decode thread."""
from __future__ import annotations

import hashlib
import os
import queue
import threading
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    name: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(path=path, name=os.path.basename(path), fps=float(fps), width=w, height=h,
                     n_frames=max(n, 0), duration=max(n, 0) / float(fps))


def video_key(path: str) -> str:
    """Cheap content key: name + size + first/last MB hash (no full read of multi-GB files)."""
    st = os.stat(path)
    h = hashlib.md5(f"{os.path.basename(path)}:{st.st_size}".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, st.st_size - (1 << 20)))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def resize_to_width(frame: np.ndarray, max_width: int | None) -> tuple[np.ndarray, float]:
    """Downscale so width <= max_width. Returns (frame, scale) where original = resized / scale."""
    if not max_width or frame.shape[1] <= max_width:
        return frame, 1.0
    s = max_width / frame.shape[1]
    out = cv2.resize(frame, (max_width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return out, s


def iter_frames(path: str, stride: int = 1, max_width: int | None = None, start: int = 0,
                end: int | None = None, prefetch: int = 32):
    """Yield (frame_idx, t_sec, frame_bgr, scale) for every `stride`-th frame.

    Decoding runs in a background thread so the GPU never waits on the CPU decoder.
    Skipped frames use grab() (no colour conversion / copy)."""
    meta = probe(path)
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def worker():
        cap = cv2.VideoCapture(path)
        try:
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            idx = start
            while not stop.is_set():
                if end is not None and idx >= end:
                    break
                if (idx - start) % stride == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame, s = resize_to_width(frame, max_width)
                    q.put((idx, idx / meta.fps, frame, s))
                else:
                    if not cap.grab():
                        break
                idx += 1
        finally:
            cap.release()
            q.put(None)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    finally:
        stop.set()
        # drain so the worker can exit
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(timeout=0.05)


def read_frame_at(path: str, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
