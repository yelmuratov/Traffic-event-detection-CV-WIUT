"""Annotated videos, event timelines and risk curves for the website (P3).

  python -m tools.render samples/a.mp4 outputs/render --pred predictions_samples.json [--gt my_labels.json]

Output: <stem>_annotated.mp4 (H.264, web-ready), <stem>_timeline.png, <stem>_events.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import CLASSES  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.video import iter_frames, probe  # noqa: E402

CLS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLS_COL = {0: (0, 200, 255), 1: (255, 128, 0), 2: (0, 255, 0), 3: (255, 0, 255), 5: (255, 255, 0), 7: (0, 128, 255)}
EV_COL = {c: tuple(int(255 * v) for v in plt.cm.tab20(i / len(CLASSES))[:3][::-1]) for i, c in enumerate(CLASSES)}


def timeline_png(events, duration, out, risk=None, gt=None, title=""):
    rows = 2 if risk else 1
    fig, axes = plt.subplots(rows, 1, figsize=(14, 2.2 + 0.35 * len(CLASSES)), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1][:rows]})
    ax = axes[0] if rows == 2 else axes
    for i, c in enumerate(CLASSES):
        segs = [(s, e - s) for s, e, l in events if l == c]
        ax.broken_barh(segs, (i - 0.35, 0.3 if gt else 0.7), color=plt.cm.tab20(i / len(CLASSES)))
        if gt:
            gsegs = [(s, e - s) for s, e, l in gt if l == c]
            ax.broken_barh(gsegs, (i + 0.05, 0.3), color="k", alpha=0.5)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES); ax.set_xlim(0, duration)
    ax.set_title(title + ("   (colour = predicted, grey = ground truth)" if gt else "")); ax.grid(axis="x", alpha=0.3)
    if risk:
        r = np.asarray(risk)
        axes[1].plot(r[:, 0], r[:, 1], color="#c0392b", lw=1); axes[1].axhline(0.5, ls="--", color="grey", lw=0.8)
        for s, e, l in (gt or []):
            if l == "accident":
                axes[1].axvspan(s - 5, s, color="orange", alpha=0.3)
        axes[1].set_ylim(0, 1); axes[1].set_ylabel("risk"); axes[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def render(video, out_dir, events=None, risk=None, gt=None, width=1280, draw_scene=True):
    os.makedirs(out_dir, exist_ok=True)
    meta, ctx = analyse(video)
    if events is None:
        from src.postprocess import postprocess
        from src.rules import run_all
        events = postprocess(run_all(ctx), meta.duration)
    stem = os.path.splitext(meta.name)[0]
    df = ctx.df
    by_frame = {f: g for f, g in df.groupby("frame")}
    frames_sorted = np.array(sorted(by_frame))
    risk_arr = np.asarray(risk) if risk else None
    s = width / meta.width
    H = int(round(meta.height * s / 2) * 2)
    tmp = os.path.join(out_dir, f"{stem}_tmp.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (width, H))
    scene_layer = None
    for fi, t, frame, _ in iter_frames(video, stride=1):
        img = cv2.resize(frame, (width, H), interpolation=cv2.INTER_AREA)
        if draw_scene and scene_layer is None:
            scene_layer = cv2.resize(ctx.scene.draw(np.zeros_like(frame), alpha=1.0), (width, H))
        if draw_scene:
            mask = scene_layer.any(2)
            img[mask] = (0.75 * img[mask] + 0.25 * scene_layer[mask]).astype(np.uint8)
        # latest processed frame at or before fi (tracks are sampled every `stride` frames)
        k = np.searchsorted(frames_sorted, fi, side="right") - 1
        if k >= 0 and fi - frames_sorted[k] <= 3:
            for r in by_frame[frames_sorted[k]].itertuples():
                p1 = (int(r.x1 * s), int(r.y1 * s)); p2 = (int(r.x2 * s), int(r.y2 * s))
                col = CLS_COL.get(r.cls, (200, 200, 200))
                cv2.rectangle(img, p1, p2, col, 2)
                cv2.putText(img, f"{CLS_NAMES.get(r.cls, r.cls)} {r.tid} {r.speed_n:.1f}", (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        active = [l for a, b, l in events if a <= t <= b]
        y = 30
        for l in active:
            cv2.rectangle(img, (10, y - 22), (10 + 14 * len(l) + 20, y + 8), EV_COL[l], -1)
            cv2.putText(img, l.upper(), (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 36
        cv2.putText(img, f"t={t:6.1f}s", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if risk_arr is not None and len(risk_arr):
            j = min(np.searchsorted(risk_arr[:, 0], t), len(risk_arr) - 1)
            rv = float(risk_arr[j, 1])
            cv2.rectangle(img, (width - 170, 45), (width - 20, 65), (60, 60, 60), -1)
            cv2.rectangle(img, (width - 170, 45), (width - 170 + int(150 * rv), 65),
                          (0, 0, 255) if rv >= 0.5 else (0, 200, 255), -1)
            cv2.putText(img, f"risk {rv:.2f}", (width - 170, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        wr.write(img)
    wr.release()
    out = os.path.join(out_dir, f"{stem}_annotated.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    os.remove(tmp)
    timeline_png(events, meta.duration, os.path.join(out_dir, f"{stem}_timeline.png"), risk, gt, meta.name)
    json.dump({"video": meta.name, "duration": meta.duration, "events": events,
               "risk": risk_arr[::5].round(3).tolist() if risk_arr is not None else []},
              open(os.path.join(out_dir, f"{stem}_events.json"), "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--pred"); ap.add_argument("--gt"); ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    name = os.path.basename(a.video)
    ev = risk = gt = None
    if a.pred:
        v = json.load(open(a.pred))["videos"].get(name, {})
        ev, risk = v.get("events"), v.get("risk")
    if a.gt:
        gt = json.load(open(a.gt)).get(name, {}).get("events")
    print(render(a.video, a.out, ev, risk, gt, a.width))
