"""Where does Part A spend its time?  (dev tool, runs in a clean process)

  python -m tools.profile samples/C3905.mp4 --decoder pyav_ref
  python -m tools.profile samples/C3905.mp4 --decoder opencv

Prints the pure decode speed on the first 30 s, then runs detect_events without the cache
and logs the split: decode wait / detector / traffic light / thumbnails / alignment / rules.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--decoder", default=config.DETECTOR["decoder"])
ap.add_argument("--seconds", type=float, default=30.0)
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
config.CACHE_DIR = None
config.DETECTOR["decoder"] = a.decoder
det = config.DETECTOR

import torch  # noqa: E402
from src.video import iter_frames, iter_frames_fast, probe  # noqa: E402

meta = probe(a.video)
print(f"cpu cores {os.cpu_count()}, torch threads {torch.get_num_threads()}, decoder {a.decoder}, "
      f"max_width {det['max_width']}", flush=True)
t0 = time.time(); n = 0
frames = iter_frames_fast(a.video, max_width=det["max_width"]) if a.decoder == "pyav_ref" else \
    iter_frames(a.video, stride=det["stride"], max_width=det["max_width"])
for fi, t, frame, s in frames:
    if t > a.seconds:
        break
    n += 1
dt = time.time() - t0
print(f"pure decode: {dt:.1f}s for {a.seconds:.0f}s of video = {dt / a.seconds:.2f}x ({n} frames)", flush=True)

from src.pipeline import detect_events  # noqa: E402
t0 = time.time()
ev = detect_events(a.video)
dt = time.time() - t0
print(f"detect_events: {dt:.0f}s = {dt / meta.duration:.2f}x realtime, {len(ev)} events", flush=True)
