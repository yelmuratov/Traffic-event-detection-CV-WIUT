"""Wall-clock budget shared by Part A and Part B.

The harness gives each video 3 x its duration for detect_events + the whole RiskEstimator pass,
and a video over budget scores as EMPTY (Part A events included). Part A marks the start; Part B
watches its own pace and thins out detection when the projected finish gets too close to the limit.
"""
from __future__ import annotations

import os
import time

TIME_FACTOR = 3.0   # official budget multiplier
SAFETY = 0.90       # aim to finish within 90 % of the budget

_START: dict[str, float] = {}


def mark_start(video_path: str) -> None:
    """Called at the top of detect_events: the harness clock starts right before it."""
    _START[os.path.basename(video_path)] = time.perf_counter()


def elapsed(video_id: str) -> float | None:
    t0 = _START.get(video_id)
    return None if t0 is None else time.perf_counter() - t0
