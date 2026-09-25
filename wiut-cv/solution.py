"""WIUT Hackathon 2026 - CV track. Interface required by run_submission.py.

Part A: detector + ByteTrack + scene-map rules  (src/pipeline.py)
Part B: causal time-to-collision risk             (src/risk.py)
"""
import logging

import numpy as np

from src.pipeline import detect_events as _detect_events
from src.risk import RiskEngine
from src.utils import set_seed

set_seed(42)
logging.getLogger("wiut").setLevel(logging.INFO)

CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    return _detect_events(video_path)


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def __init__(self):
        self._engine = RiskEngine()

    def reset(self, meta: dict) -> None:
        # meta = {"video_id", "fps", "width", "height", "n_frames"}
        self._engine.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self._engine.step(frame, t_sec)
