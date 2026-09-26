"""Central configuration. Every threshold lives here so P2 can tune in one place.

Environment overrides (dev / demo only; the official run uses the defaults):
  WIUT_CACHE      directory for cached tracks (e.g. a Google Drive folder in Colab)
  WIUT_DEMO=1     light settings for CPU demo (smaller model, bigger stride)
  WIUT_DEVICE     "0" for GPU, "cpu" to force CPU
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
SCENE_PATH = ROOT / "scene" / "scene_map.json"
FLOW_MAP_PATH = ROOT / "scene" / "flow_map.npz"
REF_FRAME_PATH = ROOT / "scene" / "ref_frame.jpg"   # frame the scene map was drawn on

SEED = 42
DEMO = os.environ.get("WIUT_DEMO", "0") == "1"
CACHE_DIR = os.environ.get("WIUT_CACHE") or None


def _device() -> str:
    env = os.environ.get("WIUT_DEVICE")
    if env:
        return env
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _device()
HALF = DEVICE != "cpu"

# ---------------------------------------------------------------- detection (Part A)
# COCO ids: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
DET_CLASSES = [0, 1, 2, 3, 5, 7]
PERSON = 0
TWO_WHEELERS = {1, 3}
VEHICLES = {2, 3, 5, 7}          # motor vehicles used by most rules
ROAD_USERS = {0, 1, 2, 3, 5, 7}

DETECTOR = {
    "weights": "yolo11n.pt" if DEMO else "yolo11m.pt",
    "imgsz": 640 if DEMO else 1280,
    "conf": 0.25,
    "iou": 0.6,
    "decoder": "pyav_ref",           # PyAV, reference frames only (~every 3rd); faster than OpenCV here
    "stride": 5 if DEMO else 3,      # frame spacing (pyav_ref gives ~3; used by the OpenCV fallback)
    "batch": 1 if DEMO else 8,       # frames per GPU call
    "max_width": 1920,               # frames are downscaled to this width before the model
    "det_every": 1,                  # run the detector on every n-th decoded frame
}
THUMBS = {"every_s": 2.0, "width": 640, "height": 360}   # grey background samples (road_obstacle)
TRACKER = {"track_buffer": 50, "track_high_thresh": 0.3, "match_thresh": 0.8}

# ---------------------------------------------------------------- risk (Part B)
RISK = {
    "weights": "yolo11n.pt" if DEMO else "yolo11s.pt",
    "imgsz": 640 if DEMO else 800,
    "target_hz": 5 if DEMO else 6,   # detections per second of video
    "history_s": 0.8,                # window for velocity fit
    "tau_s": 1.5,                    # TTC decay constant: risk = exp(-ttc/tau)
    "max_ttc_s": 5.0,
    "min_closing": 0.6,              # min closing speed (box-heights/s) to count a pair
    "sigma": 0.5,                    # miss-distance tolerance (box-heights)
    "brake_drop": 0.5,               # speed drop fraction counted as hard braking
    "brake_bonus": 0.15,
    "ema": 0.4,                      # smoothing of the output score
    "gain": 1.0,                     # final calibration: score = clip(gain * raw)
}

# ---------------------------------------------------------------- kinematics
KIN = {
    "smooth_s": 0.4,        # rolling median window for positions
    "vel_s": 0.4,           # half-window for velocity differences
    "stationary": 0.12,     # speed below this (box-heights / s) = stationary
    "moving": 0.5,          # speed above this = clearly moving
    "min_track_s": 1.0,     # ignore tracks shorter than this
}

# ---------------------------------------------------------------- rules
# Classes we actually output. Predicting a class that is not in the test set
# costs a whole class worth of F1, so only enable what is validated on labels.
ENABLED = {
    # near_miss: box contacts / TTC in this dense view gave only false alarms on the samples -> off.
    # accident uses a separate strict rule (rule_accident) instead of the contact heuristic.
    "accident": True,         # strict rule: meet in the junction, both stay stopped while traffic flows
    "near_miss": False,
    "red_light": True,
    "wrong_way": True,
    "illegal_u_turn": True,
    "stopped_vehicle": True,
    "jaywalking": True,
    "failure_to_yield": True,
    "illegal_turn": True,
    "solid_line_crossing": True,
    "stop_line": True,
    "congestion": True,
    "road_obstacle": False,   # new static non-vehicle object on the road (background comparison)
    "fire_smoke": False,      # no reliable detector yet
}

RULES = {
    "stopped_vehicle": {"min_s": 10.0, "merge_gap_s": 3.0, "max_others": 2},
    "congestion": {"min_stopped": 4, "stat_speed": 0.3, "min_s": 12.0, "smooth_s": 8.0},  # stopped cars inside the junction box
    "wrong_way": {"cos": -0.5, "min_s": 1.5, "min_speed": 0.5, "flow_min_count": 20, "flow_min_coherence": 0.7},
    "red_light": {"red_before_s": 5.0, "max_s": 3.0, "min_speed": 0.8},
    "stop_line": {"max_after_stop_s": 90.0, "red_before_s": 0.3},
    "jaywalking": {"min_s": 2.0, "crosswalk_buffer_px": 150, "rider_overlap": 0.3, "min_speed": 0.3,
                   "min_conf": 0.6, "min_path_h": 0.0},
    "failure_to_yield": {"ped_buffer_px": 20, "min_vehicle_speed": 0.8, "near_w": 1.5, "ped_min_speed": 0.5},
    "solid_line": {"min_cross_px": 5, "min_cross_w": 0.1, "persist_s": 1.5, "min_speed": 0.8, "half_s": 1.5},
    "turn": {"turn_deg": 60, "u_turn_deg": 150, "window_s": 12.0, "onset_deg": 10, "max_turn_s": 6.0},
    "near_miss": {"ttc_s": 0.7, "decel_frac": 0.5, "swerve_deg": 25, "react_s": 1.5, "min_speed": 1.2},
    "accident_strict": {"stop_within_s": 3.0, "hold_s": 10.0, "min_speed_before": 1.0,
                        "flow_radius": 8.0, "flow_speed": 0.5, "min_flow_samples": 5},
    "road_obstacle": {"diff": 35, "persist_s": 10.0, "min_area": 20, "max_area": 2500,
                      "min_fill": 0.35, "max_aspect": 4.0, "link_px": 12, "box_pad": 0.3},
    "accident": {"contact_iou": 0.15, "depth_frac": 0.12, "min_closing": 0.5, "stop_within_s": 3.0,
                 "max_s": 30.0, "min_speed_before": 1.0, "hold_s": 1.0, "hold_frac": 0.6},
}

# Segment post-processing per class: merge gaps, minimum length, boundary shifts
POST_DEFAULT = {"merge_gap": 1.0, "min_len": 0.5, "start_shift": 0.0, "end_shift": 0.0, "max_len": None}
POST = {
    # start/end shifts tuned on our dev labels (annotators mark events ~1 s before the rule fires)
    "accident": {"merge_gap": 2.0, "min_len": 1.0},
    "stopped_vehicle": {"merge_gap": 3.0, "min_len": 10.0, "start_shift": -1.0, "end_shift": 1.5},
    "congestion": {"merge_gap": 5.0, "min_len": 10.0},
    "jaywalking": {"merge_gap": 1.5, "min_len": 1.0, "start_shift": -1.5, "end_shift": 0.5},
    "wrong_way": {"merge_gap": 1.5, "min_len": 1.5},
    "red_light": {"merge_gap": 0.3, "min_len": 0.5},
    "failure_to_yield": {"merge_gap": 1.5, "min_len": 1.5, "start_shift": -1.5, "end_shift": -1.0},
    "illegal_turn": {"start_shift": -1.5, "end_shift": -1.0},
    "solid_line_crossing": {"merge_gap": 0.5, "start_shift": -0.5, "end_shift": -0.5},
}


def post_cfg(label: str) -> dict:
    cfg = dict(POST_DEFAULT)
    cfg.update(POST.get(label, {}))
    return cfg
