"""Turn raw tracker rows into a kinematics table used by every rule.

Columns added:
  gx, gy     ground-contact point (bottom-centre of the box), smoothed
  w, h       box size (h smoothed per track) - used as a perspective-aware unit
  vx, vy     velocity in px/s;   speed_n = |v| / h   (box-heights per second)
  heading    atan2(vy, vx) in radians (image coordinates, y down)
  rider      person box that sits on a bicycle/motorcycle (not a pedestrian)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .tracker import TRACK_COLS


def build_table(tracks: np.ndarray, fps: float, stride: int | None = None) -> pd.DataFrame:
    stride = stride or config.DETECTOR["stride"]
    df = pd.DataFrame(tracks, columns=TRACK_COLS)
    if df.empty:
        for c in ["gx", "gy", "w", "h", "vx", "vy", "speed_n", "heading", "rider", "dur"]:
            df[c] = pd.Series(dtype=float)
        return df
    df[["frame", "tid", "cls"]] = df[["frame", "tid", "cls"]].astype(int)
    # a track id can flip class between frames (car/truck): use its majority class
    df["cls"] = df.groupby("tid")["cls"].transform(lambda s: s.mode().iloc[0])
    df = df.sort_values(["tid", "frame"]).reset_index(drop=True)
    df["w"] = df["x2"] - df["x1"]
    df["h_raw"] = df["y2"] - df["y1"]
    df["gx_raw"] = (df["x1"] + df["x2"]) / 2
    df["gy_raw"] = df["y2"]

    dt_sample = stride / fps
    win = max(3, int(round(config.KIN["smooth_s"] / dt_sample)) | 1)
    k = max(1, int(round(config.KIN["vel_s"] / dt_sample)))
    g = df.groupby("tid", sort=False)
    df["gx"] = g["gx_raw"].transform(lambda s: s.rolling(win, center=True, min_periods=1).median())
    df["gy"] = g["gy_raw"].transform(lambda s: s.rolling(win, center=True, min_periods=1).median())
    df["h"] = g["h_raw"].transform(lambda s: s.rolling(win * 2 + 1, center=True, min_periods=1).median())

    def central_diff(col):
        return g[col].transform(lambda s: _diff(s.to_numpy(), k))
    tdiff = g["t"].transform(lambda s: _diff(s.to_numpy(), k))
    df["vx"] = central_diff("gx") / tdiff.replace(0, np.nan)
    df["vy"] = central_diff("gy") / tdiff.replace(0, np.nan)
    df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0.0)
    df["speed_n"] = np.hypot(df["vx"], df["vy"]) / df["h"].clip(lower=8)
    df["heading"] = np.arctan2(df["vy"], df["vx"])
    df["dur"] = g["t"].transform(lambda s: s.max() - s.min())
    df = df[df["dur"] >= config.KIN["min_track_s"]].reset_index(drop=True)
    df["rider"] = _rider_flags(df)
    return df.drop(columns=["h_raw", "gx_raw", "gy_raw"])


def _diff(x: np.ndarray, k: int) -> np.ndarray:
    """x[i+k] - x[i-k] with edges clamped (one-sided at the ends)."""
    n = len(x)
    if n < 2:
        return np.zeros(n)
    hi = np.minimum(np.arange(n) + k, n - 1)
    lo = np.maximum(np.arange(n) - k, 0)
    return x[hi] - x[lo]


def _rider_flags(df: pd.DataFrame) -> np.ndarray:
    """Person boxes overlapping a two-wheeler box in the same frame are riders."""
    rider = np.zeros(len(df), bool)
    persons = df[df["cls"] == config.PERSON]
    bikes = df[df["cls"].isin(list(config.TWO_WHEELERS))]
    if persons.empty or bikes.empty:
        return rider
    bikes_by_frame = {f: b[["x1", "y1", "x2", "y2"]].to_numpy() for f, b in bikes.groupby("frame")}
    thr = config.RULES["jaywalking"]["rider_overlap"]
    for idx, p in zip(persons.index, persons[["frame", "x1", "y1", "x2", "y2"]].to_numpy()):
        bb = bikes_by_frame.get(int(p[0]))
        if bb is None:
            continue
        ix = np.clip(np.minimum(p[3], bb[:, 2]) - np.maximum(p[1], bb[:, 0]), 0, None)
        iy = np.clip(np.minimum(p[4], bb[:, 3]) - np.maximum(p[2], bb[:, 1]), 0, None)
        area = max((p[3] - p[1]) * (p[4] - p[2]), 1)
        if ((ix * iy) / area).max() > thr:
            rider[idx] = True
    # a person track that is a rider most of the time is a rider all the time
    s = pd.Series(rider, index=df.index)
    frac = s.groupby(df["tid"]).transform("mean")
    return (frac > 0.5).to_numpy() & (df["cls"] == config.PERSON).to_numpy()


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between row-aligned boxes a[i] and b[i] (x1,y1,x2,y2)."""
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-6)
