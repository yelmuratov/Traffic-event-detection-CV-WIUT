"""Part B: causal accident-risk estimator (P1).

Uses only the frames it receives. Every k-th frame: detect + track (own model and tracker
state, never Part A's output), fit short-window velocities, compute pairwise time-to-collision
and hard braking, map to [0, 1], smooth, and hold the value on skipped frames.
"""
from __future__ import annotations

from collections import deque

import time

import numpy as np

from . import budget, config
from .tracker import new_model, reset_tracker, _tracker_yaml
from .utils import set_seed
from .video import resize_to_width

C = config.RISK
_SHARED = {}


def calibrate(x: float) -> float:
    """Map the smoothed conflict level to P(accident within 5 s). Normal traffic in the sample videos
    (no accidents) stays below `calib_center` 99.5 % of the time, so 0.5 is crossed only by the rarest
    conflicts. The mapping is monotonic, so the ranking (AP) is unchanged."""
    return float(1.0 / (1.0 + np.exp(-(x - C["calib_center"]) / C["calib_scale"])))


def _model():
    if "m" not in _SHARED:
        _SHARED["m"] = new_model(C["weights"])
        _SHARED["yaml"] = _tracker_yaml(config.TRACKER)
    return _SHARED["m"], _SHARED["yaml"]


class RiskEngine:
    def __init__(self):
        self.meta = {}
        self.reset({})

    def reset(self, meta: dict) -> None:
        set_seed(config.SEED)
        self.meta = meta or {}
        fps = float(self.meta.get("fps") or 25.0)
        self.fps = fps
        self.stride = max(1, int(round(fps / C["target_hz"])))
        self.duration = float(self.meta.get("n_frames") or 0) / fps
        self.i = 0
        self.score = 0.0
        self.ema = 0.0
        self.hist: dict[int, deque] = {}
        self.prev_speed: dict[int, float] = {}
        self.brake = 0.0
        self.off = False                       # detection switched off to stay inside the time budget
        self._pace = (time.perf_counter(), 0.0)  # (wall, t_sec) at the last pace check
        if "m" in _SHARED:
            reset_tracker(_SHARED["m"])

    # ------------------------------------------------------------------
    def _check_budget(self, t_sec: float) -> None:
        """Every ~5 s of video: project the finish time from the recent pace; if it would pass
        SAFETY x budget, halve the detection rate (and finally stop detecting). A video over the
        budget loses everything, so a sparser risk curve is always the better trade."""
        if self.off or self.duration <= 0:
            return
        now = time.perf_counter()
        w0, t0 = self._pace
        if t_sec - t0 < 5.0:
            return
        rate = (now - w0) / (t_sec - t0)      # wall seconds per video second, recent window
        self._pace = (now, t_sec)
        spent = budget.elapsed(self.meta.get("video_id", ""))
        if spent is None:
            return
        projected = spent + (self.duration - t_sec) * rate
        if projected > budget.SAFETY * budget.TIME_FACTOR * self.duration:
            self.stride *= 2
            if self.stride > 8 * self.fps:
                self.off = True

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.i
        self.i += 1
        if i % max(1, int(self.fps)) == 0:
            self._check_budget(t_sec)
        if self.off:
            self.score *= 0.98                 # decay the last estimate, no detection
            return self.score
        if i % self.stride:
            return self.score
        model, yaml = _model()
        k = frame.shape[1] // config.DETECTOR["max_width"]
        if k >= 2:
            small, s = frame[::k, ::k], 1.0 / k  # integer subsample: far cheaper than cv2.resize on 4K
        else:
            small, s = resize_to_width(frame, config.DETECTOR["max_width"])
        res = model.track(small, persist=True, tracker=yaml, classes=config.DET_CLASSES,
                          conf=config.DETECTOR["conf"], iou=config.DETECTOR["iou"], imgsz=C["imgsz"],
                          half=config.HALF, device=config.DEVICE, verbose=False)[0]
        self._update_hist(res, t_sec, s)
        raw = self._raw_risk(t_sec)
        self.ema = C["ema"] * raw + (1 - C["ema"]) * self.ema
        self.score = calibrate(self.ema)
        return self.score

    # ------------------------------------------------------------------
    def _update_hist(self, res, t, s):
        b = res.boxes
        seen = set()
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy() / s
            for tid, c, (x1, y1, x2, y2) in zip(b.id.cpu().numpy().astype(int), b.cls.cpu().numpy().astype(int), xyxy):
                if c not in config.VEHICLES and c != config.PERSON:
                    continue
                self.hist.setdefault(tid, deque(maxlen=32)).append((t, (x1 + x2) / 2, y2, x2 - x1, y2 - y1, c))
                seen.add(tid)
        for tid in list(self.hist):
            if tid not in seen and t - self.hist[tid][-1][0] > 1.5:
                del self.hist[tid]
                self.prev_speed.pop(tid, None)

    def _kin(self, t):
        """Current position/velocity per track from a least-squares fit over history_s."""
        out = []
        for tid, h in self.hist.items():
            pts = [p for p in h if t - p[0] <= C["history_s"]]
            if len(pts) < 3 or t - pts[-1][0] > 0.3:
                continue
            a = np.asarray([p[:5] for p in pts], float)
            tt = a[:, 0] - a[-1, 0]
            vx = np.polyfit(tt, a[:, 1], 1)[0]
            vy = np.polyfit(tt, a[:, 2], 1)[0]
            hh = max(np.median(a[:, 4]), 8.0)
            out.append((tid, a[-1, 1], a[-1, 2], vx, vy, np.median(a[:, 3]), hh, pts[-1][5]))
        return out

    def _raw_risk(self, t):
        k = self._kin(t)
        # hard braking of any vehicle
        brake = 0.0
        for tid, x, y, vx, vy, w, h, c in k:
            sp = np.hypot(vx, vy) / h
            prev = self.prev_speed.get(tid)
            if prev is not None and prev > 0.8 and sp < (1 - C["brake_drop"]) * prev and c in config.VEHICLES:
                brake = C["brake_bonus"]
            self.prev_speed[tid] = 0.7 * sp + 0.3 * prev if prev is not None else sp
        self.brake = max(brake, self.brake * 0.8)
        if len(k) < 2:
            return self.brake
        a = np.asarray([r[1:7] for r in k], float)  # x y vx vy w h
        veh = np.asarray([r[7] in config.VEHICLES for r in k])
        spd = np.hypot(a[:, 2], a[:, 3]) / a[:, 5]  # box-heights / s
        n = len(a)
        i, j = np.triu_indices(n, 1)
        keep = (veh[i] | veh[j])                                  # pedestrian groups never crash
        keep &= np.where(veh[i], spd[i], 0) + np.where(veh[j], spd[j], 0) > C["min_vehicle_speed"]
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return self.brake
        hn = (a[i, 5] + a[j, 5]) / 2
        p = (a[j, :2] - a[i, :2]) / hn[:, None]
        v = (a[j, 2:4] - a[i, 2:4]) / hn[:, None]
        dist = np.linalg.norm(p, axis=1)
        vv = (v ** 2).sum(1)
        pv = (p * v).sum(1)
        closing = -pv / np.maximum(dist, 1e-6)
        tstar = np.where(vv > 1e-6, -pv / np.maximum(vv, 1e-6), np.inf)
        dmin = np.linalg.norm(p + v * np.clip(tstar, 0, 1e3)[:, None], axis=1)
        rad = 0.5 * (a[i, 4] + a[j, 4]) / 2 / hn
        miss = np.maximum(0.0, dmin - rad)
        # same-direction pairs (car following in a lane / queue) are normal traffic: they only count
        # when the closing is violent. Crossing and head-on conflicts count fully.
        ui = a[i, 2:4] / np.maximum(np.linalg.norm(a[i, 2:4], axis=1, keepdims=True), 1e-6)
        uj = a[j, 2:4] / np.maximum(np.linalg.norm(a[j, 2:4], axis=1, keepdims=True), 1e-6)
        both_move = (spd[i] > 0.3) & (spd[j] > 0.3)
        following = both_move & ((ui * uj).sum(1) > C["follow_cos"])
        ok = (tstar > 0) & (tstar < C["max_ttc_s"]) & (closing > C["min_closing"]) & (dist < 6)
        ok &= ~following | ((closing > C["follow_closing"]) & (tstar < C["follow_ttc_s"]))
        # moving past something that stands still (queued / parked car, waiting pedestrian) is normal:
        # count it only when the mover is heading straight at it
        one_static = (spd[i] < 0.3) ^ (spd[j] < 0.3)
        aim = closing / np.maximum(np.linalg.norm(v, axis=1), 1e-6)    # 1 = straight at the other
        ok &= ~one_static | ((aim > C["aim_cos"]) & (closing > C["follow_closing"]))
        ts_ = np.clip(tstar, 0.0, C["max_ttc_s"])        # clipped: no overflow on receding pairs
        pr = np.where(ok, np.exp(-ts_ / C["tau_s"]) * np.exp(-(np.minimum(miss, 10.0) ** 2) / (2 * C["sigma"] ** 2)), 0.0)
        return float(min(1.0, pr.max() + self.brake))  # the single most dangerous pair
