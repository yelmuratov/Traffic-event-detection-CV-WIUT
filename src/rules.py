"""Hand-written event rules on the kinematics table (P2).

Every rule has the signature  rule(ctx) -> list[(start, end, label)]  and must never raise
into the pipeline (run_all wraps each one). Start/end follow the organisers' conventions
(see the Event classes table in the task PDF); postprocess.py does merging and trimming.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .scene import Scene, smooth_states
from .tracks import box_iou
from .utils import runs

log = logging.getLogger("wiut.rules")
R = config.RULES
STAT = config.KIN["stationary"]
MOVING = config.KIN["moving"]


@dataclass
class Ctx:
    df: pd.DataFrame
    scene: Scene
    signals: dict            # name -> (times, raw states)
    fps: float
    duration: float
    _cache: dict = field(default_factory=dict)

    @property
    def vehicles(self) -> pd.DataFrame:
        if "veh" not in self._cache:
            self._cache["veh"] = self.df[self.df["cls"].isin(list(config.VEHICLES))]
        return self._cache["veh"]

    @property
    def pedestrians(self) -> pd.DataFrame:
        if "ped" not in self._cache:
            self._cache["ped"] = self.df[(self.df["cls"] == config.PERSON) & (~self.df["rider"])]
        return self._cache["ped"]

    def signal_fn(self, name):
        """Return f(t) -> 1 red / 0 green / -1 unknown for signal `name`."""
        key = ("sig", name)
        if key not in self._cache:
            if name not in self.signals or len(self.signals[name][0]) == 0:
                self._cache[key] = None
            else:
                ts, st = self.signals[name]
                win = max(3, int(round(1.0 / (ts[1] - ts[0]))) if len(ts) > 1 else 3)
                self._cache[key] = (np.asarray(ts), smooth_states(st, win))
        v = self._cache[key]
        if v is None:
            return None
        ts, st = v
        return lambda t: st[np.clip(np.searchsorted(ts, np.atleast_1d(t), side="right") - 1, 0, len(st) - 1)]

    @staticmethod
    def enabled(label: str) -> bool:
        return config.ENABLED.get(label, False)

    def series(self, tid: int) -> pd.DataFrame:
        if "by_tid" not in self._cache:
            self._cache["by_tid"] = {k: g for k, g in self.df.groupby("tid")}
        return self._cache["by_tid"].get(tid, self.df.iloc[0:0])


# ============================================================ helpers
def _first_stationary_after(g: pd.DataFrame, t0: float, hold: float = 1.0):
    """Time the track becomes stationary (for >= hold s) after t0, or None."""
    g = g[g["t"] >= t0]
    for a, b in runs(g["speed_n"].to_numpy() < STAT, g["t"].to_numpy(), max_gap=0.5):
        if b - a >= hold or b >= g["t"].iloc[-1] - 1e-6:
            return a
    return None


def _side(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    return d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])


# ============================================================ stopped_vehicle
def _in_box(ctx: Ctx, x, y) -> np.ndarray:
    """Inside the junction box: on the road, outside the drawn approach lanes and queue zones."""
    m = ctx.scene.on_road(x, y)
    if ctx.scene.queue_zones:
        m &= ~ctx.scene.in_queue_zone(x, y)
    if ctx.scene.lanes:
        m &= ~ctx.scene.lane_direction(x, y)[1]
    return m


def rule_stopped_vehicle(ctx: Ctx):
    """A vehicle standing still inside the junction for >= min_s while it is not part of a jam.
    Parked cars / bus stops along the approach road are not events (they are outside the box)."""
    c = R["stopped_vehicle"]
    cands = []
    v = ctx.vehicles
    if v.empty:
        return []
    box = _in_box(ctx, v["gx"].to_numpy(), v["gy"].to_numpy())
    stat_all = v[box & (v["speed_n"].to_numpy() < STAT)]
    n_stat = stat_all.groupby("t").size()
    for tid, g in v[box].groupby("tid"):
        m = g["speed_n"].to_numpy() < STAT
        for a, b in runs(m, g["t"].to_numpy(), max_gap=1.0):
            seg = g[(g["t"] >= a) & (g["t"] <= b)]
            cands.append([a, b, seg["gx"].median(), seg["gy"].median(), seg["h"].median(), tid])
    # merge runs at the same spot (tracker ID switches on a parked car)
    cands.sort()
    merged: list = []
    for cnd in cands:
        for m_ in merged:
            if abs(cnd[2] - m_[2]) < 0.5 * m_[4] and abs(cnd[3] - m_[3]) < 0.5 * m_[4] and cnd[0] - m_[1] <= c["merge_gap_s"]:
                m_[1] = max(m_[1], cnd[1])
                break
        else:
            merged.append(list(cnd))
    out = []
    for a, b, x, y, h, tid in merged:
        if b - a < c["min_s"]:
            continue
        others = n_stat[(n_stat.index >= a) & (n_stat.index <= b)]
        if len(others) and np.median(others) - 1 >= c["max_others"]:
            continue  # many cars standing in the box = a jam (congestion), not one stopped vehicle
        out.append((a, b, "stopped_vehicle"))
    return out


def _is_queue(ctx: Ctx, a, b, x, y, h) -> bool:
    """Without drawn queue zones: a stop is a queue if >=2 other vehicles nearby are also stopped."""
    mid = (a + b) / 2
    v = ctx.vehicles
    near = v[(v["t"] - mid).abs() < 0.5]
    near = near[(np.hypot(near["gx"] - x, near["gy"] - y) < 3 * h) & (np.hypot(near["gx"] - x, near["gy"] - y) > 0.3 * h)]
    return (near["speed_n"] < STAT).sum() >= 2


# ============================================================ congestion
def rule_congestion(ctx: Ctx):
    """Gridlock: at least `min_stopped` vehicles standing still inside the junction box
    (approach queues at a red light are normal and do not count)."""
    c = R["congestion"]
    v = ctx.vehicles
    if v.empty:
        return []
    v = v[_in_box(ctx, v["gx"].to_numpy(), v["gy"].to_numpy())]
    all_t = np.sort(ctx.df["t"].unique())
    n_stat = (v["speed_n"] < c["stat_speed"]).groupby(v["t"]).sum().reindex(all_t).fillna(0)
    w = max(1, int(round(c["smooth_s"] * ctx.fps / config.DETECTOR["stride"])))
    flag = (n_stat >= c["min_stopped"]).astype(float).rolling(w, center=True, min_periods=1).mean() > 0.5
    return [(a, b, "congestion") for a, b in runs(flag.to_numpy(), all_t, max_gap=3.0, min_len=c["min_s"])]


# ============================================================ wrong_way
def rule_wrong_way(ctx: Ctx):
    c = R["wrong_way"]
    out = []
    for tid, g in ctx.vehicles.groupby("tid"):
        dirs, valid = ctx.scene.lane_direction(g["gx"].to_numpy(), g["gy"].to_numpy())
        if not valid.any():
            continue
        v = g[["vx", "vy"]].to_numpy()
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        cos = (u * dirs).sum(1)
        m = valid & (g["speed_n"].to_numpy() > c["min_speed"]) & (cos < c["cos"])
        for a, b in runs(m, g["t"].to_numpy(), max_gap=1.0, min_len=c["min_s"]):
            out.append((a, b, "wrong_way"))
    return out


# ============================================================ red_light + stop_line
def rule_signals(ctx: Ctx):
    out = []
    for sl in ctx.scene.stop_lines:
        sig = ctx.signal_fn(sl["signal"]) if sl["signal"] else None
        if sig is None:
            continue
        a, b, n = sl["a"], sl["b"], sl["dir"]
        orient = np.sign(_side(np.array([a + (b - a) / 2 + n * 10]), a, b))[0]
        L2 = float(((b - a) ** 2).sum())
        for tid, g in ctx.vehicles.groupby("tid"):
            fx = g["gx"].to_numpy() + n[0] * g["w"].to_numpy() * 0.5
            fy = g["gy"].to_numpy() + min(n[1], 0.0) * 0.3 * g["h"].to_numpy()
            p = np.column_stack([fx, fy])
            side = _side(p, a, b) * orient              # >0 = past the line
            u = ((p - a) @ (b - a)) / L2                # position along the line
            t = g["t"].to_numpy()
            idx = np.flatnonzero((side[:-1] < 0) & (side[1:] >= 0) & (u[1:] > -0.1) & (u[1:] < 1.1))
            if len(idx) == 0:
                continue
            i = idx[0]
            tc = t[i] + (t[i + 1] - t[i]) * (-side[i]) / max(side[i + 1] - side[i], 1e-6)
            if not (sig(tc)[0] == 1 and sig(tc - R["stop_line"]["red_before_s"])[0] == 1):
                continue
            # running the light only counts once red is well established (not the amber/clearing seconds)
            red_long = sig(tc - R["red_light"]["red_before_s"])[0] == 1
            if g["speed_n"].to_numpy()[max(0, i - 2): i + 3].max() < R["red_light"]["min_speed"] and \
                    _first_stationary_after(g.iloc[i + 1:], tc) is None:
                continue  # jitter of a waiting car around the line, not a real crossing
            after = g.iloc[i + 1:]
            if sl["intersection"] is not None:
                inside = sl["intersection"].contains(after["gx"].to_numpy(), after["gy"].to_numpy())
            else:
                inside = side[i + 1:] > after["h"].to_numpy() * 1.0
            t_enter = after["t"].to_numpy()[inside][0] if inside.any() else None
            t_stop = _first_stationary_after(after, tc)
            if t_enter is not None and (t_stop is None or t_enter <= t_stop):
                if not red_long:
                    continue
                inside_t = after["t"].to_numpy()[inside]
                end = min(inside_t[-1] if len(inside_t) else t[-1], tc + R["red_light"]["max_s"])
                out.append((tc, max(end, tc + 0.5), "red_light"))
            elif t_stop is not None:
                ts = np.arange(t_stop, min(t_stop + R["stop_line"]["max_after_stop_s"], ctx.duration), 0.2)
                green = ts[sig(ts) == 0] if len(ts) else []
                end = green[0] if len(green) else min(t[-1], ts[-1] if len(ts) else t_stop + 1)
                out.append((t_stop, max(end, t_stop + 0.5), "stop_line"))
    return out


# ============================================================ jaywalking
def rule_jaywalking(ctx: Ctx):
    if not ctx.scene.road:
        return []  # meaningless without a drawn carriageway
    c = R["jaywalking"]
    out = []
    for tid, g in ctx.pedestrians.groupby("tid"):
        x, y = g["gx"].to_numpy(), g["gy"].to_numpy()
        m = ctx.scene.on_road(x, y) & ~ctx.scene.in_crosswalk(x, y, c["crosswalk_buffer_px"])
        t = g["t"].to_numpy()
        for a, b in runs(m, t, max_gap=1.0, min_len=c["min_s"]):
            sel = (t >= a) & (t <= b)
            if np.median(g["speed_n"].to_numpy()[sel]) < c["min_speed"]:
                continue  # standing still (waiting at the curb, street vendor, detector noise)
            if g["conf"].to_numpy()[sel].mean() < c.get("min_conf", 0.0):
                continue  # weak detections (poles, shadows, far-away blobs)
            h = np.median(g["h"].to_numpy()[sel])
            if np.hypot(np.ptp(x[sel]), np.ptp(y[sel])) < c.get("min_path_h", 0.0) * h:
                continue  # hardly moved across the road
            out.append((a, b, "jaywalking"))
    return out


# ============================================================ failure_to_yield
def rule_failure_to_yield(ctx: Ctx):
    c = R["failure_to_yield"]
    out = []
    ped = ctx.pedestrians
    for cw in ctx.scene.crosswalks:
        pin = ped[cw.contains(ped["gx"].to_numpy(), ped["gy"].to_numpy(), c["ped_buffer_px"])]
        if pin.empty:
            continue
        pin = pin[pin["speed_n"] > c["ped_min_speed"]]      # walking, not waiting at the curb
        ped_by_frame = {f: g_[["gx", "gy"]].to_numpy() for f, g_ in pin.groupby("frame")}
        for tid, g in ctx.vehicles.groupby("tid"):
            m = cw.contains(g["gx"].to_numpy(), g["gy"].to_numpy())
            if not m.any():
                continue
            t = g["t"].to_numpy()
            fr, gx, gy, w = (g[k].to_numpy() for k in ("frame", "gx", "gy", "w"))
            for a, b in runs(m, t, max_gap=0.5):
                sel = (t >= a) & (t <= b)
                moving = np.median(g["speed_n"].to_numpy()[sel]) > c["min_vehicle_speed"]  # drives through, not creeping
                ped_near = False
                for i in np.flatnonzero(sel):
                    P = ped_by_frame.get(fr[i])
                    if P is not None and (np.hypot(P[:, 0] - gx[i], P[:, 1] - gy[i]) < c["near_w"] * w[i]).any():
                        ped_near = True
                        break
                if moving and ped_near:
                    out.append((a, max(b, a + 0.5), "failure_to_yield"))
    return out


# ============================================================ solid_line_crossing
def rule_solid_line(ctx: Ctx):
    c = R["solid_line"]
    out = []
    for line in ctx.scene.solid_lines:
        for k in range(len(line) - 1):
            a, b = line[k], line[k + 1]
            d = b - a
            L = np.linalg.norm(d)
            if L < 1:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            for tid, g in ctx.vehicles.groupby("tid"):
                p = g[["gx", "gy"]].to_numpy()
                u = ((p - a) @ d) / L ** 2
                dist = (p - a) @ nrm
                ext = 0.5 * (g["w"].to_numpy() * abs(nrm[0]) + 0.3 * g["h"].to_numpy() * abs(nrm[1]))
                t = g["t"].to_numpy()
                ok = (u > 0) & (u < 1)
                sgn = np.where(dist >= 0, 1, -1)
                idx = np.flatnonzero((sgn[:-1] * sgn[1:] < 0) & ok[:-1] & ok[1:])
                spd = g["speed_n"].to_numpy()
                w_ = g["w"].to_numpy()
                for i in idx:
                    pw = c["persist_s"]
                    win_b = (t > t[i] - pw) & (t <= t[i])
                    win_a = (t >= t[i + 1]) & (t < t[i + 1] + pw)
                    if win_b.sum() < 3 or win_a.sum() < 3:
                        continue
                    # clearly on the old side before and on the new side after (not jitter along the line)
                    if (sgn[win_b] == sgn[i]).mean() < 0.9 or (sgn[win_a] == sgn[i + 1]).mean() < 0.9:
                        continue
                    if np.abs(dist[win_a]).max() < max(c["min_cross_px"], c["min_cross_w"] * w_[i]):
                        continue
                    if np.median(spd[win_b | win_a]) < c["min_speed"]:
                        continue  # queued / creeping cars drifting over the paint
                    touch = np.abs(dist) <= ext
                    s = i
                    while s > 0 and touch[s - 1] and t[i] - t[s - 1] < c["max_s"]:
                        s -= 1
                    e = i + 1
                    while e < len(t) - 1 and touch[e] and t[e] - t[i] < c["max_s"]:
                        e += 1
                    if t[e] - t[s] > c["max_s"]:
                        continue  # a real lane change takes a few seconds, not a long drift
                    out.append((t[s], max(t[e], t[s] + 0.5), "solid_line_crossing"))
    return out


# ============================================================ turns
def _turn_extent(g: pd.DataFrame, min_deg: float, window: float):
    """Largest heading change within `window` s; returns (deg, t_onset, t_end) or None."""
    mv = g[g["speed_n"] > MOVING]
    if len(mv) < 4:
        return None
    h = np.degrees(np.unwrap(mv["heading"].to_numpy()))
    t = mv["t"].to_numpy()
    best = None
    j = 0
    for i in range(len(t)):
        while j < len(t) - 1 and t[j + 1] - t[i] <= window:
            j += 1
        seg = h[i:j + 1] - h[i]
        k = int(np.argmax(np.abs(seg)))
        if best is None or abs(seg[k]) > abs(best[0]):
            best = (seg[k], i, i + k)
    if best is None or abs(best[0]) < min_deg:
        return None
    ang, i, k = best
    on = R["turn"]["onset_deg"]
    s = i + int(np.argmax(np.abs(h[i:k + 1] - h[i]) > on))
    e = k - int(np.argmax(np.abs(h[i:k + 1][::-1] - h[k]) > on)) + 1
    e = min(max(e, s + 1), k)
    return abs(ang), t[s], t[e]


def rule_turns(ctx: Ctx):
    c = R["turn"]
    out = []
    for tid, g in ctx.vehicles.groupby("tid"):
        x, y, t = g["gx"].to_numpy(), g["gy"].to_numpy(), g["t"].to_numpy()
        # 1) forbidden movements (zone A -> zone B)
        for mv in ctx.scene.forbidden:
            in_from, in_to = mv["from"].contains(x, y), mv["to"].contains(x, y)
            if in_from.any() and in_to.any():
                t_to = t[in_to].min()                            # first time in the exit zone
                before = in_from & (t < t_to)
                if not before.any():
                    continue
                t_from = t[before].max()                         # last time in the entry zone
                ext = _turn_extent(g[(g["t"] >= t_from - 2) & (g["t"] <= t_to + 2)], 30, c["window_s"])
                a, b = (ext[1], ext[2]) if ext else (t_from, t_to)
                out.append((a, max(b, a + 0.5), mv["label"]))
        # 2) U-turns anywhere inside a no-U-turn zone
        if ctx.scene.no_u_turn:
            ext = _turn_extent(g, c["u_turn_deg"], c["window_s"])
            if ext:
                _, a, b = ext
                pa = g[g["t"] >= a].iloc[0]
                if ctx.scene._any(ctx.scene.no_u_turn, [pa["gx"]], [pa["gy"]])[0]:
                    out.append((a, max(b, a + 0.5), "illegal_u_turn"))
    return out


# ============================================================ pairwise: near_miss + accident
def _pair_table(ctx: Ctx) -> pd.DataFrame:
    df = ctx.df
    objs = df[df["cls"].isin(list(config.VEHICLES)) | ((df["cls"] == config.PERSON) & (~df["rider"]))]
    rows = []
    for f, g in objs.groupby("frame"):
        n = len(g)
        if n < 2:
            continue
        P = g[["gx", "gy"]].to_numpy()
        V = g[["vx", "vy"]].to_numpy()
        H = g["h"].to_numpy()
        W = g["w"].to_numpy()
        B = g[["x1", "y1", "x2", "y2"]].to_numpy()
        ids = g["tid"].to_numpy()
        i, j = np.triu_indices(n, 1)
        hn = (H[i] + H[j]) / 2
        p = (P[j] - P[i]) / hn[:, None]
        dist = np.linalg.norm(p, axis=1)
        keep = dist < 4
        if not keep.any():
            continue
        i, j, hn, p, dist = i[keep], j[keep], hn[keep], p[keep], dist[keep]
        v = (V[j] - V[i]) / hn[:, None]
        vv = (v ** 2).sum(1)
        pv = (p * v).sum(1)
        tstar = np.where(vv > 1e-6, -pv / np.maximum(vv, 1e-6), np.inf)
        dmin = np.linalg.norm(p + v * np.clip(tstar, 0, 1e3)[:, None], axis=1)
        rad = 0.5 * (W[i] + W[j]) / 2 / hn
        closing = -pv / np.maximum(dist, 1e-6)
        ttc = np.where((tstar > 0) & (dmin < rad + 0.3) & (closing > R["accident"]["min_closing"]), tstar, np.inf)
        # contact: bottom strips of the boxes overlap and the objects are at similar depth
        si, sj = B[i].copy(), B[j].copy()
        si[:, 1] = si[:, 3] - 0.25 * (si[:, 3] - si[:, 1])
        sj[:, 1] = sj[:, 3] - 0.25 * (sj[:, 3] - sj[:, 1])
        contact = (box_iou(si, sj) > R["accident"]["contact_iou"]) & (np.abs(B[i, 3] - B[j, 3]) < R["accident"]["depth_frac"] * hn)
        a_id, b_id = np.minimum(ids[i], ids[j]), np.maximum(ids[i], ids[j])
        t = g["t"].iloc[0]
        rows.append(np.column_stack([np.full(len(i), t), a_id, b_id, dist, ttc, contact, closing]))
    if not rows:
        return pd.DataFrame(columns=["t", "a", "b", "dist", "ttc", "contact", "closing"])
    pt = pd.DataFrame(np.vstack(rows), columns=["t", "a", "b", "dist", "ttc", "contact", "closing"])
    pt[["a", "b"]] = pt[["a", "b"]].astype(int)
    pt["contact"] = pt["contact"].astype(bool)
    return pt.sort_values(["a", "b", "t"]).reset_index(drop=True)


def _in_queue(ctx: Ctx, ga: pd.DataFrame, gb: pd.DataFrame, t0: float) -> bool:
    """Both objects inside a signal queue zone at time t0 (bumper-to-bumper waiting, not a crash)."""
    if not ctx.scene.queue_zones:
        return False
    inside = []
    for g in (ga, gb):
        r = g.iloc[(g["t"] - t0).abs().argmin()] if len(g) else None
        inside.append(r is not None and bool(ctx.scene.in_queue_zone([r["gx"]], [r["gy"]])[0]))
    return all(inside)


def _speed_at(g: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    return g[(g["t"] >= t0) & (g["t"] <= t1)]["speed_n"].to_numpy()


def rule_pairs(ctx: Ctx):
    """near_miss from time-to-collision + evasive action (disabled by default, see config)."""
    cn = R["near_miss"]
    pt = _pair_table(ctx)
    ctx._cache["pairs"] = pt
    out = []
    for (a, b), g in pt.groupby(["a", "b"]):
        ga, gb = ctx.series(a), ctx.series(b)
        t = g["t"].to_numpy()
        cont = g["contact"].to_numpy()
        # ---------------- near miss
        if cont.any() or not ctx.enabled("near_miss"):
            continue
        risky = np.isfinite(g["ttc"].to_numpy()) & (g["ttc"].to_numpy() < cn["ttc_s"])
        for s0, _ in runs(risky, t, max_gap=1.0):
            if _in_queue(ctx, ga, gb, s0):
                continue
            ev = None
            for gg in (ga, gb):
                pre = _speed_at(gg, s0 - 0.6, s0)
                if len(pre) == 0 or pre.max() < cn["min_speed"]:
                    continue
                v0 = pre.max()
                win = gg[(gg["t"] >= s0) & (gg["t"] <= s0 + cn["react_s"])]
                if win.empty:
                    continue
                decel = win["speed_n"].min() < (1 - cn["decel_frac"]) * v0
                moving = win[win["speed_n"] > MOVING]
                hd = np.degrees(np.unwrap(moving["heading"].to_numpy()))
                swerve = len(hd) > 1 and np.abs(hd - hd[0]).max() > cn["swerve_deg"]
                if decel or swerve:
                    pre_g = gg[(gg["t"] >= s0 - 1.0) & (gg["t"] <= s0 + cn["react_s"])]
                    sp = pre_g["speed_n"].to_numpy()
                    onset = pre_g["t"].to_numpy()[int(np.argmax(sp < 0.9 * v0))] if decel and (sp < 0.9 * v0).any() else s0
                    ev = onset if ev is None else min(ev, onset)
            if ev is None:
                continue
            after = g[g["t"] >= s0]
            k = int(after["dist"].to_numpy().argmin())
            dmin, t_close = after["dist"].iloc[k], after["t"].iloc[k]
            clear = after[(after["t"] > t_close) & (after["dist"] > max(1.5 * dmin, 1.2))]
            end = clear["t"].iloc[0] if len(clear) else t_close + 1.0
            out.append((ev, min(max(end, ev + 0.5), ev + 8.0), "near_miss"))
    return out


# ============================================================ accident (strict)
def rule_accident(ctx: Ctx):
    """Two vehicles come into contact OUTSIDE the signal queues, then BOTH stay stopped for
    hold_s while the traffic around them keeps flowing. Normal queues and jams fail the last test."""
    c = R["accident_strict"]
    pt = ctx._cache.get("pairs")
    if pt is None:
        pt = _pair_table(ctx)
        ctx._cache["pairs"] = pt
    veh = set(ctx.vehicles["tid"].unique())
    out = []
    for (a, b), g in pt[pt["contact"]].groupby(["a", "b"]):
        if a not in veh or b not in veh:
            continue
        full = pt[(pt["a"] == a) & (pt["b"] == b)]
        tc = g["t"].iloc[0]
        if not ((full["t"] < tc - 0.5) & ~full["contact"]).any():
            continue                                    # never seen apart: a double detection, not a crash
        ga, gb = ctx.series(a), ctx.series(b)
        ra, rb = ga.iloc[(ga["t"] - tc).abs().argmin()], gb.iloc[(gb["t"] - tc).abs().argmin()]
        cx, cy = (ra["gx"] + rb["gx"]) / 2, (ra["gy"] + rb["gy"]) / 2
        hn = (ra["h"] + rb["h"]) / 2
        if not ctx.scene.on_road([cx], [cy])[0] or \
                (ctx.scene.queue_zones and ctx.scene.in_queue_zone([cx], [cy])[0]):
            continue
        pre = np.concatenate([_speed_at(ga, tc - 2, tc), _speed_at(gb, tc - 2, tc), [0]])
        if pre.max() < c["min_speed_before"]:
            continue                                    # nobody was moving: parked / queued contact
        stops = []
        for gg in (ga, gb):
            after = gg[(gg["t"] >= tc) & (gg["t"] <= tc + c["stop_within_s"] + c["hold_s"] + 1)]
            st = None
            for s0, s1 in runs(after["speed_n"].to_numpy() < STAT, after["t"].to_numpy(), max_gap=0.7):
                if s0 - tc <= c["stop_within_s"] and s1 - s0 >= c["hold_s"]:
                    st = s0
                    break
            stops.append(st)
        if any(s_ is None for s_ in stops):
            continue
        # the rest of the traffic nearby keeps moving (rules out red-light queues and jams)
        v = ctx.vehicles
        w = v[(v["t"] > tc + c["stop_within_s"]) & (v["t"] < tc + c["stop_within_s"] + c["hold_s"])
              & ~v["tid"].isin([a, b])]
        w = w[np.hypot(w["gx"] - cx, w["gy"] - cy) < c["flow_radius"] * hn]
        if len(w) < c["min_flow_samples"] or np.median(w["speed_n"]) < c["flow_speed"]:
            continue
        out.append((tc, max(max(stops), tc + 1.0), "accident"))
    return out


# ============================================================ road obstacle (background comparison)
def rule_road_obstacle(ctx: Ctx):
    """A new object that stays still on the carriageway for persist_s and is not a tracked
    vehicle or person: compare sparse grey thumbnails against the video's median background."""
    import cv2
    th = ctx.signals.get("__thumbs__") if ctx.signals else None
    if th is None or not ctx.scene.road:
        return []
    ts, imgs = th
    if len(ts) < 20:
        return []
    c = R["road_obstacle"]
    H, W = imgs.shape[1:]
    sx, sy = W / ctx.scene.width, H / ctx.scene.height
    road = np.zeros((H, W), np.uint8)
    for p in ctx.scene.road:
        cv2.fillPoly(road, [(p.pts * [sx, sy]).astype(np.int32)], 1)
    for p in ctx.scene.ignore_zones:
        cv2.fillPoly(road, [(p.pts * [sx, sy]).astype(np.int32)], 0)
    road = cv2.erode(road, np.ones((5, 5), np.uint8)).astype(bool)
    imgs = np.stack([cv2.GaussianBlur(im, (5, 5), 0) for im in imgs]).astype(np.float32)
    ref = np.median(imgs[:, road], axis=1, keepdims=True)                       # per-frame brightness
    imgs[:, road] = imgs[:, road] * (np.median(ref) / np.maximum(ref, 1))       # lighting normalisation
    # two backgrounds (first and last third of the clip). A pixel is foreground only if it differs from
    # BOTH: slow global changes (moving tree shadows, sun) differ from just one, so this stays quiet on
    # normal footage. Trade-off: an obstacle still present at the very end of the clip is missed.
    n3 = max(3, len(ts) // 3)
    bg_early, bg_late = np.median(imgs[:n3], axis=0), np.median(imgs[-n3:], axis=0)
    df = ctx.df
    kernel = np.ones((3, 3), np.uint8)
    blobs_t = []
    for i, t in enumerate(ts):
        diff = np.minimum(np.abs(imgs[i] - bg_early), np.abs(imgs[i] - bg_late))
        fg = ((diff > c["diff"]) & road).astype(np.uint8)
        near = df[(df["t"] - t).abs() < 0.4]
        for r in near.itertuples():                                              # remove every tracked object
            pw, ph = (r.x2 - r.x1) * c["box_pad"], (r.y2 - r.y1) * c["box_pad"]
            x1, y1 = int((r.x1 - pw) * sx), int((r.y1 - ph) * sy)
            x2, y2 = int((r.x2 + pw) * sx) + 1, int((r.y2 + ph) * sy) + 1
            fg[max(0, y1):y2, max(0, x1):x2] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
        keep = []
        for k in range(1, n):
            x, y, w_, h_, area = stats[k]
            if not (c["min_area"] <= area <= c["max_area"]):
                continue
            if area / float(w_ * h_) < c["min_fill"] or max(w_, h_) / max(1, min(w_, h_)) > c["max_aspect"]:
                continue                                                         # thin shadow edges, lane paint
            keep.append(cent[k])
        blobs_t.append(keep)
    # link blobs that stay at the same place over consecutive samples
    out, active = [], []                                                         # active: [cx, cy, t_start, t_last]
    for t, blobs in zip(ts, blobs_t):
        nxt = []
        for bx, by in blobs:
            m = next((a for a in active if abs(a[0] - bx) < c["link_px"] and abs(a[1] - by) < c["link_px"]), None)
            nxt.append([bx, by, m[2] if m else t, t])
        ended = [a for a in active if not any(abs(a[0] - n_[0]) < c["link_px"] and abs(a[1] - n_[1]) < c["link_px"]
                                              for n_ in nxt)]
        for a in ended:
            if a[3] - a[2] >= c["persist_s"]:
                out.append((a[2], a[3], "road_obstacle"))
        active = nxt
    for a in active:
        if a[3] - a[2] >= c["persist_s"]:
            out.append((a[2], min(a[3] + 1.0, ctx.duration), "road_obstacle"))
    return out


# ============================================================ runner
RULE_FUNCS = [
    ("stopped_vehicle", rule_stopped_vehicle),
    ("congestion", rule_congestion),
    ("wrong_way", rule_wrong_way),
    (("red_light", "stop_line"), rule_signals),
    ("jaywalking", rule_jaywalking),
    ("failure_to_yield", rule_failure_to_yield),
    ("solid_line_crossing", rule_solid_line),
    (("illegal_turn", "illegal_u_turn"), rule_turns),
    ("near_miss", rule_pairs),
    ("accident", rule_accident),
    ("road_obstacle", rule_road_obstacle),
]


def run_all(ctx: Ctx) -> list[tuple[float, float, str]]:
    events = []
    for labels, fn in RULE_FUNCS:
        labels = labels if isinstance(labels, tuple) else (labels,)
        if not any(config.ENABLED.get(l, False) for l in labels):
            continue
        try:
            evs = fn(ctx)
            events.extend(e for e in evs if config.ENABLED.get(e[2], False))
        except Exception as exc:  # one broken rule must not kill the video
            log.exception("rule %s failed: %s", fn.__name__, exc)
    return events
