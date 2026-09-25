"""Scene map of the fixed camera: road, lanes, crosswalks, stop lines, signals, zones.

The map is drawn once by hand (see the Colab notebook's click tool) and stored in
scene/scene_map.json in pixel coordinates of a reference frame of size `size`.
It is rescaled automatically if a video has a different resolution.

A data-driven direction ("flow") map built from tracks of the sample videos is used
as a fallback for lane directions when no lanes are drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MplPath

from . import config

EMPTY_SCENE = {
    "size": [1920, 1080],
    "road": [],                 # list of polygons [[x,y],...] = carriageway
    "lanes": [],                # {"name","polygon","direction":[dx,dy],"group"}
    "crosswalks": [],           # list of polygons
    "stop_lines": [],           # {"name","line":[[x,y],[x,y]],"direction":[dx,dy],"signal","intersection":polygon}
    "signals": {},              # name -> {"red_roi":[x1,y1,x2,y2], "green_roi":[x1,y1,x2,y2]} or {"roi":[...]}
    "solid_lines": [],          # list of polylines [[x,y],...]
    "queue_zones": [],          # polygons where stopping at a signal is normal
    "ignore_zones": [],         # polygons to ignore (parking, sidewalk, far background)
    "forbidden_movements": [],  # {"name","from":polygon,"to":polygon,"label":"illegal_turn"|"illegal_u_turn"}
    "no_u_turn_zones": [],      # polygons where any U-turn is illegal
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Poly:
    def __init__(self, pts, scale=(1.0, 1.0)):
        self.pts = np.asarray(pts, float) * np.asarray(scale)
        self.path = MplPath(self.pts)

    def contains(self, x, y, radius: float = 0.0) -> np.ndarray:
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        # matplotlib: positive radius grows CCW polygons, shrinks CW ones -> test both signs
        if radius:
            return self.path.contains_points(pts, radius=radius) | self.path.contains_points(pts, radius=-radius)
        return self.path.contains_points(pts)


class Scene:
    def __init__(self, data: dict | None, width: int, height: int):
        d = dict(EMPTY_SCENE)
        d.update(data or {})
        self.raw = d
        ref_w, ref_h = d.get("size") or [width, height]
        self.sx, self.sy = width / ref_w, height / ref_h
        sc = (self.sx, self.sy)
        self.width, self.height = width, height

        self.road = [Poly(p, sc) for p in d["road"]]
        self.crosswalks = [Poly(p, sc) for p in d["crosswalks"]]
        self.queue_zones = [Poly(p, sc) for p in d["queue_zones"]]
        self.ignore_zones = [Poly(p, sc) for p in d["ignore_zones"]]
        self.no_u_turn = [Poly(p, sc) for p in d["no_u_turn_zones"]]
        self.lanes = [{"name": l.get("name", f"lane{i}"), "poly": Poly(l["polygon"], sc),
                       "dir": _unit(l["direction"]) if l.get("direction") else None,
                       "group": l.get("group", "all")} for i, l in enumerate(d["lanes"])]
        self.stop_lines = []
        for i, s in enumerate(d["stop_lines"]):
            a, b = (np.asarray(s["line"], float) * np.asarray(sc))
            self.stop_lines.append({
                "name": s.get("name", f"stop{i}"), "a": a, "b": b,
                "dir": _unit(s["direction"]), "signal": s.get("signal"),
                "intersection": Poly(s["intersection"], sc) if s.get("intersection") else None,
            })
        self.signals = {}
        for name, s in d["signals"].items():
            self.signals[name] = {k: (np.asarray(v, float) * np.array([self.sx, self.sy, self.sx, self.sy])).astype(int)
                                  for k, v in s.items() if k in ("red_roi", "green_roi", "roi")}
        self.solid_lines = [np.asarray(l, float) * np.asarray(sc) for l in d["solid_lines"]]
        self.forbidden = [{"name": m.get("name", f"move{i}"), "from": Poly(m["from"], sc), "to": Poly(m["to"], sc),
                           "label": m.get("label", "illegal_turn")} for i, m in enumerate(d["forbidden_movements"])]
        self.flow = FlowMap.load(config.FLOW_MAP_PATH, width, height)

    # ------------------------------------------------------------ queries
    @staticmethod
    def _any(polys, x, y, radius=0.0):
        x = np.atleast_1d(x)
        out = np.zeros(len(x), bool)
        for p in polys:
            out |= p.contains(x, y, radius)
        return out

    def on_road(self, x, y) -> np.ndarray:
        """True where the point is on the carriageway (everything counts if no road is drawn)."""
        x = np.atleast_1d(x)
        base = self._any(self.road, x, y) if self.road else np.ones(len(x), bool)
        if self.ignore_zones:
            base &= ~self._any(self.ignore_zones, x, y)
        return base

    def in_crosswalk(self, x, y, radius=0.0) -> np.ndarray:
        return self._any(self.crosswalks, x, y, radius)

    def in_queue_zone(self, x, y) -> np.ndarray:
        return self._any(self.queue_zones, x, y)

    def lane_direction(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Allowed direction (unit vectors, Nx2) and validity mask at each point.
        Drawn lanes win; otherwise the learned flow map is used."""
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)
        dirs = np.zeros((len(x), 2))
        valid = np.zeros(len(x), bool)
        for lane in self.lanes:
            if lane["dir"] is None:
                continue
            m = lane["poly"].contains(x, y) & ~valid
            dirs[m] = lane["dir"]
            valid |= m
        if self.flow is not None and not self.lanes:
            fd, fv = self.flow.query(x, y)
            m = fv & ~valid
            dirs[m] = fd[m]
            valid |= m
        return dirs, valid

    def lane_group(self, x, y) -> np.ndarray:
        x = np.atleast_1d(x)
        out = np.array(["all"] * len(x), dtype=object)
        for lane in self.lanes:
            out[lane["poly"].contains(x, y)] = lane["group"]
        return out

    # ------------------------------------------------------------ drawing
    def draw(self, img: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        over = img.copy()
        def fill(polys, color):
            for p in polys:
                cv2.fillPoly(over, [p.pts.astype(np.int32)], color)
        fill(self.road, (80, 80, 80))
        fill(self.queue_zones, (0, 160, 255))
        fill(self.crosswalks, (255, 255, 255))
        fill(self.ignore_zones, (60, 0, 60))
        out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
        for lane in self.lanes:
            pts = lane["poly"].pts.astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 200, 0), 2)
            if lane["dir"] is not None:
                c = pts.mean(0)
                e = c + lane["dir"] * 80
                cv2.arrowedLine(out, tuple(c.astype(int)), tuple(e.astype(int)), (255, 200, 0), 4, tipLength=0.3)
        for s in self.stop_lines:
            cv2.line(out, tuple(s["a"].astype(int)), tuple(s["b"].astype(int)), (0, 0, 255), 4)
            if s["intersection"] is not None:
                cv2.polylines(out, [s["intersection"].pts.astype(np.int32)], True, (0, 0, 255), 1)
        for l in self.solid_lines:
            cv2.polylines(out, [l.astype(np.int32)], False, (0, 255, 255), 3)
        for m in self.forbidden:
            cv2.polylines(out, [m["from"].pts.astype(np.int32)], True, (255, 0, 255), 2)
            cv2.polylines(out, [m["to"].pts.astype(np.int32)], True, (180, 0, 180), 2)
        for name, s in self.signals.items():
            for k, r in s.items():
                col = (0, 0, 255) if k == "red_roi" else (0, 255, 0) if k == "green_roi" else (0, 255, 255)
                cv2.rectangle(out, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), col, 2)
        return out


def load_scene(width: int, height: int, path: Path | str | None = None) -> Scene:
    path = Path(path or config.SCENE_PATH)
    data = json.loads(path.read_text()) if path.exists() else None
    return Scene(data, width, height)


# ============================================================ traffic-light state
def signal_state(frame: np.ndarray, sig: dict) -> int:
    """Return 1 = red, 0 = green, -1 = unknown (amber/off/occluded) from ROI colours.
    Works on the full-resolution frame (ROIs are in full-res pixel coordinates)."""
    def lit_fraction(roi, hue_ranges):
        x1, y1, x2, y2 = [int(v) for v in roi]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = np.zeros(h.shape, bool)
        for lo, hi in hue_ranges:
            m |= (h >= lo) & (h <= hi)
        m &= (s > 70) & (v > 140)
        return float(m.mean())

    RED = [(0, 10), (170, 180)]
    GREEN = [(40, 95)]
    if "red_roi" in sig or "green_roi" in sig:
        r = lit_fraction(sig["red_roi"], RED) if "red_roi" in sig else 0.0
        g = lit_fraction(sig["green_roi"], GREEN) if "green_roi" in sig else 0.0
    else:
        r = lit_fraction(sig["roi"], RED)
        g = lit_fraction(sig["roi"], GREEN)
    thr = 0.02
    if r > thr and r >= g:
        return 1
    if g > thr:
        return 0
    return -1


def smooth_states(states: np.ndarray, win: int = 5) -> np.ndarray:
    """Majority filter; unknown frames inherit the last known state (amber after green etc.)."""
    s = np.asarray(states, int).copy()
    out = s.copy()
    half = win // 2
    for i in range(len(s)):
        w = s[max(0, i - half): i + half + 1]
        known = w[w >= 0]
        if len(known):
            out[i] = 1 if (known == 1).sum() * 2 > len(known) else 0
    last = -1
    for i in range(len(out)):
        if out[i] < 0:
            out[i] = last
        else:
            last = out[i]
    return out


# ============================================================ learned direction map
class FlowMap:
    """Per-grid-cell dominant motion direction learned from tracks."""

    def __init__(self, dirs, count, coherence, width, height):
        self.dirs, self.count, self.coh = dirs, count, coherence
        self.gh, self.gw = count.shape
        self.width, self.height = width, height

    @classmethod
    def build(cls, tracks_df, width, height, grid=(36, 64), min_speed=0.5):
        """tracks_df needs columns gx, gy, vx, vy, speed_n, cls (see tracks.py)."""
        gh, gw = grid
        acc = np.zeros((gh, gw, 2))
        cnt = np.zeros((gh, gw))
        df = tracks_df[(tracks_df["speed_n"] > min_speed) & tracks_df["cls"].isin(list(config.VEHICLES))]
        ix = np.clip((df["gx"].to_numpy() / width * gw).astype(int), 0, gw - 1)
        iy = np.clip((df["gy"].to_numpy() / height * gh).astype(int), 0, gh - 1)
        v = df[["vx", "vy"]].to_numpy()
        n = np.linalg.norm(v, axis=1, keepdims=True)
        u = v / np.maximum(n, 1e-6)
        np.add.at(acc, (iy, ix), u)
        np.add.at(cnt, (iy, ix), 1)
        mean = acc / np.maximum(cnt[..., None], 1)
        coh = np.linalg.norm(mean, axis=2)
        dirs = mean / np.maximum(coh[..., None], 1e-6)
        return cls(dirs, cnt, coh, width, height)

    def save(self, path=None):
        np.savez_compressed(path or config.FLOW_MAP_PATH, dirs=self.dirs, count=self.count, coh=self.coh,
                            size=np.array([self.width, self.height]))

    @classmethod
    def load(cls, path, width, height):
        path = Path(path)
        if not path.exists():
            return None
        z = np.load(path)
        return cls(z["dirs"], z["count"], z["coh"], width, height)

    def query(self, x, y):
        c = config.RULES["wrong_way"]
        ix = np.clip((np.asarray(x) / self.width * self.gw).astype(int), 0, self.gw - 1)
        iy = np.clip((np.asarray(y) / self.height * self.gh).astype(int), 0, self.gh - 1)
        valid = (self.count[iy, ix] >= c["flow_min_count"]) & (self.coh[iy, ix] >= c["flow_min_coherence"])
        return self.dirs[iy, ix], valid

    def draw(self, img, step=1):
        out = img.copy()
        ch, cw = self.height / self.gh, self.width / self.gw
        c = config.RULES["wrong_way"]
        for i in range(0, self.gh, step):
            for j in range(0, self.gw, step):
                if self.count[i, j] < c["flow_min_count"]:
                    continue
                p = np.array([(j + 0.5) * cw, (i + 0.5) * ch])
                q = p + self.dirs[i, j] * min(cw, ch) * 0.45
                col = (0, 255, 0) if self.coh[i, j] >= c["flow_min_coherence"] else (0, 140, 255)
                cv2.arrowedLine(out, tuple(p.astype(int)), tuple(q.astype(int)), col, 2, tipLength=0.4)
        return out
