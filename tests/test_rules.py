"""Rule unit tests on synthetic tracks (no GPU, no video).   python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.rules import (Ctx, rule_congestion, rule_jaywalking, rule_signals, rule_solid_line,  # noqa: E402
                       rule_stopped_vehicle, rule_turns, rule_wrong_way)
from src.scene import Scene  # noqa: E402
from src.tracks import build_table  # noqa: E402

FPS, STRIDE = 25.0, config.DETECTOR["stride"]
W, H = 1280, 720
ROAD = [[0, 300], [1280, 300], [1280, 720], [0, 720]]


def track(tid, cls, path, t0=0.0, size=(80, 60)):
    """path: function t -> (gx, gy) over a list of times; returns tracker rows."""
    rows = []
    for t, (gx, gy) in path:
        f = int(round((t0 + t) * FPS))
        rows.append([f, f / FPS, tid, cls, 0.9, gx - size[0] / 2, gy - size[1], gx + size[0] / 2, gy])
    return rows


def times(t0, t1):
    return np.arange(t0, t1, STRIDE / FPS)


def make_ctx(rows, scene_dict, signals=None, duration=60.0):
    scene_dict = {"size": [W, H], "road": [ROAD], **scene_dict}
    scene = Scene(scene_dict, W, H)
    scene.flow = None
    df = build_table(np.asarray(rows, np.float32), FPS)
    return Ctx(df=df, scene=scene, signals=signals or {}, fps=FPS, duration=max(duration, 120.0))


def lin(p0, v, ts):
    return [(t, (p0[0] + v[0] * t, p0[1] + v[1] * t)) for t in ts]


def test_stopped_vehicle():
    ts = times(0, 30)
    path = [(t, (200 + 100 * min(t, 5), 500)) for t in ts]     # drives 5 s then parks 25 s
    ev = rule_stopped_vehicle(make_ctx(track(1, 2, path), {}))
    assert len(ev) == 1 and abs(ev[0][0] - 5.0) < 0.6 and ev[0][1] > 29


def test_wrong_way():
    lanes = [{"polygon": ROAD, "direction": [1, 0]}]
    ctx = make_ctx(track(1, 2, lin((1200, 500), (-150, 0), times(0, 6))) +
                   track(2, 2, lin((100, 400), (150, 0), times(0, 6))), {"lanes": lanes})
    ev = rule_wrong_way(ctx)
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 4


def test_red_light_and_stop_line():
    sl = {"line": [[600, 300], [600, 720]], "direction": [1, 0], "signal": "s",
          "intersection": [[700, 300], [1000, 300], [1000, 720], [700, 720]]}
    ts_sig = times(0, 60)
    sig = {"s": (ts_sig.astype(np.float32), np.where(ts_sig < 30, 1, 0).astype(np.int8))}  # red until 30 s
    runner = track(1, 2, lin((300, 500), (100, 0), times(0, 9)))                 # crosses at ~2.6 s, red
    stopper = [(t, (min(400 + 100 * t, 640), 400)) for t in times(0, 40)]       # stops just past line
    ctx = make_ctx(runner + track(2, 2, stopper), {"stop_lines": [sl]}, sig)
    labels = sorted(e[2] for e in rule_signals(ctx))
    assert labels == ["red_light", "stop_line"], labels
    sl_ev = [e for e in rule_signals(ctx) if e[2] == "stop_line"][0]
    assert abs(sl_ev[1] - 30) < 1.0          # ends when the light turns green


def test_jaywalking():
    ped = track(1, 0, lin((300, 310), (0, 60), times(0, 6)), size=(30, 80))
    cw = [[900, 300], [1000, 300], [1000, 720], [900, 720]]
    ev = rule_jaywalking(make_ctx(ped, {"crosswalks": [cw]}))
    assert len(ev) == 1


def test_solid_line():
    ts = times(0, 8)
    path = [(t, (100 + 120 * t, 420 + min(40 * t, 120))) for t in ts]  # lane change across y=480 at t=1.5
    ev = rule_solid_line(make_ctx(track(1, 2, path), {"solid_lines": [[[0, 480], [1280, 480]]]}))
    assert len(ev) == 1


def test_forbidden_turn():
    ts = times(0, 12)
    path = [(t, (600, 700 - 60 * t)) if t < 5 else (t, (600 + 90 * (t - 5), 400)) for t in ts]  # up, then right
    mv = [{"from": [[500, 500], [700, 500], [700, 720], [500, 720]],
           "to": [[800, 300], [1280, 300], [1280, 500], [800, 500]], "label": "illegal_turn"}]
    ev = rule_turns(make_ctx(track(1, 2, path), {"forbidden_movements": mv}))
    assert [e[2] for e in ev] == ["illegal_turn"]


def test_congestion():
    rows = []
    for k in range(6):
        rows += track(k + 1, 2, [(t, (200 + 150 * k + 2 * t, 500)) for t in times(0, 100)])
    ev = rule_congestion(make_ctx(rows, {}))
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 90


def test_postprocess_valid():
    raw = [(1, 3, "jaywalking"), (3.5, 5, "jaywalking"), (2, 2.2, "wrong_way"), (4, 9, "accident"), (5, 6, "near_miss")]
    ev = postprocess(raw, 8.0)
    assert ev == [[1.0, 5.0, "jaywalking"], [4.0, 8.0, "accident"]]


def test_solid_line_ignores_drift_along_line():
    ts = times(0, 10)
    path = [(t, (100 + 120 * t, 480 + 6 * np.sin(3 * t))) for t in ts]   # rides on the line, jittering
    ev = rule_solid_line(make_ctx(track(1, 2, path), {"solid_lines": [[[0, 480], [1280, 480]]]}))
    assert ev == []
