"""Part A end to end: video -> tracks -> kinematics -> rules -> clean segments."""
from __future__ import annotations

import logging
import time

from . import config
from .postprocess import postprocess
from .rules import Ctx, run_all
from .scene import load_scene
from .tracker import analyse_video
from .tracks import build_table
from .utils import set_seed
from .video import probe

log = logging.getLogger("wiut.pipeline")


def analyse(video_path: str, progress: bool = False):
    """Everything up to the rules; returned so notebooks can inspect intermediate data."""
    set_seed(config.SEED)
    meta = probe(video_path)
    t0 = time.time()
    scene = load_scene(meta.width, meta.height, video_path=video_path)
    log.info("%s: scene alignment %.1fs", meta.name, time.time() - t0)
    tracks, signals = analyse_video(meta, scene, progress=progress)
    df = build_table(tracks, meta.fps)
    ctx = Ctx(df=df, scene=scene, signals=signals, fps=meta.fps, duration=meta.duration)
    return meta, ctx


def detect_events(video_path: str, return_raw: bool = False):
    t0 = time.time()
    meta, ctx = analyse(video_path)
    t1 = time.time()
    raw = run_all(ctx)
    log.info("%s: rules %.1fs", meta.name, time.time() - t1)
    events = postprocess(raw, meta.duration)
    log.info("%s: %d events (%d raw) in %.1fs", meta.name, len(events), len(raw), time.time() - t0)
    return (events, raw, ctx) if return_raw else events
