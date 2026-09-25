"""Seeds, logging and small segment helpers shared by all modules."""
from __future__ import annotations

import logging
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_logger(name: str = "wiut") -> logging.Logger:
    logging.basicConfig(level=os.environ.get("WIUT_LOG", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def runs(mask: np.ndarray, t: np.ndarray, max_gap: float = 0.0, min_len: float = 0.0) -> list[tuple[float, float]]:
    """Contiguous True runs of `mask` over timestamps `t` -> [(t_start, t_end)].
    Runs separated by less than `max_gap` seconds are bridged; runs shorter than `min_len` dropped."""
    mask = np.asarray(mask, bool)
    t = np.asarray(t, float)
    if len(t) == 0 or not mask.any():
        return []
    idx = np.flatnonzero(mask)
    out = []
    s = e = idx[0]
    for i in idx[1:]:
        dt = t[i] - t[e]
        # adjacent samples (no False in between) are bridged unless the track itself had a long hole
        if dt <= max(max_gap, 0) + 1e-9 or (i == e + 1 and dt <= max(max_gap, 1.0)):
            e = i
        else:
            out.append((t[s], t[e]))
            s = e = i
    out.append((t[s], t[e]))
    return [(a, b) for a, b in out if b - a >= min_len]


def merge_segments(segs: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    segs = sorted(segs)
    out: list[list[float]] = []
    for a, b in segs:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
