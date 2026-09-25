"""Turn raw rule hits into clean, valid segments (matters a lot at tIoU 0.7)."""
from __future__ import annotations

from collections import defaultdict

from . import config
from .utils import merge_segments


def postprocess(raw: list[tuple[float, float, str]], duration: float) -> list[list]:
    by_cls: dict = defaultdict(list)
    for s, e, lab in raw:
        by_cls[lab].append((float(s), float(e)))
    out = []
    for lab, segs in by_cls.items():
        c = config.post_cfg(lab)
        segs = [(s + c["start_shift"], e + c["end_shift"]) for s, e in segs]
        segs = merge_segments(segs, c["merge_gap"])          # also removes same-class overlaps
        for s, e in segs:
            if c["max_len"]:
                e = min(e, s + c["max_len"])
            s, e = max(0.0, s), min(duration, e) if duration > 0 else e
            if e - s >= c["min_len"] and e > s:
                out.append([round(s, 2), round(e, 2), lab])
    out = _suppress(out)
    out.sort(key=lambda x: (x[0], x[2]))
    return _ensure_valid(out)


# (weaker, stronger): drop a weaker-class segment that overlaps a stronger one
SUPPRESS = [("near_miss", "accident")]


def _suppress(events: list[list]) -> list[list]:
    keep = []
    for s, e, lab in events:
        stronger = [w for weak, w in SUPPRESS if weak == lab]
        if any(l2 in stronger and s < e2 and s2 < e for s2, e2, l2 in events):
            continue
        keep.append([s, e, lab])
    return keep


def _ensure_valid(events: list[list]) -> list[list]:
    """Guarantee what the harness checks: start < end, known label, no same-class overlap."""
    last_end: dict = {}
    valid = []
    for s, e, lab in events:
        if lab not in config.ENABLED or s >= e:
            continue
        if lab in last_end and s < last_end[lab]:
            continue
        valid.append([s, e, lab])
        last_end[lab] = e
    return valid
