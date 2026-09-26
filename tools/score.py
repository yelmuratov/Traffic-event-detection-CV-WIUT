"""Our own Part A scorer (stand-in for the organisers' evaluate.py).

F1 per class, averaged over tIoU 0.3/0.5/0.7, then macro-averaged over every class that
appears in the labels OR the predictions (a predicted class absent from the labels scores 0).

  python -m tools.score /content/pred_rules_only.json my_labels.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

THRESHOLDS = (0.3, 0.5, 0.7)


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred: list, gt: list, thr: float) -> int:
    """Greedy one-to-one matching by tIoU; returns true positives."""
    pairs = sorted(((tiou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp


def score(pred: dict, gt: dict) -> dict:
    P, G = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for vid, d in pred.get("videos", pred).items():
        for s, e, lab in d["events"]:
            P[lab][vid].append((s, e))
    for vid, d in gt.items():
        for s, e, lab in d["events"]:
            G[lab][vid].append((s, e))
    rows = {}
    for lab in sorted(set(P) | set(G)):
        np_, ng = sum(map(len, P[lab].values())), sum(map(len, G[lab].values()))
        f1s = []
        for thr in THRESHOLDS:
            tp = sum(match(P[lab][v], G[lab][v], thr) for v in set(P[lab]) | set(G[lab]))
            prec, rec = tp / np_ if np_ else 0.0, tp / ng if ng else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        rows[lab] = {"pred": np_, "gt": ng, "f1": sum(f1s) / len(f1s), "f1@0.3": f1s[0]}
    macro = sum(r["f1"] for r in rows.values()) / max(len(rows), 1)
    return {"classes": rows, "macro_f1": macro}


if __name__ == "__main__":
    res = score(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    print(f"{'class':22s} {'pred':>5s} {'gt':>4s} {'F1@.3':>6s} {'F1avg':>6s}")
    for lab, r in res["classes"].items():
        print(f"{lab:22s} {r['pred']:5d} {r['gt']:4d} {r['f1@0.3']:6.2f} {r['f1']:6.2f}")
    print(f"\nmacro F1 (Part A): {res['macro_f1']:.3f}")
