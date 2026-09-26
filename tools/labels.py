"""Dev-set labelling helpers (everyone labels 1/3 of the sample videos).

  python -m tools.labels skeleton samples/ labels/labels_p1.json      # empty file with duration/fps
  python -m tools.labels merge labels/labels_p*.json --out my_labels.json
  python -m tools.labels check my_labels.json
  python -m tools.labels clip samples/a.mp4 12.0 19.0 out.mp4         # cut a clip to check boundaries

Format (same as the organisers' ground truth):
  {"video.mp4": {"duration": 340.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"], ...]}}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import CLASSES  # noqa: E402
from src.video import probe  # noqa: E402


def skeleton(video_dir: str) -> dict:
    out = {}
    for p in sorted(p for p in glob.glob(os.path.join(video_dir, "*")) if p.lower().endswith(".mp4")):
        m = probe(p)
        out[m.name] = {"duration": round(m.duration, 3), "fps": m.fps, "events": []}
    return out


def check(labels: dict) -> list[str]:
    errs = []
    for vid, d in labels.items():
        last = {}
        for s, e, lab in sorted(d.get("events", []), key=lambda x: (x[2], x[0])):
            if lab not in CLASSES:
                errs.append(f"{vid}: unknown label {lab}")
            if not (0 <= s < e <= d.get("duration", 1e9) + 1e-6):
                errs.append(f"{vid}: bad times {s}-{e} {lab}")
            if lab in last and s < last[lab]:
                errs.append(f"{vid}: overlapping {lab} at {s}")
            last[lab] = max(last.get(lab, 0), e)
    return errs


def merge(files: list[str]) -> dict:
    out: dict = {}
    for f in files:
        for vid, d in json.load(open(f)).items():
            if vid not in out:
                out[vid] = {"duration": d["duration"], "fps": d["fps"], "events": []}
            out[vid]["events"].extend(d.get("events", []))
    for d in out.values():
        d["events"].sort(key=lambda x: (x[0], x[2]))
    return out


def clip(video: str, t0: float, t1: float, out: str, width: int = 1280) -> str:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", video,
                    "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an", out],
                   check=True)
    return out


def _sec(tok: str) -> float:
    """'75.5' or '1:15.5' or '0:01:15.5' -> seconds."""
    parts = [float(x) for x in tok.strip().split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec


def from_txt(txt_path: str, video_dir: str) -> dict:
    """Plain-text labels -> labels dict.  One event per line:
        C3897  1:04.0 - 1:07.5  red_light
    Lines starting with # are comments. Video name without .mp4 is fine."""
    labels = skeleton(video_dir)
    names = {os.path.splitext(k)[0].lower(): k for k in labels}
    errors = []
    for n, line in enumerate(open(txt_path, encoding="utf-8"), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        try:
            vid, rest = line.split(None, 1)
            times, lab = rest.rsplit(None, 1)
            a, b = times.replace(" ", "").split("-")
            key = names[os.path.splitext(vid)[0].lower()]
            labels[key]["events"].append([round(_sec(a), 2), round(_sec(b), 2), lab.strip()])
        except Exception as exc:
            errors.append(f"line {n}: {line!r} ({exc})")
    for d in labels.values():
        d["events"].sort(key=lambda x: (x[0], x[2]))
    if errors:
        print("Could not read:\n  " + "\n  ".join(errors))
    return labels


def summary(labels: dict) -> dict:
    from collections import Counter
    c = Counter(e[2] for d in labels.values() for e in d["events"])
    return dict(sorted(c.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("skeleton"); a.add_argument("videos"); a.add_argument("out")
    a = sub.add_parser("merge"); a.add_argument("files", nargs="+"); a.add_argument("--out", required=True)
    a = sub.add_parser("check"); a.add_argument("file")
    a = sub.add_parser("txt"); a.add_argument("txt"); a.add_argument("videos"); a.add_argument("out")
    a = sub.add_parser("clip"); a.add_argument("video"); a.add_argument("t0", type=float); a.add_argument("t1", type=float); a.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "skeleton":
        json.dump(skeleton(args.videos), open(args.out, "w"), indent=1)
    elif args.cmd == "merge":
        m = merge(args.files)
        errs = check(m)
        print("\n".join(errs) or "OK", "\n", summary(m))
        json.dump(m, open(args.out, "w"), indent=1)
    elif args.cmd == "check":
        lab = json.load(open(args.file))
        print("\n".join(check(lab)) or "OK", "\n", summary(lab))
    elif args.cmd == "txt":
        lab = from_txt(args.txt, args.videos)
        print("\n".join(check(lab)) or "OK", "\n", summary(lab))
        json.dump(lab, open(args.out, "w"), indent=1)
    elif args.cmd == "clip":
        print(clip(args.video, args.t0, args.t1, args.out))
