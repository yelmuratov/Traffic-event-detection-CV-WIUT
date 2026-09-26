"""Synthetic test clips for classes that never occur in the sample videos (dev only).

  reverse  : play a piece of a sample backwards -> every car drives the wrong way  (wrong_way must fire)
  obstacle : paint a box on the road for a while                                   (road_obstacle must fire)

  python -m tools.synth reverse  samples/C3896.mp4 150 12 synth/rev.mp4
  python -m tools.synth obstacle samples/C3896.mp4 60 60 synth/obs.mp4   # box 22-40 s of a 60 s clip
"""
from __future__ import annotations

import argparse
import os
import subprocess


def reverse_clip(video: str, t0: float, dur: float, out: str, width: int = 1920) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0}", "-t", f"{dur}", "-i", video,
                    "-vf", f"scale={width}:-2,reverse", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-an", out], check=True)
    return out


def obstacle_clip(video: str, t0: float, dur: float, out: str, box=(2160, 1560, 90, 70),
                  appear: float = 22.0, disappear: float = 40.0, width: int = 1920, src_width: int = 3840) -> str:
    """box = (x, y, w, h) in full-resolution pixels of the source video; shown from appear to disappear s."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    s = width / src_width
    x, y, w, h = (int(v * s) for v in box)
    draw = (f"drawbox=x={x}:y={y}:w={w}:h={h}:color=0x6b4a2b@1:t=fill:enable='between(t,{appear},{disappear})',"
            f"drawbox=x={x}:y={y}:w={w}:h={max(2, h // 5)}:color=0x3a2615@1:t=fill:enable='between(t,{appear},{disappear})'")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0}", "-t", f"{dur}", "-i", video,
                    "-vf", f"scale={width}:-2,{draw}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-an", out], check=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["reverse", "obstacle"])
    ap.add_argument("video"); ap.add_argument("t0", type=float); ap.add_argument("dur", type=float); ap.add_argument("out")
    a = ap.parse_args()
    fn = reverse_clip if a.kind == "reverse" else obstacle_clip
    print(fn(a.video, a.t0, a.dur, a.out))
