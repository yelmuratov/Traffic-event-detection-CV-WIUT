"""EDA of the sample videos for the website (P3). Uses cached tracks (run the tracker first).

  python -m tools.eda samples/ outputs/eda

Writes PNG charts + eda.json (numbers for interactive charts on the website):
  metadata table, brightness over time (lighting), object counts per class over time,
  motion heatmap, trajectories coloured by direction, learned flow map, density per minute.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.scene import FlowMap, load_scene  # noqa: E402
from src.tracker import analyse_video  # noqa: E402
from src.tracks import build_table  # noqa: E402
from src.video import iter_frames, probe, read_frame_at  # noqa: E402

CLS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def ffprobe_info(path: str) -> dict:
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,bit_rate,pix_fmt", "-of", "json", path],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)["streams"][0]
    except Exception:
        return {}


def metadata(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        m = probe(p)
        info = ffprobe_info(p)
        rows.append({"video": m.name, "width": m.width, "height": m.height, "fps": round(m.fps, 3),
                     "frames": m.n_frames, "duration_s": round(m.duration, 1),
                     "size_MB": round(os.path.getsize(p) / 2**20, 1), "codec": info.get("codec_name"),
                     "bitrate_Mbps": round(int(info.get("bit_rate", 0) or 0) / 1e6, 2)})
    return pd.DataFrame(rows)


def brightness(path: str, every_s: float = 2.0) -> pd.DataFrame:
    m = probe(path)
    stride = max(1, int(m.fps * every_s))
    rows = []
    for _, t, f, _ in iter_frames(path, stride=stride, max_width=480):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        rows.append({"t": t, "brightness": float(hsv[..., 2].mean()), "saturation": float(hsv[..., 1].mean())})
    return pd.DataFrame(rows)


def counts_over_time(df: pd.DataFrame, bin_s: float = 5.0) -> pd.DataFrame:
    d = df.assign(bin=(df["t"] // bin_s) * bin_s, name=df["cls"].map(CLS_NAMES))
    return d.groupby(["bin", "name"])["tid"].nunique().unstack(fill_value=0)


def run(video_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    meta_df = metadata(paths)
    meta_df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    print(meta_df.to_string(index=False))
    report = {"metadata": meta_df.to_dict("records"), "videos": {}}
    all_tables = []
    for p in paths:
        m = probe(p)
        scene = load_scene(m.width, m.height)
        tracks, _ = analyse_video(m, scene)
        df = build_table(tracks, m.fps)
        df["video"] = m.name
        all_tables.append(df)
        stem = os.path.splitext(m.name)[0]
        ref = read_frame_at(p, min(5.0, m.duration / 2))

        br = brightness(p)
        cnt = counts_over_time(df)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(br["t"], br["brightness"], color="#444"); ax[0].set_ylabel("mean brightness (V)")
        cnt.plot(ax=ax[1], lw=1.5); ax[1].set_ylabel("unique objects / 5 s"); ax[1].set_xlabel("time (s)")
        fig.suptitle(f"{m.name}: lighting and object counts"); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{stem}_counts.png"), dpi=110); plt.close(fig)

        # motion heatmap on the reference frame
        heat = np.zeros((m.height, m.width), np.float32)
        mv = df[df["speed_n"] > config.KIN["moving"]]
        xs = np.clip(mv["gx"].astype(int), 0, m.width - 1); ys = np.clip(mv["gy"].astype(int), 0, m.height - 1)
        np.add.at(heat, (ys, xs), 1)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=m.width / 150)
        heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
        col = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        over = cv2.addWeighted(ref, 0.55, col, 0.45, 0) if ref is not None else col
        cv2.imwrite(os.path.join(out_dir, f"{stem}_heatmap.jpg"), over)

        # trajectories coloured by heading
        fig, ax = plt.subplots(figsize=(12, 12 * m.height / m.width))
        if ref is not None:
            ax.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), alpha=0.6)
        for tid, g in df[df["cls"].isin(list(config.VEHICLES))].groupby("tid"):
            if len(g) < 5:
                continue
            ang = np.arctan2(g["gy"].iloc[-1] - g["gy"].iloc[0], g["gx"].iloc[-1] - g["gx"].iloc[0])
            ax.plot(g["gx"], g["gy"], lw=0.8, color=plt.cm.hsv((ang + np.pi) / (2 * np.pi)), alpha=0.7)
        ax.set_axis_off(); ax.set_title(f"{m.name}: vehicle trajectories (colour = direction)")
        fig.savefig(os.path.join(out_dir, f"{stem}_trajectories.png"), dpi=100, bbox_inches="tight"); plt.close(fig)

        per_min = df.assign(minute=(df["t"] // 60).astype(int)).groupby(["minute"])["tid"].nunique()
        report["videos"][m.name] = {
            "brightness": br.round(2).to_dict("list"),
            "counts_5s": {"t": cnt.index.tolist(), **{k: cnt[k].tolist() for k in cnt.columns}},
            "unique_per_minute": per_min.to_dict(),
            "class_totals": df.groupby(df["cls"].map(CLS_NAMES))["tid"].nunique().to_dict(),
            "mean_speed_n": float(df[df["cls"].isin(list(config.VEHICLES))]["speed_n"].mean()) if len(df) else 0.0,
        }

    # learned direction map across all videos (also used by wrong_way when no lanes are drawn)
    if all_tables:
        big = pd.concat(all_tables)
        m = probe(paths[0])
        flow = FlowMap.build(big, m.width, m.height)
        flow.save()
        ref = read_frame_at(paths[0], 5.0)
        if ref is not None:
            cv2.imwrite(os.path.join(out_dir, "flow_map.jpg"), flow.draw(ref))
    json.dump(report, open(os.path.join(out_dir, "eda.json"), "w"), default=float)
    print("EDA written to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos"); ap.add_argument("out")
    a = ap.parse_args()
    run(a.videos, a.out)
