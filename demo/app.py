"""Live demo (P3): upload an .mp4 (<= 1 min) -> events table, timeline, risk curve, annotated video.

Runs on CPU (Hugging Face Spaces free tier) with WIUT_DEMO=1 (small model, bigger stride).
  WIUT_DEMO=1 python demo/app.py
"""
import os
import sys
import tempfile

os.environ.setdefault("WIUT_DEMO", "1")
os.environ.setdefault("YOLO_OFFLINE", "0")                       # the Space may download weights once
os.environ.setdefault("WIUT_CACHE", os.path.join(tempfile.gettempdir(), "wiut_cache"))  # render reuses tracks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from solution import RiskEstimator, detect_events  # noqa: E402
from src.video import iter_frames, probe  # noqa: E402
from tools.render import render  # noqa: E402

MAX_S, MAX_MB = 65, 300


def run(video, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload an .mp4 first")
    meta = probe(video)
    if meta.duration > MAX_S or os.path.getsize(video) > MAX_MB * 2**20:
        raise gr.Error(f"Please upload a clip of at most 1 minute and {MAX_MB} MB")
    progress(0.05, desc="Part A: detecting and tracking")
    events = detect_events(video)
    progress(0.45, desc="Part B: causal risk curve")
    est = RiskEstimator()
    est.reset({"video_id": meta.name, "fps": meta.fps, "width": meta.width, "height": meta.height,
               "n_frames": meta.n_frames})
    risk = []
    for fi, t, frame, _ in iter_frames(video, stride=1):
        risk.append([round(t, 3), est.step(frame, t)])
        if fi % 100 == 0:
            progress(0.45 + 0.3 * fi / max(meta.n_frames, 1), desc=f"Part B: {t:.0f}s")
    progress(0.8, desc="Rendering annotated video")
    out_dir = tempfile.mkdtemp()
    mp4 = render(video, out_dir, events, risk, width=960, out_fps=10)
    stem = os.path.splitext(meta.name)[0]
    table = pd.DataFrame(events, columns=["start_s", "end_s", "label"])
    return table, os.path.join(out_dir, f"{stem}_timeline.png"), mp4


with gr.Blocks(title="WannaCry demo") as demo:
    gr.Markdown("## WannaCry · traffic event detection\n"
                "Upload an **.mp4 from the WIUT hackathon camera** (up to **1 minute**, 300 MB). "
                "The scene map is drawn for this junction, so other cameras will not give meaningful events. "
                "Runs on a free CPU with YOLO11n: expect 2–4 minutes; the progress bar shows each stage.")
    inp = gr.Video(label="Input .mp4", sources=["upload"])
    btn = gr.Button("Detect events", variant="primary")
    with gr.Row():
        tbl = gr.Dataframe(label="Events")
        tl = gr.Image(label="Timeline + risk curve")
    out = gr.Video(label="Annotated playback")
    btn.click(run, inp, [tbl, tl, out])

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0")
