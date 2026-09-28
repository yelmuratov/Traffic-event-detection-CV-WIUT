"""Live demo (P3): upload an .mp4 (<= 1 min) -> events table, timeline, risk curve, annotated video.

Runs on CPU (Hugging Face Spaces) with WIUT_DEMO=1 (small model, bigger stride).
  WIUT_DEMO=1 python demo/app.py
"""
import os
import sys
import tempfile
from collections import Counter

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
SITE = "https://yelmuratov.github.io/Traffic-event-detection-CV-WIUT/"
REPO = "https://github.com/yelmuratov/Traffic-event-detection-CV-WIUT"
COLS = ["Start (s)", "End (s)", "Event"]
EMPTY = pd.DataFrame(columns=COLS)
IDLE = "<div class='status idle'>Upload a clip and press <b>Detect events</b>.</div>"


def _summary(events):
    if not events:
        return "<div class='status ok'>Done: no events detected in this clip.</div>"
    chips = "".join(f"<span class='chip'>{lab.replace('_', ' ')} <b>{n}</b></span>"
                    for lab, n in Counter(e[2] for e in events).most_common())
    return f"<div class='status ok'>Done: <b>{len(events)}</b> events found.<div class='chips'>{chips}</div></div>"


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
    table = pd.DataFrame([[round(s, 1), round(e, 1), lab.replace("_", " ")] for s, e, lab in events],
                         columns=COLS)
    return _summary(events), table, os.path.join(out_dir, f"{stem}_timeline.png"), mp4


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#eef1f4",
    body_background_fill_dark="#10151b",
    block_background_fill="#ffffff",
    block_background_fill_dark="#171e26",
    block_border_color="#d3d9df",
    block_border_color_dark="#2a3441",
    button_primary_background_fill="#d99100",
    button_primary_background_fill_hover="#b87b00",
    button_primary_background_fill_dark="#f2a91a",
    button_primary_text_color="#17202b",
    button_primary_text_color_dark="#10151b",
)

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
#hero {background: #17202b; color: #f3f5f7; border-radius: 12px; padding: 22px 26px;}
#hero h1 {font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-size: 34px; font-weight: 700;
          letter-spacing: .5px; margin: 0 0 4px; color: #f3f5f7;}
#hero h1 span {color: #f2a91a;}
#hero p {margin: 0; color: #c3ccd5; font-size: 15px;}
#hero .links {margin-top: 12px;}
#hero .links a {color: #17202b; background: #f2a91a; padding: 5px 12px; border-radius: 999px; font-weight: 600;
                text-decoration: none; margin-right: 8px; font-size: 13px;}
#hero .links a.ghost {background: transparent; color: #f3f5f7; border: 1px solid #56626f;}
.steps {font-size: 14px; line-height: 1.55;}
.steps ol {margin: 6px 0 0 18px; padding: 0;}
.steps .note {color: #8a6000; background: #fff4db; border: 1px solid #f0d59a; border-radius: 8px;
              padding: 8px 10px; margin-top: 10px; font-size: 13px;}
.status {border-radius: 10px; padding: 12px 14px; font-size: 14px;}
.status.idle {background: transparent; color: #56626f; border: 1px dashed #c3ccd5;}
.status.ok {background: #e7f5ee; color: #14553d; border: 1px solid #b6e0cb;}
.chips {margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;}
.chip {background: #ffffff; border: 1px solid #b6e0cb; border-radius: 999px; padding: 2px 10px; font-size: 13px;
       text-transform: capitalize;}
#run-btn {font-size: 16px; font-weight: 600; min-height: 48px;}
.steps .note b, .status.ok b, .chip, .chip b {color: inherit !important;}
.chip {color: #14553d !important;}
#hero {border: 1px solid #2a3441;}
footer {display: none !important;}
"""

with gr.Blocks(title="WannaCry · live demo", theme=THEME, css=CSS) as demo:
    gr.HTML(f"""<div id="hero">
      <h1>WannaCry <span>·</span> traffic event detection</h1>
      <p>YOLO11 + ByteTrack + scene rules find traffic events, and a causal model plots accident risk.
         WIUT Hackathon 2026, CV track.</p>
      <div class="links"><a href="{SITE}" target="_blank">Project website</a>
      <a class="ghost" href="{REPO}" target="_blank">GitHub</a></div></div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Video(label="1 · Upload a clip (.mp4)", sources=["upload"], height=300)
            btn = gr.Button("▶  Detect events", variant="primary", elem_id="run-btn")
            gr.HTML("""<div class="steps"><b>How it works</b><ol>
              <li>Upload up to <b>1 minute</b> of footage from the <b>WIUT hackathon camera</b> (max 300 MB).</li>
              <li>Press <b>Detect events</b> and follow the progress bar.</li>
              <li>See the event list, the timeline with the risk curve, and the annotated video.</li></ol>
              <div class="note">Runs on a CPU with the small YOLO11n model, so a 1-minute clip takes
              <b>3–5 minutes</b>. The scene map is drawn for this junction only; other cameras
              will not give meaningful events.</div></div>""")
        with gr.Column(scale=6):
            status = gr.HTML(IDLE)
            tbl = gr.Dataframe(value=EMPTY, headers=COLS, label="2 · Detected events",
                               interactive=False, wrap=True, max_height=320)
    tl = gr.Image(label="3 · Timeline + risk curve", show_download_button=True)
    out = gr.Video(label="4 · Annotated playback")
    btn.click(run, inp, [status, tbl, tl, out])

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", ssr_mode=False)
