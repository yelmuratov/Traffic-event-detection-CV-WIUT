# WIUT Hackathon 2026 — CV Track: Traffic Event Detection

Fixed CCTV camera → list of traffic events `[start_sec, end_sec, label]` (Part A) and a causal
per-frame accident risk (Part B).

## Run (the two official commands)

```bash
pip install -r requirements.txt
bash weights/download.sh            # once, with internet (weights can also be committed in weights/)
python run_submission.py --videos /data/test --out predictions.json
```

`run.sh` does all of it on `samples/` and runs `evaluate.py --validate-only`.

## Repository layout

```
solution.py            interface used by run_submission.py (thin wrapper around src/)
run_submission.py      starter kit — unchanged        evaluate.py   starter kit — unchanged
src/config.py          every threshold, model choice and frame stride in one place
src/video.py           probing + strided decoding in a background thread
src/tracker.py         YOLO + ByteTrack, one decoding pass, optional on-disk cache (WIUT_CACHE)
src/tracks.py          kinematics: ground point, speed in box-heights/s, heading, rider filter
src/scene.py           hand-drawn scene map (scene/scene_map.json), traffic-light colour state, learned flow map
src/rules.py           one function per event class
src/postprocess.py     merge / trim / validate segments
src/pipeline.py        Part A end to end
src/risk.py            Part B: time-to-collision + hard braking, causal
tools/                 EDA, rendering, labels, Colab helpers (dev only)
demo/app.py            Gradio live demo (CPU)
tests/                 rule unit tests on synthetic tracks (pytest)
notebooks/             Colab notebook used for development on a T4
```

## Approach

| Stage | Learned or rule-based |
|---|---|
| Object detection: YOLO11m @1280 (COCO: person, bicycle, car, motorcycle, bus, truck), every 2nd frame | learned (pre-trained, open weights) |
| Tracking: ByteTrack (Ultralytics implementation) | algorithmic |
| Scene map: road, lanes + directions, crosswalks, stop lines, signal lamp ROIs, solid lines, turn zones | hand-drawn once (camera is fixed) |
| Lane directions fallback: flow map learned from sample-video tracks | statistics |
| Traffic light state: HSV colour of the lamp ROI | rule |
| 12 event classes | rules on tracks (see `src/rules.py`) |
| Part B risk | TTC between tracked pairs + hard braking, EMA-smoothed |

`road_obstacle` and `fire_smoke` are disabled (`config.ENABLED`) because a predicted class that is not
in the test set costs a full class of F1.

## Determinism

Fixed seed 42 (`random`, `numpy`, `torch`, cuDNN deterministic). Two runs produce the same JSON.

## Datasets and licences

- COCO-pretrained YOLO11 weights — Ultralytics, AGPL-3.0.
- No other training data used so far. (List DoTA / CCD etc. here if a clip classifier is added.)

## Open-source code

Ultralytics (YOLO, ByteTrack) — AGPL-3.0. OpenCV — Apache-2.0.

## Team

| Member | Role | Did |
|---|---|---|
| P1 | Vision engineer | detection, tracking, runtime, Part B |
| P2 | Rules & evaluation | scene map, rules, post-processing, labels, scoring |
| P3 | Product & presentation | repo, EDA, website, demo, report |
