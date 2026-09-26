# WIUT Hackathon 2026 — CV Track: Traffic Event Detection

**Team WannaCry**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yelmuratov/Traffic-event-detection-CV-WIUT/blob/main/notebooks/WIUT_CV_Colab.ipynb)

A fixed 4K CCTV camera watches a signalised junction. Our system:

- **Part A:** returns every traffic event in a video as `[start_sec, end_sec, label]`, covering 14 official classes.
- **Part B:** outputs a causal, per-frame probability that an accident starts within the next 5 s.

The pipeline is a pre-trained detector and a tracker that turn the video into trajectories. Hand-written rules then read those trajectories against a hand-drawn map of the scene, and a post-processing step cleans up the resulting segments.

**Dev score (our own labels of the 4 sample videos, official `evaluate.py`): Score A = 0.346**

---

## Quick start (the two official commands)

```bash
pip install -r requirements.txt
bash weights/download.sh          # once, WITH internet: fetches the open YOLO11 weights into weights/
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

- Python ≥ 3.10; tested on Colab (T4, CUDA 12).
- After `weights/download.sh` everything runs offline.
- `run.sh` does all of the above on `samples/` and validates the output format.
- `predictions_samples.json` is our output on the 4 sample videos. It was produced by this code, and Part B was skipped for that run (`--no-risk`).

## How it works

```
video ─► decoding (PyAV, reference frames only ≈ every 3rd frame, 1920 px)
      ─► YOLO11m @1280  (person, bicycle, car, motorcycle, bus, truck)
      ─► ByteTrack      (track IDs)
      ─► kinematics     (ground point, speed in box-heights/s, heading)
      ─► scene map      (road, lanes + directions, crosswalks, stop line, signal lamp,
                         solid lines, queue zones, forbidden turn) — re-projected per video
      ─► rules          (one function per class, src/rules.py)
      ─► post-processing (merge fragments, boundary shifts, min length, validity)
      ─► [start, end, label]
```

| Stage | Learned or rule-based |
|---|---|
| Object detection: YOLO11m, COCO pre-trained, imgsz 1280 | learned (open weights, not fine-tuned) |
| Tracking: ByteTrack (Ultralytics implementation) | algorithmic |
| Scene map (`scene/scene_map.json`): drawn once on `scene/ref_frame.jpg` | manual |
| Per-video alignment: SIFT + RANSAC homography between the reference frame and each video's background | algorithmic |
| Traffic-light state: HSV colour of the red/green lamp regions | rule |
| Event classes | rules on trajectories + scene map |
| Event boundaries: per-class start/end shifts and merge gaps | tuned on our dev labels |
| Part B risk | physics rule (time-to-collision), no training |

### Event rules (Part A)

| Class | Rule (short) |
|---|---|
| `stopped_vehicle` | vehicle stationary ≥ 10 s inside the junction box (outside approach lanes and queue zones), and not part of a jam |
| `congestion` | ≥ 4 vehicles standing still inside the junction box, sustained ≥ 12 s (red-light queues on the approach do not count) |
| `red_light` | front of a vehicle crosses the stop line after the light has been red for ≥ 5 s |
| `stop_line` | vehicle stops past the stop line on red without entering the junction; ends when the light turns green |
| `jaywalking` | walking pedestrian on the carriageway, away from any crosswalk (detections with low confidence ignored) |
| `failure_to_yield` | vehicle drives through a crosswalk while a walking pedestrian is within 1.5 car widths |
| `solid_line_crossing` | lane change across one of the 4 solid lane lines before the stop line (persistent on both sides) |
| `illegal_turn` | turn into the bottom-left road from an inner lane (not the kerb lane) |
| `wrong_way` | vehicle moving against its lane's drawn direction |
| `accident` | strict: two vehicles meet in the junction and both stay stopped while surrounding traffic keeps moving |
| `near_miss`, `road_obstacle`, `fire_smoke`, `illegal_u_turn` | disabled or not triggered: too few examples to validate, and a predicted class absent from the test set costs a full class of F1 |

All thresholds live in `src/config.py`.

### Part B — accident anticipation

`RiskEstimator.step()` sees only the frames it is given (causal):

1. About 6 times per second it runs its own YOLO11s detector and tracker, independent of Part A.
2. It fits short-window velocities for every track.
3. For every pair of road users it computes the time to closest approach and the miss distance:
   `risk_pair = exp(−TTC/1.5 s) · exp(−miss² / 2σ²)`
4. The three riskiest pairs are combined, plus a bonus for hard braking, then smoothed with an EMA.

A time guard (`src/budget.py`) watches the shared 3× time budget. If the projected finish gets close to the limit, it lowers the detection rate. Our samples contain no accidents, so Part B could not be calibrated (see Limitations).

## Results on the sample videos (dev set, our labels)

The dev set is 77 events across 4 videos, labelled by us in `labels/labels.txt`. Scores are mean F1 over tIoU 0.3/0.5/0.7:

| Class | GT | Pred | F1 (mean) |
|---|---|---|---|
| stop_line | 4 | 5 | 0.67 |
| jaywalking | 22 | 24 | 0.55 |
| congestion | 9 | 3 | 0.50 |
| stopped_vehicle | 4 | 8 | 0.50 |
| illegal_turn | 8 | 5 | 0.46 |
| red_light | 2 | 2 | 0.33 |
| failure_to_yield | 13 | 17 | 0.27 |
| solid_line_crossing | 13 | 13 | 0.18 |
| illegal_u_turn | 1 | 0 | 0.00 |
| near_miss | 1 | 0 | 0.00 |
| **Score A** | | | **0.346** |

Progress: 0.12 with the first rules, 0.32 after tuning on the dev labels, and 0.346 after boundary tuning. The rules were tuned on the same labels they are scored on, so expect a lower score on unseen videos.

## Runtime

- **Budget:** 3 × video duration per video, Part A + Part B together.
- **Part A on Colab (T4, 2 CPU cores):** ≈ 1.9 × realtime.
  - About two thirds of that is decoding 4K frames, which is CPU-bound.
  - The detector takes about 25 %.
- **Harness decode on Colab:** the harness itself decodes every frame for Part B, and on 2 cores that alone takes ≈ 3.3 × realtime. So no solution fits the budget on Colab. The official machine has 8 CPU cores, where decoding is several times faster.
- **Speed choices:**
  - decoding only reference frames (PyAV `skip_frame=NONREF`);
  - batched GPU inference;
  - a background decoding thread;
  - an on-disk track cache for development (`WIUT_CACHE`).
- **Profiling:** `python -m tools.profile <video>` prints the time split.

## Determinism

- Fixed seed 42 for `random`, `numpy` and `torch`, with cuDNN in deterministic mode.
- Deterministic RANSAC via `cv2.setRNGSeed`.
- Part A output is identical across runs.
- The Part B time guard reacts to wall-clock time, so on a much slower machine the risk curve can be sparser.

## Repository layout

```
solution.py            interface used by run_submission.py (thin wrapper around src/)
run_submission.py      starter kit — unchanged
evaluate.py            starter kit — unchanged
examples/              starter kit example files
src/config.py          every threshold, model choice, frame stride and post-processing setting
src/video.py           probing + fast decoding (PyAV reference frames / OpenCV fallback)
src/tracker.py         YOLO + ByteTrack + traffic-light reading in one decoding pass, optional cache
src/tracks.py          kinematics table
src/scene.py           scene map, per-video homography alignment, traffic-light colour state
src/rules.py           one function per event class
src/postprocess.py     merge / shift / trim / validate segments
src/pipeline.py        Part A end to end
src/risk.py            Part B risk estimator
src/budget.py          shared time budget between Part A and Part B
scene/                 scene map + reference frame it was drawn on
labels/labels.txt      our dev labels of the sample videos
tools/                 labels, EDA, rendering, profiling, synthetic test clips, Colab helpers (dev only)
tests/                 rule unit tests on synthetic tracks  (python -m pytest tests -q)
demo/app.py            live demo (CPU)
notebooks/             Colab notebook used for development
predictions_samples.json   our output on the sample videos
```

## Limitations and what we would do next

- **Part B is uncalibrated.** The samples contain no accidents, so the 0.5 alarm threshold was set by reasoning, not data. Next step: calibrate on public crash datasets (DoTA, CCD).
- **Rare classes are not predicted.** illegal U-turn, near miss and road obstacle had 0–1 examples. A learned clip classifier trained on public data would help.
- **Some rules are weak.** `solid_line_crossing` and `failure_to_yield` depend on accurate ground points of partially occluded vehicles.
- **Labels may not match the organisers'.** Our dev labels come from one team and may be interpreted differently from the official annotators.

## Datasets, models and licences

- YOLO11m / YOLO11s, COCO pre-trained (Ultralytics), AGPL-3.0. Not fine-tuned.
- No external training data. Dev labels were written by us on the provided sample videos.
- Open-source code: Ultralytics (YOLO, ByteTrack) AGPL-3.0; OpenCV Apache-2.0; PyAV BSD; NumPy/pandas BSD.

## Team

Team details will be added. The roles below describe the work split:

| Member | Role | What they did | Links |
|---|---|---|---|
| _Name_ | _Role_ | _TBD_ | _GitHub / LinkedIn_ |
| _Name_ | _Role_ | _TBD_ | _GitHub / LinkedIn_ |
| _Name_ | _Role_ | _TBD_ | _GitHub / LinkedIn_ |
