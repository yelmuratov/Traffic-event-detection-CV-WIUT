#!/usr/bin/env bash
# One command: install, fetch weights, run on the sample videos, validate.
set -euo pipefail
VIDEOS=${1:-samples}
pip install -q -r requirements.txt
bash weights/download.sh
python run_submission.py --videos "$VIDEOS" --out predictions_samples.json --team "${TEAM:-our-team}"
python evaluate.py --pred predictions_samples.json --validate-only
