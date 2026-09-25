#!/usr/bin/env bash
# Run once WITH internet before evaluation. Fetches open-weight YOLO models (AGPL-3.0, Ultralytics).
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
for w in yolo11m.pt yolo11s.pt yolo11n.pt; do
  [ -f "$w" ] || curl -L --fail -o "$w" "$BASE/$w"
done
ls -lh *.pt
