#!/bin/bash
# Overnight training queue, 2026-09-22. Sequential only -- single 6GB-VRAM
# GPU (hardware.md), and this laptop already crashed a dataloader once
# under concurrent CPU load. Order follows the pipeline's own step order
# (wiki/TODO.md item 11: corner detection -> panel+staple detection ->
# orientation -> rectification -> {lead_group, style, gain} -> digitization),
# not the TODO list's own numbering, per user request 2026-09-22.
# Every config capped at epochs=20, patience=5 tonight, per user request.
set -uo pipefail
cd "$(dirname "$0")/../repo/ecg-digitization-synthetic_realistic" || exit 1
LOG=../../logs/train_queue_20260922.log

run() {
  name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$LOG"
  uv run "$@" >> "../../logs/train_${name}_20260922.log" 2>&1
  echo "$(date '+%F %T') END $name exit=$?" | tee -a "$LOG"
}

run page_corners python -m src.page_corners.train --config configs/page_corners.yml
run panel_detection python -m src.detection.train --config configs/panel_detection.yml
run orientation_classifier python -m src.orientation_classifier.train --config configs/orientation_classifier.yml
run style_classification python -m src.detection.train_style_classifier --config configs/style_classification.yml
run gain_field_detection python -m src.detection.train --config configs/gain_field_detection.yml --resume
run digitization python -m src.digitization.train --config configs/digitization.yml
run segmentation python -m src.segmentation.train --config configs/segmentation.yml

echo "$(date '+%F %T') QUEUE DONE" | tee -a "$LOG"
