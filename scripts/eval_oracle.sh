#!/bin/bash
# Headroom per step: run the oracle modes (true panels, then plus true gain, then plus true lead names) on a synthetic dataset and score each.
# Usage: scripts/eval_oracle.sh <dataset> <run name prefix>
set -e
cd "$(dirname "$0")/.."
for mode in panels gain labels; do
  uv run python -m src.digitize.oracle --config configs/digitize.yml --dataset "$1" --mode "$mode" --out-dir "@inferences/digitize_synthetic/$2_$mode"
  uv run python -m src.digitize.score --dataset "$1" --run "@inferences/digitize_synthetic/$2_$mode"
done
