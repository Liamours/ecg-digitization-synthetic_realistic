#!/bin/bash
# Digitize every page of a synthetic dataset, then score the run against its labels. Resumable like src.digitize.run.
# Usage: scripts/eval_synthetic.sh <dataset dir> <run dir> [layout]
set -e
cd "$(dirname "$0")/.."
uv run python -m src.digitize.run --config configs/digitize.yml --layout "${3:-mac400}" --out-dir "$2" "$1/pages"
uv run python -m src.digitize.score --dataset "$1" --run "$2"
