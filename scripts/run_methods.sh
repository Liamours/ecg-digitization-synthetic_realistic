#!/bin/bash
# Run every trace and lead-separation method on a saved front stage, one after another (one heavy job at a time), then
# compare them. Each run is resumable and writes its own log with an ETA. Needs the GPU environment (scripts/setup_gpu_env.sh).
# Usage: scripts/run_methods.sh [front folder]        default: @inferences/front_mac400-scan
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
py=../../.venvs/digitize_gpu/Scripts/python.exe
[ -f "$py" ] || py=../../.venvs/digitize_gpu/bin/python
front="${1:-@inferences/front_mac400-scan}"
run() { "$py" -m src.digitize.methods --config configs/digitize.yml --layout mac400 --front "$front" --mask "$1" --split "$2" --device cuda | tail -1; }
for split in ecgtizer_fragmented ecgtizer_full ecgtizer_lazy openecg_lines baseline overlap; do run openecg "$split"; done
run trace_net baseline
"$py" -m src.digitize.compare --methods "${front/front_/methods_}"
