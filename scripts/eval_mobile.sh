#!/bin/bash
# Score the phone trace network against the current one on synthetic pages with exact labels, one change at a time.
# The page text, rotation, panels, grid and gain are read once (src.digitize.front, on the GPU); only the trace stage is run
# per variant (src.digitize.methods: Open-ECG U-Net, ecgtizer `fragmented`), so each variant costs the network alone:
#   torch        the current network (PyTorch, full-size crops)                                   GPU
#   torch_input  + the phone input: crops at half size, padded to the fixed input size            GPU  -> effect of the input
#   onnx_fp32    + the float32 ONNX graph                                                         CPU  -> effect of the export
#   onnx_int8    + the static int8 graph: the phone network (`mobile` in configs/digitize.yml)    CPU  -> effect of quantization
# The page text is read with the current OCR in every variant; the ONNX OCR is not part of this comparison.
# Use pages the int8 model was not calibrated on (configs/export_onnx.yml calibrates on ecg-synthetic-260930-1856).
# Resumable: finished pages are skipped. Start it at below-normal priority (overview/hardware.md).
# Usage: scripts/eval_mobile.sh [dataset alias] [pages] [out folder]     defaults: @synthetic-260930-2311 10 @inferences/eval_mobile
set -e
cd "$(dirname "$0")/.."
data="${1:-@synthetic-260930-2311}"
pages="${2:-10}"
out="${3:-@inferences/eval_mobile}"
py="../../.venvs/digitize_gpu/Scripts/python.exe"
export PYTHONPATH=. PYTHONIOENCODING=utf-8   # torch.onnx prints an emoji, which the Windows code page cannot write to a redirected output
dir=$("$py" -c "from src import paths; print(paths.resolve('$data/pages').as_posix())")
"$py" -m src.digitize.front --config configs/digitize.yml --layout mac400 --device cuda --out-dir "$out/front" $(ls "$dir"/*.png | head -n "$pages")
trace() {
  "$py" -m src.digitize.methods --config configs/digitize.yml --layout mac400 --front "$out/front" --mask openecg --split ecgtizer_fragmented --out-dir "$out/$1" "${@:2}"
  "$py" -m src.digitize.score --dataset "$data" --run "$out/$1"
}
trace torch --device cuda
trace torch_input --device cuda --backend onnx --unet backend=torch
trace onnx_fp32 --device cpu --backend onnx --unet onnx=@models/open_ecg_digitizer-onnx/unet.ort
trace onnx_int8 --device cpu --backend onnx
