"""The Open-ECG-Digitizer U-Net as ONNX for ONNX Runtime on a phone, made lighter with ONNX Runtime's own tools.

Graph: input `image` (1, 3, H, W) at the fixed phone input size (`mobile.unet.input_size` in configs/digitize.yml), RGB
scaled to 0..1 by its own minimum and maximum; output `trace` (1, H, W), the softmax probability of the trace class.
Exported with the torch.export-based exporter (`dynamo=True`, the documented path; the TorchScript exporter is deprecated)
and fixed shapes, which leave the optimizer and static quantization more room. Writes into `out_dir`
(configs/export_onnx.yml):

- `unet.onnx`, `unet.ort`: float32; the `.ort` file is ONNX Runtime's mobile format with graph optimizations applied
- RapidOCR's text detection, recognition and angle models (`rapidocr_out_dir`): its own ONNX files copied, and their `.ort`
  files; not quantized
- `unet_int8.onnx`, `unet_int8.ort`: static int8 in QDQ format, per-channel weights, the ARM choice of the `quantize_static`
  docstring (QInt8 activations and weights, reduce_range off); static is what ONNX Runtime recommends for CNNs

The `.ort` files are checked against PyTorch on held-out synthetic panels (`check_crops`): largest probability difference and
share of mask pixels that change at the trace threshold; graph optimizations can change results. Calibration uses synthetic panels (`calibration_dir`, the training data of this project), prepared exactly as the pipeline
prepares a crop for the phone (`mobile.unet` scale and input size). The `.ort` files are written with the ONNX Runtime
version the Android app uses (`android_ort_version`); the script stops if the installed version differs. Every session and
PyTorch are capped at `threads`. Accuracy of each variant is measured by scripts/eval_mobile.sh on a synthetic set.

Usage:
    python -m src.export_onnx --config configs/export_onnx.yml
"""
import argparse
import gc
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rapidocr
import torch
import yaml
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import MinMaxCalibrater
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import paths
from src.digitize.mask import UNetMask, network_input


class TraceProbability(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module, channel: int):
        super().__init__()
        self.unet, self.channel = unet, channel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.unet(image), dim=1)[:, self.channel]


def collect_ranges(self, data_reader) -> None:
    """MinMaxCalibrater.collect_data of ONNX Runtime 1.30 with its ranges kept: with `max_intermediate_outputs` set, the upstream
    method clears the collected values before computing their ranges, so nothing is kept. This computes the ranges after every
    input and then clears, so memory does not grow with the number of calibration inputs."""
    while inputs := data_reader.get_next():
        outputs = zip(self.infer_session.get_outputs(), self.infer_session.run(None, inputs), strict=False)
        self.intermediate_outputs.append([value if o.name not in self.model_original_outputs else None for o, value in outputs])
        self.compute_data()
        self.clear_collected_data()
    if self.calibrate_tensors_range is None:
        raise ValueError("No data is collected.")


MinMaxCalibrater.collect_data = collect_ranges


class Crops(CalibrationDataReader):
    def __init__(self, crops: list[Path], prepare):
        self.items, self.prepare = iter(crops), prepare

    def get_next(self) -> dict | None:
        path = next(self.items, None)
        return None if path is None else {"image": self.prepare(cv2.imread(str(path)))}


def to_ort(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", str(path), "--output_dir", str(path.parent),
                    "--optimization_style", "Fixed"], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if ort.__version__ != cfg["android_ort_version"]:
        sys.exit(f"onnxruntime {ort.__version__} is installed, the app uses {cfg['android_ort_version']}: .ort files must be written with the app's version")
    dcfg = yaml.safe_load(paths.resolve(cfg["digitize_config"]).read_text(encoding="utf-8"))
    mobile = dcfg["mobile"]["unet"]
    width, height = mobile["input_size"]
    torch.set_num_threads(cfg["threads"])
    out = paths.resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    unet = UNetMask({**dcfg["unet"], "weights": paths.resolve(dcfg["unet"]["weights"]), "device": "cpu", "backend": "torch"})
    model = TraceProbability(unet.model, UNetMask.SIGNAL_CLASS).eval()

    ocr_out = paths.resolve(cfg["rapidocr_out_dir"])   # RapidOCR ships its models as ONNX: copied as they are, then written as .ort, not quantized (they are small, and int8 text
    ocr_out.mkdir(parents=True, exist_ok=True)         # recognition can misread the digits of the printed gain)
    for name in cfg["rapidocr_files"]:
        shutil.copy2(Path(rapidocr.__file__).parent / "models" / name, ocr_out / name)
        if name.endswith(".onnx"):
            to_ort(ocr_out / name)

    fp32 = out / "unet.onnx"
    torch.onnx.export(model, (torch.rand(1, 3, height, width),), str(fp32), input_names=["image"], output_names=["trace"],
                      opset_version=cfg["opset"], dynamo=True)
    to_ort(fp32)

    crops = sorted(paths.resolve(cfg["calibration_dir"]).glob("*.png"))
    random.Random(cfg["seed"]).shuffle(crops)
    prepare = lambda bgr: network_input(bgr, dcfg["unet"]["max_side"], mobile["scale"], mobile["input_size"])[0]
    check = [prepare(cv2.imread(str(c))) for c in crops[cfg["calibration_crops"]:cfg["calibration_crops"] + cfg["check_crops"]]]
    with torch.no_grad():
        reference = [model(torch.from_numpy(x))[0].numpy() for x in check]
    threshold = dcfg["unet"]["threshold"]
    print(f"input {width} x {height}; against PyTorch on {len(check)} synthetic panels not used for calibration\n\n"
          "| File | MB | Largest probability difference | Mask pixels changed |\n|---|---|---|---|", flush=True)

    def parity(path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = cfg["threads"], 1
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        got = [sess.run(None, {"image": x})[0][0] for x in check]
        worst = max(float(np.abs(g - r).max()) for g, r in zip(got, reference))
        changed = sum(int(((g > threshold) != (r > threshold)).sum()) for g, r in zip(got, reference)) / sum(r.size for r in reference)
        print(f"| {path.name} | {path.stat().st_size / 2**20:.1f} | {worst:.5f} | {changed:.6f} |", flush=True)

    parity(out / "unet.ort")
    del model, unet   # PyTorch is not needed for the calibration
    gc.collect()
    pre, int8 = out / "unet_pre.onnx", out / "unet_int8.onnx"
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), Crops(crops[:cfg["calibration_crops"]], prepare), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, reduce_range=False)
    pre.unlink()
    to_ort(int8)
    parity(out / "unet_int8.ort")


if __name__ == "__main__":
    main()
