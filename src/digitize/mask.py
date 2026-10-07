"""Trace pixels: the Open-ECG-Digitizer U-Net probability (pretrained, inference only), run by PyTorch or, for the phone
app, by ONNX Runtime on the exported graph (src.export_onnx). The MAC 400 threshold rules were removed on 2026-10-03: on
the 39 scans they passed 278 of 523 leads against 416 of 538 for the pretrained network."""
from pathlib import Path

import cv2
import numpy as np


def network_input(bgr: np.ndarray, max_side: int, scale: float = 1.0, size: list[int] | None = None) -> tuple[np.ndarray, int, int]:
    """(1, 3, H, W) float32 for the network: scaled by `scale` (at most to `max_side`, and to fit `size` when one is given),
    padded by edge pixels to `size` [width, height] or else to a multiple of 32 (cropping and stretching back would distort the
    panel), RGB, min-max scaled to 0..1; and the height and width before padding."""
    h, w = bgr.shape[:2]
    s = min(scale, max_side / max(h, w), *((size[0] / w, size[1] / h) if size else ()))
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    sh, sw = small.shape[:2]
    ph, pw = (size[1] - sh, size[0] - sw) if size else (-sh % 32, -sw % 32)
    small = np.pad(small, ((0, ph), (0, pw), (0, 0)), mode="edge")
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return (x - x.min()) / (x.max() - x.min()), sh, sw


class UNetMask:
    """The Open-ECG-Digitizer segmentation network (third_party/open_ecg_digitizer/unet.py, its published weights); its
    trace class as a probability image. `backend: onnx` runs `onnx` (the exported graph) with ONNX Runtime on the CPU."""

    SIGNAL_CLASS = 2

    def __init__(self, cfg: dict):
        self.max_side, self.scale, self.size = cfg["max_side"], cfg.get("scale", 1.0), cfg.get("input_size")
        self.threshold = cfg["threshold"]
        if cfg.get("backend") == "onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads, opts.inter_op_num_threads = cfg["threads"], 1   # all cores by default: the laptop stopped responding (overview/hardware.md, 2026-10-06)
            self.session = ort.InferenceSession(str(Path(cfg["onnx"])), opts, providers=["CPUExecutionProvider"])
            return
        import torch

        from third_party.open_ecg_digitizer.unet import UNet

        self.session = None
        m = UNet(num_in_channels=3, num_out_channels=4, dims=[32, 64, 128, 256, 320, 320, 320, 320], depth=2)
        ck = torch.load(Path(cfg["weights"]), weights_only=True, map_location="cpu")
        ck = ck[0] if isinstance(ck, tuple) else ck
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.items()})
        from src.digitize.ocr import resolve_device

        self.device = resolve_device(cfg.get("device", "cpu"))
        self.model = m.eval().to(self.device)
        self.torch = torch

    def probability(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x, sh, sw = network_input(bgr, self.max_side, self.scale, self.size)   # the probability is enlarged back to the crop before the threshold
        if self.session is not None:
            p = self.session.run(None, {"image": x})[0][0, :sh, :sw]
            return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        cuda = self.device == "cuda"
        with self.torch.no_grad(), self.torch.autocast(device_type=self.device, enabled=cuda):  # half precision on the GPU: the network is large for 6 GB
            p = self.torch.softmax(self.model(self.torch.from_numpy(x).to(self.device)).float(), dim=1)[0, self.SIGNAL_CLASS].cpu().numpy()[:sh, :sw]
        if cuda:
            self.torch.cuda.empty_cache()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask(self, prob: np.ndarray) -> np.ndarray:
        return (prob > self.threshold).astype(np.uint8)
