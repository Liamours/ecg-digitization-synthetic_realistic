"""Trace pixels: the Open-ECG-Digitizer U-Net probability (pretrained, the default), or a local-contrast mask for the
page-mode layouts. The MAC 400 threshold rules were removed on 2026-10-03: on the 39 scans they passed 278 of 523 leads
against 416 of 538 for the pretrained network."""
from pathlib import Path

import cv2
import numpy as np


def _drop_small(mask: np.ndarray, min_extent: int | None = None, min_area: int | None = None) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.zeros(n, bool)
    if min_extent is not None:
        small |= np.maximum(st[:, 2], st[:, 3]) < min_extent
    if min_area is not None:
        small |= st[:, 4] < min_area
    small[0] = False
    mask[small[lab]] = 0
    return mask


def local_contrast_mask(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Shaded photos: pixels much darker than the local paper level."""
    flat = gray.astype(np.int16) - cv2.medianBlur(gray, cfg["median_kernel"]).astype(np.int16)
    m = ((gray < cfg["gray_max"]) & (flat < -cfg["contrast"])).astype(np.uint8)
    return _drop_small(m, min_extent=cfg["min_extent"])


class UNetMask:
    """The Open-ECG-Digitizer segmentation network (external/Open-ECG-Digitizer); its trace class as a probability image."""

    SIGNAL_CLASS = 2

    def __init__(self, cfg: dict):
        import importlib.util

        import torch

        repo = Path(cfg["repo"]).resolve()
        spec = importlib.util.spec_from_file_location("openecg_unet", repo / "src" / "model" / "unet.py")  # by path: the repo's own package is also named src
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        UNet = module.UNet
        m = UNet(num_in_channels=3, num_out_channels=4, dims=[32, 64, 128, 256, 320, 320, 320, 320], depth=2)
        ck = torch.load(repo / cfg["weights"], weights_only=True, map_location="cpu")
        ck = ck[0] if isinstance(ck, tuple) else ck
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.items()})
        from src.digitize.ocr import resolve_device

        self.device = resolve_device(cfg.get("device", "cpu"))
        self.model = m.eval().to(self.device)
        self.max_side = cfg["max_side"]
        self.threshold = cfg["threshold"]
        self.torch = torch

    def probability(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        sh, sw = small.shape[:2]
        small = np.pad(small, ((0, -sh % 32), (0, -sw % 32), (0, 0)), mode="edge")  # pad to the network's stride; cropping and stretching back would distort the panel
        x = self.torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        x = (x - x.min()) / (x.max() - x.min())
        cuda = self.device == "cuda"
        with self.torch.no_grad(), self.torch.autocast(device_type=self.device, enabled=cuda):  # half precision on the GPU: the network is large for 6 GB
            p = self.torch.softmax(self.model(x).float(), dim=1)[0, self.SIGNAL_CLASS].cpu().numpy()[:sh, :sw]
        if cuda:
            self.torch.cuda.empty_cache()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask(self, prob: np.ndarray) -> np.ndarray:
        return (prob > self.threshold).astype(np.uint8)
