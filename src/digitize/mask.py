"""Trace pixels: a threshold for light dot-grid prints, a local-contrast threshold for shaded photos, or the
Open-ECG-Digitizer U-Net probability for prints where the grid competes with the trace."""
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


def drop_printed_text(m: np.ndarray, boxes: list[list[float]], max_px: float, pad_px: float) -> np.ndarray:
    """Remove glyph-sized components that sit inside an OCR text box: printed date, time, rate and device text is not trace.

    A component stays when it is larger than `max_px` in either direction (the trace itself) or when less than 60 percent
    of its box lies inside a text box (a trace fragment that only passes near a text box)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _a = st[i]
        if max(w, h) > max_px:
            continue
        for b in boxes:
            iw = min(x + w, b[2] + pad_px) - max(x, b[0] - pad_px)
            ih = min(y + h, b[3] + pad_px) - max(y, b[1] - pad_px)
            if iw > 0 and ih > 0 and iw * ih >= 0.6 * w * h:
                m[lab == i] = 0
                break
    return m


def drop_solid_blobs(m: np.ndarray, min_px: float, fill: float, max_px: float = float("inf")) -> np.ndarray:
    """Remove thick filled icon-sized components (the header icon, stains): a trace is a thin line, its box is mostly empty.

    A component longer than `max_px` is a shadow or dark paper, not an icon, and stays for the threshold to deal with."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min(w, h) >= min_px and max(w, h) <= max_px and a >= fill * w * h:
            m[lab == i] = 0
    return m


def threshold_mask(gray: np.ndarray, zone: tuple[int, int], label_cols: tuple[int, int], pulse_boxes: list[list[int]], cfg: dict,
                   text_boxes: list[list[float]] | None = None, px_per_mm: float | None = None) -> np.ndarray:
    """MAC 400 style: dark pixels between header and footer, minus the calibration pulses and the lead-name glyphs."""
    m = (gray < cfg["gray_max"]).astype(np.uint8)
    if px_per_mm and "dark_share_max" in cfg and m.mean() > cfg["dark_share_max"]:  # a shadowed crop: the fixed threshold marks the paper itself, judge each pixel against its own surroundings
        bg = cv2.medianBlur(gray, int(cfg["bg_kernel_mm"] * px_per_mm) | 1)
        m = (gray < cfg["dark_ratio"] * bg).astype(np.uint8)
    elif px_per_mm and "contrast_min" in cfg:  # a thin gray trace (a downscaled or faded print) is far darker than the paper around it though not below gray_max
        k = int(cfg["bg_kernel_mm"] * px_per_mm) | 1
        bg = cv2.medianBlur(gray, k)
        m |= ((bg.astype(np.int16) - gray.astype(np.int16) >= cfg["contrast_min"]) & (gray < cfg["contrast_gray_max"])).astype(np.uint8)
    m[:zone[0]] = 0
    m[zone[1]:] = 0
    for x, y, w, h in pulse_boxes:
        m[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < cfg["label_max_area"] and label_cols[0] <= x + w / 2 <= label_cols[1] and w < cfg["label_max_width"]:
            m[lab == i] = 0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)  # a label touching the trace stays connected: remove pixels near thick strokes
    core = (dist >= cfg["glyph_core"]).astype(np.uint8)
    core[:, :label_cols[0]] = 0
    core[:, label_cols[1]:] = 0
    reach = 2 * cfg["glyph_reach"] + 1
    m[cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > 0] = 0
    if px_per_mm and "text_blob_max_mm" in cfg:
        m = drop_printed_text(m, text_boxes or [], cfg["text_blob_max_mm"] * px_per_mm, cfg["text_box_pad_mm"] * px_per_mm)
        m = drop_solid_blobs(m, cfg["solid_min_mm"] * px_per_mm, cfg["solid_fill"], cfg["solid_max_mm"] * px_per_mm)
    return _drop_small(m, min_area=cfg["min_component"])


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
        small = cv2.resize(bgr, (round(w * s) // 32 * 32, round(h * s) // 32 * 32), interpolation=cv2.INTER_AREA) if s < 1 else bgr[:h // 32 * 32, :w // 32 * 32]
        x = self.torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        x = (x - x.min()) / (x.max() - x.min())
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(x), dim=1)[0, self.SIGNAL_CLASS].cpu().numpy()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask(self, prob: np.ndarray) -> np.ndarray:
        return (prob > self.threshold).astype(np.uint8)
