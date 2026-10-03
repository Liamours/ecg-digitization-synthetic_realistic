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
    """Remove components that sit inside an OCR text box: printed date, time, rate and device text is not trace.

    Only boxes no taller than `max_px` count as text lines (a taller box is OCR noise over the trace). Touching letters
    form components as wide as a word, so the size of the component is not limited; a component stays when less than 60
    percent of its box lies inside a text box (the trace itself, or a fragment that only passes near a text box)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    x, y, w, h = (st[:, i].astype(np.float64) for i in range(4))
    kill = np.zeros(n, bool)
    for b in boxes:
        if b[3] - b[1] > max_px:
            continue
        iw = np.minimum(x + w, b[2] + pad_px) - np.maximum(x, b[0] - pad_px)
        ih = np.minimum(y + h, b[3] + pad_px) - np.maximum(y, b[1] - pad_px)
        kill |= (iw > 0) & (ih > 0) & (iw * ih >= 0.6 * w * h)
    kill[0] = False  # label 0 is the background
    m[kill[lab]] = 0
    return m


def drop_solid_blobs(m: np.ndarray, min_px: float, fill: float, max_px: float = float("inf")) -> np.ndarray:
    """Remove thick filled icon-sized components (the header icon, stains): a trace is a thin line, its box is mostly empty.

    A component longer than `max_px` is a shadow or dark paper, not an icon, and stays for the threshold to deal with."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    kill = (np.minimum(w, h) >= min_px) & (np.maximum(w, h) <= max_px) & (a >= fill * w * h)
    kill[0] = False
    m[kill[lab]] = 0
    return m


def threshold_mask(gray: np.ndarray, zone: tuple[int, int], label_cols: tuple[int, int], pulse_boxes: list[list[int]], cfg: dict,
                   text_boxes: list[list[float]] | None = None, px_per_mm: float | None = None, label_boxes: list[list[float]] | None = None) -> np.ndarray:
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
    centre = st[:, 0] + st[:, 2] / 2
    kill = (st[:, 4] < cfg["label_max_area"]) & (label_cols[0] <= centre) & (centre <= label_cols[1]) & (st[:, 2] < cfg["label_max_width"])
    kill[0] = False
    m[kill[lab]] = 0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)  # a label touching the trace stays connected: remove pixels near thick strokes
    core = (dist >= cfg["glyph_core"]).astype(np.uint8)
    core[:, :label_cols[0]] = 0
    core[:, label_cols[1]:] = 0
    reach = 2 * cfg["glyph_reach"] + 1
    m[cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > 0] = 0
    if label_boxes and px_per_mm and "label_core" in cfg:  # read lead names: their thick strokes go, wherever the label sits on a tilted page
        pad = int(cfg["label_pad_mm"] * px_per_mm)
        lcore = np.zeros_like(core)
        for bx0, by0, bx1, by1 in label_boxes:
            sy, sx = slice(max(int(by0) - pad, 0), int(by1) + pad), slice(max(int(bx0) - pad, 0), int(bx1) + pad)
            lcore[sy, sx] = dist[sy, sx] >= cfg["label_core"]
        lreach = 2 * cfg["label_reach"] + 1
        m[cv2.dilate(lcore, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lreach, lreach))) > 0] = 0
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
