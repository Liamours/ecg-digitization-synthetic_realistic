"""Calibration pulse: a rectangular step 1 mV tall, so its height in millimetres is the panel's gain in mm/mV."""
import cv2
import numpy as np

from src.digitize.record import Grid


def find_pulses(gray: np.ndarray, grid: Grid, gain: float, window: tuple[int, int, int, int], cfg: dict) -> list[dict]:
    """Rectangular components about `gain` mm tall and 4 to 9 mm wide inside window x0, x1, y0, y1 (crop pixels): [{bbox: [x, y, w, h], height_mm}]."""
    x0, x1, y0, y1 = window
    dark = np.zeros(gray.shape, np.uint8)
    dark[y0:y1, x0:x1] = gray[y0:y1, x0:x1] < cfg["gray_max"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    px_mm_x, px_mm_y = grid.px_per_mm
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (cfg["height_tolerance"][0] * gain <= h / px_mm_y <= cfg["height_tolerance"][1] * gain and cfg["width_mm"][0] <= w / px_mm_x <= cfg["width_mm"][1] and area > cfg["min_area"] and area <= cfg.get("solid_fill_max", 1.0) * w * h):  # the pulse is an outline, a filled square is the header icon
            continue
        ys, xs = np.nonzero(lab == i)
        mm = grid.to_mm(np.column_stack([xs, ys]).astype(np.float64))[:, 1]
        out.append({"bbox": [int(x), int(y), int(w), int(h)], "height_mm": float(np.percentile(mm, 99) - np.percentile(mm, 1))})
    return out


def gain_from_pulses(heights: list[float], allowed: list[float], tolerance: float) -> float | None:
    """Median pulse height snapped to an allowed gain (5 or 10 mm/mV) when within `tolerance` of it."""
    if not heights:
        return None
    h = float(np.median(heights))
    for g in allowed:
        if abs(h - g) <= tolerance * g:
            return g
    return None
