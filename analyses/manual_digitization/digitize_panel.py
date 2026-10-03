"""Digitize the three leads of one MAC 400 panel crop without rectifying it:
grid map from the panel's own dots, trace mask between the header and footer,
signals in mV over seconds, and a calibration pulse reading as a self-check."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from grid_map import fit_grid, to_mm

GRAY_THRESHOLD = 130
MIN_COMPONENT = 100
LABEL_BOXES = re.compile(r"^(I{1,3}|aV[RLFUlf]|a[UV][RLF]|[UV][1-6])$")
BIN_MM = 0.1
GLYPH_CORE = 2.5
LABEL_MAX_AREA = 1500
GLYPH_REACH = 4
GAP_PX = 45


def read_gain(ocr: list[dict]) -> float | None:
    for b in ocr:
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm/m", b["text"])
        if m:
            return float(m.group(1))
    return None


def gain_from_pulses(pulses: list[dict]) -> float | None:
    """A calibration pulse is 1 mV tall: its height in millimetres is the gain. Snapped to 5 or 10 mm/mV when within 20 percent."""
    hs = [p["height_mm"] for p in pulses]
    if not hs:
        return None
    h = float(np.median(hs))
    for g in (5.0, 10.0):
        if abs(h - g) <= 0.2 * g:
            return g
    return None


def zone_y(ocr: list[dict], height: int) -> tuple[int, int]:
    head = [b["box"][3] for b in ocr if re.search(r"MAC|1\.02|^GE", b["text"])]
    foot = [b["box"][1] for b in ocr if re.search(r"^Man|25mm|mm/m|ADS|MDC|For|INNOQ", b["text"])]
    return (max(head) + 6 if head else 0), (min(foot) - 6 if foot else height)


def components(mask: np.ndarray):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return n, lab, st


def find_pulses(gray: np.ndarray, grid: dict, gain: float, zone: tuple[int, int]) -> list[dict]:
    """Rectangular calibration pulses: components about `gain` millimetres tall and under 9 mm wide."""
    mask = (gray < GRAY_THRESHOLD).astype(np.uint8)
    n, lab, st = components(mask)
    a1, a2 = grid["a1"], grid["a2"]
    mm_x, mm_y = np.hypot(*a1), np.hypot(*a2)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (0.85 * gain <= h / mm_y <= 1.15 * gain and 4.0 <= w / mm_x <= 9.0 and zone[0] < y and y + h < zone[1] + 40 and area > 80):
            continue
        ys, xs = np.nonzero(lab == i)
        top = np.percentile(ys, 1)
        base = np.percentile(ys, 99)
        # plateau and baseline levels in grid millimetres, along the second grid axis
        up = to_mm(grid, np.array([[xs.mean(), top], [xs.mean(), base]]))
        out.append({"bbox": [int(x), int(y), int(w), int(h)], "height_mm": float(abs(up[1, 1] - up[0, 1])), "left": int(x)})
    return out


def label_zone(ocr: list[dict], labels: list[str]) -> tuple[int, int]:
    xs = [b["box"] for b in ocr if LABEL_BOXES.match(b["text"].strip())]
    if xs:
        return min(b[0] for b in xs) - 8, max(b[2] for b in xs) + 8
    return 255, 345


def trace_mask(gray: np.ndarray, zone: tuple[int, int], lz: tuple[int, int], pulses: list[dict]) -> np.ndarray:
    m = (gray < GRAY_THRESHOLD).astype(np.uint8)
    m[:zone[0]] = 0
    m[zone[1]:] = 0
    for p in pulses:
        x, y, w, h = p["bbox"]
        m[max(y - 4, 0):y + h + 4, max(x - 4, 0):x + w + 4] = 0
    # lead labels: compact components inside the label columns that are not part of the long trace
    n, lab, st = components(m)
    trace_ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= LABEL_MAX_AREA]
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if i not in trace_ids and lz[0] <= x + w / 2 <= lz[1] and w < 70:
            m[lab == i] = 0
    # a label touching the trace stays connected to it: remove thick pixels (strokes wider than the trace) around it
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    core = (dist >= GLYPH_CORE).astype(np.uint8)
    core[:, :lz[0]] = 0
    core[:, lz[1]:] = 0
    m[cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GLYPH_REACH + 1, 2 * GLYPH_REACH + 1))) > 0] = 0
    n, lab, st = components(m)
    m[np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < MIN_COMPONENT])] = 0
    return m


def cut_at_gap(mask: np.ndarray) -> np.ndarray:
    """Keep the first run of trace columns from the left; a gap of GAP_PX empty columns after the midpoint ends the panel's own trace."""
    occ = mask.any(axis=0)
    xs = np.flatnonzero(occ)
    x0 = xs[0]
    empty = 0
    for x in range(x0, mask.shape[1]):
        empty = 0 if occ[x] else empty + 1
        if empty >= GAP_PX and x > mask.shape[1] * 0.4:
            mask = mask.copy()
            mask[:, x:] = 0
            break
    return mask


def row_bands(mask: np.ndarray, rows: int = 3) -> list[tuple[int, int]]:
    ys_any = np.where(mask.any(axis=1))[0]
    ymin, ymax = int(ys_any[0]), int(ys_any[-1])
    prof = cv2.GaussianBlur(mask.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), 10).ravel()
    ys = np.arange(len(prof))
    cuts = [ymin]
    for k in range(1, rows):
        lo, hi = ymin + (ymax - ymin) * (k - 0.5) / rows, ymin + (ymax - ymin) * (k + 0.5) / rows
        seg = (ys >= lo) & (ys <= hi)
        cuts.append(int(ys[seg][np.argmin(prof[seg])]))
    cuts.append(ymax + 1)
    return [(cuts[i], cuts[i + 1]) for i in range(rows)]


def track_leads(mask: np.ndarray, bands: list[tuple[int, int]]) -> tuple[list[np.ndarray], list[float]]:
    """Assign trace pixels to leads column by column by continuity. Each lead keeps a slowly moving
    baseline (starting at its row band's median); a vertical run of trace pixels goes to the lead whose
    baseline it contains, else to the nearest baseline within half a row pitch. Runs containing two
    baselines (a spike crossing rows) go to the nearest and are counted as crossings. Returns one mask
    per lead and the fraction of columns that had a crossing."""
    k = len(bands)
    base = np.array([np.median(np.nonzero(mask[b0:b1])[0]) + b0 if mask[b0:b1].any() else (b0 + b1) / 2 for b0, b1 in bands], dtype=float)
    pitch = float(np.median(np.diff(base))) if k > 1 else 100.0
    out = [np.zeros_like(mask) for _ in range(k)]
    crossings = 0
    cols = np.flatnonzero(mask.any(axis=0))
    for x in cols:
        ys = np.flatnonzero(mask[:, x])
        starts = ys[np.r_[True, np.diff(ys) > 3]]
        ends = ys[np.r_[np.diff(ys) > 3, True]]
        crossed = False
        for a, b in zip(starts, ends):
            inside = [i for i in range(k) if a - 3 <= base[i] <= b + 3]
            if len(inside) > 1:
                crossed = True
            cand = inside or [i for i in range(k) if abs((a + b) / 2 - base[i]) < 0.5 * pitch]
            if not cand:
                continue
            i = min(cand, key=lambda j: abs((a + b) / 2 - base[j]))
            out[i][a:b + 1, x] = 1
            if b - a < 0.35 * pitch:
                base[i] += 0.02 * ((a + b) / 2 - base[i])
        crossings += crossed
    return out, [crossings / max(len(cols), 1)]


def extract(mask: np.ndarray, band: tuple[int, int], grid: dict, gain: float, x_range: tuple[float, float], mm_per_s: float = 25.0):
    ys, xs = np.nonzero(mask[band[0]:band[1]])
    pts = np.column_stack([xs, ys + band[0]]).astype(np.float64)
    mm = to_mm(grid, pts)
    x_mm, y_mm = mm[:, 0], mm[:, 1]
    edges = np.arange(x_range[0], x_range[1] + BIN_MM, BIN_MM)
    which = np.digitize(x_mm, edges) - 1
    n = len(edges) - 1
    med = np.full(n, np.nan)
    for b in np.unique(which):
        if 0 <= b < n:
            med[b] = np.median(y_mm[which == b])
    local = np.array([np.nanmedian(med[max(0, i - 120):i + 120]) if np.isfinite(med[max(0, i - 120):i + 120]).any() else np.nan for i in range(n)])
    out = np.full(n, np.nan)
    for b in np.unique(which):
        if 0 <= b < n:
            v = y_mm[which == b]
            out[b] = v[np.argmax(np.abs(v - local[b]))]
    ok = np.isfinite(out)
    base = np.nanmedian(out)
    sig = np.interp(np.arange(n), np.flatnonzero(ok), out[ok])
    t = (np.arange(n) * BIN_MM) / mm_per_s  # seconds from the start of the panel's shared time axis
    x_axis = edges[:-1] + BIN_MM / 2
    return t, -(sig - base) / gain, ok, np.column_stack([x_axis, sig]), float(base)


def digitize(crop_path: Path, ocr: list[dict], labels: list[str], rows: int = 3, x_limits: tuple[int, int] | None = None, mask_override: np.ndarray | None = None) -> dict:
    gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    grid = fit_grid(gray)
    gain = read_gain(ocr)
    gain_source = "ocr"
    zone = zone_y(ocr, gray.shape[0])
    if gain is None:  # search pulses at either gain and let their height decide
        found = find_pulses(gray, grid, 10.0, zone) + find_pulses(gray, grid, 5.0, zone)
        gain, gain_source = gain_from_pulses(found), "pulse"
    if gain is None:
        raise ValueError("gain not read and no calibration pulse found")
    pulses = find_pulses(gray, grid, gain, zone)
    if mask_override is not None:  # a learned trace mask: only keep the zone between header and footer
        mask = mask_override.astype(np.uint8).copy()
        mask[:zone[0]] = 0
        mask[zone[1]:] = 0
    else:
        mask = trace_mask(gray, zone, label_zone(ocr, labels), pulses)
    if x_limits:  # the crop carries a margin of the neighbouring panels; keep only this panel's own columns
        mask[:, :x_limits[0]] = 0
        mask[:, x_limits[1]:] = 0
    mask = cut_at_gap(mask)
    bands = row_bands(mask, rows)
    lead_masks, cross = track_leads(mask, bands)
    ys, xs = np.nonzero(mask)
    x_all = to_mm(grid, np.column_stack([xs, ys]).astype(np.float64))[:, 0]
    x_range = (float(x_all.min()), float(x_all.max()))
    leads = {}
    for name, band in zip(labels, bands):
        t, mv, ok, pts_mm, base = extract(lead_masks[labels.index(name)], (0, mask.shape[0]), grid, gain, x_range)
        leads[name] = {"t": t, "mv": mv, "coverage": float(ok.mean()), "pts_mm": pts_mm, "band": band, "base_mm": base, "x0_mm": x_range[0], "crossing_fraction": cross[0]}
    return {"grid": grid, "gain": gain, "gain_source": gain_source, "zone": zone, "pulses": pulses, "mask": mask, "leads": leads, "gray": gray}


PANEL_LABELS = {0: ["I", "II", "III"], 1: ["aVR", "aVL", "aVF"], 2: ["V1", "V2", "V3"], 3: ["V4", "V5", "V6"]}


def digitize_panels(inputs: Path) -> dict:
    """Digitize every panel with a label set, from the crops, OCR boxes and panel boxes saved in `inputs`."""
    meta = {m["panel"]: m for m in json.load(open(inputs / "panels_meta.json"))}
    ocr = json.load(open(inputs / "ocr.json"))
    out = {}
    for i, labels in PANEL_LABELS.items():
        m = meta[i]
        limits = (m["box"][0] - m["crop_origin"][0], m["box"][2] - m["crop_origin"][0])
        out[i] = digitize(inputs / f"panel{i}.png", ocr[str(i)], labels, x_limits=limits)
    return out
