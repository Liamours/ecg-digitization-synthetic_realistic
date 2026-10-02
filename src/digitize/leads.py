"""Which trace pixels belong to which lead."""
import cv2
import numpy as np
from scipy.signal import find_peaks

RUN_GAP = 3
CROSS_TOL = 3
FOLLOW_RATE = 0.02


def rows_equal(mask: np.ndarray, rows: int) -> list[tuple[int, int]]:
    """Split the mask's vertical extent into `rows` bands at the emptiest rows between the row centres."""
    occupied = np.flatnonzero(mask.any(axis=1))
    ymin, ymax = int(occupied[0]), int(occupied[-1])
    prof = cv2.GaussianBlur(mask.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), 10).ravel()
    ys = np.arange(len(prof))
    cuts = [ymin]
    for k in range(1, rows):
        lo, hi = ymin + (ymax - ymin) * (k - 0.5) / rows, ymin + (ymax - ymin) * (k + 0.5) / rows
        seg = (ys >= lo) & (ys <= hi)
        cuts.append(int(ys[seg][np.argmin(prof[seg])]))
    cuts.append(ymax + 1)
    return [(cuts[i], cuts[i + 1]) for i in range(rows)]


def rows_by_peaks(mask: np.ndarray, rows: int) -> list[tuple[int, int]]:
    """Row bands from the `rows` strongest peaks of the smoothed row profile; cuts fall at the minima between neighbours."""
    prof = cv2.GaussianBlur(mask.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), 5).ravel()
    peaks, props = find_peaks(prof, distance=40, prominence=prof.max() * 0.03)
    peaks = np.sort(peaks[np.argsort(-props["prominences"])[:rows]])
    occupied = np.flatnonzero(mask.any(axis=1))
    cuts = [int(occupied[0])] + [int(a + np.argmin(prof[a:b])) for a, b in zip(peaks[:-1], peaks[1:])] + [int(occupied[-1]) + 1]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def split_rows(mask: np.ndarray, rows: int, mode: str) -> list[tuple[int, int]]:
    return rows_by_peaks(mask, rows) if mode == "peaks" else rows_equal(mask, rows)


def track_leads(mask: np.ndarray, bands: list[tuple[int, int]]) -> tuple[list[np.ndarray], float]:
    """Assign trace pixels to leads column by column by continuity.

    Each lead keeps a slowly moving baseline that starts at its row band's median. A vertical run of trace
    pixels goes to the lead whose baseline it contains, else to the nearest baseline within half a row pitch.
    A run containing two baselines (a spike crossing rows) goes to the nearest and counts as a crossing.
    Returns one mask per lead and the fraction of columns that had a crossing.
    """
    k = len(bands)
    base = np.array([np.median(np.nonzero(mask[b0:b1])[0]) + b0 if mask[b0:b1].any() else (b0 + b1) / 2 for b0, b1 in bands], dtype=float)
    pitch = float(np.median(np.diff(base))) if k > 1 else 100.0
    out = [np.zeros_like(mask) for _ in range(k)]
    crossings = 0
    cols = np.flatnonzero(mask.any(axis=0))
    for x in cols:
        ys = np.flatnonzero(mask[:, x])
        starts = ys[np.r_[True, np.diff(ys) > RUN_GAP]]
        ends = ys[np.r_[np.diff(ys) > RUN_GAP, True]]
        crossed = False
        for a, b in zip(starts, ends):
            inside = [i for i in range(k) if a - CROSS_TOL <= base[i] <= b + CROSS_TOL]
            crossed |= len(inside) > 1
            cand = inside or [i for i in range(k) if abs((a + b) / 2 - base[i]) < 0.5 * pitch]
            if not cand:
                continue
            i = min(cand, key=lambda j: abs((a + b) / 2 - base[j]))
            out[i][a:b + 1, x] = 1
            if b - a < 0.35 * pitch:
                base[i] += FOLLOW_RATE * ((a + b) / 2 - base[i])
        crossings += crossed
    return out, crossings / max(len(cols), 1)


MAX_GAP_COLUMNS = 80  # a lead that lost its pixels to a crossing trace keeps its last position this long


def _overlap(a: int, b: int, c: int, d: int) -> int:
    return max(0, min(b, d) - max(a, c))


def track_leads_overlap(mask: np.ndarray, bands: list[tuple[int, int]]) -> tuple[list[np.ndarray], float]:
    """Assign trace pixels to leads by overlap with each lead's run in the previous column.

    A run overlapping one lead's previous run continues that lead. A run that overlaps several (a stroke crossing rows
    merged them) goes to the lead with the largest overlap; the others get no pixels there and are interpolated later,
    and the column counts as a crossing. A run overlapping none starts from the nearest baseline within half a row pitch.
    Returns one mask per lead and the fraction of columns with a crossing.
    """
    k = len(bands)
    base = np.array([np.median(np.nonzero(mask[b0:b1])[0]) + b0 if mask[b0:b1].any() else (b0 + b1) / 2 for b0, b1 in bands], dtype=float)
    pitch = float(np.median(np.diff(base))) if k > 1 else 100.0
    out = [np.zeros_like(mask) for _ in range(k)]
    last: list[tuple[int, int, int] | None] = [None] * k  # a, b, column of the lead's latest run
    crossings = 0
    cols = np.flatnonzero(mask.any(axis=0))
    for x in cols:
        ys = np.flatnonzero(mask[:, x])
        starts = ys[np.r_[True, np.diff(ys) > RUN_GAP]]
        ends = ys[np.r_[np.diff(ys) > RUN_GAP, True]]
        crossed = False
        taken: dict[int, tuple[int, int]] = {}
        for a, b in zip(starts, ends):
            ov = [(_overlap(a - CROSS_TOL, b + CROSS_TOL, last[i][0], last[i][1]), i) for i in range(k) if last[i] is not None and x - last[i][2] <= MAX_GAP_COLUMNS]
            hits = [(o, i) for o, i in ov if o > 0]
            if hits:
                crossed |= len(hits) > 1
                i = max(hits)[1]
            else:
                cand = [i for i in range(k) if abs((a + b) / 2 - base[i]) < 0.5 * pitch]
                if not cand:
                    continue
                i = min(cand, key=lambda j: abs((a + b) / 2 - base[j]))
            out[i][a:b + 1, x] = 1
            lo, hi = taken.get(i, (a, b))
            taken[i] = (min(lo, a), max(hi, b))
        for i, (a, b) in taken.items():
            last[i] = (a, b, x)
            if b - a < 0.35 * pitch:
                base[i] += FOLLOW_RATE * ((a + b) / 2 - base[i])
        crossings += crossed
    return out, crossings / max(len(cols), 1)


def trim_early_starts(masks: list[np.ndarray], grid, min_lead_mm: float, margin_mm: float) -> list[np.ndarray]:
    """Cut a lead's pixels that lie left of where the other leads begin: the calibration step sits in front of the trace in one row.

    A lead whose first pixels start more than `min_lead_mm` left of the median start of all leads loses everything left of
    that median (minus `margin_mm`). Fewer than three leads: nothing is cut, there is no median to trust.
    """
    if len(masks) < 3:
        return masks
    xs = []
    for m in masks:
        ys, cx = np.nonzero(m)
        xs.append(grid.to_mm(np.column_stack([cx, ys]).astype(np.float64))[:, 0] if len(cx) else np.array([np.inf]))
    starts = np.array([np.percentile(x, 0.5) for x in xs])
    ref = float(np.median(starts[np.isfinite(starts)]))
    out = []
    for m, x, s in zip(masks, xs, starts):
        if np.isfinite(s) and s < ref - min_lead_mm:
            ys, cx = np.nonzero(m)
            m = m.copy()
            drop = x < ref - margin_mm
            m[ys[drop], cx[drop]] = 0
        out.append(m)
    return out
