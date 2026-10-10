"""Trace pixels of an upright MAC 400 panel and their separation into the three leads.

The mask is plain thresholding of the normalised image followed by removal of the grid dots (small isolated
components) and of the printed text found by the OCR. Lead separation is a dynamic-programming path per lead over the
vertical runs of trace pixels in each column: a path pays for every pixel it jumps between neighbouring columns, for
columns it has to bridge without ink, and (softly) for drifting away from its own row. That keeps a lead on its own
line when a large QRS of the neighbouring lead reaches into its row, which is where row-band methods fail.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def column_runs(col: np.ndarray, merge_gap: int = 1) -> list[tuple[int, int]]:
    ys = np.flatnonzero(col)
    if ys.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(ys) > merge_gap + 1)
    starts = np.r_[ys[0], ys[breaks + 1]]
    ends = np.r_[ys[breaks], ys[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class LeadPath:
    y: np.ndarray        # value per column (px, crop coordinates), NaN where the lead had no pixel
    lo: np.ndarray       # top of the chosen run per column (px)
    hi: np.ndarray       # bottom of the chosen run per column (px)
    own: np.ndarray      # bool: the column had a run on this lead's path
    shared: np.ndarray   # bool: that run was also on another lead's path


def track_leads(mask: np.ndarray, x0: int, x1: int, rows: list[float], s: float, passes: int = 2) -> list[LeadPath]:
    """Two passes: the first with fixed row heights, the second with each lead's baseline as found by the first
    (a running median), so that baseline wander does not pull a lead out of its band."""
    n = x1 - x0
    curves = [np.full(n, float(r)) for r in rows]
    paths = _track_once(mask, x0, x1, curves, s)
    for _ in range(passes - 1):
        w = int(round(15 * s))  # 0.6 s of paper
        new = []
        for p, c in zip(paths, curves):
            y = np.where(p.own, (p.lo + p.hi) / 2, np.nan)
            idx = np.flatnonzero(np.isfinite(y))
            if idx.size < 0.3 * n:
                new.append(c)
                continue
            yy = np.interp(np.arange(n), idx, y[idx])
            med = np.array([np.median(yy[max(0, i - w):i + w + 1]) for i in range(0, n, 4)])
            m = np.interp(np.arange(n), np.arange(0, n, 4), med)
            new.append(m)
        # keep the order and at least 5 mm between neighbouring baselines
        for i in range(1, len(new)):
            new[i] = np.maximum(new[i], new[i - 1] + 5 * s)
        curves = new
        paths = _track_once(mask, x0, x1, curves, s)
    return paths


def _track_once(mask: np.ndarray, x0: int, x1: int, rows: list[np.ndarray], s: float,
                band_mm: float = 7.0, skip_mm: float = 1.6, w_prior: float = 0.6, gap_cost_mm: float = 0.8, w_comp: float = 0.5) -> list[LeadPath]:
    """One DP path per row over the columns x0..x1 of the mask; rows are baseline heights per column."""
    cols = list(range(x0, x1))
    runs = [column_runs(mask[:, x]) for x in cols]
    n = len(cols)
    K = max(2, int(round(skip_mm * s)))
    gap_cost = gap_cost_mm * s
    band = band_mm * s
    paths = []
    all_runs = runs
    for li, r in enumerate(rows):
        # every column also offers a virtual run on the lead's own baseline: coasting there costs like a gap, so a lead
        # whose ink is faded for a stretch waits on its row instead of borrowing a neighbour's trace
        runs = [rr + [(int(r[i]) - 1, int(r[i]) + 1)] for i, rr in enumerate(all_runs)]
        virt = [len(rr) - 1 for rr in runs]
        best = [np.full(len(rr), np.inf) for rr in runs]
        back = [np.full((len(rr), 2), -1, np.int64) for rr in runs]  # (column offset index, run index)
        others = [ro for lj, ro in enumerate(rows) if lj != li]

        def idist(a, b, y):
            return 0.0 if a <= y <= b else min(abs(a - y), abs(b - y))

        prior = [np.array([w_prior * max(0.0, idist(a, b, r[i]) - band)
                           + w_comp * max(0.0, idist(a, b, r[i]) - min([idist(a, b, ro[i]) for ro in others] or [1e9]) - 1.5 * s)
                           for a, b in rr]) for i, rr in enumerate(runs)]
        for i in range(n):
            prior[i][virt[i]] = gap_cost
        # a run that sits close to another row is penalised a little more (keeps paths apart at crossings)
        for i in range(n):
            rr = runs[i]
            if not rr:
                continue
            A = np.array(rr, np.float64)
            # start anywhere in the first K columns, paying the columns skipped
            start = prior[i] + (i * gap_cost if i < K else np.inf)
            cand = start.copy()
            bk = np.full((len(rr), 2), -1, np.int64)
            for d in range(1, K + 1):
                j = i - d
                if j < 0:
                    break
                if not runs[j] or not np.isfinite(best[j]).any():
                    continue
                B = np.array(runs[j], np.float64)
                # vertical gap between the intervals (0 when they overlap or touch)
                gap = np.maximum(0.0, np.maximum(A[:, None, 0] - B[None, :, 1], B[None, :, 0] - A[:, None, 1]) - 1)
                c = best[j][None, :] + gap + (d - 1) * gap_cost
                jj = np.argmin(c, axis=1)
                cc = c[np.arange(len(rr)), jj] + prior[i]
                upd = cc < cand
                cand[upd] = cc[upd]
                bk[upd, 0] = j
                bk[upd, 1] = jj[upd]
            best[i] = cand
            back[i] = bk
        # end in any of the last K columns
        end_i, end_r, end_c = -1, -1, np.inf
        for i in range(max(0, n - K), n):
            if runs[i] and np.isfinite(best[i]).any():
                k = int(np.argmin(best[i]))
                c = best[i][k] + (n - 1 - i) * gap_cost
                if c < end_c:
                    end_i, end_r, end_c = i, k, c
        y = np.full(n, np.nan); lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        own = np.zeros(n, bool)
        i, k = end_i, end_r
        chosen = {}
        while i >= 0 and k >= 0:
            chosen[i] = k
            j, kk = back[i][k]
            i, k = int(j), int(kk)
        for i, k in chosen.items():
            if k == virt[i]:
                continue
            a, b = runs[i][k]
            lo[i], hi[i] = a, b
            own[i] = True
        paths.append((lo, hi, own, chosen))
    runs = all_runs
    # values: inside a tall run the lead moves; take the end that continues the motion.
    # First pass: the run centre of every lead, to know where the other leads are.
    centre = [np.where(own, (lo + hi) / 2, np.nan) for lo, hi, own, _ in paths]
    out = []
    for li, (lo, hi, own, chosen) in enumerate(paths):
        shared = np.zeros(n, bool)
        for lj, (lo2, hi2, own2, ch2) in enumerate(paths):
            if lj == li:
                continue
            same = own & own2 & (lo == lo2) & (hi == hi2)
            # the other lead's own centre (thin run) lies inside this lead's tall run: the run is two traces touching
            c2 = centre[lj]
            thin2 = own2 & ((hi2 - lo2) <= 0.6 * s)
            inside = own & thin2 & (c2 >= lo - 1) & (c2 <= hi + 1) & ((hi - lo) > 0.6 * s)
            shared |= same | inside
        y = np.full(n, np.nan)
        prev = rows[li][0]
        idx = np.flatnonzero(own)
        mid = (lo + hi) / 2
        for k, i in enumerate(idx):
            a, b = lo[i], hi[i]
            if b - a <= 0.5 * s:
                v = (a + b) / 2
            elif shared[i]:
                v = a if abs(a - prev) < abs(b - prev) else b
            else:
                # a slope crossing the column: its middle; a turning point (peak or trough): its outer end
                pm = mid[idx[k - 1]] if k > 0 else None
                nm = mid[idx[k + 1]] if k + 1 < len(idx) else None
                if pm is not None and nm is not None and pm > mid[i] and nm > mid[i]:
                    v = a            # neighbours lower on paper: a peak, take the top
                elif pm is not None and nm is not None and pm < mid[i] and nm < mid[i]:
                    v = b            # a trough, take the bottom
                elif pm is None or nm is None:
                    v = a if abs(a - prev) > abs(b - prev) else b
                else:
                    v = (a + b) / 2
            y[i] = v
            prev = v
        out.append(LeadPath(y, lo, hi, own, shared))
    return out


def find_pulse(mask: np.ndarray, mx: float, my: float, s: float, img: np.ndarray | None = None) -> dict | None:
    """The pulse from the trace mask; failing that, from line-shaped light ink in the grey image (a faint pulse breaks
    into pieces that the dot filter of the mask removes)."""
    p = _find_pulse(mask, mx, my, s)
    if p is None and img is not None:
        lines = _light_lines(img, mx, my, s)
        p = _find_pulse(lines, mx, my, s)
        if p is not None:
            p["source"] = "light lines"
    return p


def _light_lines(img: np.ndarray, mx: float, my: float, s: float) -> np.ndarray:
    """Vertical and horizontal line pieces of ink lighter than the mask threshold, in the pulse search window."""
    out = np.zeros(img.shape, bool)
    x0, x1 = int(max(0, mx - 16 * s)), int(max(0, mx - 1.5 * s))
    y0, y1 = int(max(0, my + 15 * s)), int(min(img.shape[0], my + 70 * s))
    b = (img[y0:y1, x0:x1] < 215).astype(np.uint8)
    L = max(3, int(round(1.2 * s)))
    ver = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    hor = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    out[y0:y1, x0:x1] = (ver | hor) > 0
    return out


def _find_pulse(mask: np.ndarray, mx: float, my: float, s: float) -> dict | None:
    """The calibration pulse left of the traces: a rectangle whose height in mm is the gain in mm/mV.

    Vertical strokes are searched in a horizontally thickened mask (a stroke may lean by a pixel) with gaps up to
    up to 3 mm bridged as long as 60 % of the stroke is inked (light thermal print breaks up). Two sides of equal height about 5 mm apart make the pulse; failing
    that, one side with a horizontal top bar of at least 3 mm.
    """
    x0, x1 = int(max(0, mx - 16 * s)), int(max(0, mx - 1.5 * s))
    y0, y1 = int(max(0, my + 15 * s)), int(min(mask.shape[0], my + 70 * s))
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    if sub.size == 0:
        return None
    thick = cv2.dilate(sub, np.ones((1, 3), np.uint8)) > 0
    gap = max(2, int(3.0 * s))
    best = []
    for x in range(thick.shape[1]):
        col = thick[:, x]
        for a, b in column_runs(col, merge_gap=gap):
            L = (b - a + 1) / s
            if 2.0 <= L <= 22 and col[a:b + 1].mean() >= 0.6:
                best.append((x, a, b, L))
    if not best:
        return None
    best.sort()
    groups = []
    for st in best:
        if groups and st[0] - groups[-1][-1][0] <= 2:
            groups[-1].append(st)
        else:
            groups.append([st])
    sides = [max(g, key=lambda t: t[3]) for g in groups]
    pairs = []
    for i in range(len(sides)):
        for j in range(i + 1, len(sides)):
            a, b = sides[i], sides[j]
            dx = (b[0] - a[0]) / s
            if 2.5 <= dx <= 10 and abs(a[3] - b[3]) < 0.15 * max(a[3], b[3]) and abs(a[1] - b[1]) < 1.0 * s:
                # the top bar joins them
                row = thick[max(0, min(a[1], b[1]) - 2):max(a[1], b[1]) + 3, a[0]:b[0] + 1].any(axis=0)
                if row.mean() > 0.8:
                    pairs.append((abs(a[3] - b[3]), a, b))
    if pairs:
        _, a, b = min(pairs)
        height = (a[3] + b[3]) / 2
        return {"height_mm": float(height), "x_left": float(a[0] + x0), "x_right": float(b[0] + x0), "top": float(min(a[1], b[1]) + y0),
                "bottom": float(max(a[2], b[2]) + y0), "sides": 2}
    # one side and its top bar
    singles = []
    for a in sides:
        x, top, bot, L = a
        for direction in (-1, 1):
            xs = range(x, max(-1, x - int(10 * s)), -1) if direction < 0 else range(x, min(thick.shape[1], x + int(10 * s)))
            run = 0
            for xx in xs:
                if thick[max(0, top - 2):top + 3, xx].any():
                    run += 1
                else:
                    break
            if 4.0 <= run / s <= 7.5:
                xf = x + direction * (run - 1)
                far = thick[top:bot + 1, max(0, xf - 2):xf + 3].any(axis=1)
                if far.mean() >= 0.4:   # the other side is there, only broken up
                    singles.append((L, a, direction, run))
    gains = (2.5, 5.0, 10.0, 20.0)
    singles = [c for c in singles if min(abs(c[0] - g) / g for g in gains) < 0.08]
    if not singles:
        return None
    L, a, direction, run = max(singles, key=lambda c: c[0])
    xl, xr = (a[0] - run, a[0]) if direction < 0 else (a[0], a[0] + run)
    return {"height_mm": float(L), "x_left": float(xl + x0), "x_right": float(xr + x0), "top": float(a[1] + y0), "bottom": float(a[2] + y0), "sides": 1}


def fill_gaps(img: np.ndarray, path: LeadPath, x0: int, s: float, win_mm: float = 0.8, max_gray: int = 200) -> int:
    """Columns the path bridged: look in the image around the bridge for the darkest pixel (faded thermal ink)."""
    y = path.y
    idx = np.flatnonzero(np.isfinite(y))
    if idx.size < 2:
        return 0
    filled = 0
    gaps = np.flatnonzero(~np.isfinite(y[idx[0]:idx[-1] + 1])) + idx[0]
    if gaps.size == 0:
        return 0
    yl = np.interp(gaps, idx, y[idx])
    w = int(round(win_mm * s))
    for g, c in zip(gaps, yl):
        col = img[:, x0 + g]
        a, b = int(max(0, c - w)), int(min(len(col), c + w + 1))
        seg = col[a:b].astype(np.int32)
        if seg.size and seg.min() < max_gray:
            # ink nearest to the bridge line
            cand = np.flatnonzero(seg < max_gray)
            j = cand[np.argmin(np.abs(cand + a - c))]
            y[g] = a + j
            filled += 1
    return filled
