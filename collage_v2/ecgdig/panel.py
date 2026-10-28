"""One upright MAC 400 panel to three leads in millivolts."""
from __future__ import annotations

from dataclasses import dataclass, field

import re

import cv2
import numpy as np

from .geometry import Marker, PanelCrop
from .glyphs import LABEL_X_MM, _chains, decode_panel, label_glyphs
from .text import LEAD_SETS, REPORT_RE, VOCAB, ocr_boxes, ocr_lead_votes, read_gain, read_leadoff, read_speed
from .trace import fill_gaps, find_pulse, track_leads

ROW_GUESS_MM = (22.5, 43.0, 59.0)   # trace rows below the marker's top edge, measured on upright strips
GAINS = (2.5, 5.0, 10.0, 20.0)


@dataclass
class PanelLead:
    name: str
    t: np.ndarray            # s, from the panel's first trace column
    mv: np.ndarray           # mV, median of the lead removed
    x_px: np.ndarray         # crop columns
    y_px: np.ndarray         # crop rows (NaN where bridged)
    own: np.ndarray
    shared: np.ndarray
    base_px: float
    lo: np.ndarray = None
    hi: np.ndarray = None
    metrics: dict = field(default_factory=dict)


@dataclass
class PanelResult:
    lead_set: str | None
    names: list[str]
    leads: list[PanelLead]
    gain: float
    gain_source: str
    speed: float
    s: float
    rows: list[float]
    x_range: tuple[int, int]
    pulse: dict | None
    texts: dict
    leadoff: list[str]
    error: str = ""
    info: dict = field(default_factory=dict)


def panel_mask(img: np.ndarray, s: float, thr: int = 180) -> np.ndarray:
    """Ink at a high threshold (thin thermal print is light grey) without the isolated grid dots."""
    b = (img < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    area, w, h = st[:, 4], st[:, 2], st[:, 3]
    keep = (area >= max(10, 0.45 * s * s)) & (np.maximum(w, h) >= 0.9 * s)
    keep[0] = False
    m = keep[lab]
    return m


def find_rows(mask: np.ndarray, x0: int, x1: int, my: float, s: float) -> list[float]:
    """Row baselines: the three strongest, well separated peaks of the horizontal ink projection.

    Flat stretches of a trace pile up at its baseline, so each lead makes a peak wherever the device put it (the
    baselines are not at fixed heights: V1 sometimes runs right under the header). Template heights only fill in when
    fewer than three peaks are found.
    """
    prof = mask[:, x0:x1].sum(axis=1).astype(np.float64)
    k = max(3, int(s * 1.2)) | 1
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    lo, hi = int(max(0, my + 4.0 * s)), int(min(len(prof), my + 74.0 * s))
    seg = prof[lo:hi]
    if seg.size == 0 or seg.max() <= 0:
        return [my + g * s for g in ROW_GUESS_MM]
    # local maxima
    pk = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1] and seg[i] > 0.12 * seg.max()]
    pk = sorted(pk, key=lambda i: -seg[i])
    sep = 7.0 * s
    chosen = []
    for i in pk:
        if all(abs(i - j) >= sep for j in chosen):
            chosen.append(i)
        if len(chosen) == 3:
            break
    rows = sorted(lo + i for i in chosen)
    if len(rows) < 3:
        guess = [my + g * s for g in ROW_GUESS_MM]
        for g in guess:
            if len(rows) < 3 and all(abs(g - r) >= sep for r in rows):
                rows.append(g)
        rows = sorted(rows)[:3]
    return [float(r) for r in rows]


def ink_extent(mask: np.ndarray, rows: list[float], x_lo: int, x_hi: int, s: float) -> tuple[int, int]:
    """First and last column where at least two of the three rows have ink near their baseline."""
    cnt = np.zeros(max(0, x_hi - x_lo))
    for r in rows:
        ya, yb = int(max(0, r - 7 * s)), int(min(mask.shape[0], r + 7 * s))
        cnt += mask[ya:yb, x_lo:x_hi].any(axis=0)
    k = max(3, int(0.6 * s))
    for need in (2, 1):
        dens = np.convolve((cnt >= need).astype(float), np.ones(k) / k, mode="same")
        idx = np.flatnonzero(dens > 0.6)
        if idx.size:
            return int(x_lo + idx[0]), int(x_lo + idx[-1] + 1)
    return x_lo, x_hi


def decide_gain(pulse: dict | None, ocr_gain: float | None) -> tuple[float, str]:
    if pulse is not None:
        h = pulse["height_mm"]
        g = min(GAINS, key=lambda v: abs(v - h))
        if abs(g - h) / g < 0.15:
            if pulse.get("sides") == 1 and ocr_gain is not None and ocr_gain != g:
                return ocr_gain, "text (one-sided pulse says %g)" % g
            if ocr_gain is None or ocr_gain == g:
                return g, "pulse" if ocr_gain is None else "pulse+text"
            return g, "pulse (text says %g)" % ocr_gain
    if ocr_gain is not None:
        return ocr_gain, "text"
    return 10.0, "assumed"


def digitize_panel(ocr, pc: PanelCrop, s: float, other_markers: list[Marker], fs: float = 500.0) -> PanelResult:
    img, mk = pc.image, pc.marker
    mask = panel_mask(img, s)
    clean = np.where(mask, 0, 255).astype(np.uint8)

    # printed text: read it, then remove it from the mask
    boxes = ocr_boxes(ocr, clean)
    texts = {"all": [b["text"] for b in boxes]}
    n_report = sum(bool(REPORT_RE.search(b["text"])) for b in boxes)
    if n_report >= 2:
        return PanelResult(None, [], [], 10.0, "", 25.0, s, [], (0, 0), None, texts, [], error="report strip (measurements/interpretation text, no traces)")
    pad = 0.4 * s
    for b in boxes:
        if VOCAB.search(b["text"]) or b["y0"] < mk.y + 7 * s or b["y1"] > mk.y + 70 * s:
            # only erase text that is clearly printed text (vocabulary) or sits in the header/footer bands
            if VOCAB.search(b["text"]) or len(b["text"]) >= 3:
                x0, x1 = int(b["x0"] - pad), int(b["x1"] + pad)
                y0, y1 = int(b["y0"] - pad), int(b["y1"] + pad)
                mask[max(0, y0):y1, max(0, x0):x1] = False
    # the marker, and glyph-sized pieces left in the header and footer bands (a trace passing there is long, so it stays)
    mask[int(mk.y - 1 * s): int(mk.y + mk.h + 1 * s), int(mk.x - 1 * s): int(mk.x + mk.w + 1 * s)] = False
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = (st[:, 3] <= 4.0 * s) & (st[:, 2] <= 7.0 * s)
    top_band = st[:, 1] + st[:, 3] <= mk.y + mk.h + 1.5 * s
    foot_band = st[:, 1] >= mk.y + 70.0 * s
    # paper edges and cut lines near the top and bottom: long, very thin, straight
    edge = (st[:, 3] <= 0.8 * s) & (st[:, 2] >= 8 * s) & ((st[:, 1] >= mk.y + 64 * s) | (st[:, 1] + st[:, 3] <= mk.y + 3 * s))
    drop = (small & (top_band | foot_band)) | edge
    drop[0] = False
    mask[drop[lab]] = False
    mask[int(mk.y + 77.0 * s):, :] = False

    # cut edges of a neighbouring strip glued over this one: long, thin, perfectly straight, slightly sloped lines
    # (a lead-off flat line is straight too, but horizontal on a deskewed strip, so it stays)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in np.flatnonzero(st[:, 2] >= 20 * s):
        if i == 0:
            continue
        x, y, w, h, _a = st[i]
        sub = lab[y:y + h, x:x + w] == i
        cols = np.flatnonzero(sub.any(axis=0))
        top = np.argmax(sub[:, cols], axis=0)
        bot = h - 1 - np.argmax(sub[::-1, cols], axis=0)
        if np.mean((bot - top + 1) <= 0.7 * s) < 0.95:
            continue
        c = (top + bot) / 2.0
        k, b0 = np.polyfit(cols, c, 1)
        if np.std(c - (k * cols + b0)) <= 0.3 * s and abs(np.degrees(np.arctan(k))) >= 0.3:
            mask[lab == i] = False
    gain_txt = read_gain(boxes)
    speed = read_speed(boxes) or 25.0
    leadoff = read_leadoff(boxes)

    pulse = find_pulse(mask, mk.x, mk.y, s, img)
    gain, gain_src = decide_gain(pulse, gain_txt)

    # a "GE <date>" header marks the second strip of an auto-mode print: no pulse, the traces run on from the strip's
    # left edge (names near the edge too)
    ht = pc.header_text or ""
    auto_cont = bool(re.search(r"\bGE\b|\d{1,2}\s*\.\s*[A-Z][a-z]{2}", ht)) and not re.search(r"MAC|400", ht)
    x_lo = max(0, int(mk.x - (14.5 if auto_cont else 2.5) * s))
    if pulse is not None and pulse["x_right"] > mk.x - 6 * s:
        x_lo = max(x_lo, int(pulse["x_right"] + 0.6 * s))
    x_hi = int(min(img.shape[1] - 1, mk.x + 74.5 * s))
    for om in other_markers:  # another strip glued over this one's right end
        if om.x > mk.x + 10 * s and -12 * s < om.y - mk.y < 70 * s:
            x_hi = min(x_hi, int(om.x - 12.5 * s))
    mask[:, :max(0, x_lo)] = False
    mask[:, x_hi:] = False

    rows = find_rows(mask, x_lo, x_hi, mk.y, s)
    x0, x1 = ink_extent(mask, rows, x_lo, x_hi, s)
    if x1 - x0 < 10 * s:
        return PanelResult(None, [], [], gain, gain_src, speed, s, rows, (x0, x1), pulse, texts, leadoff, error="no trace")

    paths = track_leads(mask, x0, x1, rows, s)
    for pth in paths:
        fill_gaps(img, pth, x0, s)

    # lead names: glyphs left in the label zones once the tracked trace pixels are taken out
    resid = mask.copy()
    for pth in paths:
        for i in np.flatnonzero(pth.own):
            x = x0 + i
            resid[int(pth.lo[i]) - 1:int(pth.hi[i]) + 2, max(0, x - 1):x + 2] = False
    ch = _chains(resid, int(max(0, mk.x + LABEL_X_MM[0] * s)), int(min(resid.shape[1], mk.x + LABEL_X_MM[1] * s)),
                 int(max(0, mk.y + 3 * s)), int(min(resid.shape[0], mk.y + 68 * s)), s)
    glyphs = [label_glyphs(resid, mk.x, mk.y, s, i, ch) for i in range(3)]
    # a chain may serve one slot only
    for i in range(3):
        for j in range(i + 1, 3):
            if glyphs[i] and glyphs[i] is glyphs[j]:
                glyphs[j] = []
    lead_set, votes = decode_panel(glyphs, ocr_lead_votes(boxes, mk.y + 7 * s, mk.y + 68 * s))
    texts["labels"] = votes
    for lo in leadoff:  # a lead-off mark in the footer (e.g. *V4*) names a lead of this panel
        for k, nm in LEAD_SETS.items():
            if lo in nm and lead_set is None:
                lead_set = k
    if lead_set in LEAD_SETS:
        names = LEAD_SETS[lead_set]
    elif lead_set == "v??":
        names = ["V?1", "V?2", "V?3"]
    else:
        names = ["row1", "row2", "row3"]

    xs = np.arange(x0, x1)
    leads = []
    for name, p, r in zip(names, paths, rows):
        own = p.own
        if own.sum() < 5:
            y = np.full(len(xs), np.nan)
        else:
            y = p.y.copy()
            good = np.isfinite(y)
            y = np.interp(np.arange(len(xs)), np.flatnonzero(good), y[good])
        base = float(np.nanmedian(p.y[own])) if own.any() else float(r)
        mv = (base - y) / (s * gain)
        t = (xs - x0) / (s * speed)
        leads.append(PanelLead(name, t, mv, xs.astype(float), p.y, own, p.shared, base, p.lo, p.hi))
    res = PanelResult(lead_set, names, leads, gain, gain_src, speed, s, rows, (x0, x1), pulse, texts, leadoff)
    res.info["votes"] = votes
    res.info["auto_continuation"] = auto_cont
    res.info["mask"] = mask
    res.info["glyphs"] = glyphs
    return res
