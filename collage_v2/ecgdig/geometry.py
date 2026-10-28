"""Page geometry: background normalisation, MAC 400 header markers, panel orientation, deskew and grid scale.

Every MAC 400 strip prints a solid black square at the left end of its header ("[] MAC 400  V1.02"). That square is
the anchor of a panel: its position fixes where the three trace rows, the calibration pulse and the footer are, and the
header text next to it tells which way is up. Nothing here depends on how the strips were glued on the page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

HEADER_RE = re.compile(r"MAC|AC\s*4|400|V1|U1|\.02|^GE$|GE\b", re.I)


def load_gray(path: str) -> np.ndarray:
    data = np.fromfile(path, np.uint8)  # works with non-ASCII Windows paths
    im = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f"cannot read {path}")
    return im


def normalise(gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Divide by the local paper brightness so that shadows and grey phone photos look like a flat white scan."""
    k = int(round(px_per_mm * 6)) | 1
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((max(3, k // 4), max(3, k // 4)), np.uint8))
    bg = cv2.medianBlur(bg, 5)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 60) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def page_px_per_mm(gray: np.ndarray) -> float:
    """A first guess from the page size: the scans are A4/F4 pages about 213 mm wide."""
    return min(gray.shape) / 213.0


def grid_period(gray: np.ndarray, lo: float, hi: float) -> float | None:
    """Period in px of the 1 mm dot lattice, from the spectrum of the row and column projections of dark pixels."""
    g = (255.0 - gray.astype(np.float32))
    best = []
    for ax in (0, 1):
        p = g.mean(axis=ax).astype(np.float64)
        if p.size < 8 * hi:
            continue
        w = int(hi * 3) | 1
        p = p - np.convolve(p, np.ones(w) / w, mode="same")
        p = p[w:-w]
        n = 1 << int(np.ceil(np.log2(len(p) * 16)))
        F = np.abs(np.fft.rfft(p * np.hanning(len(p)), n=n))
        fr = np.fft.rfftfreq(n)
        sel = (fr > 1.0 / hi) & (fr < 1.0 / lo)
        if sel.sum() < 3:
            continue
        k = int(np.argmax(np.where(sel, F, 0)))
        if 0 < k < len(F) - 1:  # parabolic peak interpolation
            a, b, c = F[k - 1], F[k], F[k + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        else:
            d = 0.0
        f0 = (k + d) / n
        snr = F[k] / (np.median(F[sel]) + 1e-9)
        best.append((snr, 1.0 / f0))
    if not best:
        return None
    snr, per = max(best)
    return float(per) if snr > 3 else None


@dataclass
class Marker:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def find_markers(gray: np.ndarray, s: float) -> list[Marker]:
    """Solid dark squares about 5 x 6 mm (the MAC 400 header block)."""
    b = (gray < 90).astype(np.uint8)
    k = max(3, int(round(s * 1.2)))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        wm, hm = w / s, h / s
        if not (3.0 <= wm <= 8.5 and 3.0 <= hm <= 8.5):
            continue
        if a / float(w * h) < 0.80 or max(w, h) / min(w, h) > 1.7:
            continue
        out.append(Marker(int(x), int(y), int(w), int(h), int(a)))
    return out


def rot_point(x: float, y: float, k: int, W: int, H: int) -> tuple[float, float]:
    """Where (x, y) of a W x H image lands after np.rot90(img, k) (counter-clockwise k quarter turns)."""
    k %= 4
    if k == 0:
        return x, y
    if k == 1:
        return y, W - 1 - x
    if k == 2:
        return W - 1 - x, H - 1 - y
    return H - 1 - y, x


def rot_box(m: Marker, k: int, W: int, H: int) -> Marker:
    pts = [rot_point(m.x, m.y, k, W, H), rot_point(m.x + m.w - 1, m.y + m.h - 1, k, W, H)]
    xs, ys = sorted([pts[0][0], pts[1][0]]), sorted([pts[0][1], pts[1][1]])
    return Marker(int(xs[0]), int(ys[0]), int(xs[1] - xs[0] + 1), int(ys[1] - ys[0] + 1), m.area)


def clean_binary(gray: np.ndarray, s: float, thr: int = 130) -> np.ndarray:
    """Dark ink without the grid dots: a 0/255 image for the OCR (text black on white)."""
    b = (gray < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = (st[1:, 4] >= max(10, 0.4 * s * s)) | (np.maximum(st[1:, 2], st[1:, 3]) >= 1.5 * s)
    return np.where(keep[lab], 0, 255).astype(np.uint8)


@dataclass
class Orientation:
    k: int                 # quarter turns (np.rot90) that make the panel upright
    score: float
    text: str
    source: str            # "ocr" or "geometry"


def header_text_score(ocr, rot_gray: np.ndarray, m: Marker, s: float) -> tuple[float, str]:
    """OCR the band right of the marker in an upright candidate; MAC 400 / V1.02 / GE there means this way is up."""
    H, W = rot_gray.shape
    x0, x1 = int(m.x + m.w - 1.0 * s), int(m.x + m.w + 45 * s)
    y0, y1 = int(m.y - 2.0 * s), int(m.y + m.h + 3.0 * s)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 5 * s or y1 - y0 < 3 * s:
        return 0.0, ""
    crop = clean_binary(rot_gray[y0:y1, x0:x1], s)
    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    res, _ = ocr(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    score, txt = 0.0, []
    for r in res or []:
        t, c = r[1], float(r[2])
        txt.append(t)
        if HEADER_RE.search(t):
            score += c * (2.0 if re.search(r"MAC|400", t, re.I) else 1.0)
    return score, " ".join(txt)


def geometry_orientation(ink: np.ndarray, m: Marker, s: float) -> int:
    """Fallback without text: the panel extends about 75 mm right of and below the marker when upright."""
    H, W = ink.shape
    best, bk = -1.0, 0
    for k in range(4):
        r = np.rot90(ink, k)
        mm = rot_box(m, k, W, H)
        y0, y1 = int(mm.y + 10 * s), int(mm.y + 65 * s)
        x0, x1 = int(mm.x + mm.w), int(mm.x + 70 * s)
        sub = r[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        frac = float(sub.mean()) if sub.size else 0.0
        if frac > best:
            best, bk = frac, k
    return bk


def deskew_angle(gray: np.ndarray, s: float, max_deg: float = 4.0) -> float:
    """Small rotation (degrees, cv2 convention) that makes the grid dot rows horizontal: projection sharpness search."""
    small_s = 4.0
    f = small_s / s if s > small_s else 1.0
    g = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    dots = ((g < 225) & (g > 60)).astype(np.float32)
    h, w = dots.shape
    c = (w / 2, h / 2)
    best = (-1.0, 0.0)
    for step, rng in ((0.5, max_deg), (0.1, 0.6)):
        center = best[1]
        for a in np.arange(center - rng, center + rng + 1e-6, step):
            M = cv2.getRotationMatrix2D(c, a, 1.0)
            r = cv2.warpAffine(dots, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            v = float(np.var(r.sum(axis=1))) + float(np.var(r.sum(axis=0)))
            if v > best[0]:
                best = (v, float(a))
    return best[1]


@dataclass
class PanelCrop:
    image: np.ndarray            # upright, deskewed, normalised grey crop
    marker: Marker               # marker box in crop coordinates
    s: float                     # px per mm in the crop
    k: int                       # quarter turns applied to the page
    angle: float                 # extra small rotation in degrees
    page_marker: Marker          # marker box on the original page
    origin: tuple[int, int]      # crop's top-left in the quarter-turned page
    orient_source: str
    header_text: str
    M_page_to_crop: np.ndarray = field(repr=False, default=None)


# Panel layout relative to the marker's top-left corner, in mm, measured on upright MAC 400 strips (25 mm/s).
CROP_MM = dict(left=-18.0, right=88.0, top=-6.0, bottom=84.0)


def crop_panel(page: np.ndarray, m: Marker, k: int, s: float) -> PanelCrop:
    H, W = page.shape
    rot = np.ascontiguousarray(np.rot90(page, k))
    mm = rot_box(m, k, W, H)
    x0 = int(round(mm.x + CROP_MM["left"] * s)); x1 = int(round(mm.x + CROP_MM["right"] * s))
    y0 = int(round(mm.y + CROP_MM["top"] * s)); y1 = int(round(mm.y + CROP_MM["bottom"] * s))
    pad = int(95 * s)
    big = cv2.copyMakeBorder(rot, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    crop = big[y0 + pad:y1 + pad, x0 + pad:x1 + pad].copy()
    lm = Marker(mm.x - x0, mm.y - y0, mm.w, mm.h, mm.area)
    # deskew on the trace area (avoids the neighbour strips at the sides as far as possible)
    cy0, cy1 = int(lm.y + 8 * s), int(lm.y + 70 * s)
    cx0, cx1 = int(max(0, lm.x - 5 * s)), int(lm.x + 70 * s)
    ang = deskew_angle(crop[cy0:cy1, cx0:cx1], s)
    if abs(ang) > 0.05:
        Mr = cv2.getRotationMatrix2D((lm.cx, lm.cy), ang, 1.0)
        crop = cv2.warpAffine(crop, Mr, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
    else:
        ang = 0.0
    return PanelCrop(crop, lm, s, k, ang, m, (x0, y0), "", "")


def refine_scale(crop: np.ndarray, m: Marker, s0: float) -> float:
    """Measure px per mm on the crop's trace area; fall back to the page guess."""
    s = s0
    y0, y1 = int(m.y + 8 * s), int(m.y + 70 * s)
    x0, x1 = int(max(0, m.x - 5 * s)), int(m.x + 70 * s)
    sub = crop[y0:y1, x0:x1]
    per = grid_period(sub, 0.7 * s0, 1.4 * s0)
    if per is None:
        return s0
    # the projection can lock onto the 5 mm lines' harmonics; keep the 1 mm candidate closest to the guess
    return float(per)
