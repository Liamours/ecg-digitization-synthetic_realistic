"""Lead names by template matching of the printed glyphs (the MAC 400 font is the same on every strip).

The OCR models read multi-letter words well but miss one- and two-letter labels such as "I", "II" or "V4" next to a
trace. The label sits at a fixed place on the strip, so the glyphs there are cut out and compared with templates of the
13 glyph classes (I a V R L F 1-6) built from this dataset (tools/build_glyph_templates.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LABEL_TOP_MM = (12.5, 32.5, 52.3)   # top of the printed lead name of each row, below the marker's top edge
LABEL_X_MM = (-6.0, 31.0)         # names start ~17.5 mm right of the marker (manual), ~7 mm (auto), ~-4 mm (auto, 2nd strip)
GW, GH = 16, 20                      # normalised glyph bitmap
TEMPLATES = Path(__file__).with_name("glyph_templates.npz")
_T = None


def norm_glyph(b: np.ndarray) -> np.ndarray:
    g = cv2.resize(b.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def _chains(mask: np.ndarray, xa: int, xb: int, ya: int, yb: int, s: float) -> list[list[dict]]:
    z = mask[ya:yb, xa:xb].astype(np.uint8)
    if z.size == 0:
        return []
    n, lab, st, _ = cv2.connectedComponentsWithStats(z, connectivity=8)
    g = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 1.3 * s <= h <= 3.2 * s and 0.6 * s <= w <= 2.7 * s:
            g.append({"x": int(x + xa), "y": int(y + ya), "bottom": int(y + ya + h), "w_mm": w / s, "h_mm": h / s, "bmp": (lab[y:y + h, x:x + w] == i)})
    g.sort(key=lambda d: d["x"])
    chains = []
    for d in g:
        for ch in chains:
            last = ch[-1]
            if abs(d["bottom"] - last["bottom"]) <= 0.6 * s and 0 <= d["x"] - (last["x"] + last["w_mm"] * s) <= 1.3 * s:
                ch.append(d)
                break
        else:
            chains.append([d])
    return chains


def label_glyphs(mask: np.ndarray, mx: float, my: float, s: float, slot: int, all_chains: list | None = None) -> list[dict]:
    """The glyph chain printed as the lead name of a row: bottom-aligned glyphs near the expected label height.

    The name sits about 12.5, 32.5 and 52.3 mm below the marker on most strips but higher on some (the first name of the
    HELLIGE-labelled strips is near 6-10 mm), so chains are searched within +-7 mm and the best one is kept.
    """
    if all_chains is None:
        xa, xb = int(max(0, mx + LABEL_X_MM[0] * s)), int(min(mask.shape[1], mx + LABEL_X_MM[1] * s))
        ya, yb = int(max(0, my + 3 * s)), int(min(mask.shape[0], my + 68 * s))
        all_chains = _chains(mask, xa, xb, ya, yb, s)
    T = my + LABEL_TOP_MM[slot] * s
    best, bs = [], -1e9
    for ch in all_chains:
        top = min(d["y"] for d in ch if d["h_mm"] > 2.0) if any(d["h_mm"] > 2.0 for d in ch) else ch[0]["y"]
        dy = abs(top - T) / s
        if dy > 7.5 or len(ch) > 3:
            continue
        sc = len(ch) - 0.25 * dy - abs(np.median([d["h_mm"] for d in ch]) - 2.45)
        if sc > bs:
            best, bs = ch, sc
    return best


SETS = {
    "limb": ["I", "II", "III"],
    "aug": ["aVR", "aVL", "aVF"],
    "v13": ["V1", "V2", "V3"],
    "v46": ["V4", "V5", "V6"],
}
SKIP_CHAR, SKIP_GLYPH = -0.35, -0.45


def load_templates():
    global _T
    if _T is None:
        d = np.load(TEMPLATES)
        _T = (list(d["names"]), d["bmps"], d["aspect"], d["height_mm"])
    return _T


def char_scores(gl: dict) -> dict[str, float]:
    names, bmps, asp, hgt = load_templates()
    v = norm_glyph(gl["bmp"]).ravel()
    a = gl["w_mm"] / gl["h_mm"]
    sc = bmps @ v - 0.6 * np.abs(np.log(a / asp)) - 1.2 * np.abs(gl["h_mm"] - hgt) / hgt
    return {str(n): float(x) for n, x in zip(names, sc)}


def word_score(chain_scores: list[dict], word: str) -> float:
    """Best in-order alignment of the glyph chain to the word's characters (glyphs or characters may be missing)."""
    m, n = len(chain_scores), len(word)
    D = np.full((m + 1, n + 1), -np.inf)
    D[0, :] = np.arange(n + 1) * SKIP_CHAR
    D[:, 0] = np.arange(m + 1) * SKIP_GLYPH
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = max(D[i - 1, j - 1] + chain_scores[i - 1].get(word[j - 1], -1.0), D[i - 1, j] + SKIP_GLYPH, D[i, j - 1] + SKIP_CHAR)
    return float(D[m, n])


def decode_panel(glyph_rows: list[list[dict]], extra: dict | None = None) -> tuple[str | None, dict]:
    """The lead set whose three names best explain the glyphs found in the three label zones (plus OCR votes)."""
    extra = extra or {}
    if not TEMPLATES.exists() or (not any(glyph_rows) and not any(extra.values())):
        return None, {}
    rows = [[char_scores(g) for g in chain] for chain in glyph_rows]
    totals = {}
    for k, words in SETS.items():
        totals[k] = sum(word_score(r, w) if r else 0.0 for r, w in zip(rows, words)) + extra.get(k, 0.0)
    order = sorted(totals, key=totals.get, reverse=True)
    best, second = order[0], order[1]
    margin = totals[best] - totals[second]
    info = {"scores": {k: round(v, 2) for k, v in totals.items()}, "margin": round(margin, 2), "ocr_votes": extra,
            "glyphs_per_row": [len(c) for c in glyph_rows]}
    if not ((totals[best] > 0.3 and margin >= 0.15) or margin >= 1.0):
        return None, info
    return best, info
