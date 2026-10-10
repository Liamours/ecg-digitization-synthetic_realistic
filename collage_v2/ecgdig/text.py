"""Printed text of a panel by RapidOCR: what to erase from the trace mask, the gain, the lead names, lead-off marks."""
from __future__ import annotations

import re

import cv2
import numpy as np

REPORT_RE = re.compile(r"Serial|Number|Last\s*Name|First\s*Name|Date\s*of|Birth|Measurement|Results|Interpretation|QT\s*/\s*QTc|RR\s*/\s*PP|P\s*/\s*QRS|Normal\s*sinus|rhythm|Borderline|criteria|Unconfirmed", re.I)
VOCAB = re.compile(r"MAC|400|V1|U1|\.0?2|^GE|Man|Auto|mm/|mm/m|mU|mV|ADS|For|MDC|INNOQ|Hz|0\.08|150|BPM|\d{6,}|Aug|Jan|Feb|Mar|Apr|May|Jun|Jul|Sep|Oct|Nov|Dec", re.I)
GAIN_RE = re.compile(r"(2[.,]5|5|10|20)\s*[mn]{1,2}\s*/\s*m", re.I)
SPEED_RE = re.compile(r"(12[.,]5|25|50)\s*[mn]{1,2}\s*/\s*[s$5]", re.I)
LEADOFF_RE = re.compile(r"[*x]\s*([UV][1-6]|aV[RLF]|I{1,3}|RA|LA|LL|C[1-6])\s*[*x]", re.I)

LEAD_SETS = {
    "limb": ["I", "II", "III"],
    "aug": ["aVR", "aVL", "aVF"],
    "v13": ["V1", "V2", "V3"],
    "v46": ["V4", "V5", "V6"],
}


def ocr_boxes(ocr, gray_clean: np.ndarray, scale: float = 1.0) -> list[dict]:
    """Det + rec on a 0/255 image (black text on white). Boxes in the input's pixel coordinates."""
    img = gray_clean
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    res, _ = ocr(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    out = []
    for r in res or []:
        pts = np.array(r[0], np.float64) / scale
        out.append({"text": r[1], "conf": float(r[2]), "x0": pts[:, 0].min(), "x1": pts[:, 0].max(), "y0": pts[:, 1].min(), "y1": pts[:, 1].max()})
    return out


def read_gain(boxes: list[dict]) -> float | None:
    for b in boxes:
        t = b["text"].replace("S", "5").replace("s", "5") if re.search(r"mm\s*/\s*m", b["text"], re.I) else b["text"]
        m = GAIN_RE.search(t)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def read_speed(boxes: list[dict]) -> float | None:
    for b in boxes:
        m = SPEED_RE.search(b["text"])
        if m:
            return float(m.group(1).replace(",", "."))
    return None


def read_leadoff(boxes: list[dict]) -> list[str]:
    out = []
    for b in boxes:
        for m in LEADOFF_RE.finditer(b["text"]):
            out.append(norm_lead(m.group(1)))
    return out


def norm_lead(t: str) -> str:
    t = t.strip().replace("U", "V").replace("u", "V").replace("l", "I").replace("|", "I")
    t = re.sub(r"^A", "a", t)
    m = re.fullmatch(r"a[Vv]([RLFrlf])", t)
    if m:
        return "aV" + m.group(1).upper()
    m = re.fullmatch(r"[Vv]([1-6])", t)
    if m:
        return "V" + m.group(1)
    if re.fullmatch(r"I{1,3}", t):
        return t
    return t


def ocr_lead_votes(boxes: list[dict], y_lo: float, y_hi: float) -> dict[str, float]:
    """Lead names the page OCR read inside the trace area (e.g. "aUR", "U5"): a vote for their set."""
    v = {k: 0.0 for k in LEAD_SETS}
    for b in boxes:
        if not (y_lo <= (b["y0"] + b["y1"]) / 2 <= y_hi):
            continue
        for tok in re.split(r"\s+", b["text"]):
            nm = norm_lead(tok)
            for k, names in LEAD_SETS.items():
                if nm in names and (len(nm) >= 2 or b["conf"] > 0.8):
                    v[k] += 1.0
    return v
