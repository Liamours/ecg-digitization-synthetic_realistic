"""Synthetic MAC 400 collage pages with known signals, to measure the digitizer against ground truth.

A page carries 4 to 6 strips laid out like the real scans (two columns, sometimes turned a quarter, slightly tilted,
sometimes overlapping the neighbour), each strip with the header block, the lead names in the device font (the glyph
templates), the calibration pulse, the footer text and three leads drawn at 25 mm/s with the strip's gain. Signals are
real recordings (MIT-BIH, two channels) combined so that the limb and augmented strips obey Einthoven and Goldberger,
time-compressed and scaled to resemble paediatric ECGs (faster rates, larger precordial voltages).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .glyphs import GH, GW, TEMPLATES

SETS = {"limb": ["I", "II", "III"], "aug": ["aVR", "aVL", "aVF"], "v13": ["V1", "V2", "V3"], "v46": ["V4", "V5", "V6"]}


def load_mitdb(folder: str, records=None) -> list[np.ndarray]:
    import wfdb
    out = []
    recs = records or sorted({p.stem for p in Path(folder).glob("*.hea")})
    for r in recs:
        try:
            sig, f = wfdb.rdsamp(str(Path(folder) / r))
        except Exception:
            continue
        if sig.shape[1] >= 2 and f["fs"] == 360:
            out.append(sig[:, :2].astype(np.float32))
    return out


def make_leads(rng, sources: list[np.ndarray], dur_s: float, fs_out: int = 500) -> dict[str, np.ndarray]:
    """12 leads (mV) for one page: two real channels, a random time compression, combined lead by lead."""
    src = sources[rng.integers(len(sources))]
    comp = rng.uniform(0.55, 1.0)                      # < 1: faster heart rate
    n_src = int(dur_s / comp * 360) + 10
    st = rng.integers(0, len(src) - n_src - 1)
    seg = src[st:st + n_src]
    t_src = np.arange(len(seg)) / 360.0 * comp
    t = np.arange(int(dur_s * fs_out)) / fs_out
    A = np.interp(t, t_src, seg[:, 0]); B = np.interp(t, t_src, seg[:, 1])
    A -= np.median(A); B -= np.median(B)
    a1 = rng.uniform(0.3, 0.9)
    I = a1 * A + rng.uniform(-0.3, 0.3) * B
    II = A
    III = II - I
    aVR = -(I + II) / 2; aVL = I - II / 2; aVF = II - I / 2
    v = {}
    for k, name in enumerate(["V1", "V2", "V3", "V4", "V5", "V6"]):
        w = (k + 1) / 6
        v[name] = (rng.uniform(0.8, 2.2) * ((1 - w) * B * (-1 if k < 2 else 1) + w * A) + rng.uniform(-0.3, 0.3) * A)
    leads = {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF, **v}
    sc = rng.uniform(0.7, 1.8)
    return {k: (x * sc).astype(np.float32) for k, x in leads.items()}


def _glyph_bank():
    d = np.load(TEMPLATES)
    bank = {}
    for n, b, a in zip(d["names"], d["bmps"], d["aspect"]):
        g = b.reshape(GH, GW)
        g = (g - g.min()) / (np.ptp(g) + 1e-9)
        bank[str(n)] = (g, float(a))
    return bank


def draw_label(img, text, x_mm, top_mm, s, bank, ink=40):
    x = x_mm
    for ch in text:
        g, a = bank[ch]
        h = 1.75 if ch == "a" else 2.45
        w = h * a
        gh, gw = max(3, int(round(h * s))), max(2, int(round(w * s)))
        gi = cv2.resize(g.astype(np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR) > 0.5
        y0 = int(round((top_mm + 2.45 - h) * s)); x0 = int(round(x * s))
        sub = img[y0:y0 + gh, x0:x0 + gw]
        sub[gi[:sub.shape[0], :sub.shape[1]]] = ink
        x += w + 0.35


def render_panel(rng, leads: dict, lead_set: str, gain: float, s: float, bank, dur_s: float, leadoff: str | None = None):
    """Upright strip image; the marker's top-left corner is at (OX, OY) mm. Returns image, truth."""
    OX, OY = 16.0, 5.0
    Wmm, Hmm = 16.0 + 82.0, 5.0 + 76.0
    W, H = int(Wmm * s), int(Hmm * s)
    img = np.full((H, W), 255, np.uint8)
    dot = int(rng.uniform(120, 175))
    # grid: dots every mm, dotted lines every 5 mm
    xs = np.arange(0.6, Wmm - 0.5, 1.0); ys = np.arange(0.6, Hmm - 0.5, 1.0)
    X, Y = np.meshgrid((xs * s).astype(int), (ys * s).astype(int))
    img[Y, X] = dot
    if s > 6:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))  # dots of 2 px like the 200 dpi scans
    for xm in xs[::5]:
        yy = (np.arange(0.6, Hmm - 0.5, 0.4) * s).astype(int)
        img[yy, int(xm * s)] = dot
    for ym in ys[::5]:
        xx = (np.arange(0.6, Wmm - 0.5, 0.4) * s).astype(int)
        img[int(ym * s), xx] = dot
    # header
    cv2.rectangle(img, (int(OX * s), int(OY * s)), (int((OX + 5) * s), int((OY + 5.9) * s)), 20, -1)
    fs_txt = 0.09 * s
    cv2.putText(img, "MAC 400", (int((OX + 5.5) * s), int((OY + 5.8) * s)), cv2.FONT_HERSHEY_SIMPLEX, fs_txt, 30, max(1, int(s / 4)), cv2.LINE_AA)
    cv2.putText(img, "V1.02", (int((OX + 31) * s), int((OY + 5.8) * s)), cv2.FONT_HERSHEY_SIMPLEX, fs_txt, 30, max(1, int(s / 4)), cv2.LINE_AA)
    names = SETS[lead_set]
    for slot, nm in enumerate(names):
        draw_label(img, nm, OX + 17.5, OY + (12.5, 32.5, 52.3)[slot], s, bank)
    footer = f"Man   25mm/s   {gain:g}mm/mV   ADS"
    cv2.putText(img, footer, (int((OX + 2) * s), int((OY + 72) * s)), cv2.FONT_HERSHEY_SIMPLEX, fs_txt, 30, max(1, int(s / 4)), cv2.LINE_AA)
    lo_txt = f"*{leadoff}*  " if leadoff else ""
    cv2.putText(img, lo_txt + "For 2030887-001  MDC72942181716008  INNOQ", (int((OX - 8) * s), int((OY + 75) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.06 * s, 40, 1, cv2.LINE_AA)
    # rows
    r1 = rng.uniform(12, 26); r2 = rng.uniform(max(r1 + 12, 33), 46); r3 = rng.uniform(max(r2 + 10, 52), 64)
    rows = [r1, r2, r3]
    # pulse
    pb = r3 + rng.uniform(-1.5, 2.5)
    px0, px1 = OX - 9.0, OX - 4.0
    pts = np.array([[OX - 11, pb], [px0, pb], [px0, pb - gain], [px1, pb - gain], [px1, pb], [OX - 3, pb]]) * s
    pts[:, 1] += OY * s
    cv2.polylines(img, [pts.astype(np.int32)], False, 35, max(1, int(round(s / 4))), cv2.LINE_AA)
    # traces: draw at 4x supersampling for smooth lines
    x_start = OX - 1.5
    thick = rng.uniform(0.14, 0.24)
    truth = {}
    big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    for slot, nm in enumerate(names):
        sig = leads[nm]
        if leadoff == nm:
            sig = np.zeros_like(sig)
        n = len(sig)
        t = np.arange(n) / 500.0
        xm = x_start + t * 25.0
        ym = OY + rows[slot] - sig * gain
        ym = np.clip(ym, 0.5, Hmm - 0.5)   # the paper clips extreme deflections
        P = np.column_stack([xm * s * 4, ym * s * 4]).astype(np.int32)
        ink = int(rng.uniform(25, 70))
        cv2.polylines(big, [P], False, ink, max(2, int(round(thick * s * 4))), cv2.LINE_AA)
        # faded thermal stretches
        if rng.random() < 0.5:
            a = rng.integers(0, n - 200); b = a + rng.integers(100, 400)
            cv2.polylines(big, [P[a:b]], False, int(rng.uniform(120, 160)), max(2, int(round(thick * s * 4))), cv2.LINE_AA)
        truth[nm] = {"t0_mm_from_marker": x_start - OX, "mv": (OY + rows[slot] - ym) / gain}
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return img, {"set": lead_set, "names": names, "gain": gain, "rows_mm": rows, "marker_xy_mm": (OX, OY), "truth": truth, "leadoff": leadoff}


def make_page(rng, sources, s: float = 7.98, page_wh=(1700, 2800), bank=None, dur_s: float = 2.9):
    bank = bank or _glyph_bank()
    leads = make_leads(rng, sources, dur_s)
    page = np.full((page_wh[1], page_wh[0]), 255, np.uint8)
    sets = ["limb", "aug", "v13", "v46"]
    if rng.random() < 0.4:
        sets.append(sets[2 + rng.integers(2)])  # a repeated attempt
    turned = rng.random() < 0.35
    panels = []
    for i, ls in enumerate(sets):
        gain = 10.0 if ls in ("limb", "aug") or rng.random() < 0.4 else 5.0
        lo = None
        if ls == "v46" and rng.random() < 0.1:
            lo = "V4"
        img, info = render_panel(rng, leads, ls, gain, s, bank, dur_s, lo)
        ang = rng.uniform(-1.5, 1.5)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
        if turned:
            img = np.ascontiguousarray(np.rot90(img, -1))  # strip glued a quarter turn clockwise
        h, w = img.shape
        col, row = i % 2, i // 2
        if turned:
            x0 = int(60 + col * (w + rng.uniform(10, 60))); y0 = int(60 + row * (h + rng.uniform(5, 40)))
        else:
            x0 = int(40 + col * (w - rng.uniform(5, 40))); y0 = int(60 + row * (h + rng.uniform(0, 25)))
        if y0 + h > page.shape[0] or x0 + w > page.shape[1]:
            continue
        region = page[y0:y0 + h, x0:x0 + w]
        np.minimum(region, img, out=region) if rng.random() < 0.3 else region.__setitem__(slice(None), img)
        # marker centre on the page
        OX, OY = info["marker_xy_mm"]
        mx, my = (OX + 2.5) * s, (OY + 2.95) * s
        c = np.array([mx, my, 1.0])
        mx, my = M @ c
        if turned:
            hh = img.shape[1]  # height before the turn
            mx, my = hh - 1 - my, mx
        info.update({"marker_center_page": [float(x0 + mx), float(y0 + my)], "turned": bool(turned), "tilt_deg": float(ang)})
        panels.append(info)
    # scan look
    page = cv2.GaussianBlur(page, (0, 0), rng.uniform(0.3, 0.7))
    noise = rng.normal(0, rng.uniform(2, 6), page.shape)
    page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = int(rng.uniform(60, 90))
    page = cv2.imdecode(cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
    return page, panels


def write_set(out_dir: str, mitdb: str, n_pages: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sources = load_mitdb(mitdb)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    bank = _glyph_bank()
    for i in range(n_pages):
        page, panels = make_page(rng, sources, bank=bank)
        cv2.imwrite(str(out / f"synth_{i:03d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for p in panels:
            for k, v in p["truth"].items():
                v["mv"] = np.round(v["mv"], 4).tolist()
        (out / f"synth_{i:03d}.json").write_text(json.dumps(panels))
