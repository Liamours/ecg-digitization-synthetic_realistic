"""Digitize the EDAN SE-1201 sheet (upright): two 5 s columns of six leads and a lead II rhythm strip,
with the solid-line grid map from line_grid.py. Same sampling and overlay steps as the MAC 400 panels."""
import json
import re
from pathlib import Path

import cv2
import numpy as np

from digitize_panel import BIN_MM, components, extract
from fukuda_page import rows_by_peaks
from line_grid import fit_line_grid

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "datasets" / "panel_templates" / "edan_raw_upright.png"
OUT = ROOT / "results" / "inferences" / "edan_fukuda_trial" / "edan"
TRACE_MAX_GRAY = 14  # the grid lines are almost as dark as the trace on this page; only the trace reaches this level
MIN_EXTENT = 100  # px; grid dashes are shorter than this, the trace is connected over much more
ZONE_Y = (372, 1640)
LEFT_X = (392, 1318)
RIGHT_X = (1345, 2310)
RHYTHM_Y = (1490, 1640)
GAIN = 10.0
LEFT_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF"]
RIGHT_LABELS = ["V1", "V2", "V3", "V4", "V5", "V6"]


MASK_IMAGE = ROOT / "results" / "inferences" / "edan_fukuda_trial" / "edan" / "unet_signal_probability.png"  # Open-ECG-Digitizer U-Net output, see openecg_mask.py
UNET_THRESHOLD = 60


def clean_mask(gray: np.ndarray) -> np.ndarray:
    if MASK_IMAGE.exists():
        return (cv2.imread(str(MASK_IMAGE), 0) > UNET_THRESHOLD).astype(np.uint8)
    m = (gray < TRACE_MAX_GRAY).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st = components(m)
    m[np.isin(lab, [i for i in range(1, n) if max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) < MIN_EXTENT])] = 0
    return m


def pulses(gray: np.ndarray, grid: dict) -> list[dict]:
    m = (gray < 60).astype(np.uint8)
    m[:, 480:] = 0
    n, lab, st = components(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        hm, wm = h / grid["px_per_mm_y"], w / grid["px_per_mm_x"]
        if 8.5 <= hm <= 11.5 and 3.5 <= wm <= 9 and ZONE_Y[0] < y:
            out.append({"bbox": [int(x), int(y), int(w), int(h)], "height_mm": float(hm), "width_mm": float(wm)})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(PAGE), cv2.IMREAD_GRAYSCALE)
    grid = fit_line_grid(gray)
    pl = pulses(gray, grid)
    mask = clean_mask(gray)
    mask[:ZONE_Y[0]] = 0
    mask[ZONE_Y[1]:] = 0
    res = {"grid": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in grid.items()}, "pulses_mm": [round(p["height_mm"], 2) for p in pl], "leads": {}}
    leads = {}
    for labels, (x0, x1) in ((LEFT_LABELS, LEFT_X), (RIGHT_LABELS, RIGHT_X)):
        col = mask.copy()
        col[:, :x0] = 0
        col[:, x1:] = 0
        col[RHYTHM_Y[0]:] = 0
        bands = rows_by_peaks(col, len(labels))
        ys, xs = np.nonzero(col)
        x_mm = (np.column_stack([xs, ys]) @ np.linalg.inv(np.stack([grid["a1"], grid["a2"]], axis=1)).T)[:, 0]
        x_range = (float(x_mm.min()), float(x_mm.max()))
        for name, band in zip(labels, bands):
            t, mv, ok, pts, base = extract(col, band, {**grid, "o": np.zeros(2)}, GAIN, x_range)
            leads[name] = {"t": t, "mv": mv, "ok": ok, "pts": pts, "band": band, "base": base, "x0": x_range[0]}
    # rhythm strip: full width, lead II
    rh = mask.copy()
    rh[:RHYTHM_Y[0]] = 0
    rh[:, :LEFT_X[0]] = 0
    rh[:, RIGHT_X[1]:] = 0
    ys, xs = np.nonzero(rh)
    A = np.stack([grid["a1"], grid["a2"]], axis=1)
    x_mm = (np.column_stack([xs, ys]) @ np.linalg.inv(A).T)[:, 0]
    t, mv, ok, pts, base = extract(rh, (RHYTHM_Y[0], RHYTHM_Y[1]), {**grid, "o": np.zeros(2)}, GAIN, (float(x_mm.min()), float(x_mm.max())))
    leads["II_rhythm"] = {"t": t, "mv": mv, "ok": ok, "pts": pts, "band": (RHYTHM_Y[0], RHYTHM_Y[1]), "base": base, "x0": float(x_mm.min())}
    for k, v in leads.items():
        res["leads"][k] = {"coverage": round(float(v["ok"].mean()), 2), "duration_s": round(float(v["t"][-1]), 2), "p2p_mV": round(float(np.ptp(np.percentile(v["mv"], [0.5, 99.5]))), 2)}
        np.savetxt(OUT / f"{k}.csv", np.column_stack([v["t"], v["mv"]]), delimiter=",", header="time_s,mV", fmt="%.4f", comments="")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    Ainv = A
    for k, v in leads.items():
        x_mm = v["x0"] + v["t"] * 25 + BIN_MM / 2
        y_mm = v["base"] - v["mv"] * GAIN
        px = np.outer(x_mm, grid["a1"]) + np.outer(y_mm, grid["a2"])
        cv2.polylines(img, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 2)
        cv2.putText(img, k, (int(px[0, 0]) - 90, int(px[0, 1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    for p in pl:
        x, y, w, h = p["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 0), 2)
    cv2.imwrite(str(OUT / "edan_overlay_upright.png"), img)
    cv2.imwrite(str(OUT / "edan_overlay_original_orientation.png"), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    print("grid: px/mm x %.3f y %.3f, tilt h %.2f v %.2f | pulses (mm) %s" % (grid["px_per_mm_x"], grid["px_per_mm_y"], grid["tilt_h_deg"], grid["tilt_v_deg"], res["pulses_mm"]))
    for k, v in res["leads"].items():
        print(f"  {k:10}", v)


if __name__ == "__main__":
    main()
