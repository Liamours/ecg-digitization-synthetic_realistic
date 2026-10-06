"""Digitize the Fukuda Denshi FX-7542 sheet (PDA (30).jpg): 12 rows, one 10 s lead each, dot grid,
5 mm/mV, calibration pulses at the left. Same grid map and sampling as the MAC 400 panels."""
import json
from pathlib import Path

import cv2
import numpy as np

import grid_map
from scipy.signal import find_peaks

from digitize_panel import BIN_MM, components, extract
from grid_map import fit_grid

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "datasets" / "ecg-mac400-phone_photo" / "PJB 19.12.2024" / "PDA (30).jpg"
OUT = ROOT / "results" / "inferences" / "edan_fukuda_trial" / "fukuda"
TRACE_MAX_GRAY = 60
MIN_EXTENT = 60
X_RANGE = (400, 2860)
Y_RANGE = (170, 1830)
GAIN = 5.0
MASK_IMAGE = OUT / "unet_signal_probability.png"
UNET_THRESHOLD = 60
LABELS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def rows_by_peaks(mask: np.ndarray, rows: int) -> list[tuple[int, int]]:
    """Row bands from the `rows` strongest peaks of the smoothed row profile; cuts fall at the profile minima between neighbours."""
    prof = cv2.GaussianBlur(mask.sum(axis=1).astype(np.float32).reshape(-1, 1), (1, 0), 5).ravel()
    peaks, props = find_peaks(prof, distance=40, prominence=prof.max() * 0.03)
    peaks = np.sort(peaks[np.argsort(-props["prominences"])[:rows]])
    cuts = [int(np.flatnonzero(mask.any(axis=1))[0])]
    for a, b in zip(peaks[:-1], peaks[1:]):
        cuts.append(int(a + np.argmin(prof[a:b])))
    cuts.append(int(np.flatnonzero(mask.any(axis=1))[-1]) + 1)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(PAGE), cv2.IMREAD_GRAYSCALE)
    grid_map.DOT_AREA = (2, 45)
    grid = fit_grid(gray[Y_RANGE[0]:Y_RANGE[1], X_RANGE[0]:X_RANGE[1]])
    mm_px = float(np.hypot(*grid["a1"]))
    print("grid: %.2f px per step, tilt %.2f deg, rms %.2f px, inliers %d of %d" % (mm_px, np.degrees(np.arctan2(grid["a1"][1], grid["a1"][0])), grid["rms_px"], grid["n_inliers"], grid["n_dots"]))
    if MASK_IMAGE.exists():  # Open-ECG-Digitizer U-Net trace probability (openecg_mask.py)
        m = (cv2.imread(str(MASK_IMAGE), 0) > UNET_THRESHOLD).astype(np.uint8)
    else:
        flat = gray.astype(np.int16) - cv2.medianBlur(gray, 51).astype(np.int16)  # local paper level removed: shading is strong on this photo
        m = ((gray < 110) & (flat < -TRACE_MAX_GRAY)).astype(np.uint8)
    m[:Y_RANGE[0]] = 0
    m[Y_RANGE[1]:] = 0
    m[:, :X_RANGE[0]] = 0
    m[:, X_RANGE[1]:] = 0
    n, lab, st = components(m)
    m[np.isin(lab, [i for i in range(1, n) if max(st[i, 2], st[i, 3]) < MIN_EXTENT])] = 0
    bands = rows_by_peaks(m, len(LABELS))
    print('bands', bands)
    A = np.stack([grid["a1"], grid["a2"]], axis=1)
    ys, xs = np.nonzero(m)
    x_all = (np.column_stack([xs, ys]).astype(float) - (grid["o"] + np.array([X_RANGE[0], Y_RANGE[0]]))) @ np.linalg.inv(A).T
    g2 = {**grid, "o": grid["o"] + np.array([X_RANGE[0], Y_RANGE[0]])}
    x_range = (float(x_all[:, 0].min()), float(x_all[:, 0].max()))
    res, leads = {"grid_px_per_step": mm_px, "leads": {}}, {}
    for name, band in zip(LABELS, bands):
        t, mv, ok, pts, base = extract(m, band, g2, GAIN, x_range)
        leads[name] = (t, mv, base)
        res["leads"][name] = {"band": [int(band[0]), int(band[1])], "coverage": round(float(ok.mean()), 2), "duration_s": round(float(t[-1]), 2), "p2p_mV": round(float(np.ptp(np.percentile(mv, [0.5, 99.5]))), 2)}
        np.savetxt(OUT / f"{name}.csv", np.column_stack([t, mv]), delimiter=",", header="time_s,mV", fmt="%.4f", comments="")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for name, (t, mv, base) in leads.items():
        x_mm = x_range[0] + t * 25 + BIN_MM / 2
        y_mm = base - mv * GAIN
        px = g2["o"] + np.outer(x_mm, g2["a1"]) + np.outer(y_mm, g2["a2"])
        cv2.polylines(img, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 2)
        cv2.putText(img, name, (int(px[0, 0]) - 90, int(px[0, 1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    cv2.imwrite(str(OUT / "fukuda_overlay.png"), img)
    cv2.imwrite(str(OUT / "fukuda_overlay_half.png"), cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    for k, v in res["leads"].items():
        print(f"  {k:4}", v)


if __name__ == "__main__":
    main()
