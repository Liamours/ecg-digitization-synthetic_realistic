"""Draw one panel-level digitization back onto the original, unfixed page.
Nothing was resampled, so the reversal is exact: a digitized sample (t, mV) becomes
grid millimetres (x = x0 + 25 t, y = baseline - gain * mV), grid millimetres become
crop pixels (o + a1 x + a2 y), and crop pixels become page pixels (+ crop origin).
Also writes every transform used to transforms.json.

Usage: python reverse_to_page.py <page image> <out dir>   (inputs read from the trial's inputs folder)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from digitize_panel import PANEL_LABELS, digitize_panels

BIN_MM = 0.1
MM_PER_S = 25.0
COLORS = {0: (0, 0, 220), 1: (0, 140, 255), 2: (0, 170, 0), 3: (200, 0, 160), 4: (150, 150, 150), 5: (150, 150, 150)}  # BGR


def to_page(grid: dict, origin: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    return grid["o"] + np.outer(x_mm, grid["a1"]) + np.outer(y_mm, grid["a2"]) + origin


def main(page_path: Path, out_dir: Path) -> None:
    inputs = Path(__file__).resolve().parents[3] / "results" / "inferences" / "manual_digitization_trial" / "inputs"
    meta = json.load(open(inputs / "panels_meta.json"))
    ocr = json.load(open(inputs / "ocr.json"))
    page = cv2.imread(str(page_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    transforms = {}
    results = digitize_panels(inputs)
    for m in meta:
        i = m["panel"]
        color = COLORS[i]
        origin = np.array(m["crop_origin"], dtype=np.float64)
        x0, y0, x1, y1 = m["box"]
        cv2.rectangle(page, (x0, y0), (x1, y1), color, 4)
        cv2.putText(page, f"panel {i}", (x0 + 8, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        for b in ocr[str(i)]:
            bx0, by0, bx1, by1 = (np.array(b["box"]) + np.tile(origin, 2)).astype(int)
            cv2.rectangle(page, (bx0, by0), (bx1, by1), (200, 120, 0), 2)
            cv2.putText(page, b["text"][:22], (bx0, max(by0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 120, 0), 2)
        if i not in PANEL_LABELS:
            continue
        res = results[i]
        g, gain = res["grid"], res["gain"]
        transforms[i] = {"crop_origin_page_px": origin.tolist(), "grid_origin_crop_px": g["o"].tolist(), "grid_a1_px_per_mm": g["a1"].tolist(), "grid_a2_px_per_mm": g["a2"].tolist(),
                         "gain_mm_per_mV": gain, "mm_per_s": MM_PER_S, "leads": {k: {"x0_mm": v["x0_mm"], "baseline_mm": v["base_mm"]} for k, v in res["leads"].items()}}
        for name, v in res["leads"].items():
            x_mm = v["x0_mm"] + v["t"] * MM_PER_S + BIN_MM / 2
            y_mm = v["base_mm"] - v["mv"] * gain
            pts = to_page(g, origin, x_mm, y_mm)
            cv2.polylines(page, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 2)
            lx, ly = pts[0]
            cv2.putText(page, name, (int(lx) - 70, int(ly) + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
    cv2.imwrite(str(out_dir / "page_with_inferences.png"), page)
    json.dump(transforms, open(out_dir / "transforms.json", "w"), indent=1)
    small = cv2.resize(page, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "page_with_inferences_half.png"), small)
    print("wrote", out_dir / "page_with_inferences.png", page.shape)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
