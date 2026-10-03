"""How much a photographed panel is warped, measured on its own printed dot lattice (1 mm steps).

The lattice is fitted with three models and the differences are the measurement: an affine map (what the digitizer uses),
a homography (a flat sheet seen at an angle: the pitch changes across the panel) and a quadratic map (a sheet that is also
curled). Reported per panel: tilt, the pitch along and across, how much the pitch changes from one side to the other, the
turn of the lattice lines from top to bottom, and the residual of each model in millimetres.

Usage:
    uv run python -m src.digitize.warp --config configs/digitize.yml --run @inferences/<run> [--layout mac400]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.digitize.gridmap import find_dots, fit_dot_grid
from src.digitize.pipeline import load_image, load_layout

GROW_ITERS = 6
FIELDS = ["page", "panel", "n_dots", "n_inliers", "extent_w_mm", "extent_h_mm", "px_per_mm_x", "px_per_mm_y", "aniso", "tilt_deg",
          "rms_affine_mm", "rms_homography_mm", "rms_quadratic_mm", "persp_x_pct", "persp_y_pct", "keystone_deg"]


def _jacobian(h: np.ndarray, i: float, j: float) -> np.ndarray:
    f = lambda a, b: cv2.perspectiveTransform(np.array([[[a, b]]], np.float64), h)[0, 0]
    return np.column_stack([f(i + 0.5, j) - f(i - 0.5, j), f(i, j + 0.5) - f(i, j - 0.5)])  # pixels per mm along i and j


def _quadratic(ij: np.ndarray, xy: np.ndarray) -> np.ndarray:
    i, j = ij[:, 0], ij[:, 1]
    basis = np.column_stack([np.ones_like(i), i, j, i * i, i * j, j * j])
    sol, *_ = np.linalg.lstsq(basis, xy, rcond=None)
    return basis @ sol


def measure_dots(dots: np.ndarray, grid_cfg: dict, gray: np.ndarray) -> dict:
    """Warp numbers from the dot centroids of one panel; `gray` only feeds the affine start (angle and pitch search)."""
    grid = fit_dot_grid(gray, grid_cfg)
    idx = np.linalg.solve(np.stack([grid.a1, grid.a2], axis=1), (dots - grid.origin).T).T
    ij = np.round(idx)
    pitch = float(np.mean(grid.px_per_mm))
    keep = np.hypot(*(idx - ij).T) < 0.22
    h = None
    for it in range(GROW_ITERS):  # the homography extends the lattice past where the affine map still matches
        if keep.sum() < 20:
            break
        h, _ = cv2.findHomography(ij[keep].astype(np.float64), dots[keep].astype(np.float64), cv2.RANSAC, 0.2 * pitch)
        if h is None:
            break
        back = cv2.perspectiveTransform(dots[None].astype(np.float64), np.linalg.inv(h))[0]
        ij = np.round(back)
        pred = cv2.perspectiveTransform(ij[None], h)[0]
        keep = np.hypot(*(pred - dots).T) < (0.3 - 0.03 * it) * pitch
    if h is None or keep.sum() < 20:
        raise ValueError("lattice not recovered")
    lat, pix = ij[keep], dots[keep].astype(np.float64)
    affine = np.column_stack([np.ones(len(lat)), lat])
    sol, *_ = np.linalg.lstsq(affine, pix, rcond=None)
    rms = lambda p: float(np.sqrt(((p - pix) ** 2).sum(1).mean())) / pitch
    left, right = lat[:, 0].min(), lat[:, 0].max()
    top, bottom = lat[:, 1].min(), lat[:, 1].max()
    mid_i, mid_j = float(np.median(lat[:, 0])), float(np.median(lat[:, 1]))
    jl, jr = _jacobian(h, left, mid_j), _jacobian(h, right, mid_j)
    jt, jb = _jacobian(h, mid_i, top), _jacobian(h, mid_i, bottom)
    jc = _jacobian(h, mid_i, mid_j)
    ang = lambda v: float(np.degrees(np.arctan2(v[1], v[0])))
    return {"n_dots": len(dots), "n_inliers": int(keep.sum()), "extent_w_mm": float(right - left), "extent_h_mm": float(bottom - top),
            "px_per_mm_x": float(np.hypot(*jc[:, 0])), "px_per_mm_y": float(np.hypot(*jc[:, 1])), "aniso": float(np.hypot(*jc[:, 1]) / np.hypot(*jc[:, 0])), "tilt_deg": ang(jc[:, 0]),
            "rms_affine_mm": rms(affine @ sol), "rms_homography_mm": rms(cv2.perspectiveTransform(lat[None], h)[0]), "rms_quadratic_mm": rms(_quadratic(lat, pix)),
            "persp_x_pct": float(100 * (np.hypot(*jr[:, 0]) / np.hypot(*jl[:, 0]) - 1)), "persp_y_pct": float(100 * (np.hypot(*jb[:, 1]) / np.hypot(*jt[:, 1]) - 1)),
            "keystone_deg": float(ang(jb[:, 0]) - ang(jt[:, 0]))}


def measure_panel(gray: np.ndarray, grid_cfg: dict) -> dict:
    dots = find_dots(gray, grid_cfg["blackhat_kernel"], grid_cfg["blackhat_contrast"], tuple(grid_cfg["dot_area"]), grid_cfg["dot_max_side"])
    if len(dots) < 200:
        raise ValueError(f"too few grid dots found ({len(dots)})")
    return measure_dots(dots, grid_cfg, gray)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=paths.resolve, required=True)
    ap.add_argument("--layout", default="mac400")
    ap.add_argument("--out", type=paths.resolve, help="csv, default <run>/warp.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid_cfg = load_layout(args.layout)["grid"]
    mx, my = cfg["panel_margin_px"]
    rows = []
    for page_json in tqdm(sorted(args.run.glob("*/page.json")), desc="pages"):
        page = json.loads(page_json.read_text(encoding="utf-8"))
        src = Path(page["source"])
        src = src if src.is_absolute() else paths.REPO / src
        if not src.exists():
            continue
        image = load_image(src)
        scale = page.get("work_scale", 1.0)  # runs before the downscale was added saved no factor and used the full size
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        up = np.rot90(image, page["rotation_ccw_deg"] // 90).copy()
        for pj in sorted(page_json.parent.glob("panel*.json")):
            meta = json.loads(pj.read_text(encoding="utf-8"))
            x0, y0, x1, y1 = meta["box"]
            crop = up[max(y0 - my, 0):min(y1 + my, up.shape[0]), max(x0 - mx, 0):min(x1 + mx, up.shape[1])]
            try:
                m = measure_panel(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), grid_cfg)
            except ValueError:
                continue
            rows.append({"page": page_json.parent.name, "panel": meta["panel_id"], **m})
    out = args.out or args.run / "warp.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    q = lambda k, f: float(np.percentile(np.abs([r[k] for r in rows]), f))
    print("| Measure | median | 90th percentile |\n|---|---|---|")
    for k, label in (("persp_x_pct", "pitch change left to right, percent"), ("persp_y_pct", "pitch change top to bottom, percent"), ("keystone_deg", "turn of lattice lines top to bottom, degrees"),
                     ("rms_affine_mm", "affine residual, mm"), ("rms_homography_mm", "homography residual, mm"), ("rms_quadratic_mm", "quadratic residual, mm")):
        print(f"| {label} | {q(k, 50):.3f} | {q(k, 90):.3f} |")
    print(f"\n{len(rows)} panels measured")


if __name__ == "__main__":
    main()
