"""Automatic ground truth for grid rectification quality (problem #1 in
context/plan-digitization.md): a well-rectified sheet's printed grid is a
perfect axis-aligned square lattice, so its own regularity IS the target --
no hand-labeling needed, and it can run over the whole dataset immediately.

Method: tile the sheet, take each tile's 2D FFT magnitude spectrum (a
periodic grid produces sharp peaks whose position encodes direction and
period), find the strongest peak nearest the horizontal axis and the
strongest nearest the vertical axis, and report:
  - skew_deg: how far those peaks sit from perfectly horizontal/vertical
    (0 for a flat, unrotated grid)
  - pitch_cv: coefficient of variation of pitch across tiles and axes
    (0 if every tile agrees on the same pitch in both directions)

Currently zero (deskew-only, no true rectification) is the expected
baseline; a real rectification algorithm should measurably lower both.

Usage:
    uv run python -m src.qa.measure_grid_regularity --input <sheet-crop.png>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

TILES = 4  # NxN tiling
PEAK_SEARCH_HALF_ANGLE_DEG = 30  # how far from the axis a peak still counts as "that axis's" peak
DC_EXCLUDE_RADIUS = 4  # pixels around the zero-frequency center to zero out
MIN_PEAK_PROMINENCE = 6.0  # peak magnitude must be this many times the band's median to be trusted, not noise
MIN_PEAK_RADIUS_FRACTION = 0.03  # ignore peaks closer to DC than this fraction of the tile's Nyquist radius (large-scale shading/gradient, not a periodic grid)


def tile_grid_stats(tile_gray: np.ndarray) -> dict | None:
    h, w = tile_gray.shape
    f = np.fft.fftshift(np.fft.fft2(tile_gray.astype(np.float64)))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    mag[cy - DC_EXCLUDE_RADIUS: cy + DC_EXCLUDE_RADIUS + 1, cx - DC_EXCLUDE_RADIUS: cx + DC_EXCLUDE_RADIUS + 1] = 0

    ys, xs = np.mgrid[0:h, 0:w]
    dy, dx = (ys - cy).astype(np.float64), (xs - cx).astype(np.float64)
    radius = np.sqrt(dy**2 + dx**2)
    angle = np.degrees(np.arctan2(dy, dx))  # -180..180, 0 = along +x (horizontal), 90/-90 = vertical
    min_radius = MIN_PEAK_RADIUS_FRACTION * min(h, w) / 2

    def best_peak(target_angle: float) -> tuple[float, float] | None:
        # angular distance to target_angle, wrapped to 0..90 (line direction is 180-periodic)
        d = np.abs(((angle - target_angle + 90) % 180) - 90)
        band = (d <= PEAK_SEARCH_HALF_ANGLE_DEG) & (radius > min_radius)
        if not band.any():
            return None
        band_values = mag[band]
        masked = np.where(band, mag, 0)
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        peak_val = masked[idx]
        if peak_val <= 0:
            return None
        # Reject a peak that isn't meaningfully above the rest of its own search
        # band -- a tile with no real periodic grid (blank margin, pure trace
        # ink) still has *some* maximum in the band, but it won't stand out.
        band_median = np.median(band_values[band_values > 0]) if (band_values > 0).any() else 0
        if band_median <= 0 or peak_val / band_median < MIN_PEAK_PROMINENCE:
            return None
        peak_angle = angle[idx]
        peak_radius = radius[idx]
        dev = ((peak_angle - target_angle + 90) % 180) - 90
        return dev, peak_radius

    horiz = best_peak(0.0)
    vert = best_peak(90.0)
    if horiz is None or vert is None:
        return None
    h_dev, h_radius = horiz
    v_dev, v_radius = vert
    if h_radius <= 0 or v_radius <= 0:
        return None
    return {
        "h_skew_deg": h_dev,
        "v_skew_deg": v_dev,
        "h_pitch_px": w / h_radius,
        "v_pitch_px": h / v_radius,
    }


def measure(gray: np.ndarray) -> dict:
    h, w = gray.shape
    th, tw = h // TILES, w // TILES
    stats = []
    for i in range(TILES):
        for j in range(TILES):
            tile = gray[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if tile.size == 0:
                continue
            s = tile_grid_stats(tile)
            if s:
                stats.append(s)
    if not stats:
        return {"tiles_used": 0}

    skews = [abs(s["h_skew_deg"]) for s in stats] + [abs(s["v_skew_deg"]) for s in stats]
    pitches = [s["h_pitch_px"] for s in stats] + [s["v_pitch_px"] for s in stats]
    pitches = np.array(pitches)
    return {
        "tiles_used": len(stats),
        "tiles_total": TILES * TILES,
        "max_skew_deg": float(np.max(skews)),
        "median_skew_deg": float(np.median(skews)),
        "pitch_cv": float(np.std(pitches) / np.mean(pitches)) if np.mean(pitches) > 0 else float("nan"),
        "pitch_median_px": float(np.median(pitches)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure grid skew and pitch-consistency (rectification quality ground truth).")
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gray = cv2.cvtColor(cv2.imread(str(args.input)), cv2.COLOR_BGR2GRAY)
    result = measure(gray)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
