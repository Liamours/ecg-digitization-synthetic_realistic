"""Detect and crop individual ECG panels from a composite multi-strip photo.

Zero-training approach: every physical GE MAC 400 / GE panel carries its own
solid black calibration square near the "MAC 400"/"GE" label. Detecting these
(plain absolute-darkness blob detection, no OCR, no illumination correction
needed since we want true black ink, not local contrast) gives one reliable
anchor per panel, avoiding the two structural blockers documented in
analyses/compositing-preprocessing/README.md:
  - background-brightness methods (threshold+contour, watershed) fail because
    panels can be physically abutting with no real brightness gap between them
  - OCR-anchor text detection (a parallel session's Attempt 1) is unreliable
    across images (0-6 anchors found)

A thin morphological opening (kernel between the printed border-line's
thickness and the calibration square's size) is required before contour
detection -- without it, a square touching the panel's top border line merges
into one wide/thin blob and gets missed entirely.

Panel size per anchor is currently a fixed per-row template, not measured per
panel -- see README.md's "bottom-3" case for where this falls short (a panel
photographed at different zoom ends up with dead space in the crop). This is
a known limitation, not yet fixed.

Usage:
    .venvs/crop-scanned-photos/Scripts/python src/inference/detect_panel_anchors.py \
        --input <path-to-composite-image> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

# (width, height) per row, in pixels, at this dataset's typical photo resolution.
# Coarse and hand-derived from one image (analyses/compositing-preprocessing/README.md);
# not adaptive per panel yet.
ROW_PANEL_SIZE = {"top": (980, 950), "bottom": (940, 900)}
ROW_SPLIT_Y = 700  # anchors above this y are "top" row, below are "bottom"

# Offset from a calibration square's top-left corner to the panel's own top-left.
ANCHOR_TO_PANEL_OFFSET = (25, 95)


def find_calibration_squares(gray: "cv2.Mat") -> list[tuple[int, int, int, int]]:
    _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Strip thin border lines while keeping solid square blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        if rect_area == 0:
            continue
        fill_ratio = cv2.contourArea(c) / rect_area
        aspect = w / h if h else 0
        if 30 < w < 110 and 30 < h < 110 and fill_ratio > 0.55 and 0.6 < aspect < 1.6:
            squares.append((x, y, w, h))

    squares.sort(key=lambda b: (b[1], b[0]))
    return squares


def crop_panels(image: "cv2.Mat", squares: list[tuple[int, int, int, int]]) -> list["cv2.Mat"]:
    height, width = image.shape[:2]
    dx, dy = ANCHOR_TO_PANEL_OFFSET
    crops = []
    for x, y, _, _ in squares:
        row = "top" if y < ROW_SPLIT_Y else "bottom"
        panel_w, panel_h = ROW_PANEL_SIZE[row]
        x0 = max(0, x - dx)
        y0 = max(0, y - dy)
        x1 = min(width, x0 + panel_w)
        y1 = min(height, y0 + panel_h)
        crops.append(image[y0:y1, x0:x1])
    return crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect calibration-square anchors and crop panels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.input))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    squares = find_calibration_squares(gray)
    print(f"Found {len(squares)} calibration-square anchors: {squares}")

    for i, crop in enumerate(crop_panels(image, squares)):
        out_path = args.output_dir / f"{args.input.stem}_panel{i}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
