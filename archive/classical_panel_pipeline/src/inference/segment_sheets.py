"""Stage 1 of the compositing pipeline: split a composite photo into its
separate physical sheets using SAM (zero-shot, no per-image tuning).

See analyses/compositing-preprocessing/README.md, "Attempt 7" -- SAM's
automatic mask generator finds the real paper-sheet boundaries directly;
classical CV (brightness thresholds, watershed) and a hand-tuned
calibration-square heuristic both failed to generalize across images because
they were hunting for boundaries between printed columns that don't
physically exist within one sheet.

Each accepted mask is deskewed via minAreaRect + perspective warp (not just
an axis-aligned bounding-box crop) since a photographed sheet can be rotated,
not only translated.

Output feeds into detect_and_digitize_columns.py (a different venv --
paper-ecg needs old numpy/scipy that conflicts with SAM/torch's versions).

Usage:
    .venvs/sam/Scripts/python src/inference/segment_sheets.py \
        --input <path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT = Path(__file__).resolve().parents[4] / "external" / "segment-anything-weights" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def load_image_exif_corrected(path: Path) -> np.ndarray:
    """BGR array with EXIF orientation actually applied.

    cv2.imread reads raw stored pixels and ignores the EXIF Orientation tag
    entirely -- confirmed 233 of 757 ekg-757 images (31%) carry a non-identity
    tag (180 degrees on 121, 270 on 92, 90 on 20), so every downstream stage
    was silently operating on a frame rotated from what a person (or any
    EXIF-respecting viewer) actually sees. ecg_esta carries no EXIF tags on
    any of its 1143 images, so this is a no-op there.
    """
    pil_img = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

# A real sheet fills a substantial fraction of the photo; this drops the many
# small distractor masks (text labels, calibration squares, pen marks) SAM's
# automatic mode also proposes.
MIN_SHEET_AREA_FRACTION = 0.08


def load_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=device)
    return sam


def generate_candidate_masks(image_rgb: np.ndarray, sam) -> list[dict]:
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        points_per_batch=32,
        crop_n_layers=0,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        min_mask_region_area=20000,
    )
    return generator.generate(image_rgb)


def find_sheet_masks(image_rgb: np.ndarray, sam) -> list[np.ndarray]:
    kept, _, _ = find_sheet_masks_with_rejects(image_rgb, sam)
    return kept


def find_sheet_masks_with_rejects(image_rgb: np.ndarray, sam) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Same filtering as find_sheet_masks, but also returns what it rejected
    and why: masks below MIN_SHEET_AREA_FRACTION (SAM's small distractor
    proposals -- text, calibration squares, pen marks), and sheet-sized
    masks drop_containing_masks then dropped as a coarser union of masks
    already kept. Lets a diagnostic tool show the discards, not just what
    survived, without running SAM's (expensive) generate() a second time."""
    results = generate_candidate_masks(image_rgb, sam)
    image_area = image_rgb.shape[0] * image_rgb.shape[1]

    too_small = [r["segmentation"] for r in results if r["area"] <= image_area * MIN_SHEET_AREA_FRACTION]
    candidates = [r["segmentation"] for r in results if r["area"] > image_area * MIN_SHEET_AREA_FRACTION]
    # Largest first, so the containment filter below always compares a
    # candidate against already-accepted *larger* masks.
    candidates.sort(key=lambda m: -m.sum())
    kept = drop_containing_masks(candidates)
    kept_ids = {id(m) for m in kept}
    coarse_union = [m for m in candidates if id(m) not in kept_ids]
    return kept, too_small, coarse_union


def drop_containing_masks(masks: list[np.ndarray], containment_thresh: float = 0.8) -> list[np.ndarray]:
    """Reject a mask that mostly just contains several smaller accepted masks.

    Observed non-determinism: on one run SAM proposed one "whole page" mask
    plus 3 correct sub-sheet masks nested inside it, instead of 3 clean
    sheets as on a prior run of the identical script/image. SAM's own
    automatic-mask-generator NMS dedupes by IoU, which doesn't catch a
    containment relationship between masks of very different sizes -- so we
    filter for it explicitly. Prefer the smaller, more specific masks over
    a coarse one that merges several real sheets together.
    """
    kept: list[np.ndarray] = []
    for mask in masks:
        contains_multiple_kept = 0
        for smaller in kept:
            overlap = np.logical_and(mask, smaller).sum()
            if overlap > containment_thresh * smaller.sum():
                contains_multiple_kept += 1
        if contains_multiple_kept >= 2:
            continue  # this mask is just a coarser union of masks we already kept
        kept.append(mask)
    return kept


def deskew_crop(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (crop, matrix) -- `matrix` is the forward homography (original
    page pixels -> deskewed crop pixels) `getPerspectiveTransform` computed,
    previously discarded right after use. Callers that need to map a box
    found in the crop back into the original page's frame invert it (see
    panels_from_sheets.py); a caller that doesn't care can still just take
    the crop and ignore the second element."""
    mask_u8 = mask.astype("uint8") * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)

    (w, h) = rect[1]
    if w < h:
        w, h = h, w  # keep sheets landscape-oriented, matching how they were photographed

    dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    src = order_points(box)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, matrix, (int(w), int(h))), matrix


def order_points(pts: np.ndarray) -> np.ndarray:
    # Sort 4 corners into (bottom-left, top-left, top-right, bottom-right) to
    # match `dst` above, regardless of minAreaRect's own point ordering.
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([bottom_left, top_left, top_right, bottom_right], dtype="float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a composite ECG photo into its physical sheets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = load_image_exif_corrected(args.input)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = load_sam()
    sheets = find_sheet_masks(image_rgb, sam)
    print(f"Found {len(sheets)} sheet(s)")

    for i, mask in enumerate(sheets):
        # Orientation (which of the 4 quadrant rotations is upright) is decided
        # per crop in detect_and_digitize_columns.py from the calibration
        # square and its header text; Tesseract OSD, used here before, rejected
        # trace-heavy crops as "too few characters" and left them sideways.
        crop, matrix = deskew_crop(image_bgr, mask)
        out_path = args.output_dir / f"{args.input.stem}_sheet{i}.png"
        cv2.imwrite(str(out_path), crop)
        # Sidecar file, not embedded in the PNG -- segment_sheets.py (SAM
        # venv) and panels_from_sheets.py (paper-ecg venv) only communicate
        # through files on disk (see pages_to_panels.py's subprocess
        # orchestration), same as the crop image itself. Downstream code
        # composes this with the discrete orientation rotation (already
        # tracked as `rotation_deg`) to map a panel/lead box found in the
        # crop back into this original page's own pixel frame.
        np.save(args.output_dir / f"{args.input.stem}_sheet{i}_matrix.npy", matrix)
        print(f"Saved {out_path} ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    main()
