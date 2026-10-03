"""Pipeline visualization, stage A (sam venv): segment one raw composite
photo into physical sheets, and additionally save an annotated copy of the
composite showing where each sheet was found -- segment_sheets.py itself
only saves the final deskewed crops, not this intermediate view.

Usage:
    .venvs/sam/Scripts/python src/qa/visualize_stage_a.py \
        --input <path> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from segment_sheets import deskew_crop, find_sheet_masks_with_rejects, load_image_exif_corrected, load_sam, order_points  # noqa: E402

BOX_COLORS = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (154, 88, 199), (0, 172, 193)]
DISCARD_RED = (0, 0, 255)  # BGR


def mask_box_points(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype("uint8") * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A: SAM segmentation + annotated composite for pipeline visualization.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = load_image_exif_corrected(args.input)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = load_sam()
    sheets, too_small, coarse_union = find_sheet_masks_with_rejects(image_rgb, sam)
    print(f"Found {len(sheets)} sheet(s), discarded {len(coarse_union)} coarse-union duplicate(s), "
          f"{len(too_small)} below the size threshold (not drawn -- SAM noise, never sheet candidates)")

    cv2.imwrite(str(args.output_dir / "00_composite.png"), image_bgr)

    annotated = image_bgr.copy()
    # Discarded candidates drawn first so a kept sheet's outline/label always
    # renders on top where the two overlap.
    for mask in coarse_union:
        box = mask_box_points(mask)
        cv2.polylines(annotated, [box.astype(np.int32)], isClosed=True, color=DISCARD_RED, thickness=6)
        label_pos = tuple(order_points(box)[1].astype(int))
        cv2.putText(annotated, "discarded (duplicate)", (label_pos[0], max(0, label_pos[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, DISCARD_RED, 3, cv2.LINE_AA)

    sheet_files = []
    for i, mask in enumerate(sheets):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        box = mask_box_points(mask)
        cv2.polylines(annotated, [box.astype(np.int32)], isClosed=True, color=color, thickness=6)
        label_pos = tuple(order_points(box)[1].astype(int))  # top-left corner
        cv2.putText(annotated, f"sheet{i}", (label_pos[0], max(0, label_pos[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4, cv2.LINE_AA)

        crop, _matrix = deskew_crop(image_bgr, mask)
        sheet_path = args.output_dir / f"sheet{i}.png"
        cv2.imwrite(str(sheet_path), crop)
        sheet_files.append(sheet_path.name)
        print(f"Saved {sheet_path} ({crop.shape[1]}x{crop.shape[0]})")

    cv2.imwrite(str(args.output_dir / "01_composite_bbox.png"), annotated)
    manifest = {"sheets": sheet_files, "discarded_duplicate_count": len(coarse_union), "discarded_too_small_count": len(too_small)}
    (args.output_dir / "stage_a_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
