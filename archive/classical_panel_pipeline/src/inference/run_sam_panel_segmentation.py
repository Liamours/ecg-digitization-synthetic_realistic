"""Zero-shot panel segmentation via Segment Anything (SAM).

Alternative to detect_panel_anchors.py's hand-tuned heuristic, which didn't
generalize across images (see analyses/compositing-preprocessing/README.md).
SAM needs no per-image constant tuning -- it predicts object boundaries
directly, at whatever scale/lighting the photo actually has.

Two modes:
  --mode auto   : SamAutomaticMaskGenerator proposes every object mask in the
                  image; we keep the ones that look like ECG panels (large,
                  roughly rectangular).
  --mode points : prompt SAM with point coordinates (e.g. the calibration-
                  square anchors detect_panel_anchors.py already finds
                  reliably) and take the returned mask per point.

Usage:
    .venvs/sam/Scripts/python src/inference/run_sam_panel_segmentation.py \
        --input <path> --output-dir <dir> --mode auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

CHECKPOINT = Path(__file__).resolve().parents[4] / "external" / "segment-anything-weights" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"


def load_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
    sam.to(device=device)
    return sam


def run_auto(image_rgb: np.ndarray, sam) -> list[dict]:
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        points_per_batch=32,
        crop_n_layers=0,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.9,
        min_mask_region_area=20000,  # drop tiny noise masks
    )
    return generator.generate(image_rgb)


def run_points(image_rgb: np.ndarray, sam, points: list[tuple[int, int]]) -> list[np.ndarray]:
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    masks = []
    for x, y in points:
        mask, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        masks.append(mask[np.argmax(scores)])
    return masks


def save_mask_crops(image_bgr: np.ndarray, masks: list[np.ndarray], output_dir: Path, stem: str) -> None:
    vis = image_bgr.copy()
    rng = np.random.default_rng(0)
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = image_bgr[y0:y1, x0:x1]
        cv2.imwrite(str(output_dir / f"{stem}_sam{i}.png"), crop)

        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)

    cv2.imwrite(str(output_dir / f"{stem}_sam_overlay.png"), vis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment ECG panels with SAM.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "points"], default="auto")
    parser.add_argument("--points", type=str, default=None, help="x1,y1;x2,y2;... (required for --mode points)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(args.input))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam = load_sam()

    if args.mode == "auto":
        results = run_auto(image_rgb, sam)
        print(f"SAM proposed {len(results)} masks (after min-area filter)")
        masks = [r["segmentation"] for r in results]
    else:
        points = [tuple(map(int, p.split(","))) for p in args.points.split(";")]
        masks = run_points(image_rgb, sam, points)
        print(f"Ran {len(points)} point prompts")

    save_mask_crops(image_bgr, masks, args.output_dir, args.input.stem)
    print(f"Saved crops and overlay to {args.output_dir}")


if __name__ == "__main__":
    main()
