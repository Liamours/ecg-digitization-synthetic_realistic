"""Auto-tune panel_geometry.find_calibration_squares's thresholds against
synthetic ground truth (dataset/synthetic-ptbxl-panels/manifest.csv's
true_square_boxes), instead of hand-adjusting them one failing case at a
time -- per the user's 2026-09-04 direction to change method now that
labeled synthetic data can be generated at scale, while staying classical/
rule-based (grid search over existing thresholds, nothing learned).

Pitch is estimated once per panel and cached, since it doesn't depend on
the thresholds being tuned and is the expensive part; only the fast
find_calibration_squares call is repeated per parameter combination.

IMPORTANT: a synthetic-set win is necessary, not sufficient. Any candidate
this script recommends must still be re-verified against the real
stage1-qa set (score_stage1.py) before being kept -- see this project's own
prior lesson (analyses/stage1-qa/README.md's reverted dot-spacing attempt)
about a synthetic/single-case fix regressing real data.

Usage:
    .venvs/paper-ecg/Scripts/python src/synthetic/tune_calibration_thresholds.py \
        --manifest ../../dataset/synthetic-ptbxl-panels/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import panel_geometry as pg  # noqa: E402

IOU_MATCH_THRESHOLD = 0.3

# The parameters most implicated by this session's direct debugging of
# synthetic panels (real marks measured well outside the current bounds),
# not a blind full-parameter sweep.
PARAM_GRID = {
    "SQUARE_SIDE_PERIODS": [(1.5, 9.0), (2.0, 9.0), (2.5, 9.0), (3.0, 9.0)],
    "THIN_MARK_MIN_PERIODS": [0.5, 0.7, 0.9, 1.1],
    "THIN_MARK_FILL_MIN": [0.75, 0.85, 0.92],
}


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_and_cache(manifest_path: Path) -> list[dict]:
    cached = []
    with manifest_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        true_boxes = []
        if row["true_square_boxes"]:
            for part in row["true_square_boxes"].split(";"):
                x, y, bw, bh = map(int, part.split(","))
                true_boxes.append((x, y, bw, bh))
        bgr = cv2.imread(row["panel_path"])
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pitch, _, _ = pg.estimate_pitch(bgr)
        cached.append({"panel_id": row["id"], "gray": gray, "pitch": pitch, "true_boxes": true_boxes})
        if (i + 1) % 20 == 0:
            print(f"  cached pitch for {i + 1}/{len(rows)} panels")
    return cached


def evaluate(cached: list[dict], params: dict) -> dict:
    original = {k: getattr(pg, k) for k in params}
    for k, v in params.items():
        setattr(pg, k, v)
    try:
        total_true = total_matched = total_detected = 0
        for entry in cached:
            total_true += len(entry["true_boxes"])
            if entry["pitch"] is None:
                continue
            detected = pg.find_calibration_squares(entry["gray"], entry["pitch"])
            total_detected += len(detected)
            used = set()
            for tb in entry["true_boxes"]:
                best_iou, best_j = 0.0, None
                for j, db in enumerate(detected):
                    if j in used:
                        continue
                    score = iou(tb, db)
                    if score > best_iou:
                        best_iou, best_j = score, j
                if best_j is not None and best_iou >= IOU_MATCH_THRESHOLD:
                    used.add(best_j)
                    total_matched += 1
        recall = total_matched / total_true if total_true else 0.0
        precision = total_matched / total_detected if total_detected else 0.0
        return {"recall": recall, "precision": precision, "matched": total_matched, "true": total_true, "detected": total_detected}
    finally:
        for k, v in original.items():
            setattr(pg, k, v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search find_calibration_squares thresholds against synthetic ground truth.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Baseline (current thresholds):")
    cached = load_and_cache(args.manifest)
    baseline_params = {k: getattr(pg, k) for k in PARAM_GRID}
    baseline = evaluate(cached, baseline_params)
    print(f"  {baseline_params} -> {baseline}")

    keys = list(PARAM_GRID)
    combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))
    print(f"\nSearching {len(combos)} combinations over {len(cached)} cached panels...")
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        result = evaluate(cached, params)
        results.append((params, result))

    # Rank by recall first (the metric this tuning pass targets, given
    # every failure diagnosed this session was a false negative, not a
    # false positive), precision as the tiebreaker.
    results.sort(key=lambda r: (r[1]["recall"], r[1]["precision"]), reverse=True)
    print(f"\nTop {args.top_n} by recall (tiebreak precision):")
    for params, result in results[: args.top_n]:
        print(f"  {params} -> recall={result['recall']:.3f} precision={result['precision']:.3f} (matched {result['matched']}/{result['true']}, detected {result['detected']})")


if __name__ == "__main__":
    main()
