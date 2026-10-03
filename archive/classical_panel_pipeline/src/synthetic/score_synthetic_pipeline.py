"""Phase C of the synthetic ground-truth compositor (transient-swimming-tide.md):
run the real pipeline (segment_sheets.py, then panels_from_sheets.py --
unchanged, the same code pages_to_panels.py orchestrates for real data) on
the synthetic pages from Phase B, match recovered panels back to the
injected ground truth by bounding-box IoU, and report the numbers this
whole effort exists to produce: discrete-rotation accuracy, tilt MAE,
bounding-box IoU, and false non_ecg rate -- pooled and split by the page's
orientation group (a: same vs b: mixed), since the mixed case is the one
this session already found a real bug in (EXIF/rotation) and is worth
tracking on its own, not averaged away.

Usage:
    uv run python -m src.synthetic.score_synthetic_pipeline \
        --pages-root ../../dataset/synthetic-ptbxl-pages --work-dir <scratch>
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM_PY = REPO_ROOT / ".venvs" / "sam" / "Scripts" / "python.exe"
PAPERECG_PY = REPO_ROOT / ".venvs" / "paper-ecg" / "Scripts" / "python.exe"
SEGMENT_SHEETS = REPO_ROOT / "src" / "inference" / "segment_sheets.py"
PANELS_FROM_SHEETS = REPO_ROOT / "src" / "inference" / "panels_from_sheets.py"


def polygon_to_bbox(row: dict) -> tuple[float, float, float, float]:
    xs = [float(row[f"x{i}"]) for i in range(4)]
    ys = [float(row[f"y{i}"]) for i in range(4)]
    return min(xs), min(ys), max(xs), max(ys)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_pipeline_on_page(page_path: Path, work_dir: Path) -> list[dict]:
    """Same two-stage call pages_to_panels.py makes for real data -- reused
    as-is, not reimplemented, so this scores the actual production code."""
    sheets_dir = work_dir / f"{page_path.stem}_sheets"
    result = subprocess.run([str(SAM_PY), str(SEGMENT_SHEETS), "--input", str(page_path), "--output-dir", str(sheets_dir)],
                             capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  segment_sheets failed: {result.stderr[-500:]}")
        return []

    panels_dir = work_dir / f"{page_path.stem}_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    recovered, next_index = [], 0
    for sheet_index, sheet_path in enumerate(sorted(sheets_dir.glob("*_sheet*.png"))):
        result = subprocess.run(
            [str(PAPERECG_PY), str(PANELS_FROM_SHEETS),
             "--page-id", page_path.stem, "--sheet-index", str(sheet_index), "--sheet", str(sheet_path),
             "--output-dir", str(panels_dir), "--start-panel-index", str(next_index)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  panels_from_sheets failed on sheet {sheet_index}: {result.stderr[-500:]}")
            continue
        rows = json.loads(result.stdout.strip() or "[]")
        recovered.extend(rows)
        next_index += len(rows)
    return recovered


def rotation_error_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def score_page(gt_rows: list[dict], recovered: list[dict]) -> list[dict]:
    """Best-IoU match: recovered panel geometry is in the SHEET's own
    deskewed frame, not the page frame the ground truth polygons use (a
    known, documented limitation of panels_from_sheets.py -- see its module
    docstring), so IoU here is necessarily approximate wherever SAM's
    deskew introduces real scale/perspective change. Treated as a soft
    match, not a strict correspondence."""
    gt_boxes = [(row, polygon_to_bbox(row)) for row in gt_rows]
    results = []
    used = set()
    for rec in recovered:
        rec_box = (rec["x"], rec["y"], rec["x"] + rec["w"], rec["y"] + rec["h"])
        best_iou, best_gt = 0.0, None
        for i, (gt_row, gt_box) in enumerate(gt_boxes):
            if i in used:
                continue
            score = iou(rec_box, gt_box)
            if score > best_iou:
                best_iou, best_gt = score, (i, gt_row)
        if best_gt is None or best_iou < 0.05:
            continue  # no plausible match at all -- not counted as a scored pair
        i, gt_row = best_gt
        used.add(i)
        true_rotation = int(gt_row["rotation_deg"])
        true_tilt = float(gt_row["tilt_deg"])
        results.append({
            "group": gt_row["group"],
            "iou": best_iou,
            "rotation_correct": rec["rotation_deg"] % 360 == true_rotation % 360,
            "tilt_error_deg": abs(rec["tilt_deg"] - true_tilt),
            "false_non_ecg": rec["ecg_or_non_ecg"] != "ecg",
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real pipeline on synthetic pages and score against injected ground truth.")
    parser.add_argument("--pages-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="score_synthetic_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.pages_root / "ground_truth.csv"
    with gt_path.open(encoding="utf-8") as f:
        all_gt = list(csv.DictReader(f))
    by_page: dict[str, list[dict]] = {}
    for row in all_gt:
        by_page.setdefault(row["page_id"], []).append(row)

    page_ids = sorted(by_page)[: args.limit] if args.limit else sorted(by_page)
    all_results = []
    for page_id in page_ids:
        page_path = Path(by_page[page_id][0]["page_path"])
        print(f"{page_id} ({len(by_page[page_id])} true panels)")
        recovered = run_pipeline_on_page(page_path, work_dir)
        results = score_page(by_page[page_id], recovered)
        print(f"  {len(recovered)} recovered, {len(results)} matched")
        all_results.extend(results)

    def report(subset: list[dict], label: str) -> None:
        if not subset:
            print(f"{label}: no matched pairs")
            return
        n = len(subset)
        rotation_acc = sum(r["rotation_correct"] for r in subset) / n
        tilt_mae = float(np.mean([r["tilt_error_deg"] for r in subset]))
        mean_iou = float(np.mean([r["iou"] for r in subset]))
        false_non_ecg_rate = sum(r["false_non_ecg"] for r in subset) / n
        print(f"{label}: n={n} rotation_acc={rotation_acc:.3f} tilt_mae={tilt_mae:.2f}deg mean_iou={mean_iou:.3f} false_non_ecg_rate={false_non_ecg_rate:.3f}")

    print()
    report(all_results, "overall")
    report([r for r in all_results if r["group"] == "a"], "group a (same-orientation)")
    report([r for r in all_results if r["group"] == "b"], "group b (mixed-orientation)")


if __name__ == "__main__":
    main()
