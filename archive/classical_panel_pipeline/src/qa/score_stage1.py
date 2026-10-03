"""Score stage 1 (preprocess and crop) output against hand-labeled ground truth.

Ground truth: analyses/stage1-qa/labels.csv, one row per source image with
true_sheet_count, true_column_count, rotation_needed_deg, and lead_columns
(semicolon-separated columns, each a slash-separated lead triplet, e.g.
"I/II/III;aVR/aVL/aVF;V1/V2/V3;V4/V5/V6").

Pipeline output: a work dir laid out as run_full_pipeline.py writes it,
<work>/<id>/sheets/*_sheet*.png and <work>/<id>/digitized/*_sheet_summary.json.

Reports per-image and aggregate: sheet recall, column recall, lead-label
accuracy over the columns the pipeline did find. Orientation accuracy is
reported only when the work dir carries a rotation log (rotation.json per
id), which the current pipeline does not write yet.

Usage:
    uv run python -m src.qa.score_stage1 --labels ../../analyses/stage1-qa/labels.csv \
        --work-dir ../../inferences/full-pipeline-test-v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score stage 1 crops against hand labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None, help="Per-image scores, optional")
    return parser.parse_args()


def parse_lead_columns(cell: str) -> list[list[str]]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [col.split("/") for col in cell.split(";") if col.strip()]


def pipeline_output(work_dir: Path, rid: str) -> tuple[int, list[list[str | None]]]:
    rec = work_dir / rid
    n_sheets = len(list((rec / "sheets").glob("*_sheet*.png"))) if (rec / "sheets").exists() else 0
    columns: list[list[str | None]] = []
    if (rec / "digitized").exists():
        for summary in sorted((rec / "digitized").glob("*_sheet_summary.json")):
            for col in json.loads(summary.read_text(encoding="utf-8"))["columns"]:
                columns.append(col["lead_names"])
    return n_sheets, columns


def pipeline_rotations(work_dir: Path, rid: str) -> list[int]:
    """rotation_deg chosen for each sheet that reached a real orientation
    decision (grid pitch found). A sheet with no grid pitch returns
    rotation_deg=0 as a default, not a decision, so it's excluded here rather
    than silently counted as a correct (or incorrect) 0-degree call."""
    rec = work_dir / rid / "digitized"
    if not rec.exists():
        return []
    degrees = []
    for summary_path in sorted(rec.glob("*_sheet_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("grid_pitch_px") is not None:
            degrees.append(summary["rotation_deg"])
    return degrees


def score_image(row: pd.Series, work_dir: Path) -> dict:
    true_cols = parse_lead_columns(row["lead_columns"])
    n_sheets, found_cols = pipeline_output(work_dir, row["id"])

    true_leads = {lead for col in true_cols for lead in col}
    found_leads = [lead for col in found_cols for lead in col if lead]
    correct_leads = sum(1 for lead in found_leads if lead in true_leads)

    return {
        "id": row["id"],
        "true_sheets": int(row["true_sheet_count"]) if pd.notna(row["true_sheet_count"]) else None,
        "found_sheets": n_sheets,
        "true_columns": len(true_cols),
        "found_columns": len(found_cols),
        "true_leads": len(true_leads),
        "found_leads_labeled": len(found_leads),
        "found_leads_correct": correct_leads,
    }


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    labeled = labels[labels["lead_columns"].notna() & (labels["lead_columns"].astype(str).str.strip() != "")]
    if labeled.empty:
        raise SystemExit("no labeled rows in labels.csv yet")

    has_output = labeled["id"].map(lambda rid: (args.work_dir / rid).exists())
    if (~has_output).any():
        print(f"skipping {(~has_output).sum()} labeled id(s) with no output under {args.work_dir}")
    labeled = labeled[has_output]
    if labeled.empty:
        raise SystemExit("no labeled ids have output in this work dir")

    per_image = pd.DataFrame([score_image(row, args.work_dir) for _, row in labeled.iterrows()])

    sheet_rows = per_image[per_image["true_sheets"].notna()]
    sheet_recall = (sheet_rows["found_sheets"].clip(upper=sheet_rows["true_sheets"]).sum() / sheet_rows["true_sheets"].sum()) if len(sheet_rows) else float("nan")
    column_recall = per_image["found_columns"].clip(upper=per_image["true_columns"]).sum() / per_image["true_columns"].sum()
    lead_recall = per_image["found_leads_correct"].sum() / per_image["true_leads"].sum()
    label_precision = per_image["found_leads_correct"].sum() / max(1, per_image["found_leads_labeled"].sum())

    print(f"images scored: {len(per_image)}")
    print(f"sheet recall:  {sheet_recall:.3f}")
    print(f"column recall: {column_recall:.3f}")
    print(f"lead recall:   {lead_recall:.3f}  (true leads recovered with the right label)")
    print(f"label precision: {label_precision:.3f}  (of OCR'd labels, fraction that are a true lead of that image)")

    # Orientation: labels.csv states one rotation_needed_deg per source image,
    # which isn't meaningful for the one row noted as MIXED ORIENTATION (its
    # sheets need different rotations each) -- excluded rather than scored
    # against a single number that can't be right for all of its sheets.
    oriented_labels = labeled[labeled["rotation_needed_deg"].notna()]
    mixed = oriented_labels["notes"].astype(str).str.contains("MIXED ORIENTATION", case=False, na=False)
    orientation_rows = [
        {"id": row["id"], "expected": int(row["rotation_needed_deg"]), "found": found}
        for _, row in oriented_labels[~mixed].iterrows()
        for found in pipeline_rotations(args.work_dir, row["id"])
    ]
    if orientation_rows:
        orient_df = pd.DataFrame(orientation_rows)
        orient_acc = (orient_df["expected"] == orient_df["found"]).mean()
        print(f"orientation accuracy: {orient_acc:.3f}  ({len(orient_df)} sheet(s) scored, {mixed.sum()} id(s) with per-sheet-mixed orientation excluded)")
    else:
        print("orientation accuracy: no sheets with a real orientation decision found")

    if args.out_csv:
        per_image.to_csv(args.out_csv, index=False)
        print(f"per-image scores -> {args.out_csv}")


if __name__ == "__main__":
    main()
