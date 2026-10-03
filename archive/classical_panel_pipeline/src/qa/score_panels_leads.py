"""Score Task 2/3 output (panels/manifest.csv, leads/manifest.csv -- the
page -> panel -> lead pipeline in src/inference/pages_to_panels.py and
panels_from_sheets.py) against the same hand-labeled ground truth
score_stage1.py uses for the older detect_and_digitize_columns.py pipeline.

Ground truth: analyses/stage1-qa/labels.csv, one row per source image with
true_sheet_count, true_column_count, rotation_needed_deg, and lead_columns
(semicolon-separated columns, each a slash-separated lead triplet).

Reports panel recall (found ecg panels vs true_column_count), lead recall
and label precision (OCR'd lead_name vs true lead_columns), and orientation
accuracy (rotation_deg found vs rotation_needed_deg, mixed-orientation ids
excluded, matching score_stage1.py's own convention).

Usage:
    uv run python -m src.qa.score_panels_leads --labels ../../analyses/stage1-qa/labels.csv \
        --dataset-root ../.. --dataset ekg-757
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score panels/leads output against hand labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def parse_lead_columns(cell: str) -> list[list[str]]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [col.split("/") for col in cell.split(";") if col.strip()]


def score_image(row: pd.Series, panels: pd.DataFrame, leads: pd.DataFrame) -> dict:
    true_cols = parse_lead_columns(row["lead_columns"])
    true_leads = {lead for col in true_cols for lead in col}

    own_panels = panels[panels["page_id"] == row["id"]]
    ecg_panels = own_panels[own_panels["ecg_or_non_ecg"] == "ecg"]
    own_leads = leads[leads["panel_id"].isin(ecg_panels["id"])]
    found_leads = [n for n in own_leads["lead_name"] if isinstance(n, str) and n.strip()]
    correct_leads = sum(1 for n in found_leads if n in true_leads)

    return {
        "id": row["id"],
        "true_columns": len(true_cols),
        "found_panels": len(own_panels),
        "found_ecg_panels": len(ecg_panels),
        "true_leads": len(true_leads),
        "found_leads_labeled": len(found_leads),
        "found_leads_correct": correct_leads,
        "rotation_degs": sorted(set(int(d) for d in own_panels["rotation_deg"].dropna().unique())),
    }


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    labeled = labels[labels["lead_columns"].notna() & (labels["lead_columns"].astype(str).str.strip() != "")]
    if labeled.empty:
        raise SystemExit("no labeled rows in labels.csv yet")

    preprocessed = args.dataset_root / "dataset" / args.dataset / "preprocessed"
    panels = pd.read_csv(preprocessed / "panels" / "manifest.csv")
    leads = pd.read_csv(preprocessed / "leads" / "manifest.csv")

    has_output = labeled["id"].isin(panels["page_id"])
    if (~has_output).any():
        print(f"skipping {(~has_output).sum()} labeled id(s) with no panels output yet")
    labeled = labeled[has_output]
    if labeled.empty:
        raise SystemExit("no labeled ids have panels output yet")

    per_image = pd.DataFrame([score_image(row, panels, leads) for _, row in labeled.iterrows()])

    panel_recall = per_image["found_ecg_panels"].clip(upper=per_image["true_columns"]).sum() / per_image["true_columns"].sum()
    lead_recall = per_image["found_leads_correct"].sum() / per_image["true_leads"].sum()
    label_precision = per_image["found_leads_correct"].sum() / max(1, per_image["found_leads_labeled"].sum())

    print(f"images scored: {len(per_image)}")
    print(f"panel recall:  {panel_recall:.3f}  (found ecg panels vs true column count, capped)")
    print(f"lead recall:   {lead_recall:.3f}  (true leads recovered with the right OCR'd label)")
    print(f"label precision: {label_precision:.3f}  (of OCR'd lead names, fraction that are a true lead of that image)")

    oriented_labels = labeled[labeled["rotation_needed_deg"].notna()]
    mixed = oriented_labels["notes"].astype(str).str.contains("MIXED ORIENTATION", case=False, na=False)
    orientation_rows = [
        {"id": row["id"], "expected": int(row["rotation_needed_deg"]), "found": found}
        for _, row in oriented_labels[~mixed].iterrows()
        for found in per_image.loc[per_image["id"] == row["id"], "rotation_degs"].iloc[0]
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
