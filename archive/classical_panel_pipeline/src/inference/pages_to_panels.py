"""Task 2 (page -> panels) orchestrator: for each page in
dataset/<name>/preprocessed/pages/manifest.csv, segment it into physical
sheets (SAM venv), find and classify panels within each sheet (paper-ecg
venv), and append rows to dataset/<name>/preprocessed/panels/manifest.csv.

Resumable: a page already present in the output manifest is skipped, so a
crashed or interrupted run can just be restarted.

Usage:
    uv run python -m src.inference.pages_to_panels --dataset-root ../.. \
        --dataset ekg-757 [--ids id1,id2,...] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM_PY = REPO_ROOT / ".venvs" / "sam" / "Scripts" / "python.exe"
PAPERECG_PY = REPO_ROOT / ".venvs" / "paper-ecg" / "Scripts" / "python.exe"
SEGMENT_SHEETS = REPO_ROOT / "src" / "inference" / "segment_sheets.py"
PANELS_FROM_SHEETS = REPO_ROOT / "src" / "inference" / "panels_from_sheets.py"

PANEL_COLUMNS = ["id", "page_id", "panel_index", "panel_path", "sheet_index", "x", "y", "w", "h",
                 "rotation_deg", "tilt_deg", "anchor_conf", "ecg_or_non_ecg", "low_confidence", "notes"]
LEAD_COLUMNS = ["id", "panel_id", "row_index", "lead_path", "x", "y", "w", "h", "lead_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2: page -> panels, run over a dataset's pages.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Project root (parent of dataset/)")
    parser.add_argument("--dataset", required=True, help="e.g. ekg-757")
    parser.add_argument("--ids", help="Comma-separated page ids to run (default: all pages not yet in the output manifest)")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of pages processed this run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessed = args.dataset_root / "dataset" / args.dataset / "preprocessed"
    pages_manifest = preprocessed / "pages" / "manifest.csv"
    panels_dir = preprocessed / "panels"
    leads_dir = preprocessed / "leads"
    panels_manifest_path = panels_dir / "manifest.csv"
    leads_manifest_path = leads_dir / "manifest.csv"
    panels_dir.mkdir(parents=True, exist_ok=True)
    leads_dir.mkdir(parents=True, exist_ok=True)

    with pages_manifest.open(encoding="utf-8") as f:
        pages = list(csv.DictReader(f))
    if args.ids:
        wanted = set(args.ids.split(","))
        pages = [p for p in pages if p["id"] in wanted]

    done_ids = set()
    if panels_manifest_path.exists():
        with panels_manifest_path.open(encoding="utf-8") as f:
            done_ids = {row["page_id"] for row in csv.DictReader(f)}
    pages = [p for p in pages if p["id"] not in done_ids]
    if args.limit:
        pages = pages[: args.limit]

    write_panel_header = not panels_manifest_path.exists() or panels_manifest_path.stat().st_size == 0
    write_lead_header = not leads_manifest_path.exists() or leads_manifest_path.stat().st_size == 0
    with panels_manifest_path.open("a", newline="", encoding="utf-8") as panels_f, \
         leads_manifest_path.open("a", newline="", encoding="utf-8") as leads_f:
        panel_writer = csv.DictWriter(panels_f, fieldnames=PANEL_COLUMNS)
        lead_writer = csv.DictWriter(leads_f, fieldnames=LEAD_COLUMNS)
        if write_panel_header:
            panel_writer.writeheader()
        if write_lead_header:
            lead_writer.writeheader()

        for i, page in enumerate(pages):
            page_id = page["id"]
            page_path = args.dataset_root / page["page_path"]
            print(f"[{i + 1}/{len(pages)}] {page_id}")
            with tempfile.TemporaryDirectory(prefix=f"sheets_{page_id}_") as tmp:
                sheets_dir = Path(tmp)
                result = subprocess.run(
                    [str(SAM_PY), str(SEGMENT_SHEETS), "--input", str(page_path), "--output-dir", str(sheets_dir)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"  segment_sheets failed (rc={result.returncode}): {result.stderr[-500:]}")
                    continue

                sheet_paths = sorted(sheets_dir.glob("*_sheet*.png"))
                next_panel_index = 0
                n_leads = 0
                for sheet_index, sheet_path in enumerate(sheet_paths):
                    result = subprocess.run(
                        [str(PAPERECG_PY), str(PANELS_FROM_SHEETS),
                         "--page-id", page_id, "--sheet-index", str(sheet_index), "--sheet", str(sheet_path),
                         "--output-dir", str(panels_dir), "--start-panel-index", str(next_panel_index)],
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        print(f"  panels_from_sheets failed on sheet {sheet_index} (rc={result.returncode}): {result.stderr[-500:]}")
                        continue
                    output = json.loads(result.stdout.strip() or '{"panels": [], "leads": []}')
                    for row in output["panels"]:
                        panel_writer.writerow(row)
                    for row in output["leads"]:
                        lead_writer.writerow(row)
                    next_panel_index += len(output["panels"])
                    n_leads += len(output["leads"])
                panels_f.flush()
                leads_f.flush()
                print(f"  {next_panel_index} panel(s), {n_leads} lead(s) from {len(sheet_paths)} sheet(s)")

    print(f"done, {len(pages)} page(s) processed, output at {panels_manifest_path} and {leads_manifest_path}")


if __name__ == "__main__":
    main()
