"""Re-run stage 1 (segment_sheets.py, fresh SAM) AND stage 2
(detect_and_digitize_columns.py) on the QA set from the original raw photos.

Unlike rerun_stage2.py, this does not reuse cached sheet crops from an
earlier run -- needed whenever a change touches how the raw image itself is
loaded (e.g. the EXIF-orientation fix, which happens before SAM ever sees
the image), since a cached "sheets/" directory would silently still reflect
the pre-fix crops.

Usage:
    uv run python -m src.qa.rerun_stage1_and_2 --labels ../../analyses/stage1-qa/labels.csv \
        --dst-work ../../inferences/stage1-qa-run10
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT.parents[1]
SAM_PY = REPO_ROOT / ".venvs" / "sam" / "Scripts" / "python.exe"
PAPERECG_PY = REPO_ROOT / ".venvs" / "paper-ecg" / "Scripts" / "python.exe"
SEGMENT_SHEETS = REPO_ROOT / "src" / "inference" / "segment_sheets.py"
STAGE2 = REPO_ROOT / "src" / "inference" / "detect_and_digitize_columns.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run stage 1 (fresh SAM) + stage 2 for the QA ids.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dst-work", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    labeled = labels[labels["lead_columns"].notna() & (labels["lead_columns"].astype(str).str.strip() != "")]
    log = args.dst_work / "rerun_stage1_and_2.log"
    args.dst_work.mkdir(parents=True, exist_ok=True)

    with log.open("a", encoding="utf-8") as logf:
        for _, row in tqdm(list(labeled.iterrows()), desc="stage1+2"):
            rid = row["id"]
            sheets = args.dst_work / rid / "sheets"
            digitized = args.dst_work / rid / "digitized"
            if digitized.exists() and any(digitized.glob("*_sheet_summary.json")):
                continue  # resumable

            source = DATASET_ROOT / row["full_path"]
            sheets.mkdir(parents=True, exist_ok=True)
            result = subprocess.run([str(SAM_PY), str(SEGMENT_SHEETS), "--input", str(source), "--output-dir", str(sheets)], capture_output=True, text=True)
            logf.write(f"== {rid} / segment_sheets (rc={result.returncode})\n{result.stdout}{result.stderr[-800:] if result.returncode else ''}\n")
            logf.flush()

            digitized.mkdir(parents=True, exist_ok=True)
            for sheet in sorted(sheets.glob("*_sheet*.png")):
                result = subprocess.run([str(PAPERECG_PY), str(STAGE2), "--input", str(sheet), "--output-dir", str(digitized)], capture_output=True, text=True)
                logf.write(f"== {rid} / {sheet.name} (rc={result.returncode})\n{result.stdout}{result.stderr[-800:] if result.returncode else ''}\n")
                logf.flush()

    print(f"done, {len(labeled)} ids, log at {log}")


if __name__ == "__main__":
    main()
