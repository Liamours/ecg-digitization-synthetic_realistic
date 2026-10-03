"""Re-run stage 2 (detect_and_digitize_columns.py) on sheet crops that an
earlier pipeline run already produced, so a stage-2 change can be scored on
the QA set without re-running SAM.

Copies <src-work>/<id>/sheets/ to <dst-work>/<id>/sheets/ for every labeled
id, digitizes each sheet into <dst-work>/<id>/digitized/, then
src/qa/score_stage1.py can score <dst-work>.

Usage:
    uv run python -m src.qa.rerun_stage2 --labels ../../analyses/stage1-qa/labels.csv \
        --src-work ../../inferences/full-pipeline-test-v6 --dst-work ../../inferences/stage1-qa-run1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPERECG_PY = REPO_ROOT / ".venvs" / "paper-ecg" / "Scripts" / "python.exe"
STAGE2 = REPO_ROOT / "src" / "inference" / "detect_and_digitize_columns.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run stage 2 on existing sheet crops for the QA ids.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--src-work", type=Path, required=True)
    parser.add_argument("--dst-work", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = [rid for rid in pd.read_csv(args.labels)["id"] if (args.src_work / rid / "sheets").exists()]
    log = args.dst_work / "rerun_stage2.log"
    args.dst_work.mkdir(parents=True, exist_ok=True)

    with log.open("a", encoding="utf-8") as logf:
        for rid in tqdm(ids, desc="stage2"):
            src_sheets = args.src_work / rid / "sheets"
            dst_sheets = args.dst_work / rid / "sheets"
            digitized = args.dst_work / rid / "digitized"
            if digitized.exists() and any(digitized.glob("*_sheet_summary.json")):
                continue  # resumable: this id was already redone
            if dst_sheets.exists():
                shutil.rmtree(dst_sheets)
            shutil.copytree(src_sheets, dst_sheets)
            digitized.mkdir(parents=True, exist_ok=True)
            for sheet in sorted(dst_sheets.glob("*_sheet*.png")):
                result = subprocess.run([str(PAPERECG_PY), str(STAGE2), "--input", str(sheet), "--output-dir", str(digitized)], capture_output=True, text=True)
                logf.write(f"== {rid} / {sheet.name} (rc={result.returncode})\n{result.stdout}{result.stderr[-800:] if result.returncode else ''}\n")
                logf.flush()

    print(f"done, {len(ids)} ids, log at {log}")


if __name__ == "__main__":
    main()
