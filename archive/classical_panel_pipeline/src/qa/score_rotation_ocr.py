"""Score rotate_and_extract_text.py's orientation decision against the
same hand-labeled ground truth used for the rest of stage 1
(analyses/stage1-qa/labels.csv's rotation_needed_deg), and show what its
OCR text extraction actually pulls out on real images.

Runs the module's decision function directly against each QA image's raw
file, so it measures the whole-image rotation logic in isolation before
any sheet-splitting.

Usage:
    uv run python -m src.qa.score_rotation_ocr --labels ../../analyses/stage1-qa/labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytesseract

from src.datasets.registry import PROJECT_ROOT
from src.datasets.rotate_and_extract_text import best_rotation, load_scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score whole-image rotation + OCR text extraction against QA labels.")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "analyses" / "stage1-qa" / "labels.csv")
    parser.add_argument("--tesseract-cmd", type=Path, default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pytesseract.pytesseract.tesseract_cmd = str(args.tesseract_cmd)

    labels = pd.read_csv(args.labels)
    rows = []
    for _, r in labels.iterrows():
        path = PROJECT_ROOT / r["full_path"]
        img = load_scaled(path)
        decision = best_rotation(img)
        correct = int(decision["rotation_deg"]) == int(r["rotation_needed_deg"])
        rows.append({
            "id": r["id"],
            "dataset": r["dataset"],
            "true_deg": r["rotation_needed_deg"],
            "pred_deg": decision["rotation_deg"],
            "correct": correct,
            "n_words": decision["n_words"],
            "low_confidence": decision["low_confidence"],
            "text": decision["text"][:200],
        })
        print(f"{r['id']:45s} true={r['rotation_needed_deg']:>3} pred={decision['rotation_deg']:>3} "
              f"{'OK' if correct else 'MISS'}  n_words={decision['n_words']:3d}  low_conf={decision['low_confidence']}")

    df = pd.DataFrame(rows)
    print()
    print(f"Overall accuracy: {df['correct'].mean():.3f} ({df['correct'].sum()}/{len(df)})")
    print()
    print("By low_confidence flag:")
    print(df.groupby("low_confidence")["correct"].agg(["mean", "count"]))
    print()
    print("By dataset:")
    print(df.groupby("dataset")["correct"].agg(["mean", "count"]))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
