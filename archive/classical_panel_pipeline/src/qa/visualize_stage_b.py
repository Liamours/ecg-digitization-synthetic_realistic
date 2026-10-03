"""Pipeline visualization, stage B (paper-ecg venv): for every sheet stage A
found, apply the real orientation fix, detect columns and rows the same way
detect_and_digitize_columns.py does, and save every intermediate crop --
not just one representative column, all of them.

Usage:
    .venvs/paper-ecg/Scripts/python src/qa/visualize_stage_b.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from detect_and_digitize_columns import build_column_lead_boxes, box_is_usable, ocr_lead_label  # noqa: E402
from panel_geometry import choose_orientation, looks_like_signal  # noqa: E402


def empty_placeholder() -> np.ndarray:
    """A small, visibly blank stand-in for a box with zero real area (too
    small / clipped at a sheet edge) -- there is no real crop to save."""
    return np.full((60, 60, 3), 235, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B: orientation fix + column/row detection for pipeline visualization.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.output_dir / "stage_a_manifest.json").read_text(encoding="utf-8"))

    sheet_summaries = []
    for sheet_name in manifest["sheets"]:
        stem = Path(sheet_name).stem
        raw = cv2.imread(str(args.output_dir / sheet_name))
        oriented, rotation_deg, pitch, squares = choose_orientation(raw)
        fixed_path = args.output_dir / f"{stem}_fixed.png"
        cv2.imwrite(str(fixed_path), oriented)  # always saved here, even at 0 deg, so every sheet has one

        summary = {"sheet": sheet_name, "fixed": fixed_path.name, "rotation_deg": rotation_deg, "pitch": pitch, "columns": []}
        if pitch is None or not squares:
            reason = "no grid pitch" if pitch is None else "no calibration squares"
            print(f"{sheet_name}: {reason}, no columns")
            summary["reason"] = reason
            sheet_summaries.append(summary)
            continue

        gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
        sheet_h, sheet_w = gray.shape
        for ci, column in enumerate(build_column_lead_boxes(gray, squares, pitch, sheet_w, sheet_h)):
            # Discarded columns are kept in the output (not skipped) so the
            # visualization can show them with their discard reason, matching
            # detect_and_digitize_columns.py's own two early-exit checks.
            usable = all(box_is_usable(box, pitch) for box in column["rows"])
            if not usable:
                lead_names = column["lead_names"]
                discard_reason = f"row box smaller than the minimum lead-strip size (anchor at the crop edge)"
            else:
                lead_names = [
                    name if name is not None else ocr_lead_label(gray, box, tuple(column["anchor"]) + (0, 0), pitch)
                    for name, box in zip(column["lead_names"], column["rows"])
                ]
                discard_reason = "OCR read no valid lead label on any row" if all(name is None for name in lead_names) else None

            x0 = min(column["anchor"][0], min(r[0] for r in column["rows"]))
            y0 = column["anchor"][1]
            x1 = max(r[0] + r[2] for r in column["rows"])
            y1 = column["rows"][-1][1] + column["rows"][-1][3]
            col_crop = oriented[max(0, y0 - 15):min(sheet_h, y1 + 15), max(0, x0 - 15):min(sheet_w, x1 + 15)]
            col_path = args.output_dir / f"{stem}_col{ci}.png"
            cv2.imwrite(str(col_path), empty_placeholder() if col_crop.size == 0 else col_crop)

            rows_out = []
            for ri, (box, name) in enumerate(zip(column["rows"], lead_names)):
                rx, ry, rw, rh = box
                row_crop = oriented[max(0, ry - 10):min(sheet_h, ry + rh + 10), max(0, rx - 10):min(sheet_w, rx + rw + 10)]
                row_path = args.output_dir / f"{stem}_col{ci}_row{ri}.png"
                # A box discarded for being too small (or clipped at a sheet
                # edge) can be zero-area -- there's nothing real to crop, so
                # save a placeholder rather than letting cv2.imwrite fail on
                # an empty array.
                cv2.imwrite(str(row_path), empty_placeholder() if row_crop.size == 0 else row_crop)
                # A column already discarded above never reaches the real
                # pipeline's row-level check, so its rows carry the column's
                # own reason rather than a row-specific one.
                if discard_reason is not None:
                    row_discarded, row_reason = True, discard_reason
                elif not looks_like_signal(gray, box, pitch):
                    row_discarded, row_reason = True, "rejected: doesn't look like a continuous trace (text/footer/etc.), zero-filled in production"
                else:
                    row_discarded, row_reason = False, None
                rows_out.append({"row": row_path.name, "lead_name": name, "discarded": row_discarded, "reason": row_reason})

            summary["columns"].append({
                "column": col_path.name, "lead_names": lead_names, "rows": rows_out,
                "discarded": discard_reason is not None, "reason": discard_reason,
            })
        n_usable = sum(1 for c in summary["columns"] if not c["discarded"])
        print(f"{sheet_name}: rotated {rotation_deg} deg, pitch {pitch:.2f}px, {n_usable} usable / {len(summary['columns'])} total column(s)")
        sheet_summaries.append(summary)

    (args.output_dir / "stage_b_manifest.json").write_text(json.dumps(sheet_summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
