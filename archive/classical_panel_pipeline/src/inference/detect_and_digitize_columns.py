"""Stage 2 of the compositing pipeline: orient one sheet crop (from
segment_sheets.py), find its lead-columns via calibration-square anchors,
read each row's printed lead label, and digitize each column with paper-ecg.

Geometry is expressed in grid periods, from `panel_geometry.py`: the crop's
grid pitch sets the calibration-square size, the opening kernel, and every
row/column box. Earlier versions sized these as fractions of the crop
(`analyses/compositing-preprocessing/README.md`, Attempt 8), which broke as
soon as SAM merged two panel rows into one crop (window and kernel doubled,
squares erased). Orientation is decided here too, per crop, from where the
header text OCRs, since Tesseract's OSD rejected trace-heavy crops as
"too few characters" and left them sideways.

A column whose right neighbour is missing gets the sheet's median column
spacing (or the nominal panel width), never "extend to the sheet edge": a
missed anchor used to produce a box spanning two lead-columns and a blended
trace with no warning.

Lead identity is read via OCR on the printed label next to each row, not
guessed from column position (a 2-column sheet turned out to be V4/V5/V6
twice, not limb leads).

Usage:
    .venvs/paper-ecg/Scripts/python src/inference/detect_and_digitize_columns.py \
        --input <sheet-crop-path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PAPER_ECG_ROOT = Path(__file__).resolve().parents[4] / "external" / "paper-ecg" / "src" / "main" / "python"
sys.path.insert(0, str(PAPER_ECG_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ecgdigitize  # noqa: E402
import ecgdigitize.image  # noqa: E402
import ecgdigitize.signal  # noqa: E402
from ecgdigitize import common  # noqa: E402
from Conversion import convertECGLeads, exportSignals  # noqa: E402
from model.InputParameters import InputParameters  # noqa: E402
from model.Lead import Lead, LeadId  # noqa: E402

from panel_geometry import choose_orientation, looks_like_signal  # noqa: E402

TIME_SCALE = 25  # mm/s, printed on every strip inspected so far

VALID_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Panel layout in grid periods (1 mm), measured on real MAC 400 / GE panels:
# the calibration square sits at the panel's top-left, the header row is
# above the first lead row, three lead rows follow, then a footer. Panel
# height varies by print (47 mm and 66 mm both observed), so row height is
# derived from each panel's own bottom (the next panel row's squares, or the
# crop edge) instead of a fixed value.
PANEL_WIDTH_PERIODS = 79.0
HEADER_PERIODS = 6.0  # "MAC 400  V1.02" / "GE <date>" sits 3-5 periods below the square's top
FOOTER_PERIODS = 2.0  # last periods before the panel bottom carry footer text, never a trace
ROW_TOP_OFFSET_PERIODS = 3.2  # equal-split fallback only: first lead row starts this far below the square's top
ROW_HEIGHT_PERIODS_RANGE = (8.0, 26.0)
MIN_BASELINE_SEPARATION_FRACTION = 0.2  # of the panel height, between two trace baselines
BASELINE_MIN_INK = 0.04  # ink fraction across the profile strip; a baseline is a continuous line, a label is not
COLUMN_X_OFFSET_PERIODS = 2.0  # column box starts this far left of the square
COLUMN_WIDTH_FRACTION_OF_SPACING = 0.90
ROW_GROUP_GAP_PERIODS = 20.0  # squares further apart vertically than this belong to different panel rows
MIN_BOX_PERIODS = 8.0  # a row box smaller than this in either direction is not a lead strip
PROFILE_STRIP_PERIODS = (10.0, 45.0)  # ink profile is taken right of the square, past the lead labels, over this span
INK_THRESHOLD = 110

# Label window in grid periods: the printed lead name sits a fixed distance
# right of the square's left edge and just above its trace baseline,
# regardless of column width or panel height. The window never reaches up
# into the header band under the square.
LABEL_X_PERIODS = (1.0, 16.0)  # from the square's left edge
LABEL_ABOVE_BASELINE_PERIODS = (0.5, 7.0)  # window spans this far above the baseline
LABEL_MIN_COMPONENT_PERIODS = 0.8  # anything shorter is a grid dot, not a letter
OCR_UPSCALE = 3
FUZZY_MATCH_CUTOFF = 0.75


def group_into_rows(squares: list[tuple[int, int, int, int]], pitch: float) -> list[list[tuple[int, int, int, int]]]:
    rows: list[list[tuple[int, int, int, int]]] = []
    for sq in sorted(squares, key=lambda b: b[1]):
        if rows and sq[1] - rows[-1][-1][1] <= ROW_GROUP_GAP_PERIODS * pitch:
            rows[-1].append(sq)
        else:
            rows.append([sq])
    return [sorted(row, key=lambda b: b[0]) for row in rows]


def clean_for_ocr(gray_crop: np.ndarray, pitch: float) -> np.ndarray:
    """Upscale, binarize, drop the dotted grid by component height (dots are
    well under half a period tall, letters over one), return black-on-white
    with a white border so Tesseract sees a page."""
    up = cv2.resize(gray_crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_h = LABEL_MIN_COMPONENT_PERIODS * pitch * OCR_UPSCALE
    keep = np.zeros_like(binary)
    for i in range(1, n):
        t, l, h, w = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_WIDTH]
        if h >= min_h:
            keep[t: t + h, l: l + w] |= binary[t: t + h, l: l + w]
    return cv2.copyMakeBorder(cv2.bitwise_not(keep), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def find_lead_labels(gray, square: tuple[int, int, int, int], pitch: float, panel_bottom: int) -> list[tuple[str, int, int]]:
    """The printed lead names in the label column right of the square, as
    (name, top_px, bottom_px) sorted top to bottom. These are the row markers:
    each trace is printed directly under its own label. Ink-profile methods
    (three strongest bands, fitted equal spacing) were tried first and locked
    onto QRS excursions and the footer text instead of the baselines."""
    ax, ay, _, _ = square
    H, W = gray.shape
    x0, x1 = ax + int(LABEL_X_PERIODS[0] * pitch), min(W, ax + int(LABEL_X_PERIODS[1] * pitch))
    y0, y1 = ay + int(HEADER_PERIODS * pitch), min(H, panel_bottom)
    if x1 - x0 < 3 * pitch or y1 - y0 < 3 * ROW_HEIGHT_PERIODS_RANGE[0] * pitch:
        return []

    strip = clean_for_ocr(gray[y0:y1, x0:x1], pitch)
    data = pytesseract.image_to_data(strip, config="--psm 11", output_type=pytesseract.Output.DICT)
    found: list[tuple[str, int, int]] = []
    for i, text in enumerate(data["text"]):
        name = match_lead_name(text)
        if name is None:
            continue
        top = y0 + (data["top"][i] - 20) // OCR_UPSCALE
        bottom = top + data["height"][i] // OCR_UPSCALE
        if any(abs(top - t) < ROW_HEIGHT_PERIODS_RANGE[0] * pitch / 2 for _, t, _ in found):
            continue  # same label read twice
        found.append((name, top, bottom))
    return sorted(found, key=lambda f: f[1])[:3]


def build_column_lead_boxes(gray, squares: list[tuple[int, int, int, int]], pitch: float, sheet_w: int, sheet_h: int) -> list[dict]:
    rows = group_into_rows(squares, pitch)
    spacings = [row[i + 1][0] - row[i][0] for row in rows for i in range(len(row) - 1)]
    fallback_spacing = float(np.median(spacings)) if spacings else PANEL_WIDTH_PERIODS * pitch

    dx = int(COLUMN_X_OFFSET_PERIODS * pitch)
    lo_h, hi_h = (int(v * pitch) for v in ROW_HEIGHT_PERIODS_RANGE)

    columns = []
    for ri, row in enumerate(rows):
        panel_bottom = (min(sq[1] for sq in rows[ri + 1]) - int(2 * pitch)) if ri + 1 < len(rows) else sheet_h
        for i, square in enumerate(row):
            ax, ay, _, _ = square
            spacing = (row[i + 1][0] - ax) if i + 1 < len(row) else fallback_spacing
            x0 = max(0, ax - dx)
            col_w = int(spacing * COLUMN_WIDTH_FRACTION_OF_SPACING)
            partial = x0 + col_w > sheet_w
            col_w = min(col_w, sheet_w - x0)

            labels = find_lead_labels(gray, square, pitch, panel_bottom)
            footer_top = panel_bottom - int(FOOTER_PERIODS * pitch)
            names: list[str | None]
            if len(labels) >= 2:
                label_tops = [t for _, t, _ in labels]
                spacing = int(np.median(np.diff(label_tops))) if len(labels) == 3 else label_tops[1] - label_tops[0]
                spacing = int(min(hi_h, max(lo_h, spacing)))
                if len(labels) == 2:
                    # Third row missing: it sits after the pair if there is room, else before it.
                    if label_tops[1] + 2 * spacing <= footer_top:
                        label_tops.append(label_tops[1] + spacing)
                        names = [labels[0][0], labels[1][0], None]
                    else:
                        label_tops.insert(0, label_tops[0] - spacing)
                        names = [None, labels[0][0], labels[1][0]]
                else:
                    names = [n for n, _, _ in labels]
                tops = [t - int(1.0 * pitch) for t in label_tops]  # a row starts just above its label
                row_h = spacing
                rows_source = "labels"
            else:
                row_top = int(ROW_TOP_OFFSET_PERIODS * pitch)
                available = footer_top - (ay + row_top)
                row_h = int(min(hi_h, max(lo_h, available / 3)))
                tops = [ay + row_top + r * row_h for r in range(3)]
                names = [labels[0][0] if labels and abs(labels[0][1] - t) < row_h else None for t in tops]
                rows_source = "equal-split"
            partial = partial or tops[-1] + row_h > sheet_h
            row_boxes = [(x0, max(0, t), col_w, min(row_h, sheet_h - max(0, t))) for t in tops]

            columns.append({
                "anchor": (ax, ay), "rows": row_boxes, "lead_names": names, "rows_source": rows_source,
                "spacing_source": "neighbour" if i + 1 < len(row) else "fallback", "partial": partial,
            })
    return columns


def box_is_usable(box: tuple[int, int, int, int], pitch: float) -> bool:
    _, _, w, h = box
    return w >= MIN_BOX_PERIODS * pitch and h >= MIN_BOX_PERIODS * pitch


def ocr_lead_label(gray, row_box: tuple[int, int, int, int], square: tuple[int, int, int, int], pitch: float) -> str | None:
    """Second attempt for a row whose label the column-wide pass did not read:
    OCR just the label window of this row (label column x-range, upper part
    of the row box)."""
    ax = square[0]
    x0, x1 = ax + int(LABEL_X_PERIODS[0] * pitch), min(gray.shape[1], ax + int(LABEL_X_PERIODS[1] * pitch))
    _, y0, _, h = row_box
    crop = gray[y0: y0 + int(h * 0.6), x0: x1]
    if crop.size == 0:
        return None
    cleaned = clean_for_ocr(crop, pitch)
    raw = pytesseract.image_to_string(cleaned, config="--psm 7").strip()
    if not raw:
        raw = pytesseract.image_to_string(cleaned, config="--psm 11").strip()
    return match_lead_name(raw)


def match_lead_name(raw_text: str) -> str | None:
    """Token-based: each OCR token is compared to the lead names on its own.
    A bare substring pass was tried first and matched lead I inside almost
    any text ("MAC 400" read as "MIAC" -> "I"), which is confidently wrong
    rather than honestly missing."""
    if not raw_text:
        return None
    upper_names = [n.upper() for n in VALID_LEAD_NAMES]
    tokens = ["".join(ch for ch in tok if ch.isalnum()).replace("U", "V") for tok in raw_text.upper().split()]  # "U" is the common misread of "V"
    tokens = [t for t in tokens if t]

    for tok in tokens:
        if tok in upper_names:
            return VALID_LEAD_NAMES[upper_names.index(tok)]
    for tok in tokens:
        if len(tok) < 2:
            continue  # a lone character is not enough evidence for a fuzzy match
        matches = difflib.get_close_matches(tok, [n for n in upper_names if len(n) >= 2], n=1, cutoff=FUZZY_MATCH_CUTOFF)
        if matches:
            return VALID_LEAD_NAMES[upper_names.index(matches[0])]
    return None


def digitize_column(image, column: dict, lead_names: list[str | None], column_index: int, pitch: float, output_dir: Path, stem: str, gray: np.ndarray) -> dict | None:
    # LeadId is paper-ecg's fixed 12-lead enum, reused only as the dict key
    # type convertECGLeads requires; the physical lead is `lead_names`.
    slot_ids = [LeadId.I, LeadId.II, LeadId.III]
    leads = {slot_ids[r]: Lead(x=x, y=y, width=w, height=h, startTime=0) for r, (x, y, w, h) in enumerate(column["rows"])}

    signals, previews = convertECGLeads(image, InputParameters(rotation=0.0, timeScale=TIME_SCALE, voltScale=10, leads=leads))
    if signals is None:
        print(f"  column {column_index}: grid detection failed on all rows")
        return None

    # A row whose ink doesn't look like a continuous trace (many small
    # components, e.g. printed text) is zero-filled here even though
    # convertECGLeads returned something for it -- the viterbi extractor
    # doesn't know the difference between a real signal and letters, and will
    # confidently trace a nonsense path through text instead of failing.
    # Zero-filled, not dropped: dropping would shrink the exported CSV below
    # 3 columns, breaking the fixed-row-position indexing downstream
    # (assemble_12_lead reads column i by row position, not by name).
    rejected_as_text = 0
    for r, box in enumerate(column["rows"]):
        slot = slot_ids[r]
        if slot in signals and not looks_like_signal(gray, box, pitch):
            signals[slot] = np.zeros_like(signals[slot])
            previews.pop(slot, None)
            rejected_as_text += 1

    # Sample rate from the crop's voted grid pitch (panel_geometry.estimate_pitch),
    # the same estimate that sized the squares and rows. The per-row estimate
    # convertECGLeads uses internally locks onto 2x/5x harmonics on these
    # dotted grids (observed 1985 Hz from one misplaced box).
    sample_rate_hz = 1.0 / ecgdigitize.signal.ecgSignalSamplingPeriod(pitch, TIME_SCALE, gridSizeInMillimeters=1.0)

    out_csv = output_dir / f"{stem}_col{column_index}.csv"
    exportSignals(signals, out_csv, separator=",")
    for lead_id, preview in previews.items():
        label = lead_names[slot_ids.index(lead_id)] or f"unread{slot_ids.index(lead_id)}"
        ecgdigitize.image.saveImage(preview, output_dir / f"{stem}_col{column_index}_{label}.png")

    meta = {
        "csv": out_csv.name,
        "lead_names": lead_names,
        "sample_rate_hz": sample_rate_hz,
        "n_rows": len(signals) - rejected_as_text,
        "rejected_as_text": rejected_as_text,
        "anchor": list(column["anchor"]),
        "rows": [list(box) for box in column["rows"]],
        "rows_source": column["rows_source"],
        "spacing_source": column["spacing_source"],
        "partial": column["partial"],
    }
    (output_dir / f"{stem}_col{column_index}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    text_note = f", {rejected_as_text} row(s) rejected as text" if rejected_as_text else ""
    print(f"  column {column_index} ({'/'.join(n or '?' for n in lead_names)}): exported {len(signals) - rejected_as_text} row(s){text_note}, ~{sample_rate_hz:.0f}Hz")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orient a sheet crop, find lead-columns, digitize each.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    raw = cv2.imread(str(args.input))
    oriented, rotation_deg, pitch, squares = choose_orientation(raw)
    summary = {"rotation_deg": rotation_deg, "grid_pitch_px": pitch, "num_columns_found": len(squares), "columns": []}

    if pitch is None or not squares:
        reason = "no grid pitch" if pitch is None else "no calibration squares"
        print(f"{args.input.name}: {reason}, nothing to digitize")
        summary["reason"] = reason
        (args.output_dir / f"{stem}_sheet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    if rotation_deg:
        cv2.imwrite(str(args.output_dir / f"{stem}_oriented.png"), oriented)
    image = ecgdigitize.image.ColorImage(oriented)
    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
    sheet_h, sheet_w = gray.shape
    print(f"{args.input.name}: rotated {rotation_deg} deg, pitch {pitch:.2f}px, {len(squares)} column anchor(s) in a {sheet_w}x{sheet_h} sheet")

    for i, column in enumerate(build_column_lead_boxes(gray, squares, pitch, sheet_w, sheet_h)):
        if not all(box_is_usable(box, pitch) for box in column["rows"]):
            print(f"  column {i}: row box smaller than {MIN_BOX_PERIODS:g} grid periods (anchor at the crop edge), skipping")
            continue
        lead_names = [
            name if name is not None else ocr_lead_label(gray, box, tuple(column["anchor"]) + (0, 0), pitch)
            for name, box in zip(column["lead_names"], column["rows"])
        ]
        if all(name is None for name in lead_names):
            print(f"  column {i}: OCR read no valid lead label on any row ({column['rows_source']} rows), skipping")
            continue
        print(f"  column {i}: OCR read {lead_names} ({column['rows_source']} rows)" + (" (partial box)" if column["partial"] else ""))
        meta = digitize_column(image, column, lead_names, i, pitch, args.output_dir, stem, gray)
        if meta:
            summary["columns"].append(meta)

    (args.output_dir / f"{stem}_sheet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
