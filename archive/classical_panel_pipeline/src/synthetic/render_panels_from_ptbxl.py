"""Phase A of the synthetic ground-truth compositor (see the approved plan,
transient-swimming-tide.md): render PTB-XL records to full 12-lead pages via
ecg-image-kit's generator, then crop each of the 4 printed columns out as one
ground-truth "panel" (3 stacked leads + its own calibration square, matching
this project's own panel definition) instead of fighting the generator's
broken isolated-lead-count layout logic.

Two vendored patches already applied to make this generator usable here:
- gen_ecg_image_from_data.py: lazy-imports HandwrittenText.generate (avoids
  needing tensorflow/keras/spacy for a plain, no-handwriting render).
- ecg_plot.py: draws the calibration pulse per column, not once per printed
  row -- this project's real dataset prints one square per panel
  independently, unlike the standard clinical single-row-pulse convention
  the generator defaults to.

Gotcha confirmed directly from real output (not from the README): each
corner in `lead_bounding_box` is a [y, x] pair, not [x, y].

Usage:
    .venvs/ecg-image-kit-gen/Scripts/python src/synthetic/render_panels_from_ptbxl.py \
        --ptbxl-data ../../dataset/ptb-xl/data --output-root ../../dataset/synthetic-ptbxl-panels \
        --count 20
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GENERATOR_DIR = Path(__file__).resolve().parents[4] / "external" / "ecg-image-kit" / "codes" / "ecg-image-generator"
GENERATOR_SCRIPT = GENERATOR_DIR / "gen_ecg_images_from_data_batch.py"

PANEL_MARGIN_PX = 15
HEADER_MARGIN_PX = 55  # room above the top lead for "MAC 400   V1.02"
FOOTER_MARGIN_PX = 70  # room below the bottom lead for two footer lines

# The exact boilerplate text seen throughout this project's own real MAC 400
# scans all session -- reproducing it verbatim so this project's own header
# OCR/vocabulary check (panel_geometry.header_score) has something real to
# find on synthetic panels too, not a stand-in string it was never tuned on.
HEADER_TEXT = "MAC 400     V1.02"
FOOTER_LINE1 = "Man   25mm/s   10mm/mV   ADS"
FOOTER_LINE2 = "For 2030887-001   MDC72942181716008   INNOQ"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


PAPER_TINT_RGB = (247, 238, 205)  # warm cream/eggshell, matching real aged MAC 400 paper rather than pure white


def tint_paper(panel_bgr: np.ndarray) -> np.ndarray:
    """Multiply blend, not an overlay -- keeps dark ink/grid dark while the
    white paper background takes on the tint, same principle as a sepia
    filter. Applied before header/footer text so that text stays crisp."""
    tint_bgr = np.array(PAPER_TINT_RGB[::-1], dtype=np.float64) / 255.0
    tinted = panel_bgr.astype(np.float64) * tint_bgr
    return np.clip(tinted, 0, 255).astype(np.uint8)


SQUARE_WIDTH_PERIODS = 3.0
SQUARE_HEIGHT_PERIODS = 2.0
SQUARE_GAP_PERIODS = 0.5  # small visible gap between the square and the trace start

# ecg-image-kit's own JSON reports x_grid/y_grid as the MAJOR (5mm) grid box
# size. This project's whole pitch convention (SQUARE_SIDE_PERIODS etc. in
# panel_geometry.py) is the fine 1mm grid period instead -- confirmed
# directly: a square sized off the raw x_grid/y_grid came out exactly 5x
# taller (in periods) than intended, and dividing by 5 landed within a
# pixel of this project's own estimate_pitch on the same image.
MAJOR_TO_MINOR_GRID_RATIO = 5


def fine_grid_pitch(major_grid_px: float) -> float:
    return major_grid_px / MAJOR_TO_MINOR_GRID_RATIO


def square_margin_px(x_grid: float) -> int:
    """Extra left margin needed to fit the square plus its gap before the
    trace, so it never has to overlap real signal to exist."""
    return int((SQUARE_WIDTH_PERIODS + SQUARE_GAP_PERIODS) * x_grid) + PANEL_MARGIN_PX


def draw_calibration_squares(crop_bgr: np.ndarray, column: list[dict], cx0: int, cy0: int, x_grid: float, y_grid: float) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Overpaints one solid black rectangle per lead, positioned entirely
    within the reserved left margin so it never covers real trace pixels --
    sized to match what this project's own real MAC 400 scans actually show
    (a wide, short rectangle, confirmed by direct measurement, see
    analyses/stage1-qa/README.md's "non-square calibration mark" section),
    not ecg-image-kit's own thin dc-pulse, which turned out too small to
    survive our detector's opening kernel and, once widened, too easily
    merged with neighboring ink instead."""
    img = crop_bgr.copy()
    w_px = max(1, int(SQUARE_WIDTH_PERIODS * x_grid))
    h_px = max(1, int(SQUARE_HEIGHT_PERIODS * y_grid))
    gap_px = int(SQUARE_GAP_PERIODS * x_grid)
    true_boxes = []
    for lead in column:
        lx0, ly0, lx1, ly1 = lead["_bbox"]
        local_x = int(lx0 - cx0) - gap_px - w_px  # sits just left of where the trace itself starts
        local_y = int((ly0 + ly1) / 2 - cy0 - h_px / 2)
        cv2.rectangle(img, (local_x, local_y), (local_x + w_px, local_y + h_px), (10, 10, 10), thickness=-1)
        true_boxes.append((local_x, local_y, w_px, h_px))
    return img, true_boxes


def draw_header_footer(panel_bgr: np.ndarray) -> np.ndarray:
    """Draws the real MAC 400 boilerplate text into the header/footer margin
    already reserved around the cropped panel -- header to the calibration
    square's right (matching every real scan this session; the square
    itself sits near the panel's left edge from ecg-image-kit's own dc-pulse
    render, so a fixed offset lands just past it without needing its exact
    width), footer left-aligned below the last lead."""
    img = Image.fromarray(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    header_font = load_font(22, bold=True)
    footer_font = load_font(16)
    small_font = load_font(11)

    draw.text((90, 12), HEADER_TEXT, font=header_font, fill=(20, 20, 20))
    footer_y = img.height - FOOTER_MARGIN_PX + 8
    draw.text((10, footer_y), FOOTER_LINE1, font=footer_font, fill=(20, 20, 20))
    draw.text((10, footer_y + 26), FOOTER_LINE2, font=small_font, fill=(60, 60, 60))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
MANIFEST_COLUMNS = ["id", "record_id", "column_index", "panel_path", "lead_names", "true_square_boxes",
                    "x", "y", "w", "h", "x_grid", "y_grid", "sampling_frequency", "source_png"]


def corner_to_xy(corner: list[float]) -> tuple[float, float]:
    """lead_bounding_box's own convention: [y, x], confirmed against a real
    render (a lead's reported box only lined up with where it visually sits
    once read this way, not the other way round)."""
    y, x = corner
    return x, y


def lead_bbox_xyxy(lead_bounding_box: dict) -> tuple[float, float, float, float]:
    corners = [corner_to_xy(lead_bounding_box[k]) for k in ("0", "1", "2", "3")]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def drop_rhythm_strip(leads: list[dict]) -> list[dict]:
    """The generator's `full_mode` rhythm strip (default lead II, one full
    10s row spanning the whole image width) appears in the same `leads`
    list as the 12 real column leads. Its bounding-box center can land
    between two real columns and derail x-based clustering (confirmed
    directly: it threw off a real render's split). Identified by segment
    duration, not by lead name (name repeats one of the 12 real leads) --
    every real column lead covers one quarter-segment of the record; the
    rhythm strip alone covers the whole thing."""
    if not leads:
        return leads
    durations = [l["end_sample"] - l["start_sample"] for l in leads]
    real_duration = min(durations)
    return [l for l, d in zip(leads, durations) if d == real_duration]


def cluster_into_columns(leads: list[dict], expected_columns: int = 4) -> list[list[dict]]:
    """Groups leads by x-position into `expected_columns` panels, each 3
    leads stacked top-to-bottom. Not relying on list order in the JSON --
    confirmed directly that leads are NOT emitted in reading order (the
    first lead in a real render's JSON was "III", not "I")."""
    leads = drop_rhythm_strip(leads)
    enriched = []
    for lead in leads:
        x0, y0, x1, y1 = lead_bbox_xyxy(lead["lead_bounding_box"])
        enriched.append({**lead, "_xc": (x0 + x1) / 2, "_yc": (y0 + y1) / 2, "_bbox": (x0, y0, x1, y1)})
    enriched.sort(key=lambda l: l["_xc"])

    # Split on the `expected_columns - 1` largest gaps between consecutive
    # x-centers, rather than assuming a fixed pixel threshold -- robust to
    # different resolutions/paddings across records.
    gaps = [(enriched[i + 1]["_xc"] - enriched[i]["_xc"], i) for i in range(len(enriched) - 1)]
    gaps.sort(reverse=True)
    split_after = sorted(i for _, i in gaps[: expected_columns - 1])

    columns, start = [], 0
    for idx in split_after:
        columns.append(enriched[start: idx + 1])
        start = idx + 1
    columns.append(enriched[start:])
    for col in columns:
        col.sort(key=lambda l: l["_yc"])
    return columns


def extract_panels(png_path: Path, json_path: Path, record_id: str, output_dir: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img = cv2.imread(str(png_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    columns = cluster_into_columns(data["leads"], expected_columns=4)
    rows_out = []
    for k, column in enumerate(columns):
        if len(column) != 3:
            continue  # a malformed group (clustering failed to find a clean 3-lead column) -- skip rather than guess
        x0 = min(l["_bbox"][0] for l in column) - square_margin_px(fine_grid_pitch(data["x_grid"]))
        y0 = min(l["_bbox"][1] for l in column) - HEADER_MARGIN_PX
        x1 = max(l["_bbox"][2] for l in column) + PANEL_MARGIN_PX
        y1 = max(l["_bbox"][3] for l in column) + FOOTER_MARGIN_PX
        cx0, cy0 = max(0, int(x0)), max(0, int(y0))
        cx1, cy1 = min(w, int(x1)), min(h, int(y1))
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        crop, true_squares = draw_calibration_squares(crop, column, cx0, cy0, fine_grid_pitch(data["x_grid"]), fine_grid_pitch(data["y_grid"]))
        crop = tint_paper(crop)
        crop = draw_header_footer(crop)

        panel_id = f"{record_id}-col{k}"
        panel_path = output_dir / f"{panel_id}.png"
        cv2.imwrite(str(panel_path), crop)
        rows_out.append({
            "id": panel_id,
            "record_id": record_id,
            "column_index": k,
            "panel_path": str(panel_path),
            "lead_names": "/".join(l["lead_name"] for l in column),
            "true_square_boxes": ";".join(f"{x},{y},{bw},{bh}" for x, y, bw, bh in true_squares),
            "x": cx0, "y": cy0, "w": cx1 - cx0, "h": cy1 - cy0,
            "x_grid": data.get("x_grid"), "y_grid": data.get("y_grid"),
            "sampling_frequency": data.get("sampling_frequency"),
            "source_png": str(png_path),
        })
    return rows_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PTB-XL records and crop ground-truth panels from them.")
    parser.add_argument("--ptbxl-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panels_dir = args.output_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"

    with tempfile.TemporaryDirectory(prefix="ptbxl_render_") as tmp:
        render_dir = Path(tmp)
        # PTB-XL stores each record at two rates (_hr 500Hz, _lr 100Hz),
        # interleaved in directory order -- rendering N raw finds would give
        # ~N/2 unique underlying signals. Over-request 2x, then keep only
        # _hr so every rendered panel is an independent record.
        subprocess.run([
            sys.executable, str(GENERATOR_SCRIPT),
            "-i", str(args.ptbxl_data), "-o", str(render_dir),
            "--max_num_images", str(args.count * 2), "-se", str(args.seed),
            "--lead_bbox", "--store_config", "1",
            "--random_bw", "1",  # light-black/gray grid, matching real MAC 400 paper, not the default red
            "--calibration_pulse", "0",  # disable the generator's own native pulse -- we draw our own controlled-size square instead, and the two were merging into one too-tall blob
            # No --wrinkles: user asked for it removed entirely, not just lighter.
        ], check=True, cwd=GENERATOR_DIR)

        all_rows = []
        json_paths = [p for p in sorted(render_dir.glob("*.json")) if "_hr" in p.stem][: args.count]
        for json_path in json_paths:
            png_path = json_path.with_suffix(".png")
            record_id = json_path.stem
            rows = extract_panels(png_path, json_path, record_id, panels_dir)
            all_rows.extend(rows)
            print(f"{record_id}: {len(rows)} panel(s)")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"done, {len(all_rows)} panel(s) from {args.count} record(s), manifest at {manifest_path}")


if __name__ == "__main__":
    main()
