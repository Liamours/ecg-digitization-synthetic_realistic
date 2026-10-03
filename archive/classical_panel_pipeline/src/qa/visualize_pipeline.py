"""End-to-end pipeline visualization: one raw composite photo in, one image
out, showing every stage the pipeline actually goes through -- composite,
composite with the SAM segmentation boxes drawn on it, every sheet found
(before and after the orientation fix), every column found on every sheet,
and every row found in every column (all of them, not one representative
example -- 4 sheets with 3 rows each means 12 row crops shown, not 1).

Orientation-fixing happens once per sheet, before any column/row
partitioning -- there is no separate per-row fix step in the real pipeline,
so this tool doesn't invent one; "row" crops are already in the corrected
frame by construction.

Orchestrates two venvs via subprocess (SAM and paper-ecg need conflicting
dependency versions, same reason run_full_pipeline.py does this), then
composes the results into one image using only this process's own PIL.

Usage:
    python src/qa/visualize_pipeline.py --input <raw_photo> --output <out.png>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
VENVS = REPO_ROOT / ".venvs"
SAM_PY = VENVS / "sam" / "Scripts" / "python.exe"
PAPERECG_PY = VENVS / "paper-ecg" / "Scripts" / "python.exe"
STAGE_A = Path(__file__).resolve().parent / "visualize_stage_a.py"
STAGE_B = Path(__file__).resolve().parent / "visualize_stage_b.py"

BG = (237, 239, 234)
INK = (32, 36, 31)
DIM = (91, 95, 86)
LINE = (211, 214, 205)
DISCARD_RED = (191, 42, 42)
DISCARD_LABEL_BG = (255, 226, 221)
THUMB_H = 220
GUTTER = 14
SECTION_GAP = 34
MARGIN = 28
BORDER_W = 4


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(22, bold=True)
FONT_LABEL = load_font(15)


def fit_thumb(path: Path, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = max(1, round(w * height / h))
    return img.resize((new_w, height), Image.LANCZOS)


def section(canvas: Image.Image, draw: ImageDraw.ImageDraw, y: int, title: str, items: list[tuple[Path, str, bool]], thumb_h: int, max_width: int) -> int:
    """items: (path, label, discarded). A discarded item gets a thick red
    border and its label drawn on a light-red chip, so a reader can tell at
    a glance what the pipeline found but then threw out, without having to
    read every label."""
    n_discarded = sum(1 for _, _, discarded in items if discarded)
    title_suffix = f"  --  {n_discarded} discarded (red)" if n_discarded else ""
    draw.text((MARGIN, y), title + title_suffix, font=FONT_TITLE, fill=INK)
    y += 34
    if not items:
        draw.text((MARGIN, y), "(none found)", font=FONT_LABEL, fill=DIM)
        return y + thumb_h // 2 + SECTION_GAP

    x = MARGIN
    row_h = 0
    for path, label, discarded in items:
        thumb = fit_thumb(path, thumb_h)
        if x + thumb.width > max_width - MARGIN and x > MARGIN:
            y += row_h + 28
            x = MARGIN
            row_h = 0
        canvas.paste(thumb, (x, y))
        if discarded:
            draw.rectangle([x - BORDER_W, y - BORDER_W, x + thumb.width + BORDER_W, y + thumb.height + BORDER_W], outline=DISCARD_RED, width=BORDER_W)
            text_w = draw.textlength(label, font=FONT_LABEL)
            draw.rectangle([x, y + thumb.height + 2, x + text_w + 6, y + thumb.height + 22], fill=DISCARD_LABEL_BG)
            draw.text((x + 3, y + thumb.height + 4), label, font=FONT_LABEL, fill=DISCARD_RED)
        else:
            draw.rectangle([x, y, x + thumb.width, y + thumb.height], outline=LINE, width=1)
            draw.text((x, y + thumb.height + 4), label, font=FONT_LABEL, fill=DIM)
        x += thumb.width + GUTTER
        row_h = max(row_h, thumb.height)
    y += row_h + 28
    return y + SECTION_GAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize every stage of the crop pipeline for one raw composite photo.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None, help="Kept after the run if given; otherwise a temp dir is used and discarded")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="visualize_pipeline_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(SAM_PY), str(STAGE_A), "--input", str(args.input), "--output-dir", str(work_dir)], check=True)
    subprocess.run([str(PAPERECG_PY), str(STAGE_B), "--output-dir", str(work_dir)], check=True)

    stage_b = json.loads((work_dir / "stage_b_manifest.json").read_text(encoding="utf-8"))

    max_width = 1600
    canvas = Image.new("RGB", (max_width, 8000), BG)  # generous placeholder, cropped to actual content height below
    draw = ImageDraw.Draw(canvas)
    y = MARGIN

    def short(reason: str | None, limit: int = 34) -> str:
        if not reason:
            return ""
        return reason if len(reason) <= limit else reason[: limit - 1] + "…"

    y = section(canvas, draw, y, "composite", [(work_dir / "00_composite.png", "raw", False)], 420, max_width)
    y = section(canvas, draw, y, "composite + sam segmentation",
                [(work_dir / "01_composite_bbox.png", f"{len(stage_b)} sheet(s) found, red = discarded duplicate", False)], 420, max_width)
    y = section(canvas, draw, y, "sheet (raw)", [(work_dir / s["sheet"], s["sheet"], False) for s in stage_b], THUMB_H, max_width)

    sheet_items = []
    for s in stage_b:
        usable_cols = [c for c in s.get("columns", []) if not c["discarded"]]
        if s.get("reason"):
            sheet_items.append((work_dir / s["fixed"], f"{s['sheet']}: {s['reason']}", True))
        elif not usable_cols:
            sheet_items.append((work_dir / s["fixed"], f"{s['sheet']}: no usable column found", True))
        else:
            sheet_items.append((work_dir / s["fixed"], f"{s['sheet']}, rot {s['rotation_deg']} deg", False))
    y = section(canvas, draw, y, "sheet (fixed)", sheet_items, THUMB_H, max_width)

    column_items = []
    row_items = []
    for s in stage_b:
        for c in s.get("columns", []):
            col_label = c["column"] if not c["discarded"] else f"{c['column']}: {short(c['reason'])}"
            column_items.append((work_dir / c["column"], col_label, c["discarded"]))
            for r in c["rows"]:
                row_label = r["lead_name"] or "?"
                if r["discarded"]:
                    row_label += f": {short(r['reason'])}"
                row_items.append((work_dir / r["row"], row_label, r["discarded"]))

    n_col_usable = sum(1 for _, _, d in column_items if not d)
    n_row_usable = sum(1 for _, _, d in row_items if not d)
    y = section(canvas, draw, y, f"column ({n_col_usable}/{len(column_items)} usable)", column_items, THUMB_H, max_width)
    y = section(canvas, draw, y, f"row ({n_row_usable}/{len(row_items)} usable)", row_items, 140, max_width)

    final = Image.new("RGB", (max_width, y + MARGIN), BG)
    final.paste(canvas.crop((0, 0, max_width, y + MARGIN)), (0, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.output)
    print(f"wrote {args.output} ({final.width}x{final.height})")

    if args.work_dir is None:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
