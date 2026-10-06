"""Page to panels: docling text, orientation, panel rectangles, ECG vs report typing.

Usage:
    uv run python -m src.run_pages --config configs/pages.yml <image-or-folder> ... [--limit N]
"""
import argparse
import csv
import json
import random
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.orient import choose_rotation, rotate_box
from src import paths
from src.panels import find_panels
from src.text_read import load_upright, read_texts

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLORS = {"ecg": "red", "report": "green", "other": "orange"}


def collect(inputs: list[Path], limit: int | None, seed: int) -> list[Path]:
    files = []
    for p in inputs:
        found = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
        files += random.Random(seed).sample(found, min(limit, len(found))) if limit and p.is_dir() else found
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("inputs", type=paths.resolve, nargs="+")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = paths.resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    summary = out / "summary.csv"
    if not summary.exists():
        summary.write_text("image,rotation_ccw,panels,ecg,report,other\n", encoding="utf-8")
    for path in tqdm(collect(args.inputs, args.limit, cfg["seed"]), desc="pages"):
        stem = f"{path.parent.name}__{path.stem}".replace(" ", "_").replace(",", "")
        if (out / f"{stem}.json").exists():
            continue
        img = load_upright(path, cfg["max_side"])
        texts = read_texts(converter, img)
        k, scores = choose_rotation(texts, img.width, img.height, cfg)
        up = img
        for _ in range(k // 90):
            up = up.transpose(Image.ROTATE_90)
        rotated = [(t["text"], rotate_box(t["bbox"], img.width, img.height, k)) for t in texts]
        panels = find_panels(rotated, up.width, up.height, cfg)
        draw = ImageDraw.Draw(up)
        for p in panels:
            draw.rectangle(p["box"], outline="orange" if p["isolated"] else COLORS[p["kind"]], width=6)
        up.thumbnail((2000, 2000))
        up.save(out / f"{stem}.png")
        record = {"source": str(path), "rotation_ccw": k, "direction_scores": scores, "panels": panels}
        (out / f"{stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        kinds = [p["kind"] for p in panels]
        with summary.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([stem, k, len(panels), kinds.count("ecg"), kinds.count("report"), kinds.count("other")])


if __name__ == "__main__":
    main()
