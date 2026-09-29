"""Label each page's style (edan, mac400, kardia, form) from docling-read text.

Usage:
    uv run python -m src.page_style --config configs/page_style.yml --dataset ekg-esta [--limit N]
"""
import argparse
import csv
import random
import re
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter
from tqdm import tqdm

from src.text_read import load_upright, read_texts

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FIELDS = ["dataset", "relative_path", "page_style", "matched"]


def classify(text: str, styles: dict[str, str], unknown: str) -> tuple[str, list[str]]:
    hits = {name: re.findall(rx, text, re.I) for name, rx in styles.items()}
    hits = {k: v for k, v in hits.items() if v}
    return (next(iter(hits)) if hits else unknown), sorted({m.strip() for v in hits.values() for m in v})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["datasets_dir"]) / args.dataset
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        files = random.Random(cfg["seed"]).sample(files, min(args.limit, len(files)))
    out = Path(cfg["out_dir"]) / f"{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            done = {r["relative_path"] for r in csv.DictReader(fh)}
    else:
        out.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
    converter = DocumentConverter()
    for p in tqdm([f for f in files if f.relative_to(root).as_posix() not in done], desc=args.dataset):
        rel = p.relative_to(root).as_posix()
        text = "\n".join(t["text"] for t in read_texts(converter, load_upright(p, cfg["max_side"])))
        style, matched = classify(text, cfg["styles"], cfg["unknown_label"])
        with out.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([args.dataset, rel, style, "|".join(matched)])


if __name__ == "__main__":
    main()
