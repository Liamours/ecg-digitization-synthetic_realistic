"""One manifest per real dataset, every dataset with the same columns: `<dataset>/_labels/manifest.csv`.

Every source file on disk gets one row (the same files src.text_extract reads: folders starting with `_` and the
`exclude` folders are left out). Shared columns, in order:

- page_id: kept from the old manifest when the file had a row there, else the relative path without its suffix, `/` as `__`
- relative_path, file_type, md5, pages (1 for an image, the page count for a PDF), width_px, height_px, dpi (images only)
- is_duplicate, duplicate_of: files with the same md5 are copies of the one with the shortest path, whose page_id they name
- page_style, style_evidence: the device or form style found in the file's docling text (`<dataset>/_text/`, src.text_extract)
  by the patterns of configs/page_style.yml, and the matched words; `-` when nothing matches, `blank` when also almost no ink
- ink_fraction (src.ink_fraction, images only), has_ecg: a device style, or ink at least `ecg_ink_fraction`

Columns the old manifest had beyond these (diagnosis labels, document and page numbers) follow, matched by relative_path.
The old manifest is copied to `_labels/_backups/manifest_<YYYYMMDD-HHMM>.csv` first. Empty cells mean the value does not apply.

Usage:
    python -m src.manifest --config configs/manifests.yml [--dataset esta]
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src import paths
from src.ink_fraction import ink_fraction
from src.text_extract import source_files

Image.MAX_IMAGE_PIXELS = None
SHARED = ["page_id", "relative_path", "file_type", "md5", "pages", "width_px", "height_px", "dpi", "is_duplicate", "duplicate_of",
          "page_style", "style_evidence", "ink_fraction", "has_ecg"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def style(text_json: Path, styles: dict[str, str], unknown: str) -> tuple[str, str, int | None]:
    """Style, matched words and page count from the file's docling JSON; the style is unknown when there is no JSON."""
    if not text_json.exists():
        return unknown, "", None
    doc = json.loads(text_json.read_text(encoding="utf-8"))
    text = "\n".join(t["text"] for t in doc["texts"])
    hits = {name: sorted({m.strip() for m in re.findall(rx, text, re.I)}) for name, rx in styles.items()}
    hits = {k: v for k, v in hits.items() if v}
    return (next(iter(hits)) if hits else unknown), "|".join(w for v in hits.values() for w in v), len(doc["pages"])


def build(alias: str, cfg: dict, styles: dict[str, str], unknown: str, stamp: str) -> list[dict]:
    root = paths.dataset(alias)
    path = root / "_labels" / "manifest.csv"
    old = {r["relative_path"]: r for r in read_csv(path) if r.get("relative_path")}
    extra = [c for c in (next(iter(old.values())) if old else {}) if c not in SHARED + ["original_filename"]]
    rows = []
    for p in tqdm(source_files(root, cfg["suffixes"], cfg["exclude"].get(alias, [])), desc=root.name):
        rel = p.relative_to(root).as_posix()
        prev = old.get(rel, {})
        page_id = prev.get("page_id") or rel.rsplit(".", 1)[0].replace("/", "__")
        digest = md5(p)
        page_style, evidence, pages = style((root / "_text" / rel).with_suffix(".json"), styles, unknown)
        width = height = dpi = ink = ""
        if p.suffix.lower() != ".pdf":
            with Image.open(p) as im:
                width, height = im.size
                dpi = round(im.info["dpi"][0]) if "dpi" in im.info else ""
            ink = round(ink_fraction(p), 3)
            pages = 1
            if page_style == unknown and ink < cfg["blank_ink_fraction"]:
                page_style = "blank"
        has_ecg = page_style in cfg["device_styles"] or (ink != "" and ink >= cfg["ecg_ink_fraction"])
        rows.append({"page_id": page_id, "relative_path": rel, "file_type": p.suffix.lower().lstrip("."), "md5": digest, "pages": pages or "",
                     "width_px": width, "height_px": height, "dpi": dpi, "is_duplicate": False, "duplicate_of": "",
                     "page_style": page_style, "style_evidence": evidence, "ink_fraction": ink, "has_ecg": has_ecg, **{c: prev.get(c, "") for c in extra}})
    original = {}   # per md5 the shortest path: a copy is saved as `name(1).jpg` or `name (2).jpg` next to `name.jpg`
    for r in sorted(rows, key=lambda r: (len(r["relative_path"]), r["relative_path"])):
        original.setdefault(r["md5"], r["page_id"])
    for r in rows:
        r["is_duplicate"], r["duplicate_of"] = (True, original[r["md5"]]) if original[r["md5"]] != r["page_id"] else (False, "")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        (path.parent / "_backups").mkdir(exist_ok=True)
        shutil.copy2(path, path.parent / "_backups" / f"manifest_{stamp}.csv")
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHARED + extra)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", nargs="*", help="dataset aliases (configs/paths.yml); default: every dataset in the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    load = lambda key: yaml.safe_load(paths.resolve(cfg[key]).read_text(encoding="utf-8"))
    style_cfg = load("style_config")
    cfg.update({k: v for k, v in load("text_config").items() if k in ("datasets", "suffixes", "exclude")})
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    for alias in args.dataset or cfg["datasets"]:
        rows = build(alias, cfg, style_cfg["styles"], style_cfg["unknown_label"], stamp)
        counts = {}
        for r in rows:
            counts[r["page_style"]] = counts.get(r["page_style"], 0) + 1
        print(f"{alias}: {len(rows)} rows, {sum(r['is_duplicate'] for r in rows)} duplicates, styles {counts}")


if __name__ == "__main__":
    main()
