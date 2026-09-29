"""Add page_style, style_evidence, ink_fraction, and has_ecg to the real-photo
dataset manifests. page_style and style_evidence (the matched text) come from
the page style CSVs (src.page_style output); a page that matched nothing and
has an ink fraction below `blank_ink_fraction` is `blank`. ink_fraction
(src.ink_fraction) is the share of paper pixels darker than their surroundings;
has_ecg is True when the page matches a device style or its ink fraction
reaches `ecg_ink_fraction`, which catches ECG strips pasted on a form and ECG
from unlisted devices.

`ekg-realistic_pages_original`: two columns appended to its manifest.
`ekg-esta`: manifest created (it had none), one row per image and per PDF.
`ekg-realistic_pages_retake`: manifest rebuilt to cover every image on disk;
its old rows keep their page_id, original_filename, md5 and duplicate info,
and get their relative_path corrected to the subfolder the file now sits in.

Existing manifests are copied to `_labels/_backups/manifest_<YYYYMMDD-HHMM>.csv`
first. Usage:
    uv run python -m src.build_manifests --config configs/manifests.yml
"""
import argparse
import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.ink_fraction import ink_fraction

Image.MAX_IMAGE_PIXELS = None
NEW_COLUMNS = ["page_style", "style_evidence", "ink_fraction", "has_ecg"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def backup(path: Path, stamp: str) -> None:
    if path.exists():
        dest = path.parent / "_backups" / f"manifest_{stamp}.csv"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)


def style_columns(entry: tuple[str, str] | None, ink: float | None, cfg: dict) -> dict:
    if entry is None or ink is None:
        return {"page_style": "", "style_evidence": "", "ink_fraction": "", "has_ecg": ""}
    style, evidence = entry
    if style == "-" and ink < cfg["blank_ink_fraction"]:
        style = "blank"
    has_ecg = style in cfg["device_styles"] or ink >= cfg["ecg_ink_fraction"]
    return {"page_style": style, "style_evidence": evidence, "ink_fraction": round(ink, 3), "has_ecg": str(has_ecg)}


def load_styles(cfg: dict, dataset: str) -> dict[str, tuple[str, str]]:
    return {r["relative_path"]: (r["page_style"], r["matched"]) for r in read_csv(Path(cfg["page_style_dir"]) / f"{dataset}.csv")}


def image_files(root: Path, cfg: dict) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in cfg["image_suffixes"] and "_labels" not in p.relative_to(root).parts)


def build_original(cfg: dict, stamp: str) -> None:
    root = Path(cfg["datasets_dir"]) / "ekg-realistic_pages_original"
    path = root / "_labels" / "manifest.csv"
    rows = read_csv(path)
    styles = load_styles(cfg, root.name)
    fields = [f for f in rows[0] if f not in NEW_COLUMNS] + NEW_COLUMNS
    backup(path, stamp)
    out = [{**r, **style_columns(styles.get(r["relative_path"]), ink_fraction(root / r["relative_path"]), cfg)} for r in tqdm(rows, desc=root.name)]
    write_csv(path, fields, out)


def build_esta(cfg: dict, stamp: str) -> None:
    root = Path(cfg["datasets_dir"]) / "ekg-esta"
    styles = load_styles(cfg, root.name)
    files = sorted(p for p in root.rglob("*") if p.is_file() and "_labels" not in p.relative_to(root).parts)
    rows = []
    for p in tqdm(files, desc="ekg-esta"):
        rel = p.relative_to(root).as_posix()
        is_image = p.suffix.lower() in cfg["image_suffixes"]
        width = height = ""
        if is_image:
            with Image.open(p) as im:
                width, height = im.size
        rows.append({
            "page_id": rel.rsplit(".", 1)[0].replace("/", "__"), "relative_path": rel, "file_type": p.suffix.lower().lstrip("."),
            "md5": md5(p), "width_px": width, "height_px": height,
            **style_columns(styles.get(rel) if is_image else None, ink_fraction(p) if is_image else None, cfg),
        })
    fields = ["page_id", "relative_path", "file_type", "md5", "width_px", "height_px"] + NEW_COLUMNS
    out = root / "_labels" / "manifest.csv"
    out.parent.mkdir(exist_ok=True)
    backup(out, stamp)
    write_csv(out, fields, rows)


def build_retake(cfg: dict, stamp: str) -> None:
    root = Path(cfg["datasets_dir"]) / "ekg-realistic_pages_retake"
    path = root / "_labels" / "manifest.csv"
    old = read_csv(path)
    styles = load_styles(cfg, root.name)
    files = image_files(root, cfg)
    by_md5 = {}
    for p in tqdm(files, desc="ekg-realistic_pages_retake"):
        by_md5[md5(p)] = p
    rows, used = [], set()
    for r in old:
        p = by_md5.get(r["md5"]) if r["relative_path"] else None
        rel = p.relative_to(root).as_posix() if p else ""
        if p:
            used.add(rel)
        rows.append({**r, "relative_path": rel, **style_columns(styles.get(rel) if rel else None, ink_fraction(root / rel) if rel else None, cfg)})
    next_id = 1 + max(int(r["page_id"].split("_")[1]) for r in old if "dup" not in r["page_id"])
    for md5_, p in by_md5.items():
        rel = p.relative_to(root).as_posix()
        if rel in used:
            continue
        with Image.open(p) as im:
            width, height = im.size
            dpi = round(im.info["dpi"][0]) if "dpi" in im.info else ""
        rows.append({
            "page_id": f"retake_{next_id:04d}", "relative_path": rel, "original_filename": p.name, "md5": md5_,
            "is_duplicate": "False", "duplicate_of": "", "width_px": width, "height_px": height, "dpi": dpi,
            **style_columns(styles.get(rel), ink_fraction(p), cfg),
        })
        next_id += 1
    fields = [f for f in old[0] if f not in NEW_COLUMNS] + NEW_COLUMNS
    backup(path, stamp)
    write_csv(path, fields, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    build_original(cfg, stamp)
    build_esta(cfg, stamp)
    build_retake(cfg, stamp)


if __name__ == "__main__":
    main()
