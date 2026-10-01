"""Label each page's style (edan, mac400, kardia, form) from OCR text.

Every page is read once with RapidOCR (the engine docling uses) in parallel
worker processes. Resumable: pages already in the dataset's CSV are skipped
and each finished page is appended at once. Progress and ETA go to the console
and to a log file.

Usage:
    uv run python -m src.page_style --config configs/page_style.yml --dataset ekg-esta [--limit N] [--out-dir DIR]
"""
import argparse
import csv
import logging
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.image_io import load_upright

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FIELDS = ["dataset", "relative_path", "page_style", "matched"]

log = logging.getLogger("page_style")
_reader = None


def classify(text: str, styles: dict[str, str], unknown: str) -> tuple[str, list[str]]:
    hits = {name: re.findall(rx, text, re.I) for name, rx in styles.items()}
    hits = {k: v for k, v in hits.items() if v}
    return (next(iter(hits)) if hits else unknown), sorted({m.strip() for v in hits.values() for m in v})


def _init_worker(threads: int) -> None:
    global _reader
    import torch
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

    torch.set_num_threads(threads)
    _reader = RapidOcrModel(
        enabled=True, artifacts_path=None, options=RapidOcrOptions(backend="torch"),
        accelerator_options=AcceleratorOptions(num_threads=threads),
    ).reader


def _label_page(path: str, max_side: int, retry_max_side: int, styles: dict[str, str], unknown: str) -> tuple[str, list[str]]:
    """Reads at `max_side`; an unmatched page is read again at `retry_max_side`,
    since small print (a device name) can vanish at low resolution."""
    for side in (max_side, retry_max_side):
        out = _reader(np.asarray(load_upright(Path(path), side)))
        style, matched = classify("\n".join(out.txts) if out.txts else "", styles, unknown)
        if style != unknown:
            break
    return style, matched


def read_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8") as fh:
        return {r["relative_path"] for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", required=True, help="dataset alias or folder name, see configs/paths.yml")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", type=paths.resolve, help="overrides out_dir in the config, e.g. for a validation run")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = paths.dataset(args.dataset)
    out_dir = args.out_dir or paths.resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"page_style-{args.dataset}-{datetime.now():%Y%m%d}.log", level=logging.INFO,
        format="%(asctime)s %(message)s", encoding="utf-8",
    )

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        files = random.Random(cfg["seed"]).sample(files, min(args.limit, len(files)))
    out = out_dir / f"{args.dataset}.csv"
    done = read_done(out)
    pending = [p for p in files if p.relative_to(root).as_posix() not in done]
    log.info("%s: %d images, %d already labeled, %d pending, %d workers x %d threads, max_side %d, retry at %d",
             args.dataset, len(files), len(files) - len(pending), len(pending), cfg["workers"], cfg["threads_per_worker"], cfg["max_side"], cfg["retry_max_side"])
    if not pending:
        return
    if not out.exists():
        out.write_text(",".join(FIELDS) + "\n", encoding="utf-8")

    counts, errors = Counter(), 0
    t0 = last = time.time()
    with ProcessPoolExecutor(cfg["workers"], initializer=_init_worker, initargs=(cfg["threads_per_worker"],)) as ex, \
            out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        futures = {ex.submit(_label_page, str(p), cfg["max_side"], cfg["retry_max_side"], cfg["styles"], cfg["unknown_label"]): p for p in pending}
        for n, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=args.dataset), 1):
            p = futures[fut]
            try:
                style, matched = fut.result()
            except Exception as exc:
                errors += 1
                log.error("%s failed: %r", p, exc)
                continue
            writer.writerow([args.dataset, p.relative_to(root).as_posix(), style, "|".join(matched)])
            fh.flush()
            counts[style] += 1
            if time.time() - last >= cfg["log_every_s"]:
                rate = n / (time.time() - t0)
                log.info("%s: %d/%d, %.1f pages/min, ETA %s", args.dataset, n, len(futures), rate * 60, timedelta(seconds=int((len(futures) - n) / rate)))
                last = time.time()
    log.info("%s finished in %s: %s, %d errors", args.dataset, timedelta(seconds=int(time.time() - t0)), dict(counts), errors)


if __name__ == "__main__":
    main()
