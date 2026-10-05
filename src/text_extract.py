"""Docling text of every source file of the real datasets: one JSON (full document) and one Markdown file per source file.

Output sits inside each dataset, next to `_labels/`: `<dataset>/_text/<relative path with .json and .md>`. Images are
turned upright first when a canonical digitize run knows the page's rotation (`rotation_runs` in the config); docling
then reads the upright page, since its OCR reads sideways text badly. PDFs are read as they are. Folders whose name
starts with `_`, and the top-level folders listed under `exclude`, hold derived files and are skipped.

Resumable: a file whose JSON exists is skipped, and the JSON is written last. Progress and ETA per file go to a log.

Usage:
    python -m src.text_extract --config configs/text_extract.yml [--dataset esta] [--sample N]
"""
import argparse
import json
import logging
import random
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from src import paths

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger("text_extract")


def source_files(root: Path, suffixes: list[str], exclude: list[str]) -> list[Path]:
    rel = lambda p: p.relative_to(root).parts
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes and rel(p)[0] not in exclude and not any(part.startswith("_") for part in rel(p)))


def rotations(run: Path) -> dict[str, int]:
    """Counterclockwise rotation that made each page upright, keyed by the run's page folder name (`<parent>__<stem>`)."""
    return {rj.parent.name: json.loads(rj.read_text(encoding="utf-8"))["image"]["rotation_ccw_deg"] for rj in run.glob("*/record.json")}


def converter(device: str):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions(backend="torch"), accelerator_options=AcceleratorOptions(device=device))
    return DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts), InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", nargs="*", help="dataset aliases (configs/paths.yml); default: every dataset in the config")
    ap.add_argument("--sample", type=int, help="N files drawn from all inputs, the same N for the same seed")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"text_extract-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    logging.getLogger("docling").setLevel(logging.WARNING)
    jobs = []   # (dataset root, file, rotation)
    for alias in args.dataset or cfg["datasets"]:
        root = paths.dataset(alias)
        rot = {}
        for run in cfg["rotation_runs"].get(alias, []):
            rot.update(rotations(paths.resolve(run)))
        jobs += [(root, p, rot.get(f"{p.parent.name}__{p.stem}".replace(" ", "_"), 0)) for p in source_files(root, cfg["suffixes"], cfg["exclude"].get(alias, []))]
    if args.sample:
        jobs = random.Random(cfg["seed"]).sample(jobs, min(args.sample, len(jobs)))
    pending = [j for j in jobs if not (j[0] / "_text" / j[1].relative_to(j[0])).with_suffix(".json").exists()]
    log.info("%d files pending, %d already done", len(pending), len(jobs) - len(pending))
    conv, recent, failed, t_start = converter(cfg["device"]), deque(maxlen=20), 0, time.time()
    for n, (root, path, k) in enumerate(tqdm(pending, desc="docling"), 1):
        out = (root / "_text" / path.relative_to(root)).with_suffix(".json")
        t0 = time.time()
        try:
            if path.suffix.lower() == ".pdf":
                doc = conv.convert(str(path)).document
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    page = Path(tmp) / "page.png"
                    ImageOps.exif_transpose(Image.open(path).convert("RGB")).rotate(k, expand=True).save(page)   # loaded as src.digitize.pipeline.load_image does, so `k` means the same
                    doc = conv.convert(str(page)).document
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".md").write_text(doc.export_to_markdown(traverse_pictures=True), encoding="utf-8")   # an ECG page is one picture to docling; its text sits inside it
            data = doc.export_to_dict()
            data["source_rotation_ccw_deg"] = k   # the page docling read is the file turned by this much
            out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")   # last: the JSON is the done marker
        except Exception as exc:
            failed += 1
            log.exception("%s failed: %r", path, exc)
        recent.append(time.time() - t0)
        eta = timedelta(seconds=int(sum(recent) / len(recent) * (len(pending) - n)))
        log.info("%s/%s: %d/%d, %.1f s, rotation %d, ETA %s", root.name, path.relative_to(root).as_posix(), n, len(pending), recent[-1], k, eta)
    log.info("done, %d files in %s, %d failed", len(pending), timedelta(seconds=int(time.time() - t_start)), failed)


if __name__ == "__main__":
    main()
