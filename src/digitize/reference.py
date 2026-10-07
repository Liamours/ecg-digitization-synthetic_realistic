"""Reference outputs for the phone port: the pipeline run with the `mobile` settings (ONNX Runtime only) on a fixed list of pages.
Every step of the Kotlin port must reproduce what is kept here.

One folder per page, `<dataset folder>__<page>`:
  input.png                  the page as the pipeline starts from it: EXIF applied, scaled to `work_side`
  page.json                  source, scale, rotation (counterclockwise degrees), panels found (box in upright page pixels, kind) and the page text reads
  panel<i>.png, .front.json  crop of an ECG panel with its text, lead names, grid, gain, trace rows and pulses
  panel<i>.prob.png          trace probability of the crop, 16-bit (value / 65535)
  panel<i>.mask.png          trace mask after the row, column and gap cuts, 0 or 255
  panel<i>.lines.csv         lead separation: one row per track, one column per crop pixel column, empty where the track has no trace
  panel<i>.json, .csv        the panel's leads: time, mV, flag
  record.csv, record.json    the page's 12 leads on one time base and their labels
`manifest.csv` in the output folder lists the pages. Resumable: a page whose record.json exists is skipped.

Usage:
    uv run python -m src.digitize.reference --config configs/digitize.yml --layout mac400_phone --reference configs/mobile_reference.yml
"""
import argparse
import csv
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml
from tqdm import tqdm

from src import paths
from src.digitize import assemble, ocr, pipeline
from src.digitize.pipeline import Digitizer, apply_backend, fit_side, load_image, load_layout

log = logging.getLogger("reference")


def tap(owner, name: str, keep) -> None:
    """Replace owner.name by a wrapper that passes each call's result to `keep` as well."""
    inner = getattr(owner, name)

    def wrapped(*args, **kwargs):
        out = inner(*args, **kwargs)
        keep(args, out)
        return out

    setattr(owner, name, wrapped)


def lines_csv(path: Path, lines: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["" if np.isnan(v) else f"{v:.3f}" for v in row] for row in lines])


def run_page(digitizer: Digitizer, cfg: dict, path: Path, out: Path, seen: dict) -> dict:
    image, work_scale = fit_side(load_image(path), cfg["work_side"])
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "input.png"), image)
    seen["reads"].clear()
    fronts, _, k, panels = digitizer.front_page(image)
    page_reads = seen["reads"][:len(seen["reads"]) - len(fronts)]   # every ECG panel reads its crop once, after the page reads
    records = []
    for f in fronts:
        seen["prob"].clear(), seen["mask"].clear(), seen["lines"].clear()
        f.save(out)
        rec = digitizer.trace(f)
        records.append(rec)
        rec.save(out)
        if seen["prob"]:
            cv2.imwrite(str(out / f"panel{f.index}.prob.png"), np.round(seen["prob"][-1] * 65535).astype(np.uint16))
        if seen["mask"]:
            cv2.imwrite(str(out / f"panel{f.index}.mask.png"), seen["mask"][-1] * 255)
        if seen["lines"]:
            lines_csv(out / f"panel{f.index}.lines.csv", seen["lines"][-1])
    t, signal, lead_labels, image_labels = assemble.page_record(records, cfg["record"], cfg["sampling"]["mm_per_s"])
    (out / "page.json").write_text(json.dumps({"source": str(path), "work_scale": round(work_scale, 4), "page_ocr_side": cfg["page_ocr_side"], "rotation_ccw_deg": k, "panels": panels,
                                               "ocr_reads": page_reads}, indent=1), encoding="utf-8")
    assemble.save(out, t, signal, lead_labels, {"source": str(path), "rotation_ccw_deg": k, **image_labels})   # last: record.json is the done marker
    return {"rotation": k, "panels": len(records), **{key: image_labels[key] for key in ("leads_ok", "leads_low_quality", "leads_flat", "leads_missing")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--reference", type=Path, required=True, help="out_dir and the list of pages")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="onnx")
    ap.add_argument("--device", help="cpu, cuda or auto; overrides the config")
    ap.add_argument("--unet", nargs="*", default=[], metavar="KEY=VALUE", help="override trace network settings after --backend")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ref = yaml.safe_load(args.reference.read_text(encoding="utf-8"))
    free = psutil.virtual_memory().available / 2**30
    if free < ref["min_free_gb"]:
        sys.exit(f"{free:.1f} GB of RAM free, {ref['min_free_gb']} needed: close other programs and run again")
    if args.device:
        cfg["device"] = args.device
    apply_backend(cfg, args.backend, args.unet)
    out_dir = paths.resolve(ref["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"reference-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    digitizer = Digitizer(cfg, load_layout(args.layout))
    seen = {"reads": [], "prob": [], "mask": [], "lines": []}
    tap(ocr, "read_text", lambda a, o: seen["reads"].append({"image_shape": list(a[0].shape[:2]), "texts": o}))
    tap(digitizer.unet, "probability", lambda a, o: seen["prob"].append(o))
    tap(pipeline, "cut_at_gap", lambda a, o: seen["mask"].append(o))
    tap(digitizer.separator, "ecgtizer_lines", lambda a, o: seen["lines"].append(o))
    manifest = out_dir / "manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8"))) if manifest.exists() else []
    done = {r["page"] for r in rows}
    files = [paths.resolve(p) for p in ref["pages"]]
    recent, t_start = deque(maxlen=20), time.time()
    for n, path in enumerate(tqdm(files, desc="reference"), 1):
        key = f"{path.parent.name}__{path.stem}".replace(" ", "_")
        if key in done:
            continue
        t0 = time.time()
        row = {"page": key, "source": str(path), **run_page(digitizer, cfg, path, out_dir / key, seen)}
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        with manifest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
        recent.append(row["seconds"])
        log.info("%s: %d/%d, %.1f s, %d panels, %d leads ok, ETA %s", key, n, len(files), row["seconds"], row["panels"], row["leads_ok"],
                 timedelta(seconds=int(sum(recent) / len(recent) * (len(files) - n))))
    log.info("done in %s", timedelta(seconds=int(time.time() - t_start)))


if __name__ == "__main__":
    main()
