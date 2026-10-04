"""Digitize ECG pages with a layout.

Each page writes <out-dir>/<page stem>/: one JSON and CSV per panel, page.json (rotation, panel confidences), record.csv and record.json (the 12 leads and their labels),
overlay_original.jpg and overlay_upright.jpg (the results drawn back on the page). <out-dir>/progress.csv gets one row per page (seconds, panels, leads ok, status, error), the log file one line per page with an ETA from the last 20 pages. Resumable: a page whose page.json
exists is skipped, so a stopped run continues where it stopped.

Usage:
    uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> [--list paths.txt] <image-or-folder> ...
"""
import argparse
import csv
import json
import logging
import random
import shutil
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.digitize import assemble, report, reverse
from src.digitize.pipeline import Digitizer, fit_side, load_image, load_layout

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
log = logging.getLogger("digitize")


def collect(inputs: list[Path]) -> list[Path]:
    files = []
    for p in inputs:
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
    return files


def save_overlay(path: Path, img: np.ndarray, max_side: int) -> None:
    """Downscaled JPEG: a full-size PNG overlay is about 13 MB, which fills the disk over a whole dataset."""
    s = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True, help="layout name in configs/layouts or a path to a layout file")
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    ap.add_argument("--list", type=paths.resolve, help="text file with one image path per line")
    ap.add_argument("--sample", type=int, help="digitize N pages drawn from the inputs, the same N for the same seed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=[], help="skip paths containing any of these strings")
    ap.add_argument("--report", action="store_true", help="write report.png (page with boxes, plus every lead) in each page folder")
    ap.add_argument("--final-only", action="store_true", help="keep per page only the final labels (record.csv, record.json) and report.png (the original page with panel boxes, text boxes and traces, plus every lead); no per-panel files. The done marker is record.json")
    ap.add_argument("--device", help="cpu, cuda or auto; overrides the config")
    ap.add_argument("--manifest", type=paths.resolve, help="a dataset's _labels/manifest.csv; with it only pages whose page_style equals --style are run")
    ap.add_argument("--style", default="mac400", help="page_style to keep when --manifest is given")
    ap.add_argument("inputs", type=paths.resolve, nargs="*", help="files or folders; @mac400-scan/scan_29 style aliases work, see configs/paths.yml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    layout = load_layout(args.layout)
    marker = "record.json" if args.final_only else "page.json"
    paths.resolve(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=paths.resolve(cfg["log_dir"]) / f"digitize-{layout['name']}-{datetime.now():%Y%m%d}.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", encoding="utf-8")
    listed = [paths.resolve(x.strip()) for x in args.list.read_text(encoding="utf-8").splitlines() if x.strip()] if args.list else []
    pool = [f for f in collect(args.inputs) + listed if not any(x in str(f) for x in args.exclude)]
    if args.manifest:  # keep only the pages the dataset's manifest gives this page style
        with args.manifest.open(encoding="utf-8") as fh:
            keep = {(args.manifest.parent.parent / r["relative_path"]).resolve() for r in csv.DictReader(fh) if r["page_style"] == args.style}
        pool = [f for f in pool if f.resolve() in keep]
    if args.sample:
        pool = random.Random(args.seed).sample(pool, min(args.sample, len(pool)))
    files = [f for f in pool if not (args.out_dir / f"{f.parent.name}__{f.stem}".replace(" ", "_") / marker).exists()]
    done_before = len(pool) - len(files)
    log.info("%s: %d pages pending, %d already done (resuming)", layout["name"], len(files), done_before)
    progress = args.out_dir / "progress.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not progress.exists():
        progress.write_text("page,seconds,panels,leads_ok,leads,status,error\n", encoding="utf-8")
    digitizer = Digitizer(cfg, layout)
    recent, failed, t_start = deque(maxlen=20), 0, time.time()
    for n, path in enumerate(tqdm(files, desc=layout["name"]), 1):
        out = args.out_dir / f"{path.parent.name}__{path.stem}".replace(" ", "_")
        t0 = time.time()
        status, error, n_panels, ok, n_leads = "ok", "", 0, 0, 0
        try:
            image, work_scale = fit_side(load_image(path), cfg["work_side"])
            out.mkdir(parents=True, exist_ok=True)  # a page with no panel still gets its page.json
            if layout["mode"] == "panel":
                records, up, k = digitizer.run_panel_page(image)
            else:
                records, up, k = digitizer.run_fixed_page(image), image, 0
            work = out / "_work" if args.final_only else out  # final-only: the per-panel files live only until the picture is drawn
            work.mkdir(parents=True, exist_ok=True)
            for r in records:
                r.save(work)
            t_rec, signal, lead_labels, image_labels = assemble.page_record(records, cfg["record"], cfg["sampling"]["mm_per_s"])
            if not args.final_only:
                assemble.save(out, t_rec, signal, lead_labels, {"source": str(path), "rotation_ccw_deg": k, **image_labels})
            overlay = reverse.draw(up, records, cfg["sampling"])
            save_overlay(work / "overlay_upright.jpg", overlay, cfg["overlay_max_side"])
            if k:
                save_overlay(work / "overlay_original.jpg", np.rot90(overlay, -(k // 90)).copy(), cfg["overlay_max_side"])
            n_panels, ok, n_leads = len(records), sum(v.flag == "ok" for r in records for v in r.leads.values()), sum(len(r.leads) for r in records)
            # page.json last: it is the done marker, so a crash before it repeats one page and never loses one silently
            (work / "page.json").write_text(json.dumps({"source": str(path), "layout": layout["name"], "rotation_ccw_deg": k, "work_scale": round(work_scale, 4),
                                                        "panels": {r.panel_id: {"confidence": round(r.confidence, 3), "error": r.error, "leads_ok": sum(v.flag == "ok" for v in r.leads.values()), "leads": len(r.leads)} for r in records}}, indent=1), encoding="utf-8")
            if args.report or args.final_only:
                report.make(work)
            if args.final_only:
                shutil.move(str(work / "report.png"), str(out / "report.png"))
                shutil.rmtree(work)
                assemble.save(out, t_rec, signal, lead_labels, {"source": str(path), "rotation_ccw_deg": k, **image_labels})  # last: record.json is the done marker
        except Exception as exc:
            log.exception("%s failed: %r", path, exc)
            status, error, failed = "error", repr(exc)[:120].replace(",", ";"), failed + 1
        sec = time.time() - t0
        recent.append(sec)
        with progress.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([path.name, round(sec, 1), n_panels, ok, n_leads, status, error])
        eta = timedelta(seconds=int(sum(recent) / len(recent) * (len(files) - n)))
        log.info("%s: %d/%d, %.1f s, %d panels, %d/%d leads ok, ETA %s (mean of the last %d pages)", path.name, n, len(files), sec, n_panels, ok, n_leads, eta, len(recent))
    log.info("%s: done, %d pages in %s, %d failed", layout["name"], len(files), timedelta(seconds=int(time.time() - t_start)), failed)


if __name__ == "__main__":
    main()
