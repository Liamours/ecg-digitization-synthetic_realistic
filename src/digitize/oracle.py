"""Headroom per step: run the digitizer on a synthetic dataset with some steps replaced by the ground truth.

Each mode keeps the earlier ones: `panels` uses the true page rotation and the true panel boxes (no OCR page read, no footer
anchors), `gain` adds the true gain, `labels` adds the true lead names. Scoring the run with src.digitize.score and
comparing it with a normal run shows how much end to end score each step is holding back.

Usage:
    uv run python -m src.digitize.oracle --config configs/digitize.yml --dataset @synthetic-260930-1856 --mode gain --out-dir @inferences/<run>
"""
import argparse
import csv
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.digitize.pipeline import Digitizer, fit_side, load_image, load_layout
from src.orient import rotate_box

MODES = ("panels", "gain", "labels")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", default="mac400")
    ap.add_argument("--dataset", type=paths.resolve, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    ap.add_argument("--min-visible", type=float, default=0.5, help="a panel with less of its box on the page is skipped, as in src.digitize.score")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    layout = load_layout(args.layout)
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"oracle-{args.mode}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("oracle")
    lab = args.dataset / "labels"
    pages = {r["page_id"]: r for r in read_rows(lab / "manifest_pages.csv")}
    panel_meta = {r["panel_id"]: r for r in read_rows(lab / "manifest_panels.csv")}
    boxes = defaultdict(list)
    for r in read_rows(lab / "page_boxes.csv"):
        if r["class"] == "panel":
            boxes[r["page_id"]].append(r)
    todo = [p for p in sorted(boxes) if not (args.out_dir / f"pages__{p}" / "page.json").exists()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.csv"
    if not progress.exists():
        progress.write_text("page,seconds,panels,leads_ok,leads,status,error\n", encoding="utf-8")
    digitizer = Digitizer(cfg, layout)
    recent, t_start = [], time.time()
    for n, pid in enumerate(tqdm(todo, desc=f"oracle {args.mode}"), 1):
        t0 = time.time()
        out = args.out_dir / f"pages__{pid}"
        out.mkdir(parents=True, exist_ok=True)
        image, _ = fit_side(load_image(args.dataset / "pages" / f"{pid}.png"), cfg["work_side"])
        W, H = int(pages[pid]["width"]), int(pages[pid]["height"])
        cw = float(boxes[pid][0]["cw_rotation_to_upright_deg"])
        k = round((360 - cw) % 360)
        up = np.rot90(image, k // 90).copy()
        recs = []
        gt = []
        for r in boxes[pid]:
            b = [float(r[f"bbox_{c}"]) for c in ("x0", "y0", "x1", "y1")]
            clip = [max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)]
            vis = max(clip[2] - clip[0], 0) * max(clip[3] - clip[1], 0) / ((b[2] - b[0]) * (b[3] - b[1]))
            if vis >= args.min_visible:
                gt.append((rotate_box(clip, W, H, k), panel_meta[r["panel_id"]]))
        gt.sort(key=lambda g: (g[0][1], g[0][0]))
        for i, (box, meta) in enumerate(gt):
            gain = float(re.match(r"\d+", meta["gain"]).group()) if args.mode in ("gain", "labels") and re.match(r"\d+", meta["gain"] or "") else None
            names = meta["row_labels"].split(";") if args.mode == "labels" else None
            recs.append(digitizer.panel(up, None, [round(v) for v in box], i, None, 1.0, cfg["gain"]["assumed"], force_gain=gain, force_labels=names))
        for r in recs:
            r.save(out)
        ok = sum(v.flag == "ok" for r in recs for v in r.leads.values())
        n_leads = sum(len(r.leads) for r in recs)
        (out / "page.json").write_text(json.dumps({"source": str(args.dataset / "pages" / f"{pid}.png"), "layout": layout["name"], "rotation_ccw_deg": k, "work_scale": 1.0, "oracle": args.mode,
                                                   "panels": {r.panel_id: {"confidence": round(r.confidence, 3), "error": r.error, "leads_ok": sum(v.flag == "ok" for v in r.leads.values()), "leads": len(r.leads)} for r in recs}}, indent=1), encoding="utf-8")
        sec = time.time() - t0
        recent = (recent + [sec])[-20:]
        with progress.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([f"{pid}.png", round(sec, 1), len(recs), ok, n_leads, "ok", ""])
        log.info("%s: %d/%d, %.1f s, %d panels, %d/%d leads ok, ETA %s", pid, n, len(todo), sec, len(recs), ok, n_leads, timedelta(seconds=int(sum(recent) / len(recent) * (len(todo) - n))))
    log.info("done in %s", timedelta(seconds=int(time.time() - t_start)))


if __name__ == "__main__":
    main()
