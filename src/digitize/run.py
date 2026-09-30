"""Digitize ECG pages with a layout.

Each page writes <out-dir>/<page stem>/: one JSON and CSV per panel, page.json (rotation, panel confidences),
overlay.png (the results drawn back on the original page) and a log line with the ETA. Resumable: a page whose page.json
exists is skipped, so a stopped run continues where it stopped.

Usage:
    uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> [--list paths.txt] <image-or-folder> ...
"""
import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src.digitize import reverse
from src.digitize.pipeline import Digitizer, load_image, load_layout

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
log = logging.getLogger("digitize")


def collect(inputs: list[Path]) -> list[Path]:
    files = []
    for p in inputs:
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True, help="layout name in configs/layouts or a path to a layout file")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--list", type=Path, help="text file with one image path per line")
    ap.add_argument("inputs", type=Path, nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    layout = load_layout(args.layout)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=Path(cfg["log_dir"]) / f"digitize-{layout['name']}-{datetime.now():%Y%m%d}.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", encoding="utf-8")
    listed = [Path(x) for x in args.list.read_text(encoding="utf-8").splitlines() if x.strip()] if args.list else []
    files = [f for f in collect(args.inputs) + listed if not (args.out_dir / f"{f.parent.name}__{f.stem}".replace(" ", "_") / "page.json").exists()]
    log.info("%s: %d pages pending", layout["name"], len(files))
    digitizer = Digitizer(cfg, layout)
    t0 = time.time()
    for n, path in enumerate(tqdm(files, desc=layout["name"]), 1):
        out = args.out_dir / f"{path.parent.name}__{path.stem}".replace(" ", "_")
        try:
            image = load_image(path)
            out.mkdir(parents=True, exist_ok=True)  # a page with no panel still gets its page.json
            if layout["mode"] == "panel":
                records, up, k = digitizer.run_panel_page(image)
            else:
                records, up, k = digitizer.run_fixed_page(image), image, 0
            for r in records:
                r.save(out)
            overlay = reverse.draw(up, records, cfg["sampling"])
            cv2.imwrite(str(out / "overlay_upright.png"), overlay)
            if k:
                cv2.imwrite(str(out / "overlay_original.png"), np.rot90(overlay, -(k // 90)).copy())
            (out / "page.json").write_text(json.dumps({"source": str(path), "layout": layout["name"], "rotation_ccw_deg": k,
                                                       "panels": {r.panel_id: {"confidence": round(r.confidence, 3), "error": r.error, "leads_ok": sum(v.flag == "ok" for v in r.leads.values()), "leads": len(r.leads)} for r in records}}, indent=1), encoding="utf-8")
        except Exception as exc:
            log.exception("%s failed: %r", path, exc)
            continue
        rate = n / (time.time() - t0)
        log.info("%s: %d/%d, ETA %s", path.name, n, len(files), timedelta(seconds=int((len(files) - n) / rate)))


if __name__ == "__main__":
    main()
