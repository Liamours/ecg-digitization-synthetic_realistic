"""Run one trace method and one lead-separation method on a saved front stage (src.digitize.front).

The front stage fixed the page rotation, the panels, lead names, gain and grid map; this reads those files and only does
the uncertain part, so methods can be compared on identical input. Trace methods (--mask): `threshold` (the rules),
`trace_net` (the network of src.train_trace_net), `openecg` (the pretrained Open-ECG-Digitizer U-Net). Lead separation
(--split): `baseline` (row bands with a slowly moving baseline), `overlap` (continuity with the previous column).

Per page it writes the same files as src.digitize.run (panel json and csv, record.csv, record.json, page.json), so the
scorer, the checks and the metrics collector read a method folder like any run. Resumable: a page whose page.json
exists is skipped.

Usage:
    python -m src.digitize.methods --config configs/digitize.yml --layout mac400 --front @inferences/front_mac400-scan --mask trace_net --split baseline [--device cuda]
"""
import argparse
import csv
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from tqdm import tqdm

from src import paths
from src.digitize import assemble
from src.digitize.pipeline import Digitizer, load_layout
from src.digitize.record import PanelFront

MASKS = ("threshold", "trace_net", "openecg")
SPLITS = ("baseline", "overlap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--front", type=paths.resolve, required=True, help="folder written by src.digitize.front")
    ap.add_argument("--mask", choices=MASKS, required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--out-dir", type=paths.resolve, help="default: <front folder>__methods/<mask>__<split>")
    ap.add_argument("--device", help="cpu, cuda or auto; overrides the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    layout = load_layout(args.layout)
    out_dir = args.out_dir or args.front.with_name(args.front.name.replace("front_", "methods_")) / f"{args.mask}__{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"methods-{out_dir.parent.name}-{out_dir.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("methods")
    pages = [p.parent for p in sorted(args.front.glob("*/page.json")) if not (out_dir / p.parent.name / "page.json").exists()]
    progress = out_dir / "progress.csv"
    if not progress.exists():
        progress.write_text("page,seconds,panels,leads_ok,leads,status,error\n", encoding="utf-8")
    digitizer = Digitizer(cfg, layout)
    recent = deque(maxlen=20)
    for n, page_dir in enumerate(tqdm(pages, desc=f"{args.mask} + {args.split}"), 1):
        t0 = time.time()
        out = out_dir / page_dir.name
        out.mkdir(parents=True, exist_ok=True)
        page = json.loads((page_dir / "page.json").read_text(encoding="utf-8"))
        records = [digitizer.trace(PanelFront.load(f), mask_kind=args.mask, tracking=args.split) for f in sorted(page_dir.glob("panel*.front.json"), key=lambda f: int(f.name[5:].split(".")[0]))]
        for r in records:
            r.save(out)
        t_rec, signal, lead_labels, image_labels = assemble.page_record(records, cfg["record"], cfg["sampling"]["mm_per_s"])
        assemble.save(out, t_rec, signal, lead_labels, {"source": page["source"], "rotation_ccw_deg": page["rotation_ccw_deg"], **image_labels})
        ok, n_leads = sum(v.flag == "ok" for r in records for v in r.leads.values()), sum(len(r.leads) for r in records)
        (out / "page.json").write_text(json.dumps({**{k: page[k] for k in ("source", "layout", "rotation_ccw_deg", "work_scale")}, "mask": args.mask, "split": args.split,
                                                   "panels": {r.panel_id: {"confidence": round(r.confidence, 3), "error": r.error, "leads_ok": sum(v.flag == "ok" for v in r.leads.values()), "leads": len(r.leads)} for r in records}}, indent=1), encoding="utf-8")
        sec = time.time() - t0
        recent.append(sec)
        with progress.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([page_dir.name, round(sec, 1), len(records), ok, n_leads, "ok", ""])
        log.info("%s: %d/%d, %.1f s, %d panels, %d/%d leads ok, ETA %s", page_dir.name, n, len(pages), sec, len(records), ok, n_leads, timedelta(seconds=int(sum(recent) / len(recent) * (len(pages) - n))))
    with progress.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{args.mask} + {args.split}: {len(rows)} pages, {sum(int(r['leads_ok']) for r in rows)} of {sum(int(r['leads']) for r in rows)} leads pass the flag -> {out_dir}")


if __name__ == "__main__":
    main()
