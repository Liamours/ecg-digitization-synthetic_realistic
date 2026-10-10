"""Batch runner: every image under the input folders to one output folder per page, plus index.csv.

python -m ecgdig.run --input "D:/.../Data-EKG-update" --out "D:/.../digitized_v2" [--workers 4] [--limit N] [--no-overlay]

Resumable: a page whose record.json exists is skipped. A page that crashes is logged and skipped, never stops the run.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
INDEX_FIELDS = ["page_id", "folder", "file", "page_type", "panels", "leads_ok", "leads_present", "lead_sets", "gains", "einthoven_corr", "seconds", "status", "record"]

_ocr = None


def _worker_init(threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    global _ocr
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR()


def page_id(root: Path, f: Path) -> str:
    rel = f.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "__".join(rel.parts))


def process(args):
    root, f, out_root, overlay = args
    from .page import digitize_page, save_page
    pid = page_id(Path(root), Path(f))
    out_dir = Path(out_root) / "pages" / pid
    try:
        res = digitize_page(_ocr, str(f), overlay=overlay)
        summ = save_page(res, out_dir, str(f))
        corr = [p["einthoven"]["corr"] for p in res.panels if p.get("einthoven")]
        row = {"page_id": pid, "folder": Path(f).parent.name, "file": Path(f).name, "page_type": res.page_type, "panels": len(res.panels),
               "leads_ok": summ.get("leads_ok", ""), "leads_present": summ.get("leads_present", ""),
               "lead_sets": ";".join(str(p["lead_set"]) for p in res.panels), "gains": ";".join(f"{p['gain_mm_per_mV']:g}" for p in res.panels),
               "einthoven_corr": ";".join(f"{c:.3f}" for c in corr), "seconds": round(res.seconds, 1), "status": "done",
               "record": f"pages/{pid}/record.npz" if res.signal is not None else ""}
        return row, None
    except Exception as e:
        return {"page_id": pid, "folder": Path(f).parent.name, "file": Path(f).name, "status": f"error: {type(e).__name__}: {e}"}, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="folders (searched recursively) or image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / f"run-{time.strftime('%Y%m%d-%H%M%S')}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("run")
    jobs = []
    for inp in a.input:
        p = Path(inp)
        files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        root = p.parent if p.is_file() else p
        for f in files:
            jobs.append((str(root), str(f), str(out), not a.no_overlay))
    index = out / "index.csv"
    done = set()
    if index.exists():
        with index.open(encoding="utf-8") as fh:
            done = {r["page_id"] for r in csv.DictReader(fh) if r.get("status") == "done"}
    else:
        with index.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writeheader()
    jobs = [j for j in jobs if page_id(Path(j[0]), Path(j[1])) not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} pages to do ({len(done)} already done), {a.workers} workers", flush=True)
    t0 = time.time()
    threads = max(1, (os.cpu_count() or 2) // max(1, a.workers))
    with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(threads,)) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            row, tb = fu.result()
            with index.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writerow(row)
            if tb:
                log.error("%s\n%s", row["page_id"], tb)
            else:
                log.info("%s %s panels=%s ok=%s %.1fs", row["page_id"], row["page_type"], row.get("panels"), row.get("leads_ok"), row.get("seconds", 0))
            el = time.time() - t0
            eta = el / k * (len(jobs) - k)
            print(f"[{k}/{len(jobs)}] {row['page_id']}: {row.get('page_type', '')} ok={row.get('leads_ok', '')} {row['status'][:60]}  ETA {eta / 60:.1f} min", flush=True)
    print("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
