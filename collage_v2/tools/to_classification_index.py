"""Write the index.csv of ecg-classification-pjb_normal (Rifqi's record contract) for a run of this digitizer.

Each page folder already holds records/<page>.npz-compatible arrays (signal 12 x 2000 in mV at 500 Hz, ok, present),
so this only copies them under one folder and writes the index with a label taken from a CSV the user provides
(columns: ecg_file, label) - e.g. pairs.csv after the diagnosis column was filled in.

python tools/to_classification_index.py --run OUT_DIR --out OUT_DIR/classification [--labels labels.csv] [--min-ok 12]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ["page", "dataset", "relative_path", "label", "leads_present", "leads_ok", "usable", "file"]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--labels", default="")
ap.add_argument("--dataset", default="Data-EKG-update")
ap.add_argument("--min-ok", type=int, default=12, help="a record is usable with at least this many leads flagged ok")
a = ap.parse_args()
labels = {}
if a.labels:
    for r in csv.DictReader(open(a.labels, encoding="utf-8")):
        labels[r["ecg_file"]] = r.get("label", "")
out = Path(a.out)
(out / "records").mkdir(parents=True, exist_ok=True)
rows = []
for r in csv.DictReader(open(Path(a.run) / "index.csv", encoding="utf-8")):
    if r.get("status") != "done" or not r.get("record"):
        continue
    src = Path(a.run) / r["record"]
    d = np.load(src)
    dst = out / "records" / f"{r['page_id']}.npz"
    np.savez_compressed(dst, signal=d["signal"], ok=d["ok"], present=d["present"])
    rows.append({"page": r["page_id"], "dataset": a.dataset, "relative_path": f"{r['folder']}/{r['file']}", "label": labels.get(r["file"], ""),
                 "leads_present": int(d["present"].sum()), "leads_ok": int(d["ok"].sum()), "usable": int(d["ok"].sum()) >= a.min_ok,
                 "file": f"records/{r['page_id']}.npz"})
with (out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} records, {sum(r['usable'] for r in rows)} usable (>= {a.min_ok} leads ok) -> {out / 'index.csv'}")
