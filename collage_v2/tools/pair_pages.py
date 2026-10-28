"""Pair every ECG page with the diagnosis form of the same patient, from the order of pages inside each folder.

In Data-EKG-update a folder such as 101-130 holds the 30 diagnosis forms first and then the 30 ECG pages, in the same
patient order. Some folders break the pattern (forms and ECG pages interleaved, missing pages, reversed halves), so the
pairing is only trusted when the folder has as many forms as ECG pages; other folders are listed for a manual check.

python tools/pair_pages.py --run OUT_DIR [--out OUT_DIR/pairs.csv]
(reads OUT_DIR/index.csv written by ecgdig.run, which has the page type of every image)
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def page_no(name: str) -> int:
    m = re.search(r"\.(\d+)\.[A-Za-z]+$", name)
    return int(m.group(1)) if m else 0


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", default="")
a = ap.parse_args()
rows = list(csv.DictReader(open(Path(a.run) / "index.csv", encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[r["folder"]].append(r)
out_rows, report = [], []
for folder, rr in sorted(by.items()):
    rr.sort(key=lambda r: page_no(r["file"]))
    ecg = [r for r in rr if r.get("page_type", "").startswith("ecg")]
    forms = [r for r in rr if r.get("page_type") == "form_or_text"]
    ok = len(ecg) == len(forms) and len(ecg) > 0
    m = re.match(r"(\d+)-(\d+)", folder)
    first = int(m.group(1)) if m else None
    report.append((folder, len(rr), len(forms), len(ecg), "paired" if ok else "check by hand"))
    for i, e in enumerate(ecg):
        f = forms[i] if ok else None
        out_rows.append({"folder": folder, "ecg_file": e["file"], "ecg_page_id": e["page_id"], "form_file": f["file"] if f else "",
                         "patient_no_guess": (first + i) if (ok and first is not None) else "", "pairing": "by order" if ok else "unpaired",
                         "leads_ok": e.get("leads_ok", ""), "page_type": e.get("page_type", ""),
                         "diagnosis_text": "", "label": ""})   # filled in by hand from the form (handwritten)
dst = Path(a.out) if a.out else Path(a.run) / "pairs.csv"
with dst.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print("| Folder | Images | Forms | ECG pages | Pairing |\n|---|---|---|---|---|")
for r in report:
    print("| " + " | ".join(map(str, r)) + " |")
print(f"\nwrote {dst}")
