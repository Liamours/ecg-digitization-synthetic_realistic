"""Summary tables of a run (written to <run>/RESULTS.md and printed).

python tools/summarize_run.py --run OUT_DIR
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
run = Path(a.run)
rows = list(csv.DictReader(open(run / "index.csv", encoding="utf-8")))
types = Counter(r.get("page_type", "error") for r in rows)
lead_status = {l: Counter() for l in LEADS}
name_src, gain_src, errors = Counter(), Counter(), Counter()
ein = {"limb": [], "aug": []}
ok_per_page, secs = [], []
strips = 0
for r in rows:
    if r.get("page_type") != "ecg_mac400":
        continue
    j = json.loads((run / "pages" / r["page_id"] / "record.json").read_text(encoding="utf-8"))
    ok_per_page.append(sum(l["status"] == "ok" for l in j["leads"]))
    secs.append(j.get("seconds", 0))
    for l in j["leads"]:
        lead_status[l["lead"]][l["status"]] += 1
    for p in j["panels"]:
        strips += 1
        if p.get("error"):
            errors["report strip" if "report" in p["error"] else ("no trace" if "no trace" in p["error"] else "other")] += 1
            continue
        name_src[str(p.get("name_source"))] += 1
        gain_src[p["gain_source"].split(" ")[0]] += 1
        e = p.get("einthoven") or {}
        if p.get("lead_set") in ein and e.get("corr") is not None:
            ein[p["lead_set"]].append(e["corr"])
okp = np.array(ok_per_page) if ok_per_page else np.zeros(1)
out = []
out.append("# Digitization run summary\n")
out.append("| Page type | Images |\n|---|---|")
out += [f"| {k} | {v} |" for k, v in types.most_common()]
out.append(f"\nECG pages: {len(ok_per_page)}, strips: {strips}, median {np.median(secs) if secs else 0:.1f} s per ECG page.\n")
out.append("| Leads flagged ok per page | Pages |\n|---|---|")
for lo, hi, lab in [(12, 12, "12 (complete)"), (9, 11, "9-11"), (6, 8, "6-8"), (1, 5, "1-5"), (0, 0, "0")]:
    out.append(f"| {lab} | {int(((okp >= lo) & (okp <= hi)).sum())} |")
out.append("\n| Lead | ok | low_quality | flat | missing |\n|---|---|---|---|---|")
for l in LEADS:
    c = lead_status[l]
    out.append(f"| {l} | {c['ok']} | {c['low_quality']} | {c['flat']} | {c['missing']} |")
out.append("\n| Strip named by | Strips |\n|---|---|")
out += [f"| {k} | {v} |" for k, v in name_src.most_common()]
out.append("\n| Gain from | Strips |\n|---|---|")
out += [f"| {k} | {v} |" for k, v in gain_src.most_common()]
if errors:
    out.append("\n| Strip not digitized | Strips |\n|---|---|")
    out += [f"| {k} | {v} |" for k, v in errors.most_common()]
out.append("\nSelf-consistency of simultaneous leads (no ground truth needed):\n")
out.append("| Strip | Identity | Strips | Median corr | Strips with corr >= 0.9 |\n|---|---|---|---|---|")
for k, ident in (("limb", "II = I + III"), ("aug", "aVR + aVL + aVF = 0")):
    v = np.array(ein[k])
    if v.size:
        out.append(f"| {k} | {ident} | {v.size} | {np.median(v):.2f} | {(v >= 0.9).mean() * 100:.0f} % |")
txt = "\n".join(out) + "\n"
(run / "RESULTS.md").write_text(txt, encoding="utf-8")
print(txt)
