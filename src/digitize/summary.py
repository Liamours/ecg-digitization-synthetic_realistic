"""Lead status of a digitize run, from the per-page record.json alone, so final-only runs are covered too.

Writes <run>/pages.csv (one row per page) and prints four tables: lead slots over the run, per source folder (the
diagnosis folder on the phone photos), per lead, and where the gain and the lead names of the digitized leads came from.

Usage:
    python -m src.digitize.summary --run @inferences/canonical_mac400-phone_photo
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import PureWindowsPath

from src import paths

STATUS = ("ok", "low_quality", "flat", "missing")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def table(title: str, rows: dict[str, Counter]) -> None:
    print(f"\n| {title} | Pages | " + " | ".join(STATUS) + " | ok share |\n|---|---|" + "---|" * (len(STATUS) + 1))
    for name, c in rows.items():
        slots = sum(c[s] for s in STATUS)
        print(f"| {name} | {c['pages'] or '-'} | " + " | ".join(str(c[s]) for s in STATUS) + f" | {c['ok'] / slots:.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=paths.resolve, required=True, help="run folder with one sub-folder per page")
    args = ap.parse_args()
    total, by_folder, by_lead, sources = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    rows = []
    for rj in sorted(args.run.glob("*/record.json")):
        rec = json.loads(rj.read_text(encoding="utf-8"))
        image, folder = rec["image"], PureWindowsPath(rec["image"]["source"]).parent.name  # sources were written on Windows
        counts = Counter({s: image[f"leads_{s}"] for s in STATUS}, pages=1)
        total += counts
        by_folder[folder] += counts
        for lead in rec["leads"]:
            by_lead[lead["lead"]][lead["status"]] += 1
            if lead["status"] != "missing":
                sources["gain"][lead["gain_source"] or "-"] += 1
                sources["lead name"][lead["name_source"] or "-"] += 1
        rows.append({"page": rj.parent.name, "folder": folder, "rotation_ccw_deg": image["rotation_ccw_deg"], "panels": image["panels"],
                     "panels_unreadable": image["panels_unreadable"], **{s: image[f"leads_{s}"] for s in STATUS}, "complete": image["complete"]})
    with (args.run / "pages.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} pages, {sum(r['complete'] for r in rows)} with all 12 leads ok, {sum(r['panels'] for r in rows)} panels")
    table("Run", {args.run.name: total})
    table("Source folder", dict(sorted(by_folder.items())))
    table("Lead", {k: by_lead[k] for k in LEADS})
    for kind, c in sources.items():
        print(f"\n| {kind.capitalize()} source (digitized leads) | Leads |\n|---|---|")
        for k, v in c.most_common():
            print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
