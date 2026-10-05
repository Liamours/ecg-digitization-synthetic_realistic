"""Why each of a page's 12 lead slots is or is not ok in a method run: which step lost it.

Reads a run folder (src.digitize.methods or src.digitize.run) and prints one table: every lead slot of every page falls
into exactly one row. A slot that is not ok is put down to the first step that failed for it:

- `low_quality`, `flat`: the lead was digitized under a name; the trace or its separation is short or out of range.
- `missing, panel has an error`: a panel is named for the lead but could not be digitized (no grid fit, no gain, no trace).
- `missing, not separated`: a panel is named for the lead and was digitized, but lead separation returned no trace for it.
- `missing, no panel named for it`: no panel on the page carries the lead's name. Split by whether the page has fewer than
  four ECG panels (a panel was not found) or four and more (a panel carries another set's names).

The table is also written to <run>/causes.csv. It also counts how the lead names of the panels were obtained (read on the panel, read on the page, or the template order),
so the share of the result that rests on text reading is visible. Real scans have no ground truth: a lead digitized under a
wrong name counts as ok here.

Usage:
    python -m src.digitize.causes --run @inferences/canonical_mac400-scan
"""
import argparse
import csv
import json
from collections import Counter

from src import paths

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
NAME_SOURCE = {"ocr": "read on the panel", "page_ocr": "read on the page", "order": "reading order of the page (not read)", "position": "template order (not read)", "oracle": "given"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=paths.resolve, required=True, help="run folder with one sub-folder per page")
    ap.add_argument("--pages", action="store_true", help="also print, for every page with a missing lead, its panels' names and how they were obtained")
    args = ap.parse_args()
    slots, names, gains, panel_errors = Counter(), Counter(), Counter(), Counter()
    pages = sorted(p.parent for p in args.run.glob("*/record.json"))
    for page in pages:
        status = {x["lead"]: x["status"] for x in json.loads((page / "record.json").read_text(encoding="utf-8"))["leads"]}
        panels = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(page.glob("panel*.json")) if not f.name.endswith(".front.json")]
        if args.pages and "missing" in status.values():
            print(f"{page.name}: missing {' '.join(k for k, v in status.items() if v == 'missing')} | " + " | ".join(f"{'-'.join(q['labels'])} ({q['label_source']}{', error' if q['error'] else ''})" for q in panels))
        for q in panels:
            names[NAME_SOURCE.get(q["label_source"], q["label_source"])] += 1
            gains[q["gain_source"] or "-"] += 1
            if q["error"]:
                panel_errors[q["error"].split(":")[0] + ": " + q["error"].split(":", 1)[1].strip()[:50]] += 1
        for lead in LEADS:
            if status.get(lead, "missing") != "missing":
                slots[status[lead]] += 1
                continue
            named = [q for q in panels if lead in q["labels"]]
            if not named:
                guessed = any(q["label_source"] in ("position", "order") for q in panels)
                slots[f"missing, no panel named for it, page has {'fewer than 4' if len(panels) < 4 else '4 or more'} ECG panels, {'some names not read' if guessed else 'all names read'}"] += 1
            elif all(q["error"] for q in named):
                slots["missing, panel has an error"] += 1
            else:
                slots["missing, not separated"] += 1
    total = sum(slots.values())
    with (args.run / "causes.csv").open("w", newline="", encoding="utf-8") as fh:  # the same table for other tools to read
        csv.writer(fh).writerows([["lead_slot", "count"]] + sorted(slots.items(), key=lambda kv: (kv[0] != "ok", -kv[1])))
    print(f"{len(pages)} pages, {total} lead slots\n\n| Lead slot | Count | Share |\n|---|---|---|")
    for k, v in sorted(slots.items(), key=lambda kv: (kv[0] != "ok", -kv[1])):
        print(f"| {k} | {v} | {100 * v / total:.0f}% |")
    for title, c in (("Lead names of a panel", names), ("Gain of a panel", gains), ("Panel error", panel_errors)):
        print(f"\n| {title} | Panels |\n|---|---|")
        for k, v in c.most_common():
            print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
