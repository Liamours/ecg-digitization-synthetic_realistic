"""Collect the numbers of every run into the metrics folder: one row per scored synthetic run, one row per digitize run.

synthetic_scores.csv comes from each run's score.json (src.digitize.score); run_leads.csv from each run's progress.csv
and, where present, the per-page record.json (lead status counts). Archived runs are included, so the history stays in
one table after their folders are moved.

Usage:
    uv run python -m src.digitize.metrics
"""
import csv
import json
from collections import Counter
from pathlib import Path

from src import paths

SCORE_FIELDS = ["run", "pages", "orientation", "panels_f1", "panels_recall", "panels_precision", "gain", "labels", "waveform", "end_to_end", "leads_expected", "leads_recovered"]
LEAD_FIELDS = ["run", "pages", "panels", "leads", "leads_flag_ok", "record_ok", "record_low_quality", "record_flat", "record_missing", "pages_complete"]


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    inf, out = paths.root("inferences"), paths.root("metrics")
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for f in sorted(inf.rglob("score.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        scores.append({"run": f.parent.relative_to(inf).as_posix(), "leads_recovered": s["funnel"].get("recovered", 0),
                       **{k: (round(s[k], 1) if isinstance(s[k], float) else s[k]) for k in SCORE_FIELDS if k in s}})
    write(out / "synthetic_scores.csv", SCORE_FIELDS, scores)
    runs = []
    for f in sorted(inf.rglob("progress.csv")):
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "panels" not in rows[0]:  # the front stage keeps its own columns and reads no leads
            continue
        status = Counter()
        complete = 0
        for rj in f.parent.glob("*/record.json"):
            image = json.loads(rj.read_text(encoding="utf-8"))["image"]
            status.update({k: image[f"leads_{k}"] for k in ("ok", "low_quality", "flat", "missing")})
            complete += image["complete"]
        has = bool(status)
        runs.append({"run": f.parent.relative_to(inf).as_posix(), "pages": len(rows), "panels": sum(int(r["panels"]) for r in rows), "leads": sum(int(r["leads"]) for r in rows),
                     "leads_flag_ok": sum(int(r["leads_ok"]) for r in rows), "record_ok": status["ok"] if has else "-", "record_low_quality": status["low_quality"] if has else "-",
                     "record_flat": status["flat"] if has else "-", "record_missing": status["missing"] if has else "-", "pages_complete": complete if has else "-"})
    write(out / "run_leads.csv", LEAD_FIELDS, runs)
    print(f"{len(scores)} scored runs -> {out / 'synthetic_scores.csv'}\n{len(runs)} runs -> {out / 'run_leads.csv'}")


if __name__ == "__main__":
    main()
