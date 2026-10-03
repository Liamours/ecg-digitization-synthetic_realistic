"""The page's 12 leads on one time base, with a label per lead and per image.

A page holds several panels of three leads; a lead group may be printed more than once (a repeated attempt). Each lead
takes its best attempt. Time zero is the left edge of the lead's panel (25 mm per second along the grid), so the leads
of a panel stay simultaneous and two methods run on the same panel share one time base. Per-lead status: `ok` (passed the quality flag), `low_quality` (digitized but failed it), `flat`
(a straight line, what the device prints when the electrode is off) and `missing` (no panel carried it). Written per page:
record.csv (time_s and one mV column per lead, empty where there is no data) and record.json (image labels, lead labels).
"""
import csv
import json
from pathlib import Path

import numpy as np

from src.digitize.record import Lead, PanelRecord

RANK = {"ok": 3, "low_quality": 2, "flat": 1}


def lead_status(lead: Lead, flat_p2p_mv: float) -> str:
    return "flat" if lead.p2p_mv < flat_p2p_mv else lead.flag


def panel_left_mm(rec: PanelRecord) -> float:
    """Grid x of the panel box's left edge at mid height: the origin of the panel's time axis."""
    left = np.array([[rec.box[0] - rec.crop_origin[0], (rec.box[1] + rec.box[3]) / 2 - rec.crop_origin[1]]], np.float64)
    return float(rec.grid.to_mm(left)[0, 0])


def page_record(records: list[PanelRecord], cfg: dict, mm_per_s: float) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Time base, leads x samples in mV (NaN where there is no data), the lead labels, and the image labels."""
    n = int(round(cfg["duration_s"] * cfg["fs"]))
    t = np.arange(n) / cfg["fs"]
    signal = np.full((len(cfg["leads"]), n), np.nan)
    labels = []
    for i, name in enumerate(cfg["leads"]):
        tries = [(r, r.leads[name]) for r in records if not r.error and name in r.leads]
        if not tries:
            labels.append({"lead": name, "status": "missing", "attempts": 0})
            continue
        rec, lead = max(tries, key=lambda x: (RANK[lead_status(x[1], cfg["flat_p2p_mv"])], x[1].coverage))
        start = (lead.x0_mm - panel_left_mm(rec)) / mm_per_s  # seconds from the panel's left edge to the lead's first sample
        signal[i] = np.interp(t, start + lead.t, lead.mv, left=np.nan, right=np.nan)
        labels.append({"lead": name, "status": lead_status(lead, cfg["flat_p2p_mv"]), "attempts": len(tries), "panel": rec.panel_id, "gain_mm_per_mV": rec.gain_mm_per_mv, "gain_source": rec.gain_source,
                       "name_source": rec.label_source, "coverage": round(lead.coverage, 3), "p2p_mV": round(lead.p2p_mv, 3), "start_s": round(start, 2), "duration_s": round(float(lead.t[-1] - lead.t[0]), 2)})
    count = lambda s: sum(x["status"] == s for x in labels)
    image = {"panels": len(records), "panels_unreadable": sum(bool(r.error) for r in records), "leads_ok": count("ok"), "leads_low_quality": count("low_quality"), "leads_flat": count("flat"),
             "leads_missing": count("missing"), "complete": count("ok") == len(cfg["leads"]), "repeated_leads": sum(x["attempts"] > 1 for x in labels)}
    return t, signal, labels, image


def save(out_dir: Path, t: np.ndarray, signal: np.ndarray, labels: list[dict], image: dict) -> None:
    with (out_dir / "record.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s"] + [f"{x['lead']}_mV" for x in labels])
        for j in range(len(t)):
            w.writerow([f"{t[j]:.4f}"] + ["" if np.isnan(v) else f"{v:.4f}" for v in signal[:, j]])
    (out_dir / "record.json").write_text(json.dumps({"image": image, "leads": labels}, indent=1), encoding="utf-8")
