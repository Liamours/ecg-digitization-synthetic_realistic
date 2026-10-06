"""Digitize the four panels of the trial page and write the results: one CSV per panel,
summary.json (gain, tilt, scale, calibration pulses, coverage), leads_summary.png.
Usage: python run_page.py
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from digitize_panel import digitize_panels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "inferences" / "manual_digitization_trial"


def main() -> None:
    results = digitize_panels(OUT / "inputs")
    meta = {"source": "datasets/ecg-mac400-scan/scan_29/Image_20260922_0025.jpg",
            "method": "manual: docling panel boxes, dot-grid local map, threshold trace mask, mm-space sampling; no model trained", "panels": {}}
    fig, axes = plt.subplots(6, 2, figsize=(14, 13))
    flat = []
    for i, r in results.items():
        labels = list(r["leads"])
        with open(OUT / f"panel{i}_{'_'.join(labels)}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s"] + [k + "_mV" for k in labels])
            for j, t in enumerate(r["leads"][labels[0]]["t"]):
                w.writerow([f"{t:.4f}"] + [f"{r['leads'][k]['mv'][j]:.4f}" for k in labels])
        g = r["grid"]
        meta["panels"][i] = {"labels": labels, "gain_mm_per_mV": r["gain"], "tilt_deg": float(np.degrees(np.arctan2(g["a1"][1], g["a1"][0]))), "px_per_mm": float(np.hypot(*g["a1"])),
                             "grid_rms_px": g["rms_px"], "pulse_heights_mm": [round(p["height_mm"], 2) for p in r["pulses"]], "coverage": {k: round(r["leads"][k]["coverage"], 2) for k in labels}}
        flat += [(k, r["leads"][k]) for k in labels]
    for ax, (k, v) in zip(axes.ravel(), flat):
        ax.plot(v["t"], v["mv"], lw=0.7)
        ax.set_ylabel(k, rotation=0, labelpad=14)
        bad = v["coverage"] < 0.9 or np.ptp(np.percentile(v["mv"], [0.5, 99.5])) > 3
        ax.set_title("low quality (artifacts, row crossing)" if bad else "", fontsize=8, color="red", loc="right")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(OUT / "leads_summary.png", dpi=80)
    json.dump(meta, open(OUT / "summary.json", "w"), indent=1)
    for i, m in meta["panels"].items():
        print(i, m["labels"], "gain", m["gain_mm_per_mV"], "pulses", m["pulse_heights_mm"], "coverage", m["coverage"])


if __name__ == "__main__":
    main()
