"""Task 4 (lead -> digitized signal): given the per-lead crops in
dataset/<name>/preprocessed/leads/manifest.csv (produced by
panels_from_sheets.py's normalize-and-split step), extract each lead's
numeric signal with paper-ecg's Viterbi digitizer.

One process for the whole manifest, not one subprocess per lead -- unlike
the sheet-level stages (which need SAM and paper-ecg in separate venvs),
this step is paper-ecg only, and per-lead subprocess startup would dominate
runtime over thousands of leads.

Pitch isn't carried in leads/manifest.csv, so it's re-measured directly on
each lead crop (panel_geometry.estimate_pitch works on any image, not just a
full sheet) -- self-contained, and avoids a schema change to a manifest a
background run is already writing.

Usage:
    .venvs/paper-ecg/Scripts/python -m src.inference.leads_to_signals \
        --dataset-root ../.. --dataset ekg-757 [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ECG_ROOT = Path(__file__).resolve().parents[4] / "external" / "paper-ecg" / "src" / "main" / "python"
sys.path.insert(0, str(REPO_ROOT / "src" / "inference"))
sys.path.insert(0, str(PAPER_ECG_ROOT))

import ecgdigitize  # noqa: E402
import ecgdigitize.image  # noqa: E402
import ecgdigitize.signal  # noqa: E402
from Conversion import convertECGLeads, exportSignals  # noqa: E402
from model.InputParameters import InputParameters  # noqa: E402
from model.Lead import Lead, LeadId  # noqa: E402

from panel_geometry import estimate_pitch, looks_like_signal  # noqa: E402

TIME_SCALE = 25  # mm/s, printed on every strip inspected so far
SIGNAL_COLUMNS = ["lead_id", "panel_id", "lead_name", "signal_path", "sample_rate_hz", "n_samples", "status"]


def digitize_one(lead_path: Path, lead_name: str | None) -> tuple[np.ndarray | None, float | None, str]:
    bgr = cv2.imread(str(lead_path))
    if bgr is None:
        return None, None, "unreadable image"
    pitch, _, _ = estimate_pitch(bgr)
    if pitch is None:
        return None, None, "no grid pitch found on lead crop"

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    box = (0, 0, w, h)
    if not looks_like_signal(gray, box, pitch):
        return None, pitch, "rejected as text/non-signal"

    image = ecgdigitize.image.ColorImage(bgr)
    leads = {LeadId.I: Lead(x=0, y=0, width=w, height=h, startTime=0)}
    signals, _ = convertECGLeads(image, InputParameters(rotation=0.0, timeScale=TIME_SCALE, voltScale=10, leads=leads))
    if not signals or LeadId.I not in signals:
        return None, pitch, "grid detection failed"
    return signals[LeadId.I], pitch, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4: digitize per-lead crops into signal CSVs.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessed = args.dataset_root / "dataset" / args.dataset / "preprocessed"
    leads_manifest_path = preprocessed / "leads" / "manifest.csv"
    signals_dir = preprocessed / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signals_manifest_path = signals_dir / "manifest.csv"

    with leads_manifest_path.open(encoding="utf-8") as f:
        leads = list(csv.DictReader(f))

    done_ids = set()
    if signals_manifest_path.exists():
        with signals_manifest_path.open(encoding="utf-8") as f:
            done_ids = {row["lead_id"] for row in csv.DictReader(f)}
    leads = [r for r in leads if r["id"] not in done_ids]
    if args.limit:
        leads = leads[: args.limit]

    write_header = not signals_manifest_path.exists() or signals_manifest_path.stat().st_size == 0
    with signals_manifest_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=SIGNAL_COLUMNS)
        if write_header:
            writer.writeheader()

        for row in tqdm(leads, desc="digitize leads"):
            # lead_path is already relative to the cwd pages_to_panels.py was
            # run from (it stores output_dir / filename directly), not to
            # --dataset-root -- joining it again would double the ../.. prefix.
            lead_path = Path(row["lead_path"])
            signal, pitch, status = digitize_one(lead_path, row["lead_name"] or None)
            signal_path = ""
            n_samples = 0
            sample_rate_hz = None
            if signal is not None:
                sample_rate_hz = 1.0 / ecgdigitize.signal.ecgSignalSamplingPeriod(pitch, TIME_SCALE, gridSizeInMillimeters=1.0)
                signal_csv = signals_dir / f"{row['id']}.csv"
                exportSignals({LeadId.I: signal}, signal_csv, separator=",")
                signal_path = str(signal_csv)
                n_samples = len(signal)

            writer.writerow({
                "lead_id": row["id"],
                "panel_id": row["panel_id"],
                "lead_name": row["lead_name"] or "",
                "signal_path": signal_path,
                "sample_rate_hz": round(sample_rate_hz, 2) if sample_rate_hz else "",
                "n_samples": n_samples,
                "status": status,
            })
            out_f.flush()

    print(f"done, {len(leads)} lead(s) processed, output at {signals_manifest_path}")


if __name__ == "__main__":
    main()
