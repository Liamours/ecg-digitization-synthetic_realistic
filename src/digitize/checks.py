"""Label-free scoring of any digitize run: the calibration pulse must read 1 mV and Einthoven's law must hold.

Reads <run>/<page>/panel*.json and panel*.csv, so it scores any digitizer that writes that format. Per panel it reports the
pulse height in mV (when a pulse was found) and the best detrended Einthoven correlation over the relations the present
leads allow; a panel passes a check when the value is inside the limit in the config. Writes <run>/checks.csv.

Usage:
    uv run python -m src.digitize.checks --config configs/digitize.yml --run @inferences/digitize_mac400-scan/baseline_round5
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.digitize.record import Lead
from src.digitize.report import read_leads
from src.digitize.selftest import check_leads, pulse_check


def score_panel(meta: dict, csv_path: Path, cfg: dict) -> dict:
    """Pulse and Einthoven values for one panel; None where the check cannot run (no pulse found, fewer than three leads)."""
    out = {"pulse_mv": None, "einthoven_corr": None, "pulse_ok": None, "einthoven_ok": None}
    pulse = pulse_check(meta.get("pulses_mm", []), meta.get("gain_mm_per_mV"))
    if pulse:
        out["pulse_mv"] = pulse["pulse_mV"]
        out["pulse_ok"] = abs(pulse["pulse_mV"] - 1.0) <= cfg["pulse_tol_mv"]
    if csv_path.exists():
        t, series = read_leads(csv_path)
        leads = {n: Lead(n, t, np.nan_to_num(mv), 0.0, 0.0, 1.0, 0.0) for n, (mv, _) in series.items()}
        corr = [c["corr_at_best_lag"] for c in check_leads(leads, cfg["max_lag_bins"]).values() if "corr_at_best_lag" in c and np.isfinite(c["corr_at_best_lag"])]
        if corr:
            out["einthoven_corr"] = round(min(corr), 3)
            out["einthoven_ok"] = out["einthoven_corr"] >= cfg["einthoven_min_corr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=paths.resolve, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["checks"]
    rows = []
    for pj in tqdm(sorted(args.run.glob("*/panel*.json")), desc="panels"):
        meta = json.loads(pj.read_text(encoding="utf-8"))
        if meta["error"]:
            continue
        rows.append({"page": pj.parent.name, "panel": meta["panel_id"], "gain_source": meta["gain_source"], **score_panel(meta, pj.with_suffix(".csv"), cfg)})
    with (args.run / "checks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    share = lambda k: (f"{100 * sum(r[k] for r in rows if r[k] is not None) / max(sum(r[k] is not None for r in rows), 1):.0f}% of {sum(r[k] is not None for r in rows)} panels")
    print(f"| Check | Pass |\n|---|---|\n| Calibration pulse reads 1 mV (within {cfg['pulse_tol_mv']} mV) | {share('pulse_ok')} |\n| Einthoven correlation at least {cfg['einthoven_min_corr']} | {share('einthoven_ok')} |")


if __name__ == "__main__":
    main()
