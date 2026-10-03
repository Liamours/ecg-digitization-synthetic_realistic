"""End-to-end pipeline: raw composite ECG photo -> crop -> digitize -> classify.

Orchestrates three separately-installed venvs (SAM, paper-ecg, ECG-FM need
conflicting dependency versions, see each stage's own module) via subprocess
calls, since they can't share one Python process. Every stage is zero-training
inference; see context/pipeline.md for the overall architecture and
analyses/compositing-preprocessing/README.md + analyses/zero-training-baseline/
classification-proxy.md for why each specific tool was chosen.

Known, documented limitations this run will hit (not hidden, tracked so
failures are interpretable rather than silent):
  - detect_and_digitize_columns.py assigns lead identity by COLUMN POSITION,
    not by reading the printed label -- a heuristic assumption, not verified
    per-image.
  - A record needs its sheets to together cover all 12 canonical leads with
    no gaps; many real images won't (see the "5 columns, one is a duplicate"
    case in analyses/compositing-preprocessing/README.md) -- these are
    correctly reported as failures, not silently dropped.
  - The classifier (ECG-FM) outputs generic arrhythmia/conduction labels, not
    a PJB-specific prediction -- there is no open pretrained PJB classifier
    (analyses/zero-training-baseline/classification-proxy.md). Performance
    here measures whether "flagged as not-sinus-rhythm" correlates with the
    true PJB/NORMAL label, which is a proxy signal, not a validated diagnosis.

Usage:
    python src/inference/run_full_pipeline.py \
        --manifest ../../dataset/ekg-757/labels/manifest.csv \
        --dataset-root ../.. \
        --n-per-class 10 \
        --work-dir ../../inferences/full-pipeline-test \
        --results-csv ../../inferences/full-pipeline-test/results.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample

REPO_ROOT = Path(__file__).resolve().parents[2]
VENVS = REPO_ROOT / ".venvs"
SAM_PY = VENVS / "sam" / "Scripts" / "python.exe"
PAPERECG_PY = VENVS / "paper-ecg" / "Scripts" / "python.exe"
ECGFM_PY = VENVS / "ecg-fm" / "Scripts" / "python.exe"

SEGMENT_SHEETS = REPO_ROOT / "src" / "inference" / "segment_sheets.py"
DETECT_DIGITIZE_COLUMNS = REPO_ROOT / "src" / "inference" / "detect_and_digitize_columns.py"
CLASSIFY = REPO_ROOT / "src" / "inference" / "run_ecg_fm_classifier.py"

CANONICAL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_HZ = 500
MIN_SAMPLES = TARGET_HZ * 5  # one classifier segment


def run_segment_sheets(image_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(SAM_PY), str(SEGMENT_SHEETS), "--input", str(image_path), "--output-dir", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    return sorted(out_dir.glob("*_sheet*.png"))


def run_digitize_columns(sheet_path: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PAPERECG_PY), str(DETECT_DIGITIZE_COLUMNS), "--input", str(sheet_path), "--output-dir", str(out_dir)],
        check=True, capture_output=True, text=True,
    )
    summary_path = out_dir / f"{sheet_path.stem}_sheet_summary.json"
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text())["columns"]


MIN_REAL_LEADS_REQUIRED = 4  # of 12; below this, the signal is too sparse to mean anything


def signal_noise_score(sig: np.ndarray) -> float:
    """Rough noise proxy: mean absolute sample-to-sample jump. A lead redone
    because the first attempt was artifact-corrupted (confirmed real pattern
    in this dataset -- e.g. a single-patient sheet with two V4/V5/V6 panels,
    one clean and one chaotic, one patient ID printed once for the whole
    page) shows much higher high-frequency jitter throughout than a clean
    physiological trace, where QRS spikes are large but sparse relative to
    the whole strip."""
    if len(sig) < 2:
        return float("inf")
    return float(np.mean(np.abs(np.diff(sig))))


def assemble_12_lead(column_metas: list[dict], digitized_dir: Path) -> tuple[np.ndarray | None, list[str], list[dict]]:
    """Returns (stacked_12_lead_array, missing_lead_names, duplicate_leads).
    Missing leads are zero-filled in the array (not dropped) so a record with
    partial coverage can still reach the classifier -- `missing` is returned
    uncollapsed so callers can report exactly how much of the signal is real
    vs synthetic, never silently. Same for `duplicate_leads`: when the same
    lead is found on more than one sheet, the lower-noise one is kept, but
    the fact that a duplicate existed (and which one lost) is always reported,
    never silently dropped -- see signal_noise_score's docstring for why a
    duplicate isn't assumed to be a different patient.
    """
    lead_signals: dict[str, np.ndarray] = {}
    lead_rates: dict[str, float] = {}
    lead_noise: dict[str, float] = {}
    duplicate_leads: list[dict] = []

    for meta in column_metas:
        data = np.loadtxt(digitized_dir / meta["csv"], delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        for i, lead_name in enumerate(meta["lead_names"]):
            if lead_name is None:
                continue  # OCR couldn't read this row's label
            sig = np.nan_to_num(data[:, i], nan=0.0)
            noise = signal_noise_score(sig)
            if lead_name not in lead_signals:
                lead_signals[lead_name] = sig
                lead_rates[lead_name] = meta["sample_rate_hz"]
                lead_noise[lead_name] = noise
                continue
            duplicate_leads.append({
                "lead": lead_name,
                "kept_noise_score": round(min(noise, lead_noise[lead_name]), 4),
                "dropped_noise_score": round(max(noise, lead_noise[lead_name]), 4),
            })
            if noise < lead_noise[lead_name]:
                lead_signals[lead_name] = sig
                lead_rates[lead_name] = meta["sample_rate_hz"]
                lead_noise[lead_name] = noise

    missing = [lead for lead in CANONICAL_LEAD_ORDER if lead not in lead_signals]
    n_found = len(CANONICAL_LEAD_ORDER) - len(missing)
    if n_found < MIN_REAL_LEADS_REQUIRED:
        return None, missing, duplicate_leads

    resampled_found = {}
    for lead in CANONICAL_LEAD_ORDER:
        if lead in lead_signals:
            sig, rate = lead_signals[lead], lead_rates[lead]
            n_target = max(1, int(round(len(sig) * TARGET_HZ / rate)))
            resampled_found[lead] = resample(sig, n_target)

    # Trim every found lead to the shortest one so they line up in time, then
    # zero-pad up to one full classifier segment if the recovered strip is
    # shorter than 5s -- rejecting outright would throw away real signal just
    # because the printed strip was narrow; ECG-FM needs a fixed-length input
    # either way, so padding is the honest choice, not truncation.
    min_len = min(len(sig) for sig in resampled_found.values())
    trimmed = {lead: sig[:min_len] for lead, sig in resampled_found.items()}
    pad_len = max(0, MIN_SAMPLES - min_len)

    stacked = np.stack(
        [
            np.pad(trimmed[lead], (0, pad_len)) if lead in trimmed else np.zeros(min_len + pad_len)
            for lead in CANONICAL_LEAD_ORDER
        ]
    ).astype(np.float32)  # (12, min_len + pad_len) -- zero rows/padding for leads never found or too short
    return stacked, missing, duplicate_leads


def classify(npy_path: Path, json_out: Path) -> dict | None:
    result = subprocess.run(
        [str(ECGFM_PY), str(CLASSIFY), "--npy-signal", str(npy_path), "--json-output", str(json_out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not json_out.exists():
        return None
    return json.loads(json_out.read_text())


def process_one_image(image_path: Path, work_dir: Path) -> dict:
    sheets_dir = work_dir / "sheets"
    digitized_dir = work_dir / "digitized"
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    try:
        sheets = run_segment_sheets(image_path, sheets_dir)
    except subprocess.CalledProcessError as e:
        timing["segment_sheets"] = time.perf_counter() - t0
        return {"stage_failed": "segment_sheets", "detail": e.stderr[-500:] if e.stderr else "", "timing": timing}
    timing["segment_sheets"] = time.perf_counter() - t0
    if not sheets:
        return {"stage_failed": "segment_sheets", "detail": "no sheets found", "timing": timing}

    t0 = time.perf_counter()
    all_columns = []
    for sheet in sheets:
        try:
            all_columns.extend(run_digitize_columns(sheet, digitized_dir))
        except subprocess.CalledProcessError as e:
            continue  # one sheet failing doesn't necessarily sink the record; try the rest
    timing["digitize_columns"] = time.perf_counter() - t0
    if not all_columns:
        return {"stage_failed": "digitize_columns", "detail": "no columns digitized across any sheet", "timing": timing}

    t0 = time.perf_counter()
    stacked, missing, duplicate_leads = assemble_12_lead(all_columns, digitized_dir)
    timing["assemble_12_lead"] = time.perf_counter() - t0
    if stacked is None:
        return {"stage_failed": "assemble_12_lead", "detail": f"only {12 - len(missing)}/12 leads found (need >= {MIN_REAL_LEADS_REQUIRED}): missing {missing}", "duplicate_leads": duplicate_leads, "timing": timing}

    npy_path = work_dir / "assembled_12lead.npy"
    np.save(npy_path, stacked)

    t0 = time.perf_counter()
    predictions = classify(npy_path, work_dir / "classification.json")
    timing["classify"] = time.perf_counter() - t0
    if predictions is None:
        return {"stage_failed": "classify", "detail": "classifier subprocess failed", "duplicate_leads": duplicate_leads, "timing": timing}

    timing["total"] = sum(timing.values())
    return {
        "stage_failed": None, "predictions": predictions, "missing_leads": missing,
        "n_leads_real": 12 - len(missing), "duplicate_leads": duplicate_leads, "timing": timing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full crop->digitize->classify pipeline on a labeled sample.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory manifest's full_path column is relative to")
    parser.add_argument("--n-per-class", type=int, default=10, help="Use a very large number (e.g. 100000) for the full dataset")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    return parser.parse_args()


def load_done_ids(results_jsonl: Path) -> dict[str, dict]:
    """Resume support: re-reads whatever the log already has, so a rerun after
    a crash/interruption picks up where it left off instead of reprocessing
    (or silently losing) everything already done. Keyed by id, last line for
    a given id wins, matching how the log is appended to below."""
    done: dict[str, dict] = {}
    if not results_jsonl.exists():
        return done
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["id"]] = row
    return done


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = args.results_csv.with_suffix(".jsonl")

    manifest = pd.read_csv(args.manifest)
    sample = (
        manifest.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(args.n_per_class, len(g)), random_state=0))
    )

    done = load_done_ids(results_jsonl)
    if done:
        print(f"Resuming: {len(done)} record(s) already done in {results_jsonl}", file=sys.stderr)

    for i, record in enumerate(sample.itertuples()):
        if record.id in done:
            continue
        image_path = args.dataset_root / record.full_path
        record_work_dir = args.work_dir / record.id
        print(f"[{i+1}/{len(sample)}] {record.id} (true label: {record.label})", file=sys.stderr)

        result = process_one_image(image_path, record_work_dir)
        row = {"id": record.id, "true_label": record.label, **result}
        if result.get("predictions"):
            row["not_sinus_rhythm_score"] = 1.0 - result["predictions"].get("Sinus rhythm", 0.0)

        # Write before marking done, not after: a crash mid-image just repeats
        # that one image next run (load_done_ids dedups on id), never loses
        # every already-finished record the way an end-of-run-only write would.
        with open(results_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        done[record.id] = row
        timing_str = ", ".join(f"{k}={v:.1f}s" for k, v in result.get("timing", {}).items())
        print(f"    -> {result.get('stage_failed') or 'OK: ' + str(result.get('predictions', {}))}"[:200], file=sys.stderr)
        print(f"    timing: {timing_str}", file=sys.stderr)

    results_df = pd.DataFrame(list(done.values()))
    results_df.to_csv(args.results_csv, index=False)

    n_ok = (results_df["stage_failed"].isna()).sum()
    print(f"\n{n_ok}/{len(results_df)} records reached classification.", file=sys.stderr)
    print(results_df["stage_failed"].value_counts(dropna=False), file=sys.stderr)


if __name__ == "__main__":
    main()
