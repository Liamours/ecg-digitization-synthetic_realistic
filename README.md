# ecg-digitization-synthetic_realistic

One scanned or photographed GE MAC 400 ECG page in, the 12 leads in millivolts with a label per lead and per page out.

## Method

Every model is a published pretrained model, and the trace and lead-separation steps are the published code of open-source ECG digitizers. No model is trained here. The steps that connect them are rules in this repository.

| Level | Step | Method | Project | License |
|---|---|---|---|---|
| Page | text recognition | RapidOCR text detection and recognition, read again once the page is upright | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | Apache-2.0 |
| Page | rotation fix | text-box shape for the axis, header above footer for the direction | this repository | - |
| Page | panel detection | footer text anchors (`For ...`), sized by footer-to-footer spacing, typed ECG or report by their text | this repository | - |
| Panel | lead naming | panel text, then page text, then the page's reading order | this repository | - |
| Panel | grid mapping | affine fit of the 1 mm dot lattice: pixels per mm and tilt, the image is never resampled | this repository | - |
| Panel | gain reading | printed gain, then calibration pulse height, then the page majority or 10 mm/mV | this repository | - |
| Panel | trace segmentation | pretrained U-Net, inference only | [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) | CC BY-SA 4.0 |
| Panel | lead separation | track cutting and `fragmented` extraction | [UMMISCO/ecgtizer](https://github.com/UMMISCO/ecgtizer) | Unlicense |
| Lead | signal sampling | trace pixels to mV over time in grid millimetres | this repository | - |
| Lead | quality check | rule-based: coverage at least 0.9 and peak to peak at most 6 mV, else `low_quality`; below 0.05 mV `flat` | this repository | - |
| Page | record assembly | best attempt per lead on one time base, 500 Hz, 4 s, time zero at the panel's left edge | this repository | - |

The third-party source files are unmodified copies in `third_party/` with their licenses (`third_party/README.md` lists source, commit and citation). The trace network's weights (90 MB) are downloaded by `scripts/get_weights.sh`.

| Set | Pages | Pages with 12 leads ok | Lead slots ok / low quality / flat / missing |
|---|---|---|---|
| `ecg-mac400-scan` | 39 | 4 | 329 / 83 / 3 / 53 |
| `ecg-mac400-phone_photo` | 754 | 103 | 6536 / 1758 / 4 / 750 |
| `ekg-esta`, MAC 400 pages | 569 | 81 | 5038 / 1148 / 26 / 616 |

The quality check is not accuracy: the real pages have no ground-truth signal. Accuracy is measured on the synthetic sets (`src.digitize.score`).

## Setup

```
uv sync
scripts/get_weights.sh
scripts/setup_gpu_env.sh
```

`uv sync` builds the repo's environment (CPU torch, about 35 s per page). `setup_gpu_env.sh` builds `../../.venvs/digitize_gpu` with CUDA torch and every dependency; with `--device cuda` text recognition, grid mapping and trace segmentation run on the GPU, about 6 s per page.

Code, configs, scripts and tests live here. Datasets, weights, outputs and logs live in the project root folders (`datasets/`, `models/`, `results/`, `logs/`), reached through `configs/paths.yml`: paths accept `@logs`, `@inferences/<run>`, `@mac400-scan/scan_29`.

## Running it

A whole dataset (every page whose manifest style is `mac400`), detached, resumable, with a log and an ETA:

```
scripts/run_canonical.ps1 -Dataset mac400-scan -Layout mac400
```

One image or folder:

```
uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> <image-or-folder> ... [--final-only] [--device cuda]
```

Layouts: `mac400` (300 dpi scans, dot pitch 9 to 15 px) and `mac400_phone` (photos and lower resolutions, 6 to 16 px). Settings are in `configs/digitize.yml`, text rules for rotation and panels in `configs/pages.yml`.

Output per page:

- `record.csv`: `time_s` and one mV column per lead (I to V6), empty where there is no data.
- `record.json`: page labels (rotation, panels, unreadable panels, lead counts by status, complete) and lead labels (status, attempts, panel, gain and its source, name source, coverage, peak to peak, start, duration). Status: `ok`, `low_quality`, `flat`, `missing`.
- `report.png`: the page upright with panel boxes, text boxes and traces, and every lead plotted.
- Without `--final-only` also one JSON and CSV per panel and `page.json`.

`python -m src.digitize.summary --run <run>` counts lead status over a run (by source folder, by lead, by gain and name source) and writes `pages.csv`.

## Code

| Part | Modules (`src/digitize/` unless noted) |
|---|---|
| The method | `pipeline.py` (`Digitizer.front`, `Digitizer.trace`), `orient.py`, `panels.py`, `ocr.py`, `text.py`, `gridmap.py`, `pulses.py`, `mask.py`, `separate.py`, `sample.py`, `selftest.py`, `record.py`, `assemble.py`, `run.py`, `report.py`, `reverse.py` |
| Evaluation | `score.py` and `oracle.py` (against the synthetic sets' exact labels, `scripts/eval_synthetic.sh`, `scripts/eval_oracle.sh`), `checks.py` (calibration pulse, Einthoven), `causes.py` (which step lost each lead), `summary.py`, `metrics.py` (tables in `results/metrics/`), `warp.py` (measured lattice warp) |
| Method comparison | `front.py` (front stage saved once), `methods.py` (one trace and separation method on it), `compare.py` (agreement between methods), `overlay.py` (lead assignment per method), `leads.py` and `tracenet.py` (this repository's row bands, overlap tracker and small trace network, kept as alternatives), `scripts/run_methods.sh` |
| Dataset tools | `src/text_extract.py` (docling text of every source file), `src/manifest.py` (one manifest per real dataset), `src/ink_fraction.py`, `configs/page_style.yml` |
| Training | `src/train_trace_net.py` (the small trace network, stopped at epoch 8; no model is trained for the method) |

```
uv run --with pytest python -m pytest tests/test_digitize.py
```

## Archive

Retired code, not maintained and not importable as is: kept so no project code lives outside the repository.

| Folder | Holds |
|---|---|
| `archive/classical_panel_pipeline/` | the SAM and calibration-square pipeline retired on 2026-09-06 |
| `archive/manual_digitization/` | the hand-built single-page trial of 2026-09-30 that `src/digitize/` grew out of |
| `archive/doctr_trial/` | the DocTr unwarping trial of 2026-09-30 (dropped) |
| `archive/docling_page_trial/` | the docling page-to-panels driver of 2026-09-26 and the RapidOCR page-style labeler, replaced by `src/digitize/` and `src/manifest.py` |
