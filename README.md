# ecg-digitization-synthetic_realistic

One scanned GE MAC 400 ECG page in, the 12 leads in millivolts with a label per lead and per page out.

## Canonical pipeline (2026-10-03), fully open source

Every model in the pipeline is a published pretrained model, and the trace and lead-separation methods are the published code of open-source ECG digitizers. No model is trained here. The steps that connect them (rotation, panel boxes, grid map, gain, sampling) are rules in this repository.

| Stage | Step | Method | Project | License |
|---|---|---|---|---|
| Front | page text | RapidOCR text detection and recognition, read again once the page is upright | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) | Apache-2.0 |
| Front | rotation | text-box shape for the axis, header above footer for the direction | this repository, on the RapidOCR text | - |
| Front | panels | footer text anchors (`For ...`), sized by footer-to-footer spacing, typed ECG or report by their text | this repository, on the RapidOCR text | - |
| Front | lead names | panel text, then page text, then the template order | this repository, on the RapidOCR text | - |
| Front | grid map | affine fit of the 1 mm dot lattice: pixels per mm and tilt, image never resampled | this repository | - |
| Front | gain | printed gain, then calibration pulse height, then the page majority or 10 mm/mV | this repository, on the RapidOCR text | - |
| Trace | trace mask | pretrained U-Net, inference only | [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer) | CC BY-SA 4.0 |
| Trace | lead separation | track cutting and `fragmented` extraction | [UMMISCO/ecgtizer](https://github.com/UMMISCO/ecgtizer) | Unlicense |
| Trace | signal | sampling in grid millimetres to mV over time, quality flag per lead | this repository | - |
| Record | 12 leads | best attempt per lead on one time base at 500 Hz, time zero at the panel's left edge | this repository | - |

The third-party source files are in `third_party/` as unmodified copies with their licenses (`third_party/README.md` lists source, commit and citation). The trace network's weights (90 MB) are downloaded by `scripts/get_weights.sh`.

Result on the 39 scans of `ecg-mac400-scan` (188 panels found, no ground truth): 385 of 507 digitized leads pass the quality flag (coverage at least 0.9, peak to peak at most 6 mV). The flag is not accuracy.

Other methods stay selectable for comparison (`tracking` in `configs/layouts/mac400.yml`, or `--split` and `--mask` of `src.digitize.methods`): ecgtizer `full` and `lazy`, Open-ECG-Digitizer's own signal extractor (`openecg_lines`), this repository's row bands (`baseline`, 418 of 538 by the same flag) and overlap tracker, and this repository's small trace network (`trace_net`).

## Setup

```
uv sync
scripts/get_weights.sh
scripts/setup_gpu_env.sh
```

`uv sync` builds the repo's environment (CPU torch), enough for the whole pipeline. `setup_gpu_env.sh` builds `../../.venvs/digitize_gpu` (CUDA torch) for the trace stage on a GPU; it is optional.

Code, configs, scripts, and tests live here. Datasets, checkpoints, inference outputs, and logs live in the project root folders (`datasets/`, `models/`, `results/`, `logs/`), referenced through `configs/paths.yml` (roots and dataset aliases). Command line paths and config paths accept `@logs`, `@inferences/<run>`, `@mac400-scan/scan_29`.

## Commands

One image or a folder, whole pipeline (CPU, about 75 s per page):

```
uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> <image-or-folder> ... [--sample N --seed S] [--report]
```

In two steps, so methods can be compared on identical input (front stage on CPU, trace stage on the GPU, about 3 s per page):

```
uv run python -m src.digitize.front --config configs/digitize.yml --layout mac400 --out-dir @inferences/front_mac400-scan @mac400-scan
PYTHONPATH=. ../../.venvs/digitize_gpu/Scripts/python.exe -m src.digitize.methods --config configs/digitize.yml --layout mac400 --front @inferences/front_mac400-scan --mask openecg --split ecgtizer_fragmented --device cuda
scripts/run_methods.sh
```

`run_methods.sh` runs every method on the saved front stage and then `src.digitize.compare`, which writes `pairs.csv`, `leads.csv` and `agreement_map.png` (where the method runs agree per page and lead) and prints the tables per method, per pair and per lead group.

Other tools:

```
python -m src.digitize.overlay --config configs/digitize.yml --layout mac400 --front <front folder> --page <page> --out-dir <dir>
uv run python -m src.digitize.score --dataset <synthetic dataset dir> --run <digitize run dir>
uv run python -m src.digitize.metrics
uv run --with pytest python -m pytest tests/test_digitize.py
```

`overlay` draws, per panel of one page, which trace pixels each lead-separation method gives to each lead. `score` compares a run with a synthetic set's exact labels. All long runs write a log with an ETA and skip finished pages on a re-run.

## Output per page

- `record.csv`: `time_s` and one mV column per lead (I to V6), 4 s at 500 Hz, empty where there is no data.
- `record.json`: image labels (rotation, panels, unreadable panels, leads ok, low quality, flat, missing, complete) and lead labels (status, attempts, panel, gain and its source, name source, coverage, peak to peak, start, duration). Status: `ok`, `low_quality`, `flat` (electrode off), `missing`.
- One JSON and CSV per panel (samples, grid map, gain, self-test results, confidence) and `page.json`.

## Modules: `src/digitize/`

| Module | Job |
|---|---|
| `pipeline.py` | `Digitizer.front` and `Digitizer.trace`; panel mode and the older page mode |
| `front.py`, `methods.py`, `compare.py`, `overlay.py` | front stage to disk, a method on a saved front stage, agreement between method runs, picture of each method's lead assignment |
| `run.py`, `report.py` | whole-page batch runner and its report pictures |
| `record.py`, `assemble.py` | `PanelFront`, `PanelRecord`; the 12-lead page record |
| `gridmap.py` | dot-lattice fit: tilt, pixels per mm, fit residual |
| `ocr.py`, `text.py` | RapidOCR text boxes; gain, lead-name set, header and footer band |
| `pulses.py` | calibration pulse height in mm |
| `mask.py` | trace pixels by the Open-ECG-Digitizer U-Net |
| `separate.py` | lead separation by ecgtizer's track extraction and by Open-ECG-Digitizer's signal extractor |
| `leads.py`, `tracenet.py` | this repository's comparison methods: row bands and overlap tracker; small trace network |
| `sample.py` | mV over time in grid millimetres |
| `selftest.py`, `checks.py` | pulse, Einthoven relations, coverage flags, confidence |
| `score.py`, `oracle.py`, `metrics.py` | score against a synthetic set's exact labels, runs with true steps, tables in `results/metrics/` |
| `warp.py`, `reverse.py` | measured warp of the dot lattice; results drawn back on the original image |

Settings are in `configs/digitize.yml`; the layout (grid, text patterns, mask kind, lead separation, lead sets) in `configs/layouts/mac400.yml`. The Fukuda, EDAN and wide-pitch layouts and the page mode are from before the scope was narrowed to MAC 400 scans and are not maintained.
