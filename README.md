# ecg-digitization-synthetic_realistic

One scanned GE MAC 400 ECG page in, the 12 leads in millivolts with a label per lead and per page out.

Code, configs, scripts, and tests only. Datasets, checkpoints, inference outputs, and logs live in the project root folders (`datasets/`, `models/`, `results/`, `logs/`, `external/`), referenced through `configs/paths.yml` (roots and dataset aliases). Command line paths and config paths accept `@logs`, `@inferences/<run>`, `@mac400-scan/scan_29`; after renaming a dataset folder on disk, change its name in `configs/paths.yml` and nothing else here.

## Pipeline

The image is never resampled: each panel gets a coordinate map from its own printed dot grid (pixels to millimetres), the trace is read in those coordinates, and every result carries the transforms needed to draw it back on the original scan.

| Stage | Step | Method |
|---|---|---|
| Front | page text | RapidOCR (pretrained), read again once the page is upright |
| Front | rotation | text-box shape for the axis, header above footer for the direction |
| Front | panels | footer text anchors (`For ...`), sized by footer-to-footer spacing, typed ECG or report by their text |
| Front | lead names | panel text, then page text, then the template order |
| Front | grid map | affine fit of the 1 mm dot lattice: pixels per mm and tilt |
| Front | gain | printed gain, then calibration pulse height, then the page majority or 10 mm/mV |
| Trace | trace mask | pretrained Open-ECG-Digitizer U-Net (`external/Open-ECG-Digitizer`, CC BY-SA 4.0) |
| Trace | lead separation | three row bands with a slowly moving baseline |
| Trace | signal | sampling in grid millimetres to mV over time, quality flag per lead |
| Record | 12 leads | best attempt per lead on one time base at 500 Hz, time zero at the panel's left edge |

The front stage is saved once per page, so trace and lead-separation methods are compared on identical input.

## Commands

```
uv run python -m src.digitize.front --config configs/digitize.yml --layout mac400 --out-dir @inferences/front_mac400-scan @mac400-scan
python -m src.digitize.methods --config configs/digitize.yml --layout mac400 --front @inferences/front_mac400-scan --mask openecg --split baseline --device cuda
python -m src.digitize.compare --methods @inferences/methods_mac400-scan
uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> <image-or-folder> ... [--sample N --seed S] [--report]
uv run python -m src.digitize.score --dataset <synthetic dataset dir> --run <digitize run dir>
uv run python -m src.digitize.metrics
uv run --with pytest python -m pytest tests/test_digitize.py
```

`front` writes per page `page.json`, `overview.jpg`, and per ECG panel `panel<i>.png` plus `panel<i>.front.json`. `methods` runs one trace mask (`openecg`, `trace_net`) and one lead separation (`baseline`, `overlap`) on a saved front stage and writes one folder per combination. `compare` writes `pairs.csv`, `leads.csv` and `agreement_map.png`: where the method runs agree per page and lead. `run` does the whole page in one go. All long runs write a log with an ETA and skip finished pages on a re-run.

The repo's own environment has CPU-only torch; the GPU runs use the project's `.venvs/digitize_gpu` (run from the repo root with `PYTHONPATH=.`).

## Output per page

- `record.csv`: `time_s` and one mV column per lead (I to V6), 4 s at 500 Hz, empty where there is no data.
- `record.json`: image labels (rotation, panels, unreadable panels, leads ok, low quality, flat, missing, complete) and lead labels (status, attempts, panel, gain and its source, name source, coverage, peak to peak, start, duration). Status: `ok` (coverage at least 0.9 and peak to peak at most 6 mV), `low_quality`, `flat` (electrode off), `missing`.
- One JSON and CSV per panel (samples, grid map, gain, self-test results, confidence) and `page.json`.

## Modules: `src/digitize/`

| Module | Job |
|---|---|
| `pipeline.py` | `Digitizer.front` and `Digitizer.trace`; panel mode and the older page mode |
| `front.py`, `methods.py`, `compare.py` | front stage to disk, a method on a saved front stage, agreement between method runs |
| `run.py`, `report.py` | whole-page batch runner and its report pictures |
| `record.py`, `assemble.py` | `PanelFront`, `PanelRecord`; the 12-lead page record |
| `gridmap.py` | dot-lattice fit: tilt, pixels per mm, fit residual |
| `ocr.py`, `text.py` | RapidOCR text boxes; gain, lead-name set, header and footer band |
| `pulses.py` | calibration pulse height in mm |
| `mask.py`, `tracenet.py` | trace pixels: the Open-ECG-Digitizer U-Net, or the repo's own small network |
| `leads.py`, `sample.py` | row bands and lead tracking; mV over time in grid millimetres |
| `selftest.py`, `checks.py` | pulse, Einthoven relations, coverage flags, confidence |
| `score.py`, `oracle.py`, `metrics.py` | score against a synthetic set's exact labels, runs with true steps, tables in `results/metrics/` |
| `warp.py`, `reverse.py` | measured warp of the dot lattice; results drawn back on the original image |

Settings are in `configs/digitize.yml`; the layout (grid, text patterns, mask kind, lead sets) in `configs/layouts/mac400.yml`. The Fukuda, EDAN and wide-pitch layouts and the page mode are from before the scope was narrowed to MAC 400 scans and are not maintained.
