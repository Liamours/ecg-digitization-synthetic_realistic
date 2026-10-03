# ecg-digitization-synthetic_realistic

Photographed paper ECG to digitized signal: page orientation, panel detection, template/type/gain reading, lead-row cutting, digitization.

Code, configs, scripts, and tests only. Datasets, checkpoints, inference outputs, and logs live in the project root folders (`datasets/`, `models/`, `results/inferences/`, `logs/`), referenced through `configs/paths.yml` (roots and dataset aliases). Command line paths and config paths accept `@logs`, `@inferences/<run>`, `@mac400-scan/scan_29`; after renaming a dataset folder on disk, change its name in `configs/paths.yml` and nothing else here.

## Current stage: page to panels

Docling reads the printed text. Orientation comes from text-box shape (axis) and header-above-footer (direction). Panels come from the `For 2030887-001` footer anchors, sized by footer-to-footer spacing, and are typed ecg or report by their text. No model is trained.

```
uv run python -m src.run_pages --config configs/pages.yml <image-or-folder> ... [--limit N]
```

## Digitization package: `src/digitize/`

Digitizes a photographed ECG page without repairing the image: each panel gets a coordinate map from its own printed grid (pixels to millimetres), the trace is read in those coordinates, and every result carries the transforms needed to draw it back on the original photo.

| Module | Job |
|---|---|
| `gridmap.py` | dot-lattice fit (MAC 400, Fukuda) or line-grid fit (EDAN); tilt, pixels per mm, fit residual |
| `ocr.py`, `text.py` | RapidOCR text boxes; gain, lead-label set, header and footer band |
| `pulses.py` | calibration pulse height in mm; gain fallback when OCR misses the gain |
| `mask.py` | trace pixels: threshold (light dot grid), local contrast (shaded photo), or the Open-ECG-Digitizer U-Net (dark grid) |
| `leads.py` | row bands and column-by-column lead tracking with crossing counts |
| `sample.py` | mV over time in grid millimetres |
| `selftest.py` | pulse, Einthoven relations, time-lag search, coverage flags, per-panel confidence |
| `reverse.py` | draw signals, panel boxes, and text boxes back on the original image |
| `pipeline.py` | panel mode (several panels per page) and page mode (one sheet, regions from the layout) |
| `run.py` | batch runner: log with ETA, resumable, one folder per page |

Layouts are config, not code: `configs/layouts/{mac400,fukuda,edan}.yml` give the grid kind, gain rule, mask kind, label sets, and regions. Global settings are in `configs/digitize.yml`. A new device is a new layout file.

```
uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> <image-or-folder> ...
uv run python -m src.digitize.run --config configs/digitize.yml --layout mac400 --out-dir <dir> <folder> --sample 5 --seed 42 --exclude <substring> --report
uv run python -m src.digitize.report --out-dir <dir>
uv run python -m src.digitize.score --dataset <synthetic dataset dir> --run <digitize run dir>
uv run --with pytest python -m pytest tests/test_digitize.py
```

`score` needs a synthetic dataset (exact labels) and gives one 0 to 100 number per step, each conditional on the one before it: orientation, panels found, gain read, lead labels, waveform, and the end-to-end mean over every expected lead (0 where a lead was lost). It also writes `score.json` and `score_leads.csv` (one row per expected lead and why it was lost) into the run folder. Real pages have no ground truth and keep the label-free checks (calibration pulse, Einthoven).

`--sample N --seed S` draws the same N pages every time; `--report` (or the report command on an old run) writes `report.png` per page: the original page with panel boxes, text boxes, and traces on the left, every digitized lead in mV over time on the right. Nothing in the run calls a model that needs training; `mac400` uses a threshold trace mask and RapidOCR for text.

Outputs per page: one JSON and CSV per panel (samples in mV with a quality flag per lead, the grid map, gain and its source, self-test results, confidence), `page.json`, and `overlay_upright.png`. Page-mode layouts expect an upright page. The U-Net mask needs the Open-ECG-Digitizer clone in `external/` (CC BY-SA 4.0, check before any product use).
