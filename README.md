# ecg-digitization-synthetic_realistic

Photographed paper ECG to digitized signal: page orientation, panel detection, template/type/gain reading, lead-row cutting, digitization.

Code, configs, scripts, and tests only. Datasets, checkpoints, inference outputs, and logs live in the project root folders (`datasets/`, `models/`, `inferences/`, `logs/`), referenced from configs by relative path.

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
uv run --with pytest python -m pytest tests/test_digitize.py
```

Outputs per page: one JSON and CSV per panel (samples in mV with a quality flag per lead, the grid map, gain and its source, self-test results, confidence), `page.json`, and `overlay_upright.png`. Page-mode layouts expect an upright page. The U-Net mask needs the Open-ECG-Digitizer clone in `external/` (CC BY-SA 4.0, check before any product use).
