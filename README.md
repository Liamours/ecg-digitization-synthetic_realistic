# ecg-digitization-synthetic_realistic

Photographed paper ECG to digitized signal: page orientation, panel detection, template/type/gain reading, lead-row cutting, digitization.

Code, configs, scripts, and tests only. Datasets, checkpoints, inference outputs, and logs live in the project root folders (`datasets/`, `models/`, `inferences/`, `logs/`), referenced from configs by relative path.

## Current stage: page to panels

Docling reads the printed text. Orientation comes from text-box shape (axis) and header-above-footer (direction). Panels come from the `For 2030887-001` footer anchors, sized by footer-to-footer spacing, and are typed ecg or report by their text. No model is trained.

```
uv run python -m src.run_pages --config configs/pages.yml <image-or-folder> ... [--limit N]
```
