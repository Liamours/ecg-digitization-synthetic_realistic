"""Front stage: from a page image to its upright panels, saved so that every trace method starts from the same input.

Per page it writes <out-dir>/<page>/: page.json (source, rotation, scale, every panel the finder saw with its kind and box),
overview.jpg (the upright page with the panel boxes and what was read), and per ECG panel panel<i>.png (the crop) and
panel<i>.front.json (lead names, gain, grid map, text boxes, calibration pulses; see record.PanelFront). No trace is read
here. <out-dir>/progress.csv gets one row per page, the log one line per page with an ETA. Resumable: a page whose
page.json exists is skipped.

Usage:
    uv run python -m src.digitize.front --config configs/digitize.yml --layout mac400 --out-dir @inferences/front_mac400-scan @mac400-scan
"""
import argparse
import csv
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from src import paths
from src.digitize.pipeline import Digitizer, fit_side, load_image, load_layout
from src.digitize.run import collect, save_overlay

KIND_COLOR = {"ecg": (200, 60, 0), "report": (0, 140, 255), "other": (128, 128, 128)}


def overview(up: np.ndarray, panels: list[dict], fronts: list) -> np.ndarray:
    img = up.copy()
    t = max(2, round(max(img.shape[:2]) / 900))
    by_box = {tuple(f.box): f for f in fronts}
    for q in panels:
        x0, y0, x1, y1 = q["box"]
        cv2.rectangle(img, (x0, y0), (x1, y1), KIND_COLOR[q["kind"]], t)
        f = by_box.get(tuple(q["box"]))
        label = q["kind"] if f is None else f"panel{f.index} {' '.join(f.labels)} | {f.gain or '-'} mm/mV ({f.gain_source or '-'}) | names: {f.label_source}" + (f" | {f.error[:40]}" if f.error else "")
        cv2.putText(img, label, (x0 + 6, y0 + 14 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, KIND_COLOR[q["kind"]], max(1, t - 1))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    ap.add_argument("--list", type=paths.resolve, help="text file with one image path per line")
    ap.add_argument("inputs", type=paths.resolve, nargs="*", help="files or folders; @mac400-scan style aliases work, see configs/paths.yml")
    ap.add_argument("--relabel", action="store_true", help="no OCR: apply the current lead-name rule and trace-row rule to the text already saved in --out-dir and rewrite each panel's names and rows")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    layout = load_layout(args.layout)
    if args.relabel:
        from src.digitize.text import fill_unread_sets, trace_zone

        digitizer, changed, total, rezoned = Digitizer(cfg, layout), 0, 0, 0
        sets = layout["label_sets"]
        hits = {"template order": 0, "reading order": 0}
        hidden = 0
        for page in sorted(p.parent for p in args.out_dir.glob("*/page.json")):
            files = sorted(page.glob("panel*.front.json"), key=lambda f: int(f.name[5:].split(".")[0]))
            metas = [json.loads(f.read_text(encoding="utf-8")) for f in files]
            read = [digitizer.lead_names(m["texts"], m["page_names"], m["box"], m["crop_origin"], m["index"]) for m in metas]
            for f, m, (labels, source) in zip(files, metas, fill_unread_sets(read, sets)):
                total += 1
                dirty = False
                if (labels, source) != (m["labels"], m["label_source"]):
                    changed += 1
                    print(f"{page.name} panel{m['index']}: {'-'.join(m['labels'])} ({m['label_source']}) -> {'-'.join(labels)} ({source})")
                    m["labels"], m["label_source"], dirty = labels, source, True
                if m.get("zone"):  # the rows between header and footer text, by the current rule
                    zone = list(trace_zone(m["texts"], cv2.imread(str(f.with_name(f"panel{m['index']}.png"))).shape[0], layout["zone"]))
                    if zone != list(m["zone"]):
                        rezoned += 1
                        print(f"{page.name} panel{m['index']}: trace rows {m['zone']} -> {zone}")
                        m["zone"], dirty = zone, True
                if dirty:
                    f.write_text(json.dumps(m, indent=1), encoding="utf-8")
            for i, (labels, source) in enumerate(read):  # how good is each fallback: hide a read name and predict it
                if source == "position":
                    continue
                hidden += 1
                masked = [(sets[j % len(sets)], "position") if j == i else r for j, r in enumerate(read)]
                hits["template order"] += sets[i % len(sets)] == labels
                hits["reading order"] += fill_unread_sets(masked, sets)[i][0] == labels
        print(f"{changed} of {total} panels renamed, {rezoned} with new trace rows")
        print(f"fallback check on {hidden} panels whose names were read, each hidden in turn: " + ", ".join(f"{k} right on {v} ({100 * v / max(hidden, 1):.0f}%)" for k, v in hits.items()))
        return
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"front-{args.out_dir.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("front")
    listed = [paths.resolve(x.strip()) for x in args.list.read_text(encoding="utf-8").splitlines() if x.strip()] if args.list else []
    name = lambda f: f"{f.parent.name}__{f.stem}".replace(" ", "_")
    files = [f for f in collect(args.inputs) + listed if not (args.out_dir / name(f) / "page.json").exists()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "progress.csv"
    if not progress.exists():
        progress.write_text("page,seconds,rotation_ccw_deg,panels_ecg,panels_other,panels_unreadable,status,error\n", encoding="utf-8")
    digitizer = Digitizer(cfg, layout)
    recent = deque(maxlen=20)
    for n, path in enumerate(tqdm(files, desc="front"), 1):
        out = args.out_dir / name(path)
        t0 = time.time()
        row = [path.name, 0, "", 0, 0, 0, "ok", ""]
        try:
            image, work_scale = fit_side(load_image(path), cfg["work_side"])
            out.mkdir(parents=True, exist_ok=True)
            fronts, up, k, panels = digitizer.front_page(image)
            for f in fronts:
                f.save(out)
            save_overlay(out / "overview.jpg", overview(up, panels, fronts), cfg["overlay_max_side"])
            # page.json last: it is the done marker
            (out / "page.json").write_text(json.dumps({"source": str(path), "layout": layout["name"], "rotation_ccw_deg": k, "work_scale": round(work_scale, 4), "upright_size": [up.shape[1], up.shape[0]],
                                                       "panels": [{"kind": q["kind"], "box": q["box"]} for q in panels]}, indent=1), encoding="utf-8")
            row[2:6] = [k, len(fronts), len(panels) - len(fronts), sum(bool(f.error) for f in fronts)]
        except Exception as exc:
            log.exception("%s failed: %r", path, exc)
            row[6:8] = ["error", repr(exc)[:120].replace(",", ";")]
        row[1] = round(time.time() - t0, 1)
        recent.append(row[1])
        with progress.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        log.info("%s: %d/%d, %.1f s, rotation %s, %s ECG panels, ETA %s", path.name, n, len(files), row[1], row[2], row[3], timedelta(seconds=int(sum(recent) / len(recent) * (len(files) - n))))


if __name__ == "__main__":
    main()
