"""Digitize every MAC 400 ECG panel of one page with the manual pipeline and write the results.
Panels come from src.run_pages (docling anchors), lead labels from the OCR when a full set is read
and otherwise from the panel's position in the page (row-major: limb, augmented, V1-V3, V4-V6, repeat).

Usage (digitization repo venv, from results/analyses/manual_digitization):
    python run_mac400_page.py <page image> <panels json from src.run_pages> <out dir>
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "repo" / "ecg-digitization-synthetic_realistic"))
from digitize_panel import digitize  # noqa: E402

SETS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
MARGIN_X, MARGIN_Y = 90, 20


def read_labels(ocr: list[dict], fallback: list[str]) -> tuple[list[str], str]:
    text = " ".join(b["text"] for b in ocr).replace("U", "V").replace("u", "V")
    for s in SETS:
        if all(name in text for name in s):
            return s, "ocr"
    return fallback, "position"


def main(page_path: Path, panels_json: Path, out_dir: Path) -> None:
    import numpy as _np
    from PIL import Image
    import src.page_style as ps  # noqa: F401  (RapidOCR worker init reused)
    ps._init_worker(8)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = Image.open(page_path).convert("RGB")
    meta = json.load(open(panels_json))
    scale = max(page.size) / 2500
    ecg = [p for p in meta["panels"] if p["kind"] == "ecg"]
    summary = {"source": str(page_path), "panels": []}
    for i, p in enumerate(ecg):
        x0, y0, x1, y1 = [round(v * scale) for v in p["box"]]
        cx0, cy0, cx1, cy1 = max(x0 - MARGIN_X, 0), max(y0 - MARGIN_Y, 0), min(x1 + MARGIN_X, page.width), min(y1 + MARGIN_Y, page.height)
        crop_path = out_dir / f"panel{i}.png"
        page.crop((cx0, cy0, cx1, cy1)).save(crop_path)
        res = ps._reader(_np.asarray(page.crop((cx0, cy0, cx1, cy1))))
        ocr = [{"text": t, "box": [int(_np.array(b)[:, 0].min()), int(_np.array(b)[:, 1].min()), int(_np.array(b)[:, 0].max()), int(_np.array(b)[:, 1].max())]}
               for b, t in zip(res.boxes if res.boxes is not None else [], res.txts or [])]
        labels, source = read_labels(ocr, SETS[i % 4])
        entry = {"panel": i, "box": [x0, y0, x1, y1], "labels": labels, "label_source": source}
        try:
            r = digitize(crop_path, ocr, labels, x_limits=(x0 - cx0, x1 - cx0))
            g = r["grid"]
            entry.update({"gain_mm_per_mV": r["gain"], "tilt_deg": float(np.degrees(np.arctan2(g["a1"][1], g["a1"][0]))), "px_per_mm": float(np.hypot(*g["a1"])), "grid_rms_px": g["rms_px"],
                          "pulse_heights_mm": [round(q["height_mm"], 2) for q in r["pulses"]]})
            entry["leads"] = {}
            for k, v in r["leads"].items():
                p2p = float(np.ptp(np.percentile(v["mv"], [0.5, 99.5])))
                flag = v["coverage"] < 0.9 or p2p > 3.0
                entry["leads"][k] = {"coverage": round(v["coverage"], 2), "p2p_mV": round(p2p, 2), "flag": "low_quality" if flag else "ok"}
            with open(out_dir / f"panel{i}_{'_'.join(labels)}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time_s"] + [k + "_mV" for k in labels] + [k + "_flag" for k in labels])
                first = r["leads"][labels[0]]["t"]
                for j, t in enumerate(first):
                    w.writerow([f"{t:.4f}"] + [f"{r['leads'][k]['mv'][j]:.4f}" for k in labels] + [entry["leads"][k]["flag"] for k in labels])
        except Exception as exc:
            entry["error"] = repr(exc)
        summary["panels"].append(entry)
        print(i, labels, source, {k: (v["coverage"], v["flag"]) for k, v in entry.get("leads", {}).items()}, entry.get("pulse_heights_mm"), entry.get("error", ""))
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
