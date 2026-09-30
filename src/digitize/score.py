"""Score a digitize run on synthetic pages against the generator's exact labels, one 0 to 100 number per step.

Steps, each conditional on the one before it, plus the end-to-end number:
  orientation  pages whose chosen rotation makes the page upright
  panels       F1 of panel boxes at IoU >= 0.5 (ground truth rotated into the pipeline's own frame)
  gain         matched panels whose printed gain (5 or 10 mm/mV) is read correctly
  labels       matched panels whose lead labels are right, in order
  waveform     mean over recovered leads of the share of the true waveform reproduced within 10% of its peak-to-peak
               (at least 0.05 mV) times the shape correlation, after the best time shift up to 0.5 s and a median offset
  end-to-end   mean lead score over every expected ground-truth lead, 0 where the lead was not recovered

Usage:
    uv run python -m src.digitize.score --dataset <synthetic dataset dir> --run <digitize run dir>
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.digitize.report import read_leads
from src.orient import rotate_box


def iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def waveform_score(gt: np.ndarray, fs: float, t: np.ndarray, mv: np.ndarray, max_shift_s: float = 0.5, tol_frac: float = 0.10) -> dict:
    """Share of the true waveform reproduced, 0 to 100: samples within tol_frac of the true peak-to-peak (at least 0.05 mV),
    over the true duration, after the best time shift and a median offset, times the shape correlation (most ECG samples sit near
    baseline, so without it a flat or inverted trace would score high). Missing stretches count as wrong and a short artifact
    costs only its own length."""
    tg = np.arange(len(gt)) / fs
    dt = float(np.median(np.diff(t)))
    dur = len(gt) / fs
    tol = max(0.05, tol_frac * float(np.ptp(gt)))
    best = {"score": 0.0, "corr": float("nan"), "amp_ratio": float("nan"), "shift_s": 0.0}
    for s in np.arange(-max_shift_s, max_shift_s + 1e-9, 0.02):
        g = np.interp(t + s, tg, gt, left=np.nan, right=np.nan)
        ok = np.isfinite(g) & np.isfinite(mv)
        if ok.sum() * dt < 0.3 * dur:
            continue
        z = mv[ok] - np.median(mv[ok] - g[ok])
        r = float(np.corrcoef(z, g[ok])[0, 1]) if z.std() > 0 and g[ok].std() > 0 else 0.0
        score = min(100.0, 100 * float((np.abs(z - g[ok]) <= tol).sum()) * dt / dur) * max(0.0, r)
        if score > best["score"]:
            best = {"score": score, "corr": r, "amp_ratio": float(z.std() / max(g[ok].std(), 1e-9)), "shift_s": float(s)}
    return best


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-visible", type=float, default=0.5, help="a ground-truth panel with less of its box on the page is not expected to be found")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    root = args.dataset.resolve().parent.parent
    labels = args.dataset / "labels"
    pages = {r["page_id"]: r for r in read_rows(labels / "manifest_pages.csv")}
    boxes = defaultdict(list)
    for r in read_rows(labels / "page_boxes.csv"):
        if r["class"] == "panel":
            boxes[r["page_id"]].append(r)
    panels = {r["panel_id"]: r for r in read_rows(labels / "manifest_panels.csv")}
    leads = defaultdict(dict)
    for r in read_rows(labels / "manifest_leds.csv"):
        leads[r["panel_id"]][r["lead_name"]] = r

    n_pages = n_orient = 0
    tp = fp = fn = 0
    ious, gain_ok, gain_n, lab_ok, lab_n = [], 0, 0, 0, 0
    lead_rows, funnel = [], Counter()
    for page_json in sorted(args.run.glob("*/page.json")):
        pid = page_json.parent.name.split("__", 1)[1]
        if pid not in pages:
            continue
        run = json.loads(page_json.read_text(encoding="utf-8"))
        W, H = int(pages[pid]["width"]), int(pages[pid]["height"])
        k = run["rotation_ccw_deg"]
        n_pages += 1
        cw = float(boxes[pid][0]["cw_rotation_to_upright_deg"]) if boxes[pid] else 0.0
        n_orient += k == round((360 - cw) % 360)
        gt = []
        for r in boxes[pid]:
            b = [float(r[f"bbox_{c}"]) for c in ("x0", "y0", "x1", "y1")]
            clip = [max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)]
            vis = max(clip[2] - clip[0], 0) * max(clip[3] - clip[1], 0) / ((b[2] - b[0]) * (b[3] - b[1]))
            gt.append({"id": r["panel_id"], "box": rotate_box(clip, W, H, k), "expected": vis >= args.min_visible})
        pipe = []
        for pj in sorted(page_json.parent.glob("panel*.json"), key=lambda p: int(p.stem[5:])):
            pipe.append({"file": pj, "meta": json.loads(pj.read_text(encoding="utf-8"))})
        pairs = sorted(((iou(g["box"], p["meta"]["box"]), i, j) for i, g in enumerate(gt) if g["expected"] for j, p in enumerate(pipe)), reverse=True)
        gi, pj_used, match = set(), set(), {}
        for v, i, j in pairs:
            if v >= args.iou and i not in gi and j not in pj_used:
                gi.add(i)
                pj_used.add(j)
                match[i] = j
                ious.append(v)
        exp = sum(g["expected"] for g in gt)
        tp += len(match)
        fn += exp - len(match)
        fp += len(pipe) - len(match)
        for i, g in enumerate(gt):
            if not g["expected"]:
                continue
            gl = leads[g["id"]]
            if i not in match:
                funnel["panel not found"] += len(gl)
                lead_rows += [[pid, g["id"], n, "panel not found", 0, "", "", "", ""] for n in gl]
                continue
            meta = pipe[match[i]]["meta"]
            printed = re.match(r"\d+", panels[g["id"]]["gain"])  # some panels print no gain at all
            if printed:
                gain_n += 1
                gain_ok += meta["gain_mm_per_mV"] == float(printed.group())
            names = list(meta["leads"])
            lab_n += 1
            lab_ok += meta["labels"] == panels[g["id"]]["row_labels"].split(";")
            csv_path = pipe[match[i]]["file"].with_suffix(".csv")
            series = read_leads(csv_path) if csv_path.exists() else (None, {})
            for name, row in gl.items():
                if name not in series[1]:
                    why = "panel unreadable" if not names else "lead label missing"
                    funnel[why] += 1
                    lead_rows.append([pid, g["id"], name, why, 0, "", "", "", ""])
                    continue
                mv, flag = series[1][name]
                w = waveform_score(np.load(root / row["digitized_path"]), float(row["signal_fs"]), series[0], mv)
                funnel["recovered"] += 1
                lead_rows.append([pid, g["id"], name, "recovered", round(w["score"], 1), round(w["corr"], 3), round(w["shift_s"], 2), round(w["amp_ratio"], 3), flag])

    rec = [r for r in lead_rows if r[3] == "recovered"]
    wave = float(np.mean([r[4] for r in rec])) if rec else 0.0
    e2e = float(np.mean([r[4] for r in lead_rows])) if lead_rows else 0.0
    prec, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 100 * 2 * prec * recall / max(prec + recall, 1e-9)
    summary = {"pages": n_pages, "orientation": 100 * n_orient / max(n_pages, 1), "panels_f1": f1, "panels_recall": 100 * recall, "panels_precision": 100 * prec,
               "panel_mean_iou": 100 * float(np.mean(ious)) if ious else 0.0, "gain": 100 * gain_ok / max(gain_n, 1), "labels": 100 * lab_ok / max(lab_n, 1),
               "waveform": wave, "end_to_end": e2e, "leads_expected": len(lead_rows), "funnel": dict(funnel)}
    (args.run / "score.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with (args.run / "score_leads.csv").open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["page", "panel", "lead", "status", "score", "corr", "shift_s", "amp_ratio", "flag"])
        w_.writerows(lead_rows)
    print("| Step | Score (0-100) | Counted over |")
    print("|---|---|---|")
    print(f"| Orientation | {summary['orientation']:.0f} | {n_pages} pages |")
    print(f"| Panels found (F1, IoU >= {args.iou}) | {f1:.0f} | recall {100 * recall:.0f}, precision {100 * prec:.0f}, mean IoU {summary['panel_mean_iou']:.0f} |")
    print(f"| Gain read | {summary['gain']:.0f} | {gain_n} matched panels |")
    print(f"| Lead labels | {summary['labels']:.0f} | {lab_n} matched panels |")
    print(f"| Waveform | {wave:.0f} | {len(rec)} recovered leads |")
    print(f"| End to end | {e2e:.0f} | {len(lead_rows)} expected leads |")
    print("leads lost at:", dict(funnel))


if __name__ == "__main__":
    main()
