"""Digitize synthetic pages and score them against their known signals.

python tools/eval_synth.py --synth DIR [--limit N] [--out results.json]

Per lead: Pearson r and SNR in dB after removing each signal's median and allowing a shift of up to 40 ms (the scoring
of the PhysioNet Challenge 2024 also allows a vertical offset and a time shift), RMSE in mV. Per strip: lead set and
gain read correctly.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidocr_onnxruntime import RapidOCR
from ecgdig.page import digitize_page


def score(gt, pred, fs=500, max_shift=20):
    n = min(len(gt), len(pred))
    best = None
    for sh in range(-max_shift, max_shift + 1):
        a = gt[max(0, sh):n + min(0, sh)]
        b = pred[max(0, -sh):n - max(0, sh)]
        m = min(len(a), len(b)); a, b = a[:m], b[:m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < fs * 0.5:
            continue
        a, b = a[ok] - np.median(a[ok]), b[ok] - np.median(b[ok])
        off = np.mean(a - b); b = b + off
        err = a - b
        snr = 10 * np.log10(np.sum(a ** 2) / max(np.sum(err ** 2), 1e-12))
        if best is None or snr > best[0]:
            r = float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 0 and np.std(a) > 0 else 0.0
            best = (float(snr), r, float(np.sqrt(np.mean(err ** 2))), sh / fs)
    return best


ap = argparse.ArgumentParser(); ap.add_argument("--synth", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="")
a = ap.parse_args()
ocr = RapidOCR()
rows, panel_rows = [], []
pages = sorted(Path(a.synth).glob("synth_*.jpg"))
if a.limit:
    pages = pages[:a.limit]
for p in pages:
    truth = json.loads(p.with_suffix(".json").read_text())
    res = digitize_page(ocr, str(p), overlay=False)
    for gt in truth:
        gx, gy = gt["marker_center_page"]
        best, bd = None, 1e9
        for pi, info in enumerate(res.panels):
            x, y, w, h = info["marker_page_xywh"]
            d = np.hypot(x + w / 2 - gx, y + h / 2 - gy)
            if d < bd:
                best, bd = pi, d
        if best is None or bd > 30:
            panel_rows.append({"page": p.name, "set": gt["set"], "found": False})
            for nm in gt["names"]:
                rows.append({"page": p.name, "lead": nm, "found": False})
            continue
        info = res.panels[best]
        pres, _ = res._results[best]
        panel_rows.append({"page": p.name, "set": gt["set"], "found": True, "set_pred": info["lead_set"], "gain": gt["gain"], "gain_pred": info["gain_mm_per_mV"],
                           "turned": gt["turned"], "einthoven": info.get("einthoven")})
        for slot, nm in enumerate(gt["names"]):
            mv = np.array(gt["truth"][nm]["mv"])
            if slot >= len(pres.leads):
                rows.append({"page": p.name, "lead": nm, "found": False}); continue
            L = pres.leads[slot]
            pred = np.interp(np.arange(len(mv)) / 500.0, L.t, L.mv, left=np.nan, right=np.nan)
            sc = score(mv, pred)
            rows.append({"page": p.name, "lead": nm, "set": gt["set"], "found": True, "status": L.metrics.get("status"), "leadoff": gt.get("leadoff") == nm,
                         "snr_db": sc[0] if sc else None, "r": sc[1] if sc else None, "rmse_mv": sc[2] if sc else None, "shift_s": sc[3] if sc else None,
                         "p2p_true": float(np.ptp(mv))})
    print(p.name, round(res.seconds, 1), flush=True)

found = [r for r in rows if r.get("found") and r.get("snr_db") is not None and not r.get("leadoff")]
snr = np.array([r["snr_db"] for r in found]); rr = np.array([r["r"] for r in found]); rm = np.array([r["rmse_mv"] for r in found])
okm = np.array([r["status"] == "ok" for r in found])
pf = [p for p in panel_rows if p["found"]]
summary = {
    "pages": len(pages), "strips": len(panel_rows), "strips_found": len(pf),
    "lead_set_correct": float(np.mean([p["set_pred"] == p["set"] for p in pf])) if pf else None,
    "lead_set_unread": float(np.mean([p["set_pred"] is None for p in pf])) if pf else None,
    "gain_correct": float(np.mean([p["gain_pred"] == p["gain"] for p in pf])) if pf else None,
    "leads_scored": len(found), "snr_db_median": float(np.median(snr)), "snr_db_mean": float(np.mean(snr)),
    "r_median": float(np.median(rr)), "r_mean": float(np.mean(rr)), "rmse_mv_median": float(np.median(rm)),
    "frac_r_above_0.9": float(np.mean(rr > 0.9)), "frac_flagged_ok": float(okm.mean()),
    "r_median_flagged_ok": float(np.median(rr[okm])) if okm.any() else None,
    "r_median_flagged_not_ok": float(np.median(rr[~okm])) if (~okm).any() else None,
    "snr_median_flagged_ok": float(np.median(snr[okm])) if okm.any() else None,
}
print(json.dumps(summary, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps({"summary": summary, "leads": rows, "strips": panel_rows}, indent=1))
