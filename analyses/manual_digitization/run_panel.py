"""Digitize one panel crop from results/inferences/manual_digitization_trial/inputs (crops and OCR boxes saved there):
    python run_panel.py <panel index> <comma separated lead labels>
"""
import json, sys
import numpy as np, cv2
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from digitize_panel import digitize
S = Path(__file__).resolve().parents[3] / "results" / "inferences" / "manual_digitization_trial" / "inputs"
def run(i, labels):
    ocr = json.load(open(S / "ocr.json"))
    res = digitize(S / f"panel{i}.png", ocr[str(i)], labels)
    g = res["grid"]
    print(f"panel {i}: gain {res['gain']} mm/mV | grid rms {g['rms_px']:.2f}px tilt {np.degrees(np.arctan2(g['a1'][1], g['a1'][0])):.2f} deg {np.hypot(*g['a1']):.2f} px/mm | pulses (mm): {[round(p['height_mm'], 2) for p in res['pulses']]}")
    img = cv2.cvtColor(res["gray"], cv2.COLOR_GRAY2BGR); A = np.stack([g["a1"], g["a2"]], axis=1)
    for k, v in res["leads"].items():
        px = np.array([g["o"] + A @ np.array([x, y]) for x, y in v["pts_mm"]]).astype(int)
        for a, b in zip(px[:-1], px[1:]): cv2.line(img, tuple(a), tuple(b), (0, 160, 0), 1)
        print(f"  {k}: band {v['band']} coverage {100*v['coverage']:.0f}% p2p {np.percentile(v['mv'],99.5)-np.percentile(v['mv'],0.5):.2f} mV duration {v['t'][-1]:.2f} s")
    cv2.imwrite(str(S / f"panel{i}_overlay.png"), img)
    fig, ax = plt.subplots(len(labels), 1, figsize=(11, 2.4 * len(labels)), sharex=True)
    for a, (k, v) in zip(ax, res["leads"].items()): a.plot(v["t"], v["mv"], lw=0.8); a.set_ylabel(k + " (mV)"); a.grid(True, alpha=0.3)
    ax[-1].set_xlabel("time (s)"); plt.tight_layout(); plt.savefig(S / f"panel{i}_signals.png", dpi=80); plt.close()
    return res
if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2].split(","))
