"""Digitize the four panels of the trial page twice, with the threshold trace mask and with the
Open-ECG-Digitizer U-Net mask (probability above OE_THRESHOLD), and print coverage and quality per lead.
Run from results/analyses/manual_digitization with the digitization repo venv (the U-Net mask is read from a saved page probability image)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from digitize_panel import PANEL_LABELS, digitize

ROOT = Path(__file__).resolve().parents[3]
OE_THRESHOLD = 60  # of 255; the U-Net's signal probability peaks near 0.7 on this page
inputs = ROOT / "results" / "inferences" / "manual_digitization_trial" / "inputs"
prob = cv2.imread(sys.argv[1], 0)
meta = {m["panel"]: m for m in json.load(open(inputs / "panels_meta.json"))}
ocr = json.load(open(inputs / "ocr.json"))
for i, labels in PANEL_LABELS.items():
    m = meta[i]
    ox, oy = m["crop_origin"]
    crop_prob = prob[oy:oy + cv2.imread(str(inputs / f"panel{i}.png")).shape[0], ox:ox + cv2.imread(str(inputs / f"panel{i}.png")).shape[1]]
    lim = (m["box"][0] - ox, m["box"][2] - ox)
    a = digitize(inputs / f"panel{i}.png", ocr[str(i)], labels, x_limits=lim)
    b = digitize(inputs / f"panel{i}.png", ocr[str(i)], labels, x_limits=lim, mask_override=(crop_prob > OE_THRESHOLD))
    for k in labels:
        f = lambda r: (round(r["leads"][k]["coverage"], 2), round(float(np.ptp(np.percentile(r["leads"][k]["mv"], [0.5, 99.5]))), 2), round(r["leads"][k]["crossing_fraction"], 2))
        print(f"panel {i} {k:3} threshold cov/p2p/cross {f(a)} | U-Net {f(b)}")
