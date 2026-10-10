"""Collect label glyphs from many pages and cluster them, so the glyph classes can be named once by eye.

python tools/collect_glyphs.py --out glyphs.npz PAGE.jpg ...
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidocr_onnxruntime import RapidOCR
from ecgdig.page import digitize_page
from ecgdig.glyphs import norm_glyph

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("pages", nargs="+")
a = ap.parse_args()
ocr = RapidOCR()
V, A, M = [], [], []
for p in a.pages:
    r = digitize_page(ocr, p, overlay=False)
    print(p, r.page_type, len(r.panels), round(r.seconds, 1), flush=True)
    for pi, rows in enumerate(r.glyphs):
        for slot, chain in enumerate(rows):
            for gi, g in enumerate(chain):
                V.append(norm_glyph(g["bmp"]).ravel()); A.append(g["w_mm"] / g["h_mm"]); M.append(f"{Path(p).name}|{pi}|{slot}|{gi}|{len(chain)}|{g['w_mm']:.2f}|{g['h_mm']:.2f}")
np.savez(a.out, v=np.array(V), a=np.array(A), meta=np.array(M))
print(len(V), "glyphs")
