"""Build ecgdig/glyph_templates.npz from glyphs collected by tools/collect_glyphs.py and a k-means labelling.

The cluster-to-character map below was set by looking at the cluster mosaic once (see README, "Lead names").
Digits 3 and the letter a look alike after normalisation, so position in the label decides: a is the first glyph of a
three-glyph label, 3 the second glyph of a two-glyph label.
"""
import sys
import numpy as np
from pathlib import Path

d = np.load(sys.argv[1]); V, A, M = d["v"], d["a"], d["meta"]
L = np.load(sys.argv[2])
CL = {0: "6", 3: "I", 4: "L", 5: "V", 6: "F", 7: "4", 10: "1", 11: "I", 12: "2", 13: "V", 14: "V", 15: "5", 17: "R", 1: "V"}
names, bmps, asp, hgt = [], [], [], []
chars = {}
for i, m in enumerate(M):
    _, pi, slot, gi, n, w, h = m.split("|")
    gi, n, h = int(gi), int(n), float(h)
    c = CL.get(int(L[i]))
    if int(L[i]) in (16, 0, 1):
        if n == 3 and gi == 0 and h < 2.15:
            c = "a"
        elif n == 2 and gi == 1 and h >= 2.15 and int(L[i]) == 16:
            c = "3"
        elif int(L[i]) == 0 and h >= 2.15:
            c = "6"
        elif int(L[i]) == 1 and gi == 0:
            c = "V"
        else:
            c = None
    if c:
        chars.setdefault(c, []).append(i)
for c, idx in sorted(chars.items()):
    X = V[idx]
    mu = X.mean(0)
    r = X @ (mu / np.linalg.norm(mu))
    keep = np.array(idx)[r >= np.percentile(r, 30)]
    mu = V[keep].mean(0)
    mu = mu - mu.mean(); mu /= np.linalg.norm(mu)
    names.append(c); bmps.append(mu); asp.append(float(np.median(A[keep])))
    hgt.append(float(np.median([float(M[i].split("|")[6]) for i in keep])))
    print(c, len(idx), "kept", len(keep), "aspect", round(asp[-1], 2), "h", round(hgt[-1], 2))
out = Path(__file__).resolve().parents[1] / "ecgdig" / "glyph_templates.npz"
np.savez(out, names=np.array(names), bmps=np.array(bmps), aspect=np.array(asp), height_mm=np.array(hgt))
print("wrote", out)
