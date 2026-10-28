"""Smoke test: one synthetic collage page (no data files needed) goes through the whole pipeline.

python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fake_source(n_s=60, fs=360, hr=90, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * fs)) / fs
    rr = 60 / hr
    beats = np.arange(0.3, n_s, rr)
    def wave(c, w, a):
        return a * np.exp(-0.5 * ((t[:, None] - (beats + c)[None, :]) / w) ** 2).sum(1)
    A = wave(-0.16, 0.025, 0.15) + wave(-0.02, 0.008, -0.1) + wave(0, 0.01, 1.2) + wave(0.025, 0.01, -0.3) + wave(0.25, 0.05, 0.3)
    B = -0.6 * A + wave(0.0, 0.012, 0.5) + 0.02 * rng.normal(size=len(t))
    return np.column_stack([A, B]).astype(np.float32)


def test_synthetic_page(tmp_path):
    import cv2
    from rapidocr_onnxruntime import RapidOCR
    from ecgdig.synth import make_page
    from ecgdig.page import digitize_page, save_page
    rng = np.random.default_rng(3)
    page, truth = make_page(rng, [fake_source()])
    p = tmp_path / "page.jpg"
    cv2.imwrite(str(p), page)
    res = digitize_page(RapidOCR(), str(p))
    assert res.page_type == "ecg_mac400"
    assert len(res.panels) == len(truth)
    save_page(res, tmp_path / "out", str(p))
    assert (tmp_path / "out" / "record.npz").exists()
    d = np.load(tmp_path / "out" / "record.npz")
    assert d["signal"].shape == (12, 2000)
    # the limb strip: digitized II against the truth
    gt = [t for t in truth if t["set"] == "limb"][0]
    pi = [i for i, info in enumerate(res.panels) if info["lead_set"] == "limb"]
    assert pi, "limb strip not named"
    r, _ = res._results[pi[0]]
    L = [l for l in r.leads if l.name == "II"][0]
    mv = np.array(gt["truth"]["II"]["mv"])
    pred = np.interp(np.arange(len(mv)) / 500, L.t, L.mv)
    n = min(len(mv), len(pred))
    best = max(np.corrcoef(mv[max(0, s):n + min(0, s)], pred[max(0, -s):n - max(0, s)])[0, 1] for s in range(-15, 16))
    assert best > 0.85
