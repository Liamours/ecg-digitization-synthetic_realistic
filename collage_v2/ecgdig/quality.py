"""Quality numbers per lead and the Einthoven/Goldberger checks a panel can do on itself.

The three leads of a MAC 400 strip are recorded at the same moment, so on a limb strip II = I + III and on an augmented
strip aVR + aVL + aVF = 0 must hold sample by sample. How well the digitized leads satisfy that is an accuracy check that
needs no ground truth: a lead that jumped to its neighbour or lost its peaks breaks the identity.
"""
from __future__ import annotations

import numpy as np

FLAT_P2P_MV = 0.08


def _robust_p2p(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return 0.0
    return float(np.percentile(x, 99.5) - np.percentile(x, 0.5))


def lead_metrics(L, res, s: float) -> dict:
    own = L.own
    n = len(own)
    if n == 0 or own.sum() < 5:
        return {"status": "missing", "coverage": 0.0, "score": -9.0}
    cov = float(own.mean())
    shared = float(L.shared[own].mean()) if own.any() else 0.0
    # a jump: consecutive pixel runs on the path that do not touch and are more than 2 mm apart (the path left its line)
    idx = np.flatnonzero(own)
    lo, hi = L.lo[idx], L.hi[idx]
    gaps = np.maximum(0, np.maximum(lo[1:] - hi[:-1], lo[:-1] - hi[1:])) / s if idx.size > 1 else np.zeros(1)
    jump = float((gaps > 2.0).sum()) / max(1.0, n / s / 25.0)   # per second
    p2p = _robust_p2p(L.mv)
    sat = float((np.abs(L.mv) > 5.0).mean())
    status = "ok"
    if p2p < FLAT_P2P_MV or L.name in getattr(res, "leadoff", []):
        status = "flat"
    elif cov < 0.85 or shared > 0.30 or jump > 1.0 or p2p > 8.0:
        status = "low_quality"
    score = cov - shared - 0.2 * jump - 0.5 * sat
    return {"status": status, "coverage": round(cov, 3), "shared": round(shared, 3), "jumps_per_s": round(jump, 2), "p2p_mV": round(p2p, 3), "score": round(score, 3)}


def _rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def einthoven_limb(leads) -> dict:
    I, II, III = (l.mv - np.nanmedian(l.mv) for l in leads)
    e = II - (I + III)
    e = e - np.nanmedian(e)
    ok = np.isfinite(e)
    if ok.sum() < 20:
        return {}
    c = float(np.corrcoef(II[ok], (I + III)[ok])[0, 1])
    return {"identity": "II = I + III", "corr": round(c, 3), "residual_rms_mV": round(_rms(e), 3), "rel_residual": round(_rms(e) / max(_rms(II), 1e-6), 3)}


def einthoven_aug(leads) -> dict:
    R, L_, F = (l.mv - np.nanmedian(l.mv) for l in leads)
    e = R + L_ + F
    e = e - np.nanmedian(e)
    ok = np.isfinite(e)
    if ok.sum() < 20:
        return {}
    ref = np.mean([_rms(R), _rms(L_), _rms(F)])
    c = float(np.corrcoef(R[ok], -(L_ + F)[ok])[0, 1])
    return {"identity": "aVR + aVL + aVF = 0", "corr": round(c, 3), "residual_rms_mV": round(_rms(e), 3), "rel_residual": round(_rms(e) / max(ref, 1e-6), 3)}
