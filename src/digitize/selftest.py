"""Label-free checks that need no ground-truth signal: calibration pulse, Einthoven's law, time lag, coverage.

Einthoven (best_lag_bins: samples by which the derived lead must be rolled to match the measured one): II = I + III and aVR + aVL + aVF = 0 hold sample by sample when the leads were recorded together, and
aVR = -(I + II) / 2, aVL = I - II / 2, aVF = II - I / 2. A row-to-row time offset (a wrong coordinate map) shows as a
best correlation at a non-zero lag, so the lag search doubles as a self-test of the grid map.
"""
import numpy as np
from scipy.ndimage import median_filter

from src.digitize.record import Lead

DETREND_BINS = 125


def _detrend(x: np.ndarray) -> np.ndarray:
    return x - median_filter(x, size=DETREND_BINS, mode="nearest")


def _compare(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict:
    lags = {k: float(np.corrcoef(np.roll(x, k), y)[0, 1]) for k in range(-max_lag, max_lag + 1)}
    best = max(lags, key=lags.get)
    return {"corr": round(lags[0], 3), "best_lag_bins": best, "corr_at_best_lag": round(lags[best], 3),
            "rms_residual_mV": round(float(np.sqrt(((x - y) ** 2).mean())), 3), "reference_p2p_mV": round(float(np.ptp(y)), 3)}


def check_leads(leads: dict[str, Lead], max_lag: int) -> dict:
    """Every Einthoven relation the present leads allow."""
    n = min((len(v.mv) for v in leads.values()), default=0)
    d = {k: _detrend(v.mv[:n]) for k, v in leads.items()}
    out = {}
    if all(k in d for k in ("I", "II", "III")):
        out["II=I+III"] = _compare(d["I"] + d["III"], d["II"], max_lag)
    if all(k in d for k in ("aVR", "aVL", "aVF")):
        s = d["aVR"] + d["aVL"] + d["aVF"]
        out["aVR+aVL+aVF=0"] = {"rms_sum_mV": round(float(np.sqrt((s ** 2).mean())), 3), "mean_lead_std_mV": round(float(np.mean([d[k].std() for k in ("aVR", "aVL", "aVF")])), 3)}
    if all(k in d for k in ("I", "II")):
        for k, ref in (("aVR", -(d.get("I", 0) + d.get("II", 0)) / 2), ("aVL", d.get("I", 0) - d.get("II", 0) / 2), ("aVF", d.get("II", 0) - d.get("I", 0) / 2)):
            if k in d:
                out[f"{k} from I,II"] = _compare(ref, d[k], max_lag)
    return out


def pulse_check(pulses_mm: list[float], gain: float | None) -> dict:
    if not pulses_mm or gain is None:
        return {}
    h = float(np.median(pulses_mm))
    return {"pulse_mm": round(h, 2), "pulse_mV": round(h / gain, 3)}


def flag_leads(leads: dict[str, Lead], cfg: dict) -> None:
    for v in leads.values():
        bad = v.coverage < cfg["coverage_min"] or v.p2p_mv > cfg["p2p_max_mv"]
        v.flag = "low_quality" if bad else "ok"


def confidence(leads: dict[str, Lead], pulse: dict, checks: dict) -> float:
    """Product of the mean coverage, the mean share of crossing-free columns, the pulse agreement, and the best Einthoven correlation."""
    if not leads:
        return 0.0
    cov = float(np.mean([v.coverage for v in leads.values()]))
    clean = 1.0 - float(np.mean([v.crossing_fraction for v in leads.values()]))
    pulse_factor = 1.0 if not pulse else float(np.clip(1.0 - abs(pulse["pulse_mV"] - 1.0), 0.0, 1.0))
    corr = [c["corr_at_best_lag"] for c in checks.values() if "corr_at_best_lag" in c]
    return cov * clean * pulse_factor * (float(np.clip(min(corr), 0.0, 1.0)) if corr else 1.0)
