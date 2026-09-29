"""One lead's trace pixels to millivolts over time, sampled in the panel's own grid millimetres."""
import numpy as np

from src.digitize.record import Grid


def sample_lead(mask: np.ndarray, grid: Grid, gain: float, x_range: tuple[float, float], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Per time bin, the trace point farthest from the local median (peaks keep their full height); gaps are interpolated.

    The bin is at least `min_bin_px` pixels wide (a thin trace leaves gaps between bins narrower than a pixel), so coverage
    is comparable across resolutions. Returns time in seconds, mV, a mask of bins that had trace pixels, the baseline in
    grid millimetres, and the bin width in millimetres.
    """
    ys, xs = np.nonzero(mask)
    mm = grid.to_mm(np.column_stack([xs, ys]).astype(np.float64))
    x_mm, y_mm = mm[:, 0], mm[:, 1]
    bin_mm = max(cfg["bin_mm"], cfg["min_bin_px"] / grid.px_per_mm[0])
    edges = np.arange(x_range[0], x_range[1] + bin_mm, bin_mm)
    n = len(edges) - 1
    which = np.digitize(x_mm, edges) - 1
    med = np.full(n, np.nan)
    for b in np.unique(which):
        if 0 <= b < n:
            med[b] = np.median(y_mm[which == b])
    half = cfg["baseline_window_bins"]
    local = np.array([np.nanmedian(med[max(0, i - half):i + half]) if np.isfinite(med[max(0, i - half):i + half]).any() else np.nan for i in range(n)])
    out = np.full(n, np.nan)
    for b in np.unique(which):
        if 0 <= b < n:
            v = y_mm[which == b]
            out[b] = v[np.argmax(np.abs(v - local[b]))]
    ok = np.isfinite(out)
    if not ok.any():
        raise ValueError("no trace pixels in this lead")
    base = float(np.nanmedian(out))
    sig = np.interp(np.arange(n), np.flatnonzero(ok), out[ok])
    t = np.arange(n) * bin_mm / cfg["mm_per_s"]
    return t, -(sig - base) / gain, ok, base, bin_mm
