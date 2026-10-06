"""Local coordinate map of one panel from its own dot grid: pixels to millimetres,
without resampling the image. Fits p = o + m * a1 + n * a2 to the printed dots,
where (m, n) are integer millimetre indices and a1, a2 are the pixel vectors of
one millimetre along the grid's two axes."""
import cv2
import numpy as np

DOT_AREA = (4, 45)
KERNEL = 9


def find_dots(gray: np.ndarray) -> np.ndarray:
    """Centroids of small dark blobs (grid dots); traces and text are larger and get dropped."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    mask = (blackhat > 28).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if DOT_AREA[0] <= stats[i, cv2.CC_STAT_AREA] <= DOT_AREA[1]
            and max(stats[i, 2], stats[i, 3]) <= 9]
    return cent[keep]


def _initial(dots: np.ndarray):
    """Angle and one-millimetre pitch from the dot point set: the strongest periodicity of x and y projections."""
    best = None
    for ang in np.arange(-6, 6.01, 0.1):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        u = dots[:, 0] * c + dots[:, 1] * s
        v = -dots[:, 0] * s + dots[:, 1] * c
        for pitch in np.arange(9.0, 15.0, 0.05):
            score = abs(np.exp(2j * np.pi * u / pitch).sum()) + abs(np.exp(2j * np.pi * v / pitch).sum())
            if best is None or score > best[0]:
                best = (score, ang, pitch)
    return best[1], best[2]


def fit_grid(gray: np.ndarray, iters: int = 8):
    dots = find_dots(gray)
    ang, pitch = _initial(dots)
    r = np.radians(ang)
    a1 = pitch * np.array([np.cos(r), np.sin(r)])
    a2 = pitch * np.array([-np.sin(r), np.cos(r)])
    o = dots[np.argmin(np.hypot(*(dots - dots.mean(0)).T))]
    tol = 0.22
    for _ in range(iters):
        A = np.stack([a1, a2], axis=1)
        idx = np.linalg.solve(A, (dots - o).T).T
        near = np.round(idx)
        res = np.hypot(*(idx - near).T)
        inl = res < tol
        M = np.column_stack([np.ones(inl.sum()), near[inl]])
        sol, *_ = np.linalg.lstsq(M, dots[inl], rcond=None)
        o, a1, a2 = sol[0], sol[1], sol[2]
        tol = max(0.12, tol * 0.85)
    A = np.stack([a1, a2], axis=1)
    idx = np.linalg.solve(A, (dots - o).T).T
    res_px = np.hypot(*((idx - np.round(idx)) @ A.T).T)
    inl = res_px < 1.5
    return {"o": o, "a1": a1, "a2": a2, "n_dots": len(dots), "n_inliers": int(inl.sum()), "rms_px": float(np.sqrt((res_px[inl] ** 2).mean()))}


def to_mm(grid: dict, pts: np.ndarray) -> np.ndarray:
    A = np.stack([grid["a1"], grid["a2"]], axis=1)
    return np.linalg.solve(A, (pts - grid["o"]).T).T
