"""Local coordinate map of a panel from its own printed grid, without resampling the image.

Two grid kinds: a dot lattice (MAC 400, Fukuda) fitted to the printed dots, and a solid-line grid (EDAN)
fitted to the direction and spacing of the horizontal and the vertical lines. Both return a Grid whose
axes a1 (along time) and a2 (across voltage) are pixel vectors of one grid step.
"""
import cv2
import numpy as np

from src.digitize.record import Grid

MAX_DOTS = 3000
ITERS = 8


def find_dots(gray: np.ndarray, kernel: int, contrast: int, area: tuple[int, int], max_side: int) -> np.ndarray:
    """Centroids of small dark blobs; traces and text are larger and get dropped."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    mask = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > contrast).astype(np.uint8)
    n, _, st, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = (st[1:, cv2.CC_STAT_AREA] >= area[0]) & (st[1:, cv2.CC_STAT_AREA] <= area[1]) & (np.maximum(st[1:, 2], st[1:, 3]) <= max_side)
    return cent[1:][keep]


def _best(dots: np.ndarray, angles: np.ndarray, pitches: np.ndarray) -> tuple[float, float]:
    best = (-1.0, 0.0, 0.0)
    for a in angles:
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        u = dots[:, 0] * c + dots[:, 1] * s
        v = -dots[:, 0] * s + dots[:, 1] * c
        score = np.abs(np.exp(2j * np.pi * u[None] / pitches[:, None]).sum(1)) + np.abs(np.exp(2j * np.pi * v[None] / pitches[:, None]).sum(1))
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), float(a), float(pitches[i]))
    return best[1], best[2]


def initial_angle_pitch(dots: np.ndarray, pitch_range: tuple[float, float], angle_range: float, rng: np.random.Generator) -> tuple[float, float]:
    """Coarse-to-fine search of the lattice angle and pitch on a subsample of the dots."""
    pts = dots if len(dots) <= MAX_DOTS else dots[rng.choice(len(dots), MAX_DOTS, replace=False)]
    ang, pitch = _best(pts, np.arange(-angle_range, angle_range + 1e-9, 0.2), np.arange(pitch_range[0], pitch_range[1], 0.1))
    return _best(pts, np.arange(ang - 0.25, ang + 0.2501, 0.02), np.arange(pitch - 0.15, pitch + 0.1501, 0.01))


def fit_dot_grid(gray: np.ndarray, cfg: dict) -> Grid:
    dots = find_dots(gray, cfg["blackhat_kernel"], cfg["blackhat_contrast"], tuple(cfg["dot_area"]), cfg["dot_max_side"])
    if len(dots) < 200:
        raise ValueError(f"too few grid dots found ({len(dots)})")
    ang, pitch = initial_angle_pitch(dots, tuple(cfg["pitch_px_range"]), cfg["angle_range_deg"], np.random.default_rng(42))
    r = np.radians(ang)
    a1 = pitch * np.array([np.cos(r), np.sin(r)])
    a2 = pitch * np.array([-np.sin(r), np.cos(r)])
    o = dots[np.argmin(np.hypot(*(dots - dots.mean(0)).T))]
    tol = 0.22
    for _ in range(ITERS):
        idx = np.linalg.solve(np.stack([a1, a2], axis=1), (dots - o).T).T
        near = np.round(idx)
        inl = np.hypot(*(idx - near).T) < tol
        sol, *_ = np.linalg.lstsq(np.column_stack([np.ones(inl.sum()), near[inl]]), dots[inl], rcond=None)
        o, a1, a2 = sol[0], sol[1], sol[2]
        tol = max(0.12, tol * 0.85)
    A = np.stack([a1, a2], axis=1)
    idx = np.linalg.solve(A, (dots - o).T).T
    res = np.hypot(*((idx - np.round(idx)) @ A.T).T)
    inl = res < 1.5
    return Grid("dot", o, a1, a2, float(np.sqrt((res[inl] ** 2).mean())), int(inl.sum()))


def _lines_only(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return np.clip(cv2.GaussianBlur(g, (0, 0), 25) - g, 0, None)


def _profile_at(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return r[h // 8:-h // 8, w // 8:-w // 8].mean(axis=1)


def _axis_angle(img: np.ndarray, angle_range: float) -> float:
    coarse = max(np.arange(-angle_range, angle_range + 1e-9, 0.1), key=lambda a: _profile_at(img, a).var())
    return float(max(np.arange(coarse - 0.1, coarse + 0.101, 0.02), key=lambda a: _profile_at(img, a).var()))


def _spacing(profile: np.ndarray, lo: int, hi: int) -> float:
    p = profile - profile.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac /= ac[0]
    seg = ac[lo:hi]
    i = int(np.argmax(seg))
    y0, y1, y2 = seg[max(i - 1, 0)], seg[i], seg[min(i + 1, len(seg) - 1)]
    return lo + i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)


def fit_line_grid(gray: np.ndarray, cfg: dict) -> Grid:
    """Direction and spacing of the horizontal and vertical grid lines from their projection profiles."""
    img = _lines_only(gray)
    ah = _axis_angle(img, cfg["angle_range_deg"])
    imt = np.ascontiguousarray(img.T)
    av = _axis_angle(imt, cfg["angle_range_deg"])
    lo, hi = cfg["spacing_px_range"]
    py = _spacing(_profile_at(img, ah), lo, hi)
    px = _spacing(_profile_at(imt, av), lo, hi)
    # av comes from the transposed image, so its sense is mirrored against ah (found with the Einthoven check on EDAN)
    a1 = px * np.array([np.cos(np.radians(-ah)), np.sin(np.radians(-ah))])
    a2 = py * np.array([np.sin(np.radians(av)), np.cos(np.radians(av))])
    return Grid("line", np.zeros(2), a1, a2)


def fit_grid(gray: np.ndarray, cfg: dict) -> Grid:
    return fit_dot_grid(gray, cfg) if cfg["kind"] == "dot" else fit_line_grid(gray, cfg)
