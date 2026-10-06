"""Local coordinate map for a page printed with a solid-line grid (EDAN): the direction
and spacing of the horizontal and the vertical grid lines, each found from its own
projection profile, so a small shear or foreshortening shows up as a difference
between the two axes. Returns the same grid dict as grid_map.fit_grid."""
import cv2
import numpy as np

ANGLES_COARSE = np.arange(-4, 4.01, 0.1)


def _lines_only(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return np.clip(cv2.GaussianBlur(g, (0, 0), 25) - g, 0, None)  # dark thin structures, no paper shading


def _profile_at(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return r[h // 8:-h // 8, w // 8:-w // 8].mean(axis=1)


def axis_angle(img: np.ndarray) -> float:
    coarse = max(ANGLES_COARSE, key=lambda a: _profile_at(img, a).var())
    fine = np.arange(coarse - 0.1, coarse + 0.101, 0.02)
    return float(max(fine, key=lambda a: _profile_at(img, a).var()))


def spacing(profile: np.ndarray, lo: int = 5, hi: int = 26) -> float:
    p = profile - profile.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac /= ac[0]
    seg = ac[lo:hi]
    i = int(np.argmax(seg))
    y0, y1, y2 = seg[max(i - 1, 0)], seg[i], seg[min(i + 1, len(seg) - 1)]
    return lo + i + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)


def fit_line_grid(gray: np.ndarray) -> dict:
    img = _lines_only(gray)
    ah = axis_angle(img)                       # tilt of the horizontal lines
    av = axis_angle(np.ascontiguousarray(img.T))   # tilt of the vertical lines, measured on the transposed image
    py = spacing(_profile_at(img, ah))
    px = spacing(_profile_at(np.ascontiguousarray(img.T), av))
    # an angle from cv2.getRotationMatrix2D is counterclockwise in image coordinates; a1 follows the horizontal lines, a2 the vertical lines
    a1 = px * np.array([np.cos(np.radians(-ah)), np.sin(np.radians(-ah))])
    a2 = py * np.array([np.sin(np.radians(av)), np.cos(np.radians(av))])  # av comes from the transposed image, so its sense is mirrored against ah; the sign was found with the Einthoven check
    return {"o": np.array([0.0, 0.0]), "a1": a1, "a2": a2, "n_dots": 0, "n_inliers": 0, "rms_px": float("nan"),
            "tilt_h_deg": ah, "tilt_v_deg": av, "px_per_mm_x": px, "px_per_mm_y": py}
