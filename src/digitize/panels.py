import re
import statistics as st


def find_panels(texts: list[tuple[str, list[float]]], width: int, height: int, cfg: dict) -> list[dict]:
    """Panel rectangles from footer anchors; column width and row height come from footer-to-footer spacing.

    texts are (text, bbox) in the upright frame. kind is ecg, report, or other; isolated marks a panel with no neighbor.
    """
    foot, report, ecg = (re.compile(cfg[k], re.I) for k in ("foot_regex", "report_regex", "ecg_regex"))
    foots = sorted((b for t, b in texts if foot.search(t)), key=lambda b: (b[3], b[0]))
    if not foots:
        return []
    rows: list[list[list[float]]] = []
    for f in foots:
        if rows and f[3] - st.mean(x[3] for x in rows[-1]) < cfg["row_gap_frac"] * height:
            rows[-1].append(f)
        else:
            rows.append([f])
    for r in rows:
        r.sort(key=lambda b: b[0])
    dx = [r[i + 1][0] - r[i][0] for r in rows for i in range(len(r) - 1)]
    col_w = st.median(dx) if dx else cfg["lone_width_ratio"] * st.median(f[2] - f[0] for f in foots)
    bottoms = [st.mean(f[3] for f in r) + 0.02 * col_w for r in rows]
    lo, hi = cfg["spacing_range"]
    fits = lambda g: lo * col_w <= g <= hi * col_w
    dy = [g for g in (bottoms[i + 1] - bottoms[i] for i in range(len(bottoms) - 1)) if fits(g)]
    row_h = st.median(dy) if dy else cfg["lone_height_ratio"] * col_w
    panels = []
    for ri, r in enumerate(rows):
        above = ri > 0 and fits(bottoms[ri] - bottoms[ri - 1])
        below = ri + 1 < len(rows) and fits(bottoms[ri + 1] - bottoms[ri])
        for i, f in enumerate(r):
            off = cfg["left_offset_frac"] * col_w  # the footer text starts 2.3 to 2.6 boxes into a 20 box panel (measured on the reference crops)
            left = max(f[0] - off, 0)
            right = min(r[i + 1][0] - cfg["right_gap_frac"] * col_w if i + 1 < len(r) else left + col_w, width)  # a smaller gap than the left offset keeps trace ends on real photos
            top = max(bottoms[ri - 1] if above else bottoms[ri] - row_h, 0)
            bottom = min(bottoms[ri], height)
            inside = [t for t, b in texts if left <= (b[0] + b[2]) / 2 <= right and top <= (b[1] + b[3]) / 2 <= bottom]
            if any(report.search(t) for t in inside):
                kind = "report"
            elif sum(bool(ecg.search(t)) for t in inside) >= cfg["ecg_min_hits"]:
                kind = "ecg"
            else:
                kind = "other"
            panels.append({"box": [round(left), round(top), round(right), round(bottom)], "kind": kind, "isolated": len(r) == 1 and not above and not below})
    return panels


def iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
