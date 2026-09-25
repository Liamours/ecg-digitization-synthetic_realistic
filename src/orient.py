import re
import statistics as st


def rotate_box(b: list[float], w: int, h: int, k: int) -> list[float]:
    for _ in range(k // 90):
        b = [b[1], w - b[2], b[3], w - b[0]]
        w, h = h, w
    return b


def choose_rotation(texts: list[dict], w: int, h: int, cfg: dict) -> tuple[int, dict]:
    """Counter-clockwise degrees that make the page upright.

    Axis from text-box shape, direction from the header sitting above the footer.
    """
    boxes = [t["bbox"] for t in texts if len(t["text"]) >= cfg["min_text_len"]]
    tall = sum((b[3] - b[1]) > 1.5 * (b[2] - b[0]) for b in boxes)
    wide = sum((b[2] - b[0]) > 1.5 * (b[3] - b[1]) for b in boxes)
    cands = (90, 270) if tall > wide else (0, 180)
    head, foot = re.compile(cfg["head_regex"], re.I), re.compile(cfg["foot_regex"], re.I)
    heads = [t["bbox"] for t in texts if head.search(t["text"]) and "V1" not in t["text"]]
    foots = [t["bbox"] for t in texts if foot.search(t["text"])]

    def cue(k: int) -> float:
        if not heads or not foots:
            return 0.0
        return st.mean(rotate_box(b, w, h, k)[1] for b in foots) - st.mean(rotate_box(b, w, h, k)[1] for b in heads)

    scores = {k: cue(k) for k in cands}
    return max(cands, key=scores.get), scores
