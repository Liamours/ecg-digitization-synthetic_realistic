"""What the printed text says: gain, lead-label set, and the header and footer band that bounds the traces."""
import re


def read_gain(texts: list[dict], allowed: list[float]) -> float | None:
    """Gain printed as N mm/mV; only allowed values count (OCR reads 10mm/mV as 0mm/mV, and the pulse then decides)."""
    for t in texts:
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm/m", t["text"])
        if m and float(m.group(1)) in allowed:
            return float(m.group(1))
    return None


def read_labels(texts: list[dict], label_sets: list[list[str]], position: int) -> tuple[list[str], str]:
    """The label set owning the lead names read as whole text items (OCR confuses U with V and drops punctuation, and
    the header V1.02 must not count); with no name read or a tie between sets, the set for this panel's position."""
    words = {re.sub(r"[^A-Za-z0-9]", "", t["text"]).replace("U", "V").replace("u", "V") for t in texts}
    hits = [len(words & set(s)) for s in label_sets]
    if max(hits) >= 1 and hits.count(max(hits)) == 1:
        return label_sets[hits.index(max(hits))], "ocr"
    return label_sets[position % len(label_sets)], "position"


def fill_unread_sets(named: list[tuple[list[str], str]], label_sets: list[list[str]]) -> list[tuple[list[str], str]]:
    """Names for the panels of a page whose names were not read (source `position`), from the panels that were read.

    The device prints the lead sets in reading order, the first and the last set sometimes twice (I-II-III, aVR-aVL-aVF,
    V1-V2-V3, V4-V5-V6; or with repeats). An unread panel lies between the nearest read panel before it and after it,
    so its set lies between theirs; of those it takes the one no panel of the page has yet, else the first or last set
    (the ones the device repeats), else the set of the panel before it. The source becomes `order`."""
    k = len(label_sets)
    known = [label_sets.index(labels) if source != "position" and labels in label_sets else None for labels, source in named]
    out = list(known)
    for i, s in enumerate(known):
        if s is not None or named[i][1] != "position":
            continue
        prev = next((known[j] for j in range(i - 1, -1, -1) if known[j] is not None), 0)
        nxt = next((known[j] for j in range(i + 1, len(known)) if known[j] is not None), k - 1)
        cand = list(range(prev, max(prev, nxt) + 1))
        free = [c for c in cand if c not in out]
        out[i] = free[0] if free else 0 if 0 in cand else k - 1 if k - 1 in cand else prev
    return [(label_sets[out[i]], "order") if known[i] is None and named[i][1] == "position" else named[i] for i in range(len(named))]


def trace_zone(texts: list[dict], height: int, cfg: dict) -> tuple[int, int]:
    """Rows between the header text and the footer text; the margin keeps text out of the trace mask.

    A header word counts only in the upper half of the crop and a footer word only in the lower half: the device's second
    print format repeats `MAC 400` in its footer row, which once put the header below the footer and left no rows at all."""
    head = re.compile(cfg["header_regex"], re.I)
    foot = re.compile(cfg["footer_regex"], re.I)
    mid = lambda t: (t["bbox"][1] + t["bbox"][3]) / 2
    heads = [t["bbox"][3] for t in texts if head.search(t["text"]) and mid(t) < height / 2]
    foots = [t["bbox"][1] for t in texts if foot.search(t["text"]) and mid(t) > height / 2]
    return (int(max(heads) + cfg["margin_px"]) if heads else 0), (int(min(foots) - cfg["margin_px"]) if foots else height)

