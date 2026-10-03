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


def trace_zone(texts: list[dict], height: int, cfg: dict) -> tuple[int, int]:
    """Rows between the header text and the footer text; the margin keeps text out of the trace mask."""
    head = re.compile(cfg["header_regex"], re.I)
    foot = re.compile(cfg["footer_regex"], re.I)
    heads = [t["bbox"][3] for t in texts if head.search(t["text"])]
    foots = [t["bbox"][1] for t in texts if foot.search(t["text"])]
    return (int(max(heads) + cfg["margin_px"]) if heads else 0), (int(min(foots) - cfg["margin_px"]) if foots else height)

