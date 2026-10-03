"""Glue: one page to a list of PanelRecords.

Two layout modes. `panel`: the page holds several MAC 400 style panels found from their printed footers; each panel gets its
own grid map, gain, and label set. `page`: the page is one sheet (Fukuda, EDAN) whose regions are given as fractions of the
image in the layout file; the page must be upright.
"""
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

from src import paths
from src.digitize import leads as leadmod
from src.digitize import mask as maskmod
from src.digitize import ocr, pulses, selftest, text
from src.digitize.gridmap import fit_grid
from src.digitize.record import Grid, Lead, PanelRecord
from src.orient import choose_rotation
from src.panels import find_panels, iou

ROOT = Path(__file__).resolve().parents[2]


def load_layout(name_or_path: str) -> dict:
    p = Path(name_or_path)
    p = p if p.exists() else ROOT / "configs" / "layouts" / f"{name_or_path}.yml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_image(path: Path) -> np.ndarray:
    """Upright BGR image (EXIF orientation applied)."""
    return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(path).convert("RGB"))), cv2.COLOR_RGB2BGR)


def fit_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Scale a page down once when its long side exceeds `max_side`: the grid finder expects dots about 9 to 15 px apart (a 300 dpi scan), and an 870 dpi scan has them 35 px apart."""
    s = min(1.0, max_side / max(image.shape[:2]))
    return (cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image), s


def cut_at_gap(mask: np.ndarray, gap_px: int) -> np.ndarray:
    """Keep the run of trace columns from the left; an empty stretch of `gap_px` columns past 40 percent of the width ends this panel's own trace."""
    occ = mask.any(axis=0)
    if not occ.any():
        raise ValueError("no trace pixels left in the mask")
    empty = 0
    for x in range(int(np.flatnonzero(occ)[0]), mask.shape[1]):
        empty = 0 if occ[x] else empty + 1
        if empty >= gap_px and x > mask.shape[1] * 0.4:
            mask = mask.copy()
            mask[:, x:] = 0
            break
    return mask


class Digitizer:
    def __init__(self, cfg: dict, layout: dict):
        self.cfg, self.layout = cfg, layout
        self._unet = None

    @property
    def unet(self) -> maskmod.UNetMask:
        if self._unet is None:
            self._unet = maskmod.UNetMask({**self.cfg["unet"], "repo": paths.resolve(self.cfg["unet"]["repo"]), "device": self.cfg["device"]})
        return self._unet

    # ---- shared -------------------------------------------------------------------------------------------------
    def digitize_region(self, mask: np.ndarray, grid: Grid, gain: float, labels: list[str], rows: int, rows_mode: str) -> tuple[dict[str, Lead], float]:
        """Row assignment, sampling, and lead flags for one region's trace mask."""
        bands = leadmod.split_rows(mask, rows, rows_mode)
        track = leadmod.track_leads_overlap if self.layout.get("tracking") == "overlap" else leadmod.track_leads
        lead_masks, crossing = track(mask, bands)
        trim = self.cfg["trim"]
        lead_masks = leadmod.trim_early_starts(lead_masks, grid, trim["early_start_mm"], trim["margin_mm"], trim.get("late_end_mm"))
        ys, xs = np.nonzero(np.any(lead_masks, axis=0))
        x_all = grid.to_mm(np.column_stack([xs, ys]).astype(np.float64))[:, 0]
        x_range = (float(np.percentile(x_all, 0.2)), float(np.percentile(x_all, 99.8)))  # stray pixels at a region's edge must not stretch the time axis
        samp = {**self.cfg["sampling"]}
        leads = {}
        for name, lm in zip(labels, lead_masks):
            try:
                t, mv, ok, base, bin_mm = sample_lead_safe(lm, grid, gain, x_range, samp)
            except ValueError:
                continue
            leads[name] = Lead(name, t, mv, x_range[0], base, float(ok.mean()), crossing, bin_mm=bin_mm)
        selftest.flag_leads(leads, self.cfg["flags"])
        return leads, crossing

    def finish(self, rec: PanelRecord) -> PanelRecord:
        checks = selftest.check_leads(rec.leads, self.cfg["selftest"]["max_lag_bins"])
        pulse = selftest.pulse_check(rec.pulses_mm, rec.gain_mm_per_mv)
        rec.selftest = {"pulse": pulse, "einthoven": checks}
        rec.confidence = selftest.confidence(rec.leads, pulse, checks)
        if rec.gain_source == "assumed":
            rec.confidence *= 0.5  # the shape is read, the amplitude scale is a guess
        return rec

    def orient_and_read(self, small: np.ndarray, finding: dict) -> tuple[int, list[dict], np.ndarray]:
        """Read the page text, choose the counterclockwise rotation that makes it upright, and read it again once it is upright:
        OCR reads turned text far worse. The header-above-footer cue picks the direction; when it is missing (no header or no
        footer read) both directions are read and the one that yields more known printed words wins."""
        cfg = self.cfg
        read = lambda im: ocr.read_text(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cfg["ocr_threads"], cfg["device"])
        vocab = re.compile(finding["vocab_regex"], re.I)
        words = lambda ts: sum(bool(vocab.search(t["text"])) for t in ts)
        frame = lambda r: small if r == 0 else np.rot90(small, r // 90).copy()
        texts = read(small)
        k, scores = choose_rotation(texts, small.shape[1], small.shape[0], finding)
        if not any(scores.values()):
            reads = {r: texts if r == 0 else read(frame(r)) for r in scores}
            k = max(reads, key=lambda r: words(reads[r]))
            return k, reads[k], frame(k)
        if k == 0:
            return 0, texts, small
        turned = frame(k)
        again = read(turned)
        k2, _ = choose_rotation(again, turned.shape[1], turned.shape[0], finding)
        if k2 == 180:  # still upside down: the first direction was wrong
            turned = np.rot90(turned, 2).copy()
            return (k + 180) % 360, read(turned), turned
        return k, again, turned

    # ---- panel mode ---------------------------------------------------------------------------------------------
    def run_panel_page(self, image: np.ndarray) -> tuple[list[PanelRecord], np.ndarray, int]:
        """Returns the records, the upright page they refer to, and the counterclockwise rotation that made it upright."""
        lay, cfg = self.layout, self.cfg
        finding = yaml.safe_load((ROOT / lay["panel_finding"]).read_text(encoding="utf-8"))
        h0, w0 = image.shape[:2]
        s = min(1.0, cfg["page_ocr_side"] / max(h0, w0))
        small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
        k, texts, small = self.orient_and_read(small, finding)
        up = np.rot90(image, k // 90).copy()
        rotated = [(t["text"], t["bbox"]) for t in texts]
        uh, uw = small.shape[:2]
        panels = find_panels(rotated, uw, uh, finding)
        scale = up.shape[1] / uw
        prob = self.unet.probability(up) if lay["mask"]["kind"] == "unet" else None
        records = []
        ecg = [q for q in panels if q["kind"] == "ecg"]
        assumed = self.page_gain(texts, [q["box"] for q in ecg])
        for i, p in enumerate(ecg):
            records.append(self.panel(up, prob, [round(v * scale) for v in p["box"]], i, texts, scale, assumed))
        return records, up, k

    def page_gain(self, texts: list[dict], boxes: list[list[float]]) -> float:
        """The gain most panels of this page print (boxes and texts in the same frame), else the configured default."""
        allowed = self.cfg["gain"]["allowed"]
        read = []
        for b in boxes:
            inside = [t for t in texts if b[0] <= (t["bbox"][0] + t["bbox"][2]) / 2 <= b[2] and b[1] <= (t["bbox"][1] + t["bbox"][3]) / 2 <= b[3]]
            g = text.read_gain(inside, allowed)
            if g is not None:
                read.append(g)
        return Counter(read).most_common(1)[0][0] if read else self.cfg["gain"]["assumed"]

    def panel(self, up: np.ndarray, prob: np.ndarray | None, box: list[int], index: int, page_texts: list[dict] | None = None, scale: float = 1.0, assumed_gain: float | None = None,
              force_gain: float | None = None, force_labels: list[str] | None = None) -> PanelRecord:
        """force_gain and force_labels replace what the page would say; they exist for the oracle runs (src.digitize.oracle)."""
        lay, cfg = self.layout, self.cfg
        mx, my = cfg["panel_margin_px"]
        x0, y0, x1, y1 = box
        cx0, cy0, cx1, cy1 = max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, up.shape[1]), min(y1 + my, up.shape[0])
        crop = up[cy0:cy1, cx0:cx1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texts = ocr.read_text(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cfg["ocr_threads"], cfg["device"])
        labels, label_source = text.read_labels(texts, lay["label_sets"], index)
        if label_source == "position" and page_texts is not None and cfg.get("merge_page_text"):
            # the crop read no usable lead name: ask the page read before falling back to the template order
            label_re = re.compile(lay["labels"]["label_regex"])
            extra = [{"text": t["text"], "bbox": [0, 0, 1, 1]} for t in page_texts if label_re.match(t["text"].strip()) and cx0 <= scale * (t["bbox"][0] + t["bbox"][2]) / 2 < cx1 and cy0 <= scale * (t["bbox"][1] + t["bbox"][3]) / 2 < cy1]
            labels2, source2 = text.read_labels(texts + extra, lay["label_sets"], index)
            if source2 == "ocr":
                labels, label_source = labels2, "page_ocr"
        if force_labels is not None:
            labels, label_source = list(force_labels), "oracle"
        rec = PanelRecord(f"panel{index}", lay["name"], box, [cx0, cy0], labels, label_source, None, "", Grid("dot", np.zeros(2), np.zeros(2), np.zeros(2)),
                          text_boxes=[{"text": t["text"], "bbox": [t["bbox"][0] + cx0, t["bbox"][1] + cy0, t["bbox"][2] + cx0, t["bbox"][3] + cy0]} for t in texts])
        try:
            grid = fit_grid(gray, lay["grid"])
            zone = text.trace_zone(texts, gray.shape[0], lay["zone"])
            window = (0, gray.shape[1], zone[0], zone[1] + 40)
            gain, source = (force_gain, "oracle") if force_gain is not None else (text.read_gain(texts, cfg["gain"]["allowed"]), "ocr")
            if gain is None:
                found = [p for g in cfg["gain"]["allowed"] for p in pulses.find_pulses(gray, grid, g, window, lay["pulses"])]
                gain, source = pulses.gain_from_pulses([p["height_mm"] for p in found], cfg["gain"]["allowed"], cfg["gain"]["tolerance"]), "pulse"
            if gain is None and assumed_gain is not None:  # GE style panels print no gain: keep the trace, say the scale is assumed
                gain, source = assumed_gain, "assumed"
            if gain is None:
                raise ValueError("gain not read and no calibration pulse found")
            found = pulses.find_pulses(gray, grid, gain, window, lay["pulses"])
            rec.grid, rec.gain_mm_per_mv, rec.gain_source, rec.pulses_mm = grid, gain, source, [p["height_mm"] for p in found]
            if prob is not None:
                mask = (prob[cy0:cy1, cx0:cx1] > cfg["unet"]["threshold"]).astype(np.uint8)
                mask[:zone[0]] = 0
                mask[zone[1]:] = 0
            else:
                label_re = re.compile(lay["labels"]["label_regex"])
                from_page = [{"text": t["text"], "bbox": [scale * t["bbox"][0] - cx0, scale * t["bbox"][1] - cy0, scale * t["bbox"][2] - cx0, scale * t["bbox"][3] - cy0]}
                             for t in (page_texts or []) if label_re.match(t["text"].strip()) and cx0 <= scale * (t["bbox"][0] + t["bbox"][2]) / 2 < cx1 and cy0 <= scale * (t["bbox"][1] + t["bbox"][3]) / 2 < cy1]
                names = [t for t in texts + from_page if label_re.match(t["text"].strip())]  # the crop read and the page read, in crop pixels
                lab_cols = text.label_columns(texts, tuple(lay["labels"]["fallback_columns"]), lay["labels"])
                printed = [t["bbox"] for t in texts if not label_re.match(t["text"].strip()) and len(re.sub(r"\W", "", t["text"])) >= 2]
                mask = maskmod.threshold_mask(gray, zone, lab_cols, [p["bbox"] for p in found], lay["mask"], printed, grid.px_per_mm[0], [t["bbox"] for t in names])
            mask[:, :x0 - cx0] = 0
            mask[:, x1 - cx0:] = 0
            mask = cut_at_gap(mask, lay["cut_gap_px"])
            rec.leads, _ = self.digitize_region(mask, grid, gain, labels, lay["rows"], lay["rows_mode"])
        except Exception as exc:  # a panel that cannot be read keeps its record and its error
            rec.error = f"{type(exc).__name__}: {exc}"
        return self.finish(rec)

    # ---- page mode ----------------------------------------------------------------------------------------------
    def run_fixed_page(self, image: np.ndarray) -> list[PanelRecord]:
        lay, cfg = self.layout, self.cfg
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fx0, fx1, fy0, fy1 = lay["grid"].get("fit_region", [0, 1, 0, 1])
        gx0, gx1, gy0, gy1 = int(fx0 * w), int(fx1 * w), int(fy0 * h), int(fy1 * h)
        grid = fit_grid(gray[gy0:gy1, gx0:gx1], lay["grid"]).shifted(gx0, gy0) if lay["grid"]["kind"] == "dot" else fit_grid(gray, lay["grid"])
        gain = lay["gain"]["fixed"]
        found = []
        if "pulses" in lay:
            pw = lay["pulses"]["window"]
            found = pulses.find_pulses(gray, grid, gain, (int(pw[0] * w), int(pw[1] * w), int(pw[2] * h), int(pw[3] * h)), lay["pulses"])
        if lay["mask"]["kind"] == "unet":
            full = self.unet.mask(self.unet.probability(image))
        else:
            full = maskmod.local_contrast_mask(gray, lay["mask"])
        records = []
        for r in lay["regions"]:
            rx0, rx1, ry0, ry1 = int(r["x"][0] * w), int(r["x"][1] * w), int(r["y"][0] * h), int(r["y"][1] * h)
            mask = np.zeros_like(full)
            mask[ry0:ry1, rx0:rx1] = full[ry0:ry1, rx0:rx1]
            rec = PanelRecord(r["id"], lay["name"], [rx0, ry0, rx1, ry1], [0, 0], r["labels"], "layout", gain, "layout", grid, [p["height_mm"] for p in found])
            try:
                rec.leads, _ = self.digitize_region(mask, grid, gain, r["labels"], r["rows"], r["rows_mode"])
            except Exception as exc:
                rec.error = f"{type(exc).__name__}: {exc}"
            records.append(self.finish(rec))
        return records


def sample_lead_safe(mask, grid, gain, x_range, cfg):
    from src.digitize.sample import sample_lead

    return sample_lead(mask, grid, gain, x_range, cfg)
