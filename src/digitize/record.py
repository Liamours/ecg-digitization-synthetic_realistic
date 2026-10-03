"""Data records passed between stages and written to disk."""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Grid:
    """Local coordinate map: page pixel p = origin + x * a1 + y * a2, with (x, y) in millimetres."""
    kind: str
    origin: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    rms_px: float = float("nan")
    n_inliers: int = 0

    def to_mm(self, pts: np.ndarray) -> np.ndarray:
        return np.linalg.solve(np.stack([self.a1, self.a2], axis=1), (pts - self.origin).T).T

    def to_px(self, mm: np.ndarray) -> np.ndarray:
        return self.origin + mm @ np.stack([self.a1, self.a2], axis=0)

    @property
    def px_per_mm(self) -> tuple[float, float]:
        return float(np.hypot(*self.a1)), float(np.hypot(*self.a2))

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.a1[1], self.a1[0])))

    def shifted(self, dx: float, dy: float) -> "Grid":
        return Grid(self.kind, self.origin + np.array([dx, dy]), self.a1, self.a2, self.rms_px, self.n_inliers)


@dataclass
class Lead:
    name: str
    t: np.ndarray
    mv: np.ndarray
    x0_mm: float
    baseline_mm: float
    coverage: float
    crossing_fraction: float
    flag: str = "ok"
    bin_mm: float = 0.1

    @property
    def p2p_mv(self) -> float:
        return float(np.ptp(np.percentile(self.mv, [0.5, 99.5])))


@dataclass
class PanelFront:
    """A panel before its traces are read: the crop and what the page says about it. Saved as panel<i>.png and panel<i>.front.json,
    so every trace method can start from the same input without running OCR and the grid fit again."""
    index: int
    layout: str
    box: list[int]                 # panel rectangle in upright page pixels
    crop_origin: list[int]         # page pixels of the crop's top left corner (the box plus a margin)
    crop: np.ndarray               # BGR
    texts: list[dict]              # OCR of the crop, crop pixels
    page_names: list[dict]         # lead-name boxes the page read found inside the crop, crop pixels
    labels: list[str]
    label_source: str
    grid: Grid | None = None
    zone: tuple[int, int] | None = None   # rows between the header and footer text, crop pixels
    gain: float | None = None
    gain_source: str = ""
    pulses: list[dict] = field(default_factory=list)   # calibration pulses found: bbox in crop pixels, height in mm
    error: str = ""

    def save(self, out_dir: Path) -> None:
        import cv2

        cv2.imwrite(str(out_dir / f"panel{self.index}.png"), self.crop)
        g = self.grid
        meta = {"index": self.index, "layout": self.layout, "box": self.box, "crop_origin": self.crop_origin, "labels": self.labels, "label_source": self.label_source,
                "gain_mm_per_mV": self.gain, "gain_source": self.gain_source, "zone": list(self.zone) if self.zone else None, "pulses": self.pulses, "error": self.error,
                "grid": None if g is None else {"kind": g.kind, "origin": g.origin.tolist(), "a1": g.a1.tolist(), "a2": g.a2.tolist(), "rms_px": g.rms_px, "n_inliers": g.n_inliers,
                                                 "tilt_deg": g.tilt_deg, "px_per_mm": g.px_per_mm},
                "texts": self.texts, "page_names": self.page_names}
        (out_dir / f"panel{self.index}.front.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, json_path: Path) -> "PanelFront":
        import cv2

        m = json.loads(json_path.read_text(encoding="utf-8"))
        g = m["grid"]
        grid = None if g is None else Grid(g["kind"], np.array(g["origin"]), np.array(g["a1"]), np.array(g["a2"]), g["rms_px"], g["n_inliers"])
        return cls(m["index"], m["layout"], m["box"], m["crop_origin"], cv2.imread(str(json_path.with_name(f"panel{m['index']}.png"))), m["texts"], m["page_names"], m["labels"], m["label_source"],
                   grid, tuple(m["zone"]) if m["zone"] else None, m["gain_mm_per_mV"], m["gain_source"], m["pulses"], m["error"])


@dataclass
class PanelRecord:
    panel_id: str
    layout: str
    box: list[int]                 # page pixels, upright frame: x0, y0, x1, y1
    crop_origin: list[int]         # page pixels of the crop the grid map and masks live in
    labels: list[str]
    label_source: str
    gain_mm_per_mv: float | None
    gain_source: str
    grid: Grid
    pulses_mm: list[float] = field(default_factory=list)
    leads: dict[str, Lead] = field(default_factory=dict)
    selftest: dict = field(default_factory=dict)
    confidence: float = 0.0
    error: str = ""
    text_boxes: list[dict] = field(default_factory=list)  # OCR boxes in page pixels, kept so they can be drawn back

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id, "layout": self.layout, "box": self.box, "crop_origin": self.crop_origin,
            "labels": self.labels, "label_source": self.label_source, "gain_mm_per_mV": self.gain_mm_per_mv, "gain_source": self.gain_source,
            "grid": {"kind": self.grid.kind, "origin": self.grid.origin.tolist(), "a1": self.grid.a1.tolist(), "a2": self.grid.a2.tolist(),
                     "rms_px": self.grid.rms_px, "n_inliers": self.grid.n_inliers, "tilt_deg": self.grid.tilt_deg, "px_per_mm": self.grid.px_per_mm},
            "pulses_mm": self.pulses_mm, "selftest": self.selftest, "confidence": round(self.confidence, 3), "error": self.error, "text_boxes": self.text_boxes,
            "leads": {k: {"x0_mm": v.x0_mm, "baseline_mm": v.baseline_mm, "coverage": round(v.coverage, 3), "crossing_fraction": round(v.crossing_fraction, 3),
                          "p2p_mV": round(v.p2p_mv, 3), "flag": v.flag, "bin_mm": round(v.bin_mm, 4)} for k, v in self.leads.items()},
        }

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{self.panel_id}.json").write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        if not self.leads:
            return
        first = next(iter(self.leads.values()))
        with (out_dir / f"{self.panel_id}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s"] + [f"{k}_mV" for k in self.leads] + [f"{k}_flag" for k in self.leads])
            for j, t in enumerate(first.t):
                w.writerow([f"{t:.4f}"] + [f"{v.mv[j]:.4f}" if j < len(v.mv) else "" for v in self.leads.values()] + [v.flag for v in self.leads.values()])
