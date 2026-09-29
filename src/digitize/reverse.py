"""Draw a result back on the original image. Nothing was resampled, so the map is exact: a sample (t, mV) becomes
grid millimetres (x = x0 + mm_per_s * t, y = baseline - gain * mV), grid millimetres become crop pixels through the
Grid, and crop pixels become page pixels by adding the crop origin."""
import cv2
import numpy as np

from src.digitize.record import PanelRecord

PANEL_COLOR = (200, 0, 0)
TEXT_COLOR = (200, 120, 0)
SIGNAL_COLOR = (255, 0, 255)


def lead_pixels(rec: PanelRecord, name: str, cfg: dict) -> np.ndarray:
    lead = rec.leads[name]
    x = lead.x0_mm + lead.t * cfg["mm_per_s"] + lead.bin_mm / 2
    y = lead.baseline_mm - lead.mv * rec.gain_mm_per_mv
    return rec.grid.to_px(np.column_stack([x, y])) + np.array(rec.crop_origin)


def draw(image: np.ndarray, records: list[PanelRecord], cfg: dict) -> np.ndarray:
    """Panel boxes, text boxes, and the digitized signals on a copy of the image."""
    out = image.copy()
    for rec in records:
        x0, y0, x1, y1 = rec.box
        cv2.rectangle(out, (x0, y0), (x1, y1), PANEL_COLOR, 4)
        cv2.putText(out, rec.panel_id, (x0 + 8, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, PANEL_COLOR, 3)
        for t in rec.text_boxes:
            bx0, by0, bx1, by1 = (int(v) for v in t["bbox"])
            cv2.rectangle(out, (bx0, by0), (bx1, by1), TEXT_COLOR, 2)
            cv2.putText(out, t["text"][:22], (bx0, max(by0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        for name, lead in rec.leads.items():
            pts = lead_pixels(rec, name, cfg)
            color = SIGNAL_COLOR if lead.flag == "ok" else (0, 165, 255)
            cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, 2)
            cv2.putText(out, name, (int(pts[0, 0]) - 70, int(pts[0, 1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return out
