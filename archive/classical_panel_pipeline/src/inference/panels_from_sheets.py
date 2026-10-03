"""Task 2 (page -> panels), the paper-ecg-venv half: given sheet crops
already produced by segment_sheets.py, find each panel (lead column),
classify it ecg/non_ecg, measure its tilt, crop it, and emit manifest rows
as JSON for the orchestrator (pages_to_panels.py) to collect.

Panel x/y/w/h are in the ORIGINAL PAGE's pixel frame (2026-09-06; previously
the sheet's own oriented-crop frame, see git history for that version's
notes). Each panel box is found in the oriented-sheet frame same as before,
then mapped back through the discrete orientation rotation (`rotation_deg`,
already tracked) composed with segment_sheets.py's perspective deskew matrix
(now persisted as a `<sheet>_matrix.npy` sidecar next to the sheet crop,
previously computed and discarded). The reported box is the axis-aligned
bounding box of the mapped quadrilateral's 4 corners, not the true
quadrilateral -- matches datasets/synthetic-ptbxl-panels's own
manifest_panels.csv bbox_* convention, so a detector or ground truth
expressed as a plain rectangle compares directly (IoU) against this without
extra translation on either side.

Lead x/y/w/h are NOT part of this fix -- they remain local to their own
normalized panel crop (see deskew_and_normalize_panel), a third coordinate
frame distinct from both the sheet frame and this page frame. Mapping leads
into page-space too would additionally need deskew_and_normalize_panel's own
per-panel rotation+crop threaded through, which nothing here does yet.

Usage:
    .venvs/paper-ecg/Scripts/python src/inference/panels_from_sheets.py \
        --page-id <id> --sheet-index <k> --sheet <sheet-crop.png> \
        --output-dir <panels-dir> --start-panel-index <n>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_and_digitize_columns import build_column_lead_boxes, box_is_usable, find_lead_labels  # noqa: E402
from panel_geometry import DARKNESS_THRESHOLD, ROTATIONS, choose_orientation, find_calibration_squares, looks_like_signal  # noqa: E402

def _unrotate_point(x: float, y: float, degrees: int, oriented_w: int, oriented_h: int) -> tuple[float, float]:
    """Maps a point in the oriented-sheet frame back to the deskewed-sheet
    frame (segment_sheets.py's own output frame, before choose_orientation's
    rotation) -- the exact inverse of applying ROTATIONS[degrees] via
    cv2.rotate, verified by round-trip test against cv2.rotate itself at
    every discrete code and several points (2026-09-06), not derived by eye:
    cv2.rotate's 90-degree codes swap width/height, so which axis flips and
    which just carries over differs by code and is easy to get backwards.
    Dispatches on the FORWARD code (ROTATIONS[degrees], the one actually
    applied to produce the oriented frame) -- inverting the code first and
    then dispatching on that looks equally reasonable but is wrong, since
    each branch below is already the inverse formula for its own forward
    code, not a formula to compose with a second inversion."""
    code = ROTATIONS[degrees]
    if code is None:
        return x, y
    if code == cv2.ROTATE_90_CLOCKWISE:
        return y, oriented_w - 1 - x
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return oriented_h - 1 - y, x
    return oriented_w - 1 - x, oriented_h - 1 - y  # ROTATE_180


def map_box_to_page(box_xyxy: tuple[float, float, float, float], rotation_deg: int, oriented_shape: tuple[int, int], inv_deskew_matrix: np.ndarray | None) -> tuple[float, float, float, float]:
    """Maps a panel box's 4 corners from the oriented-sheet frame back
    through the discrete rotation then the perspective deskew, into the
    original page's pixel frame, and returns the axis-aligned bounding box
    of the 4 mapped corners (not the true quadrilateral -- see module
    docstring). `inv_deskew_matrix` is None when no sidecar matrix file was
    found (e.g. an older run's sheet crops) -- falls back to sheet-frame
    coordinates (post-rotation-undo only) rather than raising, since a
    caller with no matrix genuinely cannot recover page-space and the
    previous sheet-frame behavior is a documented, known fallback, not a
    silent wrong answer."""
    x0, y0, x1, y1 = box_xyxy
    oriented_h, oriented_w = oriented_shape
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    corners = [_unrotate_point(x, y, rotation_deg, oriented_w, oriented_h) for x, y in corners]
    if inv_deskew_matrix is not None:
        pts = np.array([[x, y, 1.0] for x, y in corners]).T  # 3xN homogeneous
        mapped = inv_deskew_matrix @ pts
        corners = [(mapped[0, i] / mapped[2, i], mapped[1, i] / mapped[2, i]) for i in range(mapped.shape[1])]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)

PANEL_MARGIN_PX = 15
JPEG_QUALITY = 92  # disk on this shared machine repeatedly hit critical during a full-dataset run at PNG size;
                    # quality 92 is visually near-lossless and OTSU thresholding (the only thing that reads these
                    # images back) is robust to the small ringing JPEG introduces at this quality

# Normalized panel size, in 5mm major grid boxes -- the printed calibration
# line itself ("Man 25mm/s 10mm/mV") sets the unit: 18 boxes wide = 90mm =
# 3.6s of trace per lead at 25mm/s, 15 boxes tall over 3 stacked leads = 5
# boxes = 25mm = 2.5mV of headroom per row at 10mm/mV, both physically
# sensible trace windows. Whole panel first, split into rows after.
MAJOR_BOX_PERIODS = 5.0  # 1 major box = 5 fine (1mm) grid periods
PANEL_BOX_WIDTH = 18
PANEL_BOX_HEIGHT = 15
ROWS_PER_PANEL = 3
WORKING_PAD_BOX = 4  # extra major boxes of padding around the dirty anchor so deskew rotation has room


def find_row_cuts(bgr_normalized: np.ndarray, num_rows: int, major_px: float) -> list[int]:
    """Row boundaries from the trace's own ink density, not a fixed
    equal-thirds split -- leads are not reliably centered in their third of
    the panel (plan-preprocessing-tasks.md Task 3, step 1). Adapted from
    external/ecgtizer/ecgtizer/PDF2XML.py's tracks_extraction(): row centers
    are peaks in the row-wise ink-density profile (a trace row has far more
    dark ink than the blank gap rows between leads), cuts fall at the
    midpoints between consecutive peaks. Falls back to equal thirds only
    when peak-finding doesn't recover exactly num_rows peaks -- a real
    layout mismatch or too little ink to trust the profile."""
    gray = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    _, binary = cv2.threshold(gray, DARKNESS_THRESHOLD, 1, cv2.THRESH_BINARY_INV)
    row_profile = binary.sum(axis=1).astype(float)
    k = max(3, int(major_px * 0.5)) | 1
    smoothed = np.convolve(row_profile, np.ones(k) / k, mode="same")

    min_gap = int(0.6 * h / num_rows)
    peaks, _ = find_peaks(smoothed, distance=min_gap, prominence=max(1.0, smoothed.max() * 0.1))

    if len(peaks) != num_rows:
        return [round(i * h / num_rows) for i in range(num_rows + 1)]

    gaps = np.diff(peaks)
    half = int(np.median(gaps) / 2)
    cuts = [max(0, peaks[0] - half)]
    cuts += [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
    cuts.append(min(h, peaks[-1] + half))
    return cuts


def deskew_and_normalize_panel(oriented: np.ndarray, anchor: tuple[int, int], pitch: float, tilt_deg: float) -> tuple[np.ndarray | None, list[np.ndarray], float]:
    """Panel-body pipeline step (page->panel->lead): rotate the panel's
    working region to remove its residual tilt, relocate the calibration
    square in that deskewed frame, then crop a fixed 18x15-major-box region
    from it and split that into 3 equal row bands, one per lead.

    The target region (up to 90mm wide at this project's real measured
    pitch) can exceed the sheet's own available area -- a too-small sheet
    crop, or an overestimated pitch (a real, separate, already-documented
    failure mode in estimate_pitch's own docstring) both produce this.
    Padding with edge replication rather than failing keeps every real
    panel producing 3 lead rows, per plan-preprocessing-tasks.md's own
    requirement to always write 3 rows per ECG panel and flag confidence
    rather than silently dropping a panel. Returns (normalized_panel,
    [lead_crop, ...], pad_frac) -- pad_frac is the fraction of the
    normalized panel's area that is fabricated edge-replication rather than
    real sheet content (0.0 when nothing was padded); None normalized only
    when the calibration square is lost entirely after deskewing."""
    major_px = pitch * MAJOR_BOX_PERIODS
    target_w = int(round(PANEL_BOX_WIDTH * major_px))
    target_h = int(round(PANEL_BOX_HEIGHT * major_px))
    pad = int(round(WORKING_PAD_BOX * major_px))

    H, W = oriented.shape[:2]
    ax, ay = anchor
    wx0, wy0 = max(0, ax - pad), max(0, ay - pad)
    wx1, wy1 = min(W, ax + target_w + pad), min(H, ay + target_h + pad)
    working = oriented[wy0:wy1, wx0:wx1]
    if working.size == 0:
        return None, [], 0.0

    center = (working.shape[1] / 2, working.shape[0] / 2)
    rot_mat = cv2.getRotationMatrix2D(center, tilt_deg, 1.0)
    rotated = cv2.warpAffine(working, rot_mat, (working.shape[1], working.shape[0]),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    squares = find_calibration_squares(gray, pitch)
    if squares:
        fx, fy, _, _ = min(squares, key=lambda s: s[0] + s[1])  # nearest the working crop's own top-left
    else:
        # Deskew rotation is a small residual angle around the crop's own
        # center; if the square is lost after it (edge blur from
        # interpolation, or it sat right at the working crop's boundary),
        # the pre-rotation anchor position is still a good fallback --
        # confirmed the square itself is real and was found before rotating.
        pre_squares = find_calibration_squares(cv2.cvtColor(working, cv2.COLOR_BGR2GRAY), pitch)
        if not pre_squares:
            return None, [], 0.0
        fx, fy, _, _ = min(pre_squares, key=lambda s: s[0] + s[1])

    real_w = max(0, min(target_w, rotated.shape[1] - fx))
    real_h = max(0, min(target_h, rotated.shape[0] - fy))
    pad_frac = 1.0 - (real_w * real_h) / (target_w * target_h)
    pad_bottom = max(0, fy + target_h - rotated.shape[0])
    pad_right = max(0, fx + target_w - rotated.shape[1])
    if pad_frac > 0.0:
        rotated = cv2.copyMakeBorder(rotated, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)

    normalized = rotated[fy:fy + target_h, fx:fx + target_w]

    cuts = find_row_cuts(normalized, ROWS_PER_PANEL, major_px)
    leads = [normalized[cuts[j]:cuts[j + 1], :] for j in range(ROWS_PER_PANEL)]
    return normalized, leads, pad_frac


def square_fill_confidence(gray: np.ndarray, square: tuple[int, int, int, int]) -> float:
    """Re-measures the accepted calibration square's own fill ratio directly
    against the darkness threshold, as a simple, honest anchor_conf: how
    solidly filled the accepted region actually is, not a learned score."""
    x, y, w, h = square
    if w <= 0 or h <= 0:
        return 0.0
    crop = gray[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    dark = (crop < DARKNESS_THRESHOLD).mean()
    return float(dark)


def panel_tilt_deg(gray_panel: np.ndarray) -> float:
    """Small residual skew, from the panel's own near-horizontal line
    segments (grid lines, borders, trace baselines), not the calibration
    square. A square has no well-defined rotation angle by itself --
    minAreaRect on a near-square blob is ambiguous about which side counts
    as width vs height, and flips discontinuously between ~0 and ~90 on
    essentially the same input (confirmed: every panel measured -90.0
    exactly, regardless of real orientation -- a degenerate result, not a
    real reading). Hough segments have no such ambiguity: length and angle
    are both well-defined, so the long, near-horizontal ones (grid rules,
    panel borders) dominate a length-weighted average over short, noisy
    trace wiggles."""
    edges = cv2.Canny(gray_panel, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                             minLineLength=max(20, int(gray_panel.shape[1] * 0.3)), maxLineGap=10)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if abs(angle) <= 20:  # near-horizontal candidates only
            angles.append(angle)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    return float(np.average(angles, weights=weights))


def classify_ecg_or_non_ecg(gray: np.ndarray, rows: list[tuple[int, int, int, int]], pitch: float) -> str:
    signal_rows = sum(1 for box in rows if looks_like_signal(gray, box, pitch))
    return "ecg" if signal_rows >= 2 else "non_ecg"


def process_sheet(sheet_path: Path, page_id: str, sheet_index: int, start_panel_index: int, output_dir: Path) -> tuple[list[dict], list[dict]]:
    leads_dir = output_dir.parent / "leads"
    leads_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = sheet_path.with_name(sheet_path.stem + "_matrix.npy")
    inv_deskew_matrix = np.linalg.inv(np.load(matrix_path)) if matrix_path.exists() else None

    raw = cv2.imread(str(sheet_path))
    oriented, rotation_deg, pitch, squares = choose_orientation(raw)
    panels_out: list[dict] = []
    leads_out: list[dict] = []
    if pitch is None or not squares:
        return panels_out, leads_out

    gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
    sheet_h, sheet_w = gray.shape
    panel_index = start_panel_index
    for column in build_column_lead_boxes(gray, squares, pitch, sheet_w, sheet_h):
        row_boxes = column["rows"]
        usable = all(box_is_usable(box, pitch) for box in row_boxes)

        # partial is about the REAL panel content reaching the sheet edge,
        # not about whether our own cosmetic margin fit -- checking the
        # padded box here flagged nearly every panel "partial" even when
        # fully visible, since a panel just needs to sit within
        # PANEL_MARGIN_PX of the edge to trip it for no real reason.
        raw_x0 = min(column["anchor"][0], min(r[0] for r in row_boxes))
        raw_y0 = column["anchor"][1]
        raw_x1 = max(r[0] + r[2] for r in row_boxes)
        raw_y1 = row_boxes[-1][1] + row_boxes[-1][3]
        partial = raw_x0 <= 0 or raw_y0 <= 0 or raw_x1 >= sheet_w or raw_y1 >= sheet_h

        cx0 = max(0, raw_x0 - PANEL_MARGIN_PX)
        cy0 = max(0, raw_y0 - PANEL_MARGIN_PX)
        cx1 = min(sheet_w, raw_x1 + PANEL_MARGIN_PX)
        cy1 = min(sheet_h, raw_y1 + PANEL_MARGIN_PX)
        crop = oriented[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue

        panel_id = f"{page_id}-p{panel_index}"
        panel_path = output_dir / f"{panel_id}.jpg"
        cv2.imwrite(str(panel_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        ecg_or_non_ecg = classify_ecg_or_non_ecg(gray, row_boxes, pitch) if usable else "non_ecg"
        anchor_conf = square_fill_confidence(gray, column["anchor"] + (int(pitch), int(pitch)))
        tilt = panel_tilt_deg(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        notes = []
        if partial:
            notes.append("partial panel: clipped at sheet edge")
        if not usable:
            notes.append("row box smaller than the minimum lead-strip size")

        normalized, lead_crops, pad_frac = deskew_and_normalize_panel(oriented, column["anchor"][:2], pitch, tilt)
        if normalized is None:
            notes.append("normalize failed: calibration square not relocated after deskew")
        else:
            if pad_frac > 0.0:
                severity = "severe, pitch may be overestimated for this panel" if pad_frac > 0.3 else "minor"
                notes.append(f"normalize padded ({severity}, {pad_frac:.0%} fabricated): target box region ran past the available sheet area")
            cv2.imwrite(str(panel_path), normalized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])  # normalized crop replaces the dirty one as the panel's own record
            # find_lead_labels searches relative to a square's own position -- the
            # calibration square sits at (0, 0) in `normalized` by construction
            # (deskew_and_normalize_panel crops starting exactly at the square's
            # own detected top-left), a far more predictable frame than the old
            # dirty, pre-deskew sheet coordinates this used to search in.
            normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            major_px = pitch * MAJOR_BOX_PERIODS
            labels = find_lead_labels(normalized_gray, (0, 0, int(major_px), int(major_px)), pitch, panel_bottom=normalized.shape[0]) if usable else []
            names_by_row = {i: name for i, (name, _, _) in enumerate(labels)}
            y_offset = 0
            for j, lead_crop in enumerate(lead_crops):
                lead_id = f"{panel_id}-l{j}"
                lead_path = leads_dir / f"{lead_id}.jpg"
                cv2.imwrite(str(lead_path), lead_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                leads_out.append({
                    "id": lead_id,
                    "panel_id": panel_id,
                    "row_index": j,
                    "lead_path": str(lead_path),
                    "x": 0, "y": y_offset, "w": normalized.shape[1], "h": lead_crop.shape[0],
                    "lead_name": names_by_row.get(j),
                })
                y_offset += lead_crop.shape[0]

        # w/h describe the file actually written to panel_path: the
        # normalized crop's own size once normalization succeeds, the dirty
        # crop's size otherwise. x/y stay the dirty crop's sheet-frame
        # position either way -- normalized's own position is expressed
        # through a rotation, not a plain offset, so it can't be reported as
        # one pair of sheet-frame coordinates without the same frame
        # limitation already stated in this file's own module docstring.
        panel_w, panel_h = (normalized.shape[1], normalized.shape[0]) if normalized is not None else (cx1 - cx0, cy1 - cy0)
        page_x0, page_y0, page_x1, page_y1 = map_box_to_page((cx0, cy0, cx0 + panel_w, cy0 + panel_h), rotation_deg, (sheet_h, sheet_w), inv_deskew_matrix)

        rows_out_notes = "; ".join(notes)
        panels_out.append({
            "id": panel_id,
            "page_id": page_id,
            "panel_index": panel_index,
            "panel_path": str(panel_path),
            "sheet_index": sheet_index,
            "x": round(page_x0, 1), "y": round(page_y0, 1), "w": round(page_x1 - page_x0, 1), "h": round(page_y1 - page_y0, 1),
            "rotation_deg": rotation_deg,
            "tilt_deg": round(tilt, 2),
            "anchor_conf": round(anchor_conf, 3),
            "ecg_or_non_ecg": ecg_or_non_ecg,
            "low_confidence": bool(partial or not usable or anchor_conf < 0.5 or normalized is None or pad_frac > 0.3),
            "notes": rows_out_notes,
        })
        panel_index += 1
    return panels_out, leads_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 per-sheet worker: find panels, classify, crop.")
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--sheet-index", type=int, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-panel-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels, leads = process_sheet(args.sheet, args.page_id, args.sheet_index, args.start_panel_index, args.output_dir)
    print(json.dumps({"panels": panels, "leads": leads}))


if __name__ == "__main__":
    main()
