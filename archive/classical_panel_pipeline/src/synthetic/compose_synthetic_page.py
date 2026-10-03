"""Phase B of the synthetic ground-truth compositor (transient-swimming-tide.md):
place Phase A's ground-truth panels onto a synthetic page, each with an
independently-known rotation and tilt, and record the exact ground truth.

Two explicit page groups (not left to independent-randomization chance):
  a. same-orientation -- every panel on the page shares one discrete rotation
  b. mixed-orientation -- at least two panels differ
matching the real, already-documented case of a page mixing orientations
(ekg757-normal-tambahan-0059) as its own scorable scenario, not folded into
an average with the (more common) consistent case.

v1 scope: one panel per "sheet" (a real margin/gap around every panel),
matching the simpler of the two real sheet layouts already documented in
analyses/compositing-preprocessing/README.md. Panels butted edge-to-edge on
one physical sheet is a real but separate case, not built here.

Usage:
    uv run python -m src.synthetic.compose_synthetic_page \
        --panels-manifest ../../dataset/synthetic-ptbxl-panels/manifest.csv \
        --output-root ../../dataset/synthetic-ptbxl-pages --num-pages 20
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_PANELS_PER_PAGE = 4
MAX_PANELS_PER_PAGE = 6
DISCRETE_ROTATIONS = (0, 90, 180, 270)
TILT_RANGE_DEG = (-1.5, 1.5)
MARGIN_PX = 40
PAGE_BG = (232, 228, 218)

# Page-level noise, matching real conditions already documented in
# dataset/ekg-757/preprocessed/README.md ("Text appears outside panels,
# inside panels, and sometimes over the traces themselves") plus staple
# marks and opaque black stickers observed directly in real scans this
# session. Added here (page level, after panels are placed) rather than in
# Phase A, since these are page artifacts independent of any one panel's
# own ground truth.
HANDWRITING_WORDS = ["hasil", "normal", "ulang", "cek", "revisi", "baik", "tgl", "anak", "ASD", "VSD", "PDA", "OK"]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["ariali.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def draw_staple_marks(draw: ImageDraw.ImageDraw, placed_boxes: list[tuple[float, float, float, float]], rng: random.Random) -> None:
    """Short diagonal dark line near a panel's corner -- a real artifact seen
    directly in this project's own scans (a staple or pen mark at a page
    edge), deliberately allowed to clip into the panel's own side."""
    for _ in range(rng.randint(1, 3)):
        x0, y0, x1, y1 = rng.choice(placed_boxes)
        corner_x = rng.choice([x0, x1])
        corner_y = rng.choice([y0, y1])
        length = rng.uniform(25, 50)
        angle = rng.uniform(20, 70)
        dx, dy = length * np.cos(np.radians(angle)), length * np.sin(np.radians(angle))
        # let the mark extend inward (toward the panel) as often as outward
        sign_x, sign_y = rng.choice([-1, 1]), rng.choice([-1, 1])
        draw.line([(corner_x, corner_y), (corner_x + sign_x * dx, corner_y + sign_y * dy)],
                  fill=(20, 20, 20), width=rng.randint(2, 3))


def draw_handwriting_noise(draw: ImageDraw.ImageDraw, page_w: int, page_h: int,
                            placed_boxes: list[tuple[float, float, float, float]], rng: random.Random) -> None:
    """Sparse random numbers/words in a lightly-italic font, only placed
    where they don't overlap any panel -- deliberately not realistic
    handwriting-synthesis (that needs ecg-image-kit's tensorflow/keras/spacy
    stack this project avoided installing for disk-space reasons); this is
    plain background clutter text, matching the "outside panels" case."""
    font = load_font(rng.randint(16, 22))
    for _ in range(rng.randint(2, 5)):
        text = rng.choice(HANDWRITING_WORDS) if rng.random() < 0.5 else str(rng.randint(0, 9999))
        for _attempt in range(15):
            x, y = rng.uniform(0, max(1, page_w - 100)), rng.uniform(0, max(1, page_h - 30))
            box = (x, y, x + len(text) * 12, y + 24)
            if not any(boxes_overlap(box, pb) for pb in placed_boxes):
                draw.text((x, y), text, font=font, fill=(50, 50, 70))
                break


def draw_black_stickers(draw: ImageDraw.ImageDraw, placed_boxes: list[tuple[float, float, float, float]], rng: random.Random) -> None:
    """Small opaque black rectangles, deliberately allowed to sit at or over
    a panel's edge -- a real, solid-black-square-shaped confound for
    find_calibration_squares, and the reason this one is flagged important:
    it directly tests whether an anchor detector can be fooled by a
    same-looking but unrelated black patch, not just find the real square."""
    for _ in range(rng.randint(1, 2)):
        x0, y0, x1, y1 = rng.choice(placed_boxes)
        w_, h_ = rng.uniform(30, 65), rng.uniform(20, 45)
        # centered near a random point along the panel's own border
        edge_x = rng.uniform(x0, x1)
        edge_y = rng.choice([y0, y1])
        draw.rectangle([edge_x - w_ / 2, edge_y - h_ / 2, edge_x + w_ / 2, edge_y + h_ / 2], fill=(12, 12, 12))


def apply_shadow(canvas: Image.Image, rng: random.Random) -> Image.Image:
    """Faint directional lighting gradient across the whole page -- a
    cast-shadow/uneven-light effect, not a hard edge. Multiplicative, so ink
    darkness relative to its own local background is preserved."""
    w, h = canvas.size
    arr = np.array(canvas).astype(np.float64)
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    proj = (proj - proj.min()) / max(1e-6, proj.max() - proj.min())
    strength = rng.uniform(0.12, 0.3)
    gradient = 1.0 - strength * proj
    arr *= gradient[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_page_perspective(canvas: Image.Image, gt_rows: list[dict], rng: random.Random) -> tuple[Image.Image, list[dict]]:
    """A mild whole-page perspective warp -- the paper not perfectly flat /
    photographed at a slight angle, matching real acquisition rather than
    just the in-plane rotation+tilt already applied per panel. This is the
    actual target distortion behind this whole synthetic system (see
    plan-digitization.md's "grid rectification" backlog item): a photographed
    page's grid pitch can genuinely differ by direction/position, which
    in-plane rotation alone can't reproduce. One shared homography for the
    whole page, applied to the canvas and to every recorded ground-truth
    corner point identically, so the numbers stay exact."""
    w, h = canvas.size
    max_shift_frac = 0.035  # mild: corners displaced by at most ~3.5% of that dimension
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(0, max_shift_frac * w), rng.uniform(0, max_shift_frac * h)],
        [w - rng.uniform(0, max_shift_frac * w), rng.uniform(0, max_shift_frac * h)],
        [w - rng.uniform(0, max_shift_frac * w), h - rng.uniform(0, max_shift_frac * h)],
        [rng.uniform(0, max_shift_frac * w), h - rng.uniform(0, max_shift_frac * h)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(bgr, matrix, (w, h), borderValue=tuple(reversed(PAGE_BG)))
    warped_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    for row in gt_rows:
        for i in range(4):
            point = np.array([row[f"x{i}"], row[f"y{i}"], 1.0])
            transformed = matrix @ point
            transformed /= transformed[2]
            row[f"x{i}"] = round(float(transformed[0]), 1)
            row[f"y{i}"] = round(float(transformed[1]), 1)
    return warped_img, gt_rows

GT_COLUMNS = ["page_id", "panel_index", "source_panel_id", "record_id", "lead_names",
              "group", "rotation_deg", "tilt_deg", "total_angle_deg",
              "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "page_path"]


def rotate_panel(img: Image.Image, total_angle_deg: float) -> Image.Image:
    """Rotates counter-clockwise by total_angle_deg (discrete rotation +
    tilt combined into one transform), expanding the canvas to fit -- PIL's
    own expand=True handles the 90/180/270 dimension swap and an arbitrary
    small tilt in one consistent operation, rather than juggling cv2.rotate
    (discrete) and warpAffine (continuous) separately."""
    return img.rotate(total_angle_deg, expand=True, fillcolor=PAGE_BG, resample=Image.BICUBIC)


def corners_after_rotation(w: int, h: int, total_angle_deg: float) -> np.ndarray:
    """The panel's own 4 corners (upright, before rotation), forward-
    transformed by the same rotation PIL just applied, so the recorded
    ground-truth polygon matches the pasted pixels exactly."""
    theta = np.radians(total_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # PIL's rotate(expand=True) is counter-clockwise about the image center,
    # for the ORIGINAL image's own dimensions -- rotate the original box's
    # corners (centered at origin) the same way, then re-center into the
    # expanded canvas below.
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    rotated = corners @ rot.T
    new_w = abs(w * cos_t) + abs(h * sin_t)
    new_h = abs(w * sin_t) + abs(h * cos_t)
    return rotated + np.array([new_w / 2, new_h / 2])


def assign_group_and_rotations(n_panels: int, rng: random.Random) -> tuple[str, list[int]]:
    group = rng.choice(["a", "b"])
    if group == "a":
        r = rng.choice(DISCRETE_ROTATIONS)
        return group, [r] * n_panels
    # mixed: draw independently, then force at least 2 distinct values if an
    # unlucky draw came out all-the-same (rare but possible with 4 choices).
    rotations = [rng.choice(DISCRETE_ROTATIONS) for _ in range(n_panels)]
    if len(set(rotations)) < 2:
        idx = rng.randrange(n_panels)
        rotations[idx] = rng.choice([r for r in DISCRETE_ROTATIONS if r != rotations[0]])
    return group, rotations


def compose_page(panels: list[dict], page_id: str, output_dir: Path, rng: random.Random) -> list[dict]:
    group, rotations = assign_group_and_rotations(len(panels), rng)
    placed = []
    for panel, rotation in zip(panels, rotations):
        tilt = rng.uniform(*TILT_RANGE_DEG)
        total_angle = rotation + tilt
        img = Image.open(panel["panel_path"]).convert("RGB")
        w, h = img.size
        rotated = rotate_panel(img, total_angle)
        corners = corners_after_rotation(w, h, total_angle)
        placed.append({"panel": panel, "group": group, "rotation_deg": rotation, "tilt_deg": round(tilt, 3),
                        "total_angle_deg": round(total_angle, 3), "rotated_img": rotated, "corners": corners})

    # Simple non-overlapping row-flow layout: left-to-right, wrap on
    # overflow, real margins between panels -- not random bin-packing, since
    # what matters for ground truth is precise per-panel geometry, not a
    # visually random arrangement.
    max_row_width = 2200
    x_cursor, y_cursor, row_height = MARGIN_PX, MARGIN_PX, 0
    positions = []
    for p in placed:
        pw, ph = p["rotated_img"].size
        if x_cursor + pw + MARGIN_PX > max_row_width and x_cursor > MARGIN_PX:
            x_cursor = MARGIN_PX
            y_cursor += row_height + MARGIN_PX
            row_height = 0
        positions.append((x_cursor, y_cursor))
        x_cursor += pw + MARGIN_PX
        row_height = max(row_height, ph)
    page_w = max_row_width
    page_h = y_cursor + row_height + MARGIN_PX

    canvas = Image.new("RGB", (page_w, page_h), PAGE_BG)
    page_path = output_dir / f"{page_id}.png"
    rows_out = []
    panel_boxes = []
    for i, (p, (px, py)) in enumerate(zip(placed, positions)):
        canvas.paste(p["rotated_img"], (px, py))
        corners = p["corners"] + np.array([px, py])
        pw, ph = p["rotated_img"].size
        panel_boxes.append((px, py, px + pw, py + ph))
        rows_out.append({
            "page_id": page_id, "panel_index": i,
            "source_panel_id": p["panel"]["id"], "record_id": p["panel"]["record_id"],
            "lead_names": p["panel"]["lead_names"], "group": p["group"],
            "rotation_deg": p["rotation_deg"], "tilt_deg": p["tilt_deg"], "total_angle_deg": p["total_angle_deg"],
            "x0": round(corners[0, 0], 1), "y0": round(corners[0, 1], 1),
            "x1": round(corners[1, 0], 1), "y1": round(corners[1, 1], 1),
            "x2": round(corners[2, 0], 1), "y2": round(corners[2, 1], 1),
            "x3": round(corners[3, 0], 1), "y3": round(corners[3, 1], 1),
            "page_path": str(page_path),
        })

    # Page-level noise, drawn on the canvas after every panel is placed so
    # it can never disturb the geometry already recorded above.
    draw = ImageDraw.Draw(canvas)
    draw_staple_marks(draw, panel_boxes, rng)
    draw_handwriting_noise(draw, page_w, page_h, panel_boxes, rng)
    draw_black_stickers(draw, panel_boxes, rng)

    canvas = apply_shadow(canvas, rng)
    canvas, rows_out = apply_page_perspective(canvas, rows_out, rng)

    canvas.save(page_path)
    return rows_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Composite ground-truth panels onto synthetic pages.")
    parser.add_argument("--panels-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-pages", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pages_dir = args.output_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    with args.panels_manifest.open(encoding="utf-8") as f:
        all_panels = list(csv.DictReader(f))
    if not all_panels:
        raise SystemExit("panels manifest is empty")

    all_rows = []
    for i in range(args.num_pages):
        page_id = f"synth-page-{i:04d}"
        n = rng.randint(MIN_PANELS_PER_PAGE, MAX_PANELS_PER_PAGE)
        panels = rng.choices(all_panels, k=n)  # sampling with replacement is fine: geometry ground truth is per-placement, not per-source-panel
        rows = compose_page(panels, page_id, pages_dir, rng)
        all_rows.extend(rows)
        print(f"{page_id}: {n} panel(s), group {rows[0]['group']}")

    gt_path = args.output_root / "ground_truth.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"done, {args.num_pages} page(s), {len(all_rows)} panel placement(s), ground truth at {gt_path}")


if __name__ == "__main__":
    main()
