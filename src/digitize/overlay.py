"""Picture of what each lead-separation method does on the panels of one saved page: one tile per method, the pixels
given to each lead in its own colour, unassigned trace left grey. Also prints each lead's flag and coverage.

Usage:
    python -m src.digitize.overlay --config configs/digitize.yml --layout mac400 --front @inferences/front_mac400-scan --page scan_29__Image_20260922_0015 --out-dir @inferences/methods_mac400-scan/overlays [--device cuda]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import paths
from src.digitize import leads as leadmod
from src.digitize.pipeline import Digitizer, cut_at_gap, load_layout
from src.digitize.record import PanelFront
from src.digitize.separate import METHODS

COLORS = [(0, 0, 255), (0, 160, 0), (255, 0, 0)]  # BGR, one per row of a panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--front", type=paths.resolve, required=True, help="folder written by src.digitize.front")
    ap.add_argument("--page", required=True, help="page folder name inside the front folder")
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    ap.add_argument("--device", help="cpu, cuda or auto; overrides the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    d = Digitizer(cfg, load_layout(args.layout))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted((args.front / args.page).glob("panel*.front.json")):
        front = PanelFront.load(f)
        if front.error:
            continue
        (x0, _, x1, _), (cx0, _), crop = front.box, front.crop_origin, front.crop
        p = d.unet.probability(crop)
        mask = (p > cfg["unet"]["threshold"]).astype(np.uint8)  # the same cuts as Digitizer.trace
        counts = [("trace probability above the threshold", int(mask.sum()))]
        mask[:front.zone[0]] = 0
        mask[front.zone[1]:] = 0
        counts.append((f"between the header and footer text, rows {front.zone[0]} to {front.zone[1]} of {crop.shape[0]}", int(mask.sum())))
        mask[:, :x0 - cx0] = 0
        mask[:, x1 - cx0:] = 0
        counts.append((f"inside the panel box, columns {x0 - cx0} to {x1 - cx0} of {crop.shape[1]}", int(mask.sum())))
        print(f"{args.page} panel{front.index} trace pixels: " + "; ".join(f"{k}: {v}" for k, v in counts) + f"; highest probability {p.max():.2f}")
        seen = cv2.cvtColor((255 * np.clip(p, 0, 1)).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # what the trace network sees, with the rows kept between the text bands
        for y in front.zone:
            cv2.line(seen, (0, int(y)), (seen.shape[1], int(y)), (0, 200, 255), 2)
        cv2.imwrite(str(args.out_dir / f"{args.page}_panel{front.index}_trace_probability.jpg"), np.vstack([crop, seen]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            mask = cut_at_gap(mask, d.layout["cut_gap_px"])
        except ValueError as exc:
            print(f"{args.page} panel{front.index}: {exc}")
            continue
        bands = leadmod.split_rows(mask, d.layout["rows"], d.layout["rows_mode"])
        tiles = []
        for how in ("baseline", "overlap", *METHODS):
            t0 = time.time()
            try:
                if how in ("baseline", "overlap"):
                    masks = (leadmod.track_leads_overlap if how == "overlap" else leadmod.track_leads)(mask, bands)[0]
                else:
                    masks = d.separator.split(how, mask, p * mask, bands)
                err = ""
            except Exception as exc:
                masks, err = [], f"{type(exc).__name__}: {exc}"
            sec = time.time() - t0
            rec = d.trace(front, tracking=how)
            print(f"{args.page} panel{front.index} {how:20s} {sec:5.1f} s  " + " ".join(f"{k}:{v.flag}({v.coverage:.2f})" for k, v in rec.leads.items()) + f" {err} {rec.error}")
            img = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            img = (0.5 * img + 127).astype(np.uint8)
            for m, c in zip(masks, COLORS):
                img[cv2.dilate(m, np.ones((2, 2), np.uint8)) > 0] = c
            cv2.putText(img, how, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            tiles.append(img)
        cv2.imwrite(str(args.out_dir / f"{args.page}_panel{front.index}.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
