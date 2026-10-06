"""Run DocTr (external/DocTr) on CPU over a few pages and save the input,
the geometrically unwarped image, and the illumination-corrected image side by
side. The third-party code is left untouched: its .cuda() calls are turned into
no-ops here and the weights are loaded on the CPU.

Usage (from the project root):
    .venvs/doctr/Scripts/python.exe results/analyses/doctr_trial/run_doctr_trial.py --out-dir results/inferences/doctr_trial --max-side 1600 <image> ...
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

DOCTR = Path(__file__).resolve().parents[3] / "external" / "DocTr"
sys.path.insert(0, str(DOCTR))
torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self

from GeoTr import GeoTr  # noqa: E402
from IllTr import IllTr  # noqa: E402
from inference_ill import rec_ill  # noqa: E402
from seg import U2NETP  # noqa: E402


class GeoTrSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk = (self.msk(x)[0] > 0.5).float()
        bm = self.GeoTr(msk * x)
        return (2 * (bm / 286.8) - 1) * 0.99


def load(model: nn.Module, path: Path, strip: int) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    own = model.state_dict()
    model.load_state_dict({**own, **{k[strip:]: v for k, v in state.items() if k[strip:] in own}})
    return model.eval()


def unwarp(model: GeoTrSeg, im: np.ndarray) -> np.ndarray:
    h, w, _ = im.shape
    small = torch.from_numpy(cv2.resize(im, (288, 288)).transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        bm = model(small).cpu()
    flow = [cv2.blur(cv2.resize(bm[0, i].numpy(), (w, h)), (3, 3)) for i in (0, 1)]
    grid = torch.from_numpy(np.stack(flow, axis=2)).unsqueeze(0)
    out = F.grid_sample(torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float(), grid, align_corners=True)
    return (out[0] * 255).permute(1, 2, 0).numpy()[:, :, ::-1].astype(np.uint8)  # BGR uint8


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 16), "white")
    out.paste(img, (0, 16))
    ImageDraw.Draw(out).text((3, 2), text, fill=(200, 0, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("images", type=Path, nargs="+")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre = DOCTR / "model_pretrained"
    geo_model = GeoTrSeg()
    load(geo_model.msk, pre / "seg.pth", 6)
    load(geo_model.GeoTr, pre / "geotr.pth", 7)
    ill_model = load(IllTr(), pre / "illtr.pth", 7)
    geo_model.eval()
    for path in tqdm(args.images, desc="pages"):
        stem = f"{path.parent.name}__{path.stem}".replace(" ", "_").replace(",", "")
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        im = np.asarray(img) / 255.0
        t0 = time.time()
        geo = unwarp(geo_model, im)
        t1 = time.time()
        ill_path = args.out_dir / f"{stem}_ill.png"
        cv2.imwrite(str(args.out_dir / f"{stem}_geo.png"), geo)
        rec_ill(ill_model, geo, saveRecPath=str(ill_path))
        t2 = time.time()
        geo_img = Image.fromarray(geo[:, :, ::-1])
        ill_img = Image.open(ill_path).convert("RGB")
        tiles = [label(img, f"input {img.size}"), label(geo_img, f"unwarped ({t1 - t0:.0f} s)"), label(ill_img, f"unwarped + illumination ({t2 - t1:.0f} s)")]
        h = max(t.height for t in tiles)
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 12, h), "white")
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.width + 6
        sheet.save(args.out_dir / f"{stem}_side_by_side.png")


if __name__ == "__main__":
    main()
