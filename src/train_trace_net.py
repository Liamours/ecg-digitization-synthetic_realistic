"""Train the trace segmentation network on synthetic panels with exact trace masks.

A panel's mask is built from its three lead masks at their rows in the panel. Training reads the train dataset, validation
a dataset of another seed. Every epoch writes a line to the log and to progress.csv and saves last.pt (model, optimizer,
epoch), so a stopped run continues with --run <name>; best.pt is the model with the highest validation Dice.

Usage:
    python -m src.train_trace_net --config configs/trace_net.yml [--run <name>] [--limit N] [--epochs E]
"""
import argparse
import csv
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import paths
from src.digitize.tracenet import TraceNet, to_tensor


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def panel_items(dataset: Path, columns: list[str]) -> list[tuple[Path, list[tuple[Path, int, int]]]]:
    """(image path, [(lead mask path, y_top, y_bottom)]) for every panel image listed in `columns`."""
    root = dataset.parent.parent  # label paths start at the project root
    leads = defaultdict(list)
    for r in read_rows(dataset / "labels" / "manifest_leds.csv"):
        leads[r["panel_id"]].append((root / r["mask_image_path"], int(r["y_top"]), int(r["y_bottom"])))
    items = []
    for r in read_rows(dataset / "labels" / "manifest_panels.csv"):
        for c in columns:
            if r.get(c) and leads[r["panel_id"]] and (root / r[c]).exists():
                items.append((root / r[c], leads[r["panel_id"]]))
    return items


def load_pair(item: tuple, dilate: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(item[0]))
    mask = np.zeros(image.shape[:2], np.uint8)
    for path, y0, y1 in item[1]:
        m = cv2.imread(str(path), 0)
        h = min(y1 - y0, m.shape[0], mask.shape[0] - y0)
        w = min(m.shape[1], mask.shape[1])
        mask[y0:y0 + h, :w] = np.maximum(mask[y0:y0 + h, :w], (m[:h, :w] > 127).astype(np.uint8))
    if dilate:
        mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return image, mask


def augment(image: np.ndarray, mask: np.ndarray, cfg: dict, crop: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = cfg["augment"]
    s = rng.uniform(*a["scale"])
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = (cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
    img = image.astype(np.float32) / 255.0
    if rng.random() < a["fade_prob"]:  # a faint grey trace: its ink moves toward the local paper colour
        keep = rng.uniform(*a["fade"])
        paper = cv2.blur(cv2.dilate(img, np.ones((9, 9), np.uint8)), (15, 15))
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)[..., None]
        img = img * (1 - soft * (1 - keep)) + paper * soft * (1 - keep)
    img = img * rng.uniform(*a["gain"], size=3).astype(np.float32)
    if rng.random() < a["gray_prob"]:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    img = np.clip(img, 0, 1) ** rng.uniform(*a["gamma"])
    sigma = rng.uniform(*a["blur_sigma"])
    if sigma > 0.15:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = np.clip(img + rng.normal(0, rng.uniform(*a["noise_std"]), img.shape).astype(np.float32), 0, 1)
    ch, cw = crop
    ph, pw = max(ch - img.shape[0], 0), max(cw - img.shape[1], 0)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
        mask = np.pad(mask, ((0, ph), (0, pw)))
    y, x = rng.integers(0, img.shape[0] - ch + 1), rng.integers(0, img.shape[1] - cw + 1)
    return (img[y:y + ch, x:x + cw] * 255).astype(np.uint8), mask[y:y + ch, x:x + cw]


class Panels(Dataset):
    def __init__(self, items: list, cfg: dict, train: bool):
        self.items, self.cfg, self.train = items, cfg, train
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        image, mask = load_pair(self.items[i], self.cfg["mask_dilate_px"])
        if self.train:
            image, mask = augment(image, mask, self.cfg, tuple(self.cfg["crop"]), np.random.default_rng(self.cfg["seed"] + 100003 * self.epoch + i))
        else:  # the whole panel, padded to a size the network can halve `depth` times
            m = 2 ** self.cfg["model"]["depth"]
            ph, pw = -image.shape[0] % m, -image.shape[1] % m
            image, mask = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="edge"), np.pad(mask, ((0, ph), (0, pw)))
        return to_tensor(image), torch.from_numpy(mask.astype(np.float32))[None]


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, target) + dice.mean()


@torch.no_grad()
def validate(net: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    net.eval()
    inter = pred = true = 0.0
    for x, y in loader:
        p = (torch.sigmoid(net(x.to(device))) > 0.5).float().cpu()
        inter, pred, true = inter + float((p * y).sum()), pred + float(p.sum()), true + float(y.sum())
    return 2 * inter / max(pred + true, 1), inter / max(pred + true - inter, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", help="continue this run folder; a new one is named by the start time")
    ap.add_argument("--limit", type=int, help="first N panels of each set, for a quick check")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    epochs = args.epochs or cfg["epochs"]
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = paths.resolve(cfg["out_dir"]) / (args.run or datetime.now().strftime("%Y%m%d-%H%M"))
    out.mkdir(parents=True, exist_ok=True)
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"train_trace_net-{out.name}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("train")
    train_items = panel_items(paths.resolve(cfg["train_dataset"]), cfg["image_columns"])[:args.limit]
    val_items = panel_items(paths.resolve(cfg["val_dataset"]), cfg["image_columns"])[:args.limit]
    train_set = Panels(train_items, cfg, True)
    train = DataLoader(train_set, batch_size=cfg["batch"], shuffle=True, num_workers=cfg["workers"], drop_last=True, persistent_workers=cfg["workers"] > 0)
    val = DataLoader(Panels(val_items, cfg, False), batch_size=1, num_workers=0)
    net = TraceNet(cfg["model"]["base"], cfg["model"]["depth"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    start, best = 0, 0.0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"], ck["best"]
    progress = out / "progress.csv"
    if not progress.exists():
        progress.write_text("epoch,train_loss,val_dice,val_iou,seconds\n", encoding="utf-8")
    log.info("run %s on %s: %d train and %d validation panel images, %d parameters, epochs %d to %d", out.name, device, len(train_items), len(val_items), sum(p.numel() for p in net.parameters()), start + 1, epochs)
    times = []
    for epoch in range(start, epochs):
        t0 = time.time()
        train_set.epoch = epoch
        net.train()
        total = 0.0
        for x, y in tqdm(train, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss)
        sched.step()
        dice, iou = validate(net, val, device)
        sec = time.time() - t0
        times.append(sec)
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch + 1},{total / max(len(train), 1):.4f},{dice:.4f},{iou:.4f},{sec:.0f}\n")
        if dice > best:
            best = dice
            torch.save({"state": net.state_dict(), "base": cfg["model"]["base"], "depth": cfg["model"]["depth"], "train_px_per_mm": cfg["train_px_per_mm"], "epoch": epoch + 1, "val_dice": dice}, out / "best.pt")
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch + 1, "best": best}, last)
        eta = timedelta(seconds=int(np.mean(times) * (epochs - epoch - 1)))
        log.info("epoch %d/%d: loss %.4f, val dice %.4f, iou %.4f, %.0f s, best %.4f, ETA %s", epoch + 1, epochs, total / max(len(train), 1), dice, iou, sec, best, eta)
        print(f"epoch {epoch + 1}/{epochs}: loss {total / max(len(train), 1):.4f}, val dice {dice:.4f}, iou {iou:.4f}, {sec:.0f} s, ETA {eta}")
    print(f"best validation dice {best:.4f}; weights {out / 'best.pt'}")


if __name__ == "__main__":
    main()
