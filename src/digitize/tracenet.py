"""Trace segmentation network: which pixels of a panel crop are ECG trace.

A small U-Net trained on synthetic panels whose trace masks are exact (src.train_trace_net). Printed text, lead names, the calibration step, the header icon and the grid are background by construction of
the labels. The network works at the generator's scale (`train_px_per_mm`), so a crop is resized to that scale using the
pitch of its own fitted grid, and the probability map is resized back.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TraceNet(nn.Module):
    def __init__(self, base: int = 24, depth: int = 4):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(3 if i == 0 else chans[i - 1], chans[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth))])
        self.merge = nn.ModuleList([block(2 * chans[i], chans[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(base, 1, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else nn.functional.max_pool2d(x, 2))
            skips.append(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1])):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1).float() / 255.0


class TraceMask:
    """Loads a checkpoint once and returns trace probabilities for panel crops."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        ck = torch.load(Path(cfg["weights"]), map_location="cpu", weights_only=True)
        self.net = TraceNet(ck["base"], ck["depth"])
        self.net.load_state_dict(ck["state"])
        self.device = device
        self.net.eval().to(device)
        self.px_per_mm = ck["train_px_per_mm"]
        self.threshold = cfg["threshold"]
        self.multiple = 2 ** ck["depth"]

    def probability(self, bgr: np.ndarray, px_per_mm: float) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.px_per_mm / px_per_mm
        nh, nw = (max(self.multiple, int(round(v * s / self.multiple)) * self.multiple) for v in (h, w))
        x = to_tensor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR))[None].to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.net(x))[0, 0].cpu().numpy()
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def mask(self, bgr: np.ndarray, px_per_mm: float) -> np.ndarray:
        return (self.probability(bgr, px_per_mm) > self.threshold).astype(np.uint8)
