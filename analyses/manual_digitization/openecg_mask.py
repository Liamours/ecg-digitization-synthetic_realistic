"""Run only the Open-ECG-Digitizer U-Net (external/Open-ECG-Digitizer, weights unet_weights_07072025.pt) on an image:
class probabilities for grid, text/background, signal (trace), background; no perspective correction, no layout step.
Usage (from the project root, venv .venvs/openecg): python results/analyses/manual_digitization/openecg_mask.py <image> <out png> [max_side]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "external" / "Open-ECG-Digitizer"
sys.path.insert(0, str(REPO))
from src.model.unet import UNet  # noqa: E402

SIGNAL_CLASS = 2
_model = None


def model() -> UNet:
    global _model
    if _model is None:
        m = UNet(num_in_channels=3, num_out_channels=4, dims=[32, 64, 128, 256, 320, 320, 320, 320], depth=2)
        ck = torch.load(REPO / "weights" / "unet_weights_07072025.pt", weights_only=True, map_location="cpu")
        if isinstance(ck, tuple):
            ck = ck[0]
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.items()})
        _model = m.eval()
    return _model


def signal_probability(bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Probability map of the trace class at the input's own resolution."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (round(w * s) // 32 * 32, round(h * s) // 32 * 32), interpolation=cv2.INTER_AREA) if s < 1 else bgr[:h // 32 * 32, :w // 32 * 32]
    x = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = (x - x.min()) / (x.max() - x.min())
    with torch.no_grad():
        p = torch.softmax(model()(x), dim=1)[0, SIGNAL_CLASS].numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    p = signal_probability(img, int(sys.argv[3]) if len(sys.argv) > 3 else 1600)
    cv2.imwrite(sys.argv[2], (p * 255).astype(np.uint8))
    print("signal probability: mean %.4f, pixels > 0.5: %d" % (p.mean(), int((p > 0.5).sum())))
