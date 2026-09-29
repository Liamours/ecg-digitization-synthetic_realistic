from pathlib import Path

import cv2
import numpy as np  # cv2 arrives with docling's OCR dependencies

INK_CONTRAST = 25  # gray levels darker than the local paper level
PAPER_MIN = 40  # darker than this is scan border, not paper


def ink_fraction(path: Path, width: int = 800) -> float:
    """Share of paper pixels darker than their surroundings. ECG paper (grid
    and trace) scores about 0.09 to 0.27; a blank or handwritten form scores
    below 0.05 and an empty scan about 0."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    scale = width / max(gray.shape)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    paper = gray > PAPER_MIN
    if paper.sum() == 0:
        return 0.0
    background = cv2.medianBlur(gray, 31).astype(np.float32)
    dark = (background - gray.astype(np.float32)) > INK_CONTRAST
    return float((dark & paper).sum() / paper.sum())
