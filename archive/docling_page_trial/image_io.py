from pathlib import Path

from PIL import Image, ImageOps


def load_upright(path: Path, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    img.thumbnail((max_side, max_side))
    return img
