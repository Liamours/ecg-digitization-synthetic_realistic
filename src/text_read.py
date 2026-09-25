import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter
from PIL import Image, ImageOps


def load_upright(path: Path, max_side: int) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    img.thumbnail((max_side, max_side))
    return img


def read_texts(converter: DocumentConverter, img: Image.Image) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "page.png"
        img.save(f)
        doc = converter.convert(str(f)).document
    page = next(iter(doc.pages.values()))
    scale = img.height / page.size.height
    texts = []
    for t in doc.texts:
        for pr in t.prov:
            b = pr.bbox.to_top_left_origin(page.size.height)
            texts.append({"text": t.text, "bbox": [round(v * scale) for v in (b.l, b.t, b.r, b.b)]})
    return texts
