import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter
from PIL import Image

from src.image_io import load_upright  # noqa: F401  (re-exported for src.run_pages)


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
