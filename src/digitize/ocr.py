"""Text boxes from RapidOCR (the OCR engine docling uses), called directly."""
import numpy as np

_reader = None


def get_reader(threads: int = 8):
    global _reader
    if _reader is None:
        import torch
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import RapidOcrOptions
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        torch.set_num_threads(threads)
        _reader = RapidOcrModel(enabled=True, artifacts_path=None, options=RapidOcrOptions(backend="torch"),
                                accelerator_options=AcceleratorOptions(num_threads=threads)).reader
    return _reader


def read_text(rgb: np.ndarray, threads: int = 8) -> list[dict]:
    """[{text, bbox: [x0, y0, x1, y1]}] in the pixel frame of `rgb`."""
    res = get_reader(threads)(rgb)
    if res.boxes is None:
        return []
    return [{"text": t, "bbox": [float(np.array(b)[:, 0].min()), float(np.array(b)[:, 1].min()), float(np.array(b)[:, 0].max()), float(np.array(b)[:, 1].max())]}
            for b, t in zip(res.boxes, res.txts or [])]
