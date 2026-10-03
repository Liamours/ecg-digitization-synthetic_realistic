"""Lead separation by open-source digitizers, run on one panel's trace mask.

Each method returns lines (one row position per image column, NaN where the lead has no trace); `lines_to_masks` gives
each of the panel's row bands its longest line, so the result goes through the same trimming and sampling as the other
lead-separation methods (src.digitize.leads).

`openecg_lines`: Open-ECG-Digitizer's SignalExtractor (third_party/open_ecg_digitizer/signal_extractor.py) on the trace
probability: connected regions become lines, a region holding more than one trace is split by tracing a horizontal path
through it, and line pieces are matched end to start and merged.

`ecgtizer_lazy`, `ecgtizer_full`, `ecgtizer_fragmented`: ecgtizer's track cutting (rows cut midway between the peaks of
the row variance, third_party/ecgtizer/PDF2XML.py) and one of its three per-column extractions
(third_party/ecgtizer/extraction_functions.py). `ecgtizer_fragmented` is the canonical pipeline's lead separation.

The third-party files are unmodified copies; sources, commits and licenses are in third_party/README.md.
"""
import numpy as np

METHODS = ("openecg_lines", "ecgtizer_lazy", "ecgtizer_full", "ecgtizer_fragmented")


REACH = 2  # px a line may sit off the trace run it belongs to


def lines_to_masks(lines: np.ndarray, bands: list[tuple[int, int]], mask: np.ndarray) -> list[np.ndarray]:
    """One mask per row band: the trace pixels owned by the line whose median row lies in the band (of several, the one
    with most columns).

    A line gives one row per column, the middle of a steep stroke, so a lead takes the whole vertical run of trace pixels
    its line passes through and the sampler keeps the peak heights. A run two lines pass through is split midway between
    them; a line point on no trace pixel owns nothing."""
    h, w = mask.shape
    lines = np.asarray(lines, np.float64).reshape(-1, w)
    valid = np.isfinite(lines)
    mid = np.array([np.median(ln[v]) if v.any() else np.nan for ln, v in zip(lines, valid)])
    chosen = []
    for b0, b1 in bands:
        inside = [i for i in range(len(lines)) if b0 <= mid[i] < b1]
        chosen.append(max(inside, key=lambda j: valid[j].sum()) if inside else None)
    out = [np.zeros_like(mask) for _ in bands]
    for x in np.flatnonzero(mask.any(axis=0)):
        ys = np.flatnonzero(mask[:, x])
        cut = np.flatnonzero(np.diff(ys) > 1)
        for a, b in zip(ys[np.r_[0, cut + 1]], ys[np.r_[cut, len(ys) - 1]]):
            own = sorted((lines[i, x], k) for k, i in enumerate(chosen) if i is not None and valid[i, x] and a - REACH <= lines[i, x] <= b + REACH)
            edges = [a] + [int(round((p[0] + q[0]) / 2)) for p, q in zip(own[:-1], own[1:])] + [b + 1]
            for (_, k), lo, hi in zip(own, edges[:-1], edges[1:]):
                out[k][lo:hi, x] = 1
    return out


class Separator:
    def __init__(self):
        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            from third_party.open_ecg_digitizer.signal_extractor import SignalExtractor

            self._extractor = SignalExtractor()
        return self._extractor

    @property
    def ecgtizer(self):
        from third_party.ecgtizer import PDF2XML  # imported on first use: it needs pdf2image and matplotlib

        return PDF2XML

    def split(self, method: str, mask: np.ndarray, soft: np.ndarray, bands: list[tuple[int, int]]) -> list[np.ndarray]:
        lines = self.openecg_lines(soft) if method == "openecg_lines" else self.ecgtizer_lines(mask, method.split("_", 1)[1])
        return lines_to_masks(lines, bands, mask)

    def openecg_lines(self, soft: np.ndarray) -> np.ndarray:
        """`soft`: the trace probability, zero outside the panel's trace area."""
        import torch

        ext = self.extractor
        fmap = torch.from_numpy(np.ascontiguousarray(soft, np.float32))
        fmap = (fmap - fmap.mean()).clamp(min=0)
        fmap = fmap / (fmap.max() + 1e-9)  # as InferenceWrapper.process_sparse_prob
        pieces = [ln for ln in ext._iterative_extraction(fmap) if (~torch.isnan(ln)).sum() > ext.min_line_width]  # SignalExtractor.__call__ without its print-only peak count
        out = np.full((0, soft.shape[1]), np.nan)
        if not pieces:
            return out
        stacked = torch.stack(pieces)
        first = int(torch.nonzero(stacked.nan_to_num(0.0).abs().sum(0) > 0)[0])  # match_and_merge_lines crops to the first column that holds a line
        merged, _ = ext.match_and_merge_lines(stacked)
        if not merged:
            return out
        got = torch.stack(merged).numpy()
        out = np.full((len(got), soft.shape[1]), np.nan)
        out[:, first:first + got.shape[1]] = got
        return out

    def ecgtizer_lines(self, mask: np.ndarray, how: str) -> np.ndarray:
        ez = self.ecgtizer
        h, w = mask.shape
        gray = np.where(mask > 0, 0, 255).astype(np.uint8)  # ecgtizer binarizes a dark trace on light paper; the mask goes in as that picture
        tracks, peaks, v_start = ez.tracks_extraction(gray, "classic", 300, "multilead", NOISE=True)
        # the row each track starts at, as tracks_extraction cuts them (it returns the track images without their offsets)
        if len(peaks) >= 2:
            gap = int(np.median(np.diff(peaks)))
            cuts = [max(0, peaks[0] - gap // 2)] + [int((a + b) / 2) for a, b in zip(peaks[:-1], peaks[1:])] + [min(h, peaks[-1] + gap // 2)]
        else:
            cuts = [0, h]
        if len(cuts) == 6:
            del cuts[0]
        tops = [c + (int(0.02 * h) if i == 0 else 0) for i, c in enumerate(cuts[:-1])]
        extract = {"lazy": ez.lazy_extraction, "full": ez.full_extraction, "fragmented": ez.fragmented_extraction}[how]
        out = np.full((len(tracks), w), np.nan)
        for i, top in zip(sorted(tracks), tops):
            track = np.ascontiguousarray(tracks[i])
            if track.size == 0:
                continue
            y = np.asarray(extract(track), np.float64)[:track.shape[1]]
            lit = (track == 255).any(axis=0)[:len(y)]
            out[i, v_start:v_start + len(y)] = np.where(lit, y + top, np.nan)  # columns without trace stay empty; the extractors hold or zero them
        return out
