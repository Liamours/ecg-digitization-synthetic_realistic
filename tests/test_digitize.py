"""Fast checks of the digitize package on synthetic data: no OCR, no U-Net, no dataset needed."""
import numpy as np

from src.digitize.gridmap import fit_dot_grid
from src.digitize.leads import split_rows, track_leads
from src.digitize.record import Grid, Lead
from src.digitize.sample import sample_lead
from src.digitize.selftest import check_leads, pulse_check
from src.digitize.text import read_gain, read_labels

GRID_CFG = {"blackhat_kernel": 9, "blackhat_contrast": 28, "dot_area": [4, 45], "dot_max_side": 9, "pitch_px_range": [9.0, 15.0], "angle_range_deg": 6.0}
SAMPLING = {"bin_mm": 0.1, "min_bin_px": 1.2, "mm_per_s": 25.0, "baseline_window_bins": 120}
PX = 12.0


def dot_page(angle_deg: float, size: int = 900) -> tuple[np.ndarray, Grid]:
    """A white page with a dot lattice of 12 px per step tilted by angle_deg."""
    import cv2

    img = np.full((size, size), 255, np.uint8)
    r = np.radians(angle_deg)
    a1, a2 = PX * np.array([np.cos(r), np.sin(r)]), PX * np.array([-np.sin(r), np.cos(r)])
    o = np.array([37.3, 51.1])
    for i in range(-5, 100):
        for j in range(-5, 100):
            x, y = o + i * a1 + j * a2
            if 4 < x < size - 4 and 4 < y < size - 4:
                cv2.circle(img, (int(round(x)), int(round(y))), 1, 0, -1)
    return img, Grid("dot", o, a1, a2)


def test_dot_grid_recovers_tilt_and_pitch():
    img, truth = dot_page(-1.4)
    fit = fit_dot_grid(img, GRID_CFG)
    assert abs(fit.tilt_deg - (-1.4)) < 0.1
    assert abs(fit.px_per_mm[0] - PX) < 0.1 and abs(fit.px_per_mm[1] - PX) < 0.1
    assert fit.rms_px < 1.0


def test_sampling_recovers_a_tilted_trace():
    """A sine of known mV drawn along the tilted grid comes back with the right amplitude and period."""
    img, grid = dot_page(-2.0)
    gain = 10.0
    x_mm = np.arange(5, 60, 0.05)
    mv = 0.5 * np.sin(2 * np.pi * x_mm / 25.0 * 1.2)   # 1.2 Hz at 25 mm/s
    px = grid.to_px(np.column_stack([x_mm, 20 - mv * gain]))
    mask = np.zeros(img.shape, np.uint8)
    mask[np.round(px[:, 1]).astype(int), np.round(px[:, 0]).astype(int)] = 1
    t, out, ok, base, bin_mm = sample_lead(mask, grid, gain, (5.0, 60.0), SAMPLING)
    assert abs(np.ptp(out) - 1.0) < 0.1
    assert ok.mean() > 0.95


def test_rows_and_tracking_separate_two_leads():
    mask = np.zeros((200, 300), np.uint8)
    xs = np.arange(300)
    mask[(50 + 10 * np.sin(xs / 20)).astype(int), xs] = 1
    mask[(150 + 10 * np.sin(xs / 25)).astype(int), xs] = 1
    bands = split_rows(mask, 2, "equal")
    masks, crossing = track_leads(mask, bands)
    assert masks[0].sum() > 250 and masks[1].sum() > 250 and crossing == 0


def test_einthoven_and_lag_search():
    t = np.arange(600) * 0.004
    beat = np.exp(-((t % 0.8 - 0.3) ** 2) / 0.001)
    i, iii = 0.6 * beat, 0.4 * beat
    leads = {k: Lead(k, t, v, 0.0, 0.0, 1.0, 0.0) for k, v in (("I", i), ("II", i + iii), ("III", iii))}
    out = check_leads(leads, 40)
    assert out["II=I+III"]["corr"] > 0.99
    shifted = {**leads, "II": Lead("II", t, np.roll(i + iii, 5), 0.0, 0.0, 1.0, 0.0)}
    assert check_leads(shifted, 40)["II=I+III"]["best_lag_bins"] == 5


def test_pulse_and_text_rules():
    assert pulse_check([9.9, 10.1], 10.0)["pulse_mV"] == 1.0
    assert read_gain([{"text": "25mm/s"}, {"text": "5mm/mU"}], [5.0, 10.0]) == 5.0
    assert read_gain([{"text": "0mm/mV"}], [5.0, 10.0]) is None
    sets = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
    assert read_labels([{"text": "*U4"}], sets, 0) == (sets[3], "ocr")
    assert read_labels([{"text": "MAC 400"}, {"text": "U1.02"}], sets, 2) == (sets[2], "position")
    assert read_labels([{"text": "V1.02"}, {"text": "aVR"}], sets, 0) == (sets[1], "ocr")


def test_report_writes_png(tmp_path):
    """report.make reads only run.py's files: page.json, panel json and csv, an overlay."""
    import json

    import cv2

    from src.digitize.report import make

    cv2.imwrite(str(tmp_path / "overlay_upright.png"), np.full((200, 150, 3), 255, np.uint8))
    (tmp_path / "page.json").write_text(json.dumps({"source": "x.jpg", "layout": "mac400", "rotation_ccw_deg": 0, "panels": {"panel0": {}}}), encoding="utf-8")
    (tmp_path / "panel0.json").write_text(json.dumps({"gain_mm_per_mV": 10.0, "gain_source": "ocr", "error": ""}), encoding="utf-8")
    t = np.arange(0, 2, 0.004)
    rows = ["time_s,I_mV,I_flag"] + [f"{x:.4f},{np.sin(6 * x):.4f},ok" for x in t]
    (tmp_path / "panel0.csv").write_text("\n".join(rows), encoding="utf-8")
    assert make(tmp_path).stat().st_size > 5000


def test_waveform_score():
    """The same waveform on a shifted clock scores near its coverage, a flipped one and a half-length one score low."""
    from src.digitize.score import waveform_score

    fs = 500.0
    rng = np.random.default_rng(0)
    gt = np.convolve(rng.normal(size=1600), np.ones(25) / 25, mode="same") * 4  # smooth and not periodic, so a shift cannot fake a flip
    t = np.arange(0, 3.0, 0.004)
    same = np.interp(t + 0.1, np.arange(len(gt)) / fs, gt)
    assert waveform_score(gt, fs, t, same)["score"] > 90  # 3.0 s of 3.2 s, so coverage caps it near 94
    assert waveform_score(gt, fs, t, -same)["score"] < 15
    assert waveform_score(gt, fs, t, np.zeros_like(t))["score"] < 15
    assert waveform_score(gt, fs, t[: len(t) // 2], same[: len(t) // 2])["score"] < 55


def test_entry_points_import():
    """The command line modules are not otherwise imported by the tests, so a syntax slip in one would go unseen."""
    import src.digitize.pipeline  # noqa: F401
    import src.digitize.report  # noqa: F401
    import src.digitize.run  # noqa: F401
    import src.digitize.score  # noqa: F401
