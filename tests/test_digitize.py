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
    import src.digitize.checks  # noqa: F401
    import src.digitize.pipeline  # noqa: F401
    import src.digitize.report  # noqa: F401
    import src.digitize.run  # noqa: F401
    import src.digitize.score  # noqa: F401


def test_paths_resolve_roots_and_dataset_aliases():
    from pathlib import Path

    from src import paths

    assert paths.resolve("@logs") == paths.root("logs")
    assert paths.resolve("@mac400-scan/scan_29/a.jpg") == paths.dataset("mac400-scan") / "scan_29" / "a.jpg"
    assert paths.dataset("some_unlisted_folder") == paths.root("datasets") / "some_unlisted_folder"
    assert str(paths.resolve("plain/relative.txt")) == str(Path("plain/relative.txt"))
    assert paths.dataset("mac400-scan").is_dir() and paths.dataset("synthetic-260930-1856").is_dir() and paths.dataset("mac400-scan") == paths.dataset("ecg-mac400-scan")


def test_calibration_step_in_front_of_one_lead_is_cut():
    from src.digitize.leads import trim_early_starts

    grid = Grid("dot", np.zeros(2), np.array([10.0, 0.0]), np.array([0.0, 10.0]))  # 10 px per mm
    masks = []
    for i in range(3):
        m = np.zeros((90, 600), np.uint8)
        m[15 + 30 * i, 100:590] = 1  # the trace starts at 10 mm
        masks.append(m)
    masks[1][15 + 30 - 8:15 + 30 + 1, 20:100] = 1  # a step area from 2 mm to 10 mm in the second row
    out = trim_early_starts(masks, grid, 5.0, 0.5)
    assert np.nonzero(out[1])[1].min() >= 95 and out[0].sum() == masks[0].sum() and out[2].sum() == masks[2].sum()
    assert trim_early_starts(masks[:2], grid, 5.0, 0.5)[1].sum() == masks[1].sum()


def test_oversized_page_is_scaled_to_the_working_side():
    from src.digitize.pipeline import fit_side

    big, s = fit_side(np.zeros((7000, 4600, 3), np.uint8), 3600)
    assert max(big.shape[:2]) == 3600 and abs(s - 3600 / 7000) < 1e-9
    small, s = fit_side(np.zeros((3300, 2550, 3), np.uint8), 3600)
    assert small.shape[:2] == (3300, 2550) and s == 1.0


def test_checks_script_scores_pulse_and_einthoven(tmp_path):
    from src.digitize.checks import score_panel

    t = np.arange(0, 6.0, 0.01)
    i, iii = np.sin(2 * np.pi * 1.2 * t), 0.5 * np.sin(2 * np.pi * 2.3 * t)
    with (tmp_path / "p.csv").open("w", newline="", encoding="utf-8") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow(["t_s", "I_mV", "II_mV", "III_mV", "I_flag", "II_flag", "III_flag"])
        for k in range(len(t)):
            w.writerow([t[k], i[k], i[k] + iii[k], iii[k], "ok", "ok", "ok"])
    cfg = {"pulse_tol_mv": 0.2, "einthoven_min_corr": 0.8, "max_lag_bins": 40}
    good = score_panel({"pulses_mm": [10.0], "gain_mm_per_mV": 10.0}, tmp_path / "p.csv", cfg)
    assert good["pulse_ok"] and good["einthoven_ok"]
    assert not score_panel({"pulses_mm": [5.0], "gain_mm_per_mV": 10.0}, tmp_path / "p.csv", cfg)["pulse_ok"]
    assert score_panel({}, tmp_path / "missing.csv", cfg)["einthoven_ok"] is None


def test_printed_text_and_solid_blobs_leave_the_trace_alone():
    from src.digitize.mask import drop_printed_text, drop_solid_blobs

    m = np.zeros((100, 400), np.uint8)
    m[50, 10:390] = 1  # the trace, long and thin
    m[20:30, 100:104] = 1  # a glyph stroke inside a text box
    m[22:26, 120:130] = 1  # another glyph inside the same box
    m[60:90, 300:330] = 1  # a solid icon, away from any text box
    out = drop_printed_text(m.copy(), [[95, 15, 135, 35]], 60.0, 2.0)
    assert out[50].sum() == 380 and out[20:30, 100:130].sum() == 0 and out[60:90, 300:330].sum() == 900
    out = drop_solid_blobs(out, 20.0, 0.6)
    assert out[60:90, 300:330].sum() == 0 and out[50].sum() == 380


def test_pulse_finder_skips_a_filled_square():
    from src.digitize.pulses import find_pulses

    grid = Grid("dot", np.zeros(2), np.array([10.0, 0.0]), np.array([0.0, 10.0]))  # 10 px per mm
    gray = np.full((200, 400), 255, np.uint8)
    gray[50:100, 20:80] = 0   # a filled 6 x 5 mm square, the header icon
    gray[50:101, 200:204] = 0  # an outline pulse 6 mm wide and 5 mm tall: two vertical strokes and a top and bottom line
    gray[50:54, 200:260] = 0
    gray[50:101, 256:260] = 0
    cfg = {"gray_max": 130, "height_tolerance": [0.85, 1.15], "width_mm": [4.0, 9.0], "min_area": 80, "solid_fill_max": 0.5}
    found = find_pulses(gray, grid, 5.0, (0, 400, 0, 200), cfg)
    assert len(found) == 1 and found[0]["bbox"][0] == 200
