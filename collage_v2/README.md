# ecg-digitization-v2

Digitizer for the pasted MAC 400 strip pages of **Data-EKG-update** (RSUP Dr. Sardjito, poliklinik anak). One page image
in, the 12 leads in millivolts at 500 Hz out, with a quality label per lead and a QA picture per page. CPU only, no
training, no GPU needed.

Output follows the record contract of `ecg-classification-pjb_normal` (`signal` 12 x 2000 in mV, `ok`, `present`), so
the classifier repo can read it directly (`tools/to_classification_index.py` writes its `index.csv`).

## Why a new pipeline

The pages of Data-EKG-update are collages: 4 to 8 cut MAC 400 strips glued on a sheet, some turned a quarter, tilted,
overlapping their neighbour, repeated attempts, report strips mixed in. The existing pipeline in
`ecg-digitization-synthetic_realistic` finds panels from the footer text, is tuned on 39 scans of `ecg-mac400-scan`, and
needs the Open-ECG-Digitizer U-Net weights. This one is built around what every strip has, whatever the collage:

| Step | Method | Why it holds on this data |
|---|---|---|
| Page type | grid-ink fraction and header markers | separates ECG pages, diagnosis forms, blank pages |
| Strip finding | the solid black square of the MAC 400 header | one per strip, survives cuts, overlap and rotation |
| Orientation | OCR of the header text next to the square, 4 quarter turns | "MAC 400 / V1.02 / GE" reads only when upright |
| Deskew, scale | projection sharpness of the dot grid; spectrum of the 1 mm dot lattice | px/mm measured per strip (scans are ~8 px/mm, phone photos differ) |
| Trace mask | normalised image < 180, grid dots removed as small isolated components | thermal print is light grey; dots are as dark as the trace but never connected |
| Text removal | RapidOCR boxes of known vocabulary, glyph-sized pieces in header/footer bands | trace passing through the header survives |
| Gain | height of the calibration pulse (2 sides, or 1 side + top bar), checked against the printed "10mm/mV" | the pulse is 1 mV by definition; OCR of the footer is the fallback |
| Rows | three strongest separated peaks of the horizontal ink projection | baselines are not at fixed heights (V1 can run under the header) |
| Lead separation | dynamic programming per lead over the vertical pixel runs of each column, two passes (second pass follows each lead's wandering baseline), virtual "coast on baseline" state for faded stretches | a lead stays on its own line when a big QRS of the neighbour crosses into its row |
| Value per column | run middle on slopes, outer end at peaks/troughs, nearer end when two leads touch | unbiased timing and kept peak amplitude |
| Faded ink | gaps the path bridged are re-read from the grey image near the bridge | |
| Lead names | template matching of the label glyphs (MAC 400 font, templates built from this dataset), word decoding over the 4 groups, OCR votes, footer lead-off marks (`*V4*`), Goldberger identity, page completion | OCR alone misses one- and two-letter labels such as "I", "II", "V4" |
| Quality | coverage, shared runs, path jumps per second, peak-to-peak, flat (electrode off) | per lead `ok / low_quality / flat / missing` |
| Record | best attempt per lead across repeated strips | |

Important: the three leads of one strip are simultaneous, but the four strips (I-III, aVR-aVF, V1-V3, V4-V6) are four
different acquisitions of about 2.9 s each. In `record.npz` every strip starts at t = 0, so the 12 leads are aligned by
strip start, not by heartbeat. A model that mixes leads of different strips sample by sample should know that.

## Accuracy

Measured with `tools/eval_synth.py` on synthetic collage pages (`ecgdig/synth.py`): real MIT-BIH signals combined into
12 leads that obey Einthoven/Goldberger, time-compressed and scaled like paediatric ECGs, drawn as MAC 400 strips (header
block, label glyphs in the device font, pulse, footer, dot grid, faded thermal stretches, 0 to 1.5 degree tilt, quarter
turns, overlap, blur, JPEG). Scored like the PhysioNet Challenge 2024 (median offset removed, up to 40 ms shift).

| Metric (30 pages, 133 strips, 389 leads) | Value |
|---|---|
| strips found | 132 / 133 |
| lead group named correctly / unread / wrong | 89.4 % / 9.8 % / 0.8 % |
| gain correct | 100 % |
| Pearson r per lead, median | 0.93 |
| SNR per lead, median | 8.5 dB |
| RMSE per lead, median | 0.10 mV |
| leads flagged `ok` | 94 % (median r 0.93; leads not flagged ok: median r 0.79) |

The synthetic pages are harder than most real pages in one way (strong baseline wander of the MIT-BIH records, voltages
clipped at the paper edge) and easier in another (no glue, staples, crossed-out text).

Run on all of Data-EKG-update (1151 images, 6 October 2026, `RESULTS.md` of the run):

| | Value |
|---|---|
| images: ECG pages / diagnosis forms / other layouts | 577 / 570 / 4 |
| strips found on the ECG pages | 3012 (30 report strips, 8 without trace) |
| ECG pages with all 12 leads flagged ok | 232 (40 %) |
| ECG pages with at least 9 leads ok | 456 (79 %) |
| strips named by glyphs / page completion / Goldberger / unnamed | 2501 / 118 / 5 / 350 |
| gain from pulse or printed text / assumed 10 mm/mV | 2507 / 407 |
| Einthoven II = I + III on limb strips, median corr | 0.75 |

There is no ground truth for the real pages. The Einthoven correlation is not an accuracy score on its own: on the
synthetic pages the true limb strips reach a median of about 0.8 while their leads correlate 0.93 with the truth (the
identity adds up the errors of three leads and is very sensitive to timing at the QRS). A real-page median of 0.75 is
therefore in the same range as the synthetic test.

## Setup (Windows, PowerShell)

```
cd "D:\Telkom University\Riset\Riset ECG\Digitalisasi dataset\ecg-digitization-v2"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q
```

## Run

```
python -m ecgdig.run --input "..\Data-EKG-update" --out "..\digitized_v2" --workers 4
python tools\pair_pages.py --run "..\digitized_v2"
python tools\summarize_run.py --run "..\digitized_v2"
```

Resumable: run the same command again after an interruption and finished pages are skipped. `--no-overlay` skips the QA
pictures (faster), `--limit 20` for a quick try.

Accuracy test on synthetic pages (needs a folder of MIT-BIH records, e.g. `ecg1lead-starter\data\mitdb`):

```
python -c "from ecgdig.synth import write_set; write_set('synth', r'..\..\Eksperimen\ecg1lead-starter\data\mitdb', 30, seed=7)"
python tools\eval_synth.py --synth synth --out synth_eval.json
```

## Output of a run

```
digitized_v2/
  index.csv                      one row per image: page type, strips, leads ok, lead groups, gains, Einthoven corr
  pairs.csv                      ECG page -> diagnosis form of the same patient (tools/pair_pages.py)
  RESULTS.md                     summary tables (tools/summarize_run.py)
  pages/<folder>__<file>/
    record.npz                   signal (12 x 2000, mV, NaN = no data), ok, present, status, leads, fs
    record.csv                   time_s and one column per lead
    record.json                  per lead: status, strip, gain and its source, coverage, jumps, p2p
                                 per strip: rotation, deskew, px/mm, lead group and how it was named, gain source,
                                 pulse height, Einthoven/Goldberger check, label decoding scores, errors
    overlay.jpg                  every strip upright with the digitized leads drawn over it
```

Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

## Lead names: how a strip gets its group

`name_source` in `record.json`:

- `glyphs`: the label glyphs read (most strips).
- `goldberger_identity`: unread labels, but aVR + aVL + aVF = 0 holds (corr >= 0.9).
- `page_completion`: the only group still missing on the page, or the two V groups split between two unread strips by
  their glyph scores. Less certain; filter on it if needed.
- `null`: not named; its leads are not put in the record.

The glyph templates (`ecgdig/glyph_templates.npz`) were built once from this dataset with
`tools/collect_glyphs.py` and `tools/build_glyph_templates.py`.

## Known limits

- Pages of other devices (e.g. the long 12-lead rhythm sheets in some folders) are labelled `ecg_other_layout` and not
  digitized.
- Strips with heavy movement artefact (crying child) are digitized but usually flagged `low_quality`; check
  `overlay.jpg`.
- A strip whose label glyphs are covered by the trace and that is not the only missing group stays unnamed.
- Gain falls back to 10 mm/mV (`assumed`) when there is no pulse on the cut strip and the footer is unreadable.
- Phone photos with strong perspective (some files in `Dataset tambahan`) are not unwarped.

## Modules

| File | Job |
|---|---|
| `ecgdig/geometry.py` | load, normalise, markers, orientation, crop, deskew, px/mm |
| `ecgdig/panel.py` | one upright strip: mask, text removal, pulse/gain, rows, tracking, names, mV |
| `ecgdig/trace.py` | DP lead tracking, gap re-reading, calibration pulse |
| `ecgdig/glyphs.py` | label glyphs, templates, word decoding |
| `ecgdig/text.py` | OCR boxes, gain/speed/lead-off text, report strip words |
| `ecgdig/quality.py` | per-lead flags, Einthoven/Goldberger checks |
| `ecgdig/page.py` | whole page, naming fallbacks, best attempt per lead, outputs |
| `ecgdig/run.py` | batch runner (parallel, resumable, log) |
| `ecgdig/synth.py` | synthetic collage pages with ground truth |
| `tools/` | evaluation, pairing, summary, export for the classifier, glyph template building |
