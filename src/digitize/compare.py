"""Compare the method runs made on one front stage (src.digitize.methods): where do they agree on a lead?

With no ground truth on real scans, agreement between independent methods is the available confidence signal. For
every page and lead, each pair of methods is compared on the page's record.csv (the same time base): the correlation
after the best small time shift, and the root mean square difference in mV. Agreement is not accuracy: methods can
agree and all be wrong.

Writes into the methods folder: pairs.csv (one row per page, lead and method pair), leads.csv (one row per page and
lead: the status under each method and the mean pairwise correlation), and agreement_map.png (pages by leads, coloured
by that mean). Prints one table per method and one per pair.

Usage:
    python -m src.digitize.compare --methods @inferences/methods_mac400-scan
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import paths  # noqa: E402

MAX_SHIFT = 50   # samples of the record's time base: methods may start a lead at a slightly different column
MIN_OVERLAP = 250


def read_record(page_dir: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    with (page_dir / "record.csv").open(encoding="utf-8") as fh:
        head, *body = csv.reader(fh)
    names = [h[:-3] for h in head[1:]]
    signal = np.array([[float(v) if v else np.nan for v in r[1:]] for r in body]).T
    status = {x["lead"]: x["status"] for x in json.loads((page_dir / "record.json").read_text(encoding="utf-8"))["leads"]}
    return names, signal, status


def agree(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Best correlation over small shifts, the RMS difference at that shift after removing each median, and the samples compared."""
    best = (np.nan, np.nan, 0)
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1, 5):
        x, y = (a[s:], b[:len(b) - s]) if s >= 0 else (a[:s], b[-s:])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_OVERLAP or x[ok].std() == 0 or y[ok].std() == 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if not np.isfinite(best[0]) or r > best[0]:
            best = (r, float(np.sqrt(np.mean(((x[ok] - np.median(x[ok])) - (y[ok] - np.median(y[ok]))) ** 2))), int(ok.sum()))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=paths.resolve, required=True, help="folder holding one sub-folder per method run")
    args = ap.parse_args()
    runs = sorted(d for d in args.methods.iterdir() if d.is_dir() and (d / "progress.csv").exists())
    pages = sorted({p.parent.name for r in runs for p in r.glob("*/record.csv")})
    pair_rows, lead_rows = [], []
    leads: list[str] = []
    for page in pages:
        data = {r.name: read_record(r / page) for r in runs if (r / page / "record.csv").exists()}
        leads = next(iter(data.values()))[0]
        for i, lead in enumerate(leads):
            corrs = []
            for a, b in combinations(data, 2):
                r, rms, n = agree(data[a][1][i], data[b][1][i])
                pair_rows.append({"page": page, "lead": lead, "method_a": a, "method_b": b, "corr": "" if np.isnan(r) else round(r, 3), "rms_mV": "" if np.isnan(rms) else round(rms, 3), "samples": n})
                if np.isfinite(r):
                    corrs.append(r)
            lead_rows.append({"page": page, "lead": lead, **{m: data[m][2].get(lead, "missing") for m in data}, "pairs": len(corrs), "mean_corr": round(float(np.mean(corrs)), 3) if corrs else ""})
    for name, rows in (("pairs.csv", pair_rows), ("leads.csv", lead_rows)):
        with (args.methods / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    grid = np.full((len(pages), len(leads)), np.nan)
    for r in lead_rows:
        if r["mean_corr"] != "":
            grid[pages.index(r["page"]), leads.index(r["lead"])] = r["mean_corr"]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(pages) + 1.2))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(leads)), leads, fontsize=7)
    ax.set_yticks(range(len(pages)), [p.split("__")[-1][-16:] for p in pages], fontsize=6)
    ax.set_title("Mean correlation between methods, per page and lead (blank: fewer than two methods read it)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.methods / "agreement_map.png", dpi=130)
    print("| Method | Leads ok | Leads low quality | Leads flat | Leads missing |\n|---|---|---|---|---|")
    for r in runs:
        st = [x[r.name] for x in lead_rows if r.name in x]
        print(f"| {r.name} | {st.count('ok')} | {st.count('low_quality')} | {st.count('flat')} | {st.count('missing')} |")
    print("\n| Pair | Leads compared | Median correlation | Share with correlation at least 0.9 |\n|---|---|---|---|")
    for a, b in combinations([r.name for r in runs], 2):
        c = np.array([x["corr"] for x in pair_rows if x["method_a"] == a and x["method_b"] == b and x["corr"] != ""])
        if len(c):
            print(f"| {a} vs {b} | {len(c)} | {np.median(c):.2f} | {100 * (c >= 0.9).mean():.0f}% |")
    m = np.array([x["mean_corr"] for x in lead_rows if x["mean_corr"] != ""])
    print(f"\n{len(m)} leads read by at least two methods; mean pairwise correlation at least 0.9 on {100 * (m >= 0.9).mean():.0f}%, below 0.5 on {100 * (m < 0.5).mean():.0f}%")


if __name__ == "__main__":
    main()
