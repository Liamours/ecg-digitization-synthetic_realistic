"""One picture per digitized page: the original page with panel boxes, text boxes, and traces on the left, every
digitized lead (mV over time) on the right. Reads only what run.py wrote, so it also works on old runs.

Usage:
    uv run python -m src.digitize.report --out-dir <run dir or one page folder>
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import paths  # noqa: E402


def read_leads(csv_path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, str]]]:
    with csv_path.open(encoding="utf-8") as fh:
        head, *body = csv.reader(fh)
    n = (len(head) - 1) // 2
    t = np.array([float(r[0]) for r in body])
    leads = {head[1 + i][:-3]: (np.array([float(r[1 + i]) if r[1 + i] else np.nan for r in body]), body[0][1 + n + i]) for i in range(n)}
    return t, leads


def make(page_dir: Path, max_side: int = 1500) -> Path:
    page = json.loads((page_dir / "page.json").read_text(encoding="utf-8"))
    names = ["overlay_original.jpg", "overlay_upright.jpg", "overlay_original.png", "overlay_upright.png"]
    img = cv2.imread(str(next(page_dir / n for n in names if (page_dir / n).exists())))
    s = min(1.0, max_side / max(img.shape[:2]))
    img = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    plots, unreadable = [], []
    for pj in sorted(page_dir.glob("panel*.json"), key=lambda p: int(p.stem[5:])):
        meta = json.loads(pj.read_text(encoding="utf-8"))
        if not pj.with_suffix(".csv").exists():
            unreadable.append(f"{pj.stem} ({meta['error'] or 'no leads'})")
            continue
        t, leads = read_leads(pj.with_suffix(".csv"))
        plots += [(pj.stem, meta, t, name, mv, flag) for name, (mv, flag) in leads.items()]
    rows = max(math.ceil(len(plots) / 2), 4)
    fig = plt.figure(figsize=(20, max(9, 1.3 * rows + 1.5)))
    gs = fig.add_gridspec(rows, 3, width_ratios=[2.2, 1.5, 1.5], wspace=0.25, hspace=1.0)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"panels {len(page['panels'])}" + (f", unreadable: {', '.join(unreadable)}" if unreadable else ""), fontsize=9)
    for k, (pid, meta, t, name, mv, flag) in enumerate(plots):
        a = fig.add_subplot(gs[k // 2, 1 + k % 2])
        a.plot(t, mv, lw=0.6, color="k" if flag == "ok" else "tab:red")
        a.set_title(f"{pid} {name}  {meta['gain_mm_per_mV']:g} mm/mV ({meta['gain_source']})" + ("" if flag == "ok" else f"  {flag}"), fontsize=8, color="k" if flag == "ok" else "tab:red")
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3)
        a.set_ylabel("mV", fontsize=7)
        a.set_xlabel("s", fontsize=7)
    fig.suptitle(f"{Path(page['source']).name}   layout {page['layout']}   rotation {page['rotation_ccw_deg']} deg", fontsize=11)
    out = page_dir / "report.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    args = ap.parse_args()
    pages = [args.out_dir] if (args.out_dir / "page.json").exists() else sorted(p.parent for p in args.out_dir.glob("*/page.json"))
    for p in tqdm(pages, desc="reports"):
        make(p)


if __name__ == "__main__":
    main()
