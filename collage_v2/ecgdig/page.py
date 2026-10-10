"""A whole page: find the strips, digitize each, keep the best attempt of every lead, write the record and QA pictures."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import (Marker, crop_panel, find_markers, geometry_orientation, header_text_score, load_gray, normalise,
                       page_px_per_mm, refine_scale, rot_box)
from .panel import PanelResult, digitize_panel
from .quality import einthoven_aug, einthoven_limb, lead_metrics
from .text import LEAD_SETS as LEAD_GROUPS

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STATUS_RANK = {"ok": 3, "low_quality": 2, "flat": 1, "missing": 0}


@dataclass
class PageResult:
    page_type: str                     # ecg_mac400, form_or_text, blank, ecg_other_layout
    panels: list[dict] = field(default_factory=list)
    signal: np.ndarray | None = None   # 12 x N mV
    status: list[str] = field(default_factory=list)
    lead_info: list[dict] = field(default_factory=list)
    overlay: np.ndarray | None = None
    seconds: float = 0.0
    glyphs: list = field(default_factory=list)
    _results: list = field(default_factory=list)


def page_type_of(gray: np.ndarray, n_markers: int) -> str:
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    grey = float(((small > 120) & (small < 235)).mean())
    if n_markers > 0:
        return "ecg_mac400"
    if grey > 0.10:   # other ECG sheets are 0.2 to 0.35; filled-in diagnosis forms stay below 0.03
        return "ecg_other_layout"
    ink = float((small < 160).mean())
    return "blank" if ink < 0.01 else "form_or_text"


def orient(ocr, page: np.ndarray, m: Marker, s: float) -> tuple[int, float, str]:
    H, W = page.shape
    ink = (page < 180).astype(np.uint8)
    k0 = geometry_orientation(ink, m, s)
    order = [k0] + [k for k in range(4) if k != k0]
    best = (-1.0, k0, "")
    for k in order:
        sc, txt = header_text_score(ocr, np.rot90(page, k), rot_box(m, k, W, H), s)
        if sc > best[0]:
            best = (sc, k, txt)
        if sc >= 1.0:
            break
    return best[1], best[0], best[2]


def panel_overlay(pc, res: PanelResult, max_w: int = 900) -> np.ndarray:
    vis = cv2.cvtColor(pc.image, cv2.COLOR_GRAY2BGR)
    vis = (vis.astype(np.float32) * 0.45 + 140).astype(np.uint8)
    colors = [(0, 0, 230), (0, 150, 0), (220, 0, 0)]
    for li, L in enumerate(res.leads):
        y = L.y_px
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        yi = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
        pts = np.column_stack([L.x_px, yi]).astype(np.int32)
        cv2.polylines(vis, [pts], False, colors[li % 3], 1, cv2.LINE_AA)
        st = L.metrics.get("status", "")
        cv2.putText(vis, f"{L.name} {st}", (int(res.x_range[0]), int(res.rows[li] - 3.5 * res.s)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[li % 3], 2)
    if res.pulse:
        p = res.pulse
        cv2.rectangle(vis, (int(p["x_left"]), int(p["top"])), (int(p["x_right"]), int(p["bottom"])), (200, 0, 200), 2)
    cv2.putText(vis, f"set={res.lead_set} gain={res.gain:g} mm/mV ({res.gain_source}) {res.error}", (5, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    f = max_w / vis.shape[1]
    return cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else vis


def digitize_page(ocr, path: str, fs: int = 500, duration_s: float = 4.0, overlay: bool = True) -> PageResult:
    t0 = time.time()
    gray = load_gray(path)
    s0 = page_px_per_mm(gray)
    page = normalise(gray, s0)
    markers = find_markers(page, s0)
    ptype = page_type_of(page, len(markers))
    out = PageResult(ptype)
    if ptype != "ecg_mac400":
        out.seconds = time.time() - t0
        return out
    H, W = page.shape
    results, tiles = [], []
    for i, m in enumerate(markers):
        pc = None
        try:
            k, score, htxt = orient(ocr, page, m, s0)
            pc = crop_panel(page, m, k, s0)
            pc.header_text = htxt
            s = refine_scale(pc.image, pc.marker, s0)
            others = []
            for j, o in enumerate(markers):
                if j != i:
                    ob = rot_box(o, k, W, H)
                    others.append(Marker(ob.x - pc.origin[0], ob.y - pc.origin[1], ob.w, ob.h, 0))
            res = digitize_panel(ocr, pc, s, others, fs)
        except Exception as e:  # one bad strip must not stop the page
            import traceback
            res = PanelResult(None, [], [], 10.0, "", 25.0, s0, [], (0, 0), None, {}, [], error=f"{type(e).__name__}: {e}")
            res.info["traceback"] = traceback.format_exc()
            if "pc" not in locals() or pc is None:
                continue
        for L in res.leads:
            L.metrics = lead_metrics(L, res, s)
        if res.lead_set == "limb" and len(res.leads) == 3:
            res.info["einthoven"] = einthoven_limb(res.leads)
        if res.lead_set == "aug" and len(res.leads) == 3:
            res.info["einthoven"] = einthoven_aug(res.leads)
        info = {"panel": i, "marker_page_xywh": [m.x, m.y, m.w, m.h], "quarter_turns": k, "header_score": round(score, 2), "header_text": htxt,
                "deskew_deg": round(pc.angle, 2), "px_per_mm": round(s, 3), "lead_set": res.lead_set, "leads": res.names, "name_source": "glyphs" if res.lead_set else None,
                "gain_mm_per_mV": res.gain, "gain_source": res.gain_source, "speed_mm_per_s": res.speed,
                "pulse_height_mm": round(res.pulse["height_mm"], 2) if res.pulse else None,
                "label_decoding": res.texts.get("labels", {}),
                "lead_off_marks": res.leadoff, "error": res.error, "einthoven": res.info.get("einthoven"),
                "trace_mm": round((res.x_range[1] - res.x_range[0]) / s, 1) if res.x_range[1] > res.x_range[0] else 0,
                "lead_metrics": {L.name: L.metrics for L in res.leads}}
        out.panels.append(info)
        results.append((res, pc))
        out.glyphs.append(res.info.get("glyphs", []))
        if overlay:
            tiles.append(panel_overlay(pc, res))
    # black corner blocks of other devices (EDAN sheets) look like a MAC 400 marker; with no strip showing MAC 400 header
    # text and no strip named, the page is another layout
    if out.panels and max(p["header_score"] for p in out.panels) < 0.5 and all(r.lead_set is None for r, _ in results):
        other = PageResult("ecg_other_layout")
        other.seconds = time.time() - t0
        return other
    # strips whose names could not be read: (1) an augmented strip satisfies aVR + aVL + aVF = 0, which names it;
    # (2) when exactly one group is still missing on the page and
    # exactly one unread strip is left, that strip gets it. Both are marked in name_source.
    def rename(pi, r, k, src):
        r.lead_set = k
        for L, nm in zip(r.leads, LEAD_GROUPS[k]):
            L.name = nm
        r.names = list(LEAD_GROUPS[k])
        out.panels[pi].update({"lead_set": k, "leads": r.names, "name_source": src, "lead_metrics": {L.name: L.metrics for L in r.leads}})

    for pi, (r, pc) in enumerate(results):
        if r.lead_set is None and len(r.leads) == 3 and not r.error:
            el = einthoven_limb(r.leads).get("corr", 0.0) or 0.0
            ea = einthoven_aug(r.leads).get("corr", 0.0) or 0.0
            out.panels[pi]["identity_corr"] = {"limb": el, "aug": ea}
            # only the augmented identity is specific enough to name a strip (tested on synthetic pages: true augmented
            # strips reach corr >= 0.9, precordial strips stayed below 0.65, limb strips negative). II = I + III is not:
            # neighbouring precordial leads satisfy V2 ~ V1 + V3 nearly as well.
            ag = einthoven_aug(r.leads)
            if ea >= 0.9 and ag.get("rel_residual", 9) <= 0.6:
                rename(pi, r, "aug", "goldberger_identity")
                r.info["einthoven"] = ag
                out.panels[pi]["einthoven"] = ag
    found = {r.lead_set for r, _ in results if r.lead_set in LEAD_GROUPS}
    missing = [k for k in LEAD_GROUPS if k not in found]
    unread = [pi for pi, (r, _) in enumerate(results) if r.lead_set is None and len(r.leads) == 3 and not r.error]
    if len(missing) == 1 and len(unread) == 1:
        rename(unread[0], results[unread[0]][0], missing[0], "page_completion")
    elif set(missing) == {"v13", "v46"} and len(unread) == 2:
        # two unread precordial strips: the one whose glyphs lean more to V1-V3 takes V1-V3
        d = []
        for pi in unread:
            sc = (results[pi][0].texts.get("labels") or {}).get("scores", {})
            d.append(sc.get("v13", 0.0) - sc.get("v46", 0.0))
        if abs(d[0] - d[1]) >= 0.2:
            a, b = (unread[0], unread[1]) if d[0] > d[1] else (unread[1], unread[0])
            rename(a, results[a][0], "v13", "page_completion")
            rename(b, results[b][0], "v46", "page_completion")
    # a strip without pulse or readable gain takes the gain of another strip of the same lead group and print mode
    # on the page
    for pi, (r, pc) in enumerate(results):
        if r.gain_source == "assumed" and r.lead_set:
            donors = [r2 for r2, _ in results if r2 is not r and r2.lead_set == r.lead_set and r2.gain_source not in ("assumed", "")
                      and r2.info.get("auto_continuation") == r.info.get("auto_continuation")]
            if donors and donors[0].gain != r.gain:
                f = r.gain / donors[0].gain
                for L in r.leads:
                    L.mv = L.mv * f
                    L.metrics = lead_metrics(L, r, r.s)
                r.gain = donors[0].gain
            if donors:
                r.gain_source = "same group on page"
                out.panels[pi].update({"gain_mm_per_mV": r.gain, "gain_source": r.gain_source, "lead_metrics": {L.name: L.metrics for L in r.leads}})
    # one record: per lead, the best attempt over the panels
    n = int(round(duration_s * fs))
    tgrid = np.arange(n) / fs
    sig = np.full((len(LEADS), n), np.nan, np.float32)
    status, linfo = [], []
    for li, name in enumerate(LEADS):
        tries = []
        for pi, (res, pc) in enumerate(results):
            for L in res.leads:
                if L.name == name:
                    tries.append((STATUS_RANK.get(L.metrics.get("status"), 0), L.metrics.get("score", 0.0), pi, L, res))
        if not tries:
            status.append("missing")
            linfo.append({"lead": name, "status": "missing", "attempts": 0})
            continue
        tries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, pi, L, res = tries[0]
        sig[li] = np.interp(tgrid, L.t, L.mv, left=np.nan, right=np.nan).astype(np.float32)
        status.append(L.metrics["status"])
        linfo.append({"lead": name, "status": L.metrics["status"], "attempts": len(tries), "panel": pi, "gain_mm_per_mV": res.gain,
                      "gain_source": res.gain_source, "duration_s": round(float(L.t[-1]), 2), **{k: v for k, v in L.metrics.items() if k != "status"}})
    out.signal, out.status, out.lead_info = sig, status, linfo
    out._results = results
    if overlay and tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        rows = []
        for i in range(0, len(tiles), 2):
            pair = tiles[i:i + 2]
            if len(pair) == 1:
                pair.append(np.full_like(pair[0], 255))
            h = max(p.shape[0] for p in pair)
            pair = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in pair]
            rows.append(np.hstack(pair))
        out.overlay = np.vstack(rows)
    out.seconds = time.time() - t0
    return out


def save_page(res: PageResult, out_dir: Path, source: str, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source": source, "page_type": res.page_type, "seconds": round(res.seconds, 1), **(extra or {})}
    if res.signal is not None:
        ok = np.array([s == "ok" for s in res.status])
        present = np.array([s != "missing" for s in res.status])
        np.savez_compressed(out_dir / "record.npz", signal=res.signal, ok=ok, present=present, status=np.array(res.status), leads=np.array(LEADS), fs=500)
        n = res.signal.shape[1]
        with (out_dir / "record.csv").open("w", encoding="utf-8") as fh:
            fh.write("time_s," + ",".join(f"{l}_mV" for l in LEADS) + "\n")
            for j in range(n):
                vals = ["" if np.isnan(v) else f"{v:.4f}" for v in res.signal[:, j]]
                fh.write(f"{j / 500:.4f}," + ",".join(vals) + "\n")
        summary.update({"leads_ok": int(ok.sum()), "leads_present": int(present.sum()), "panels": len(res.panels)})
    (out_dir / "record.json").write_text(json.dumps({**summary, "leads": res.lead_info, "panels": res.panels}, indent=1, default=_json), encoding="utf-8")
    if res.overlay is not None:
        cv2.imencode(".jpg", res.overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(out_dir / "overlay.jpg"))
    return summary


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
