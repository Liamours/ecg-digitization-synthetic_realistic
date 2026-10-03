"""Pass rates from a digitize output folder: pages, panels, leads, gain source, errors, by dataset."""
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for page in sorted(root.glob("*/page.json")):
    d = json.loads(page.read_text(encoding="utf-8"))
    for pid in d["panels"]:
        r = json.loads((page.parent / f"{pid}.json").read_text(encoding="utf-8"))
        rows.append((("retake" if "retake" in d["source"] else "original"), page.parent.name, r))
for group in ("retake", "original", "all"):
    sel = [x for x in rows if group == "all" or x[0] == group]
    pages = {x[1] for x in sel}
    leads = [(k, v) for _, _, r in sel for k, v in r["leads"].items()]
    ok = sum(v["flag"] == "ok" for _, v in leads)
    err = Counter(r["error"].split(":")[0] + ":" + r["error"].split(":", 1)[-1][:40] for _, _, r in sel if r["error"])
    gain = Counter(r["gain_source"] for _, _, r in sel if not r["error"])
    conf = sorted(r["confidence"] for _, _, r in sel if not r["error"])
    pul = [abs(r["selftest"]["pulse"]["pulse_mV"] - 1) for _, _, r in sel if r["selftest"].get("pulse")]
    ein = [r["selftest"]["einthoven"]["II=I+III"]["corr"] for _, _, r in sel if "II=I+III" in r["selftest"].get("einthoven", {}) and r["label_source"] != "position"]
    print(f"{group}: pages {len(pages)}, panels {len(sel)} (errors {sum(bool(r['error']) for _, _, r in sel)}), leads {len(leads)}, ok {ok} ({100*ok/max(len(leads),1):.0f}%), gain source {dict(gain)}, median confidence {conf[len(conf)//2] if conf else '-'}")
    print("   pulse error (|mV-1|) median %.3f, panels with pulse %d; errors: %s" % (sorted(pul)[len(pul) // 2] if pul else float('nan'), len(pul), dict(err)))
