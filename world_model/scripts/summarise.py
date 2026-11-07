"""Summarise a collection directory (reads only grasp.json files).

    python world_model/scripts/summarise.py $Q/data/wm_v1
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in sorted(root.rglob("grasp.json"))]
group = lambda p: "A-D" if p in "ABCD" else ("G00-G49" if int(p[1:]) < 50 else "G50-G149")
CONDS = ["train", "pitch_interp", "pitch_extrap", "pos_interp", "pos_extrap", "combined"]

print(f"grasps {len(rows)}  errors {sum('error' in r for r in rows)}  "
      f"data {sum(r.get('bytes', 0) for r in rows) / 1e9:.2f} GB  "
      f"median {np.median([r['seconds'] for r in rows]):.0f} s/grasp")
for e in [r for r in rows if "error" in r][:5]:
    print("  error:", e["grasp"], e["error"][:150])

print("\nreference insertion (expert, from the aligned safe standoff)")
for g in ["A-D", "G00-G49", "G50-G149"]:
    line = []
    for c in CONDS:
        rs = [r for r in rows if group(r["port"]) == g and r["condition"] == c]
        if rs:
            line.append(f"{c} {sum(bool(r.get('reference_ok')) for r in rs)}/{len(rs)}")
    print(f"  {g:9s} " + "  ".join(line))
fails = Counter(str(r.get("reference_failure"))[:70] for r in rows if not r.get("reference_ok"))
print("  failures:", dict(fails.most_common(5)))

print("\ntrajectories by kind (outcome counts)")
kinds = defaultdict(Counter)
for r in rows:
    for t in r["trajectories"]:
        kinds[t["kind"]][t.get("outcome")] += 1
for k, c in kinds.items():
    print(f"  {k:10s} {sum(c.values()):5d}  " + "  ".join(f"{o}:{n}" for o, n in c.most_common()))

print("\nfar-start insertion success (insert + perturbed) by group x condition")
for g in ["A-D", "G00-G49", "G50-G149"]:
    line = []
    for c in CONDS:
        ts = [t for r in rows if group(r["port"]) == g and r["condition"] == c
              for t in r["trajectories"] if t["kind"] in ("insert", "perturbed")]
        if ts:
            line.append(f"{c} {sum(t.get('outcome') == 'inserted' for t in ts)}/{len(ts)}")
    print(f"  {g:9s} " + "  ".join(line))

fail = Counter(str(t.get("failure"))[:70] for r in rows for t in r["trajectories"]
               if t.get("outcome") in ("expert_failed", "placement_failed"))
print("\nexpert failures:", dict(fail.most_common(8)))
contact = [t for r in rows for t in r["trajectories"] if t["kind"] == "contact" and "offset" in t]
if contact:
    lat = np.array([np.hypot(*t["offset"]["yz_mm"]) for t in contact])
    ins = np.array([t.get("outcome") == "inserted" for t in contact])
    for lo, hi in ((0, 0.01), (0.01, 0.5), (0.5, 1), (1, 2), (2, 3.1)):
        m = (lat >= lo) & (lat < hi)
        if m.any():
            print(f"  contact trials, lateral offset {lo}-{hi} mm: inserted {ins[m].sum()}/{m.sum()}")
