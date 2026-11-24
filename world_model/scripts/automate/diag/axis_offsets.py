"""Per assembly: where the plug axis, the CAD origin and the tip centre lie (plug frame), and what follows from it.

    r_axis  = |(x_c, y_c)|: plug axis (assets.AssemblySpec.axis_xy_m) from the CAD origin. The two values of the switch
              config.GRASP_SPIN_AXIS ("plug_axis", approved by the user 2026-11-22, and "cad_origin", the spec formula)
              differ for these assemblies: at the box's largest spin s the plug sits 2 r_axis sin(s / 2) apart in the
              hand between the two (shift_at_max_spin_mm).
    r_tip   = |tip_xy - axis_xy|: tip centre (where depth_left / lateral are recorded, checks.py) from the plug axis; a
              spin of 4 deg about the axis moves it sideways by 2 r_tip sin(2 deg) (lateral_at_4deg_spin_mm).
Reads assets.SPEC_FILE and grasp.RANGES_FILE (read-only); writes one JSON.
    python -u axis_offsets.py --out /mnt/storage3/qingxin/automate/m1b/v6/axis_offsets.json [--min-mm 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-mm", type=float, default=0.05, help="list assemblies with r_axis above this [mm]")
    args = ap.parse_args()
    rows, missing = [], []
    for aid in A.assembly_ids():
        try:
            s = A.load_spec(aid)
        except FileNotFoundError:
            missing.append(aid)
            continue
        try:
            box = G.load_ranges(aid).get("box")
        except FileNotFoundError:
            box = None
        ax, tip = np.asarray(s.axis_xy_m, float), np.asarray(s.tip_centre_m, float)[:2]
        r_axis, r_tip = float(np.linalg.norm(ax)), float(np.linalg.norm(tip - ax))
        smax = max(abs(v) for v in box["dpsi_deg"]) if box else G.spin_limit_deg(s.symmetry_n)
        rows.append({"assembly_id": aid, "symmetry_n": s.symmetry_n, "category": s.category,
                     "axis_xy_mm": [round(v * 1e3, 3) for v in ax], "r_axis_mm": round(r_axis * 1e3, 3),
                     "max_spin_deg": smax, "has_box": box is not None,
                     "shift_at_max_spin_mm": round(2 * r_axis * np.sin(np.radians(smax) / 2) * 1e3, 3),
                     "r_tip_mm": round(r_tip * 1e3, 3),
                     "lateral_at_4deg_spin_mm": round(2 * r_tip * np.sin(np.radians(2.0)) * 1e3, 3)})
    rows.sort(key=lambda r: -r["r_axis_mm"])
    affected = [r for r in rows if r["r_axis_mm"] > args.min_mm]
    out = {"spec_file": A.SPEC_FILE, "ranges_file": G.RANGES_FILE, "switches": C.switches(),
           "spin_axis_affected": affected, "n_affected": len(affected), "threshold_mm": args.min_mm,
           "tip_off_axis": sorted([r for r in rows if r["r_tip_mm"] > 1.0], key=lambda r: -r["r_tip_mm"]),
           "missing_spec": missing, "all": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"{len(affected)} assemblies with the plug axis > {args.min_mm} mm from the CAD origin; "
          f"{len(out['tip_off_axis'])} with the tip centre > 1 mm off the axis; {len(missing)} without {A.SPEC_FILE}")
    for r in affected:
        print(r["assembly_id"], r["axis_xy_mm"], f"shift {r['shift_at_max_spin_mm']} mm at {r['max_spin_deg']} deg",
              "box" if r["has_box"] else "no box")


if __name__ == "__main__":
    main()
