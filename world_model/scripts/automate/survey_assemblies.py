"""Assembly survey (milestone 1, item 4): insertion depth and radial clearance of every AutoMate assembly with an
optimised grasp, from the meshes (definitions in wm.sim_automate.assets). No Isaac app needed.

    python survey_assemblies.py --out /mnt/storage3/qingxin/automate/m1/survey.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.sim_automate import assets as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1/survey.json")
    ap.add_argument("--target_mm", type=float, default=0.5, help="our ManiSkill clearance per side")
    args = ap.parse_args()
    rows, t0 = [], time.time()
    for aid in A.assembly_ids():
        try:
            s = A.load_spec(aid)
            rows.append(s.as_dict())
        except Exception as exc:  # report, do not stop the survey
            rows.append({"assembly_id": aid, "error": f"{type(exc).__name__}: {exc}"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
    ok = [r for r in rows if "error" not in r]
    print(f"{len(ok)} of {len(rows)} assemblies measured in {time.time() - t0:.0f} s -> {args.out}")
    hdr = ("id", "c_min mm", "off mm", "d_min mm", "L mm", "L_tight mm", "L_top mm", "d_dis mm", "plug xy mm",
           "fingertip above face mm", "plug top above fingertip mm")
    print(" | ".join(hdr))
    for r in sorted(ok, key=lambda r: abs(r["clearance_m"] * 1000 - args.target_mm) if math.isfinite(r["clearance_m"]) else 1e9):
        above = (r["fingertip_z_m"] - r["entrance_z_m"]) * 1000
        room = (r["plug_top_z_m"] - r["fingertip_z_m"]) * 1000
        print(f"{r['assembly_id']} | {r['clearance_m'] * 1000:.3f} | {r['seated_offcentre_m'] * 1000:.3f} | "
              f"{r['min_gap_m'] * 1000:.3f} | {r['pin_length_m'] * 1000:.1f} | {r['depth_tight_m'] * 1000:.1f} | "
              f"{r['depth_top_m'] * 1000:.1f} | "
              f"{r['disassembly_dist_m'] * 1000:.1f} | {r['plug_xy_m'][0] * 1000:.1f} x {r['plug_xy_m'][1] * 1000:.1f} | "
              f"{above:.1f} | {room:.1f}")
    for r in rows:
        if "error" in r:
            print("ERROR", r["assembly_id"], r["error"])


if __name__ == "__main__":
    main()
