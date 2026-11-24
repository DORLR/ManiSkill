"""Mesh analysis of every AutoMate assembly with an optimised grasp (milestone 1b, no Isaac app):
tight-fit depth L, entrance, symmetry, category (wm.sim_automate.fit), radial clearance (assets), N_max, and the
per-assembly grasp ranges from the hand/finger-socket collision check at H* (wm.sim_automate.grasp). Writes
assets/<id>/{assets.SPEC_FILE, grasp.RANGES_FILE} when missing (the ranges file of the current spin-axis switch,
config.GRASP_SPIN_AXIS); never overwrites them or older versions.

    python -u analyse_assemblies.py --out /mnt/storage3/qingxin/automate/m1b/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402


def obj_overlap(aid: str, n: int = 20000) -> dict:
    """Deepest overlap of plug.obj and socket.obj at the assembled pose [mm] (checks.overlap: inside by ray parity for
    a watertight mesh, by the generalised winding number for one that is not (22 of the 200 meshes), unsigned
    distance as depth, both meshes sampled, in chunks), with the watertight flags and the inside test used."""
    import numpy as np
    import trimesh
    plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
    sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    return K.overlap(plug, sock, np.eye(4), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1b/analysis.json")
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    ids = args.ids.split(",") if args.ids else A.assembly_ids()
    rows = []
    for aid in ids:
        t0 = time.time()
        try:
            s = A.load_spec(aid, compute=True)
            r = G.load_ranges(aid, compute=True)
            d = s.as_dict()
            d["fit"].pop("sweep_coarse", None)
            ov = obj_overlap(aid)
            d["obj_overlap_mm"] = ov["max_penetration_mm"]
            d["obj_overlap"] = ov
            rows.append({**d, "ranges": r, "switches": C.switches(), "spec_file": A.SPEC_FILE,
                         "ranges_file": G.RANGES_FILE, "seconds": round(time.time() - t0, 1)})
            b = r.get("box") or {}
            print(f"{aid} L {s.pin_length_m * 1e3:.1f} d_dis {s.disassembly_dist_m * 1e3:.1f} c {s.clearance_m * 1e3:.3f} "
                  f"n {s.symmetry_n} {s.category} box {json.dumps(b)} {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:
            rows.append({"assembly_id": aid, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})
            print(aid, "ERROR", exc, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    print("wrote", args.out, len(rows))


if __name__ == "__main__":
    main()
