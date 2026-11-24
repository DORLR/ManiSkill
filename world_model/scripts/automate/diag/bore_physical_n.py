"""Option 2 of r4-decisions-06 (the bore rule): the physical symmetry n of an assembly from the
plug AND the socket (fit.physical_symmetry: the seated plug's sections rotated about the plug axis against the socket
sections; "round" when no angle of a 0.5 deg scan is blocked, else the largest n in 2..12 whose rotation 360 / n is
not blocked, else 1; None when the seated sections already overlap, fit.O0_MAX), next to the current rule's n (plug
alone, assets.SPEC_FILE).

For every assembly of --ids (default 00320, 00863) whose n_phys differs from the current n, and only with --write,
candidate files are written next to the current ones under new names, never over them:
    <assets>/<id>/geometry_v6_physn.json   the current spec with symmetry_n = n_phys and the category re-derived by
                                           fit.py's rule (round / polygonal / keyed; multi-pin, sleeve, flat kept)
    <assets>/<id>/ranges_v4_physn.json     grasp.ranges(id, n_phys): the grasp box with spin limit min(P/4, 15 deg)
                                           (hand / finger - socket collision check at H*, hpp-fcl; CPU)
assets.SPEC_FILE and grasp.RANGES_FILE are read, never written. The user's decision of 2026-11-23 applies option 2 only
to the named list assets.SPIN_ROUND_2026_11_23 (symmetry "round"), through geometry_v7.json / ranges_v5.json written
by spin_round_v7.py from this script's --controls all output; the _physn candidates are not used by any code.
--controls adds assemblies whose n is only reported (a check that the rule finds keyed parts keyed); --controls all
reports every assembly with a spec.

    python -u bore_physical_n.py --out bore_physical_n.json [--ids 00320,00863] [--controls 00015,00081] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import fit  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402

GEOMETRY_CANDIDATE = "geometry_v6_physn.json"
RANGES_CANDIDATE = "ranges_v4_physn.json"


def category(spec: dict, n) -> str:
    """fit.py's category rule with symmetry n (the first three rules do not depend on n and are kept)."""
    c = spec.get("category")
    if c in ("multi-pin", "sleeve", "flat"):
        return c
    if n == "round":
        return "round"
    return "polygonal" if isinstance(n, int) and n >= 3 else "keyed"


def run(aid: str, write: bool) -> dict:
    import trimesh
    t0 = time.time()
    spec = A.load_spec(aid)
    plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
    sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    phys = fit.physical_symmetry(plug, sock, spec.axis_xy_m)
    n_now, n_phys = spec.symmetry_n, phys["n_phys"]
    row = {"assembly_id": aid, "n_current": n_now, "category_current": spec.category, "n_phys": n_phys,
           "category_phys": None if n_phys is None else category(spec.as_dict(), n_phys),
           "spin_limit_current_deg": G.spin_limit_deg(n_now),
           "spin_limit_phys_deg": None if n_phys is None else G.spin_limit_deg(n_phys), "physical_symmetry": phys,
           "box_current": G.load_ranges(aid).get("box")}
    if write and n_phys is not None and n_phys != n_now:
        d = A.ASSET_DIR / aid
        gf, rf = d / GEOMETRY_CANDIDATE, d / RANGES_CANDIDATE
        new = spec.as_dict()
        new.pop("max_decisions", None)
        new.update(symmetry_n=n_phys, category=row["category_phys"])   # AssemblySpec fields only (load_spec)
        new["fit"] = {**new["fit"], "n_plug_rule": n_now, "physical_symmetry": phys,
                      "symmetry_rule": "physical (r4-decisions-06 option 2, candidate)"}
        gf.write_text(json.dumps(new, default=float))
        rg = G.ranges(aid, n_phys)
        rg["symmetry_rule"] = "physical (r4-decisions-06 option 2, candidate)"
        rf.write_text(json.dumps(rg, default=float))
        row.update(written=[str(gf), str(rf)], box_phys=rg.get("box"), box_phys_verified=rg.get("verified"))
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="00320,00863")
    ap.add_argument("--controls", default="")
    ap.add_argument("--write", action="store_true", help="write the candidate files (new names only)")
    args = ap.parse_args()
    rows = []
    for aid in [x for x in args.ids.split(",") if x]:
        rows.append(run(aid, args.write))
        print(json.dumps({k: rows[-1][k] for k in ("assembly_id", "n_current", "n_phys", "spin_limit_phys_deg",
                                                   "seconds")}), flush=True)
    ctl = A.assembly_ids() if args.controls == "all" else [x for x in args.controls.split(",") if x]
    done = {r["assembly_id"] for r in rows}
    for aid in [x for x in ctl if x not in done and (A.ASSET_DIR / x / A.SPEC_FILE).exists()]:
        try:
            r = run(aid, False)
        except Exception as exc:                                  # reported, the other assemblies go on
            r = {"assembly_id": aid, "error": f"{type(exc).__name__}: {exc}", "n_current": None, "n_phys": None,
                 "seconds": None}
        r["control"] = True
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("assembly_id", "n_current", "n_phys", "seconds")}), flush=True)
    Path(args.out).write_text(json.dumps({"spec_file": A.SPEC_FILE, "ranges_file": G.RANGES_FILE,
                                          "tol_overlap": fit.TOL_OVERLAP, "step_deg": fit.PHYS_STEP_DEG,
                                          "rows": rows}, indent=1, default=float))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
