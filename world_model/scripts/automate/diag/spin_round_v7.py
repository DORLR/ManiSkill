"""Writes geometry_v7.json and ranges_v5.json for every assembly (the user's decision of 2026-11-23 on the spin range,
r4-decisions-06 option 2 applied to the named list assets.SPIN_ROUND_2026_11_23 only). No Isaac; CPU.

    assembly not in the list   geometry_v7.json = geometry_v6.json and ranges_v5.json = ranges_v4.json, byte for byte
    assembly in the list       geometry_v7.json = geometry_v6.json with symmetry_n = "round"; category, axis and every
                               other field unchanged; fit gets n_plug_rule (the plug rule's n), symmetry_rule and the
                               row of the all-assembly physical check (--physical), which must say n_phys "round" with
                               overlap0 = max_overlap = 0;
                               ranges_v5.json = grasp.ranges(id, "round"): the usual box pipeline (7 x 7 x 5 grid of
                               the base box with s = 15 deg, largest all-free box, 9 x 9 x 9 verification with 10 %
                               shrinks; hand / finger - socket hpp-fcl check at H*), as for the _physn candidates
geometry_v6.json / ranges_v4.json and every other file are read, never written. An existing v7 / v5 file is left as it is
when its content equals what would be written, otherwise the script stops (never overwrites). The spin axis switch must
be "plug_axis" (the approved value; ranges_v5.json belongs to it).

    python -u spin_round_v7.py --physical .../bore_physical_n_all.json --out .../spin_round_v7.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402

GEOMETRY_OLD, GEOMETRY_NEW = "geometry_v6.json", "geometry_v7.json"
RANGES_OLD, RANGES_NEW = "ranges_v4.json", "ranges_v5.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def put(path: Path, data: bytes) -> str:
    """Write data to path unless it exists; an existing file must hold the same bytes. Returns "written" / "same"."""
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"{path} exists with other content; not overwritten")
        return "same"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physical", required=True, help="bore_physical_n.py --controls all output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert A.SPEC_FILE == GEOMETRY_NEW and G.RANGES_FILE == RANGES_NEW, (A.SPEC_FILE, G.RANGES_FILE)
    assert C.GRASP_SPIN_AXIS == "plug_axis", C.GRASP_SPIN_AXIS
    phys = {r["assembly_id"]: r for r in json.loads(Path(a.physical).read_text())["rows"]}
    listed = set(A.SPIN_ROUND_2026_11_23)
    ids = sorted(set(A.assembly_ids()) | {p.parent.name for p in A.ASSET_DIR.glob(f"*/{GEOMETRY_OLD}")})
    assert listed <= set(ids), listed - set(ids)
    rows, t_all = [], time.time()
    # the listed assemblies are checked first: a physical-check row that does not say round stops before any write
    for aid in sorted(listed):
        p = phys.get(aid)
        ps = (p or {}).get("physical_symmetry") or {}
        if not p or p.get("n_phys") != "round" or ps.get("overlap0") != 0.0 or ps.get("max_overlap") != 0.0:
            raise RuntimeError(f"{aid}: physical check row does not say round with zero overlap: "
                               f"{None if p is None else {k: p.get(k) for k in ('n_phys', 'n_current')}} "
                               f"{ps.get('overlap0')} {ps.get('max_overlap')}")
    for aid in ids:
        t0 = time.time()
        d = A.ASSET_DIR / aid
        g_old, r_old = d / GEOMETRY_OLD, d / RANGES_OLD
        row = {"assembly_id": aid, "listed": aid in listed}
        gb = g_old.read_bytes()
        rb = r_old.read_bytes() if r_old.exists() else None
        if aid not in listed:
            row["geometry"] = put(d / GEOMETRY_NEW, gb)
            row["ranges"] = put(d / RANGES_NEW, rb) if rb is not None else f"no {RANGES_OLD}"
            row["identical_to_old"] = True
        else:
            g = json.loads(gb)
            n_old = g["symmetry_n"]
            assert phys[aid].get("n_current") == n_old, (aid, phys[aid].get("n_current"), n_old)
            ps = phys[aid]["physical_symmetry"]
            g["symmetry_n"] = "round"
            g["fit"] = {**g["fit"], "n_plug_rule": n_old, "symmetry_rule": A.SPIN_ROUND_RULE,
                        "physical_symmetry": {k: ps.get(k) for k in ("n_phys", "reliable", "overlap0", "max_overlap",
                                                                     "n_blocked", "tol", "step_deg", "n_planes")}}
            row["geometry"] = put(d / GEOMETRY_NEW, json.dumps(g, default=float).encode())
            spec = A.load_spec(aid)                                   # reads v7: symmetry_n round, fields valid
            assert spec.symmetry_n == "round" and spec.category == json.loads(gb)["category"], (aid, spec.category)
            old = json.loads(rb) if rb is not None else {}
            rg = G.ranges(aid, "round")
            rg.update(symmetry_rule=A.SPIN_ROUND_RULE, n_plug_rule=n_old)
            row["ranges"] = put(d / RANGES_NEW, json.dumps(rg, default=float).encode())
            row.update(n_old=n_old, category=spec.category, box_old=old.get("box"), box_new=rg.get("box"),
                       verified_old=old.get("verified"), verified_new=rg.get("verified"), shrinks_new=rg.get("shrinks"),
                       worst_margin_old_mm=None if old.get("worst_margin_m") is None else old["worst_margin_m"] * 1e3,
                       worst_margin_new_mm=None if rg.get("worst_margin_m") is None else rg["worst_margin_m"] * 1e3,
                       base_free_fraction_new=rg.get("base_free_fraction"), identical_to_old=False)
            cand = d / "ranges_v4_physn.json"
            if cand.exists():
                row["box_physn_candidate"] = json.loads(cand.read_text()).get("box")
        row["sha256_ranges_v5"] = sha((d / RANGES_NEW).read_bytes()) if (d / RANGES_NEW).exists() else None
        row["sha256_ranges_v4"] = sha(rb) if rb is not None else None
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        if aid in listed:
            print(json.dumps({k: row.get(k) for k in ("assembly_id", "n_old", "box_old", "box_new", "verified_new",
                                                      "shrinks_new", "seconds")}), flush=True)
    changed = [r["assembly_id"] for r in rows if r.get("sha256_ranges_v5") != r.get("sha256_ranges_v4")]
    Path(a.out).write_text(json.dumps({"geometry": [GEOMETRY_OLD, GEOMETRY_NEW], "ranges": [RANGES_OLD, RANGES_NEW],
                                       "listed": sorted(listed), "ranges_sha256_changed": changed,
                                       "switches": C.switches(), "seconds": round(time.time() - t_all, 1),
                                       "rows": rows}, indent=1, default=float))
    print(f"{len(rows)} assemblies; ranges sha256 changed for {len(changed)}: {changed}; wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
