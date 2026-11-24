"""Milestone-1b per-assembly table (all assemblies) as JSON: mesh analysis (m1b/analysis.json, with the geometry and
the grasp ranges replaced by the CURRENT files assets.SPEC_FILE and grasp.RANGES_FILE through report_m1b.current), simulation checks (m1b/<batch>/<id>.json, default batch_r3_cpu; with --rerun, an assembly whose process
died is taken from <rerun>/<id>.json when present, and marked so). Same depth-check rule as report_m1b.py; "stale" =
report_m1b.stale_fields (provenance that differs from the current files / switches: such a result must be re-run).
    python table_m1b.py --root /mnt/storage3/qingxin/automate/m1b --assets /mnt/storage3/qingxin/automate/assets \
        --out /mnt/storage3/qingxin/automate/m1b/diag/table_m1b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

CAT = {"round": "round", "polygonal": "non-round", "keyed": "non-round", "flat": "flat", "multi-pin": "multi-pin",
       "sleeve": "sleeve", "unknown": "unknown"}


def gap_at(sweep, d):
    best = None
    for x, g in sweep:
        if x <= d + 1e-9:
            best = g
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/storage3/qingxin/automate/m1b")
    ap.add_argument("--assets", default="/mnt/storage3/qingxin/automate/assets")
    ap.add_argument("--batch", default="batch_r3_cpu")
    ap.add_argument("--rerun", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from wm.sim_automate import assets as A
    A.ASSET_DIR = Path(a.assets)
    import report_m1b as RM
    RM.A_DIR = A.ASSET_DIR
    root = Path(a.root)
    rows = []
    cur = [RM.current(r) for r in json.loads((root / "analysis.json").read_text())]
    print("symmetry n changed since geometry_v4:", RM.changed_n(cur))
    for r in cur:
        aid = r["assembly_id"]
        if "error" in r:
            rows.append({"id": aid, "error": r["error"]})
            continue
        f = r["fit"]
        L, dd, Lt = r["pin_length_m"] * 1e3, r["disassembly_dist_m"] * 1e3, r["depth_top_m"] * 1e3
        if abs(dd - L) <= 4:
            chk = "agree"
        elif abs(dd - Lt) <= 4:
            g = gap_at(f["sweep_coarse"], (L + 1) / 1e3)
            chk = f"lead-in {Lt - L:.1f} mm" + ("" if g is None else f" (gap at L+1 mm: {g * 1e3:.1f} mm)")
        elif dd < L - 4:
            chk = "early-free"
        else:
            chk = f"other (L_top {Lt:.1f} mm)"
        n = r["symmetry_n"]
        rg = r["ranges"]
        row = {"id": aid, "category": CAT.get(r["category"], r["category"]), "category_fit": r["category"],
               "twelve_gon": n == 12 and (f.get("radial_spread") or 1) < 0.04,
               "n": n, "P_deg": None if n in (None, "round") else 360.0 / n, "spin_limit_deg": rg.get("spin_limit_deg"),
               "c_mm": r["clearance_m"] * 1e3, "L_mm": L, "d_dis_mm": dd, "L_top_mm": Lt, "depth_check": chk,
               "N_max": r["max_decisions"], "size_mm": f.get("equiv_diameter_m", float("nan")) * 1e3,
               "long_mm": f.get("long_side_m", float("nan")) * 1e3, "obj_overlap_mm": r.get("obj_overlap_mm")}
        base = {"dtheta_deg": (-20.0, 20.0), "dp_m": (-0.010, 0.010), "dpsi_deg": (-(rg.get("spin_limit_deg") or 15.0), rg.get("spin_limit_deg") or 15.0)}
        if rg.get("box"):
            b = rg["box"]
            row["box"] = {"dtheta_deg": b["dtheta_deg"], "dp_mm": [v * 1e3 for v in b["dp_m"]], "dpsi_deg": b["dpsi_deg"]}
            row["width_fraction"] = {k: (b[k][1] - b[k][0]) / (base[k][1] - base[k][0]) for k in base}
            row["shrunk"] = any(v < 1 - 1e-9 for v in row["width_fraction"].values())
            row["box_verified"], row["box_shrinks"] = rg["verified"], rg["shrinks"]
            row["worst_margin_mm"] = None if rg.get("worst_margin_m") is None else rg["worst_margin_m"] * 1e3
        else:
            nom = rg.get("nominal") or {}
            row["box"] = None
            row["box_reason"] = rg.get("reason", "")
            row["base_free_fraction"] = rg.get("base_free_fraction")
        row["base_free_fraction"] = rg.get("base_free_fraction")
        bf = root / a.batch / f"{aid}.json"
        b = json.loads(bf.read_text()) if bf.exists() else {}
        row["source"] = a.batch
        if a.rerun and "i" not in b and (root / a.rerun / f"{aid}.json").exists():
            b = json.loads((root / a.rerun / f"{aid}.json").read_text())
            row["source"] = a.rerun + f" (the {a.batch} process died in this assembly)"
        if b:
            row["stale"] = RM.stale_fields(b, aid)
        if "error" in b:
            row["error"] = b["error"]
        if "i" in b:
            x = b["i"]
            row["i"] = {k: x[k] for k in ("pass", "move_ok", "depth_left_mm", "lateral_mm", "tilt_deg", "max_penetration_mm",
                                          "hand_finger_socket_N", "plug_socket_N")}
        if "ii" in b:
            x = b["ii"]
            row["ii"] = {k: x.get(k) for k in ("pass", "placed", "completed", "push_end", "depth_left_mm", "lateral_mm",
                                               "tilt_deg", "success_end", "success_any", "height_mm", "keypoint_dist_mm",
                                               "max_plug_socket_N", "max_hand_finger_socket_N")}
        if "iii" in b:
            x = b["iii"]
            row["iii"] = x if "skipped" in x else {k: x.get(k) for k in ("pass", "grasp", "mesh_margin_mm", "move_ok",
                                                                             "depth_left_mm", "success_end",
                                                                             "hand_finger_socket_N", "robot_socket_pairs_N",
                                                                             "hand_finger_max_N", "plug_socket_N")}
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    n_i = sum(1 for r in rows if r.get("i", {}).get("pass"))
    n_ii = sum(1 for r in rows if r.get("ii", {}).get("pass"))
    n_iii = sum(1 for r in rows if r.get("iii", {}).get("pass"))
    n_st = sum(1 for r in rows if r.get("stale"))
    print(f"assemblies {len(rows)}, (i) pass {n_i}, (ii) pass {n_ii}, (iii) pass {n_iii}, stale results {n_st}, "
          f"wrote {a.out}")


if __name__ == "__main__":
    main()
