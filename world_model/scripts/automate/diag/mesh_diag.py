"""Milestone-1b diagnosis, mesh side (no Isaac app): for each assembly
  * mesh health of plug.obj / socket.obj (watertight, winding, volume, bodies);
  * where checks.penetration finds plug points inside the socket at the assembled pose (signed distance > 0.05 mm),
    and whether a second inside-test (ray containment) agrees;
  * section overlap profile at the assembled pose: for every plane z (0.2 mm grid, fit.py sections), the area of
    plug-section AND socket-section and its thickness estimate t = 2 A / perimeter (thin sliver of width t);
  * the nominal grasp's finger/hand-socket check (grasp.FingerSocketCheck) and a sweep over dp (fingers moved along
    the plug axis; dp < 0 = away from the tip) at dtheta = dpsi = 0: first dp where the check is free, first dp where
    nothing collides.
    python -u mesh_diag.py --ids 00141,00855 --out /mnt/storage3/qingxin/automate/m1b/diag/mesh
Writes <out>/<id>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import fit as F  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402


def health(m) -> dict:
    return {"watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent),
            "is_volume": bool(m.is_volume), "bodies": int(len(m.split(only_watertight=False))),
            "volume_cm3": float(m.volume * 1e6), "euler": int(m.euler_number), "faces": int(len(m.faces))}


def penetration_detail(plug, sock, n=20000) -> dict:
    import trimesh
    pts, _ = trimesh.sample.sample_surface_even(plug, n, seed=0)          # same sampling as checks.penetration
    pts = np.vstack([pts, plug.vertices])
    sd = trimesh.proximity.signed_distance(sock, pts)
    bad = sd > 5e-5
    out = {"n_points": int(len(pts)), "n_inside_sd": int(bad.sum()), "max_sd_mm": float(max(sd.max(), 0) * 1e3)}
    if bad.any():
        p = pts[bad]
        out["inside_z_mm"] = [float(p[:, 2].min() * 1e3), float(p[:, 2].max() * 1e3)]
        out["inside_r_mm"] = [float(np.hypot(p[:, 0], p[:, 1]).min() * 1e3), float(np.hypot(p[:, 0], p[:, 1]).max() * 1e3)]
        k = int(np.argmax(sd))
        out["deepest_point_mm"] = (pts[k] * 1e3).round(3).tolist()
        try:
            cont = sock.contains(p)
            out["ray_contains_of_inside"] = int(cont.sum())
        except Exception as exc:                                              # noqa: BLE001
            out["ray_contains_of_inside"] = f"error {exc}"
        # distance of those points to the socket surface, unsigned (how far the point is from any socket face)
        _, d, _ = trimesh.proximity.closest_point(sock, p)
        out["inside_unsigned_dist_mm_max"] = float(d.max() * 1e3)
    return out


def overlap_profile(plug, sock) -> dict:
    import shapely
    z0 = max(float(plug.bounds[0, 2]), float(sock.bounds[0, 2]))
    z1 = min(float(plug.bounds[1, 2]), float(sock.bounds[1, 2]))
    zs = np.arange(z0 + F.DZ / 2, z1, F.DZ)
    if len(zs) == 0:
        return {"rows": []}
    P = F._sections(plug, zs)
    S = F._sections(sock, zs)
    rows = []
    for z, p, s in zip(zs, P, S):
        if p.is_empty or s.is_empty:
            continue
        inter = shapely.intersection(p, s)
        a = float(inter.area)
        if a > 1e-10:                                                       # > 1e-4 mm^2
            rows.append({"z_mm": float(z * 1e3), "area_mm2": a * 1e6, "thick_mm": float(2 * a / max(inter.length, 1e-12) * 1e3),
                         "plug_area_mm2": float(p.area * 1e6)})
    out = {"n_planes": int(len(zs)), "n_overlap_planes": len(rows)}
    if rows:
        out["z_range_mm"] = [rows[0]["z_mm"], rows[-1]["z_mm"]]
        k = int(np.argmax([r["thick_mm"] for r in rows]))
        out["max_thick_mm"], out["max_thick_z_mm"] = rows[k]["thick_mm"], rows[k]["z_mm"]
        out["max_area_mm2"] = max(r["area_mm2"] for r in rows)
    out["rows"] = rows
    return out


def finger_sweep(aid) -> dict:
    chk = G.FingerSocketCheck(aid)
    nom = chk.check([0.0, 0.0, 0.0])
    rows = []
    for dp in np.arange(0.010, -0.0301, -0.0005):
        r = chk.check([0.0, float(dp), 0.0])
        rows.append({"dp_mm": round(float(dp) * 1e3, 2), "free": r["free"], "hits": r["hits"],
                     "margin_mm": None if r["margin_m"] is None else r["margin_m"] * 1e3, "q_f_mm": r["q_f"] * 1e3})
    neg = [r for r in rows if r["dp_mm"] <= 0]                               # dp = 0, -0.5, -1, ...
    first_free = next((r["dp_mm"] for r in neg if r["free"]), None)
    first_nohit = next((r["dp_mm"] for r in neg if not r["hits"]), None)
    return {"nominal": {k: (v * 1e3 if isinstance(v, float) else v) for k, v in nom.items()},
            "first_free_dp_mm_towards_top": first_free, "first_no_contact_dp_mm_towards_top": first_nohit, "rows": rows}


def main():
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1b/diag/mesh")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for aid in args.ids.split(","):
        plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
        sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
        res = {"assembly_id": aid, "plug": health(plug), "socket": health(sock),
               "plug_bounds_mm": (plug.bounds * 1e3).round(3).tolist(), "socket_bounds_mm": (sock.bounds * 1e3).round(3).tolist(),
               "penetration": penetration_detail(plug, sock), "overlap": overlap_profile(plug, sock),
               "fingers": finger_sweep(aid)}
        (out / f"{aid}.json").write_text(json.dumps(res, indent=1, default=float))
        o, p, f = res["overlap"], res["penetration"], res["fingers"]
        print(aid, "plug", res["plug"], "socket", res["socket"], flush=True)
        print("   pen", {k: v for k, v in p.items()}, flush=True)
        print("   overlap planes", o["n_overlap_planes"], "z", o.get("z_range_mm"), "max thick", o.get("max_thick_mm"),
              "at", o.get("max_thick_z_mm"), "max area", o.get("max_area_mm2"), flush=True)
        print("   fingers nominal", f["nominal"], "first free dp", f["first_free_dp_mm_towards_top"],
              "first no-contact dp", f["first_no_contact_dp_mm_towards_top"], flush=True)


if __name__ == "__main__":
    main()
