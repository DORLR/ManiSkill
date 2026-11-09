"""Control for check_pitch_geometry.py: can the plug's points alone fix the plug's axis direction?

Pairs of inserted frames (reference insertions) of two grasps of the same shape: in each, the plug
axis in the camera frame is known (inserted -> plug axis = insertion axis, FK + board fit). ICP between
the two plug point clouds (camera frames) gives a rotation; error = angle between the rotated axis of
grasp A and the axis of grasp B. Needed precision: ~0.4 deg (the size of the misalignments to read).

    python world_model/scripts/analysis/check_plug_axis_icp.py --out $Q/runs/checks_v3g
"""

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from check_pitch_geometry import icp, plug_points_cam, rot_axis
from wm.data import index
from wm.data.datasets import port_axes_ok


def pair(item):
    a, b = item
    data = []
    for e in (a, b):
        with np.load(e["path"]) as f:
            j = len(f["img_step"]) - 1
            P = plug_points_cam(f, j)
            Rcw = f["cam_T_world"][j][:3, :3].astype(np.float64)
        R_P = np.asarray(e["R_P"], float)
        data.append((P, Rcw @ R_P[:, 0], Rcw @ R_P[:, 1]))
    (Pa, axa, _), (Pb, axb, lat_b) = data
    if len(Pa) < 30 or len(Pb) < 30:
        return None
    best = None
    for deg in range(-45, 46, 5):
        R, t, rms = icp(Pa, Pb, rot_axis(lat_b, deg))
        if best is None or rms < best[2]:
            best = (R, t, rms)
    est = best[0] @ axa
    cos = lambda u, v: np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
    return {"err_deg": float(np.degrees(np.arccos(cos(est, axb)))),
            "true_axis_diff_deg": float(np.degrees(np.arccos(cos(axa, axb)))), "rms_mm": best[2] * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=300)
    args = ap.parse_args()
    refs = defaultdict(list)
    for e in index.load(args.data):
        if e["kind"] == "reference" and e.get("outcome") == "inserted" and port_axes_ok(e):
            refs[e["port"]].append(e)
    items = [(rs[k], rs[k + 1]) for rs in refs.values() for k in range(min(3, len(rs) - 1))]
    rng = np.random.default_rng(0)
    items = [items[i] for i in rng.permutation(len(items))[:args.pairs]]
    with mp.get_context("spawn").Pool(12) as pool:
        rows = [r for r in pool.map(pair, items, chunksize=2) if r]
    err, diff = np.array([r["err_deg"] for r in rows]), np.array([r["true_axis_diff_deg"] for r in rows])
    q = lambda x: {"median": round(float(np.median(x)), 3), "p25": round(float(np.percentile(x, 25)), 3),
                   "p75": round(float(np.percentile(x, 75)), 3), "p90": round(float(np.percentile(x, 90)), 3)}
    out = {"pairs": len(rows), "true_axis_difference_deg": q(diff), "icp_axis_error_deg": q(err), "by_axis_difference": {}}
    for lo, hi in ((0, 5), (5, 15), (15, 60)):
        m = (diff >= lo) & (diff < hi)
        if m.any():
            out["by_axis_difference"][f"{lo}-{hi}deg"] = {"pairs": int(m.sum()), "icp_axis_error_deg": q(err[m]),
                                                          "median_rms_mm": round(float(np.median([r["rms_mm"] for r, k in zip(rows, m) if k])), 3)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plug_axis_icp.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
