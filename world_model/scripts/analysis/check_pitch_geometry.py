"""Is the plug's pitch misalignment in the observation at all?  A learning-free geometric check.

For each contact trial at push start, from that single frame (true depth + plug segmentation):
the plug's axis direction is found by registering the frame's plug points to a template: the plug
points of the inserted frame of *another* grasp of the same shape, whose plug axis is known there
(inserted -> plug axis = insertion axis). Registration: point-to-point ICP in the camera frame from
several initial rotations about the pitch axis (no grasp parameter is used), best residual kept.
The plug axis is then compared with the hole axis (port axes from the board fit of this trajectory)
-> estimated pitch / yaw misalignment, correlated with the sampled offset (simulator truth), per grasp
condition. High correlation also under grasp-pitch extrapolation = the information is in the frame
and the networks fail to extract it; low = this view / depth does not determine it.

    python world_model/scripts/analysis/check_pitch_geometry.py --out $Q/runs/checks_v3g --workers 12
"""

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from wm.analysis.probes import PUSH_PHASE
from wm.data import index
from wm.data.datasets import port_axes_ok

PLUG = 1
DEPTH_UNIT_M = 1e-4
N_PTS = 1500


def plug_points_cam(f, j: int) -> np.ndarray:
    """Plug points of image j in the camera frame (OpenCV), metres."""
    depth = f["depth"][j].astype(np.float64) * DEPTH_UNIT_M
    seg = f["seg"][j]
    v, u = np.nonzero((seg == PLUG) & (depth > 0))
    if len(u) < 30:
        return np.zeros((0, 3))
    K = f["K"].astype(np.float64)
    z = depth[v, u]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    p = np.stack([x, y, z], 1)
    if len(p) > N_PTS:
        p = p[np.random.default_rng(0).choice(len(p), N_PTS, replace=False)]
    return p


def rot_axis(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def icp(src, dst, R0, iters=40):
    """Point-to-point ICP src -> dst (rigid), start rotation R0 about the centroids. -> R, t, rms."""
    tree = cKDTree(dst)
    cs, cd = src.mean(0), dst.mean(0)
    R, t = R0, cd - R0 @ cs
    for _ in range(iters):
        moved = src @ R.T + t
        d, idx = tree.query(moved)
        keep = d < np.percentile(d, 90)                     # trim outliers / partial overlap
        a, b = src[keep], dst[idx[keep]]
        ma, mb = a.mean(0), b.mean(0)
        H = (a - ma).T @ (b - mb)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mb - R @ ma
    d, _ = tree.query(src @ R.T + t)
    return R, t, float(np.sqrt(np.mean(np.minimum(d, np.percentile(d, 90)) ** 2)))


def work(item):
    trial, template = item
    try:
        with np.load(template["path"]) as f:
            jt = len(f["img_step"]) - 1                      # inserted frame of the template grasp
            P_ref = plug_points_cam(f, jt)
            Rcw_ref = f["cam_T_world"][jt][:3, :3].astype(np.float64)       # world -> camera rotation
        axis_ref_cam = Rcw_ref @ np.asarray(template["R_P"], float)[:, 0]  # plug axis = insertion axis there
        with np.load(trial["path"]) as f:
            phase = f["phase"]
            push = np.flatnonzero(phase == PUSH_PHASE)
            j = int(np.searchsorted(f["img_step"], push[0], side="right") - 1)
            P_now = plug_points_cam(f, j)
            Rcw = f["cam_T_world"][j][:3, :3].astype(np.float64)
            meta = json.loads(str(f["meta"]))
        if len(P_ref) < 30 or len(P_now) < 30:
            return None
        R_P = np.asarray(trial["R_P"], float)
        pitch_axis_cam = Rcw @ R_P[:, 1]                     # lateral port axis seen from the camera
        best = None
        for deg in range(-45, 46, 5):                        # hypotheses: no grasp parameter is used
            R, t, rms = icp(P_ref, P_now, rot_axis(pitch_axis_cam, deg))
            if best is None or rms < best[2]:
                best = (R, t, rms)
        a_cam = best[0] @ axis_ref_cam
        a_port = R_P.T @ (Rcw.T @ a_cam)                     # plug axis in port axes (hole axis = +x)
        pitch = np.degrees(np.arctan2(a_port[2], a_port[0]))
        yaw = np.degrees(np.arctan2(a_port[1], a_port[0]))
        off = meta["offset"]["rpy_deg"]
        return {"condition": trial["condition"], "split": trial["split"], "est_pitch": pitch, "est_yaw": yaw,
                "true_roll": off[0], "true_pitch": off[1], "true_yaw": off[2], "rms_mm": best[2] * 1000,
                "n_now": len(P_now), "n_ref": len(P_ref)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-condition", type=int, default=150)
    args = ap.parse_args()
    entries = [e for e in index.load(args.data) if port_axes_ok(e)]
    refs = defaultdict(list)
    for e in entries:
        if e["kind"] == "reference" and e.get("outcome") == "inserted":
            refs[e["port"]].append(e)
    rng = np.random.default_rng(0)
    items, per = [], defaultdict(int)
    trials = [e for e in entries if e["kind"] == "contact" and e.get("outcome") in ("inserted", "jammed")
              and e["split"] in ("train", "ood_val")]
    for i in rng.permutation(len(trials)):
        e = trials[int(i)]
        key = (e["split"], e["condition"])
        if per[key] >= args.per_condition:
            continue
        others = [r for r in refs[e["port"]] if r["grasp"] != e["grasp"]]
        if not others:
            continue
        items.append((e, others[0]))
        per[key] += 1
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for r in pool.map(work, items, chunksize=4) if r is not None]
    errors = [r["error"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    out = {"items": len(items), "used": len(rows), "errors": len(errors), "first_errors": errors[:3], "by_condition": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["condition"])].append(r)
    for (split, cond), rs in sorted(groups.items()):
        est_p = np.array([r["est_pitch"] for r in rs]); tru_p = np.array([r["true_pitch"] for r in rs])
        est_y = np.array([r["est_yaw"] for r in rs]); tru_y = np.array([r["true_yaw"] for r in rs])
        c = lambda a, b: round(float(abs(np.corrcoef(a, b)[0, 1])), 3)
        out["by_condition"][f"{split}/{cond}"] = {
            "trials": len(rs), "pitch_corr": c(est_p, tru_p), "yaw_corr": c(est_y, tru_y),
            "pitch_corr_with_true_yaw": c(est_p, tru_y),
            "median_icp_rms_mm": round(float(np.median([r["rms_mm"] for r in rs])), 3),
            "median_plug_points": int(np.median([r["n_now"] for r in rs]))}
    p = Path(args.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "pitch_geometry.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
