"""Plug-in-hand labels for the plug network (FK only).

M = R_pre^T R_P: the port axes in the hand frame at the pre-insertion pose, constant per grasp
(H_pre and R_P come from the one insertion per grasp, by forward kinematics). Across grasps M
turns about one fixed axis by an angle theta (it follows the grasp pitch). This script finds the
axis and a reference from the training packs, writes theta (deg) per view into each packed dir,
and saves the parametrisation M(theta) = M_ref @ Rot(axis, theta) to --frame.

python scripts/plug_labels.py --packed data/packed_v6 data/packed_v7 data/packed_v6_abcd --frame runs_v7/plug_frame.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def grasp_M(f):
    d = np.load(f)
    return d["H_pre"][:3, :3].T @ d["R_P"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", nargs="+", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--fit-on", type=int, default=2, help="number of leading packs (training) used to fit the axis")
    args = ap.parse_args()
    Ms = {}
    for p in args.packed:
        files = json.loads((Path(p) / "ports.json").read_text())["files"]
        Ms[p] = [grasp_M(f) for f in files]
    fit = [M for p in args.packed[:args.fit_on] for M in Ms[p]]
    M_ref = fit[0]
    rv = np.array([Rot.from_matrix(M_ref.T @ M).as_rotvec() for M in fit])
    big = rv[np.linalg.norm(rv, axis=1) > np.radians(2)]
    axis = np.linalg.svd(big, full_matrices=False)[2][0]
    if (big @ axis).mean() < 0:
        axis = -axis
    th = rv @ axis
    off = np.linalg.norm(rv - np.outer(th, axis), axis=1)
    centre = np.degrees(np.median(th))
    M_ref = M_ref @ Rot.from_rotvec(axis * np.radians(centre)).as_matrix()   # re-centre: theta ~ 0 mid-range
    Path(args.frame).parent.mkdir(parents=True, exist_ok=True)
    Path(args.frame).write_text(json.dumps({"M_ref": M_ref.tolist(), "axis": axis.tolist()}))
    print(f"axis {np.round(axis, 4)}; off-axis residual max {np.degrees(off.max()):.3f} deg over {len(fit)} grasps")
    for p in args.packed:
        grasp = np.load(Path(p) / "grasp.npy")
        theta_g = np.array([np.degrees(Rot.from_matrix(M_ref.T @ M).as_rotvec() @ axis) for M in Ms[p]])
        res = [np.degrees(np.linalg.norm(Rot.from_matrix(M_ref.T @ M).as_rotvec() - axis * np.radians(t)))
               for M, t in zip(Ms[p], theta_g)]
        np.save(Path(p) / "theta.npy", theta_g[grasp].astype(np.float32))
        print(f"{p}: {len(theta_g)} grasps, theta {theta_g.min():.1f}..{theta_g.max():.1f} deg, off-axis max {max(res):.3f} deg")


if __name__ == "__main__":
    main()
