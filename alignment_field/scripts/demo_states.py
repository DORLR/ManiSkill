"""Relative state u of every demonstration frame, from the logged joint angles only.

python scripts/demo_states.py --data "data/v4_demos/*.npz" --out data/v4_demo_states.npz
Per demonstration (as in a teleoperation log: joint angles of every frame):
  H_star  hand pose of the last frame (FK); the demonstration ends inserted
  e_x     insertion axis = principal direction of the hand positions while the
          pins are in the holes (frames within pin length of H_star; the motion
          there is constrained to the hole axis)
  R_P     port_axes(e_x) (gravity from the robot base, as at test time)
  H_pre   H_star retreated by pin length + GAP0 along e_x (paf.collect.establish_reference)
  u       relative_state(FK(q_t), H_pre, R_P) / SIGMA for every frame
No simulator state is read; the simulator is only used for its kinematic model
(FK) and the port's nominal pin length (CAD).
"""

import argparse
import glob
import json
import multiprocessing as mp
from collections import defaultdict

import numpy as np


def _work(job):
    port, files = job
    from paf import config as C
    from paf.env_adapter import Robot
    from paf.geometry import make_T, port_axes, relative_state
    robot = Robot(port)
    robot.reset(1, -40.0, -0.020)
    q_full = robot.qpos().copy()
    planner = robot.base._planner()
    pm = planner.pinocchio_model
    idx = planner.user_link_names.index("panda_hand_tcp")
    from paf.env_adapter import _np, pose_to_T
    base = robot.base.agent.robot.pose
    T_base = pose_to_T(_np(base.p)[0], _np(base.q)[0])

    def fk(q7):
        q = q_full.copy()
        q[:7] = q7
        pm.compute_forward_kinematics(q)
        p = pm.get_link_pose(idx)
        return T_base @ pose_to_T(np.asarray(p.p), np.asarray(p.q))

    pin = robot.port_spec.pin_length_m
    back = pin + C.GAP0
    rows = []
    for fi, f in files:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if "qpos" not in d.files:
            continue
        for k in np.unique(d["demo"]):
            Q = d["qpos"][d["demo"] == k].astype(np.float64)
            H = [fk(q) for q in Q]
            P = np.array([h[:3, 3] for h in H])
            H_star = H[-1]
            near = np.linalg.norm(P - P[-1], axis=1) <= pin
            if near.sum() < 3:
                continue
            X = P[near] - P[near].mean(0)
            ax = np.linalg.svd(X, full_matrices=False)[2][0]
            first = np.where(near)[0][0]
            if np.dot(P[-1] - P[first], ax) < 0:
                ax = -ax
            line_res = np.linalg.norm(X - np.outer(X @ ax, ax), axis=1).max()
            R_P = port_axes(ax)
            H_pre = make_T(H_star[:3, :3], H_star[:3, 3] - back * R_P[:, 0])
            U = np.array([relative_state(h, H_pre, R_P) for h in H]) / C.SIGMA
            rows.append(dict(port=port, file=fi, demo=int(k), pitch=meta["pitch"], grasp_pos=meta["grasp_pos"],
                             u=U, stroke=back / C.SIGMA_T, line_res_mm=1000 * line_res, n_near=int(near.sum())))
    robot.env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v4_demos/*.npz")
    ap.add_argument("--out", default="data/v4_demo_states.npz")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    by_port = defaultdict(list)
    for fi, f in enumerate(sorted(glob.glob(args.data))):
        by_port[f.split("/")[-1].rsplit("_", 1)[0]].append((fi, f))
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(_work, sorted(by_port.items())):
            rows += r
            print(f"{len(rows)} demos", flush=True)
    u = np.concatenate([r["u"] for r in rows]).astype(np.float32)
    rep = lambda key: np.concatenate([np.full(len(r["u"]), r[key]) for r in rows])
    np.savez(args.out, u=u, port=rep("port"), grasp_file=rep("file"), demo=rep("demo"), pitch=rep("pitch"),
             grasp_pos=rep("grasp_pos"), stroke=rep("stroke"),
             frame=np.concatenate([np.arange(len(r["u"])) for r in rows]),
             line_res_mm=np.array([r["line_res_mm"] for r in rows]), n_near=np.array([r["n_near"] for r in rows]))
    print(f"saved {len(u)} frames of {len(rows)} demos -> {args.out}")


if __name__ == "__main__":
    main()
