"""Port plan T8 (Isaac, GPU 1, one capped scope; started by online_smoke.sh): does a saved AutoMate state come back?

    python -u online_restore_check.py --grasps <collection> --grasp <key> --out <t8.json> [--repeats 3] [--seed 9100000]

On one grasp: reset with attempt 0's seed and socket pose, the reference, the collector's placement (wm.online.automate320,
exactly as an online attempt), then `repeats` times:
    save        robot.base.get_state_dict() (Robot.get_state: scene, PD targets, last command, episode, grasp)
    decision    8 control steps ramping the joint targets to the measured joints + the joint offset that moves the hand
                1 mm along the FK insertion axis (common320.offset on shared320's Jacobian): one scripted decision
    restore     robot.base.set_state_dict(saved)
and records: the hand deviation from the saved hand right after the restore and after one control step holding the
restored command [mm, deg], the joint deviation [rad], the socket pose against the saved one after that step
(AutomateScene.check_socket_pose: raises above 1e-5 m / 1e-5 rad; the error is recorded, not raised), and the spread
of the decision's end hand between repeats [mm] (PhysX contact caches are not part of the state). No model, no torch
import. Thresholds are not decided here: the numbers go to the user (plan item 1: only then --probe-max > 0).
Before the Isaac start, as the online prepare (exit 3 otherwise): automate320.check_switches (plan.json switches as a
whole, physics device), automate320.check_inputs (the collection's frozen assets: WM_AUTOMATE_ASSETS =
<collection>/inputs/assets, asset_files, transfer) and automate320.check_plan_seeds on attempt 0's reset seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--grasps", required=True)
ap.add_argument("--grasp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--seed", type=int, default=9_100_000)
args = ap.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    print(f"REFUSED: CUDA_VISIBLE_DEVICES must be 1, found {os.environ.get('CUDA_VISIBLE_DEVICES')!r}", flush=True)
    os._exit(3)
port = args.grasp.rpartition("_g")[0]
g = json.loads((Path(args.grasps) / port / args.grasp / "grasp.json").read_text())
plan = json.loads((Path(args.grasps) / "plan.json").read_text())
gi = next(int(p["gi"]) for p in plan["grasps"] if p["grasp"] == args.grasp)
from wm.online import automate320 as A  # noqa: E402  (no Isaac import)

try:
    A.check_switches(plan)
    assets = A.check_inputs(Path(args.grasps), plan)
    A.check_plan_seeds(plan, args.seed, [A.attempt_streams(args.seed, port, gi, 0, None)["reset_seed"]])
except SystemExit as exc:
    print(f"REFUSED: {exc}", flush=True)
    os._exit(3)

from wm.sim_automate.app import launch, shutdown  # noqa: E402

app = launch(headless=True, cameras=True)
code = 1
try:
    from wm.data.kinematics import HandFK
    from wm.online.common320 import offset
    from wm.online.shared320 import AttemptBase
    from wm.sim.geometry import pose_distance
    from wm.sim_automate.checks import ResetError
    from wm.sim_automate.robot import Robot
    t0 = time.time()
    robot = Robot(port)
    streams = A.attempt_streams(args.seed, port, gi, 0, None)
    st = A.reset_and_reference(robot, g, 0, streams)
    placed = "outcome" not in st["entry"] and A.place_start(robot, st, streams)
    res = {"grasp": args.grasp, "seed": args.seed, "assets": assets, "entry": st["entry"], "placed": bool(placed),
           "repeats": []}
    if placed:
        base = AttemptBase(robot, st["H"], st["axis"], HandFK(robot), st["F"])
        socket0 = robot.sc.socket_T(robot.env_id)
        snap, hand0, q0 = robot.base.get_state_dict(), robot.hand_pose().copy(), robot.qpos().copy()
        q9 = robot.qpos().astype(np.float64)
        d = offset(base.jacobian(q9), st["axis"] * 0.001)
        ends = []
        for rep in range(args.repeats):
            for j in range(1, 9):
                robot._command(q9[:7] + d * j / 8.0, 1)
            ends.append(robot.hand_pose().copy())
            robot.base.set_state_dict(snap)
            dt0, dr0 = pose_distance(robot.hand_pose(), hand0)
            dq0 = float(np.abs(robot.qpos() - q0).max())
            robot._command(robot._q_cmd if robot._q_cmd is not None else q0[:7], 1)
            dt1, dr1 = pose_distance(robot.hand_pose(), hand0)
            try:
                sock = robot.sc.check_socket_pose(robot.env_id, socket0)
                sock_err = None
            except ResetError as exc:
                sock, sock_err = None, str(exc)[:500]
            res["repeats"].append({"restore_dev_mm": dt0 * 1e3, "restore_dev_deg": float(np.degrees(dr0)),
                                   "restore_joint_dev_rad": dq0, "after_step_dev_mm": dt1 * 1e3,
                                   "after_step_dev_deg": float(np.degrees(dr1)), "socket": sock, "socket_error": sock_err,
                                   "decision_end_vs_first_mm": float(np.linalg.norm(ends[-1][:3, 3] - ends[0][:3, 3]) * 1e3),
                                   "decision_move_mm": float(np.linalg.norm(ends[-1][:3, 3] - hand0[:3, 3]) * 1e3)})
            robot.base.set_state_dict(snap)
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "entry"}, indent=1, default=float), flush=True)
    code = 0
except BaseException:
    import traceback
    traceback.print_exc()
finally:
    shutdown(code)
