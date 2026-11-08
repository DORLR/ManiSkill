"""Forward kinematics of the Panda hand (TCP) for arrays of joint angles.

The same computation as `paf.env_adapter.Robot.hand_pose()` (MPlib's Pinocchio model of the
robot, link `panda_hand_tcp`, then the fixed robot base pose), done for many joint vectors at
once. Used offline to turn the commanded joint targets of every recorded step into commanded
hand poses (the world model's action condition is an end-effector trajectory, as in Ctrl-World),
and online to do the same for the policy's actions. Real-robot equivalent: the robot's own FK.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._paths import use_alignment_field

use_alignment_field()


def _pose_to_T(p, q) -> np.ndarray:
    """position (3,) and quaternion (w, x, y, z) -> 4x4."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class HandFK:
    """FK from a live environment's planner model (built once; the robot model and base pose are the
    same for every port shape)."""

    def __init__(self, robot):
        planner = robot.base._planner()
        self.pm = planner.pinocchio_model
        self.idx = planner.user_link_names.index("panda_hand_tcp")
        base = robot.base.agent.robot.pose
        to_np = lambda x: np.asarray(x.cpu() if hasattr(x, "cpu") else x)
        self.base_T = _pose_to_T(to_np(base.p)[0], to_np(base.q)[0])

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """q (N, 9) full joint vectors (7 arm + 2 finger) -> (N, 4, 4) hand poses, world."""
        out = np.empty((len(q), 4, 4))
        for i, qi in enumerate(np.asarray(q, np.float64)):
            self.pm.compute_forward_kinematics(qi)
            local = self.pm.get_link_pose(self.idx)
            out[i] = self.base_T @ _pose_to_T(np.asarray(local.p), np.asarray(local.q))
        return out

    def commanded(self, action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Commanded hand poses (T, 4, 4): FK of each step's 7 arm targets, fingers as measured."""
        q = np.concatenate([action[:, :7], qpos[:len(action), 7:9]], 1)
        return self(q)


def build_fk(port: str = "A") -> "HandFK":
    """A HandFK from a freshly reset environment (any port and a grasp inside the training range:
    the arm model and the base pose do not depend on them)."""
    from ..env.robot import Robot
    robot = Robot(port)
    robot.reset(0, -45.0, -0.024)
    return HandFK(robot), robot


def _ee_worker(args):
    """One worker: FK of the commanded targets of a list of trajectories (env built once)."""
    items, = args
    fk, robot = build_fk()
    out, worst = 0, 0.0
    for src, dst in items:
        if Path(dst).exists():
            continue
        with np.load(src) as f:
            qpos, action, hand = f["qpos"], f["action"], f["hand"]
        if len(action):
            cmd = fk.commanded(action, qpos)
            # FK of the measured joints must reproduce the recorded hand poses (same FK as recording)
            meas = fk(qpos[:: max(1, len(qpos) // 5)])
            worst = max(worst, float(np.abs(meas - hand[:: max(1, len(qpos) // 5)]).max()))
        else:
            cmd = np.zeros((0, 4, 4))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, hand_cmd=cmd.astype(np.float32))
        out += 1
    robot.env.close()
    return out, worst


def compute_commanded_hands(entries: list[dict], data_root, out_root, workers: int = 6) -> dict:
    """Write <out_root>/<relative trajectory path> with `hand_cmd` (T, 4, 4) for every trajectory.
    Returns counts and the largest |FK(measured) - recorded hand| entry seen (a self-check)."""
    import multiprocessing as mp
    items = [(e["path"], str(Path(out_root) / Path(e["path"]).relative_to(data_root))) for e in entries]
    chunks = [items[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_ee_worker, [(c,) for c in chunks])
    return {"written": sum(r[0] for r in res), "fk_check_max_abs": max(r[1] for r in res)}
