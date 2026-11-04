"""The only module that talks to Chongyu's WMInsertionChargerRandomized-v2.

The environment code is used unchanged. Two kinds of access are separated:

* method-side (real-robot equivalent): joint angles, hand pose from forward
  kinematics, wrist RGB-D image, camera intrinsics/extrinsics (FK + hand-eye),
  joint-position commands, and the port geometry spec (a CAD model a real
  user also has). These are what `Obs` and `Robot` expose.
* truth-side: simulator object poses and Chongyu's geometry metrics. These
  live in `paf/truth.py` and may only be imported by evaluation code
  (enforced by tests/test_no_truth_leak.py).

The scripted expert is used once per grasp to perform "one successful
insertion", the stand-in for a teleoperated insertion on the real robot. Only
the hand poses it produces (FK) are recorded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/my_nvidia_icd.json")

import gymnasium as gym  # noqa: E402
import mplib  # noqa: E402
import numpy as np  # noqa: E402

import mani_skill.envs  # noqa: E402,F401  registers the task
from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2 import (  # noqa: E402
    ContinuousScriptedPolicyConfig,
    RandomContinuousScriptedPolicy,
)

from .geometry import pose_to_T, T_to_pose, interpolate_T, pose_distance  # noqa: E402

ENV_ID = "WMInsertionChargerRandomized-v2"


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class Obs:
    rgb: np.ndarray          # (H, W, 3) uint8
    depth_mm: np.ndarray     # (H, W) int16, millimetres, 0 = no return
    K: np.ndarray            # (3, 3) intrinsics
    cam_T_world: np.ndarray  # (4, 4) extrinsic (OpenCV convention), world -> camera
    H: np.ndarray            # (4, 4) hand pose from FK
    qpos: np.ndarray         # (9,) measured joint angles


class Robot:
    """Thin wrapper: reset, observe, move the hand, run the expert insertion."""

    def __init__(self, port_type: str, seed: int = 10043):
        self.port_type = port_type
        self.env = gym.make(ENV_ID, port_type=port_type, obs_mode="rgbd", seed=seed)
        self.base = self.env.unwrapped
        self.port_spec = self.base.port_spec

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int, pitch_deg: float, grasp_pos_m: float):
        _, info = self.env.reset(
            seed=int(seed),
            options={"grasp_pitch_deg": float(pitch_deg), "grasp_position_m": float(grasp_pos_m)},
        )
        self.planner = self.base._planner()
        return info

    # ---------------------------------------------------------- proprio/FK
    def qpos(self) -> np.ndarray:
        return _np(self.base.agent.robot.get_qpos())[0].astype(np.float64)

    def hand_pose(self) -> np.ndarray:
        # The simulator's TCP link pose is the forward kinematics of the
        # measured joint angles (rigid articulation); checked against MPlib FK
        # in tests/test_geometry.py.
        pose = self.base.agent.tcp.pose
        return pose_to_T(_np(pose.p)[0], _np(pose.q)[0])

    # ------------------------------------------------------------- observe
    def observe(self) -> Obs:
        obs = self.base.get_obs(self.base.get_info())
        cam = obs["sensor_data"]["hand_camera"]
        par = obs["sensor_param"]["hand_camera"]
        ext = np.eye(4)
        ext[:3, :] = _np(par["extrinsic_cv"])[0]
        return Obs(
            rgb=_np(cam["rgb"])[0],
            depth_mm=_np(cam["depth"])[0, ..., 0],
            K=_np(par["intrinsic_cv"])[0],
            cam_T_world=ext,
            H=self.hand_pose(),
            qpos=self.qpos(),
        )

    # ---------------------------------------------------------------- move
    def _ik(self, T: np.ndarray, seed_q: np.ndarray) -> np.ndarray | None:
        p, q = T_to_pose(T)
        local = self.planner._transform_goal_to_wrt_base(mplib.Pose(p, q))
        sol = None
        # Same threshold as Chongyu's expert first; widen the search only if needed.
        for n_init, thr in ((1, 1e-5), (20, 1e-5), (20, 1e-4)):
            status, sol = self.planner.IK(local, seed_q, n_init_qpos=n_init, threshold=thr)
            if sol is not None:
                break
        if sol is None:
            return None
        sols = np.atleast_2d(np.asarray(sol, dtype=np.float64))
        sol = sols[np.argmin(np.abs(sols[:, :7] - seed_q[:7]).max(axis=1))]  # closest branch
        if np.abs(sol[:7] - seed_q[:7]).max() > 0.25:  # refuse branch jumps
            return None
        return sol

    def _command(self, q7: np.ndarray, n: int = 1):
        action = np.r_[q7[:7], -1.0].astype(np.float32)
        for _ in range(n):
            self.base._step_action(action)  # physics only; render on observe()

    def move_to(self, T_goal: np.ndarray, max_lin: float = 0.002, max_ang: float = np.radians(1.0),
                settle_steps: int = 40, tol_m: float = 1e-4, tol_rad: float = np.radians(0.05)) -> bool:
        """Move the hand to T_goal through small waypoints, then settle.
        Returns False if IK fails (the hand stays where it is)."""
        start = self.hand_pose()
        dt, dr = pose_distance(start, T_goal)
        n = max(1, int(np.ceil(max(dt / max_lin, dr / max_ang))))
        seed_q = self.qpos()
        targets = []
        for k in range(1, n + 1):
            sol = self._ik(interpolate_T(start, T_goal, k / n), seed_q)
            if sol is None:
                return False
            targets.append(sol)
            seed_q = sol
        for sol in targets:
            self._command(sol, 2)
        for _ in range(settle_steps):
            self._command(targets[-1], 1)
            dt, dr = pose_distance(self.hand_pose(), T_goal)
            v = np.abs(_np(self.base.agent.robot.get_qvel())[0][:7]).max()
            if dt < tol_m and dr < tol_rad and v < 2e-3:
                break
        return True

    # ------------------------------------------------- one expert insertion
    def expert_insert(self) -> dict:
        """Chongyu's clean scripted expert. Records only FK hand poses."""
        poses = []

        def on_step(phase, _action, _info):
            poses.append((phase, self.hand_pose()))

        cfg = ContinuousScriptedPolicyConfig(profile="clean")
        # The randomized-board variant is the expert Chongyu's collector uses.
        result = RandomContinuousScriptedPolicy(cfg).run(self.env, on_step=on_step)
        return {"completed": bool(result.get("completed")), "failure": result.get("failure_reason"),
                "steps": int(result.get("steps", 0)), "poses": poses}

    # ------------------------------------------------------- straight push
    def push_along(self, axis: np.ndarray, distance: float, step: float, jam_lag: float,
                   margin: float) -> dict:
        """Insertion test used to define ID: push the hand straight along the
        insertion axis by `distance` (FK only, no simulator truth). Success if
        the hand advanced at least distance - margin. Returns to the start."""
        H0 = self.hand_pose()
        seed_q = self.qpos()
        n = int(np.ceil(distance / step))
        commanded = 0.0
        for k in range(1, n + 1):
            commanded = min(distance, k * step)
            T = H0.copy()
            T[:3, 3] = H0[:3, 3] + commanded * axis
            sol = self._ik(T, seed_q)
            if sol is None:
                break
            seed_q = sol
            self._command(sol, 1)
            advanced = float(axis @ (self.hand_pose()[:3, 3] - H0[:3, 3]))
            if commanded - advanced > jam_lag:
                break
        for _ in range(10):
            self._command(seed_q, 1)
        advanced = float(axis @ (self.hand_pose()[:3, 3] - H0[:3, 3]))
        ok = self.move_to(H0)
        return {"success": bool(advanced >= distance - margin), "advanced_mm": advanced * 1000.0,
                "commanded_mm": commanded * 1000.0, "returned": ok}
