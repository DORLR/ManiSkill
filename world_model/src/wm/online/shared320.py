"""Simulator-free parts of the online attempt (attempt320.OnlineTester) that both simulators need (the user's decision 1
of 2026-11-25: one copy, imported by the ManiSkill adapter wm.online.maniskill320 and the AutoMate adapter
wm.online.automate320). numpy, scipy and wm.sim.geometry only: nothing here imports ManiSkill, SAPIEN, MPlib or Isaac.

Moved here from wm.collect.branch / wm.collect.explore (those files are unchanged: ManiSkill collection scripts keep
importing them; the code below is theirs, line for line, except where named):
    EXECUTE, IMAGE_EVERY, MAX_DECISIONS, STALL_DECISIONS, STALL_M     branch.py constants (same values)
    frame, offset_pose                                                explore.py
    FullRecorder                                                      branch.FullRecorder; the simulator-specific reads
                                                                      (joint velocities, the truth 7-vector, the wrist
                                                                      picture, the Frame class, the render scale) come
                                                                      from a RecorderIO given by the adapter, so the
                                                                      ManiSkill adapter's RecorderIO executes exactly
                                                                      branch.FullRecorder's reads in the same order
    AttemptBase.depth                                                 explore.Explorer.depth
    AttemptBase.snapshot / restore                                    branch.Brancher.snapshot / restore
    AttemptBase.jacobian                                              attempt320.OnlineTester.jacobian (FK central
                                                                      differences on the adapter's HandFK)
AttemptBase holds what the attempt loop reads: robot, H_star (float64), F (columns a, y, z), L (the pin length / the
tight-fit depth, robot.port_spec.pin_length_m), fk (wm.data.kinematics.HandFK of the robot).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..sim.geometry import pose_distance

EXECUTE = 8                   # control steps per decision (0.4 s)                      (branch.EXECUTE)
IMAGE_EVERY = 4               # 5 Hz images                                             (branch.IMAGE_EVERY)
MAX_DECISIONS = 150           # 60 s, the closed-loop protocol (ManiSkill)              (branch.MAX_DECISIONS)
STALL_DECISIONS, STALL_M = 8, 0.0002                                                    # (branch.STALL_*)


def frame(axis: np.ndarray) -> np.ndarray:
    """Columns: a (insertion axis), y, z (up, perpendicular to a). (explore.frame)"""
    a = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0]) - a[2] * a
    z /= np.linalg.norm(z)
    return np.stack([a, np.cross(z, a), z], 1)


def offset_pose(H_star: np.ndarray, F: np.ndarray, depth_m: float, lateral_m, rotvec_deg) -> np.ndarray:
    """H* moved by `depth_m` along a (negative = back), `lateral_m` (y, z) across it, and rotated about the
    hand by `rotvec_deg` (components along a, y, z). (explore.offset_pose)"""
    T = H_star.copy()
    T[:3, 3] = H_star[:3, 3] + F @ np.array([depth_m, lateral_m[0], lateral_m[1]])
    T[:3, :3] = Rotation.from_rotvec(F @ np.radians(rotvec_deg)).as_matrix() @ H_star[:3, :3]
    return T


@dataclass(frozen=True)
class RecorderIO:
    """The simulator-specific reads of FullRecorder (given by the adapter):
        qvel(robot)       -> (9,) joint velocities (any float dtype; stored as float32)
        truth_rel(robot)  -> (7,) float32 plug in the goal frame (simulator truth; evaluation only)
        picture(robot)    -> (rgb (h, w, 3) uint8, depth (h, w) uint16 [depth_unit], K (3, 3) of the crop,
                              cam_T_world (4, 4)) of the stored picture (the crop at the render scale)
        frame_cls         the simulator's Frame dataclass (rgb, depth, seg, K, cam_T_world, qpos, hand)
        cam_scale         render scale (1 = the native stream; the policy input is the picture reduced by it)"""
    qvel: Callable
    truth_rel: Callable
    picture: Callable
    frame_cls: type
    cam_scale: int = 1


class FullRecorder:
    """Per control step: joints, joint velocities, FK hand, truth (eval only), executed and policy targets.
    Every IMAGE_EVERY steps: the full wrist frame. `frame` is the 320 crop of the latest picture (the policy's input).
    (branch.FullRecorder with the simulator reads of `io`.)"""

    def __init__(self, robot, io: RecorderIO):
        self.robot, self.io = robot, io

    def begin(self):
        self.t = 0
        self.buf = {k: [] for k in ("qpos", "qvel", "hand", "eval_rel", "action", "action_policy",
                                    "rgb", "depth", "img_step", "cam_T_world")}
        self._sample()
        self._image()

    def step(self, q_exec: np.ndarray, q_policy: np.ndarray):
        self.buf["action"].append(np.asarray(q_exec, np.float32)[:7])
        self.buf["action_policy"].append(np.asarray(q_policy, np.float32)[:7])
        self.t += 1
        self._sample()
        if self.t % IMAGE_EVERY == 0:
            self._image()

    def end(self) -> dict:
        if self.buf["img_step"][-1] != self.t:
            self._image()
        b = self.buf
        return {"qpos": np.asarray(b["qpos"], np.float32), "qvel": np.asarray(b["qvel"], np.float32),
                "hand": np.asarray(b["hand"], np.float32), "eval_rel": np.asarray(b["eval_rel"], np.float32),
                "action": np.asarray(b["action"], np.float32).reshape(-1, 7),
                "action_policy": np.asarray(b["action_policy"], np.float32).reshape(-1, 7),
                "rgb": np.asarray(b["rgb"], np.uint8), "depth": np.asarray(b["depth"], np.uint16),
                "img_step": np.asarray(b["img_step"], np.int32),
                "cam_T_world": np.asarray(b["cam_T_world"], np.float32), "K": np.asarray(self.K, np.float32)}

    def _sample(self):
        r = self.robot
        self.buf["qpos"].append(r.qpos().astype(np.float32))
        self.buf["qvel"].append(np.asarray(self.io.qvel(r)).astype(np.float32))
        self.buf["hand"].append(r.hand_pose().astype(np.float32))
        self.buf["eval_rel"].append(self.io.truth_rel(r))

    def _image(self):
        """The stored picture (io.picture: the crop of the rendered stream at the render scale); the policy gets the
        same picture reduced to its 320 x 320 input."""
        r, s = self.robot, self.io.cam_scale
        rgb, d, K, ext = self.io.picture(r)
        self.K = K
        self.buf["rgb"].append(rgb)
        self.buf["depth"].append(d)
        self.buf["img_step"].append(self.t)
        self.buf["cam_T_world"].append(ext.astype(np.float32))
        if s == 1:
            rgb_p, d_p, Kp = rgb, d, K
        else:
            h, w = rgb.shape[0] // s, rgb.shape[1] // s
            rgb_p = np.round(rgb.reshape(h, s, w, s, 3).astype(np.float32).mean((1, 3))).astype(np.uint8)
            d_p = np.ascontiguousarray(d[::s, ::s])
            Kp = K.copy()
            Kp[:2] /= s
        self.frame = self.io.frame_cls(rgb=rgb_p, depth=d_p, seg=np.zeros(d_p.shape, np.uint8), K=Kp, cam_T_world=ext,
                                       qpos=r.qpos(), hand=r.hand_pose())
        self.frame.full = (rgb, d)          # the stored full picture, for a policy that reads it (wm/tdmpc/policy_resnet.py)


class AttemptBase:
    """What the attempt loop reads of the grasp and the robot. F: the adapter's frame of the axis (default frame(axis),
    as explore.Explorer); only F[:, 0] (the axis) enters the attempt's motion (face gap, exploration shift) and its
    reports. fk: wm.data.kinematics.HandFK of the robot (built by the adapter, once per robot)."""

    def __init__(self, robot, H_star: np.ndarray, axis: np.ndarray, fk, F: np.ndarray | None = None):
        self.robot, self.H_star = robot, np.asarray(H_star, np.float64)
        self.F = frame(np.asarray(axis, np.float64)) if F is None else np.asarray(F, np.float64)
        self.L = float(robot.port_spec.pin_length_m)
        self.fk = fk

    def depth(self, T: np.ndarray) -> float:
        """Hand position along a relative to inserted (negative = in front). (explore.Explorer.depth)"""
        return float(self.F[:, 0] @ (T[:3, 3] - self.H_star[:3, 3]))

    # ------------------------------------------------------------ same start (branch.Brancher)
    def snapshot(self):
        self._snap = self.robot.base.get_state_dict()
        self._snap_hand = self.robot.hand_pose()

    def restore(self) -> list[float]:
        """Back to the saved start. Returns the deviation of the hand from the saved one (mm, deg)."""
        self.robot.base.set_state_dict(self._snap)
        dt, dr = pose_distance(self.robot.hand_pose(), self._snap_hand)
        return [float(dt * 1000), float(np.degrees(dr))]

    # ------------------------------------------------------------ joint offsets that move the hand
    def jacobian(self, q9: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """(6, 7): hand position [m] and rotation vector [rad] (world frame) per joint [rad], by central differences of FK."""
        J = np.zeros((6, 7))
        for i in range(7):
            qp, qm = q9.copy(), q9.copy()
            qp[i] += eps
            qm[i] -= eps
            Tp, Tm = self.fk(np.stack([qp, qm]))
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            J[3:, i] = Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
        return J


__all__ = ["EXECUTE", "IMAGE_EVERY", "MAX_DECISIONS", "STALL_DECISIONS", "STALL_M", "frame", "offset_pose",
           "RecorderIO", "FullRecorder", "AttemptBase"]
