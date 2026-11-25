"""ManiSkill adapter of the online attempt (attempt320.OnlineTester): the behaviour of attempt320 before the AutoMate
port (git c7b37d7), unchanged (tests/test_online_port_maniskill_cpu.py T1 runs both on the same stub robot and agent).

    ManiSkillTester(robot, H_star, axis)   wm.env.robot.Robot; placement (remember_standoff, place) by
                                           wm.collect.branch.Brancher, as before; the attempt loop, the recorder,
                                           snapshot / restore and the Jacobian from wm.online.shared320 / attempt320
    recorder       shared320.FullRecorder with branch.FullRecorder's reads (MANISKILL_IO: joint velocities through
                   env_adapter._np, data.recorder._truth_rel, the wrist camera with its float-depth twin, the crop
                   wm.env.robot.CROP at wm.env.robot.CAM_SCALE)
    left_m         attempt320.LEFT_M (6 cm)
    truth          explore.truth_outcome + demo320.is_success (the one ManiSkill success criterion, at the end of every
                   decision and at the end of the attempt)
Imports ManiSkill, SAPIEN and MPlib (through wm.collect.branch, wm.env.robot, wm.sim.env_adapter).
"""

from __future__ import annotations

import numpy as np

from ..collect.branch import Brancher
from ..collect.explore import truth_outcome
from ..data.recorder import _truth_rel
from ..env import robot as _robot
from ..sim import env_adapter as _ea
from ..tdmpc import demo320 as D
from ..tdmpc.demo320 import is_success
from .attempt320 import LEFT_M, OnlineTester
from .shared320 import RecorderIO


def _qvel(r) -> np.ndarray:
    return _ea._np(r.base.agent.robot.get_qvel())[0]


def _picture(r):
    """branch.FullRecorder._image's capture (the crop of the rendered stream, at the render scale)."""
    cam, comp = r._camera()
    r.base.scene.update_render()
    cam.capture()
    r._twin.take_picture()
    r0, r1, c0, c1 = _robot.CROP
    rgb = np.ascontiguousarray(_ea._np(cam.get_picture("Color"))[0, r0:r1, c0:c1, :3]).astype(np.uint8)
    depth = -np.asarray(r._twin.get_picture("Position"))[r0:r1, c0:c1, 2].astype(np.float64)
    depth[depth < _robot.DEPTH_MIN_M] = 0.0
    d = np.clip(np.round(depth / _robot.DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
    ext = np.eye(4)
    ext[:3, :] = np.asarray(comp.get_extrinsic_matrix())
    K = np.asarray(comp.get_intrinsic_matrix(), dtype=np.float64).copy()
    K[0, 2] -= c0
    K[1, 2] -= r0
    return rgb, d, K, ext


MANISKILL_IO = RecorderIO(qvel=_qvel, truth_rel=_truth_rel, picture=_picture, frame_cls=_robot.Frame,
                          cam_scale=_robot.CAM_SCALE)


class ManiSkillTester(OnlineTester):
    def __init__(self, robot, H_star: np.ndarray, axis: np.ndarray):
        self.placer = Brancher(robot, H_star, axis)            # placement: Explorer.place / Brancher.place, as before
        super().__init__(robot, H_star, axis, fk=self.placer.fk)
        self.io, self.left_m, self.label_meta = MANISKILL_IO, LEFT_M, {}

    # ------------------------------------------------------------ placement (wm.collect.branch / explore, unchanged)
    def remember_standoff(self) -> bool:
        return self.placer.remember_standoff()

    def place(self, start: dict) -> str:
        return self.placer.place(start)

    @property
    def start_error_mm(self) -> float:
        return self.placer.start_error_mm

    # ------------------------------------------------------------ truth (reporting only)
    def truth_step(self, rel7: np.ndarray) -> tuple[bool, float, float]:
        tr = truth_outcome(rel7)
        return bool(is_success(tr)), tr["depth_left_mm"], tr["lateral_mm"]

    def truth_final(self, out: dict) -> dict:
        truth = truth_outcome(out["eval_rel"][-1])
        return {"success": is_success(truth), **{key: float(truth[key]) for key in D.SUCCESS}}


__all__ = ["ManiSkillTester", "MANISKILL_IO"]
