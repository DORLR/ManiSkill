"""Records one trajectory while the expert runs (step callback of the expert)."""

from __future__ import annotations

import numpy as np
from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted import _copy_pose

from ..env.robot import Robot
from . import schema


def _truth_rel(robot: Robot) -> np.ndarray:
    """Charger pose in the socket-goal frame (simulator truth, evaluation only)."""
    rel = _copy_pose(robot.base.goal_pose).inv() * _copy_pose(robot.base.charger.pose)
    return np.r_[np.asarray(rel.p, float), np.asarray(rel.q, float)].astype(np.float32)


class Recorder:
    def __init__(self, robot: Robot, images: bool = True, image_every: int = schema.IMAGE_EVERY):
        self.robot, self.images, self.every = robot, images, image_every

    def begin(self):
        self.t = 0
        self.buf = {k: [] for k in ("qpos", "hand", "eval_rel", "action", "phase",
                                    "rgb", "depth", "seg", "img_step", "cam_T_world")}
        self.K = None
        self._sample(image=self.images)

    def on_step(self, phase: str, action: np.ndarray, info: dict):
        self.buf["action"].append(np.asarray(action, np.float32)[:7])
        self.buf["phase"].append(schema.phase_id(phase))
        self.t += 1
        self._sample(image=self.images and self.t % self.every == 0)

    def end(self) -> dict:
        if self.images and (not self.buf["img_step"] or self.buf["img_step"][-1] != self.t):
            self._image()
        b = self.buf
        out = {
            "qpos": np.asarray(b["qpos"], np.float32),
            "hand": np.asarray(b["hand"], np.float32),
            "eval_rel": np.asarray(b["eval_rel"], np.float32),
            "action": np.asarray(b["action"], np.float32).reshape(-1, 7),
            "phase": np.asarray(b["phase"], np.uint8),
        }
        if self.images:
            out.update(rgb=np.asarray(b["rgb"], np.uint8), depth=np.asarray(b["depth"], np.uint16),
                       seg=np.asarray(b["seg"], np.uint8),
                       img_step=np.asarray(b["img_step"], np.int32),
                       cam_T_world=np.asarray(b["cam_T_world"], np.float32),
                       K=np.asarray(self.K, np.float32))
        return out

    # ------------------------------------------------------------------
    def _sample(self, image: bool):
        self.buf["qpos"].append(self.robot.qpos().astype(np.float32))
        self.buf["hand"].append(self.robot.hand_pose().astype(np.float32))
        self.buf["eval_rel"].append(_truth_rel(self.robot))
        if image:
            self._image()

    def _image(self):
        f = self.robot.frame()
        self.buf["rgb"].append(f.rgb)
        self.buf["depth"].append(f.depth)
        self.buf["seg"].append(f.seg)
        self.buf["img_step"].append(self.t)
        self.buf["cam_T_world"].append(f.cam_T_world.astype(np.float32))
        self.K = f.K
