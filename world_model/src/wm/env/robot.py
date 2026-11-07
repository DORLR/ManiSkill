"""Robot access for data collection.

Builds on alignment_field/paf/env_adapter.py: Chongyu's task used unchanged, the
wrist camera set to the D405 default stream by a runtime override, generated
port shapes registered in memory. Differences, all at runtime (no file edited):

* `obs_mode="state"`: the task no longer renders the camera at every control
  step; images are taken only when the recorder asks (`frame()`).
* The episode time limit is raised (gym.make argument), because one episode
  holds the reference insertion and every recorded trajectory of a grasp.
* Sub-millimetre depth: ManiSkill's camera pipeline ("minimal" shader) stores
  positions as int16 millimetres (measured: mean error 0.49 mm, max 1.0 mm).
  A twin camera with SAPIEN's "default" shader is mounted on the same link with
  the same pose and intrinsics (checked equal) and gives float positions; RGB
  stays from the original camera, so colours are unchanged. Depth is stored in
  the D405 depth unit, 0.1 mm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .._paths import use_alignment_field

use_alignment_field()

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
from paf import config as _C  # noqa: E402
from paf import env_adapter as _ea  # noqa: E402

MAX_EPISODE_STEPS = 1_000_000
DEPTH_UNIT_M = 1e-4               # stored depth unit: 0.1 mm (D405 default)
CROP = _C.REL_CROP                # rows r0:r1, cols c0:c1 -> 320 x 320 around pins and holes
DEPTH_MIN_M = _C.DEPTH_MIN_MM / 1000.0
_FLOAT_SHADER = os.path.join(os.path.dirname(sapien.__file__), "vulkan_shader", "default")


SEG_CLASSES = ("other", "plug", "robot", "board", "table")  # uint8 class ids 0..4


@dataclass
class Frame:
    """One wrist-camera observation, cropped. Everything here exists on a real robot
    (the segmentation as an annotation or an off-the-shelf segmenter's output)."""
    rgb: np.ndarray          # (320, 320, 3) uint8
    depth: np.ndarray        # (320, 320) uint16, units of DEPTH_UNIT_M, 0 = no return
    seg: np.ndarray          # (320, 320) uint8, index into SEG_CLASSES
    K: np.ndarray            # (3, 3) intrinsics of the crop
    cam_T_world: np.ndarray  # (4, 4) world -> camera (OpenCV), from FK and the hand-eye mount
    qpos: np.ndarray         # (9,) measured joint angles
    hand: np.ndarray         # (4, 4) hand (TCP) pose from FK


class Robot(_ea.Robot):
    """paf Robot without per-step rendering, with a long episode limit and float depth.

    Use `frame()` for images; the parent's `observe()` is not available in state mode.
    """

    def __init__(self, port_type: str, seed: int = 10043):  # parent init not called on purpose
        self.port_type = port_type
        self.env = gym.make(_ea.ENV_ID, port_type=port_type, obs_mode="state", seed=seed,
                            max_episode_steps=MAX_EPISODE_STEPS)
        self.base = self.env.unwrapped
        self.port_spec = self.base.port_spec
        self._twin = None

    def reset(self, seed: int, pitch_deg: float, grasp_pos_m: float):
        info = super().reset(seed, pitch_deg, grasp_pos_m)
        self._attach_twin()
        self._seg_lut = self._segmentation_lut()
        return info

    def _segmentation_lut(self) -> np.ndarray:
        """Per-scene segmentation id -> class id (SEG_CLASSES)."""
        lut = np.zeros(1 << 16, np.uint8)
        for link in self.base.agent.robot.get_links():
            cls = 1 if link.name == "wm_charger" else 2
            for o in link._objs:
                lut[int(o.entity.per_scene_id)] = cls
        for actor in self.base.scene.get_all_actors():
            name = actor.name.lower()
            cls = 3 if ("receptacle" in name or "socket" in name or "board" in name) else \
                4 if ("table" in name or "ground" in name) else 0
            for i in np.atleast_1d(np.asarray(actor.per_scene_id)):
                lut[int(i)] = cls
        return lut

    # ------------------------------------------------------------------ camera
    def _camera(self):
        cam = self.base._sensors["hand_camera"]
        return cam, cam.camera._render_cameras[0]

    def _attach_twin(self):
        """(Re)mount the float-depth twin on the wrist camera's link (scene may be rebuilt on reset)."""
        _, comp = self._camera()
        if self._twin is not None and self._twin.entity is comp.entity:
            return
        twin = sapien.render.RenderCameraComponent(comp.width, comp.height, _FLOAT_SHADER)
        twin.set_local_pose(comp.local_pose)
        K = comp.get_intrinsic_matrix()
        twin.set_perspective_parameters(comp.near, comp.far, K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0)
        comp.entity.add_component(twin)
        self._twin = twin

    def frame(self) -> Frame:
        cam, comp = self._camera()
        self.base.scene.update_render()
        cam.capture()
        self._twin.take_picture()
        rgb = _ea._np(cam.get_picture("Color"))[0, ..., :3]
        seg_id = _ea._np(cam.get_picture("PositionSegmentation"))[0, ..., 3].astype(np.int64) & 0xFFFF
        depth = -np.asarray(self._twin.get_picture("Position"))[..., 2].astype(np.float64)
        depth[depth < DEPTH_MIN_M] = 0.0            # D405 minimum range; also no return
        ext = np.eye(4)
        ext[:3, :] = np.asarray(comp.get_extrinsic_matrix())
        r0, r1, c0, c1 = CROP
        K = np.asarray(comp.get_intrinsic_matrix(), dtype=np.float64).copy()
        K[0, 2] -= c0
        K[1, 2] -= r0
        d = np.clip(np.round(depth[r0:r1, c0:c1] / DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
        return Frame(rgb=np.ascontiguousarray(rgb[r0:r1, c0:c1]).astype(np.uint8), depth=d,
                     seg=self._seg_lut[seg_id[r0:r1, c0:c1]], K=K,
                     cam_T_world=ext, qpos=self.qpos(), hand=self.hand_pose())
