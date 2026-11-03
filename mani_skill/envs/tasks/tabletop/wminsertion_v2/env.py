"""Preheld charger insertion with fixed grasp and wrist RGB/depth.

Built on the original ManiSkill 3.0.0b4 PlugCharger-v1 task. Camera mount,
native controls, pin dimensions and 0.5 mm per-side clearance are retained.
"""

from __future__ import annotations
import copy
import math
from importlib.metadata import version
from typing import Any
import mplib
import numpy as np
import sapien
import torch
import trimesh
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.envs.utils import randomization
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from .common import _sp, _raw, _pitch, _pose_error, _native_rgb256_config
from .geometry import (
    ACTIVE_GRASP_PITCH_RANGE_DEG,
    HOLE_OFFSET_LIMIT_M,
    RESET_IK_SEED,
    RNG_NAMESPACE,
    sample_pitch,
    sample_hole_offset,
    validate_hole_offset,
    validate_pitch_range,
    partition_receptacle,
)
from .robot import CHARGER_LINK_NAME, ROBOT_UID, FixedLinkGrasp

ENV_ID = "WMInsertionChargerRandomized-v2"
FIXED_LINK_ENV_ID = ENV_ID
FUTURE_GRASP_PITCH_RANGE_DEG = (-60.0, -25.0)
GRIP_CENTER_IN_CHARGER_M = (-0.020, 0.0, 0.0015)
FRONT_FIXED_SOCKET_POSE = (
    0.055732615292072296,
    -0.08228035271167755,
    0.10000000149011612,
    0.06713979691267014,
    0.0,
    0.0,
    -0.997743546962738,
)
FRONT_FIXED_PIN_LENGTH_M = 0.016
FRONT_FIXED_TIP_GAP_M = 0.012
FRONT_FIXED_TABLETOP_Z_M = 0.012


def _front_fixed_charger_pose() -> list[float]:
    """Place horizontal pins in front of the fixed socket, still on the table.

    G is the native insertion goal (socket * Rz(pi)); its x axis points into
    the socket. The pin root is 28mm behind G in the tabletop XY projection,
    giving the 16mm pin tips a 12mm projected wall gap and zero lateral offset.
    Table height is independent of the elevated socket insertion height.
    """
    goal = _sp(FRONT_FIXED_SOCKET_POSE) * sapien.Pose(q=euler2quat(0, 0, np.pi))
    front = goal * sapien.Pose(
        [-(FRONT_FIXED_PIN_LENGTH_M + FRONT_FIXED_TIP_GAP_M), 0.0, 0.0]
    )
    position = np.asarray(front.p, dtype=np.float64).copy()
    position[2] = FRONT_FIXED_TABLETOP_Z_M
    return _raw(sapien.Pose(position, goal.q))


class _FixedSeedIKPlanner:
    """Delegate unchanged planning, with a reset seed immediately inside IK.

    The archived parent sets the MPlib seed *after* calling ``_planner``.
    Seeding here preserves that parent's code while making the actual IK
    search independent of the episode and both requested scene factors.
    """

    def __init__(self, planner, seed: int):
        self._planner = planner
        self._seed = seed

    def __getattr__(self, name):
        return getattr(self._planner, name)

    def IK(self, *args, **kwargs):
        if not hasattr(mplib, "set_global_seed"):
            raise RuntimeError(
                "This variant requires MPlib set_global_seed for deterministic reset IK"
            )
        mplib.set_global_seed(self._seed)
        return self._planner.IK(*args, **kwargs)


@register_env(ENV_ID, max_episode_steps=800)
class WMInsertionChargerEnvV2(PlugChargerEnv):
    """One CPU-physics scene; t=0 is preheld on the table before lift.

    Default RGB-D is native millimetre depth plus RGB from hand_camera.
    Use obs_mode="rgb" for RGB only. Calibration overrides belong in reset
    options: grasp_pitch_deg and hole_offset_yz_m.
    """

    SUPPORTED_ROBOTS = [ROBOT_UID]

    def __init__(
        self,
        *,
        pitch_range_deg=ACTIVE_GRASP_PITCH_RANGE_DEG,
        hole_offset_enabled=True,
        reset_ik_seed=RESET_IK_SEED,
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        if kwargs.get("num_envs", 1) != 1 or kwargs.get("sim_backend", "cpu") not in (
            "cpu",
            "auto",
        ):
            raise ValueError("WMInsertion-v2 requires one CPU physics environment")
        if kwargs.get("robot_uids", ROBOT_UID) != ROBOT_UID:
            raise ValueError("WMInsertion-v2 uses its fixed-child Panda")
        if kwargs.get("reconfiguration_freq", 1) != 1:
            raise ValueError("Translated hole geometry requires reconfiguration_freq=1")
        if (
            float(robot_init_qpos_noise) != 0.0
            or isinstance(reset_ik_seed, bool)
            or reset_ik_seed != RESET_IK_SEED
        ):
            raise ValueError("Use robot_init_qpos_noise=0 and reset_ik_seed=9101")
        if not isinstance(hole_offset_enabled, bool):
            raise ValueError("hole_offset_enabled must be bool")
        self.pitch_range_deg = validate_pitch_range(pitch_range_deg)
        self.hole_offset_enabled = hole_offset_enabled
        self.reset_ik_seed = RESET_IK_SEED
        self.native_package_versions = {p: version(p) for p in ("mani_skill", "sapien")}
        if self.native_package_versions != {
            "mani_skill": "3.0.0b4",
            "sapien": "3.0.0b1",
        }:
            raise RuntimeError(
                f"Expected ManiSkill3.0.0b4/SAPIEN3.0.0b1; found {self.native_package_versions}"
            )
        self.spawn_layout, self.socket_visuals, self.wrist_only = (
            "front_fixed",
            "stable",
            True,
        )
        self.fixed_socket_pose = _raw(FRONT_FIXED_SOCKET_POSE)
        self.fixed_charger_pose = _front_fixed_charger_pose()
        self.legacy_fixed_charger_pose = [
            -0.049560967832803726,
            0.1660775989294052,
            0.012000000104308128,
            0.9964408278465271,
            0.0,
            0.0,
            -0.08429525792598724,
        ]
        self._attachment = self._episode_record = None
        self._zero_held_pose = self._initial_held_pose = None
        self._ik_planner = None
        self._hole_offset_yz_m = (0.0, 0.0)
        self._hole_offset_overridden = False
        self._socket_partition = ()
        kwargs.update(
            num_envs=1, sim_backend="cpu", robot_uids=ROBOT_UID, reconfiguration_freq=1
        )
        kwargs.setdefault("obs_mode", "rgbd")
        kwargs.setdefault("control_mode", "pd_joint_pos")
        kwargs["sensor_configs"] = _native_rgb256_config(kwargs.get("sensor_configs"))
        super().__init__(robot_init_qpos_noise=0.0, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []  # The mounted hand_camera is supplied by the robot.

    def _release_attachment(self):
        attachment = self._attachment
        if attachment is not None:
            attachment.release()
            self._attachment = None

    def reset(self, seed=None, options=None):
        self._release_attachment()
        self._episode_record = self._zero_held_pose = self._initial_held_pose = None
        self._ik_planner = None
        options = dict(options or {})
        if "grasp_pitch_deg" in options:
            self._validate_pitch_override(options["grasp_pitch_deg"])
        if "hole_offset_yz_m" in options:
            validate_hole_offset(options["hole_offset_yz_m"])
        options["reconfigure"] = True
        try:
            return super().reset(seed=seed, options=options)
        except BaseException:
            self._release_attachment()
            raise

    def _load_scene(self, options):
        self._hole_offset_overridden = "hole_offset_yz_m" in options
        self._hole_offset_yz_m = (
            validate_hole_offset(options["hole_offset_yz_m"])
            if self._hole_offset_overridden
            else sample_hole_offset(int(self._episode_seed), self.hole_offset_enabled)
        )
        self.scene_builder = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.scene_builder.build()
        self.charger = self.agent.robot.links_map[CHARGER_LINK_NAME]
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        boxes = partition_receptacle(
            peg_size, receptacle_size, gap, self._hole_offset_yz_m
        )
        self._socket_partition = boxes
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial()
        material.set_base_color([1, 1, 1, 1])
        material.metallic = 0.0
        material.roughness = 0.1
        for box in boxes:
            pose = sapien.Pose(box.center_m)
            half_size = list(box.half_size_m)
            # Rendered openings and physical openings share the exact boxes.
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=material)
        return builder.build_kinematic(name="receptacle")

    def _planner(self):
        if self._ik_planner is None:
            # The source URDF has no wm_charger name; its unchanged native link
            # list is required by MPlib.  The fixture has no extra active joint.
            self._ik_planner = mplib.Planner(
                urdf=self.agent.urdf_path,
                srdf=self.agent.urdf_path.replace(".urdf", ".srdf"),
                user_link_names=self.agent.native_ik_link_names,
                user_joint_names=[
                    joint.get_name() for joint in self.agent.robot.get_active_joints()
                ],
                move_group="panda_hand_tcp",
                joint_vel_limits=np.ones(7) * 0.5,
                joint_acc_limits=np.ones(7) * 0.5,
            )
        base = _sp(self.agent.robot.pose)
        self._ik_planner.set_base_pose(mplib.Pose(base.p, base.q))
        return _FixedSeedIKPlanner(self._ik_planner, self.reset_ik_seed)

    def _validate_pitch_override(self, value) -> float:
        value = float(value)
        low, high = self.pitch_range_deg
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"grasp_pitch_deg must lie in active range [{low},{high}]")
        return value

    def _sample_pitch(self, options: dict) -> float:
        if "grasp_pitch_deg" in options:
            return self._validate_pitch_override(options["grasp_pitch_deg"])
        return sample_pitch(int(self._episode_seed), self.pitch_range_deg)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.gpu_sim_enabled:
            raise RuntimeError(
                "Fixed-link charger reset currently requires CPU physics"
            )
        # Retain native table/robot initialization and its episode-RNG draws.
        # The table helper recognizes only the public Panda uid, not this child
        # class uid; restore the real registration immediately afterwards.
        uid = self.robot_uids
        try:
            self.robot_uids = "panda_wristcam"
            self.scene_builder.initialize(env_idx)
        finally:
            self.robot_uids = uid
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0.0,
                    -np.pi * 5 / 8,
                    0.0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0.0, 0.0]))
            # Preserve native RNG consumption without ever setting a link pose.
            xy = randomization.uniform(
                [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
            )
            discarded_native_plug = Pose.create_from_pq(pos, ori)
            xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.1
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            )
            native_sampled_socket = Pose.create_from_pq(pos, ori)
            # Consume every native draw in both layouts, then override only
            # the reset pose. Legacy initialization is otherwise unchanged.
            self.receptacle.set_pose(
                _sp(self.fixed_socket_pose)
                if self.spawn_layout == "front_fixed"
                else native_sampled_socket
            )
            self.goal_pose = self.receptacle.pose * sapien.Pose(
                q=euler2quat(0, 0, np.pi)
            )

        plug = _sp(self.fixed_charger_pose)
        obb = trimesh.primitives.Box(
            extents=np.asarray(self._base_size) * 2,
            transform=(
                plug * sapien.Pose([-0.020, 0.0, 0.0])
            ).to_transformation_matrix(),
        )
        approaching = np.array([0.0, 0.0, -1.0])
        grasp = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=_sp(self.agent.tcp.pose).to_transformation_matrix()[:3, 1],
            depth=0.025,
        )
        if (
            np.linalg.norm(grasp["center"] - (plug * sapien.Pose([-0.020, 0.0, 0.0])).p)
            > 1e-6
        ):
            raise RuntimeError("Native grasp helper no longer selects housing middle")
        center = (plug * sapien.Pose(GRIP_CENTER_IN_CHARGER_M)).p
        canonical = _sp(
            self.agent.build_grasp_pose(approaching, grasp["closing"], center)
        )
        zero_held = canonical.inv() * plug
        requested = self._sample_pitch(options)
        candidates = []
        for command in (requested, -requested):
            tcp = canonical * sapien.Pose(q=euler2quat(0, np.radians(command), 0))
            held = tcp.inv() * plug
            candidates.append(
                (
                    abs(_pitch(zero_held, held)["pitch_deg"] - requested),
                    command,
                    tcp,
                    held,
                )
            )
        error, command, desired_tcp, desired_held = min(
            candidates, key=lambda row: row[0]
        )
        if error > 1e-3:
            raise RuntimeError("Canonical grasp does not realize requested pitch")
        planner = self._planner()
        if hasattr(mplib, "set_global_seed"):
            mplib.set_global_seed(int(self._episode_seed))
        q_start = (
            np.asarray(common.to_numpy(self.agent.robot.get_qpos())).reshape(-1).copy()
        )
        local = planner._transform_goal_to_wrt_base(
            mplib.Pose(desired_tcp.p, desired_tcp.q)
        )
        status, qpos = planner.IK(
            local, q_start, n_init_qpos=20, threshold=1e-4, return_closest=True
        )
        if qpos is None:
            raise RuntimeError(f"Tabletop held reset has no IK solution: {status}")
        qpos = np.asarray(qpos, dtype=np.float32).copy()
        qpos[-2:] = self._base_size[1]
        q = torch.as_tensor(qpos[None, :], dtype=torch.float32, device=self.device)
        self.agent.robot.set_qpos(q)
        actual_tcp = _sp(self.agent.tcp.pose)
        tcp_error = _pose_error(actual_tcp, desired_tcp)
        if tcp_error["translation_mm"] > 0.2 or tcp_error["rotation_deg"] > 0.2:
            raise RuntimeError(f"Held reset IK misses target: {tcp_error}")
        # Anchor changes occur only during reset, outside any physics step.
        # Capture the achieved FK hand transform so charger stays at exact P0.
        hand = _sp(self.agent.robot.links_map["panda_hand"].pose)
        joint = self.charger._objs[0].joint
        joint.pose_in_child = sapien.Pose()
        joint.pose_in_parent = hand.inv() * plug
        self.agent.robot.set_qpos(q)  # CPU articulation FK/cache synchronization.
        self.agent.robot.set_qvel(torch.zeros_like(q))
        self.agent.robot.links_map[CHARGER_LINK_NAME].disable_gravity = False
        self.agent.controller.reset()
        spawn_error = _pose_error(_sp(self.charger.pose), plug)
        if spawn_error["translation_mm"] > 0.01 or spawn_error["rotation_deg"] > 0.01:
            raise RuntimeError(
                f"Fixed child FK does not retain tabletop charger pose: {spawn_error}"
            )
        held = actual_tcp.inv() * _sp(self.charger.pose)
        achieved = _pitch(zero_held, held)["pitch_deg"]
        if abs(achieved - requested) > 0.2:
            raise RuntimeError("Achieved fixed-link reset pitch misses requested angle")
        self._zero_held_pose, self._initial_held_pose = zero_held, held
        pin_length = float(self._peg_size[0]) * 2.0
        pin_tip_in_goal = _sp(self.goal_pose).inv() * (
            _sp(self.charger.pose) * sapien.Pose([pin_length, 0.0, 0.0])
        )
        self._episode_record = {
            "env_id": FIXED_LINK_ENV_ID,
            "native_package_versions": dict(self.native_package_versions),
            "episode_seed": int(self._episode_seed),
            "initialization": "fixed_tabletop_charger_preheld_housing_middle",
            "physical_pickup_executed": False,
            "reset_physics_steps": 0,
            "grasp_pitch_sampling": "continuous_uniform",
            "requested_grasp_pitch_deg": requested,
            "local_tcp_y_command_deg": command,
            "calibration_angle_override": "grasp_pitch_deg" in options,
            "active_grasp_pitch_range_deg": list(ACTIVE_GRASP_PITCH_RANGE_DEG),
            "future_candidate_grasp_pitch_range_deg": list(
                FUTURE_GRASP_PITCH_RANGE_DEG
            ),
            "future_candidate_range_active": False,
            "grip_center_in_charger_m": list(GRIP_CENTER_IN_CHARGER_M),
            "grip_longitudinal_location": "housing_middle",
            "grip_vertical_offset_mm": 1.5,
            "spawn_layout": self.spawn_layout,
            "fixed_charger_pose": list(self.fixed_charger_pose),
            "charger_spawn_position_randomized": False,
            "legacy_fixed_charger_pose": list(self.legacy_fixed_charger_pose),
            "fixed_socket_pose": self.fixed_socket_pose,
            "fixed_native_goal_pose": (
                _raw(self.goal_pose) if self.spawn_layout == "front_fixed" else None
            ),
            "charger_spawn_yaw_randomized": False,
            "socket_spawn": (
                "fixed_reference"
                if self.spawn_layout == "front_fixed"
                else "native_randomized_each_episode"
            ),
            "socket_spawn_reference_seed": (
                910200 if self.spawn_layout == "front_fixed" else None
            ),
            "native_sampled_receptacle_pose_before_layout_override": _raw(
                native_sampled_socket
            ),
            "initial_pin_tip_projected_gap_mm": -float(pin_tip_in_goal.p[0]) * 1000.0,
            "initial_pin_tip_projected_lateral_mm": float(pin_tip_in_goal.p[1])
            * 1000.0,
            "initial_pin_tip_projected_vertical_mm": float(pin_tip_in_goal.p[2])
            * 1000.0,
            "front_fixed_requested_tip_gap_mm": (
                FRONT_FIXED_TIP_GAP_M * 1000.0
                if self.spawn_layout == "front_fixed"
                else None
            ),
            "physical_pin_length_m": pin_length,
            "spawn_layout_updates": "reset_initialization_only_before_physics",
            "initial_receptacle_pose": _raw(self.receptacle.pose),
            "initial_native_goal_pose": _raw(self.goal_pose),
            "native_sampled_charger_pose_before_fixed_override": _raw(
                discarded_native_plug
            ),
            "canonical_tcp_pose": _raw(canonical),
            "zero_held_pose": _raw(zero_held),
            "requested_held_pose": _raw(desired_held),
            "initial_achieved_held_pose": _raw(held),
            "initial_tcp_pose": _raw(actual_tcp),
            "initial_charger_pose": _raw(self.charger.pose),
            "initial_qpos": qpos.astype(float).tolist(),
            "ik_status": str(status),
            "ik_tcp_error": tcp_error,
            "initial_achieved_grasp_pitch_deg": achieved,
            "fixed_child_spawn_error": spawn_error,
            "camera_mount_and_fov": "native_unmodified",
            "camera_resolution": [256, 256],
            "observation_camera_mode": (
                "wrist_only" if self.wrist_only else "native_base_and_wrist"
            ),
            "geometry_socket_collision": "native_unmodified",
            "socket_visuals": self.socket_visuals,
            "socket_coplanar_gold_decorations_removed": self.socket_visuals == "stable",
            "solver_position_iteration_override": "robot_articulation_minimum128_for_finger_charger_contact",
            "controller_reset_at_initialized_qpos": True,
            "charger_representation": "articulation_fixed_child_link",
            "independent_charger_actor_created": False,
            "original_URDF_modified": False,
            "joint_anchor_updates": "reset_only_before_physics",
            "charger_gravity_enabled": True,
        }
        self._attachment = FixedLinkGrasp(self).attach()
        dy, dz = self._hole_offset_yz_m
        self.goal_pose = (
            self.receptacle.pose
            * sapien.Pose([0.0, dy, dz])
            * sapien.Pose(q=euler2quat(0.0, 0.0, np.pi))
        )
        pin_length = float(self._peg_size[0]) * 2.0
        pin_tip_in_goal = _sp(self.goal_pose).inv() * (
            _sp(self.charger.pose) * sapien.Pose([pin_length, 0.0, 0.0])
        )
        gap, clearance = float(self._peg_gap), float(self._clearance)
        self._episode_record.update(
            {
                "env_id": ENV_ID,
                "variant": "fixed_outer_board_random_grasp_and_aperture_pair",
                "active_grasp_pitch_range_deg": list(self.pitch_range_deg),
                "authorized_grasp_pitch_range_deg": list(ACTIVE_GRASP_PITCH_RANGE_DEG),
                "grasp_pitch_sampling": "continuous_uniform_dedicated_episode_rng",
                "hole_offset_enabled": self.hole_offset_enabled,
                "calibration_hole_offset_override": self._hole_offset_overridden,
                "hole_offset_yz_m": [dy, dz],
                "hole_offset_bounds_board_yz_m": [
                    [-HOLE_OFFSET_LIMIT_M, HOLE_OFFSET_LIMIT_M]
                ]
                * 2,
                "hole_offset_sampling": "independent_uniform_y_z_dedicated_episode_rng",
                "randomization_rng_namespace": RNG_NAMESPACE,
                "randomization_factors_use_independent_rng_streams": True,
                "hole_pair_center_board_m": [0.0, dy, dz],
                "hole_centers_board_m": [[0.0, dy - gap, dz], [0.0, dy + gap, dz]],
                "hole_center_spacing_m": 2 * gap,
                "hole_half_size_yz_m": [
                    float(self._peg_size[1]) + clearance,
                    float(self._peg_size[2]) + clearance,
                ],
                "hole_single_side_clearance_m": clearance,
                "board_outer_half_size_m": [float(v) for v in self._receptacle_size],
                "board_actor_pose_fixed_across_episodes": True,
                "board_outer_boundary_fixed": True,
                "whole_board_translation_applied": False,
                "insertion_axis_board_m": [-1.0, 0.0, 0.0],
                "aperture_insertion_axis_position_offset_m": 0.0,
                "socket_collision_partition": [
                    box.as_dict() for box in self._socket_partition
                ],
                "geometry_socket_collision": "native_fixed_exterior_five_box_shifted_slot_partition",
                "socket_visuals": "stable",
                "socket_visual_boxes_match_collision_boxes": True,
                "socket_coplanar_gold_decorations_removed": True,
                "robot_init_qpos_noise": 0.0,
                "reset_ik_seed": self.reset_ik_seed,
                "reset_ik_seed_applied_immediately_inside_ik": True,
                "scene_reconfiguration_frequency": 1,
                "scene_reconfigured_for_aperture_geometry": True,
                "native_goal_follows_aperture_pair": True,
                "native_goal_pose_fixed_across_episodes": not (
                    self.hole_offset_enabled or self._hole_offset_overridden
                ),
                "native_goal_construction": "board_pose * Pose([0,dy,dz]) * Rz(pi)",
                "fixed_native_goal_pose": _raw(self.goal_pose),
                "initial_native_goal_pose": _raw(self.goal_pose),
                "initial_pin_tip_projected_gap_mm": -float(pin_tip_in_goal.p[0])
                * 1000.0,
                "initial_pin_tip_projected_lateral_mm": float(pin_tip_in_goal.p[1])
                * 1000.0,
                "initial_pin_tip_projected_vertical_mm": float(pin_tip_in_goal.p[2])
                * 1000.0,
            }
        )
        # The archived exploration range was never sampled and is no longer
        # the active contract for this new variant.
        self._episode_record.pop("future_candidate_grasp_pitch_range_deg", None)
        self._episode_record.pop("future_candidate_range_active", None)

    def audit(self) -> dict[str, Any]:
        """Read achieved H and constraint drift without changing the scene."""
        if self._episode_record is None:
            return {"env_id": ENV_ID, "initialized": False}
        report = copy.deepcopy(self._episode_record)
        tcp = _sp(self.agent.tcp.pose)
        plug = _sp(self.charger.pose)
        held = tcp.inv() * plug
        metrics = _pitch(self._zero_held_pose, held)
        report.update(
            initialized=True,
            current_tcp_pose=_raw(tcp),
            current_charger_pose=_raw(plug),
            current_held_pose=_raw(held),
            achieved_grasp_pitch_deg=metrics["pitch_deg"],
            grasp_forward_axis_in_zero_reference=metrics[
                "forward_axis_in_zero_reference"
            ],
            grasp_lateral_residual=metrics["lateral_residual"],
            achieved_H_drift=_pose_error(held, self._initial_held_pose),
            attachment=(
                self._attachment.audit() if self._attachment is not None else None
            ),
        )
        return report

    def get_info(self):
        info = super().get_info()
        if self._episode_record is not None:
            info["charger_v2"] = self.audit()
        return info

    def close(self):
        try:
            self._release_attachment()
        finally:
            super().close()
