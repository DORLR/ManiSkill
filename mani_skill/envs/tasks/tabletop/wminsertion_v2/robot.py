"""Load the unchanged Panda URDF and add an articulation-internal fixed tool.

This uses the same zero-DoF articulation joint type as the fixed Franka root,
with a parent hand body rather than the world.  No external D6 is constructed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import sapien

from mani_skill import format_path
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots import PandaWristCam
from mani_skill.utils import sapien_utils

from .common import _pose_error, _raw, _sp

ROBOT_UID = "panda_wristcam_wminsertion_v2"
CHARGER_LINK_NAME = "wm_charger"
FIXED_JOINT_NAME = "wm_charger_fixed_joint"
MINIMUM_POSITION_SOLVER_ITERATIONS = 128


@register_agent()
class PandaWristCamFixedCharger(PandaWristCam):
    """Native Panda and cameras, with the native charger as a fixed child."""

    uid = ROBOT_UID

    def _load_articulation(self):
        # Match the installed b4 BaseAgent loader settings before parsing.
        loader = self.scene.create_urdf_loader()
        asset_path = format_path(str(self.urdf_path))
        loader.name = (
            self.uid
            if self._agent_idx is None
            else f"{self.uid}-agent-{self._agent_idx}"
        )
        loader.fix_root_link = self.fix_root_link
        loader.load_multiple_collisions_from_file = self.load_multiple_collisions
        loader.disable_self_collisions = self.disable_self_collisions
        if self.urdf_config is not None:
            config = sapien_utils.parse_urdf_config(self.urdf_config)
            sapien_utils.check_urdf_config(config)
            sapien_utils.apply_urdf_config(loader, config)
        builders, actors, cameras = loader.parse(asset_path)
        if len(builders) != 1 or actors or cameras:
            raise RuntimeError(
                "Expected the native Panda's single articulation and external cameras"
            )
        builder = builders[0]
        builder.disable_self_collisions = loader.disable_self_collisions
        hand = next(link for link in builder.link_builders if link.name == "panda_hand")
        child = builder.create_link_builder(hand)
        child.set_name(CHARGER_LINK_NAME)
        child.set_joint_name(FIXED_JOINT_NAME)
        child.set_joint_properties("fixed", [], sapien.Pose(), sapien.Pose())
        # Exactly the native task's three collision/visual boxes.  Density and
        # mass/inertia construction retain the native builder defaults.
        metal = sapien.render.RenderMaterial()
        metal.set_base_color([1, 1, 1, 1])
        metal.metallic, metal.roughness, metal.specular = 1.0, 0.0, 1.0
        for y in (0.007, -0.007):
            pose = sapien.Pose([0.008, y, 0.0])
            child.add_box_collision(pose, [0.008, 0.00075, 0.0032])
            child.add_box_visual(pose, [0.008, 0.00075, 0.0032], material=metal)
        plastic = sapien.render.RenderMaterial()
        plastic.set_base_color([1, 1, 1, 1])
        plastic.metallic, plastic.roughness = 0.0, 0.1
        pose = sapien.Pose([-0.020, 0.0, 0.0])
        child.add_box_collision(pose, [0.020, 0.015, 0.012])
        child.add_box_visual(pose, [0.020, 0.015, 0.012], material=plastic)
        self.robot = builder.build()
        # The fixed connection itself has no grasp DoF.  Extra position work is
        # needed for the unchanged finite-force finger/charger contact: the
        # first native-15-iteration probe penetrated 1.362mm while closing.
        self.solver_iteration_record = []
        for articulation in self.robot._objs:
            position = int(articulation.get_solver_position_iterations())
            velocity = int(articulation.get_solver_velocity_iterations())
            requested = max(position, MINIMUM_POSITION_SOLVER_ITERATIONS)
            articulation.set_solver_position_iterations(requested)
            actual_position = int(articulation.get_solver_position_iterations())
            actual_velocity = int(articulation.get_solver_velocity_iterations())
            if actual_position != requested or actual_velocity != velocity:
                raise RuntimeError(
                    "Fixed-link position solver refinement changed unexpected settings"
                )
            self.solver_iteration_record.append(
                {
                    "position_before": position,
                    "velocity_before": velocity,
                    "position_after": actual_position,
                    "velocity_after": actual_velocity,
                }
            )
        self.robot_link_ids = [link.name for link in self.robot.get_links()]
        self.native_urdf_path = str(asset_path)
        self.native_urdf_sha256 = hashlib.sha256(
            Path(asset_path).read_bytes()
        ).hexdigest()
        self.native_ik_link_names = [
            name for name in self.robot_link_ids if name != CHARGER_LINK_NAME
        ]

    def set_control_mode(self, control_mode=None):
        super().set_control_mode(control_mode)
        # BaseAgent compensates robot gravity by disabling it on every link.
        # The formerly independent native charger must still receive gravity.
        self.robot.links_map[CHARGER_LINK_NAME].disable_gravity = False


class FixedLinkGrasp:
    """Audit facade with the D6 prototype's interface, without external joints."""

    def __init__(self, base):
        self.base = base
        self.parent = base.agent.robot.links_map["panda_hand"]._objs[0]
        self.child = base.charger._objs[0]
        self.reference_hand_h = None
        self.reference_tcp_h = None
        self.released = False

    def attach(self):
        if self.child.joint.type != "fixed":
            raise RuntimeError("Charger child must have a fixed articulation joint")
        if len(self.base.agent.robot.get_active_joints()) != 9:
            raise RuntimeError("Adding charger changed the native robot's nine DoF")
        self.reference_hand_h = self.parent.entity.pose.inv() * self.child.entity.pose
        self.reference_tcp_h = (
            _sp(self.base.agent.tcp.pose).inv() * self.child.entity.pose
        )
        return self

    def release(self):
        # The fixed joint belongs to the articulation lifetime.  There are no
        # runtime constraint components or solver settings to remove/restore.
        self.released = True

    def audit(self):
        articulation = self.parent.articulation
        held = self.parent.entity.pose.inv() * self.child.entity.pose
        return {
            "implementation": "original_URDF_builder_articulation_fixed_child",
            "mode": "articulation_fixed_child",
            "parent_link": "panda_hand",
            "child_link": CHARGER_LINK_NAME,
            "joint_type": self.child.joint.type,
            "fixed_joint_count": 1,
            "active_joint_count": 1,
            "child_joint_dof": 0,
            "external_D6_count": 0,
            "root_joint_type": self.base.agent.robot.root._objs[0].joint.type,
            "root_link_name": self.base.agent.robot.root.name,
            "robot_active_joint_count": len(self.base.agent.robot.get_active_joints()),
            "reference_hand_h": _raw(self.reference_hand_h),
            "reference_tcp_h": _raw(self.reference_tcp_h),
            "current_hand_h": _raw(held),
            "hand_H_drift": _pose_error(held, self.reference_hand_h),
            "solver_position_iterations": int(
                articulation.get_solver_position_iterations()
            ),
            "solver_velocity_iterations": int(
                articulation.get_solver_velocity_iterations()
            ),
            "solver_iteration_override": True,
            "minimum_position_solver_iterations": MINIMUM_POSITION_SOLVER_ITERATIONS,
            "solver_initialization": self.base.agent.solver_iteration_record,
            "solver_refinement_reason": "finite_force_finger_charger_contact; native15_probe_penetration1.362mm",
            "charger_disable_gravity": bool(self.child.disable_gravity),
            "charger_mass_kg": float(self.child.mass),
            "charger_cmass_local_pose": _raw(self.child.cmass_local_pose),
            "charger_principal_inertia_kg_m2": list(map(float, self.child.inertia)),
            "charger_body_type": type(self.child).__name__,
            "charger_collision_shapes": [
                {
                    "type": type(shape).__name__,
                    "half_size_m": list(map(float, shape.half_size)),
                    "local_pose": _raw(shape.local_pose),
                    "contact_offset_m": float(shape.contact_offset),
                    "rest_offset_m": float(shape.rest_offset),
                    "collision_groups": list(shape.get_collision_groups()),
                }
                for shape in self.child.get_collision_shapes()
            ],
            "original_urdf_path": self.base.agent.native_urdf_path,
            "original_urdf_sha256": self.base.agent.native_urdf_sha256,
            "original_urdf_modified": False,
            "pose_writes_during_step": False,
            "released": self.released,
            "attached": not self.released,
            "connection_lifetime": "robot_articulation",
        }
