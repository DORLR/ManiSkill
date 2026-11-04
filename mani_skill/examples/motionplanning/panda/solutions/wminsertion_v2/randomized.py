"""Continuous teacher adapted to shifted holes inside the fixed board.

Accept the new environment ID and split the initial transport into a world-Z
lift at the actual initial TCP XY, then a slow lateral move at safe standoff.
Exact IK, command/actual guards, and later approach/insertion are inherited.
"""
import time
import numpy as np
import sapien
from .continuous import ContinuousScriptedPolicyConfig, ContinuousScriptedPolicy, _ContinuousExecution
from .scripted import _PolicyFailure, _copy_pose
from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _raw, _pose_error
from mani_skill.envs.tasks.tabletop.wminsertion_v2.env import ENV_ID


class _RandomExecution(_ContinuousExecution):
    def _send_transport_tick(self, record, source, destination, speed):
        spacing = _pose_error(self._tcp(source), self._tcp(destination))
        self._send(destination, metadata={"scheduled_speed_m_s": speed,
                   "transport_segment": record["phase"], "outside_wall_guards_active": True})
        record["maximum_command_tcp_spacing_mm"] = max(
            record.get("maximum_command_tcp_spacing_mm", 0.), spacing["translation_mm"])
        record["maximum_actual_tcp_endpoint_speed_m_s"] = max(
            record.get("maximum_actual_tcp_endpoint_speed_m_s", 0.),
            self.metrics["tcp_endpoint_linear_velocity_m_s"])
        record["maximum_actual_tcp_endpoint_angular_speed_deg_s"] = max(
            record.get("maximum_actual_tcp_endpoint_angular_speed_deg_s", 0.),
            self.metrics["tcp_endpoint_angular_velocity_deg_s"])

    def _lift(self, safe):
        source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        initial_tcp, safe_tcp = self._tcp(source), self._tcp(safe)
        safe_delta = np.asarray(safe_tcp.p) - np.asarray(initial_tcp.p)
        # Retain the original positive-lift and orientation admission checks.
        # Offset holes change safe TCP XY, so first lift at the original XY.
        if safe_delta[2] <= 0. or _pose_error(initial_tcp, safe_tcp)["rotation_deg"] > .2:
            raise _PolicyFailure("front_spawn_not_below_safe_standoff_target")
        vertical_target = sapien.Pose(
            [float(initial_tcp.p[0]), float(initial_tcp.p[1]), float(safe_tcp.p[2])], initial_tcp.q)
        vertical = self.goal.inv() * vertical_target * self.h
        delta = np.asarray(vertical_target.p) - np.asarray(initial_tcp.p)
        if np.linalg.norm(delta[:2]) > .0001 or delta[2] <= 0. or _pose_error(initial_tcp, vertical_target)["rotation_deg"] > .2:
            raise _PolicyFailure("front_spawn_not_below_safe_standoff_target")
        self.result.update(
            initial_transport="worldZ_at_initial_TCP_XY_then_lateral_at_safe_standoff",
            vertical_lift_tcp_target_pose=_raw(vertical_target),
            lateral_transport_required_mm=float(np.linalg.norm(safe_delta[:2])*1000.),
            lateral_transport_skipped=True)
        record = self._begin("front_vertical_lift", vertical_target)
        record.update(vertical_lift_m=float(delta[2]), maximum_speed_m_s=self.config.lift_speed_m_s,
                      decelerates_to_close_speed=True, nominal_xy_displacement_mm=0.,
                      source_tcp_pose=_raw(initial_tcp), orientation_held_at_initial_TCP=True,
                      waypoint_conversion="G^-1 * world_TCP * H", fixed_hold_steps=0)
        while _pose_error(self._tcp(source), vertical_target)["translation_mm"] > 1e-4:
            remaining = float(np.linalg.norm(np.asarray(vertical_target.p) - np.asarray(self._tcp(source).p)))
            mix = min(1., remaining / self.config.lift_deceleration_distance_m)
            speed = self.config.close_speed_m_s + (self.config.lift_speed_m_s-self.config.close_speed_m_s)*mix
            next_source = self._advance(source, vertical, speed)
            self._send_transport_tick(record, source, next_source, speed)
            source = next_source
        record.update(final_actual_charger_pose=_raw(self.base.charger.pose), no_lift_settle=True,
                      final_actual_tcp_error=_pose_error(_copy_pose(self.base.agent.tcp.pose), vertical_target))
        self._end()

        error = _pose_error(self._tcp(source), safe_tcp)
        if error["translation_mm"] <= 1e-4 and error["rotation_deg"] <= 1e-4:
            return
        self.result["lateral_transport_skipped"] = False
        record = self._begin("front_lateral_transport", safe_tcp)
        record.update(maximum_speed_m_s=self.config.close_speed_m_s,
                      source_tcp_pose=_raw(self._tcp(source)),
                      lateral_tcp_distance_mm=float(np.linalg.norm(safe_delta[:2])*1000.),
                      nominal_world_z_change_mm=float((safe_tcp.p[2]-self._tcp(source).p[2])*1000.),
                      safe_tip_gap_m=self.config.transport_tip_gap_m,
                      source_command_tip_gap_m=self._gap(source), destination_command_tip_gap_m=self._gap(safe),
                      waypoint_conversion="G^-1 * world_TCP * H", fixed_hold_steps=0,
                      exact_ik_and_existing_command_actual_guards=True)
        while True:
            error = _pose_error(self._tcp(source), safe_tcp)
            if error["translation_mm"] <= 1e-4 and error["rotation_deg"] <= 1e-4:
                break
            next_source = self._advance(source, safe, self.config.close_speed_m_s)
            self._send_transport_tick(record, source, next_source, self.config.close_speed_m_s)
            source = next_source
        record.update(final_actual_charger_pose=_raw(self.base.charger.pose), no_lateral_settle=True,
                      final_actual_tcp_error=_pose_error(_copy_pose(self.base.agent.tcp.pose), safe_tcp))
        self._end()

    def run(self):
        started = time.perf_counter()
        try:
            episode = getattr(self.base, "_episode_record", None) or {}
            if episode.get("env_id") != ENV_ID or episode.get("spawn_layout") != "front_fixed":
                raise _PolicyFailure("requires_reset_randomized_fixed_child_charger_v2_environment")
            if self.base.agent.control_mode != "pd_joint_pos" or getattr(self.base, "num_envs", 1) != 1:
                raise _PolicyFailure("requires_one_native_pd_joint_pos_environment")
            if abs(float(self.base.control_freq) - 20.) > 1e-6:
                raise _PolicyFailure("requires_native20Hz_control")
            self.planner = self.base._planner()
            initial_tcp = _copy_pose(self.base.agent.tcp.pose)
            self.h = initial_tcp.inv() * _copy_pose(self.base.charger.pose)
            self.goal = _copy_pose(self.base.goal_pose)
            self._initialize_pin_geometry()
            self._initialize_random(episode)
            self._initialize_command_guards()
            self._record_slot_guards()
            near = self._outside_pose(self.config.target_tip_gap_m, np.zeros(2), np.array([1., 0., 0., 0.]))
            self.near = near
            safe = self._outside_pose(self.config.transport_tip_gap_m, np.zeros(2), np.array([1., 0., 0., 0.]))
            self.result.update(frozen_achieved_grasp_pose=_raw(self.h), frozen_goal_pose=_raw(self.goal),
                initial_tcp_pose=_raw(initial_tcp), control_freq_hz=float(self.base.control_freq),
                pin_corners_charger_frame_m=self.pin_corners.tolist(), safe_transport_tcp_pose=_raw(self._tcp(safe)),
                preinsert_tcp_pose=_raw(self._tcp(near)), final_tcp_target_pose=_raw(self._tcp(sapien.Pose())),
                final_stroke_m=-float(near.p[0]), lateral_transport_skipped=True,
                push_axis_world=self.goal.to_transformation_matrix()[:3, 0].astype(float).tolist())
            self.metrics = self._snapshot()
            self.require_outside_wall = True
            self._lift(safe)
            if self.config.profile == "mixed":
                self._approach(safe)
            self._align(near)
            self._insert()
            self.result["completed"] = True
        except Exception as exc:
            self.result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            if self.stage is not None and not self.stage["completed"]:
                self.stage.update(end_step=self.steps, steps=self.steps - self.stage["start_step"],
                                  after=dict(self.metrics), failure_reason=self.result["failure_reason"])
        finally:
            self.result.update(steps=self.steps, last_phase=self.policy.phase,
                wall_seconds=time.perf_counter() - started, final_metrics=dict(self.metrics),
                minimum_outside_command_tip_gap_mm=None if self.minimum_command_tip_gap_m is None else self.minimum_command_tip_gap_m * 1000.)
            if "control_freq_hz" in self.result:
                self.result["sim_seconds"] = self.steps / self.result["control_freq_hz"]
            if hasattr(self, "goal"):
                try:
                    self.result["final_charger_goal_error"] = _pose_error(_copy_pose(self.base.charger.pose), self.goal)
                except Exception as exc:
                    self.result["final_charger_goal_error_unavailable"] = f"{type(exc).__name__}: {exc}"
        return self.result

class RandomContinuousScriptedPolicy(ContinuousScriptedPolicy):
    def run(self, env, on_step=None):
        self.phase = "initialization"
        execution = _RandomExecution(self, env, on_step)
        execution.result["environment_variant"] = ENV_ID
        execution.result["control_adapter"] = (
            "randomized env ID guard; worldZ initialTCP lift plus <=3mm/s lateral safe-standoff transport; "
            "archived exactIK, command/actual guards and later phases")
        return execution.run()
