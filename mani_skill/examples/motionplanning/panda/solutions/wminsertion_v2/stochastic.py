"""Streamed near-wall motion perturbations with the frozen Charger-v2 G/H.

Only the bounded 12-to-about-6 mm approach is perturbed.  The target drifts
during motion, and/or correlated arm-position residuals are added AFTER exact
IK.  There is no per-noisy-target settle.  Recovery starts from the measured
resulting state and uses the existing exact alignment, gates and fixed push.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import itertools
import math
import time
from typing import Any

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _pose_error, _raw, _sp
from .scripted import StepCallback, _PolicyFailure, _copy_pose
from .near_wall import (
    NearWallScriptedPolicy, NearWallScriptedPolicyConfig, _NearWallExecution, _slerp,
)


@dataclass(frozen=True)
class StochasticScriptedPolicyConfig(NearWallScriptedPolicyConfig):
    profile: str = "mixed"
    stochastic_steps: int = 56
    stochastic_end_tip_gap_m: float = 0.006
    drift_renew_steps: int = 5
    drift_smoothing: float = 0.25
    drift_gap_amplitude_m: float = 0.0008
    drift_yz_amplitude_m: float = 0.0015
    drift_rotation_amplitude_deg: float = 1.5
    joint_noise_sigma_rad: float = 0.0013
    joint_noise_bound_rad: float = 0.004
    joint_noise_correlation: float = 0.85
    max_action_delta_rad: float = 0.02
    minimum_noisy_command_tip_gap_m: float = 0.002
    noise_resample_attempts: int = 12
    noise_fk_path_samples: int = 4

    def __post_init__(self):
        # Validate every inherited bound without asking the static-perturbation
        # parent to recognize the new profile names.
        inherited = {f.name: getattr(self, f.name) for f in fields(NearWallScriptedPolicyConfig)}
        inherited["profile"] = "clean"
        NearWallScriptedPolicyConfig(**inherited)
        if self.profile not in {"action_noise", "target_drift", "mixed"}:
            raise ValueError("profile must be action_noise, target_drift or mixed")
        for name in ("stochastic_steps", "drift_renew_steps", "noise_resample_attempts", "noise_fk_path_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 40 <= self.stochastic_steps <= 70:
            raise ValueError("The stochastic window must contain 40 through 70 control actions")
        if not 2 <= self.drift_renew_steps <= 10:
            raise ValueError("Renew moving drift targets every 2 through 10 control actions")
        for name in ("stochastic_end_tip_gap_m", "drift_gap_amplitude_m", "drift_yz_amplitude_m",
                     "drift_rotation_amplitude_deg", "joint_noise_sigma_rad", "joint_noise_bound_rad",
                     "max_action_delta_rad", "minimum_noisy_command_tip_gap_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("drift_smoothing", "joint_noise_correlation"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if not self.target_tip_gap_m < self.stochastic_end_tip_gap_m < self.transport_tip_gap_m:
            raise ValueError("Stochastic approach must end between safe transport and exact recovery")
        if self.stochastic_end_tip_gap_m - self.drift_gap_amplitude_m <= self.minimum_noisy_command_tip_gap_m:
            raise ValueError("Every moving nominal target must leave the noisy-command clearance margin")
        if self.minimum_noisy_command_tip_gap_m < 0.002:
            raise ValueError("Noisy FK commands must retain at least 2 mm pin-corner clearance")
        if self.minimum_actual_tip_gap_m < 0.001:
            raise ValueError("Actual pre-insertion clearance guard must retain at least 1 mm")
        if self.joint_noise_sigma_rad >= self.joint_noise_bound_rad:
            raise ValueError("Joint noise sigma must be below its hard rejection bound")
        if self.max_action_delta_rad > 0.025:
            raise ValueError("Per-action joint changes must stay within the native 0.5 rad/s planning bound at 20 Hz")


class StochasticScriptedPolicy(NearWallScriptedPolicy):
    """One already-reset fixed-child environment; callback API is unchanged."""

    def __init__(self, config: StochasticScriptedPolicyConfig | None = None):
        super().__init__(config or StochasticScriptedPolicyConfig())

    def run(self, env: Any, on_step: StepCallback | None = None) -> dict[str, Any]:
        self.phase = "initialization"
        return _StochasticExecution(self, env, on_step).run()


class _StochasticExecution(_NearWallExecution):
    def __init__(self, policy: StochasticScriptedPolicy, env: Any, on_step: StepCallback | None):
        self.action_metadata: dict[str, Any] = {}
        self.bc_segment_id: int | None = 0
        self.last_sent_arm: np.ndarray | None = None
        super().__init__(policy, env, on_step)
        self.result.update(
            policy="charger_v2_streamed_stochastic_recovery_20261102",
            close_motion="streamed stochastic window; exact-IK recovery and original fixed push",
            stochastic_action_semantics="absolute native 7-joint positions in radians plus unchanged close=-1; residual added after IK",
            noise_process="seeded symmetric zero-mean AR(1) proposals; unsafe/bounded proposals rejected, never geometry-clipped",
            bc_filter="retain all raw rows; stochastic actions excluded; BC chunks cannot cross segment boundary",
            actual_gap_guard="module pre/post 20Hz endpoints; evaluation callback additionally audits all 100Hz physics substeps",
        )

    def _begin(self, phase: str, target: sapien.Pose | None = None) -> dict[str, Any]:
        self.action_metadata = {}
        self.bc_segment_id = (None if phase.startswith("perturbation") else
                              1 if phase.startswith("recovery") or phase in {"axial_push", "final_settle"} else 0)
        record = super()._begin(phase, target)
        if phase == "perturbation_stochastic":
            self.action_kind = self.config.profile
        record.update(action_kind=self.action_kind, bc_segment_id=self.bc_segment_id)
        return record

    def _step_metadata(self) -> dict[str, Any]:
        return {**super()._step_metadata(), "bc_segment_id": self.bc_segment_id,
                "outside_wall_required": self.require_outside_wall, **self.action_metadata}

    def _step(self, q: np.ndarray):
        if self.require_outside_wall:
            gap = self._gap(self.goal.inv() * _copy_pose(self.base.charger.pose))
            if gap < self.config.minimum_actual_tip_gap_m:
                raise _PolicyFailure("actual_pin_corner_tip_gap_guard_before_action")
        super()._step(q)
        self.last_sent_arm = np.asarray(q[:7], dtype=np.float32).astype(float)

    def _initialize_random(self, episode: dict[str, Any]):
        seed = (int(episode["episode_seed"]) + self.config.seed_offset) % (2 ** 64)
        self.rng = np.random.default_rng(seed)
        self.sampled = {
            "requested_profile": self.config.profile, "profile": self.config.profile,
            "env_episode_seed": int(episode["episode_seed"]), "seed_offset": self.config.seed_offset,
            "sampling_seed": seed, "tip_gap_m": self.config.stochastic_end_tip_gap_m,
            "stochastic_steps": self.config.stochastic_steps, "drift_renew_steps": self.config.drift_renew_steps,
            "drift_frame": "socket goal local gap,Y,Z and static XYZ roll,pitch,yaw",
        }
        self.result["sampled_parameters"] = dict(self.sampled)

    def _fk_tcp(self, q: np.ndarray) -> sapien.Pose:
        self.planner.pinocchio_model.compute_forward_kinematics(q)
        local = self.planner.pinocchio_model.get_link_pose(self.planner.link_name_2_idx[self.planner.move_group])
        return _copy_pose(self.base.agent.robot.pose) * sapien.Pose(local.p, local.q)

    def _initialize_command_guards(self):
        self.joint_limits = np.asarray(self.planner.joint_limits, dtype=float)
        if self.joint_limits.ndim != 2 or self.joint_limits.shape[0] < 7 or self.joint_limits.shape[1] != 2:
            raise _PolicyFailure("invalid_mplib_joint_limits")
        if not np.isfinite(self.joint_limits[:7]).all():
            raise _PolicyFailure("non_finite_mplib_arm_joint_limits")
        check = _pose_error(self._fk_tcp(self._qpos()), _copy_pose(self.base.agent.tcp.pose))
        self.result["noisy_command_fk_calibration"] = check
        if check["translation_mm"] > 0.1 or check["rotation_deg"] > 0.1:
            raise _PolicyFailure("planner_fk_does_not_match_actual_tcp")
        self.result["noisy_command_guard"] = {
            "minimum_all_pin_corner_gap_mm": self.config.minimum_noisy_command_tip_gap_m * 1000.,
            "max_action_delta_rad": self.config.max_action_delta_rad,
            "max_actual_branch_delta_rad": self.config.ik_max_joint_change_rad,
            "joint_limits_rad": self.joint_limits[:7].tolist(),
            "collision_api": "MPlib0.2.1 check_for_self_collision(state=q),check_for_env_collision(state=q)",
            "collision_scope": "native planner model; physical socket/table contacts also audited by existing actual guards",
            "fk_interpolation_samples": self.config.noise_fk_path_samples,
        }

    def _validate_command(self, q: np.ndarray) -> tuple[str | None, dict[str, float]]:
        """Validate the float32 values that will actually be sent, without state setters."""
        c, arm = self.config, q[:7]
        if q.shape != (9,) or not np.isfinite(q).all():
            return "non_finite_or_invalid_command", {}
        if np.any(arm < self.joint_limits[:7, 0]) or np.any(arm > self.joint_limits[:7, 1]):
            return "joint_limit", {}
        actual = self._qpos()
        branch_delta = float(np.abs(arm - actual[:7]).max())
        if branch_delta > c.ik_max_joint_change_rad:
            return "actual_joint_branch_delta", {}
        previous = actual[:7] if self.last_sent_arm is None else self.last_sent_arm
        action_delta = float(np.abs(arm - previous).max())
        if action_delta > c.max_action_delta_rad:
            return "per_action_joint_delta", {}
        # .2.1 takes state=, not the older qpos=/with_point_cloud= interface.
        if self.planner.check_for_self_collision(state=q):
            return "mplib_self_collision", {}
        if self.planner.check_for_env_collision(state=q):
            return "mplib_environment_collision", {}
        gaps = []
        for fraction in np.linspace(0., 1., c.noise_fk_path_samples + 1)[1:]:
            intermediate = actual + float(fraction) * (q - actual)
            relative = self.goal.inv() * self._fk_tcp(intermediate) * self.h
            gaps.append(self._gap(relative))
        diagnostics = {"fk_tip_gap_mm": gaps[-1] * 1000.,
                       "fk_path_minimum_tip_gap_mm": min(gaps) * 1000.,
                       "max_action_delta_rad": action_delta, "max_actual_branch_delta_rad": branch_delta}
        if min(gaps) < c.minimum_noisy_command_tip_gap_m:
            return "fk_all_pin_corner_clearance", diagnostics
        return None, diagnostics

    def _rate_limited_target(self, source: sapien.Pose, destination: sapien.Pose) -> sapien.Pose:
        """Limit streamed target spacing; do not alter a target to repair wall geometry."""
        source_gap, destination_gap = self._gap(source), self._gap(destination)
        fraction = 1.
        for _ in range(20):
            point = self._outside_pose(source_gap + fraction * (destination_gap - source_gap),
                np.asarray(source.p)[1:] + fraction * (np.asarray(destination.p)[1:] - np.asarray(source.p)[1:]),
                _slerp(source.q, destination.q, fraction))
            spacing = _pose_error(self._tcp(source), self._tcp(point))
            if (spacing["translation_mm"] <= self.config.axial_step_m * 1000. + 1e-7 and
                    spacing["rotation_deg"] <= self.config.se3_rotation_step_deg + 1e-7):
                return point
            fraction *= 0.5
        raise _PolicyFailure("stochastic_target_rate_limit_failed")

    def _stochastic_approach(self, safe: sapien.Pose) -> np.ndarray:
        c = self.config
        has_drift, has_noise = c.profile in {"target_drift", "mixed"}, c.profile in {"action_noise", "mixed"}
        record = self._begin("perturbation_stochastic", self._tcp(safe))
        record.update(stochastic_steps=c.stochastic_steps, target_renew_every_steps=c.drift_renew_steps,
                      per_target_settle_steps=0, supervised_recovery_starts_from="actual post-window charger/TCP/qpos")
        q_nominal, previous = self._qpos(), safe
        drift, drift_endpoint, residual = np.zeros(6), np.zeros(6), np.zeros(7)
        amplitude = np.array([c.drift_gap_amplitude_m, c.drift_yz_amplitude_m, c.drift_yz_amplitude_m,
                              c.drift_rotation_amplitude_deg, c.drift_rotation_amplitude_deg, c.drift_rotation_amplitude_deg])
        renewals, accepted, fallback = 0, 0, 0
        for index in range(c.stochastic_steps):
            renewed = has_drift and index % c.drift_renew_steps == 0
            prior_error = _pose_error(_copy_pose(self.base.agent.tcp.pose), self._tcp(previous)) if renewed else None
            if renewed:
                drift_endpoint = self.rng.uniform(-amplitude, amplitude)
                renewals += 1
            if has_drift:
                drift += c.drift_smoothing * (drift_endpoint - drift)
            fraction = (index + 1) / c.stochastic_steps
            baseline_gap = c.transport_tip_gap_m + fraction * (c.stochastic_end_tip_gap_m - c.transport_tip_gap_m)
            desired = self._outside_pose(baseline_gap + drift[0], drift[1:3], euler2quat(*np.radians(drift[3:])))
            relative = self._rate_limited_target(previous, desired)
            self._set_command(relative)
            q_nominal = self._exact_ik(self.target, q_nominal)
            # The parent sends float32 arm positions; validate/record that exact representation.
            nominal = q_nominal.copy()
            nominal[:7] = nominal[:7].astype(np.float32).astype(float)
            reason, nominal_diagnostics = self._validate_command(nominal)
            if reason:
                raise _PolicyFailure(f"stochastic_nominal_command_guard: {reason}")
            sent, diagnostics, rejections = nominal.copy(), nominal_diagnostics, []
            noise_accepted = False
            if has_noise:
                for attempt in range(c.noise_resample_attempts):
                    proposal = c.joint_noise_correlation * residual + math.sqrt(1. - c.joint_noise_correlation ** 2) * self.rng.normal(0., c.joint_noise_sigma_rad, 7)
                    if np.abs(proposal).max() > c.joint_noise_bound_rad:
                        rejections.append({"attempt": attempt + 1, "reason": "residual_bound", "proposal_rad": proposal.tolist()})
                        continue
                    candidate = nominal.copy()
                    candidate[:7] = (nominal[:7] + proposal).astype(np.float32).astype(float)
                    reason, candidate_diagnostics = self._validate_command(candidate)
                    if reason:
                        rejections.append({"attempt": attempt + 1, "reason": reason,
                                           "proposal_rad": proposal.tolist(), **candidate_diagnostics})
                        continue
                    sent, diagnostics, noise_accepted = candidate, candidate_diagnostics, True
                    accepted += 1
                    break
                if not noise_accepted:
                    fallback += 1
            residual = sent[:7] - nominal[:7]
            self.action_metadata = {
                "stochastic_index": index + 1, "stochastic_steps": c.stochastic_steps,
                "q_nominal": nominal[:7].tolist(), "q_noisy": sent[:7].tolist(),
                "joint_noise_residual_rad": residual.tolist(), "noise_enabled": has_noise,
                "noise_accepted": noise_accepted, "noise_fallback_no_noise": has_noise and not noise_accepted,
                "noise_rejections": rejections, "nominal_fk_tip_gap_mm": nominal_diagnostics["fk_tip_gap_mm"],
                "noisy_fk_tip_gap_mm": diagnostics["fk_tip_gap_mm"],
                "noisy_fk_path_minimum_tip_gap_mm": diagnostics["fk_path_minimum_tip_gap_mm"],
                "sent_action_max_delta_rad": diagnostics["max_action_delta_rad"],
                "sent_action_max_actual_branch_delta_rad": diagnostics["max_actual_branch_delta_rad"],
                "target_drift_enabled": has_drift, "target_drift_renewed": bool(renewed),
                "target_drift_renewal_index": renewals, "target_drift_endpoint_m_deg": drift_endpoint.tolist(),
                "target_drift_filtered_m_deg": drift.tolist(), "previous_target_error_at_renewal": prior_error,
                "scheduled_baseline_tip_gap_mm": baseline_gap * 1000.,
                "nominal_target_charger_goal_pose": _raw(relative), "desired_drifting_charger_goal_pose": _raw(desired),
                "noisy_target_settle_steps": 0,
            }
            self._step(sent)
            previous = relative
        record.update(target_renewals=renewals, accepted_noise_actions=accepted,
                      no_noise_fallback_actions=fallback, final_actual_tcp_pose=_raw(self.base.agent.tcp.pose),
                      final_actual_charger_pose=_raw(self.base.charger.pose), final_actual_qpos=self._qpos().tolist())
        self._end()
        return self._qpos()

    def run(self) -> dict[str, Any]:
        # Only phase orchestration differs from the parent. Lift, recovery,
        # convergence checks and the final physical push reuse its methods.
        started = time.perf_counter()
        try:
            episode = getattr(self.base, "_episode_record", None) or {}
            if episode.get("env_id") != "WMInsertionChargerFixedLink-v2":
                raise _PolicyFailure("requires_reset_fixed_child_charger_v2_environment")
            if self.base.agent.control_mode != "pd_joint_pos" or getattr(self.base, "num_envs", 1) != 1:
                raise _PolicyFailure("requires_one_native_pd_joint_pos_environment")
            if abs(float(self.base.control_freq) - 20.) > 1e-6:
                raise _PolicyFailure("requires_native_20Hz_control_for_streamed_perturbations")
            self.planner = self.base._planner()
            initial_tcp = _copy_pose(self.base.agent.tcp.pose)
            self.h = initial_tcp.inv() * _copy_pose(self.base.charger.pose)
            self.goal = _copy_pose(self.base.goal_pose)
            half, gap = np.asarray(self.base._peg_size, dtype=float), float(self.base._peg_gap)
            self.pin_corners = np.asarray([[half[0] + sx * half[0], y + sy * half[1], sz * half[2]]
                for y in (-gap, gap) for sx, sy, sz in itertools.product((-1., 1.), repeat=3)])
            self._initialize_random(episode)
            self._initialize_command_guards()
            identity = np.array([1., 0., 0., 0.])
            near = self._outside_pose(self.config.target_tip_gap_m, np.zeros(2), identity)
            safe = self._outside_pose(self.config.transport_tip_gap_m, np.zeros(2), identity)
            final = self._tcp(sapien.Pose())
            self.result.update(frozen_achieved_grasp_pose=_raw(self.h), frozen_goal_pose=_raw(self.goal),
                initial_tcp_pose=_raw(initial_tcp), control_freq_hz=float(self.base.control_freq),
                pin_corners_charger_frame_m=self.pin_corners.tolist(),
                safe_transport_tcp_pose=_raw(self._tcp(safe)), preinsert_tcp_pose=_raw(self._tcp(near)),
                final_tcp_target_pose=_raw(final), final_stroke_m=-float(near.p[0]),
                push_axis_world=self.goal.to_transformation_matrix()[:3, 0].astype(float).tolist())
            self.metrics = self._snapshot()
            if episode.get("spawn_layout") == "front_fixed":
                self.require_outside_wall = True
                q = self._front_vertical_lift(self._tcp(safe))
                self._settle("front_lift_settle", self._tcp(safe), self.config.prealign_min_hold_steps, q=q)
                self.result["lateral_transport_skipped"] = True
            else:
                lift = sapien.Pose(np.asarray(initial_tcp.p) + [0., 0., self.config.lift_m], initial_tcp.q)
                self._transport("lift_transport", lift)
                self._settle("lift_settle", lift, self.config.lift_min_hold_steps)
                self._transport("safe_standoff_transport", self._tcp(safe))
                self._settle("safe_standoff_settle", self._tcp(safe), self.config.prealign_min_hold_steps)
                self.result["lateral_transport_skipped"] = False
            self.require_outside_wall = True
            self._stochastic_approach(safe)
            # Deliberately omit a command seed: both Cartesian source and qpos
            # come from the actual resulting state, never an unexecuted goal.
            q = self._close_move("recovery_align", near)
            q = self._settle_outside("recovery_settle", near, self.config.prealign_min_hold_steps, q)
            self.require_outside_wall = False
            record = self._begin("axial_push", self._tcp(near))
            stroke = -float(near.p[0])
            count = int(math.ceil(stroke / self.config.axial_step_m - 1e-10))
            record.update(waypoints=count, stroke_m=stroke, hold_steps_per_waypoint=self.config.waypoint_hold_steps)
            for index in range(1, count + 1):
                relative = sapien.Pose([-max(0., stroke - index * self.config.axial_step_m), 0., 0.])
                self._set_command(relative)
                q = self._exact_ik(self.target, q)
                for _ in range(self.config.waypoint_hold_steps):
                    self._step(q)
                record["completed_waypoints"] = index
            self._end()
            self._settle("final_settle", final, self.config.final_min_hold_steps, q=q)
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
            if "frozen_goal_pose" in self.result:
                try:
                    self.result["final_charger_goal_error"] = _pose_error(_sp(self.base.charger.pose), self.goal)
                except Exception as exc:
                    self.result["final_charger_goal_error_unavailable"] = f"{type(exc).__name__}: {exc}"
        return self.result


def run_stochastic_scripted_policy(env: Any, config: StochasticScriptedPolicyConfig | None = None,
                                   on_step: StepCallback | None = None) -> dict[str, Any]:
    return StochasticScriptedPolicy(config).run(env, on_step=on_step)


__all__ = ["StochasticScriptedPolicyConfig", "StochasticScriptedPolicy", "run_stochastic_scripted_policy"]
