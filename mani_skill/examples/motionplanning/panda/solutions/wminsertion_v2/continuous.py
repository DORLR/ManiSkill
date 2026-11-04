"""Continuous 20 Hz guarded Charger-v2 teacher, including small insertion noise.

G and achieved H stay frozen.  Every tick solves a new bounded Cartesian
command and executes native joint PD; no simulation-state setters are used.
Phase names are labels, not mandatory stationary holds.  The final five-sample
truth-convergence check belongs to control; any video display hold is external.
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

from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _pose_error, _raw
from .scripted import StepCallback, _PolicyFailure, _copy_pose
from .near_wall import _slerp
from .stochastic import (
    StochasticScriptedPolicy, StochasticScriptedPolicyConfig, _StochasticExecution,
)


@dataclass(frozen=True)
class ContinuousScriptedPolicyConfig(StochasticScriptedPolicyConfig):
    profile: str = "mixed"
    lift_speed_m_s: float = .030
    close_speed_m_s: float = .003
    insertion_speed_m_s: float = .003
    close_angular_speed_deg_s: float = 2.0
    lift_deceleration_distance_m: float = .006
    max_insert_tracking_lag_m: float = .0003
    insertion_noise_yz_bound_m: float = .00004
    insertion_noise_x_bound_m: float = .000015
    insertion_noise_rotation_bound_deg: float = .03
    insertion_noise_correlation: float = .85
    insertion_noise_taper_distance_m: float = .001
    insertion_noise_final_scale: float = .10
    tracking_bias_gain: float = .75
    tracking_bias_filter: float = .15
    tracking_bias_bound_rad: float = .003
    terminal_condition: str = "insertion_geometry"

    def __post_init__(self):
        inherited = {f.name: getattr(self, f.name) for f in fields(StochasticScriptedPolicyConfig)}
        inherited["profile"] = "mixed"
        StochasticScriptedPolicyConfig(**inherited)
        if self.profile not in {"clean", "mixed"}:
            raise ValueError("profile must be clean or mixed")
        for name in ("lift_speed_m_s", "close_speed_m_s", "insertion_speed_m_s",
                     "close_angular_speed_deg_s", "lift_deceleration_distance_m",
                     "max_insert_tracking_lag_m", "insertion_noise_yz_bound_m",
                     "insertion_noise_x_bound_m", "insertion_noise_rotation_bound_deg",
                     "insertion_noise_taper_distance_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lift_speed_m_s > .030 or max(self.close_speed_m_s, self.insertion_speed_m_s) > .003:
            raise ValueError("Continuous motion is bounded to lift30mm/s and close/insert3mm/s")
        if self.close_angular_speed_deg_s > 5.:
            raise ValueError("Close orientation changes must be <=0.25deg per20Hz tick")
        if self.max_insert_tracking_lag_m > self.position_tolerance_mm / 1000.:
            raise ValueError("Insertion command tracking lead cannot exceed the actual-position tolerance")
        if self.insertion_noise_yz_bound_m > .00005 or self.insertion_noise_rotation_bound_deg > .05:
            raise ValueError("Insertion target noise is bounded to50um perYZ axis and0.05deg perRPY axis")
        if self.insertion_noise_x_bound_m > .00002:
            raise ValueError("Insertion axial micro-jitter is bounded to20um")
        for name in ("insertion_noise_correlation", "insertion_noise_final_scale"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if not np.isfinite(self.tracking_bias_gain) or not 0. <= self.tracking_bias_gain <= 1.:
            raise ValueError("tracking_bias_gain must be within [0,1]; zero restores the legacy teacher")
        if not 0. < self.tracking_bias_filter <= 1. or not 0. < self.tracking_bias_bound_rad <= .003:
            raise ValueError("Tracking filter must be in (0,1], bias bound in (0,.003] radians")
        if self.terminal_condition not in {"insertion_geometry", "legacy_tcp"}:
            raise ValueError("Unknown terminal_condition")


class ContinuousScriptedPolicy(StochasticScriptedPolicy):
    """Caller resets one front_fixed fixed-child v2 environment first."""

    def __init__(self, config: ContinuousScriptedPolicyConfig | None = None):
        super().__init__(config or ContinuousScriptedPolicyConfig())

    def run(self, env: Any, on_step: StepCallback | None = None) -> dict[str, Any]:
        self.phase = "initialization"
        return _ContinuousExecution(self, env, on_step).run()


class _ContinuousExecution(_StochasticExecution):
    def __init__(self, policy, env, on_step):
        self.previous_charger_relative = None
        self.last_relative = None
        self.last_nominal_q = None
        self.insertion_active = False
        self.insertion_noise_state = np.zeros(6)
        self.approach_residual = np.zeros(7)
        self.drive_bias = np.zeros(7)
        self.drive_bias_samples = 0
        self.tracking_metadata = {}
        self.perturbation_seen = False
        super().__init__(policy, env, on_step)
        self.result.update(
            policy="charger_v2_continuous_20Hz_20261102",
            close_motion="one updated exactIK command per20Hz tick; no fixed waypoint/phase holds",
            ignored_legacy_hold_fields=["waypoint_hold_steps", "lift_min_hold_steps", "prealign_min_hold_steps", "final_min_hold_steps"],
            admission_gate="actual transverse<=0.3mm/attitude<=0.2deg and all-pin slotfit in actual3-to6mm pin-tip-gap corridor; transverse<=0.5mm/s, angular<=0.5deg/s; forward axial<=3mm/s allowed for5consecutive samples",
            stopped_gate_reference="previous scripted.py additionally required totalTCP<=0.5mm/s and all-arm qvel<=0.02rad/s",
            insertion_noise="socket-frame bounded filtered targetXYZ/RPY noise beforeIK, not additive joint noise; final1mm taper to10percent; exactG final verification",
            actual_guard="policy pre/post20Hz endpoints; caller must retain independent all100Hz contact/partial-pin checks",
            bc_filter="bad approach actions excluded; guarded insertion target noise remains an executed valid expert action; failed episodes need external eligibility verdict",
            fixed_hold_steps=0, video_display_hold_steps=0,
            tracking_compensation="filtered previous sent q minus measured q; bounded joint feedforward after IK; original guards unchanged",
        )

    def _begin(self, phase, target=None):
        record = super()._begin(phase, target)
        if phase.startswith("perturbation"):
            self.perturbation_seen = True
        self.bc_segment_id = None if phase.startswith("perturbation") else (1 if self.perturbation_seen else 0)
        if phase in {"axial_push", "final_verify"}:
            self.action_kind = "insert"
        elif phase in {"continuous_approach", "recovery_align"}:
            self.action_kind = "recovery" if self.perturbation_seen else "align"
        record.update(action_kind=self.action_kind, bc_segment_id=self.bc_segment_id,
                      fixed_hold_steps=0, recomputed_command_every_tick=True)
        return record

    def _initialize_random(self, episode):
        super()._initialize_random(episode)
        self.sampled.update(insertion_target_noise=self.config.profile == "mixed",
                            insertion_noise_frame="frozen socket-goal staticXYZ metres/degrees")
        self.result["sampled_parameters"] = dict(self.sampled)

    def _slot_geometry(self, relative):
        transform = relative.to_transformation_matrix()
        points = self.pin_corners @ transform[:3, :3].T + transform[:3, 3]
        minimum_margin, entering_margin = float("inf"), float("inf")
        any_entering = False
        for group, (cy, cz, hy, hz) in zip(points.reshape(-1, 8, 3), self.pin_slots):
            margins = np.minimum(hy - np.abs(group[:, 1] - cy),
                                 hz - np.abs(group[:, 2] - cz))
            minimum_margin = min(minimum_margin, float(margins.min()))
            # Conservatively require the WHOLE physical pin projection to fit
            # once any of its corners reaches the entrance plane.
            if float(group[:, 0].max()) >= 0.:
                any_entering = True
                entering_margin = min(entering_margin, float(margins.min()))
        return {"all_pin_slot_fit": bool(minimum_margin >= 0.),
                "minimum_slot_margin_mm": minimum_margin * 1000.,
                "pins_entering_plane": any_entering,
                "partial_pin_slot_safe": bool(not any_entering or entering_margin >= 0.),
                "minimum_entering_pin_slot_margin_mm": None if not any_entering else entering_margin * 1000.}

    def _initialize_pin_geometry(self):
        """Use each physical pin and its matching slot, including asymmetric D."""
        from mani_skill.envs.tasks.tabletop.wminsertion_v2.ports import get_port_spec
        # The archived fixed-link A environment predates the port_spec field.
        spec = getattr(self.base, "port_spec", None) or get_port_spec("A")
        self.port_spec = spec
        self.pin_slots = [(pin.center_yz_m[0], pin.center_yz_m[1],
                           pin.half_size_yz_m[0] + spec.clearance_m,
                           pin.half_size_yz_m[1] + spec.clearance_m)
                          for pin in spec.pins]
        self.pin_corners = np.asarray([
            [spec.pin_half_length_m + sx * spec.pin_half_length_m,
             pin.center_yz_m[0] + sy * pin.half_size_yz_m[0],
             pin.center_yz_m[1] + sz * pin.half_size_yz_m[1]]
            for pin in spec.pins
            for sx, sy, sz in itertools.product((-1., 1.), repeat=3)])
        self.result.update(port_type=spec.port_type, port_geometry=spec.as_dict(),
                           port_geometry_sha256=spec.sha256)

    def _record_slot_guards(self):
        self.result["noisy_command_guard"].update(
            insertion_clearance_rule="each physical pin against its corresponding slot",
            pin_slots_goal_frame_m=[list(slot) for slot in self.pin_slots],
            slot_clearance_mm=self.port_spec.clearance_m * 1000.)

    def _compensate_tracking(self, command):
        """Estimate payload following error using joint feedback, never a pose write.

        Only quiet, outside-wall alignment samples update the estimate. Every corrected
        command still passes the same joint, FK aperture and clearance guards.
        The uncompensated command remains an explicit admissible fallback.
        """
        c = self.config
        if c.tracking_bias_gain == 0.:
            self.tracking_metadata = {"tracking_estimate_updated": False, "tracking_applied_scale": 0.}
            return command, np.zeros(7), None
        qvel = np.asarray(self.base.agent.robot.get_qvel().cpu()).reshape(-1)[:7]
        actual = self._qpos()[:7]
        update = (self.last_sent_arm is not None
                  and self.policy.phase in {"continuous_approach", "recovery_align"} and self.require_outside_wall
                  and np.max(np.abs(qvel)) < .02
                  and self.metrics.get("tcp_endpoint_linear_velocity_m_s", float("inf")) < .0003
                  and self.metrics.get("tcp_endpoint_angular_velocity_deg_s", float("inf")) < .5
                  and self.metrics.get("socket_contact_force_n", float("inf")) < 1.)
        if update:
            error = np.clip(self.last_sent_arm - actual, -c.tracking_bias_bound_rad, c.tracking_bias_bound_rad)
            self.drive_bias = (1. - c.tracking_bias_filter) * self.drive_bias + c.tracking_bias_filter * error
            self.drive_bias_samples += 1
        elif self.require_outside_wall:
            self.drive_bias_samples = 0
        desired = c.tracking_bias_gain * self.drive_bias
        self.tracking_metadata = {"q_actual_pre": actual.tolist(), "tracking_estimate_updated": update,
            "tracking_calibration_samples": self.drive_bias_samples, "tracking_applied_scale": 0.,
            "tracking_estimate_frozen_in_insertion": self.insertion_active}
        if not np.any(desired):
            return command, np.zeros(7), None
        base_relative = self.goal.inv() * self._fk_tcp(command) * self.h
        for scale in (1., .5, .25, .125, 0.):
            candidate = command.copy()
            candidate[:7] = (command[:7] + scale * desired).astype(np.float32).astype(float)
            if scale and np.any(desired):
                relative = self.goal.inv() * self._fk_tcp(candidate) * self.h
                # Correct lateral/attitude following error only. Do not use a
                # bias correction to advance the commanded insertion depth.
                position = np.asarray(relative.p).copy()
                position[0] = base_relative.p[0]
                try:
                    candidate = self._exact_ik(self._tcp(sapien.Pose(position, relative.q)), candidate)
                except _PolicyFailure:
                    continue
                candidate[:7] = candidate[:7].astype(np.float32).astype(float)
            if np.max(np.abs(candidate[:7] - command[:7])) > c.tracking_bias_bound_rad:
                continue
            if self.last_sent_arm is not None:
                prior = self._qpos().copy()
                prior[:7] = self.last_sent_arm
                spacing = _pose_error(self._fk_tcp(prior), self._fk_tcp(candidate))
                if spacing["translation_mm"] > c.axial_step_m * 1000. + 1e-6 or spacing["rotation_deg"] > c.se3_rotation_step_deg:
                    continue
            reason, diagnostics = self._validate_command(candidate)
            if reason is None:
                self.tracking_metadata = {"q_actual_pre": actual.tolist(), "tracking_estimate_updated": update,
                    "tracking_calibration_samples": self.drive_bias_samples, "tracking_applied_scale": scale,
                    "tracking_estimate_frozen_in_insertion": self.insertion_active}
                return candidate, candidate[:7] - command[:7], diagnostics
        raise _PolicyFailure("tracking_compensation_and_uncompensated_command_rejected")

    def _snapshot(self):
        data = super()._snapshot()
        if not hasattr(self, "goal"):
            return data
        relative = self.goal.inv() * _copy_pose(self.base.charger.pose)
        previous = self.previous_charger_relative
        dt = float(self.base.control_timestep)
        velocity = np.zeros(3) if previous is None else (np.asarray(relative.p) - np.asarray(previous.p)) / dt
        self.previous_charger_relative = _copy_pose(relative)
        attitude = _pose_error(relative, sapien.Pose(relative.p))
        data.update(actual_charger_goal_pose=_raw(relative),
                    actual_axial_velocity_m_s=float(velocity[0]),
                    actual_transverse_velocity_m_s=float(np.linalg.norm(velocity[1:])),
                    actual_transverse_error_mm=float(np.linalg.norm(relative.p[1:]) * 1000.),
                    actual_goal_attitude_error_deg=attitude["rotation_deg"],
                    **self._slot_geometry(relative))
        if hasattr(self, "near"):
            ready, evidence = self._readiness(self.near, data)
            data.update(actual_insertion_ready=ready, actual_insertion_readiness=evidence)
        return data

    def _actual_insert_guard(self):
        m, c = self.metrics, self.config
        if m["actual_transverse_error_mm"] > c.position_tolerance_mm:
            raise _PolicyFailure("actual_insertion_transverse_error_guard")
        if m["actual_goal_attitude_error_deg"] > c.orientation_tolerance_deg:
            raise _PolicyFailure("actual_insertion_attitude_guard")
        if not m["partial_pin_slot_safe"]:
            raise _PolicyFailure("actual_partial_pin_slot_cross_section_guard")

    def _step(self, q):
        if self.insertion_active:
            self._actual_insert_guard()
        super()._step(q)
        if self.insertion_active:
            self._actual_insert_guard()

    def _validate_command(self, q):
        c, arm = self.config, q[:7]
        if q.shape != (9,) or not np.isfinite(q).all():
            return "non_finite_or_invalid_command", {}
        if np.any(arm < self.joint_limits[:7, 0]) or np.any(arm > self.joint_limits[:7, 1]):
            return "joint_limit", {}
        actual = self._qpos()
        branch = float(np.abs(arm - actual[:7]).max())
        previous = actual[:7] if self.last_sent_arm is None else self.last_sent_arm
        delta = float(np.abs(arm - previous).max())
        if branch > c.ik_max_joint_change_rad:
            return "actual_joint_branch_delta", {}
        if delta > c.max_action_delta_rad:
            return "per_action_joint_delta", {}
        if self.planner.check_for_self_collision(state=q):
            return "mplib_self_collision", {}
        if self.planner.check_for_env_collision(state=q):
            return "mplib_environment_collision", {}
        gaps, margins, partial = [], [], []
        # Include actual start and four intermediate/end configurations. This
        # is a discrete FK guard, not a claim of continuous collision checking.
        for fraction in np.linspace(0., 1., c.noise_fk_path_samples + 1):
            intermediate = actual + float(fraction) * (q - actual)
            relative = self.goal.inv() * self._fk_tcp(intermediate) * self.h
            geom = self._slot_geometry(relative)
            gaps.append(self._gap(relative))
            margins.append(geom["minimum_slot_margin_mm"])
            partial.append(geom["partial_pin_slot_safe"])
        diagnostics = {"fk_tip_gap_mm": gaps[-1] * 1000.,
                       "fk_path_minimum_tip_gap_mm": min(gaps) * 1000.,
                       "fk_path_minimum_slot_margin_mm": min(margins),
                       "fk_partial_pin_slot_safe": bool(all(partial)),
                       "max_action_delta_rad": delta, "max_actual_branch_delta_rad": branch}
        if self.require_outside_wall and min(gaps) < c.minimum_noisy_command_tip_gap_m:
            return "fk_all_pin_corner_outside_clearance", diagnostics
        if not self.require_outside_wall and not all(partial):
            return "fk_partial_pin_slot_cross_section", diagnostics
        return None, diagnostics

    def _advance(self, source, destination, speed, outside=True):
        error = _pose_error(self._tcp(source), self._tcp(destination))
        dt = float(self.base.control_timestep)
        fraction = min(1., speed * dt * 1000. / max(error["translation_mm"], 1e-12),
                       self.config.close_angular_speed_deg_s * dt / max(error["rotation_deg"], 1e-12))
        for _ in range(24):
            p = np.asarray(source.p) + fraction * (np.asarray(destination.p) - np.asarray(source.p))
            q = _slerp(source.q, destination.q, fraction)
            point = (self._outside_pose(self._gap(source) + fraction * (self._gap(destination) - self._gap(source)), p[1:], q)
                     if outside else sapien.Pose(p, q))
            step = _pose_error(self._tcp(source), self._tcp(point))
            if step["translation_mm"] <= speed * dt * 1000. + 1e-6 and step["rotation_deg"] <= .25 + 1e-7:
                return point
            fraction *= .5
        raise _PolicyFailure("continuous_target_spacing_limit_failed")

    def _send(self, nominal_relative, *, approach_noise=False, insertion_noise=False, metadata=None):
        c = self.config
        nominal_target = self._tcp(nominal_relative)
        if self.last_relative is not None and self.policy.phase != "front_vertical_lift":
            spacing = _pose_error(self._tcp(self.last_relative), nominal_target)
            if spacing["translation_mm"] > c.axial_step_m * 1000. + 1e-6 or spacing["rotation_deg"] > c.se3_rotation_step_deg + 1e-7:
                raise _PolicyFailure("continuous_nominal_target_spacing_guard")
        seed = self._qpos() if self.last_nominal_q is None else self.last_nominal_q
        nominal = self._exact_ik(nominal_target, seed)
        nominal[:7] = nominal[:7].astype(np.float32).astype(float)
        reason, diagnostics = self._validate_command(nominal)
        if reason:
            raise _PolicyFailure(f"continuous_nominal_command_guard: {reason}")
        sent, sent_relative = nominal.copy(), nominal_relative
        rejections, accepted, pose_noise = [], False, np.zeros(6)
        noisy = approach_noise or insertion_noise
        if noisy:
            for attempt in range(c.noise_resample_attempts):
                candidate, candidate_relative = nominal.copy(), nominal_relative
                if approach_noise:
                    proposal = c.joint_noise_correlation * self.approach_residual + math.sqrt(1. - c.joint_noise_correlation ** 2) * self.rng.normal(0., c.joint_noise_sigma_rad, 7)
                    if np.abs(proposal).max() > c.joint_noise_bound_rad:
                        rejections.append({"attempt": attempt + 1, "reason": "residual_bound", "proposal_rad": proposal.tolist()})
                        continue
                    candidate[:7] = (nominal[:7] + proposal).astype(np.float32).astype(float)
                    trial_noise = np.zeros(6)
                else:
                    remaining = max(0., -float(nominal_relative.p[0]))
                    taper = c.insertion_noise_final_scale + (1. - c.insertion_noise_final_scale) * min(1., remaining / c.insertion_noise_taper_distance_m)
                    bounds = np.array([c.insertion_noise_x_bound_m, c.insertion_noise_yz_bound_m, c.insertion_noise_yz_bound_m,
                                       c.insertion_noise_rotation_bound_deg, c.insertion_noise_rotation_bound_deg, c.insertion_noise_rotation_bound_deg])
                    rho = c.insertion_noise_correlation
                    # Bounded AR filtering of bounded zero-mean innovations;
                    # retain normalized state so the amplitude really tapers.
                    state = rho * self.insertion_noise_state + (1. - rho) * self.rng.uniform(-1., 1., 6)
                    trial_noise = state * bounds * taper
                    noise_rotation = sapien.Pose(q=nominal_relative.q) * sapien.Pose(q=euler2quat(*np.radians(trial_noise[3:])))
                    candidate_relative = sapien.Pose(np.asarray(nominal_relative.p) + trial_noise[:3], noise_rotation.q)
                    if float(candidate_relative.p[0]) > 0.:
                        rejections.append({"attempt": attempt + 1, "reason": "no_axial_overtravel", "target_noise_m_deg": trial_noise.tolist()})
                        continue
                    if self.last_relative is not None:
                        spacing = _pose_error(self._tcp(self.last_relative), self._tcp(candidate_relative))
                        if spacing["translation_mm"] > c.insertion_speed_m_s * float(self.base.control_timestep) * 1000. + 1e-6 or spacing["rotation_deg"] > c.se3_rotation_step_deg:
                            rejections.append({"attempt": attempt + 1, "reason": "noisy_target_spacing", **spacing})
                            continue
                    candidate = self._exact_ik(self._tcp(candidate_relative), nominal)
                    candidate[:7] = candidate[:7].astype(np.float32).astype(float)
                reason, candidate_diagnostics = self._validate_command(candidate)
                if reason:
                    rejections.append({"attempt": attempt + 1, "reason": reason, "target_noise_m_deg": trial_noise.tolist(), **candidate_diagnostics})
                    continue
                sent, sent_relative, diagnostics = candidate, candidate_relative, candidate_diagnostics
                pose_noise, accepted = trial_noise, bool(np.any(candidate[:7] != nominal[:7]))
                if insertion_noise:
                    self.insertion_noise_state = state
                break
        residual = sent[:7] - nominal[:7]
        if approach_noise:
            self.approach_residual = residual.copy()
        noisy_command = sent.copy()
        sent, tracking_offset, tracking_diagnostics = self._compensate_tracking(sent)
        if tracking_diagnostics is not None:
            diagnostics = tracking_diagnostics
        prior = self.last_relative
        spacing = None if prior is None else _pose_error(self._tcp(prior), self._tcp(sent_relative))
        self._set_command(sent_relative)
        self.action_metadata = {
            "q_nominal": nominal[:7].tolist(), "q_noisy": noisy_command[:7].tolist(), "q_sent": sent[:7].tolist(),
            "tracking_bias_estimate_rad": self.drive_bias.tolist(),
            "tracking_compensation_rad": tracking_offset.tolist(),
            **self.tracking_metadata,
            "joint_noise_residual_rad": residual.tolist(), "noise_enabled": noisy,
            "noise_accepted": accepted, "noise_fallback_no_noise": noisy and not accepted,
            "noise_rejections": rejections, "insertion_target_noise_enabled": insertion_noise, "insertion_noise_enabled": insertion_noise,
            "insertion_target_noise_m_deg": pose_noise.tolist(), "insertion_noise_offset_m_deg": pose_noise.tolist(),
            "insertion_noise_taper_scale": (None if not insertion_noise else taper),
            "noise_semantics": "socket_target_pose_beforeIK" if insertion_noise else "joint_residual_afterIK" if approach_noise else "none",
            "nominal_target_charger_goal_pose": _raw(nominal_relative),
            "perturbed_target_charger_goal_pose": _raw(sent_relative),
            "command_tcp_spacing": spacing, "fixed_hold_steps": 0,
            "nominal_fk_tip_gap_mm": self._gap(self.goal.inv() * self._fk_tcp(nominal) * self.h) * 1000.,
            "noisy_fk_tip_gap_mm": diagnostics["fk_tip_gap_mm"],
            "noisy_fk_path_minimum_tip_gap_mm": diagnostics["fk_path_minimum_tip_gap_mm"],
            "noisy_fk_path_minimum_slot_margin_mm": diagnostics["fk_path_minimum_slot_margin_mm"],
            "noisy_fk_partial_pin_slot_safe": diagnostics["fk_partial_pin_slot_safe"],
            "sent_action_max_delta_rad": diagnostics["max_action_delta_rad"],
            **(metadata or {}),
        }
        self._step(sent)
        self.last_relative, self.last_nominal_q = _copy_pose(sent_relative), nominal.copy()
        return sent

    def _lift(self, safe):
        source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        target = self._tcp(safe)
        initial_tcp = self._tcp(source)
        delta = np.asarray(target.p) - np.asarray(initial_tcp.p)
        if np.linalg.norm(delta[:2]) > .0001 or delta[2] <= 0. or _pose_error(initial_tcp, target)["rotation_deg"] > .2:
            raise _PolicyFailure("front_spawn_not_below_safe_standoff_target")
        record = self._begin("front_vertical_lift", target)
        record.update(vertical_lift_m=float(delta[2]), maximum_speed_m_s=self.config.lift_speed_m_s,
                      decelerates_to_close_speed=True)
        while _pose_error(self._tcp(source), target)["translation_mm"] > 1e-4:
            remaining = float(np.linalg.norm(np.asarray(target.p) - np.asarray(self._tcp(source).p)))
            mix = min(1., remaining / self.config.lift_deceleration_distance_m)
            speed = self.config.close_speed_m_s + (self.config.lift_speed_m_s - self.config.close_speed_m_s) * mix
            source = self._advance(source, safe, speed)
            self._send(source, metadata={"scheduled_speed_m_s": speed})
        record.update(final_actual_charger_pose=_raw(self.base.charger.pose), no_lift_settle=True)
        self._end()

    def _approach(self, safe):
        c = self.config
        record = self._begin("perturbation_stochastic", self._tcp(safe))
        source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        record.update(stochastic_steps=c.stochastic_steps, recovery_starts_from="actual post-window state",
                      phase_source_actual_charger_goal_pose=_raw(source))
        drift, endpoint, renewals = np.zeros(6), np.zeros(6), 0
        amplitude = np.array([c.drift_gap_amplitude_m, c.drift_yz_amplitude_m, c.drift_yz_amplitude_m,
                              c.drift_rotation_amplitude_deg, c.drift_rotation_amplitude_deg, c.drift_rotation_amplitude_deg])
        for index in range(c.stochastic_steps):
            renewed = index % c.drift_renew_steps == 0
            if renewed:
                endpoint = self.rng.uniform(-amplitude, amplitude)
                renewals += 1
            drift += c.drift_smoothing * (endpoint - drift)
            fraction = (index + 1) / c.stochastic_steps
            baseline_gap = self._gap(source) + fraction * (c.stochastic_end_tip_gap_m - self._gap(source))
            yz = (1. - fraction) * np.asarray(source.p)[1:] + drift[1:3]
            desired = self._outside_pose(baseline_gap + drift[0], yz, euler2quat(*np.radians(drift[3:])))
            previous = source if self.last_relative is None else self.last_relative
            command = self._advance(previous, desired, c.close_speed_m_s)
            self._send(command, approach_noise=True, metadata={
                "stochastic_index": index + 1, "stochastic_steps": c.stochastic_steps,
                "target_drift_enabled": True, "target_drift_renewed": renewed,
                "target_drift_renewal_index": renewals, "target_drift_endpoint_m_deg": endpoint.tolist(),
                "target_drift_filtered_m_deg": drift.tolist(), "noisy_target_settle_steps": 0,
                "desired_drifting_charger_goal_pose": _raw(desired)})
        record.update(target_renewals=renewals)
        self._end()

    def _readiness(self, near, metrics=None):
        c, m = self.config, (self.metrics if metrics is None else metrics)
        error = _pose_error(_copy_pose(self.base.agent.tcp.pose), self._tcp(near))
        gap_mm = m["actual_minimum_pin_corner_tip_gap_mm"]
        checks = {"transverse_position": m["actual_transverse_error_mm"] <= c.position_tolerance_mm,
                  "orientation": m["actual_goal_attitude_error_deg"] <= c.orientation_tolerance_deg,
                  "actual_axial_corridor": c.target_tip_gap_m * 1000. <= gap_mm <= c.stochastic_end_tip_gap_m * 1000.,
                  "all_pin_slot_fit": m["all_pin_slot_fit"],
                  "transverse_velocity": m["actual_transverse_velocity_m_s"] <= c.tcp_linear_velocity_tolerance_m_s,
                  "angular_velocity": m["tcp_endpoint_angular_velocity_deg_s"] <= c.tcp_angular_velocity_tolerance_deg_s,
                  "axial_velocity": -c.tcp_linear_velocity_tolerance_m_s <= m["actual_axial_velocity_m_s"] <= c.insertion_speed_m_s,
                  "tracking_bias_calibrated": c.tracking_bias_gain == 0. or self.drive_bias_samples >= c.settle_consecutive_steps}
        return all(checks.values()), {"checks": checks, "tcp_error_to_stopped_3mm_reference": error,
                                     "actual_tip_gap_mm": gap_mm,
                                     "admission_tip_gap_corridor_mm": [c.target_tip_gap_m * 1000., c.stochastic_end_tip_gap_m * 1000.],
                                     "actual_transverse_error_mm": m["actual_transverse_error_mm"],
                                     "actual_goal_attitude_error_deg": m["actual_goal_attitude_error_deg"],
                                     "actual_axial_velocity_m_s": m["actual_axial_velocity_m_s"],
                                     "actual_transverse_velocity_m_s": m["actual_transverse_velocity_m_s"],
                                     "actual_angular_velocity_deg_s": m["tcp_endpoint_angular_velocity_deg_s"],
                                     "minimum_slot_margin_mm": m["minimum_slot_margin_mm"]}

    def _align(self, near):
        phase = "recovery_align" if self.perturbation_seen else "continuous_approach"
        record = self._begin(phase, self._tcp(near))
        actual_source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        record.update(phase_source_actual_charger_goal_pose=_raw(actual_source), condition_based_admission=True,
                      consecutive_steps_required=self.config.settle_consecutive_steps)
        command = actual_source
        consecutive, endpoint_ticks = 0, 0
        while True:
            # The first recovery command is constructed from measured state,
            # then blended against the previous sent Cartesian command.
            # A small speed reserve avoids treating float32 endpoint-velocity
            # rounding around exactly3mm/s as a reason to stop for admission.
            # The physical3mm/s gate itself is unchanged.
            speed = .9 * self.config.close_speed_m_s
            desired = self._advance(command, near, speed)
            if self.last_relative is not None:
                desired = self._advance(self.last_relative, desired, speed)
            self._send(desired, metadata={"admission_consecutive_before_action": consecutive})
            command = desired
            ready, evidence = self._readiness(near)
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= self.config.settle_consecutive_steps:
                record.update(admission_actual=evidence, admission_step=self.steps,
                              admission_actual_charger_goal_pose=_raw(self.goal.inv() * _copy_pose(self.base.charger.pose)),
                              consecutive_admitted_steps=consecutive, endpoint_condition_ticks=endpoint_ticks)
                self._end()
                return
            if _pose_error(self._tcp(command), self._tcp(near))["translation_mm"] < .001:
                endpoint_ticks += 1
                if endpoint_ticks >= self.config.settle_timeout_steps:
                    record["last_admission_evidence"] = evidence
                    raise _PolicyFailure("moving_actual_alignment_admission_timeout")

    def _terminal_ready(self):
        from mani_skill.envs.tasks.tabletop.wminsertion_v2.metrics import geometry_from_poses
        c, m = self.config, self.metrics
        geometry = geometry_from_poses(self.base.charger.pose, self.goal, self.port_spec)
        checks = {"axial_depth": geometry["depth_ok"], "pin_slot_fit": geometry["pin_slot_fit"],
                  "transverse_position": geometry["lateral_error_mm"] <= c.position_tolerance_mm,
                  "orientation": geometry["angle_error_deg"] <= c.orientation_tolerance_deg,
                  "tcp_velocity": m["tcp_endpoint_linear_velocity_m_s"] <= c.tcp_linear_velocity_tolerance_m_s,
                  "angular_velocity": m["tcp_endpoint_angular_velocity_deg_s"] <= c.tcp_angular_velocity_tolerance_deg_s,
                  "joint_velocity": m["arm_joint_velocity_rad_s"] <= c.arm_joint_velocity_tolerance_rad_s}
        legacy = self._converged() and m["all_pin_slot_fit"]
        ready = legacy if c.terminal_condition == "legacy_tcp" else all(checks.values())
        return ready, {"checks": checks, "geometry": geometry,
                       "strict_tcp_converged": legacy, "terminal_condition": c.terminal_condition}

    def _insert(self):
        c = self.config
        self.require_outside_wall, self.insertion_active = False, True
        source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        # Nominal insertion starts at the last physical alignment target;
        # tracking-lead throttle below explicitly uses the actual measured x.
        x = float(self.last_relative.p[0])
        self.result.update(stopped_3mm_reference_stroke_m=self.result["final_stroke_m"], final_stroke_m=-x)
        record = self._begin("axial_push", self._tcp(sapien.Pose([x, 0., 0.])))
        record.update(phase_source_actual_charger_goal_pose=_raw(source), maximum_axial_speed_m_s=c.insertion_speed_m_s,
                      actual_start_tip_gap_mm=self._gap(source) * 1000., nominal_start_root_x_m=x,
                      actual_commanded_stroke_m=-x,
                      insertion_target_noise=c.profile == "mixed", accepted_noise_actions=0, throttled_actions=0)
        while x < -1e-10:
            self._actual_insert_guard()
            actual_x = float((self.goal.inv() * _copy_pose(self.base.charger.pose)).p[0])
            scheduled = min(0., x + c.insertion_speed_m_s * float(self.base.control_timestep))
            throttled = actual_x + c.max_insert_tracking_lag_m < scheduled
            next_x = max(x, min(scheduled, actual_x + c.max_insert_tracking_lag_m))
            # Blend any remaining sub-tolerance alignment correction and the
            # prior small target noise without an abrupt phase-boundary jump.
            relative = self._advance(self.last_relative, sapien.Pose([next_x, 0., 0.]),
                                     c.insertion_speed_m_s, outside=False)
            x = float(relative.p[0])
            self._send(relative, insertion_noise=c.profile == "mixed", metadata={
                "axial_tracking_throttled": throttled, "actual_axial_position_before_action_m": actual_x,
                "scheduled_nominal_axial_position_m": scheduled, "maximum_axial_speed_m_s": c.insertion_speed_m_s})
            record["accepted_noise_actions"] += int(self.action_metadata["noise_accepted"])
            record["throttled_actions"] += int(throttled)
        self._end()
        final = self._tcp(sapien.Pose())
        record = self._begin("final_verify", final)
        record.update(condition_based_only=True, consecutive_steps_required=c.settle_consecutive_steps,
                      timeout_steps=c.settle_timeout_steps, noise_disabled_for_exact_final_G=True,
                      terminal_condition=c.terminal_condition)
        consecutive = 0
        for _ in range(c.settle_timeout_steps):
            self._send(sapien.Pose(), metadata={"terminal_verification": True})
            ready, evidence = self._terminal_ready()
            record["terminal_evidence"] = evidence
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= c.settle_consecutive_steps:
                record["consecutive_converged_steps"] = consecutive
                self._end()
                return
        raise _PolicyFailure("final_actual_pose_or_velocity_convergence_timeout")

    def run(self):
        started = time.perf_counter()
        try:
            episode = getattr(self.base, "_episode_record", None) or {}
            if episode.get("env_id") != "WMInsertionChargerFixedLink-v2" or episode.get("spawn_layout") != "front_fixed":
                raise _PolicyFailure("requires_reset_front_fixed_child_charger_v2_environment")
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


def run_continuous_scripted_policy(env: Any, config: ContinuousScriptedPolicyConfig | None = None,
                                   on_step: StepCallback | None = None) -> dict[str, Any]:
    return ContinuousScriptedPolicy(config).run(env, on_step=on_step)


__all__ = ["ContinuousScriptedPolicyConfig", "ContinuousScriptedPolicy", "run_continuous_scripted_policy"]
