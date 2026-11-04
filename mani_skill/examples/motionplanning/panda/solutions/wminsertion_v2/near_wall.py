"""Near-wall 6D recovery demonstrations using the frozen Charger-v2 G and H.

The charger frame is at the *pin root*.  A 3 mm tip gap therefore places its
aligned root at G.x=-19 mm, since the unchanged physical pins are 16 mm long.
Perturbations remain outside the wall and are explicitly excluded from BC;
their resulting observations precede the supervised correction actions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import itertools
import math
import time
from typing import Any

import mplib
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _pose_error, _raw, _sp
from .scripted import (
    ChargerV2ScriptedPolicy, ScriptedPolicyConfig, StepCallback, _Execution,
    _PolicyFailure, _copy_pose,
)


@dataclass(frozen=True)
class NearWallScriptedPolicyConfig(ScriptedPolicyConfig):
    # Retained baseline field denotes the aligned root-to-G final stroke.
    # Actual stroke is recomputed from physical pin corners and target_tip_gap.
    preinsert_distance_m: float = 0.019
    target_tip_gap_m: float = 0.003
    transport_tip_gap_m: float = 0.012
    se3_rotation_step_deg: float = 0.25
    profile: str = "random"
    clean_probability: float = 0.25
    seed_offset: int = 20261102
    perturb_tip_gap_min_m: float = 0.002
    perturb_tip_gap_max_m: float = 0.005
    perturb_translation_min_m: float = 0.001
    perturb_translation_max_m: float = 0.003
    perturb_rotation_min_deg: float = 1.0
    perturb_rotation_max_deg: float = 3.0
    minimum_actual_tip_gap_m: float = 0.001

    def __post_init__(self):
        # The parent's asdict validation intentionally knows only its fields.
        ScriptedPolicyConfig(**{f.name: getattr(self, f.name)
                                for f in fields(ScriptedPolicyConfig)})
        if self.profile not in {"random", "clean", "overshoot", "undershoot"}:
            raise ValueError("profile must be random, clean, overshoot or undershoot")
        if not np.isfinite(self.clean_probability) or not 0 <= self.clean_probability <= 1:
            raise ValueError("clean_probability must be in [0, 1]")
        if isinstance(self.seed_offset, bool) or not isinstance(self.seed_offset, int) or self.seed_offset < 0:
            raise ValueError("seed_offset must be a nonnegative integer")
        for name in ("target_tip_gap_m", "transport_tip_gap_m", "se3_rotation_step_deg",
                     "perturb_tip_gap_min_m", "perturb_tip_gap_max_m",
                     "perturb_translation_min_m", "perturb_translation_max_m",
                     "perturb_rotation_min_deg", "perturb_rotation_max_deg",
                     "minimum_actual_tip_gap_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.perturb_tip_gap_min_m < self.target_tip_gap_m < self.perturb_tip_gap_max_m:
            raise ValueError("target tip gap must lie inside the perturbation gap range")
        if self.transport_tip_gap_m <= self.perturb_tip_gap_max_m:
            raise ValueError("free transport must end outside all near-wall perturbations")
        if self.minimum_actual_tip_gap_m >= self.perturb_tip_gap_min_m:
            raise ValueError("Actual gap guard must leave room below every perturbation target")
        if self.se3_rotation_step_deg > 0.25:
            raise ValueError("Close-motion orientation spacing must be at most 0.25 degrees")
        for lo, hi in ((self.perturb_translation_min_m, self.perturb_translation_max_m),
                       (self.perturb_rotation_min_deg, self.perturb_rotation_max_deg)):
            if lo >= hi:
                raise ValueError("Perturbation magnitudes require continuous nonzero ranges")


class NearWallScriptedPolicy(ChargerV2ScriptedPolicy):
    """Run on one already-reset fixed-child v2 env; callback API is unchanged.

    ``info['scripted_policy']['use_for_bc']`` is false for every deliberately
    bad perturbation action, including its hold.  Keep all raw observations;
    the first recovery action uses the actual post-perturbation observation.
    """

    def __init__(self, config: NearWallScriptedPolicyConfig | None = None):
        super().__init__(config or NearWallScriptedPolicyConfig())

    def run(self, env: Any, on_step: StepCallback | None = None) -> dict[str, Any]:
        self.phase = "initialization"
        return _NearWallExecution(self, env, on_step).run()


def _slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = a + fraction * (b - a)
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1 - fraction) * angle) * a
            + math.sin(fraction * angle) * b) / math.sin(angle)


class _NearWallExecution(_Execution):
    def __init__(self, policy: NearWallScriptedPolicy, env: Any,
                 on_step: StepCallback | None):
        self.external_on_step = on_step
        super().__init__(policy, env, self._dispatch_step)
        self.use_for_bc, self.action_kind = True, "initialization"
        self.require_outside_wall = False
        self.command_tip_gap_m: float | None = None
        self.minimum_command_tip_gap_m: float | None = None
        self.sampled: dict[str, Any] = {}
        self.result.update(
            policy="charger_v2_near_wall_6d_recovery_20261102",
            goal_feedback="frozen_G_and_achieved_H; privileged 6D recovery outside wall",
            bc_filter="exclude actions with use_for_bc=false; retain all raw observations",
            close_motion="exact IK, <=0.5mm TCP spacing, <=0.25deg, >=3steps per waypoint",
        )

    def _exact_ik(self, target: sapien.Pose, seed: np.ndarray) -> np.ndarray:
        try:
            return super()._exact_ik(target, seed)
        except _PolicyFailure as exc:
            if '_ik_failed' not in str(exc):
                raise
            # MPlib's CLIK default eps=1e-5 uses a different residual from its
            # wrapper's position+quaternion distance. A valid local solution
            # can miss that wrapper threshold by a tiny amount. Refine the
            # SAME target more tightly, then rerun the original wrapper's joint
            # limit/collision/residual checks. No physical state is changed.
            local=self.planner._transform_goal_to_wrt_base(mplib.Pose(target.p,target.q))
            eps=min(1e-7,self.config.ik_threshold*.01)
            refined,success,_=self.planner.pinocchio_model.compute_IK_CLIK(
                self.planner.link_name_2_idx[self.planner.move_group],local,seed,
                eps=eps,max_iter=1000)
            attempt={'phase':self.policy.phase,'original_failure':str(exc),'clik_eps':eps,
                     'outer_threshold_unchanged':self.config.ik_threshold,'refinement_converged':bool(success)}
            self.result.setdefault('ik_local_refinements',[]).append(attempt)
            if not success:
                raise _PolicyFailure(f'{self.policy.phase}_local_ik_refinement_failed') from exc
            q=super()._exact_ik(target,np.asarray(refined,dtype=float).reshape(-1))
            if np.abs(q[:7]-seed[:7]).max()>self.config.ik_max_joint_change_rad:
                raise _PolicyFailure(f'{self.policy.phase}_ik_refinement_branch_jump')
            attempt['accepted']=True
            return q

    def _begin(self, phase: str, target: sapien.Pose | None = None) -> dict[str, Any]:
        self.use_for_bc = not phase.startswith("perturbation")
        self.action_kind = ("perturbation" if not self.use_for_bc else
                            "recovery" if phase.startswith("recovery") else
                            "insert" if phase in {"axial_push", "final_settle"} else
                            "align" if phase.startswith(("near_wall", "prealign")) else
                            "transport")
        record = super()._begin(phase, target)
        record.update(use_for_bc=self.use_for_bc, action_kind=self.action_kind)
        # Coarse planner points have no Cartesian waypoint guarantee; report
        # their target separately, never pretend it is their current command.
        self.command_tip_gap_m = None
        if target is not None and not phase.endswith("transport") and hasattr(self, "goal"):
            self._set_command(self.goal.inv() * target * self.h)
        return record

    def _step_metadata(self) -> dict[str, Any]:
        return dict(use_for_bc=self.use_for_bc, action_kind=self.action_kind,
                    command_tip_gap_mm=None if self.command_tip_gap_m is None else
                    self.command_tip_gap_m * 1000., sampled_parameters=dict(self.sampled),
                    sampled_tip_gap_mm=None if not self.sampled else self.sampled["tip_gap_m"] * 1000.)

    def _dispatch_step(self, phase: str, action: np.ndarray, info: dict[str, Any]):
        # Preserve BC flags even if an actual-state reader raises and the base
        # class substitutes state_error diagnostics for that executed action.
        info["scripted_policy"].update(self._step_metadata())
        return self.external_on_step(phase, action, info) if self.external_on_step else None

    def _step(self, q: np.ndarray):
        super()._step(q)
        if (self.require_outside_wall and
                self.metrics["actual_minimum_pin_corner_tip_gap_mm"] < self.config.minimum_actual_tip_gap_m * 1000.):
            raise _PolicyFailure("actual_pin_corner_tip_gap_guard_before_insertion")

    def _gap(self, relative: sapien.Pose) -> float:
        transform = relative.to_transformation_matrix()
        return -float((self.pin_corners @ transform[:3, :3].T + transform[:3, 3])[:, 0].max())

    def _outside_pose(self, tip_gap: float, yz: np.ndarray,
                      quaternion: np.ndarray) -> sapien.Pose:
        rotation = sapien.Pose(q=quaternion).to_transformation_matrix()[:3, :3]
        xmax = float((self.pin_corners @ rotation.T)[:, 0].max())
        return sapien.Pose([-tip_gap - xmax, float(yz[0]), float(yz[1])], quaternion)

    def _tcp(self, relative: sapien.Pose) -> sapien.Pose:
        return self.goal * relative * self.h.inv()

    def _set_command(self, relative: sapien.Pose):
        self.target = self._tcp(relative)
        self.command_tip_gap_m = self._gap(relative)
        if self.require_outside_wall:
            if self.command_tip_gap_m <= 0:
                raise _PolicyFailure("commanded_pin_corner_crosses_wall_before_insertion")
            self.minimum_command_tip_gap_m = (
                self.command_tip_gap_m if self.minimum_command_tip_gap_m is None else
                min(self.minimum_command_tip_gap_m, self.command_tip_gap_m))

    def _snapshot(self) -> dict[str, Any]:
        data = super()._snapshot()
        data.update(self._step_metadata())
        if hasattr(self, "goal"):
            gap = self._gap(self.goal.inv() * _copy_pose(self.base.charger.pose))
            data["actual_minimum_pin_corner_tip_gap_mm"] = gap * 1000.
            data["outside_wall_required"] = self.require_outside_wall
        return data

    def _sample(self, episode: dict[str, Any]):
        c = self.config
        episode_seed = int(episode["episode_seed"])
        seed = (episode_seed + c.seed_offset) % (2 ** 64)
        rng = np.random.default_rng(seed)
        profile = c.profile
        if profile == "random":
            profile = ("clean" if rng.random() < c.clean_probability else
                       str(rng.choice(["overshoot", "undershoot"])))
        gap, yz, rpy = c.target_tip_gap_m, np.zeros(2), np.zeros(3)
        if profile != "clean":
            bounds = ((c.perturb_tip_gap_min_m, c.target_tip_gap_m) if profile == "overshoot"
                      else (c.target_tip_gap_m, c.perturb_tip_gap_max_m))
            gap = float(rng.uniform(*bounds))
            yz = rng.uniform(c.perturb_translation_min_m, c.perturb_translation_max_m, 2)
            yz *= rng.choice([-1., 1.], 2)
            rpy = rng.uniform(c.perturb_rotation_min_deg, c.perturb_rotation_max_deg, 3)
            rpy *= rng.choice([-1., 1.], 3)
        self.sampled = {
            "requested_profile": c.profile, "profile": profile,
            "env_episode_seed": episode_seed, "seed_offset": c.seed_offset,
            "sampling_seed": seed, "tip_gap_m": gap,
            "axial_tip_gap_error_m": gap - c.target_tip_gap_m,
            "lateral_height_m": yz.tolist(), "roll_pitch_yaw_deg": rpy.tolist(),
            "rotation_convention": "socket goal local static XYZ roll/pitch/yaw",
        }
        self.result["sampled_parameters"] = dict(self.sampled)
        return self._outside_pose(gap, yz, euler2quat(*np.radians(rpy)))

    def _close_move(self, phase: str, destination: sapien.Pose,
                    q: np.ndarray | None = None) -> np.ndarray:
        """Interpolate orientation and clearance, recomputing root x per point."""
        c = self.config
        source = self.goal.inv() * _copy_pose(self.base.charger.pose)
        source_gap, destination_gap = self._gap(source), self._gap(destination)
        if min(source_gap, destination_gap) <= 0:
            raise _PolicyFailure(f"{phase}_must_start_and_end_outside_wall")
        error = _pose_error(self._tcp(source), self._tcp(destination))
        count = max(1, int(math.ceil(max(error["translation_mm"] / (1000 * c.axial_step_m),
                                          error["rotation_deg"] / c.se3_rotation_step_deg))))

        def points(n: int) -> list[sapien.Pose]:
            return [self._outside_pose(source_gap + t * (destination_gap - source_gap),
                        np.asarray(source.p)[1:] + t * (np.asarray(destination.p)[1:] - np.asarray(source.p)[1:]),
                        _slerp(source.q, destination.q, t))
                    for t in np.linspace(0., 1., n + 1)]

        # Root x is curved by the rotating pin corners, as is TCP through H.
        # Check actual Cartesian spacing, rather than only endpoint distance.
        while True:
            if count * c.waypoint_hold_steps > c.max_steps - self.steps:
                raise _PolicyFailure(f"{phase}_path_exceeds_remaining_step_budget")
            path = points(count)
            spacing = [_pose_error(self._tcp(a), self._tcp(b)) for a, b in zip(path, path[1:])]
            ratio = max(max(e["translation_mm"] / (1000 * c.axial_step_m),
                            e["rotation_deg"] / c.se3_rotation_step_deg) for e in spacing)
            if ratio <= 1. + 1e-5:
                break
            count = max(count + 1, int(math.ceil(count * ratio)))
        record = self._begin(phase, self._tcp(destination))
        record.update(waypoints=count, hold_steps_per_waypoint=c.waypoint_hold_steps,
                      max_tcp_spacing_mm=max(e["translation_mm"] for e in spacing),
                      max_rotation_spacing_deg=max(e["rotation_deg"] for e in spacing),
                      minimum_command_tip_gap_mm=min(self._gap(p) for p in path) * 1000.)
        q = self._qpos() if q is None else q
        for index, relative in enumerate(path[1:], 1):
            self._set_command(relative)
            q = self._exact_ik(self.target, q)
            for _ in range(c.waypoint_hold_steps):
                self._step(q)
            record["completed_waypoints"] = index
        self._end()
        return q

    def _settle_outside(self, phase: str, relative: sapien.Pose, min_steps: int,
                        q: np.ndarray | None = None) -> np.ndarray:
        return self._settle(phase, self._tcp(relative), min_steps, q=q)

    def _front_vertical_lift(self, target: sapien.Pose) -> np.ndarray:
        initial=_copy_pose(self.base.agent.tcp.pose)
        delta=np.asarray(target.p)-np.asarray(initial.p)
        error=_pose_error(initial,target)
        if np.linalg.norm(delta[:2])*1000.>.1 or error['rotation_deg']>.2 or delta[2]<=0:
            raise _PolicyFailure('front_spawn_not_below_safe_standoff_target')
        count=int(math.ceil(float(delta[2])/.005))
        record=self._begin('front_vertical_lift',target)
        record.update(waypoints=count,vertical_lift_m=float(delta[2]),
                      nominal_xy_displacement_mm=float(np.linalg.norm(delta[:2])*1000.),
                      max_command_spacing_mm=5.,planner='exactIK_vertical_waypoints',
                      hold_steps_per_waypoint=self.config.waypoint_hold_steps)
        q=self._qpos()
        for fraction in np.linspace(0.,1.,count+1)[1:]:
            self.target=sapien.Pose(np.asarray(initial.p)+fraction*delta,
                                    _slerp(initial.q,target.q,float(fraction)))
            self.command_tip_gap_m=self._gap(self.goal.inv()*self.target*self.h)
            q=self._exact_ik(self.target,q)
            for _ in range(self.config.waypoint_hold_steps):self._step(q)
        self._end()
        return q

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            episode = getattr(self.base, "_episode_record", None) or {}
            if episode.get("env_id") != "WMInsertionChargerFixedLink-v2":
                raise _PolicyFailure("requires_reset_fixed_child_charger_v2_environment")
            if self.base.agent.control_mode != "pd_joint_pos" or getattr(self.base, "num_envs", 1) != 1:
                raise _PolicyFailure("requires_one_native_pd_joint_pos_environment")
            self.planner = self.base._planner()
            initial_tcp = _copy_pose(self.base.agent.tcp.pose)
            self.h = initial_tcp.inv() * _copy_pose(self.base.charger.pose)
            self.goal = _copy_pose(self.base.goal_pose)
            half = np.asarray(self.base._peg_size, dtype=float)
            gap = float(self.base._peg_gap)
            self.pin_corners = np.asarray([[half[0] + sx * half[0], y + sy * half[1], sz * half[2]]
                for y in (-gap, gap) for sx, sy, sz in itertools.product((-1., 1.), repeat=3)])
            perturbed = self._sample(episode)
            near = self._outside_pose(self.config.target_tip_gap_m, np.zeros(2), np.array([1., 0., 0., 0.]))
            safe = self._outside_pose(self.config.transport_tip_gap_m, np.zeros(2), np.array([1., 0., 0., 0.]))
            final = self._tcp(sapien.Pose())
            self.result.update(frozen_achieved_grasp_pose=_raw(self.h), frozen_goal_pose=_raw(self.goal),
                initial_tcp_pose=_raw(initial_tcp), control_freq_hz=float(self.base.control_freq),
                pin_corners_charger_frame_m=self.pin_corners.tolist(),
                safe_transport_tcp_pose=_raw(self._tcp(safe)), preinsert_tcp_pose=_raw(self._tcp(near)),
                perturbation_tcp_pose=_raw(self._tcp(perturbed)), final_tcp_target_pose=_raw(final),
                final_stroke_m=-float(near.p[0]),
                push_axis_world=self.goal.to_transformation_matrix()[:3, 0].astype(float).tolist())
            self.metrics = self._snapshot()
            lift = sapien.Pose(np.asarray(initial_tcp.p) + [0., 0., self.config.lift_m], initial_tcp.q)
            if episode.get('spawn_layout')=='front_fixed':
                self.require_outside_wall=True
                q=self._front_vertical_lift(self._tcp(safe))
                self._settle('front_lift_settle',self._tcp(safe),self.config.prealign_min_hold_steps,q=q)
                self.result['lateral_transport_skipped']=True
            else:
                self._transport("lift_transport", lift)
                self._settle("lift_settle", lift, self.config.lift_min_hold_steps)
                self._transport("safe_standoff_transport", self._tcp(safe))
                self._settle("safe_standoff_settle", self._tcp(safe), self.config.prealign_min_hold_steps)
                self.result['lateral_transport_skipped']=False
            self.require_outside_wall = True
            q = self._close_move("near_wall_approach", near)
            q = self._settle_outside("near_wall_settle", near, self.config.prealign_min_hold_steps, q)
            if self.sampled["profile"] != "clean":
                q = self._close_move("perturbation_move", perturbed, q)
                q = self._settle_outside("perturbation_settle", perturbed,
                                         self.config.settle_consecutive_steps, q)
                q = self._close_move("recovery_align", near, q)
                q = self._settle_outside("recovery_settle", near, self.config.prealign_min_hold_steps, q)
            # The aligned 6D actual-pose AND actual-velocity gate above passes
            # before permitting any physical pin corner through the wall.
            self.require_outside_wall = False
            record = self._begin("axial_push", self._tcp(near))
            stroke = -float(near.p[0])
            count = int(math.ceil(stroke / self.config.axial_step_m - 1e-10))
            record.update(waypoints=count, stroke_m=stroke,
                          hold_steps_per_waypoint=self.config.waypoint_hold_steps)
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
                minimum_outside_command_tip_gap_mm=None if self.minimum_command_tip_gap_m is None
                else self.minimum_command_tip_gap_m * 1000.)
            if "control_freq_hz" in self.result:
                self.result["sim_seconds"] = self.steps / self.result["control_freq_hz"]
            if "frozen_goal_pose" in self.result:
                try:
                    self.result["final_charger_goal_error"] = _pose_error(
                        _sp(self.base.charger.pose), self.goal)
                except Exception as exc:
                    self.result["final_charger_goal_error_unavailable"] = f"{type(exc).__name__}: {exc}"
        return self.result


def run_near_wall_scripted_policy(env: Any, config: NearWallScriptedPolicyConfig | None = None,
                                  on_step: StepCallback | None = None) -> dict[str, Any]:
    return NearWallScriptedPolicy(config).run(env, on_step=on_step)


__all__ = ["NearWallScriptedPolicyConfig", "NearWallScriptedPolicy",
           "run_near_wall_scripted_policy"]
