"""Privileged scripted baseline for the fixed-child Charger v2 environment.

The caller resets the environment first.  This policy freezes the achieved
TCP-to-charger transform H and native socket goal G, then uses targets
``G * Trans(-distance, 0, 0) * H.inv()`` throughout the insertion.  MPlib
transports the held charger in free space; exact IK supplies the alignment and
slow axial waypoints.  All motion is executed by the existing finite-force
``pd_joint_pos`` controller through ``env.step``.  No pose/joint-state setters,
contact-dependent correction, rendering, or image/file I/O are used here.

``completed`` describes completion of the commanded motion and convergence
gates.  It is not a physical-insertion or numerical-contact-validity verdict;
the evaluation runner must audit physical pin geometry and contact stability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

import mplib
import numpy as np
import sapien

from mani_skill.utils import common

from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _pose_error, _raw, _sp


# Called once after EVERY completed env.step.  A nonempty returned string, or
# an exception raised by the callback, aborts and is retained in the result.
StepCallback = Callable[[str, np.ndarray, dict[str, Any]], str | None]


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    """Distances are metres; convergence tolerances name their own units."""

    max_steps: int = 800
    lift_m: float = 0.08
    preinsert_distance_m: float = 0.05
    axial_step_m: float = 0.0005
    waypoint_hold_steps: int = 3
    lift_min_hold_steps: int = 5
    prealign_min_hold_steps: int = 20
    final_min_hold_steps: int = 60
    settle_timeout_steps: int = 100
    settle_consecutive_steps: int = 5
    position_tolerance_mm: float = 0.3
    orientation_tolerance_deg: float = 0.2
    tcp_linear_velocity_tolerance_m_s: float = 0.0005
    tcp_angular_velocity_tolerance_deg_s: float = 0.5
    arm_joint_velocity_tolerance_rad_s: float = 0.02
    max_socket_contact_force_n: float = 80.0
    max_socket_penetration_mm: float = 0.5
    ik_threshold: float = 1e-5
    ik_max_joint_change_rad: float = 0.15
    screw_attempts: int = 2
    transport_rrt_fallback: bool = True
    transport_rrt_planning_seconds: float = 2.0

    def __post_init__(self):
        integer_fields = (
            "max_steps", "waypoint_hold_steps", "lift_min_hold_steps",
            "prealign_min_hold_steps", "final_min_hold_steps",
            "settle_timeout_steps", "settle_consecutive_steps", "screw_attempts",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if name not in integer_fields and name != "transport_rrt_fallback":
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.max_steps > 800:
            raise ValueError("The scripted baseline is bounded to at most 800 steps")
        if self.waypoint_hold_steps < 3:
            raise ValueError("Each axial IK waypoint requires at least 3 control steps")
        if self.axial_step_m > 0.0005:
            raise ValueError("Axial IK spacing must be at most 0.5 mm")
        if self.preinsert_distance_m < self.axial_step_m:
            raise ValueError("preinsert_distance_m must be at least axial_step_m")
        if max(self.lift_min_hold_steps, self.prealign_min_hold_steps,
               self.final_min_hold_steps, self.settle_consecutive_steps) > self.settle_timeout_steps:
            raise ValueError("All convergence holds must fit settle_timeout_steps")


class _PolicyFailure(RuntimeError):
    pass


def _array(value: Any) -> np.ndarray:
    return np.asarray(common.to_numpy(value), dtype=np.float64).reshape(-1)


def _copy_pose(value: Any) -> sapien.Pose:
    value = _sp(value)
    return sapien.Pose(p=np.asarray(value.p).copy(), q=np.asarray(value.q).copy())


def _tcp_velocity(tcp: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read actual instantaneous world-frame velocity, never planner velocity.

    The CPU ManiSkill Link wraps one SAPIEN articulation-link component.  Both
    public wrapper properties and the native component accessor are supported.
    Missing velocity data fails the gate rather than assuming zero velocity.
    """
    body = tcp
    if not (hasattr(body, "linear_velocity") and hasattr(body, "angular_velocity")):
        objects = getattr(tcp, "_objs", ())
        if len(objects) != 1:
            raise _PolicyFailure("actual_tcp_velocity_unavailable")
        body = objects[0]
    linear, angular = _array(body.linear_velocity), _array(body.angular_velocity)
    if linear.shape != (3,) or angular.shape != (3,):
        raise _PolicyFailure("actual_tcp_velocity_shape_invalid")
    return linear, angular


class ChargerV2ScriptedPolicy:
    """Reusable baseline; ``run`` never resets, closes, or renders ``env``.

    Example::

        env.reset(seed=910200)
        result = ChargerV2ScriptedPolicy().run(env, on_step=capture_and_audit)

    ``on_step(phase, action, info)`` receives the executed action and native
    step info, with scalar endpoint diagnostics in ``info['scripted_policy']``.
    The hook can set an external probe phase for the *next* step; to label
    physics substeps of the current step, read ``policy.phase`` from the probe.
    ``phase`` is assigned before ``env.step`` and remains the last phase after
    return.  A wrapper horizon below this policy's bound is honored as failure.
    """

    def __init__(self, config: ScriptedPolicyConfig | None = None):
        self.config = config or ScriptedPolicyConfig()
        self.phase = "not_started"

    def run(self, env: Any, on_step: StepCallback | None = None) -> dict[str, Any]:
        self.phase = "initialization"
        execution = _Execution(self, env, on_step)
        return execution.run()


class _Execution:
    def __init__(self, policy: ChargerV2ScriptedPolicy, env: Any,
                 on_step: StepCallback | None):
        self.policy, self.config = policy, policy.config
        self.env, self.base, self.on_step = env, env.unwrapped, on_step
        self.steps = 0
        self.stage: dict[str, Any] | None = None
        self.target: sapien.Pose | None = None
        self.metrics: dict[str, Any] = {}
        self.previous_endpoint_tcp: sapien.Pose | None = None
        self.result: dict[str, Any] = {
            "policy": "charger_v2_fixed_h_scripted_20261102",
            "completed": False, "failure_reason": None, "steps": 0,
            "config": asdict(self.config), "stages": [],
            "controller": "native_pd_joint_pos",
            "goal_feedback": "frozen_G_and_achieved_H; no contact correction",
            "native_success_handling": "record loose5mm success; continue to full target and convergence gates; no autoreset",
            "convergence_velocity_measurement": "actual TCP endpoint delta/control_dt, each of5consecutive endpoints; raw PhysX velocities also logged",
        }

    def _qpos(self) -> np.ndarray:
        q = _array(self.base.agent.robot.get_qpos())
        if q.shape != (9,) or not np.isfinite(q).all():
            raise _PolicyFailure("expected_one_finite_native_9dof_qpos")
        return q

    def _snapshot(self) -> dict[str, Any]:
        tcp = _copy_pose(self.base.agent.tcp.pose)
        linear, angular = _tcp_velocity(self.base.agent.tcp)
        qpos, qvel = self._qpos(), _array(self.base.agent.robot.get_qvel())
        if qvel.shape != (9,) or not np.isfinite(np.r_[qpos, qvel, linear, angular, _raw(tcp)]).all():
            raise _PolicyFailure("non_finite_or_invalid_actual_state")
        data = {
            "actual_tcp_pose": _raw(tcp),
            "tcp_linear_velocity_m_s": float(np.linalg.norm(linear)),
            "tcp_angular_velocity_deg_s": float(np.degrees(np.linalg.norm(angular))),
            "arm_joint_velocity_rad_s": float(np.abs(qvel[:7]).max()),
            "socket_contact_force_n": 0.0,
            "socket_penetration_mm": 0.0,
        }
        endpoint_error=_pose_error(tcp,self.previous_endpoint_tcp) if self.previous_endpoint_tcp is not None else {'translation_mm':0.,'rotation_deg':0.}
        dt=float(self.base.control_timestep)
        data['tcp_endpoint_linear_velocity_m_s']=endpoint_error['translation_mm']/1000./dt
        data['tcp_endpoint_angular_velocity_deg_s']=endpoint_error['rotation_deg']/dt
        self.previous_endpoint_tcp=_copy_pose(tcp)
        # Standalone endpoint guard. The evaluation callback additionally audits
        # all100Hz substeps; this force is impulse/dt, not a changed drive cap.
        for contact in self.base.scene.get_contacts():
            if not any('receptacle' in body.entity.name for body in contact.bodies):
                continue
            impulse=np.sum([np.asarray(point.impulse) for point in contact.points],axis=0)
            data['socket_contact_force_n']+=float(np.linalg.norm(impulse)*self.base.sim_freq)
            data['socket_penetration_mm']=max(data['socket_penetration_mm'],max(
                (max(0.,-float(point.separation)*1000.) for point in contact.points),default=0.))
        if self.target is not None:
            data["tcp_error"] = _pose_error(tcp, self.target)
        return data

    def _begin(self, phase: str, target: sapien.Pose | None = None) -> dict[str, Any]:
        self.policy.phase, self.target = phase, target
        self.stage = {"phase": phase, "start_step": self.steps, "completed": False}
        if target is not None:
            self.stage["target_tcp_pose"] = _raw(target)
            self.stage["before"] = _pose_error(_sp(self.base.agent.tcp.pose), target)
        self.result["stages"].append(self.stage)
        return self.stage

    def _end(self):
        assert self.stage is not None
        self.stage.update(completed=True, end_step=self.steps,
                          steps=self.steps - self.stage["start_step"], after=dict(self.metrics))

    def _step(self, q: np.ndarray):
        if self.steps >= self.config.max_steps:
            raise _PolicyFailure(f"policy_step_budget_{self.config.max_steps}")
        arm = _array(q)[:7]
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise _PolicyFailure("invalid_arm_joint_command")
        action = np.r_[arm, -1.0].astype(np.float32)
        _, _, terminated, truncated, native_info = self.env.step(action)
        self.steps += 1
        state_error = None
        try:
            self.metrics = self._snapshot()
        except Exception as exc:
            state_error = exc
            self.metrics = {"state_error": f"{type(exc).__name__}: {exc}"}
        info = dict(native_info)
        info["scripted_policy"] = {
            "step": self.steps, "phase": self.policy.phase, **self.metrics,
            "environment_terminated": bool(_array(terminated).any()),
            "environment_truncated": bool(_array(truncated).any()),
        }
        if self.on_step is not None:
            reason = self.on_step(self.policy.phase, action.copy(), info)
            if isinstance(reason, str) and reason:
                raise _PolicyFailure(f"callback_abort: {reason}")
        if state_error is not None:
            raise state_error
        if self.metrics['socket_contact_force_n'] > self.config.max_socket_contact_force_n:
            raise _PolicyFailure('socket_contact_force_guard')
        if self.metrics['socket_penetration_mm'] > self.config.max_socket_penetration_mm:
            raise _PolicyFailure('socket_penetration_guard')
        if info["scripted_policy"]["environment_truncated"]:
            raise _PolicyFailure("environment_truncated")
        # This exact task declares success within5mm, before full insertion.
        # Like its native expert, finish the explicit trajectory after that flag.
        # Actual fail/non-success termination and TimeLimit remain aborts.
        success=bool(_array(native_info.get('success',False)).any())
        failed=bool(_array(native_info.get('fail',False)).any())
        if info["scripted_policy"]["environment_terminated"] and (failed or not success):
            raise _PolicyFailure("environment_terminated")

    def _transport(self, phase: str, target: sapien.Pose) -> np.ndarray:
        record = self._begin(phase, target)
        record["planning_attempts"] = []
        started = time.perf_counter()
        qstart = self._qpos()
        target_mplib = mplib.Pose(target.p, target.q)
        path = None
        for _ in range(self.config.screw_attempts):
            candidate = self.planner.plan_screw(target_mplib, qstart,
                time_step=self.base.control_timestep)
            record["planning_attempts"].append("screw:" + str(candidate["status"]))
            if candidate["status"] == "Success":
                path, record["planner"] = candidate, "native_mplib_screw"
                break
        if path is None and self.config.transport_rrt_fallback:
            candidate = self.planner.plan_pose(target_mplib, qstart,
                time_step=self.base.control_timestep,
                planning_time=self.config.transport_rrt_planning_seconds)
            record["planning_attempts"].append("rrt:" + str(candidate["status"]))
            if candidate["status"] == "Success":
                path, record["planner"] = candidate, "native_mplib_rrt_free_transport"
        record["planning_wall_seconds"] = time.perf_counter() - started
        if path is None or len(path.get("position", ())) == 0:
            raise _PolicyFailure(f"{phase}_planning_failed")
        record["planned_steps"] = len(path["position"])
        if record["planned_steps"] > self.config.max_steps - self.steps:
            raise _PolicyFailure(f"{phase}_path_exceeds_remaining_step_budget")
        for q in path["position"]:
            self._step(q)
        self._end()
        return self._qpos()

    def _exact_ik(self, target: sapien.Pose, seed: np.ndarray) -> np.ndarray:
        local = self.planner._transform_goal_to_wrt_base(mplib.Pose(target.p, target.q))
        status, solution = self.planner.IK(local, seed, n_init_qpos=1,
            threshold=self.config.ik_threshold, return_closest=True)
        if solution is None:
            raise _PolicyFailure(f"{self.policy.phase}_ik_failed: {status}")
        solution = _array(solution)
        if solution.size < 7 or not np.isfinite(solution).all():
            raise _PolicyFailure(f"{self.policy.phase}_invalid_ik_solution")
        if float(np.abs(solution[:7] - seed[:7]).max()) > self.config.ik_max_joint_change_rad:
            raise _PolicyFailure(f"{self.policy.phase}_ik_branch_jump")
        # Keep a full native 9-DoF seed even if an MPlib build returns only the
        # seven move-group joints. Finger commands remain the native close=-1.
        q = seed.copy()
        q[:7] = solution[:7]
        return q

    def _converged(self) -> bool:
        c, m = self.config, self.metrics
        return bool(
            m["tcp_error"]["translation_mm"] <= c.position_tolerance_mm
            and m["tcp_error"]["rotation_deg"] <= c.orientation_tolerance_deg
            and m["tcp_endpoint_linear_velocity_m_s"] <= c.tcp_linear_velocity_tolerance_m_s
            and m["tcp_endpoint_angular_velocity_deg_s"] <= c.tcp_angular_velocity_tolerance_deg_s
            and m["arm_joint_velocity_rad_s"] <= c.arm_joint_velocity_tolerance_rad_s
        )

    def _settle(self, phase: str, target: sapien.Pose, min_steps: int,
                q: np.ndarray | None = None) -> np.ndarray:
        record = self._begin(phase, target)
        q = self._exact_ik(target, self._qpos()) if q is None else q
        record.update(min_hold_steps=min_steps, timeout_steps=self.config.settle_timeout_steps,
                      consecutive_steps_required=self.config.settle_consecutive_steps,
                      exact_ik_command_arm_qpos=q[:7].tolist())
        consecutive = 0
        for index in range(self.config.settle_timeout_steps):
            self._step(q)
            consecutive = consecutive + 1 if self._converged() else 0
            if index + 1 >= min_steps and consecutive >= self.config.settle_consecutive_steps:
                record["consecutive_converged_steps"] = consecutive
                self._end()
                return q
        record["consecutive_converged_steps"] = consecutive
        raise _PolicyFailure(f"{phase}_actual_pose_or_velocity_convergence_timeout")

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            episode = getattr(self.base, "_episode_record", None) or {}
            if episode.get("env_id") != "WMInsertionChargerFixedLink-v2":
                raise _PolicyFailure("requires_reset_fixed_child_charger_v2_environment")
            if self.base.agent.control_mode != "pd_joint_pos":
                raise _PolicyFailure("requires_native_pd_joint_pos_control_mode")
            if getattr(self.base, "num_envs", 1) != 1:
                raise _PolicyFailure("requires_one_environment")
            self.planner = self.base._planner()
            initial_tcp = _copy_pose(self.base.agent.tcp.pose)
            h = initial_tcp.inv() * _copy_pose(self.base.charger.pose)
            goal = _copy_pose(self.base.goal_pose)
            self.result.update(frozen_achieved_grasp_pose=_raw(h), frozen_goal_pose=_raw(goal),
                               initial_tcp_pose=_raw(initial_tcp),
                               control_freq_hz=float(self.base.control_freq))
            self.metrics = self._snapshot()
            lift = sapien.Pose(p=np.asarray(initial_tcp.p) + [0., 0., self.config.lift_m],
                               q=initial_tcp.q)
            preinsert = goal * sapien.Pose([-self.config.preinsert_distance_m, 0., 0.]) * h.inv()
            final = goal * h.inv()
            self.result.update(preinsert_tcp_pose=_raw(preinsert), final_tcp_target_pose=_raw(final),
                push_axis_world=goal.to_transformation_matrix()[:3, 0].astype(float).tolist())

            self._transport("lift_transport", lift)
            self._settle("lift_settle", lift, self.config.lift_min_hold_steps)
            self._transport("preinsert_transport", preinsert)
            q = self._settle("prealign_settle", preinsert, self.config.prealign_min_hold_steps)

            record = self._begin("axial_push", preinsert)
            count = int(math.ceil(self.config.preinsert_distance_m / self.config.axial_step_m - 1e-10))
            record.update(waypoints=count, max_spacing_mm=self.config.axial_step_m * 1000.,
                          hold_steps_per_waypoint=self.config.waypoint_hold_steps,
                          feedback="IK seeded by previous command; fixed axis; no contact correction")
            completed = 0
            for index in range(1, count + 1):
                distance = max(0., self.config.preinsert_distance_m - index * self.config.axial_step_m)
                self.target = goal * sapien.Pose([-distance, 0., 0.]) * h.inv()
                q = self._exact_ik(self.target, q)
                for _ in range(self.config.waypoint_hold_steps):
                    self._step(q)
                completed = index
                record["completed_waypoints"] = completed
            self._end()
            self._settle("final_settle", final, self.config.final_min_hold_steps, q=q)
            self.result["completed"] = True
        except Exception as exc:
            self.result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            if self.stage is not None and not self.stage["completed"]:
                self.stage.update(end_step=self.steps, steps=self.steps-self.stage["start_step"],
                                  after=dict(self.metrics), failure_reason=self.result["failure_reason"])
        finally:
            self.result.update(steps=self.steps, last_phase=self.policy.phase,
                               wall_seconds=time.perf_counter()-started, final_metrics=dict(self.metrics))
            if "control_freq_hz" in self.result:
                self.result["sim_seconds"] = self.steps/self.result["control_freq_hz"]
            if "frozen_goal_pose" in self.result:
                try:
                    self.result["final_charger_goal_error"] = _pose_error(
                        _sp(self.base.charger.pose), _sp(self.result["frozen_goal_pose"]))
                except Exception as exc:
                    self.result["final_charger_goal_error_unavailable"] = f"{type(exc).__name__}: {exc}"
        return self.result


def run_scripted_policy(env: Any, config: ScriptedPolicyConfig | None = None,
                        on_step: StepCallback | None = None) -> dict[str, Any]:
    """Run the baseline on one already-reset episode, returning plain records."""
    return ChargerV2ScriptedPolicy(config).run(env, on_step=on_step)


__all__ = ["ScriptedPolicyConfig", "ChargerV2ScriptedPolicy", "run_scripted_policy", "StepCallback"]
