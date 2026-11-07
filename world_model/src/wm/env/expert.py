"""Demonstration expert: Chongyu's scripted phases, started from any pose.

The expert stands in for a human teleoperator, so it may use simulator truth
(goal pose, pin geometry). Recorded model inputs never contain that truth.

Why a wrapper: Chongyu's RandomContinuousScriptedPolicy assumes the reset spawn.
It lifts and then transports laterally at 3 mm/s, which for some grasps exceeds
its fixed 800-step budget (audit 2026-11-07: 127 truncated attempts). Here

* moves between trajectories (to the safe standoff, to a start pose, retreats)
  are planner motions that are not recorded, like an environment reset;
* every recorded trajectory reuses his tested phases from the hand's current
  pose: `_approach` (stochastic perturbation), `_align` (exact-IK recovery with
  tracking compensation) and `_insert` (guarded axial push);
* two trajectory kinds are added on top of his phases: a contact trial (align,
  offset, straight push without his insertion guards, stop on jam) and random
  motion in front of the board (his outside-the-wall guards stay active).

His files are not edited; subclasses only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.continuous import (
    ContinuousScriptedPolicy,
    ContinuousScriptedPolicyConfig,
)
from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.randomized import _RandomExecution
from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted import _PolicyFailure, _copy_pose

from .robot import Robot
from .sampling import Offset

SAFE_GAP_M = 0.012          # Chongyu's transport_tip_gap_m
PUSH_SPEED_M_S = 0.003      # his insertion speed
JAM_LAG_M = 0.0015          # contact trial stops when the plug lags the command by 1.5 mm
INSERTED_MARGIN_M = 0.001   # inserted = within 1 mm of full depth
StepCallback = Callable[[str, np.ndarray, dict], object]


def _quat(rpy_deg) -> np.ndarray:
    return np.asarray(euler2quat(*np.radians(rpy_deg)), dtype=float)


class _Execution(_RandomExecution):
    """One of Chongyu's executions, initialised at the current pose (no spawn lift)."""

    def __init__(self, env, on_step: StepCallback | None, profile: str, seed_offset: int):
        cfg = ContinuousScriptedPolicyConfig(profile=profile, seed_offset=int(seed_offset))
        policy = ContinuousScriptedPolicy(cfg)
        policy.phase = "initialization"
        super().__init__(policy, env, on_step)
        self._setup()

    def _setup(self):
        # Same initialisation as _RandomExecution.run, without the spawn checks and lift.
        c = self.config
        self.planner = self.base._planner()
        self.h = _copy_pose(self.base.agent.tcp.pose).inv() * _copy_pose(self.base.charger.pose)
        self.goal = _copy_pose(self.base.goal_pose)
        self._initialize_pin_geometry()
        self._initialize_random(self.base._episode_record)
        self._initialize_command_guards()
        self._record_slot_guards()
        unit = np.array([1.0, 0.0, 0.0, 0.0])
        self.near = self._outside_pose(c.target_tip_gap_m, np.zeros(2), unit)
        self.safe = self._outside_pose(c.transport_tip_gap_m, np.zeros(2), unit)
        self.result.update(final_stroke_m=-float(self.near.p[0]))
        self.metrics = self._snapshot()
        self.require_outside_wall = True

    # ------------------------------------------------------------ truth helpers
    def relative(self) -> sapien.Pose:
        """Actual charger pose in the goal frame (truth)."""
        return self.goal.inv() * _copy_pose(self.base.charger.pose)

    def offset_pose(self, off: Offset) -> sapien.Pose:
        return self._outside_pose(off.tip_gap_m, np.asarray(off.yz_m), _quat(off.rpy_deg))

    # ------------------------------------------------------------ phases
    def _move_outside(self, phase: str, target: sapien.Pose, max_steps: int = 600):
        self._begin(phase, self._tcp(target))
        command = self.relative() if self.last_relative is None else self.last_relative
        for _ in range(max_steps):
            if self._pose_error_mm(command, target) < 1e-3:
                break
            command = self._advance(command, target, 0.9 * self.config.close_speed_m_s)
            self._send(command)
        else:
            raise _PolicyFailure(f"{phase}_did_not_reach_target")
        self._end()

    def _pose_error_mm(self, a: sapien.Pose, b: sapien.Pose) -> float:
        """TCP translation plus a rotation term (1 deg counted as 1 mm) between two goal-frame poses."""
        ta, tb = self._tcp(a), self._tcp(b)
        dr = (ta.inv() * tb).to_transformation_matrix()[:3, :3]
        angle = float(np.degrees(np.arccos(np.clip((np.trace(dr) - 1.0) / 2.0, -1.0, 1.0))))
        return float(np.linalg.norm(np.asarray(ta.p) - np.asarray(tb.p)) * 1000.0) + angle

    def _contact_push(self, start: sapien.Pose) -> dict:
        """Straight push along the goal axis from `start` without insertion guards."""
        self.require_outside_wall, self.insertion_active = False, False
        dt = float(self.base.control_timestep)
        x = float(start.p[0])
        self._begin("contact_push", self._tcp(sapien.Pose([0.0, *start.p[1:]], start.q)))
        seed = self._qpos()
        outcome = "inserted"
        try:
            while x < 0.0:
                actual_x = float(self.relative().p[0])
                if x - actual_x > JAM_LAG_M:
                    outcome = "jammed"
                    break
                x = min(0.0, x + PUSH_SPEED_M_S * dt)
                q = self._exact_ik(self._tcp(sapien.Pose([x, *start.p[1:]], start.q)), seed)
                q[:7] = q[:7].astype(np.float32).astype(float)
                self._step(q)
                seed = q
            for _ in range(10):  # let the arm settle at the last command
                self._step(seed)
        except _PolicyFailure as exc:  # contact force / penetration guards count as a jam
            outcome = f"guard:{str(exc)[:60]}"
        self._end()
        depth_left = -float(self.relative().p[0])
        inserted = depth_left <= INSERTED_MARGIN_M
        return {"outcome": "inserted" if inserted else outcome, "depth_left_mm": depth_left * 1000.0}

    # ------------------------------------------------------------ public entry
    def execute(self, kind: str, **kw) -> dict:
        info = {}
        try:
            if kind in ("insert", "perturbed"):
                if kind == "perturbed":
                    self._approach(self.safe)
                self._align(self.near)
                self._insert()
            elif kind == "contact":
                self._align(self.near)
                target = self.offset_pose(kw["offset"])
                self._move_outside("contact_offset", target)
                info = self._contact_push(target)
            elif kind == "random":
                for i, wp in enumerate(kw["waypoints"]):
                    self._move_outside(f"random_motion_{i}", self.offset_pose(wp))
            else:
                raise ValueError(kind)
            self.result["completed"] = True
        except _PolicyFailure as exc:
            self.result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        self.result.update(steps=self.steps, last_phase=self.policy.phase, **info)
        return self.result


class Expert:
    """Places the hand (not recorded) and runs recorded expert trajectories."""

    def __init__(self, robot: Robot):
        self.robot = robot

    def _execution(self, on_step=None, profile: str = "clean", seed_offset: int = 0) -> _Execution:
        return _Execution(self.robot.env, on_step, profile, seed_offset)

    # ------------------------------------------------- unrecorded placement
    def _tcp_T(self, ex: _Execution, rel: sapien.Pose) -> np.ndarray:
        return ex._tcp(rel).to_transformation_matrix()

    def to_safe_from_spawn(self) -> bool:
        """After reset: lift in world z at the spawn XY, then move to the aligned safe standoff."""
        ex = self._execution()
        safe = self._tcp_T(ex, ex.safe)
        lift = self.robot.hand_pose().copy()
        lift[2, 3] = safe[2, 3]
        return self.robot.move_to(lift) and self.robot.move_to(safe)

    def retreat(self) -> bool:
        """Back out along the insertion axis to the safe gap, keeping lateral offset and attitude."""
        ex = self._execution()
        rel = ex.relative()
        out = ex._outside_pose(SAFE_GAP_M, np.asarray(rel.p[1:]), np.asarray(rel.q))
        if float(rel.p[0]) >= float(out.p[0]):  # already further out than the safe gap
            return True
        return self.robot.move_to(self._tcp_T(ex, out))

    def move_to_offset(self, off: Offset) -> bool:
        """Retreat, then go to the start; if the direct planner path is refused (IK branch
        jump), go through the aligned safe standoff first."""
        ex = self._execution()
        if not self.retreat():
            return False
        target = self._tcp_T(ex, ex.offset_pose(off))
        if self.robot.move_to(target):
            return True
        return self.robot.move_to(self._tcp_T(ex, ex.safe)) and self.robot.move_to(target)

    # ------------------------------------------------- recorded trajectories
    def run(self, kind: str, on_step: StepCallback, seed_offset: int, **kw) -> dict:
        profile = "mixed" if kind == "perturbed" else "clean"
        return self._execution(on_step, profile, seed_offset).execute(kind, **kw)
