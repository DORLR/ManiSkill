"""State-based insertion environment for the PPO baseline (gymnasium; trained with Stable-Baselines3).

PRIVILEGED baseline: the observation contains the hand's offset from the seated pose H* of the grasp (from FK and
the grasp's reference insertion), which a vision policy does not get. It tells how well a learner does when the
geometry is handed to it; its numbers are an upper reference, not a competitor under the same information.

One step = one decision (0.4 s = 8 control steps at 20 Hz), as the policies and the world model.
  observation  joints (7) [rad]; their change over the last decision (7) [mrad]; hand offset from H* in the insertion
               axis frame: along, lateral x, lateral y [mm]; rotation from H* as a rotation vector [deg]   (20 numbers)
  action       7 numbers in [-1, 1]: the joint target for the decision = measured joints + action * per-joint scale
               (the 99.9th percentile of the demonstrations' per-decision joint motion), held for 8 control steps
  reward       the world model's reward (wm/tdmpc/wmdata320.py): FK depth gained [mm] + shaping * (gamma * phi' - phi)
               + 10 when seated (1.5 mm, 4 deg); the potential counts as 0 at a terminal state
  episode end  seated (terminal); hand more than 6 cm from where it started (terminal); a stall of the hand (8
               decisions, < 0.2 mm) or 150 decisions (truncated), as the closed-loop test of the policies
Starts: the policies' start box (collect/branch.py sample_start), placed with FK only after the grasp's reference
insertion is run once (as the closed-loop test does). A worker stays on one grasp for `episodes_per_grasp` episodes.
The simulator's truth is read only into `info` at the end of an episode (success by the one criterion), never into
the observation or the reward.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..collect import branch as B
from ..collect.branch import Brancher
from ..collect.explore import truth_outcome
from ..data.recorder import _truth_rel
from ..env.expert import Expert
from ..env.robot import Robot
from ..tdmpc import wmdata320 as WD
from ..tdmpc.demo320 import is_success

EXECUTE, MAX_DECISIONS = WD.EXECUTE, 150
LEFT_M = 0.06


def _rotvec_deg(R: np.ndarray) -> np.ndarray:
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return np.degrees(a) * w


class StateInsertEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grasps: list[dict], act_scale: np.ndarray, seed: int, shaping: float, gamma: float,
                 episodes_per_grasp: int = 16, max_decisions: int = MAX_DECISIONS):
        super().__init__()
        self.grasps, self.scale = list(grasps), np.asarray(act_scale, np.float64)
        self.rng = np.random.default_rng(seed)
        self.shaping, self.gamma, self.per_grasp, self.max_decisions = shaping, gamma, episodes_per_grasp, max_decisions
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.robot, self.ex, self.g, self.done_on_grasp = None, None, None, 0

    # ---- grasp and start
    def _load(self, g: dict) -> bool:
        if self.robot is None or self.robot.port_type != g["port"]:
            if self.robot is not None:
                self.robot.env.close()
            self.robot = Robot(g["port"])
        r = self.robot
        r.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        expert = Expert(r)
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            return False
        self.ex = Brancher(r, r.hand_pose(), np.asarray(g["axis"]))
        self.meta = {"H_star": self.ex.H_star, "axis": np.asarray(g["axis"]), "pin_length_m": self.ex.L}
        return self.ex.remember_standoff()

    def place(self, start: dict | None = None) -> bool:
        for _ in range(B.PLACEMENT_ATTEMPTS):
            s = start if start is not None else B.sample_start(self.rng)
            if self.ex.place(s) == "ok":
                return True
            if start is not None:
                return False
        return False

    def use_grasp(self, g: dict) -> bool:
        """Switch to a given grasp (evaluation); False if its reference insertion or stand-off fails."""
        self.g, self.done_on_grasp = g, 0
        return self._load(g)

    # ---- state, labels
    def _labels(self):
        lab = WD.decision_labels(self.meta, self.robot.hand_pose()[None], np.array([0]))
        return {k: v[0] for k, v in lab.items()}

    def _obs(self, lab) -> np.ndarray:
        q = self.robot.qpos()[:7]
        R = self.ex.H_star[:3, :3].T @ self.robot.hand_pose()[:3, :3]
        return np.concatenate([q, (q - self.q_prev) * 1000, lab["probe"][:3], _rotvec_deg(R)]).astype(np.float32)

    # ---- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            if self.ex is None or self.done_on_grasp >= self.per_grasp:
                self.g, self.done_on_grasp = self.grasps[int(self.rng.integers(len(self.grasps)))], 0
                if not self._load(self.g):
                    self.ex = None
                    continue
            if self.place():
                break
            self.done_on_grasp += 1
        self.done_on_grasp += 1
        return self.start_episode(), {}

    def start_episode(self) -> np.ndarray:
        """After a placement: the episode's bookkeeping and the first observation."""
        self.k, self.q_prev = 0, self.robot.qpos()[:7].copy()
        self.hands = [self.robot.hand_pose()[:3, 3].copy()]
        self.lab = self._labels()
        return self._obs(self.lab)

    def step(self, action):
        r = self.robot
        q = r.qpos()[:7].copy()
        target = q + np.clip(np.asarray(action, np.float64), -1, 1) * self.scale
        for _ in range(EXECUTE):
            r._command(target, 1)
        self.k += 1
        self.q_prev = q
        lab = self._labels()
        rew = (lab["depth"] - self.lab["depth"]) * 1000 + self.shaping * (self.gamma * lab["phi"] - self.lab["phi"])
        hand = r.hand_pose()[:3, 3].copy()
        self.hands.append(hand)
        terminated, truncated, why = False, False, ""
        if lab["seated"]:
            terminated, why = True, "seated"
            rew += WD.BONUS
        elif np.linalg.norm(hand - self.hands[0]) > LEFT_M:
            terminated, why = True, "left"
        elif len(self.hands) > B.STALL_DECISIONS and np.linalg.norm(self.hands[-1] - self.hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
            truncated, why = True, "stalled"
        elif self.k >= self.max_decisions:
            truncated, why = True, "time"
        if terminated:
            rew -= self.shaping * self.gamma * lab["phi"]                # potential 0 at a terminal state
        self.lab = lab
        info = {}
        if terminated or truncated:
            for _ in range(4):                                          # the closed-loop test's final hold
                r._command(target, 1)
            t = truth_outcome(_truth_rel(r))
            info = {"is_success": is_success(t), "stopped": why, "decisions": self.k, "seated_fk": bool(self._labels()["seated"]),
                    **{k: float(t[k]) for k in ("depth_left_mm", "angle_deg", "lateral_mm")}}
        return self._obs(lab), float(rew), terminated, truncated, info


__all__ = ["StateInsertEnv"]
