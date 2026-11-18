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
  episode end  training: seated (terminal); hand more than 6 cm from where it started (terminal); a stall of the hand
               (8 decisions, < 0.2 mm) or 150 decisions (truncated).
               test (stop_on_seated=False): no stop on "seated", exactly the stops of the policies' closed-loop test
               in its "free" mode (left 6 cm, stall, 150 decisions), so that the numbers compare.
Starts: the policies' start box (collect/branch.py sample_start), placed with FK only after the grasp's reference
insertion is run once (as the closed-loop test does). A worker gets whole ports and stays on one grasp for
`episodes_per_grasp` episodes; the next grasp comes from the same port while it has untried ones, so the simulator
(one per port) is rebuilt rarely. A grasp whose reference insertion or stand-off fails is dropped, as is one with no
successful placement in a whole turn, or with ERRORS_IN_A_ROW (5) simulator errors with no successful decision between
them (a grasp whose load, placement or first decision always fails; a reload after an error spans hundreds of
control steps, so sporadic errors could otherwise hit two reloads in a row); only a successful decision resets the
count; every drop is printed. After a simulator error the same grasp is loaded again and its turn goes on (a new
turn when its episodes were used up). Training (catch_errors=True): an exception of the simulator in reset or step
is caught: the simulator is closed and rebuilt at the next reset, the episode is cut (truncated; is_success None, so
that the logged success rate leaves it out); an exception in the final hold or truth read of an episode that ended
keeps the decision's real result (truth missing in info); only a worker left without any grasp raises RuntimeError
(which ends the training).
A worker's first reset waits `start_delay_s` (the simulators of the workers are then not built at the same moment).
`info` at the end also holds the first decision at which FK said "seated" (None if never), so that a test can count
attempts that were seated once but failed at the end.
The simulator's truth is read only into `info` at the end of an episode (success by the one criterion), never into
the observation or the reward.
"""

from __future__ import annotations

import sys
import time

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
ERRORS_IN_A_ROW = 5


def _rotvec_deg(R: np.ndarray) -> np.ndarray:
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return np.degrees(a) * w


class StateInsertEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grasps: list[dict], act_scale: np.ndarray, seed: int, shaping: float, gamma: float,
                 episodes_per_grasp: int = 16, max_decisions: int = MAX_DECISIONS, stop_on_seated: bool = True,
                 catch_errors: bool = True, start_delay_s: float = 0.0):
        super().__init__()
        self.grasps, self.scale = list(grasps), np.asarray(act_scale, np.float64)
        self.stop_on_seated, self.fresh, self.catch, self.start_delay_s = stop_on_seated, [], catch_errors, start_delay_s
        self.errors, self.placed_on_grasp, self.last_obs, self.reload = {}, 0, np.zeros(20, np.float32), False
        self.rng = np.random.default_rng(seed)
        self.shaping, self.gamma, self.per_grasp, self.max_decisions = shaping, gamma, episodes_per_grasp, max_decisions
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.robot, self.ex, self.g, self.done_on_grasp = None, None, None, 0

    # ---- grasp and start
    def _load(self, g: dict) -> bool:
        self.ex = None
        if self.robot is None or self.robot.port_type != g["port"]:
            if self.robot is not None:
                old, self.robot = self.robot, None
                try:
                    old.env.close()                                        # the old port's simulator: not this grasp's error
                except Exception as exc:
                    print(f"state_env: closing the simulator: {type(exc).__name__}: {exc}"[:300], file=sys.stderr, flush=True)
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

    # ---- errors
    def _drop(self, g: dict, why: str):
        self.grasps = [h for h in self.grasps if h is not g]
        print(f"state_env: grasp {g['grasp']} dropped ({why}); {len(self.grasps)} left", file=sys.stderr, flush=True)
        self.fresh = [h for h in self.fresh if h is not g]
        if not self.grasps:
            raise RuntimeError("no usable grasp is left in this worker")

    def _broken(self, where: str, exc: Exception):
        """A simulator error in training: close the simulator (rebuilt at the next reset: the same grasp, or the next
        one when its turn is over); ERRORS_IN_A_ROW errors with no successful decision between them drop the grasp."""
        if not self.catch:
            raise exc
        print(f"state_env: {where} on {self.g and self.g['grasp']}: {type(exc).__name__}: {exc}"[:400], file=sys.stderr, flush=True)
        old, self.robot, self.ex = self.robot, None, None
        try:
            if old is not None:
                old.env.close()
        except Exception:
            pass
        if self.g is not None:
            self.errors[self.g["grasp"]] = self.errors.get(self.g["grasp"], 0) + 1
            self.reload = self.errors[self.g["grasp"]] < ERRORS_IN_A_ROW  # the same grasp again, its turn goes on
            if not self.reload:
                self._drop(self.g, f"{ERRORS_IN_A_ROW} simulator errors in a row")

    # ---- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.start_delay_s:
            time.sleep(self.start_delay_s)
            self.start_delay_s = 0.0
        while True:
            if self.ex is None or self.done_on_grasp >= self.per_grasp:
                if self.reload and self.done_on_grasp < self.per_grasp:   # after a simulator error: same grasp, same turn
                    self.reload = False
                else:
                    self.reload = False
                    if self.ex is not None and self.placed_on_grasp == 0:  # a whole turn without one placement
                        self._drop(self.g, "no placement in a whole turn")
                    self.g, self.done_on_grasp, self.placed_on_grasp = self._next_grasp(), 0, 0
                try:
                    ok = self._load(self.g)
                except Exception as exc:
                    self._broken("load", exc)
                    continue
                if not ok:
                    self.ex = None
                    self._drop(self.g, "reference insertion or stand-off failed")
                    continue
            try:
                ok = self.place()
                obs = self.start_episode() if ok else None
            except Exception as exc:
                self._broken("place", exc)
                continue
            if ok:
                break
            self.done_on_grasp += 1
        self.done_on_grasp += 1
        self.placed_on_grasp += 1
        return obs, {}

    def _next_grasp(self) -> dict:
        """An untried grasp of the current port if there is one, else one of the port drawn next (a new cycle when all were tried)."""
        port = self.robot.port_type if self.robot is not None else (self.g["port"] if self.g is not None else None)
        if not self.fresh:
            self.fresh = list(self.grasps)
        same = [g for g in self.fresh if g["port"] == port]
        if not same:
            p = self.fresh[int(self.rng.integers(len(self.fresh)))]["port"]
            same = [g for g in self.fresh if g["port"] == p]
        g = same[int(self.rng.integers(len(same)))]
        self.fresh = [h for h in self.fresh if h is not g]
        return g

    def start_episode(self) -> np.ndarray:
        """After a placement: the episode's bookkeeping and the first observation."""
        self.k, self.q_prev, self.first_seated = 0, self.robot.qpos()[:7].copy(), None
        self.hands = [self.robot.hand_pose()[:3, 3].copy()]
        self.lab = self._labels()
        self.last_obs = self._obs(self.lab)
        return self.last_obs

    def step(self, action):
        try:
            out = self._step(action)
        except Exception as exc:
            self._broken("step", exc)                                  # raises outside training
            return self.last_obs, 0.0, False, True, {"is_success": None, "stopped": "error", "decisions": getattr(self, "k", 0),
                                                     "error": f"{type(exc).__name__}"}
        self.errors[self.g["grasp"]] = 0                                   # a successful decision: errors no longer in a row
        return out

    def _step(self, action):
        r = self.robot
        q = r.qpos()[:7].copy()
        target = (q + np.clip(np.asarray(action, np.float64), -1, 1) * self.scale).astype(np.float32).astype(np.float64)   # as the policies' test
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
            rew += WD.BONUS
            if self.first_seated is None:
                self.first_seated = self.k
        if lab["seated"] and self.stop_on_seated:
            terminated, why = True, "seated"
        elif np.linalg.norm(hand - self.hands[0]) > LEFT_M:
            terminated, why = True, "left"
        elif len(self.hands) > B.STALL_DECISIONS and np.linalg.norm(self.hands[-1] - self.hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
            truncated, why = True, "stalled"
        elif self.k >= self.max_decisions:
            truncated, why = True, "time"
        if terminated:
            rew -= self.shaping * self.gamma * lab["phi"]                # potential 0 at a terminal state
        self.lab = lab
        obs, info = self._obs(lab), {}                                  # the observation of the decision's end, before any hold
        self.last_obs = obs
        if terminated or truncated:
            try:
                for _ in range(4):                                      # the closed-loop test's final hold
                    r._command(target, 1)
                t = truth_outcome(_truth_rel(r))
                info = {"is_success": is_success(t), "stopped": why, "decisions": self.k, "seated_fk": bool(self._labels()["seated"]),
                        "first_seated_decision": self.first_seated,
                        **{k: float(t[k]) for k in ("depth_left_mm", "angle_deg", "lateral_mm")}}
            except Exception as exc:                                    # the decision itself stands
                self._broken("final hold", exc)                         # raises outside training
                info = {"is_success": None, "stopped": why, "decisions": self.k, "first_seated_decision": self.first_seated,
                        "error": f"{type(exc).__name__} in the final hold"}
        return obs, float(rew), terminated, truncated, info


__all__ = ["StateInsertEnv"]
