"""PoolVecEnv (wm/baselines/pool_vec_env.py) on toy environments with a slow reset, on a CPU, no simulator:
SubprocVecEnv's contract per slot (complete episodes, terminal_observation, TimeLimit.truncated, the next episode's
first observation), every engine's episodes played once each and in order, a second reset, the speed gain over
SubprocVecEnv, an engine's error (in a step, in a background reset, or the engine dying) reaching the learner and
the pool refusing further use, close() while an engine is inside a long reset, and PPO + VecMonitor + VecNormalize on
top.

    python world_model/tests/test_pool_vec_env.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.baselines.pool_vec_env import PoolVecEnv  # noqa: E402


class Toy(gym.Env):
    """obs = (engine, episode, t, 0); episode k lasts 3 + (engine + k) % 5 steps; it ends terminated when k is even,
    truncated when k is odd; reward = 10 * engine + t. Reset sleeps reset_s (the first one first_reset_s), a step
    step_s. fail_at / die_at: raise / end the process in that step (counted over the engine's life); reset_fail_at:
    raise in that reset (0 = the first)."""

    def __init__(self, engine, reset_s=0.0, step_s=0.0, fail_at=None, die_at=None, reset_fail_at=None, first_reset_s=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.engine, self.reset_s, self.step_s, self.fail_at = engine, reset_s, step_s, fail_at
        self.die_at, self.reset_fail_at, self.first_reset_s = die_at, reset_fail_at, first_reset_s
        self.k, self.t, self.steps = -1, 0, 0

    def _obs(self):
        return np.array([self.engine, self.k, self.t, 0], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(self.first_reset_s if self.k == -1 and self.first_reset_s is not None else self.reset_s)
        if self.reset_fail_at is not None and self.k + 1 == self.reset_fail_at:
            raise ValueError(f"toy reset failure in engine {self.engine}")
        self.k, self.t = self.k + 1, 0
        return self._obs(), {"engine": self.engine, "episode": self.k}

    def step(self, action):
        time.sleep(self.step_s)
        self.steps += 1
        if self.fail_at is not None and self.steps == self.fail_at:
            raise ValueError(f"toy failure in engine {self.engine}")
        if self.die_at is not None and self.steps == self.die_at:
            os._exit(3)
        self.t += 1
        end = self.t >= 3 + (self.engine + self.k) % 5
        return self._obs(), float(10 * self.engine + self.t), end and self.k % 2 == 0, end and self.k % 2 == 1, {}


def fn(i, **kw):
    return lambda: Toy(i, **kw)


def test_contract():
    n, m = 3, 7
    venv = PoolVecEnv([fn(i, reset_s=0.002 * (i % 3)) for i in range(m)], n)
    try:
        obs = venv.reset()
        cur = [tuple(o[:3].astype(int)) for o in obs]
        assert all(c[2] == 0 for c in cur)
        played = {i: [] for i in range(m)}                      # (episode, length) per engine, in the order finished
        for _ in range(400):
            obs, rew, done, infos = venv.step(np.zeros((n, 2), np.float32))
            for s in range(n):
                e, k, t = cur[s]
                if done[s]:
                    term = infos[s]["terminal_observation"].astype(int)
                    assert term[0] == e and term[1] == k and term[2] == t + 1, "terminal observation of the same episode"
                    assert term[2] == 3 + (e + k) % 5, "the episode ran to its end"
                    assert infos[s]["TimeLimit.truncated"] == (k % 2 == 1)
                    assert rew[s] == 10 * e + t + 1
                    played[e].append(k)
                    nxt = tuple(obs[s][:3].astype(int))
                    assert nxt[2] == 0, "the slot's next observation is a fresh episode's first"
                    assert venv.reset_infos[s] == {"engine": nxt[0], "episode": nxt[1]}
                    cur[s] = nxt
                else:
                    o = tuple(obs[s][:3].astype(int))
                    assert o == (e, k, t + 1) and rew[s] == 10 * e + t + 1 and not infos[s]["TimeLimit.truncated"]
                    assert "terminal_observation" not in infos[s]
                    cur[s] = o
        for e, ks in played.items():
            assert ks == list(range(len(ks))), f"engine {e}: episodes {ks} (each once, in order)"
        used = sum(len(v) > 0 for v in played.values())
        assert used == m, f"only {used} of {m} engines played"
        assert venv.env_is_wrapped(gym.Wrapper) == [False] * n
        assert sorted(venv.get_attr("engine")) == sorted(c[0] for c in cur)
        # a second reset: every slot starts a fresh episode, abandoned episodes are not resumed
        obs = venv.reset()
        again = [tuple(o[:3].astype(int)) for o in obs]
        assert all(a[2] == 0 for a in again)
        for a in again:
            assert a[1] > max(played[a[0]], default=-1), "a fresh episode of its engine"
    finally:
        venv.close()
    return f"{sum(len(v) for v in played.values())} episodes over {m} engines in {n} slots, each once and in order"


def test_speed():
    from stable_baselines3.common.vec_env import SubprocVecEnv
    n, steps = 4, 60
    res = {}
    for name, make in (("subproc", lambda: SubprocVecEnv([fn(i, reset_s=0.3, step_s=0.01) for i in range(n)], start_method="spawn")),
                       ("pool", lambda: PoolVecEnv([fn(i, reset_s=0.3, step_s=0.01) for i in range(3 * n)], n))):
        venv = make()
        try:
            venv.reset()
            t = time.time()
            for _ in range(steps):
                venv.step(np.zeros((n, 2), np.float32))
            res[name] = time.time() - t
        finally:
            venv.close()
    assert res["pool"] * 2.5 < res["subproc"], res
    return f"{steps} rounds: SubprocVecEnv {res['subproc']:.2f} s, pool {res['pool']:.2f} s"


def test_balance():
    """Engines with a slow reset keep about the same share of decisions when spare engines are placed."""
    n, m = 2, 8
    venv = PoolVecEnv([fn(i, reset_s=0.03 if i % 2 else 0.0, step_s=0.003) for i in range(m)], n)
    try:
        venv.reset()
        for _ in range(600):
            venv.step(np.zeros((n, 2), np.float32))
        tr = venv.transitions
    finally:
        venv.close()
    assert sum(tr) == 600 * n and max(tr) <= 1.5 * min(tr), tr
    return f"decisions per engine {tr}"


def _must_fail(venv, text, seconds=30.0):
    """Steps until an error, at most `seconds` (engines that start late join only after their process is up)."""
    try:
        venv.reset()
        t = time.time()
        while time.time() - t < seconds:
            venv.step(np.zeros((venv.num_envs, 2), np.float32))
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
        try:
            venv.step(np.zeros((venv.num_envs, 2), np.float32))
            raise AssertionError("a broken pool was used again")
        except RuntimeError as again:
            assert "had an error" in str(again), str(again)
        return
    finally:
        venv.close()
    raise AssertionError(f"no error with {text!r}")


def test_error():
    st = 0.002
    _must_fail(PoolVecEnv([fn(i, step_s=st, fail_at=5 if i == 1 else None) for i in range(4)], 2), "toy failure in engine 1")
    _must_fail(PoolVecEnv([fn(i, step_s=st, reset_fail_at=2 if i == 3 else None) for i in range(4)], 2), "toy reset failure in engine 3")
    _must_fail(PoolVecEnv([fn(i, step_s=st, die_at=4 if i == 0 else None) for i in range(4)], 2), "exit code 3")
    _must_fail(PoolVecEnv([fn(i, step_s=st, reset_fail_at=0 if i == 3 else None) for i in range(4)], 2), "toy reset failure in engine 3")
    return "errors in a step, in a background reset, at the first reset, and an engine's death stop the learner; the pool refuses further use"


def test_close_during_reset():
    venv = PoolVecEnv([fn(i, first_reset_s=60.0 if i >= 2 else None) for i in range(4)], 2, close_timeout_s=3.0)
    try:
        venv.reset()
        for _ in range(5):
            venv.step(np.zeros((2, 2), np.float32))
    finally:
        t = time.time()
        venv.close()
        took = time.time() - t
    assert took < 15, took
    assert not any(p.is_alive() for p in venv.processes)
    return f"close() with two engines inside a 60 s reset: {took:.1f} s"


def test_ppo():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
    with tempfile.TemporaryDirectory() as d:
        venv = VecNormalize(VecMonitor(PoolVecEnv([fn(i, reset_s=0.01) for i in range(6)], 3), filename=f"{d}/m"))
        try:
            model = PPO("MlpPolicy", venv, n_steps=32, batch_size=48, n_epochs=2, device="cpu", seed=0, verbose=0)
            model.learn(total_timesteps=192)
            assert model.num_timesteps == 192
        finally:
            venv.close()
        rows = Path(f"{d}/m.monitor.csv").read_text().splitlines()[2:]
        assert len(rows) > 10
    return f"PPO 192 steps, {len(rows)} episodes in the monitor"


if __name__ == "__main__":
    for t in (test_contract, test_speed, test_balance, test_error, test_close_during_reset, test_ppo):
        print(f"{t.__name__} OK:", t(), flush=True)
