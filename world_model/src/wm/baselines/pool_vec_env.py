"""A Stable-Baselines3 VecEnv whose n_envs slots are served by a larger pool of environment processes ("engines").

Why (measured 2026-11-18 on uoft, run ppo_state_s0): one decision of StateInsertEnv takes 0.09 s, but the start of
an episode (placing the hand) 3.3 s (median, up to 12 s) and a change of grasp 10 s. SubprocVecEnv steps its workers
in lock-step and resets a worker inside its step, so with 24 workers nearly every round waited for somebody's reset:
18 decisions/s instead of ~200.

Here a slot whose episode ends takes over an engine that has already placed its next episode, and the engine that
finished places its own next episode in the background. What SB3 sees is SubprocVecEnv's contract: per slot a
sequence of complete episodes; at an episode's end done = True, info["terminal_observation"] = the episode's last
observation, info["TimeLimit.truncated"] = truncated and not terminated, and the returned observation is the first
one of the slot's next episode; reset_infos holds the info of each slot's latest reset. Every engine is an ordinary
environment with its own grasps and random generator, so its episodes are the ones it would play under
SubprocVecEnv. What differs: which engine's episode a slot plays next. Among the engines whose next episode is placed,
the slot takes the one that has played the fewest decisions so far, so every engine keeps about the same share of the
data as a SubprocVecEnv worker had (an engine whose resets are slow can still fall behind when no other engine is
ready). Which engine lands in which slot depends on timing, so a run is not reproducible step for step.

Seeds given through VecEnv.seed() are not passed on (the engines are seeded when they are built; StateInsertEnv
draws everything from its own generator, so under SubprocVecEnv such seeds changed nothing either). get_attr /
set_attr / env_method reach the engines in the slots only. An error in an engine stops the training with the
engine's traceback (also printed by the engine to stderr), as with SubprocVecEnv; after any error the object refuses
further use. close() gives the engines `close_timeout_s` in total, then terminates them (they keep nothing worth
saving; an engine may be inside a long first reset).
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from multiprocessing.connection import wait
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvIndices
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

LOG_EVERY_EPISODES = 2000
SLOW_RESET_S = 120.0


def _engine(remote, parent_remote, env_fn_wrapper) -> None:
    """One engine: an environment answering one command at a time; an error is printed, sent back, and ends it."""
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = None
    try:
        env = _patch_env(env_fn_wrapper.var())
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = env.step(data)
            elif cmd == "reset":
                out = env.reset()
            elif cmd == "get_spaces":
                out = (env.observation_space, env.action_space)
            elif cmd == "get_attr":
                out = env.get_wrapper_attr(data)
            elif cmd == "set_attr":
                out = setattr(env, data[0], data[1])
            elif cmd == "env_method":
                out = env.get_wrapper_attr(data[0])(*data[1], **data[2])
            elif cmd == "is_wrapped":
                out = is_wrapped(env, data)
            elif cmd == "close":
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the engine")
            remote.send(("ok", out))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        tb = traceback.format_exc()
        try:
            remote.send(("error", tb))
        except Exception:
            pass
        try:
            print(f"pool_vec_env engine failed:\n{tb}", file=sys.stderr, flush=True)
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        remote.close()


class PoolVecEnv(VecEnv):
    """`env_fns` builds the engines (at least n_envs of them); `n_envs` slots are what the learner sees."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], n_envs: int, start_method: str = "spawn",
                 close_timeout_s: float = 30.0):
        if not 0 < n_envs <= len(env_fns):
            raise ValueError(f"{len(env_fns)} engines for {n_envs} slots")
        self.waiting, self.closed, self.broken = False, False, False
        self.close_timeout_s = close_timeout_s
        ctx = mp.get_context(start_method)
        self.remotes, work = zip(*[ctx.Pipe() for _ in env_fns])
        self.processes = []
        for w, r, fn in zip(work, self.remotes, env_fns):
            p = ctx.Process(target=_engine, args=(w, r, CloudpickleWrapper(fn)), daemon=True)
            p.start()
            self.processes.append(p)
            w.close()
        self.n_engines = len(env_fns)
        self.slot_engine: list[int | None] = [None] * n_envs
        self.preparing: dict[int, float] = {}                # engine -> when its reset was sent (its reply is pending)
        self.ready: list[tuple[int, Any, dict]] = []         # placed, not in a slot yet, in the order they were collected
        self.episodes = [0] * self.n_engines                 # episodes handed to slots, per engine
        self.transitions = [0] * self.n_engines              # decisions played, per engine
        self.placed_once: set[int] = set()                   # engines past their first reset (which may include a start delay)
        self.wait_s, self.start_wait_s, self._t0 = 0.0, 0.0, time.time()   # slots blocked in steps / in reset()
        self._send(0, "get_spaces", None)
        observation_space, action_space = self._recv(0)
        super().__init__(n_envs, observation_space, action_space)

    # ---- engine I/O
    def _dead(self, e: int, what: str) -> RuntimeError:
        self.broken = True
        self.processes[e].join(1)
        return RuntimeError(f"engine {e} {what} (process exit code {self.processes[e].exitcode})")

    def _send(self, e: int, cmd: str, data) -> None:
        try:
            self.remotes[e].send((cmd, data))
        except OSError:
            try:                                           # it may have sent its error before it ended
                if self.remotes[e].poll(0):
                    self._recv(e)
            except RuntimeError:
                raise
            except Exception:
                pass
            raise self._dead(e, f"cannot be reached for '{cmd}'") from None

    def _recv(self, e: int):
        try:
            kind, out = self.remotes[e].recv()
        except (EOFError, OSError):
            raise self._dead(e, "ended without an answer") from None
        if kind == "error":
            self.broken = True
            raise RuntimeError(f"engine {e} failed:\n{out}")
        return out

    def _prepare(self, e: int) -> None:
        if not self._idle(e):
            raise RuntimeError(f"engine {e} is not idle")
        self._send(e, "reset", None)
        self.preparing[e] = time.time()

    def _collect(self, timeout: float | None) -> None:
        if not self.preparing:
            return
        conns = {self.remotes[e]: e for e in self.preparing}
        for c in wait(list(conns), timeout):
            e = conns[c]
            del self.preparing[e]
            obs, info = self._recv(e)
            self.placed_once.add(e)
            self.ready.append((e, obs, info))

    def _take(self, in_reset: bool = False) -> tuple[int, Any, dict]:
        """Of the engines with a placed episode, the one with the fewest decisions played (ties: collected first);
        waits for one when none is placed."""
        self._collect(0)
        if not self.ready:
            if not self.preparing:
                raise RuntimeError("no engine is placing an episode and none is placed")
            t = time.time()
            while not self.ready:
                self._collect(None)
            if in_reset:
                self.start_wait_s += time.time() - t
            else:
                self.wait_s += time.time() - t
        i = min(range(len(self.ready)), key=lambda j: self.transitions[self.ready[j][0]])
        e, obs, info = self.ready.pop(i)
        self.episodes[e] += 1
        n = sum(self.episodes)
        if n % LOG_EVERY_EPISODES == 0:
            now = time.time()
            slow = sorted(e for e, t in self.preparing.items() if e in self.placed_once and now - t > SLOW_RESET_S)
            print(f"pool_vec_env: {n} episodes in {(now - self._t0) / 60:.1f} min; slots waited {self.wait_s / 60:.1f} min "
                  f"in steps (+ {self.start_wait_s / 60:.1f} min in reset); decisions per engine min {min(self.transitions)} "
                  f"max {max(self.transitions)}; engines placing for more than {SLOW_RESET_S:.0f} s (first resets not counted): {slow}",
                  file=sys.stderr, flush=True)
        return e, obs, info

    def _idle(self, e: int) -> bool:
        return e not in self.preparing and e not in self.slot_engine and all(r[0] != e for r in self.ready)

    def _usable(self) -> None:
        if self.broken or self.closed:
            raise RuntimeError("this PoolVecEnv had an error or is closed")
        if self.waiting:
            raise RuntimeError("between step_async and step_wait")

    # ---- VecEnv API
    def reset(self):
        self._usable()
        try:
            for s, e in enumerate(self.slot_engine):    # an episode in progress is abandoned: its engine places a new one
                if e is not None:
                    self.slot_engine[s] = None
                    self._prepare(e)
            for e in range(self.n_engines):
                if self._idle(e):
                    self._prepare(e)
            obs = []
            for s in range(self.num_envs):
                e, o, info = self._take(in_reset=True)
                self.slot_engine[s] = e
                self.reset_infos[s] = info
                obs.append(o)
        except BaseException:
            self.broken = True
            raise
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def step_async(self, actions: np.ndarray) -> None:
        self._usable()
        if any(e is None for e in self.slot_engine):
            raise RuntimeError("step before reset")
        try:
            for e, a in zip(self.slot_engine, actions):
                self._send(e, "step", a)
        except BaseException:
            self.broken = True
            raise
        self.waiting = True

    def step_wait(self):
        if self.broken or not self.waiting:
            raise RuntimeError("step_wait without a pending step_async, or after an error")
        try:
            results = [self._recv(e) for e in self.slot_engine]
            self.waiting = False
            obs, rews, dones, infos = [], [], [], []
            for s, (o, r, terminated, truncated, info) in enumerate(results):
                self.transitions[self.slot_engine[s]] += 1
                done = bool(terminated or truncated)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    info["terminal_observation"] = o
                obs.append(o)
                rews.append(r)
                dones.append(done)
                infos.append(info)
            ended = [s for s in range(self.num_envs) if dones[s]]
            for s in ended:                               # every finished engine starts placing before any slot waits
                e, self.slot_engine[s] = self.slot_engine[s], None
                self._prepare(e)
            for s in ended:
                self.slot_engine[s], obs[s], self.reset_infos[s] = self._take()
        except BaseException:
            self.broken = True
            raise
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def close(self) -> None:
        if self.closed:
            return
        for r in self.remotes:                            # pending answers are not read: every engine stops at "close"
            try:
                r.send(("close", None))
            except OSError:
                pass
        deadline = time.time() + self.close_timeout_s
        for p in self.processes:
            p.join(max(0.0, deadline - time.time()))
        late = [p for p in self.processes if p.is_alive()]
        for p in late:
            p.terminate()
        for p in late:
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
        for r in self.remotes:
            r.close()
        print(f"pool_vec_env: closed after {sum(self.episodes)} episodes in {(time.time() - self._t0) / 60:.1f} min; slots "
              f"waited {self.wait_s / 60:.1f} min in steps (+ {self.start_wait_s / 60:.1f} min in reset); {len(late)} engines "
              f"terminated", file=sys.stderr, flush=True)
        self.closed = True

    def _slot_engines(self, indices: VecEnvIndices) -> list[int]:
        """Engines of the given slots; before the first reset, engines 0..n_envs-1 (all of them idle then)."""
        self._usable()
        out = []
        for i in self._get_indices(indices):
            e = self.slot_engine[i] if self.slot_engine[i] is not None else i
            if e in self.preparing:
                raise RuntimeError(f"engine {e} is placing an episode; ask a slot's engine between steps")
            out.append(e)
        return out

    def _ask(self, indices: VecEnvIndices, cmd: str, data) -> list[Any]:
        es = self._slot_engines(indices)
        for e in es:
            self._send(e, cmd, data)
        return [self._recv(e) for e in es]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._ask(indices, "get_attr", attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._ask(indices, "set_attr", (attr_name, value))

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        return self._ask(indices, "env_method", (method_name, method_args, method_kwargs))

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return self._ask(indices, "is_wrapped", wrapper_class)

    def get_images(self):
        return [None for _ in range(self.num_envs)]


__all__ = ["PoolVecEnv"]
