"""TD-MPC2 itself, unchanged (the control run): its own TDMPC2 class, encoder (64 x 64 pictures, three stacked),
bootstrapped values, policy head -- only fed from the branch set instead of its own replay buffer, with its own
configuration file plus the entries that describe the task (picture observation, 56 action numbers, episodes that end).
Its `compile` switch is off in our runs: with torch.compile the first 1000 updates did not finish in 13 minutes on
the shared machine (all CPU cores busy), without it an update takes 0.1 s (measured 2026-11-16).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from . import TDMPC2_DIR, use_tdmpc2
from .data import ACTION_DIM, eval_points, returns, scale_action
from .model import Cfg

SIZE = 64
EPISODE_LENGTH = 150


def make_cfg(**over) -> Cfg:
    import yaml
    raw = yaml.safe_load((TDMPC2_DIR / "config.yaml").read_text())
    raw.pop("defaults", None)
    for k, v in raw.items():                                # yaml reads 3e-4 as text
        if isinstance(v, str):
            try:
                raw[k] = float(v)
            except ValueError:
                pass
    raw.update(task="wm-branch", obs="rgb", episodic=True, obs_shape={"rgb": [9, SIZE, SIZE]}, action_dim=ACTION_DIM,
               episode_length=EPISODE_LENGTH, multitask=False, task_dim=0, tasks=["wm-branch"], model_size=None,
               enable_wandb=False, save_video=False)
    raw.update(over)
    raw["bin_size"] = (raw["vmax"] - raw["vmin"]) / (raw["num_bins"] - 1)
    return Cfg(**raw)


def make_agent(cfg: Cfg):
    use_tdmpc2()
    from tdmpc2 import TDMPC2
    return TDMPC2(cfg)


def load_cache(root: str | Path, branches: list[dict]) -> list[dict]:
    """The 64 x 64 copies written by scripts/data/cache_branch64.py, in the order of `branches` (missing ones: None)."""
    out = []
    for b in branches:
        f = Path(root) / (b["rel"] + ".npz")
        if not f.exists():
            out.append(None)
            continue
        with np.load(f) as z:
            out.append({k: z[k] for k in ("rgb", "delta", "reward", "terminated", "q_img")} | {"face_k": int(z["face_k"])})
    return out


class RamBuffer:
    """What TDMPC2.update() asks of a buffer: sample() -> obs (H+1, B, 9, 64, 64) uint8, action (H, B, 56),
    reward (H, B, 1), terminated (H, B, 1), task None. Windows are drawn uniformly over all decisions with `horizon`
    decisions left in their branch (as its SliceSampler with strict_length)."""

    def __init__(self, cache: list[dict], stats: dict, cfg: Cfg, device="cuda:0"):
        h = cfg.horizon
        cache = [c for c in cache if c is not None and len(c["reward"]) >= h]
        self.frames = torch.from_numpy(np.concatenate([c["rgb"] for c in cache])).permute(0, 3, 1, 2).contiguous()
        self.action = torch.from_numpy(np.concatenate([scale_action(c["delta"], stats) for c in cache]))
        self.reward = torch.from_numpy(np.concatenate([c["reward"] for c in cache]))
        self.terminated = torch.from_numpy(np.concatenate([c["terminated"] for c in cache]))
        f0, t0, fmin, foff, toff = [], [], [], 0, 0
        for c in cache:
            n = len(c["reward"])
            k = np.arange(n - h + 1)
            f0.append(foff + 2 * k)
            t0.append(toff + k)
            fmin.append(np.full(len(k), foff))
            foff, toff = foff + len(c["rgb"]), toff + n
        self.f0, self.t0, self.fmin = (torch.from_numpy(np.concatenate(x)) for x in (f0, t0, fmin))
        self.cfg, self.device, self.h = cfg, device, h
        self.num_eps, self.windows = len(cache), len(self.f0)
        self._j = torch.arange(h + 1)[None, :, None] * 2 - torch.tensor([2, 1, 0])[None, None, :]

    def sample(self):
        B, h = self.cfg.batch_size, self.h
        i = torch.randint(self.windows, (B,))
        fi = torch.maximum(self.f0[i][:, None, None] + self._j, self.fmin[i][:, None, None])        # (B, H+1, 3)
        dev = self.device
        obs = self.frames[fi.reshape(-1)].to(dev, non_blocking=True).reshape(B, h + 1, 9, SIZE, SIZE).transpose(0, 1)
        ti = self.t0[i][:, None] + torch.arange(h)[None]
        return (obs.contiguous(), self.action[ti].transpose(0, 1).contiguous().to(dev),
                self.reward[ti].transpose(0, 1).unsqueeze(-1).contiguous().to(dev),
                self.terminated[ti].transpose(0, 1).unsqueeze(-1).contiguous().to(dev), None)


def points(cache: list[dict], stats: dict, gamma: float, hmax: int) -> dict:
    """The scored decisions of all cached branches: obs (P, 9, 64, 64), action (P, hmax, 56), valid, ret, k, branch, tags."""
    obs, action, valid, ret, ks, br, tags = [], [], [], [], [], [], []
    for bi, c in enumerate(cache):
        if c is None:
            continue
        n = len(c["reward"])
        act, g = scale_action(c["delta"], stats), returns(c["reward"], gamma)
        for tag, k in eval_points(n, c["face_k"]):
            idx = np.maximum(np.arange(2 * k - 2, 2 * k + 1), 0)
            obs.append(c["rgb"][idx].transpose(0, 3, 1, 2).reshape(9, SIZE, SIZE))
            a = np.zeros((hmax, ACTION_DIM), np.float32)
            m = min(hmax, n - k)
            a[:m] = act[k:k + m]
            action.append(a); valid.append(m); ret.append(g[k]); ks.append(k); br.append(bi); tags.append(tag)
    return {"obs": torch.from_numpy(np.stack(obs)), "action": torch.from_numpy(np.stack(action)),
            "valid": torch.tensor(valid), "ret": np.asarray(ret), "k": np.asarray(ks), "branch": np.asarray(br), "tags": tags}


@torch.no_grad()
def scores(agent, obs: torch.Tensor, action: torch.Tensor, valid: torch.Tensor, hs=(1, 2, 3)) -> dict:
    """Per head (heads, N), with TD-MPC2's own parts as in its _estimate_value: `q0` = Q(z, a_0); `qpi` = Q(z, pi(z));
    `im<h>` = imagined rewards of the first h given actions + discounted Q(z_h, pi(z_h))."""
    from common import math
    m, cfg = agent.model, agent.cfg
    inv = lambda x: math.two_hot_inv(x, cfg)[..., 0]
    z = m.encode(obs, None)
    q_pi = lambda z: inv(m.Q(z, m.pi(z, None)[1]["mean"], None, return_type="all"))
    out = {"q0": inv(m.Q(z, action[:, 0], None, return_type="all")), "qpi": q_pi(z)}
    G, disc = torch.zeros(len(z), device=z.device), torch.ones(len(z), device=z.device)
    for t in range(max(hs)):
        on = (t < valid).float()
        G = G + disc * on * inv(m.reward(z, action[:, t], None))
        z = torch.where(on[:, None] > 0, m.next(z, action[:, t], None), z)
        disc = disc * (1 - on + on * float(agent.discount))
        if t + 1 in hs:
            out[f"im{t + 1}"] = G[None] + disc[None] * q_pi(z)
    return out
