"""Residual RL on top of the frozen latent FM policy, trained in the world model's imagination (PPO).

The FM policy proposes 16 joint targets per decision (frozen, sampled as usual); a small Gaussian
policy adds a bounded correction to the 8 that are executed. Starting with a zero correction, so at
iteration 0 the agent is exactly the FM policy.

Observation of the residual policy: the 4 context latents (as the FM sees them), the measured joints,
and the FM's proposal for the executed 8 targets (relative to the joints). Output: 56 values in
[-1, 1] (Gaussian, clipped) times RESIDUAL_RAD per joint target.

Reward per 0.4 s step, all read from the imagined latent / imagined proprioception (no simulator truth):
    progress     distance to inserted before - after (mm)
    success      + success_bonus when the distance falls below the success threshold
    fence (v2)   - fence_penalty once, when the hand leaves the region around its start (episode ends)
    drift (v2)   - drift_coef * lateral hand motion of the step (mm), across the insertion axis
    hard push    - push_coef * max(0, gap - gap_free_mm), gap = |commanded - measured hand position| (mm):
                 pushing against something instead of probing
    uncertainty  - unc_coef * |distance(sample 1) - distance(sample 2)| (mm): the world model disagrees
    size         - size_coef * mean(a^2): stay close to the FM
    time         - time_cost
PPO with GAE; episodes of `horizon` imagined steps, bootstrapped by the critic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..models.policy import ACTION_DIM, CHUNK, LatentFMPolicy
from ..data.datasets import WM_STEP

RESIDUAL_RAD = 0.004          # v1 default max correction per executed joint target (rad); v2 sets cfg["residual_rad"]
N_RES = WM_STEP * ACTION_DIM  # 56


class FrozenFM:
    """The latent FM policy as the base of the residual: batched sampling of absolute targets."""

    def __init__(self, policy_ckpt: str, device, flow_steps: int = 10):
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self.model = LatentFMPolicy(**p["cfg"]["model"])
        self.model.load_state_dict(p["ema"])
        self.model.to(device).eval().requires_grad_(False)
        self.mean, self.std = torch.tensor(p["mean"], device=device), torch.tensor(p["std"], device=device)
        self.scale, self.steps = float(p["scale"]), flow_steps

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """z (B, 4, 64, C) scaled latents, q (B, 7) -> relative targets (B, 16, 7)."""
        a = self.model.sample(z, q, steps=self.steps, n=1)[:, 0] * self.std + self.mean
        return a.view(-1, CHUNK, ACTION_DIM)


class ResidualActorCritic(nn.Module):
    def __init__(self, latent_dim: int = 4 * 64 * 8, hidden: int = 512):
        super().__init__()
        inp = latent_dim + 7 + N_RES
        mlp = lambda out: nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))
        self.actor, self.critic = mlp(N_RES), mlp(1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)                        # start as the FM policy (zero correction)
        self.log_std = nn.Parameter(torch.full((N_RES,), math.log(0.3)))

    @staticmethod
    def features(z, q, base_rel):
        """z (B,4,64,C), q (B,7) rad, base_rel (B,16,7) rad relative -> (B, D)."""
        return torch.cat([z.flatten(1), q, base_rel[:, :WM_STEP].flatten(1) * 100.0], 1)

    def dist(self, f):
        return torch.distributions.Normal(self.actor(f), self.log_std.exp())

    def value(self, f):
        return self.critic(f)[:, 0]


def gae(rew, val, last_val, nonterm, gamma, lam):
    """rew, val, nonterm (T, B); last_val (B,). nonterm[t] = 0 if the episode ended at step t."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    g = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * nonterm[t] - val[t]
        g = delta + gamma * lam * nonterm[t] * g
        adv[t] = g
    return adv, adv + val


def train(env, base: FrozenFM, cfg: dict, run_dir: Path, log, device) -> Path:
    ac = ResidualActorCritic().to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg["lr"])
    run_dir.mkdir(parents=True, exist_ok=True)
    w = cfg["reward"]
    bound = cfg.get("residual_rad", RESIDUAL_RAD)
    for it in range(1, cfg["iterations"] + 1):
        obs = env.reset()
        F, A, LP, V, R, NT = [], [], [], [], [], []
        stats = {"progress": 0.0, "success": 0.0, "gap": 0.0, "unc": 0.0, "res": 0.0, "fenced": 0.0}
        for t in range(env.horizon):
            base_rel = base(obs["z"], obs["q"])
            f = ResidualActorCritic.features(obs["z"], obs["q"], base_rel)
            with torch.no_grad():
                d = ac.dist(f)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
                v = ac.value(f)
            corr = (a.clamp(-1, 1) * bound).view(-1, WM_STEP, ACTION_DIM)
            targets = (base_rel + obs["q"][:, None]).clone()
            targets[:, :WM_STEP] += corr
            obs, info = env.step(targets.cpu().numpy().astype(np.float64))
            alive = ~info["was_done"]
            progress = info["d_before"] - info["d_after"]
            r = (w["progress"] * progress + w["success_bonus"] * info["success"]
                 - w["push_coef"] * np.maximum(0.0, info["push_gap_mm"] - w["gap_free_mm"])
                 - w["unc_coef"] * info["uncertainty_mm"]
                 - w["size_coef"] * (a.clamp(-1, 1) ** 2).mean(-1).cpu().numpy() - w["time_cost"]
                 - w.get("fence_penalty", 0.0) * info["fenced"] - w.get("drift_coef", 0.0) * info["lateral_step_mm"]) * alive
            nonterm = (~(info["success"] | info["fenced"] | info["was_done"])).astype(np.float32)
            F.append(f), A.append(a), LP.append(lp), V.append(v)
            R.append(torch.tensor(r, dtype=torch.float32, device=device))
            NT.append(torch.tensor(nonterm, device=device))
            n_alive = max(1, int(alive.sum()))
            stats["progress"] += float((progress * alive).sum()) / n_alive
            stats["success"] += float(info["success"].mean())
            stats["fenced"] += float(info["fenced"].mean())
            stats["gap"] += float((info["push_gap_mm"] * alive).sum()) / n_alive
            stats["unc"] += float((info["uncertainty_mm"] * alive).sum()) / n_alive
            stats["res"] += float(a.clamp(-1, 1).abs().mean())
        with torch.no_grad():
            base_rel = base(obs["z"], obs["q"])
            last_v = ac.value(ResidualActorCritic.features(obs["z"], obs["q"], base_rel))
        V_, R_, NT_ = torch.stack(V), torch.stack(R), torch.stack(NT)
        adv, ret = gae(R_, V_, last_v * NT_[-1], NT_, cfg["gamma"], cfg["lam"])
        F_, A_, LP_ = torch.cat(F), torch.cat(A), torch.cat(LP)
        adv, ret = adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(F_)
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n, device=device)
            for k in range(0, n, cfg["minibatch"]):
                idx = perm[k:k + cfg["minibatch"]]
                d = ac.dist(F_[idx])
                ratio = (d.log_prob(A_[idx]).sum(-1) - LP_[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[idx]).mean()
                vl = (ac.value(F_[idx]) - ret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        H = env.horizon
        log(iteration=it, reward=float(R_.sum(0).mean()), progress_mm=stats["progress"], success_rate=stats["success"],
            push_gap_mm=stats["gap"] / H, fenced_rate=stats["fenced"], uncertainty_mm=stats["unc"] / H, residual_abs=stats["res"] / H,
            start_distance_mm=float(np.median(env.d_start)), std=float(ac.log_std.exp().mean()))
        if it % cfg["save_every"] == 0 or it == cfg["iterations"]:
            torch.save({"model": ac.state_dict(), "iteration": it, "cfg": cfg}, run_dir / f"residual_it{it}.pt")
            torch.save({"model": ac.state_dict(), "iteration": it, "cfg": cfg}, run_dir / "last.pt")
    return run_dir / "last.pt"
