"""The agreed model (2026-11-15): TD-MPC2's latent structure, fed with the wrist pictures and the joints.

Kept from TD-MPC2 (its own layers and losses are imported, not copied): SimNorm latent, MLP dynamics, reward head and
a 5-head value ensemble with two-hot targets, latent consistency loss against the encoder's own stop-gradient output,
loss weights, learning rates, gradient clipping.
Changed, all at once (the user's decision):
  encoder   three pictures (RGB + depth, 640 x 640, 0.2 s apart) through one ResNet-18 each, plus the joints
  values    learned from the real discounted returns of the recorded branches instead of bootstrapped targets, on the
            imagined latents; each head leaves out a fixed fifth of the branches; besides Q(z, a) a state value V(z),
            so that an imagined end state can be scored without a further action
  no policy head, no planner over raw actions: candidates come from the policy that recorded the data
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import use_tdmpc2
from .data import ACTION_DIM, JOINTS

DEFAULTS = dict(horizon=4, latent_dim=512, mlp_dim=512, enc_dim=256, num_enc_layers=2, num_q=5, dropout=0.01,
                simnorm_dim=8, num_bins=101, vmin=-10, vmax=10, frame_dim=512, action_dim=ACTION_DIM,
                consistency_coef=20, reward_coef=0.1, value_coef=0.1, rho=0.5, lr=3e-4, enc_lr_scale=0.3,
                grad_clip_norm=20, gamma=1 - 1 / 30, shift_px=4, amp=True, multitask=False, task_dim=0, stem_stride=2)
# gamma: TD-MPC2's rule for an episode of 150 decisions, (150 / 5 - 1) / (150 / 5)


class Cfg(types.SimpleNamespace):
    def get(self, k, d=None):
        return getattr(self, k, d)


def make_cfg(**over) -> Cfg:
    c = Cfg(**{**DEFAULTS, **over})
    c.bin_size = (c.vmax - c.vmin) / (c.num_bins - 1)
    return c


class FrameNet(nn.Module):
    """One picture (RGB + depth) -> a feature vector. ResNet-18 with group norm (no batch statistics: the same
    network gives the targets), its last map kept as a map (1 x 1 conv to 32 channels, then flattened).
    `stem_stride` 2: the usual ResNet stem, last map 20 x 20. `stem_stride` 4: the first convolution steps over 4
    pixels instead of 2 (it still reads every pixel of the 640 x 640 picture), last map 10 x 10, a quarter of the
    computation: with stride 2 the shared GPU manages 16 windows per second (measured 2026-11-16), too few to train
    this model in a day."""

    def __init__(self, out_dim: int, amp: bool, stem_stride: int = 2):
        super().__init__()
        net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(8, c))
        self.stem = nn.Sequential(nn.Conv2d(4, 64, 7, stem_stride, 3, bias=False), nn.GroupNorm(8, 64), nn.ReLU())
        self.trunk = nn.Sequential(net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
        self.squeeze = nn.Conv2d(512, 32, 1)
        side = 640 // (16 * stem_stride)
        self.out = nn.Sequential(nn.Linear(32 * side * side, out_dim), nn.LayerNorm(out_dim))
        self.amp = amp

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (N, 3, 640, 640) uint8, depth (N, 640, 640) int16 in 0.1 mm."""
        x = torch.cat([rgb.float() / 255.0 - 0.5, (depth.float() * 1e-3).clamp(0, 3).unsqueeze(1)], 1)
        x = self.stem(x.contiguous(memory_format=torch.channels_last))     # float32: 0.1 mm of depth must survive
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            x = self.squeeze(self.trunk(x))
        return self.out(x.float().flatten(1))


class LatentWM(nn.Module):
    def __init__(self, cfg: Cfg, stats: dict):
        super().__init__()
        use_tdmpc2()
        from common import init, layers, math
        self.cfg, self.math = cfg, math
        self.frame = FrameNet(cfg.frame_dim, cfg.amp, cfg.get("stem_stride", 2)).to(memory_format=torch.channels_last)
        L, A, M = cfg.latent_dim, cfg.action_dim, cfg.mlp_dim
        self._fuse = layers.mlp(3 * cfg.frame_dim + 3 * JOINTS, max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim], L,
                                act=layers.SimNorm(cfg))
        self._dynamics = layers.mlp(L + A, 2 * [M], L, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(L + A, 2 * [M], cfg.num_bins)
        for m in (self._fuse, self._dynamics, self._reward):
            m.apply(init.weight_init)
        self._Qs = layers.Ensemble([layers.mlp(L + A, 2 * [M], cfg.num_bins, dropout=cfg.dropout).apply(init.weight_init)
                                    for _ in range(cfg.num_q)])
        self._Vs = layers.Ensemble([layers.mlp(L, 2 * [M], cfg.num_bins, dropout=cfg.dropout).apply(init.weight_init)
                                    for _ in range(cfg.num_q)])
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"], self._Vs.params["2", "weight"]])
        for k in ("q_mean", "q_std", "dq_std"):
            self.register_buffer(k, torch.tensor(stats[k], dtype=torch.float32))

    # ---- parts
    def frames(self, rgb: torch.Tensor, depth: torch.Tensor, chunk: int = 96) -> torch.Tensor:
        """rgb (..., 3, 640, 640) uint8, depth (..., 640, 640) -> (..., frame_dim)."""
        lead = depth.shape[:-2]
        rgb, depth = rgb.reshape(-1, *rgb.shape[-3:]), depth.reshape(-1, *depth.shape[-2:])
        f = torch.cat([self.frame(rgb[i:i + chunk], depth[i:i + chunk]) for i in range(0, len(rgb), chunk)])
        return f.reshape(*lead, -1)

    def fuse(self, feat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """feat (N, 3, D) and q (N, 3, 7) for the instants [t-2, t-1, t] -> latent (N, L)."""
        j = torch.cat([(q[:, 2] - self.q_mean) / self.q_std, (q[:, 2] - q[:, 1]) / self.dq_std,
                       (q[:, 1] - q[:, 0]) / self.dq_std], -1)
        return self._fuse(torch.cat([feat.flatten(1), j], -1))

    def encode(self, rgb, depth, q, chunk: int = 96):
        """rgb (N, 3, 3, 640, 640), depth (N, 3, 640, 640), q (N, 3, 7) -> latent."""
        return self.fuse(self.frames(rgb, depth, chunk), q)

    def next(self, z, a):
        return self._dynamics(torch.cat([z, a], -1))

    def reward(self, z, a):
        return self._reward(torch.cat([z, a], -1))

    def Q(self, z, a):
        return self._Qs(torch.cat([z, a], -1))

    def V(self, z):
        return self._Vs(z)

    def scalar(self, logits):
        return self.math.two_hot_inv(logits, self.cfg)

    # ---- training
    def loss(self, batch: dict, augment: bool = True) -> tuple[torch.Tensor, dict]:
        """batch as data.Windows (on the GPU). TD-MPC2's update with the changes named in the module docstring."""
        cfg, math = self.cfg, self.math
        H = cfg.horizon
        rgb, depth, q, action = batch["rgb"].permute(0, 1, 4, 2, 3), batch["depth"], batch["q"], batch["action"].transpose(0, 1)
        B = rgb.shape[0]
        if augment and cfg.shift_px:                                  # one shift per window, the same for all its pictures
            s = torch.randint(-cfg.shift_px, cfg.shift_px + 1, (B, 2)).tolist()
            rgb = torch.stack([torch.roll(x, sh, (-2, -1)) for x, sh in zip(rgb, s)])
            depth = torch.stack([torch.roll(x, sh, (-2, -1)) for x, sh in zip(depth, s)])
        was_training = self.training
        with torch.no_grad():                                         # targets: the encoder itself, no gradient
            self.eval()
            ft = self.frames(rgb[:, 2:], depth[:, 2:])                # pictures 2 .. 2H+2 of the window
            z_tgt = torch.stack([self.fuse(ft[:, 2 * j - 2:2 * j + 1], q[:, 2 * j:2 * j + 3]) for j in range(1, H + 1)])
        self.train(was_training)
        z = self.fuse(self.frames(rgb[:, :3], depth[:, :3]), q[:, :3])
        zs, consistency = [z], 0.0
        for t in range(H):
            z = self.next(z, action[t])
            consistency = consistency + F.mse_loss(z, z_tgt[t]) * cfg.rho ** t
            zs.append(z)
        zs = torch.stack(zs)                                          # (H+1, B, L)
        reward_pred, qs, vs = self.reward(zs[:-1], action), self.Q(zs[:-1], action), self.V(zs)
        rew, ret, mask = batch["reward"].transpose(0, 1), batch["ret"].transpose(0, 1), batch["head_mask"].transpose(0, 1)
        norm = mask.sum(1).clamp(min=1.0)                             # per head: its share of the batch
        reward_loss, q_loss, v_loss = 0.0, 0.0, 0.0
        for t in range(H + 1):
            w = cfg.rho ** t
            if t < H:
                reward_loss = reward_loss + math.soft_ce(reward_pred[t], rew[t, :, None], cfg).mean() * w
            for h in range(cfg.num_q):
                v_loss = v_loss + (math.soft_ce(vs[h, t], ret[t, :, None], cfg)[:, 0] * mask[h]).sum() / norm[h] * w
                if t < H:
                    q_loss = q_loss + (math.soft_ce(qs[h, t], ret[t, :, None], cfg)[:, 0] * mask[h]).sum() / norm[h] * w
        consistency, reward_loss = consistency / H, reward_loss / H
        q_loss, v_loss = q_loss / (H * cfg.num_q), v_loss / ((H + 1) * cfg.num_q)
        total = cfg.consistency_coef * consistency + cfg.reward_coef * reward_loss + cfg.value_coef * (q_loss + v_loss)
        with torch.no_grad():
            info = {"consistency": consistency, "reward_loss": reward_loss, "q_loss": q_loss, "v_loss": v_loss, "total": total,
                    "z_spread": zs[0].std(0).mean(),         # 0 = every picture gives the same latent (collapse)
                    "v_abs_err": (self.scalar(vs[:, 0]).mean(0)[:, 0] - ret[0]).abs().mean(),
                    "reward_abs_err": (self.scalar(reward_pred)[..., 0] - rew).abs().mean()}
        return total, {k: float(v) for k, v in info.items()}

    # ---- scoring
    @torch.no_grad()
    def scores(self, z: torch.Tensor, action: torch.Tensor, valid: torch.Tensor, hs=(1, 2, 4)) -> dict:
        """Per head (heads, N): `v0` = V(z); `q0` = Q(z, a_0); `im<h>` = the imagined rewards of the first h given
        actions plus the discounted V of the imagined latent (fewer steps where the branch has fewer left).
        action (N, hmax, A), valid (N,)."""
        g = self.cfg.gamma
        out = {"v0": self.scalar(self.V(z))[..., 0], "q0": self.scalar(self.Q(z, action[:, 0]))[..., 0]}
        G, disc = torch.zeros(len(z), device=z.device), torch.ones(len(z), device=z.device)
        for t in range(max(hs)):
            on = (t < valid).float()
            G = G + disc * on * self.scalar(self.reward(z, action[:, t]))[:, 0]
            z = torch.where(on[:, None] > 0, self.next(z, action[:, t]), z)
            disc = disc * (1 - on + on * g)
            if t + 1 in hs:
                out[f"im{t + 1}"] = G[None] + disc[None] * self.scalar(self.V(z))[..., 0]
        return out


def optimizer(model: LatentWM) -> torch.optim.Optimizer:
    cfg = model.cfg
    enc = list(model.frame.parameters()) + list(model._fuse.parameters())
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": enc, "lr": cfg.lr * cfg.enc_lr_scale}, {"params": rest}], lr=cfg.lr)
