"""Latent-free FM policy (baseline for the latent policy).

Same visual encoder architecture as the VAE (RelationTokenizer: DINOv2 + PointNet++ plug/port
tokens, plug<->port cross-attention, 63 relation queries + joint token), same 4-frame context,
same transformer + flow-matching head and action chunk as LatentFMPolicy. The difference: no
VAE bottleneck and no plug-point / Chamfer supervision; the encoder is trained from scratch
(DINOv2 frozen, as in the VAE) by the behaviour-cloning loss only.
"""

from __future__ import annotations

import torch

from .policy import LatentFMPolicy
from .tokenizer import RelationTokenizer

OBS_KEYS = ("rgb", "depth", "K", "cam_T_world", "hand", "qpos", "R_P")


class ImageFMPolicy(LatentFMPolicy):
    def __init__(self, tokenizer: dict | None = None, context: int = 4, d: int = 256, layers: int = 4,
                 heads: int = 8, hidden: int = 512):
        tok = RelationTokenizer(**(tokenizer or {}))
        super().__init__(latent_ch=tok.d, context=context, d=d, layers=layers, heads=heads, hidden=hidden)
        self.tokenizer = tok

    def tokens(self, obs: dict) -> torch.Tensor:
        """obs: per-frame tensors (B, F, ...) except K and R_P (B, ...). -> (B, F, 64, d)."""
        B, Fr = obs["depth"].shape[:2]
        fl = lambda x: x.flatten(0, 1)
        rep = lambda x: x[:, None].expand(-1, Fr, *x.shape[1:]).flatten(0, 1)
        t = self.tokenizer(fl(obs["rgb"]), fl(obs["depth"]), rep(obs["K"]), fl(obs["cam_T_world"]),
                           fl(obs["hand"]), rep(obs["R_P"]), fl(obs["qpos"]))["tokens"]
        return t.view(B, Fr, *t.shape[1:])

    def condition(self, obs) -> torch.Tensor:
        return super().condition(self.tokens(obs) if isinstance(obs, dict) else obs)

    @torch.no_grad()
    def sample(self, obs, steps: int = 10, n: int = 1) -> torch.Tensor:
        cond = self.condition(obs).repeat_interleave(n, 0)
        a = torch.randn(cond.shape[0], self.velocity[-1].out_features, device=cond.device)
        for k in range(steps):
            a = a + self.forward(a, torch.full((a.shape[0],), k / steps, device=cond.device), cond) / steps
        return a.view(-1, n, a.shape[-1])
