"""Flow-matching policy on the shared latent.

Input: the last 4 latents (64 tokens x C each, the joint token included), the same context
the world model sees. Output: a chunk of 16 joint targets (0.8 s at 20 Hz) relative to the
measured joints; the first 8 (one world-model step) are executed, then the policy replans.

Network: a small transformer over the context tokens (frame and token embeddings) gives a
condition vector; a velocity MLP conditioned on it and on the flow time maps a noisy action
chunk to its velocity (conditional flow matching, linear path from noise to data).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tokenizer import N_TOKENS

CHUNK = 16
ACTION_DIM = 7


def _time_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * 1000 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class LatentFMPolicy(nn.Module):
    def __init__(self, latent_ch: int = 8, context: int = 4, d: int = 256, layers: int = 4,
                 heads: int = 8, hidden: int = 512):
        super().__init__()
        self.context = context
        self.inp = nn.Linear(latent_ch, d)
        self.token_pos = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(context, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        a = CHUNK * ACTION_DIM
        self.velocity = nn.Sequential(nn.Linear(a + d + 128, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                      nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, a))

    def condition(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, context, 64, C) -> (B, d)."""
        B = z.shape[0]
        h = self.inp(z) + self.token_pos[None, None] + self.frame_pos[None, :, None]
        h = torch.cat([self.cls.expand(B, -1, -1), h.flatten(1, 2)], 1)
        return self.norm(self.encoder(h)[:, 0])

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Velocity of the flow at (a_t, t). a_t (B, CHUNK*7) normalised actions, t (B,) in [0, 1]."""
        return self.velocity(torch.cat([a_t, cond, _time_embedding(t)], -1))

    def loss(self, z: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        cond = self.condition(z)
        a0 = torch.randn_like(a1)
        t = torch.rand(a1.shape[0], device=a1.device)
        a_t = (1 - t[:, None]) * a0 + t[:, None] * a1
        return (self.forward(a_t, t, cond) - (a1 - a0)).pow(2).mean()

    @torch.no_grad()
    def sample(self, z: torch.Tensor, steps: int = 10, n: int = 1) -> torch.Tensor:
        """n action chunks per input (normalised), Euler integration from noise. -> (B, n, CHUNK*7)."""
        cond = self.condition(z).repeat_interleave(n, 0)
        a = torch.randn(cond.shape[0], CHUNK * ACTION_DIM, device=z.device)
        for k in range(steps):
            t = torch.full((a.shape[0],), k / steps, device=z.device)
            a = a + self.forward(a, t, cond) / steps
        return a.view(z.shape[0], n, -1)
