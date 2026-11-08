"""Latent-free FM policy (baseline for the latent policy, v3).

Same visual encoder architecture as the autoencoder (models/encoder.py: DINOv2 + raw RGB + raw depth
+ viewing rays, self-attention, 64 queries), same 4-frame context, same joint input, same transformer
+ flow-matching head and action chunk as LatentFMPolicy. The only difference: no latent bottleneck
and none of the autoencoder's reconstruction / pose supervision; the encoder (DINOv2 frozen, as in
the autoencoder) is trained from scratch by the behaviour-cloning loss alone.
"""

from __future__ import annotations

import torch

from .encoder import ImageEncoder
from .policy import LatentFMPolicy

OBS_KEYS = ("rgb", "depth", "K", "cam_T_world", "hand", "R_P", "q")


class ImageFMPolicy(LatentFMPolicy):
    def __init__(self, encoder: dict | None = None, context: int = 4, d: int = 256, layers: int = 4,
                 heads: int = 8, hidden: int = 512):
        enc = ImageEncoder(**(encoder or {}))
        super().__init__(latent_ch=enc.d, context=context, d=d, layers=layers, heads=heads, hidden=hidden)
        self.encoder_img = enc

    def tokens(self, obs: dict) -> torch.Tensor:
        """obs: rgb / depth / cam_T_world / hand (B, F, ...), K / R_P (B, ...) -> (B, F, 64, d)."""
        B, Fr = obs["depth"].shape[:2]
        fl = lambda x: x.flatten(0, 1)
        rep = lambda x: x[:, None].expand(-1, Fr, *x.shape[1:]).flatten(0, 1)
        t = self.encoder_img(fl(obs["rgb"]), fl(obs["depth"]), rep(obs["K"]), fl(obs["cam_T_world"]),
                             fl(obs["hand"]), rep(obs["R_P"]))["tokens"]
        return t.view(B, Fr, *t.shape[1:])

    def condition(self, obs, q=None) -> torch.Tensor:
        if isinstance(obs, dict):
            return super().condition(self.tokens(obs), obs["q"])
        return super().condition(obs, q)

    def loss(self, obs: dict, q, a1: torch.Tensor) -> torch.Tensor:
        return super().loss(obs, q, a1)

    @torch.no_grad()
    def sample(self, obs: dict, q=None, steps: int = 10, n: int = 1) -> torch.Tensor:
        return super().sample(obs, q, steps, n)
