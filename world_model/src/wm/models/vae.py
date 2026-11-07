"""Token VAE: frame -> latent (64 tokens x C channels) -> what the latent must keep.

Encoder: RelationTokenizer -> per-token Gaussian (mean, log-variance), C channels.
Decoder (from the latent only):
    plug / port points   learned point queries -> xyz; Chamfer to the observed points
    joints               from the joint token; L1 to the measured joint angles
    plug pose            "where does the plug have to go to be inserted": rotation (6D)
                         + translation in hand-centred port axes, trained with the
                         plug-point loss (wm.training.losses.plug_point_loss)
The world model predicts latents; this decoder turns predicted latents back into
plug pose and joints, which is how predictions are evaluated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry import rot6d_to_matrix
from .perception import mlp
from .tokenizer import N_TOKENS, RelationTokenizer


class TokenVAE(nn.Module):
    def __init__(self, tokenizer: dict | None = None, latent_ch: int = 8, dec_d: int = 256,
                 dec_layers: int = 2, n_dec_plug: int = 256, n_dec_port: int = 512):
        super().__init__()
        self.tokenizer = RelationTokenizer(**(tokenizer or {}))
        d = self.tokenizer.d
        self.latent_ch, self.n_dec_plug = latent_ch, n_dec_plug
        self.to_moments = nn.Linear(d, 2 * latent_ch)
        self.from_latent = nn.Linear(latent_ch, dec_d)
        self.pos = nn.Parameter(torch.randn(N_TOKENS, dec_d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(dec_d, 8, 2 * dec_d, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer(), dec_layers)
        self.point_queries = nn.Parameter(torch.randn(n_dec_plug + n_dec_port, dec_d) * 0.02)
        self.point_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dec_d, 8, 2 * dec_d, batch_first=True, norm_first=True), dec_layers)
        self.point_out = mlp((dec_d, 128, 3))
        self.joint_out = mlp((dec_d, 128, 7))
        self.pose_q = nn.Parameter(torch.randn(1, 1, dec_d) * 0.02)
        self.pose_pool = nn.MultiheadAttention(dec_d, 8, batch_first=True)
        self.pose_out = mlp((dec_d, 256, 9))

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        aux = self.tokenizer(batch["rgb"], batch["depth"], batch["K"], batch["cam_T_world"],
                             batch["hand"], batch["R_P"], batch["qpos"])
        mean, logvar = self.to_moments(aux.pop("tokens")).chunk(2, -1)
        return mean, logvar.clamp(-10, 5), aux

    def decode(self, z: torch.Tensor) -> dict:
        """z (B, 64, C) -> plug/port points (m), joints (rad), plug pose (R, t in m)."""
        h = self.dec(self.from_latent(z) + self.pos)
        q = self.point_queries[None].expand(z.shape[0], -1, -1)
        pts = self.point_out(self.point_dec(q, h)) * 0.05
        pooled = self.pose_pool(self.pose_q.expand(z.shape[0], -1, -1), h, h, need_weights=False)[0][:, 0]
        pose = self.pose_out(pooled)
        return {"plug_points": pts[:, :self.n_dec_plug], "port_points": pts[:, self.n_dec_plug:],
                "joints": self.joint_out(h[:, -1]), "R": rot6d_to_matrix(pose[:, :6]), "t": pose[:, 6:] * 0.01}

    def forward(self, batch: dict, sample: bool = True) -> dict:
        mean, logvar, aux = self.encode(batch)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return {**self.decode(z), **aux, "mean": mean, "logvar": logvar}
