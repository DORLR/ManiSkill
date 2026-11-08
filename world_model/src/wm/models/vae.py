"""Token autoencoder: frame -> latent (64 tokens x C channels) -> what the latent must keep.

Encoder: RelationTokenizer -> LayerNorm -> linear, C channels per token. Deterministic since v2:
the KL term of v1 (weight 0.001) is removed; LayerNorm bounds the token magnitudes and the
latents are rescaled to unit RMS before the world model, which is what the KL was there for.
(Class name kept as TokenVAE so v1 checkpoints and scripts keep working; load_vae() maps v1's
mean/log-variance head onto the mean.)
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
        # LayerNorm before the moments keeps token magnitudes bounded (without it the latent mean
        # drifted to ~50 and then exploded to ~6e5 at step 1145 of the first run; KL became 8e9).
        self.to_latent = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent_ch))
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

    def encode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """-> latent (B, 64, C) and the tokenizer's sampled plug/port points."""
        aux = self.tokenizer(batch["rgb"], batch["depth"], batch["K"], batch["cam_T_world"],
                             batch["hand"], batch["R_P"], batch["qpos"])
        return self.to_latent(aux.pop("tokens")), aux

    def _trunk(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.from_latent(z) + self.pos)

    def _pose(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pose_pool(self.pose_q.expand(h.shape[0], -1, -1), h, h, need_weights=False)[0][:, 0]
        pose = self.pose_out(pooled)
        return rot6d_to_matrix(pose[:, :6]), pose[:, 6:] * 0.01

    def decode_pose(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Only the plug pose (R, t in m): the motion still needed to be inserted."""
        return self._pose(self._trunk(z))

    def decode(self, z: torch.Tensor) -> dict:
        """z (B, 64, C) -> plug/port points (m), joints (rad), plug pose (R, t in m)."""
        h = self._trunk(z)
        q = self.point_queries[None].expand(z.shape[0], -1, -1)
        pts = self.point_out(self.point_dec(q, h)) * 0.05
        R, t = self._pose(h)
        return {"plug_points": pts[:, :self.n_dec_plug], "port_points": pts[:, self.n_dec_plug:],
                "joints": self.joint_out(h[:, -1]), "R": R, "t": t}

    def forward(self, batch: dict) -> dict:
        z, aux = self.encode(batch)
        return {**self.decode(z), **aux, "z": z}


def load_vae(path: str, device=None) -> TokenVAE:
    """Load a checkpoint (v1: Gaussian head -> its mean half; v2: as saved), eval mode."""
    ck = torch.load(path, map_location="cpu")
    model = TokenVAE(**ck["cfg"]["model"])
    state = dict(ck["model"])
    if "to_moments.1.weight" in state:                     # v1 checkpoint
        c = model.latent_ch
        for k in ("0.weight", "0.bias"):
            state[f"to_latent.{k}"] = state.pop(f"to_moments.{k}")
        state["to_latent.1.weight"] = state.pop("to_moments.1.weight")[:c]
        state["to_latent.1.bias"] = state.pop("to_moments.1.bias")[:c]
    model.load_state_dict(state)
    model.eval()
    return model.to(device) if device is not None else model
