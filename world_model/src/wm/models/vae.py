"""Latent autoencoder (v3): frame -> latent (64 tokens x C channels) -> what the latent must keep.

Encoder: ImageEncoder (models/encoder.py) -> LayerNorm -> linear to C channels per token.
Deterministic: no KL (removed in v2; LayerNorm bounds the token magnitudes and the latents are
rescaled to unit RMS before the world model). No joint token: the robot state (joints / hand pose)
is given to the world model and the policy separately, as in Ctrl-World.

Decoder, from the latent only:
    patch decoder   one query per image patch (529) reads the latent and reconstructs that patch:
                    raw depth (incl. where the camera has no return: a validity logit per pixel),
                    raw RGB, and the frozen DINOv2 feature -- the whole frame as the camera gave it
    pose head       the rigid motion the plug still needs to be fully inserted, in hand-centred
                    port axes (6D rotation + translation); trained against FK of the one
                    insertion of this grasp (training/losses.py). The world model's predictions
                    are read through this head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry import rot6d_to_matrix
from .encoder import N_PATCH, N_TOKENS, PATCH, ImageEncoder


def mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers[:-1])


class TokenVAE(nn.Module):
    """Name kept from v1 for continuity of scripts; deterministic autoencoder since v2."""

    def __init__(self, encoder: dict | None = None, latent_ch: int = 8, dec_d: int = 256,
                 dec_layers: int = 2, patch_layers: int = 2, heads: int = 8):
        super().__init__()
        self.encoder = ImageEncoder(**(encoder or {}))
        d = self.encoder.d
        self.latent_ch = latent_ch
        self.to_latent = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent_ch))
        # decoder trunk over the 64 latent tokens
        self.from_latent = nn.Linear(latent_ch, dec_d)
        self.pos = nn.Parameter(torch.randn(N_TOKENS, dec_d) * 0.02)
        enc_layer = lambda: nn.TransformerEncoderLayer(dec_d, heads, 2 * dec_d, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(enc_layer(), dec_layers)
        # patch decoder: 529 patch queries cross-attend to the trunk
        self.patch_queries = nn.Parameter(torch.randn(N_PATCH, dec_d) * 0.02)
        self.patch_dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dec_d, heads, 2 * dec_d, batch_first=True, norm_first=True), patch_layers)
        self.out_depth = nn.Linear(dec_d, PATCH * PATCH)          # depth / 0.2 m
        self.out_valid = nn.Linear(dec_d, PATCH * PATCH)          # logit: camera returned a depth
        self.out_rgb = nn.Linear(dec_d, 3 * PATCH * PATCH)        # [0, 1]
        self.out_dino = nn.Linear(dec_d, self.encoder.dino.dim)
        # pose head: attention pooling over the trunk, then 6D rotation + translation
        self.pose_q = nn.Parameter(torch.randn(1, 1, dec_d) * 0.02)
        self.pose_pool = nn.MultiheadAttention(dec_d, heads, batch_first=True)
        self.pose_out = mlp((dec_d, 256, 9))

    # ------------------------------------------------------------------ encode
    def encode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """-> latent (B, 64, C) and the frame's reconstruction targets (aux)."""
        aux = self.encoder(batch["rgb"], batch["depth"], batch["K"], batch["cam_T_world"], batch["hand"], batch["R_P"])
        return self.to_latent(aux.pop("tokens")), aux

    # ------------------------------------------------------------------ decode
    def _trunk(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.from_latent(z) + self.pos)

    def _pose(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pose_pool(self.pose_q.expand(h.shape[0], -1, -1), h, h, need_weights=False)[0][:, 0]
        pose = self.pose_out(pooled)
        return rot6d_to_matrix(pose[:, :6]), pose[:, 6:] * 0.01          # translation in metres

    def decode_pose(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Only the pose head: (R, t) the plug still has to move to be inserted (cheap; used by the
        world model's pose loss, the checks and the planner)."""
        return self._pose(self._trunk(z))

    def decode(self, z: torch.Tensor) -> dict:
        """z (B, 64, C) -> per-patch depth / validity / RGB / DINO and the plug pose (R, t)."""
        h = self._trunk(z)
        q = self.patch_dec(self.patch_queries[None].expand(z.shape[0], -1, -1), h)
        R, t = self._pose(h)
        return {"depth_patch": self.out_depth(q), "valid_logit": self.out_valid(q),
                "rgb_patch": self.out_rgb(q), "dino": self.out_dino(q), "R": R, "t": t}

    def forward(self, batch: dict) -> dict:
        z, aux = self.encode(batch)
        return {"z": z, "pred": self.decode(z), "target": aux}


def load_vae(path: str, device=None) -> TokenVAE:
    """Load a v3 checkpoint ({"model": state, "cfg": {...}}), eval mode. v1/v2 checkpoints belong to
    the archived point-cloud code ($Q/archive/world_model_code_v1_v2_2026-11-08.tar.gz)."""
    ck = torch.load(path, map_location="cpu")
    model = TokenVAE(**ck["cfg"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(device) if device is not None else model
