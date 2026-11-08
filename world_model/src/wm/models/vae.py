"""Latent autoencoder (v3): frame -> latent (64 tokens x C channels) -> what the latent must keep.

Encoder: ImageEncoder (models/encoder.py) -> LayerNorm -> linear to C channels per token.
Deterministic: no KL (removed in v2; LayerNorm bounds the token magnitudes and the latents are
rescaled to unit RMS before the world model). No joint token: the robot state (joints / hand pose)
is given to the world model and the policy separately, as in Ctrl-World.

Decoder, from the latent only:
    patch decoder   one query per image patch (529) reads the latent and reconstructs that patch:
                    raw depth (incl. where the camera has no return: a validity logit per pixel),
                    raw RGB, and the frozen DINOv2 feature -- the whole frame as the camera gave it
    geometric pose  (v3g, replaces v3's direct pose regressor, which confused the grasp pitch with
                    the plug's misalignment) every pixel also gets a predicted motion "where will this
                    point be when the plug is inserted" (3-D, hand-centred port axes) and a weight;
                    the rigid motion the plug still needs is then *solved* from the pixels' current
                    3-D points p (decoded depth + intrinsics + camera pose) and p + motion by a
                    weighted rigid fit (geometry.weighted_rigid_fit, the Kabsch objective). Nothing
                    in the solver is learned. Trained against FK of the one insertion of this grasp
                    (training/losses.py). The world model's predictions are read through it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry import pixel_points, weighted_rigid_fit
from .encoder import DEPTH_SCALE_M, IMG, N_PATCH, N_TOKENS, PATCH, ImageEncoder

MOTION_SCALE_M = 0.01      # predicted per-pixel motion in centimetres


def unpatchify(x: torch.Tensor, channels: int) -> torch.Tensor:
    """(B, 529, channels * 14 * 14) row-major patches -> (B, channels, 322, 322) (inverse of encoder.patchify)."""
    return torch.nn.functional.fold(x.transpose(1, 2), (IMG, IMG), kernel_size=PATCH, stride=PATCH)


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
        self.out_motion = nn.Linear(dec_d, 3 * PATCH * PATCH)      # per pixel: motion to inserted [cm]
        self.out_weight = nn.Linear(dec_d, PATCH * PATCH)          # per pixel: weight logit in the fit

    # ------------------------------------------------------------------ encode
    def encode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """-> latent (B, 64, C) and the frame's reconstruction targets (aux)."""
        aux = self.encoder(batch["rgb"], batch["depth"], batch["K"], batch["cam_T_world"], batch["hand"], batch["R_P"])
        return self.to_latent(aux.pop("tokens")), aux

    # ------------------------------------------------------------------ decode
    def _trunk(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.from_latent(z) + self.pos)

    def decode(self, z: torch.Tensor) -> dict:
        """z (B, 64, C) -> per-patch depth / validity / RGB / DINO / motion / weight."""
        h = self._trunk(z)
        q = self.patch_dec(self.patch_queries[None].expand(z.shape[0], -1, -1), h)
        return {"depth_patch": self.out_depth(q), "valid_logit": self.out_valid(q), "rgb_patch": self.out_rgb(q),
                "dino": self.out_dino(q), "motion_patch": self.out_motion(q), "weight_logit": self.out_weight(q)}

    @staticmethod
    def solve_pose(dec: dict, geom: dict, depth_patch: torch.Tensor | None = None):
        """Rigid motion the plug still needs, (R, t in m), from decoded per-pixel motions and weights.

        geom: K, cam_T_world, hand, R_P of the frame (FK, hand-eye mount, board-plane fit). Current 3-D
        points come from `depth_patch` if given, else from the decoded depth (what is available for a
        predicted latent); they are not differentiated through (depth has its own loss). Pixels count
        with weight sigmoid(weight) x P(depth return)."""
        with torch.autocast("cuda", enabled=False):      # geometry and the fit in fp32
            return TokenVAE._solve_fp32(dec, geom, depth_patch)

    @staticmethod
    def _solve_fp32(dec: dict, geom: dict, depth_patch: torch.Tensor | None):
        d = dec["depth_patch"] if depth_patch is None else depth_patch
        depth_m = unpatchify(d.detach().float(), 1)[:, 0] * DEPTH_SCALE_M
        p = pixel_points(depth_m, geom["K"], geom["cam_T_world"], geom["hand"], geom["R_P"], crop=320)
        motion = unpatchify(dec["motion_patch"].float(), 3).flatten(2).transpose(1, 2) * MOTION_SCALE_M
        w = torch.sigmoid(unpatchify(dec["weight_logit"].float(), 1).flatten(1))
        if depth_patch is not None:
            valid = (depth_m.flatten(1) > 0).float()
        else:
            valid = torch.sigmoid(unpatchify(dec["valid_logit"].detach().float(), 1).flatten(1))
        return weighted_rigid_fit(p.detach(), p.detach() + motion, w * valid)

    def decode_pose(self, z: torch.Tensor, geom: dict):
        """What the world model and the checks need: (R, t) solved from the decoded frame."""
        return self.solve_pose(self.decode(z), geom)

    def forward(self, batch: dict) -> dict:
        z, aux = self.encode(batch)
        dec = self.decode(z)
        geom = {k: batch[k] for k in ("K", "cam_T_world", "hand", "R_P")}
        R, t = self.solve_pose(dec, geom)
        return {"z": z, "pred": {**dec, "R": R, "t": t}, "target": aux}


def load_vae(path: str, device=None) -> TokenVAE:
    """Load a v3 checkpoint ({"model": state, "cfg": {...}}), eval mode. v1/v2 checkpoints belong to
    the archived point-cloud code ($Q/archive/world_model_code_v1_v2_2026-11-08.tar.gz)."""
    ck = torch.load(path, map_location="cpu")
    model = TokenVAE(**ck["cfg"]["model"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model.to(device) if device is not None else model
