"""Latent autoencoder v4 / v5 over the two crops: frame -> latent (tokens x C) -> what it must keep.

Encoder: CropEncoder (models/crop_encoder.py) -> LayerNorm -> linear to C channels per token.
Deterministic (no KL), no joint token (the robot state goes to the world model separately).
Latent: 64 tokens (v4, read-out "queries") or 128 tokens = 8 x 8 per crop, each one 16 x 16 px region
(v5, read-out "grid"; why: crop_encoder.py).

Decoder, from the latent only.
    v4: one query per crop token (2 x 256) reads all latent tokens (cross-attention).
    v5: every latent token is split back into the 2 x 2 crop tokens of its own region (one linear layer),
        then self-attention over the 2 x 256 crop tokens (so a region is rebuilt from its own token, with
        the others as context).
Each crop token reconstructs its 8 x 8 pixel patch -- RGB, depth, depth validity, the frozen DINOv2 feature -- and,
for the pose, a per-pixel motion "where this point is when the plug is inserted" (3-D, hand-centred
port axes) and a weight. The remaining rigid motion (R, t) of the plug is solved from the crop pixels'
3-D points p (depth + crop intrinsics + camera pose) and p + motion by the weighted rigid fit of v3g
(geometry.weighted_rigid_fit); nothing in the solver is learned. Labels: FK of the grasp's one
insertion; the plug segmentation as a loss mask only.

Reading a pose from a latent (also a predicted one) needs that frame's geometry: K, camera pose, hand
pose, port axes and the two crop origins.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry import pixel_points, weighted_rigid_fit
from .crop_encoder import CROP, G, N_CROP_TOK, TOK, CropEncoder, crop_K
from .encoder import DEPTH_SCALE_M

MOTION_SCALE_M = 0.01


def unpatch(x: torch.Tensor, channels: int) -> torch.Tensor:
    """(B, 256, channels * 64) -> (B, channels, 128, 128) (inverse of crop_encoder.patches)."""
    return torch.nn.functional.fold(x.transpose(1, 2), (CROP, CROP), kernel_size=TOK, stride=TOK)


GEOM_KEYS = ("K", "cam_T_world", "hand", "R_P", "plug_uv0", "hole_uv0")


class CropAE(nn.Module):
    def __init__(self, encoder: dict | None = None, latent_ch: int = 16, dec_d: int = 256, dec_layers: int = 2,
                 patch_layers: int = 3, heads: int = 8):
        super().__init__()
        self.encoder = CropEncoder(**(encoder or {}))
        d = self.encoder.d
        self.latent_ch = latent_ch
        self.to_latent = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent_ch))
        self.from_latent = nn.Linear(latent_ch, dec_d)
        self.grid, self.n_tokens = self.encoder.readout == "grid", self.encoder.n_tokens
        self.pos = nn.Parameter(torch.randn(self.n_tokens, dec_d) * 0.02)
        self.trunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dec_d, heads, 2 * dec_d, batch_first=True, norm_first=True), dec_layers)
        self.patch_queries = nn.Parameter(torch.randn(2 * N_CROP_TOK, dec_d) * 0.02)
        if self.grid:
            self.split = nn.Linear(dec_d, self.encoder.merge ** 2 * dec_d)
            self.patch_dec = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dec_d, heads, 2 * dec_d, batch_first=True, norm_first=True), patch_layers)
        else:
            self.patch_dec = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(dec_d, heads, 2 * dec_d, batch_first=True, norm_first=True), patch_layers)
        px = TOK * TOK
        self.out_rgb = nn.Linear(dec_d, 3 * px)
        self.out_depth = nn.Linear(dec_d, px)
        self.out_valid = nn.Linear(dec_d, px)
        self.out_dino = nn.Linear(dec_d, self.encoder.dino_dim)
        self.out_motion = nn.Linear(dec_d, 3 * px)
        self.out_weight = nn.Linear(dec_d, px)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        aux = self.encoder(batch["rgb"], batch["depth"], batch["plug_uv0"], batch["hole_uv0"], batch["K"],
                           batch["cam_T_world"], batch["hand"], batch["R_P"])
        return self.to_latent(aux.pop("tokens")), aux

    def split_tokens(self, t: torch.Tensor) -> torch.Tensor:
        """One crop's latent-token features (B, 8 * 8, merge^2 * c) -> its crop tokens (B, 16 * 16, c), row-major
        (inverse of CropEncoder.merge_tokens)."""
        B = t.shape[0]
        m, g = self.encoder.merge, G // self.encoder.merge
        return t.view(B, g, g, m, m, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, N_CROP_TOK, -1)

    def decode(self, z: torch.Tensor) -> dict:
        """z (B, n_tokens, C) -> per crop ("plug", "hole") patch predictions (B, 256, ...)."""
        h = self.trunk(self.from_latent(z) + self.pos)
        if self.grid:
            h = self.split(h)
            n = self.n_tokens // 2
            q = self.patch_dec(torch.cat([self.split_tokens(h[:, :n]), self.split_tokens(h[:, n:])], 1) + self.patch_queries)
        else:
            q = self.patch_dec(self.patch_queries[None].expand(z.shape[0], -1, -1), h)
        out = {}
        for k, sl in (("plug", slice(0, N_CROP_TOK)), ("hole", slice(N_CROP_TOK, 2 * N_CROP_TOK))):
            qq = q[:, sl]
            out[k] = {"rgb_patch": self.out_rgb(qq), "depth_patch": self.out_depth(qq), "valid_logit": self.out_valid(qq),
                      "dino": self.out_dino(qq), "motion_patch": self.out_motion(qq), "weight_logit": self.out_weight(qq)}
        return out

    # ------------------------------------------------------------------ pose
    @staticmethod
    def crop_points(depth_patch: torch.Tensor, geom: dict, which: str) -> torch.Tensor:
        """3-D points (hand-centred port axes) of every pixel of a crop: (B, 128*128, 3)."""
        depth_m = unpatch(depth_patch.float(), 1)[:, 0] * DEPTH_SCALE_M
        Kc = crop_K(geom["K"], geom[f"{which}_uv0"])
        return pixel_points(depth_m, Kc, geom["cam_T_world"], geom["hand"], geom["R_P"], crop=CROP)

    @staticmethod
    def solve_pose(dec: dict, geom: dict, target: dict | None = None):
        """(R, t in m) of the plug's remaining motion from both crops' pixels. With `target` (the
        encoder's aux), the true depth gives the points (training); else the decoded depth and
        validity (any latent, also predicted)."""
        with torch.autocast("cuda", enabled=False):
            P, Q, W = [], [], []
            for k in ("plug", "hole"):
                d = dec[k]
                if target is not None:
                    dp = target[k]["depth_patch"]
                    valid = unpatch(target[k]["valid_patch"].float(), 1).flatten(1)
                else:
                    dp = d["depth_patch"].detach()
                    valid = torch.sigmoid(unpatch(d["valid_logit"].detach().float(), 1).flatten(1))
                p = CropAE.crop_points(dp, geom, k).detach()
                motion = unpatch(d["motion_patch"].float(), 3).flatten(2).transpose(1, 2) * MOTION_SCALE_M
                w = torch.sigmoid(unpatch(d["weight_logit"].float(), 1).flatten(1))
                P.append(p), Q.append(p + motion), W.append(w * valid)
            return weighted_rigid_fit(torch.cat(P, 1), torch.cat(Q, 1), torch.cat(W, 1))

    def decode_pose(self, z: torch.Tensor, geom: dict):
        return self.solve_pose(self.decode(z), geom)

    def forward(self, batch: dict) -> dict:
        z, aux = self.encode(batch)
        dec = self.decode(z)
        geom = {k: batch[k] for k in GEOM_KEYS}
        R, t = self.solve_pose(dec, geom, aux)
        return {"z": z, "pred": dec, "R": R, "t": t, "target": aux}


def load_crop_ae(path: str, device=None) -> CropAE:
    ck = torch.load(path, map_location="cpu")
    m = CropAE(**ck["cfg"]["model"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m.to(device) if device is not None else m
