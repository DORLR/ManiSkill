"""Image encoder v4: two native-resolution crops (plug pins, socket opening) and their relation.

Input per frame (all real-robot quantities): the 320 x 320 wrist RGB-D image, the two crop origins
placed from the UNet (perception/crops.py), intrinsics, camera pose (FK + hand-eye mount), hand pose
(FK), port axes (board-plane fit). Depth is used as the camera gives it (0 = no return).

Per crop (128 x 128 px, cut from the image at native resolution, never resized), 16 x 16 tokens of
8 x 8 px each:
    CNN feature        ImageNet ResNet-18 up to stride 8 (trained), local detail at full resolution
    raw RGB patch      the 192 pixel values (exact colours / edges for sub-pixel positions)
    raw depth patch    64 values (m / 0.2), and where the camera returned a depth
    DINOv2 feature     frozen ViT-S/14 on the crop (126 px = 9 x 9 patches), interpolated to 16 x 16:
                       generic features for shapes never seen
    viewing ray        of the token's pixel centre, in port axes, with Fourier features (frequencies up
                       to 2^5 pi: neighbouring pixels differ by ~0.003 rad), so the absolute geometry of
                       every token -- hence where the two crops are relative to each other -- is known
    local position     learned 16 x 16 embedding, and a crop-type embedding (plug / opening)
Per crop one position token: the crop centre's viewing ray (Fourier) + its pixel position.
One geometry token: camera orientation in port axes, camera position relative to the hand.

Attention:
    1 per-crop self-attention (2 layers, shared, crop-type embedding tells them apart)
    2 relation: cross-attention both ways, plug tokens attend to opening tokens and opening tokens to
      plug tokens (3 layers each direction, separate weights)
    3 joint self-attention over [geometry, 2 position tokens, 256 plug, 256 opening tokens] (4 layers)
    4 read-out -> latent tokens
        "queries" (v4): 64 learned queries read the result (2 layers) -> 64 tokens. Measured 2026-11-11 on
            crop_ae_v4e: the 64 tokens came out as near copies of one vector (token-specific share of the
            variance 0.4 %, ~15 effective dimensions) -- nothing makes the queries read different places.
        "grid" (v5): no queries. The 2 x 2 neighbouring crop tokens are merged (concatenated, one linear
            layer), so every latent token belongs to one 16 x 16 px region of one crop: 8 x 8 tokens for
            the plug crop, then 8 x 8 for the opening crop (row-major), 128 in all; 2 more self-attention
            layers over them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..perception.crops import CROP
from .encoder import DEPTH_SCALE_M, DEPTH_UNIT_M, Dinov2, geometry_features

TOK = 8                      # pixels per token side
G = CROP // TOK              # 16 tokens per crop side
N_CROP_TOK = G * G           # 256
N_TOKENS = 64
N_FREQ = 6


def fourier(x: torch.Tensor, n: int = N_FREQ) -> torch.Tensor:
    """(..., k) -> (..., k * (1 + 2n)): x, sin(2^i pi x), cos(2^i pi x)."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


def cut(img: torch.Tensor, uv0: torch.Tensor) -> torch.Tensor:
    """img (B, H, W, ...) , uv0 (B, 2) long (u0, v0) -> (B, CROP, CROP, ...)."""
    B = img.shape[0]
    r = uv0[:, 1:2].long() + torch.arange(CROP, device=img.device)[None]          # (B, CROP) rows
    c = uv0[:, 0:1].long() + torch.arange(CROP, device=img.device)[None]          # (B, CROP) cols
    b = torch.arange(B, device=img.device)[:, None, None]
    return img[b, r[:, :, None], c[:, None, :]]


def crop_K(K: torch.Tensor, uv0: torch.Tensor) -> torch.Tensor:
    """Intrinsics of the crop (principal point shifted by the origin)."""
    Kc = K.float().clone()
    Kc[:, 0, 2] -= uv0[:, 0].float()
    Kc[:, 1, 2] -= uv0[:, 1].float()
    return Kc


def token_rays(K: torch.Tensor, uv0: torch.Tensor, cam_T_world: torch.Tensor, R_P: torch.Tensor) -> torch.Tensor:
    """Unit viewing ray (port axes) of every token's pixel centre: (B, 256, 3)."""
    B = K.shape[0]
    c = torch.arange(G, device=K.device, dtype=torch.float32) * TOK + (TOK - 1) / 2
    v, u = torch.meshgrid(c, c, indexing="ij")
    pix = torch.stack([u.flatten(), v.flatten(), torch.ones_like(u.flatten())], 0)[None].expand(B, -1, -1)
    rays = torch.linalg.solve(crop_K(K, uv0), pix)
    R_wc = cam_T_world[:, :3, :3].float().transpose(1, 2)
    return F.normalize((R_P.float().transpose(1, 2) @ R_wc @ rays).transpose(1, 2), dim=-1)


def centre_ray(K: torch.Tensor, uv0: torch.Tensor, cam_T_world: torch.Tensor, R_P: torch.Tensor) -> torch.Tensor:
    pix = torch.stack([uv0[:, 0].float() + (CROP - 1) / 2, uv0[:, 1].float() + (CROP - 1) / 2,
                       torch.ones_like(uv0[:, 0].float())], 1)[..., None]
    ray = torch.linalg.solve(K.float(), pix)
    R_wc = cam_T_world[:, :3, :3].float().transpose(1, 2)
    return F.normalize((R_P.float().transpose(1, 2) @ R_wc @ ray)[..., 0], dim=-1)


def patches(x: torch.Tensor) -> torch.Tensor:
    """(B, C, 128, 128) -> (B, 256, C * 64) row-major 8 x 8 patches."""
    return F.unfold(x, kernel_size=TOK, stride=TOK).transpose(1, 2)


class CropBackbone(nn.Module):
    """ImageNet ResNet-18 to stride 8: (B, 3, 128, 128) [0, 1] -> (B, 256, 128)."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).flatten(2).transpose(1, 2)


def _enc(d, heads, n):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), n)


def _dec(d, heads, n):
    return nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), n)


class CropEncoder(nn.Module):
    def __init__(self, d: int = 256, heads: int = 8, crop_layers: int = 2, relation_layers: int = 3,
                 joint_layers: int = 4, query_layers: int = 2, dino: bool = True, readout: str = "queries",
                 merge: int = 2):
        super().__init__()
        assert readout in ("queries", "grid") and G % merge == 0
        self.d, self.readout, self.merge = d, readout, merge
        self.n_tokens = N_TOKENS if readout == "queries" else 2 * (G // merge) ** 2
        self.backbone = CropBackbone()
        self.dino = Dinov2() if dino else None
        self.dino_dim = 384
        self.in_cnn = nn.Linear(128, d)
        self.in_rgb = nn.Linear(3 * TOK * TOK, d)
        self.in_depth = nn.Linear(2 * TOK * TOK, d)                  # depth and its validity
        self.in_dino = nn.Linear(self.dino_dim, d)
        self.in_ray = nn.Linear(3 * (1 + 2 * N_FREQ), d)
        self.local_pos = nn.Parameter(torch.randn(N_CROP_TOK, d) * 0.02)
        self.crop_type = nn.Parameter(torch.randn(2, 1, d) * 0.02)  # 0 plug, 1 opening
        self.pos_token = nn.Sequential(nn.Linear(3 * (1 + 2 * N_FREQ) + 2, d), nn.SiLU(), nn.Linear(d, d))
        self.geom = nn.Sequential(nn.Linear(9, d), nn.SiLU(), nn.Linear(d, d))
        self.geom_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.crop_attn = _enc(d, heads, crop_layers)
        self.plug_to_hole = _dec(d, heads, relation_layers)
        self.hole_to_plug = _dec(d, heads, relation_layers)
        self.joint = _enc(d, heads, joint_layers)
        if readout == "queries":
            self.queries = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
            self.query_layers = _dec(d, heads, query_layers)
        else:
            self.merge_proj = nn.Linear(merge * merge * d, d)
            self.grid_pos = nn.Parameter(torch.randn(self.n_tokens, d) * 0.02)
            self.grid_layers = _enc(d, heads, query_layers)
        self.norm = nn.LayerNorm(d)

    def merge_tokens(self, t: torch.Tensor) -> torch.Tensor:
        """One crop's tokens (B, 16 * 16, c), row-major -> (B, 8 * 8, merge^2 * c): the tokens of each
        merge x merge block side by side (inverse: crop_ae.CropAE.split_tokens)."""
        B, _, c = t.shape
        m, g = self.merge, G // self.merge
        return t.view(B, g, m, g, m, c).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, m * m * c)

    def _dino(self, rgb01: torch.Tensor) -> torch.Tensor:
        """(B, 3, 128, 128) -> (B, 256, 384) frozen DINOv2 features on a 16 x 16 grid."""
        if self.dino is None:                                         # unit tests only
            return rgb01.new_zeros(rgb01.shape[0], N_CROP_TOK, self.dino_dim)
        x = F.interpolate(rgb01, size=(126, 126), mode="bilinear", align_corners=False)
        f = self.dino(x)                                              # (B, 81, 384)
        f = f.transpose(1, 2).unflatten(2, (9, 9))
        return F.interpolate(f, size=(G, G), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)

    def crop_tokens(self, rgb, depth, uv0, K, cam_T_world, R_P, kind: int) -> dict:
        """One crop's tokens and its reconstruction targets."""
        x = cut(rgb, uv0).permute(0, 3, 1, 2).float() / 255.0                          # (B, 3, 128, 128)
        dm = cut(depth, uv0).float()[:, None] * (DEPTH_UNIT_M / DEPTH_SCALE_M)         # (B, 1, 128, 128)
        valid = (dm > 0).float()
        rgb_p, depth_p, valid_p = patches(x), patches(dm), patches(valid)
        dino = self._dino(x)
        rays = token_rays(K, uv0, cam_T_world, R_P)
        tok = (self.in_cnn(self.backbone(x)) + self.in_rgb(rgb_p) + self.in_depth(torch.cat([depth_p, valid_p], -1))
               + self.in_dino(dino) + self.in_ray(fourier(rays)) + self.local_pos + self.crop_type[kind])
        cr = centre_ray(K, uv0, cam_T_world, R_P)
        uvn = (uv0.float() + CROP / 2) / 320.0 * 2 - 1
        pos = self.pos_token(torch.cat([fourier(cr), uvn], -1))[:, None] + self.crop_type[kind]
        return {"tokens": tok, "pos": pos, "rgb_patch": rgb_p, "depth_patch": depth_p, "valid_patch": valid_p,
                "dino": dino}

    def forward(self, rgb, depth, plug_uv0, hole_uv0, K, cam_T_world, hand, R_P) -> dict:
        """rgb (B,320,320,3) uint8, depth (B,320,320) int [0.1 mm], plug_uv0 / hole_uv0 (B,2) crop
        origins, K (B,3,3), cam_T_world / hand (B,4,4), R_P (B,3,3) -> n_tokens tokens + per-crop targets."""
        p = self.crop_tokens(rgb, depth, plug_uv0, K, cam_T_world, R_P, 0)
        h = self.crop_tokens(rgb, depth, hole_uv0, K, cam_T_world, R_P, 1)
        tp, th = self.crop_attn(p["tokens"]), self.crop_attn(h["tokens"])
        tp, th = self.plug_to_hole(tp, th), self.hole_to_plug(th, tp)                  # relation, both ways
        g = self.geom(geometry_features(cam_T_world, hand, R_P))[:, None] + self.geom_type
        x = self.joint(torch.cat([g, p["pos"], h["pos"], tp, th], 1))
        if self.readout == "queries":
            q = self.query_layers(self.queries[None].expand(rgb.shape[0], -1, -1), x)
        else:
            tp, th = x[:, 3:3 + N_CROP_TOK], x[:, 3 + N_CROP_TOK:]
            q = self.merge_proj(torch.cat([self.merge_tokens(tp), self.merge_tokens(th)], 1))
            q = self.grid_layers(q + self.grid_pos)
        return {"tokens": self.norm(q), "plug": p, "hole": h}
