"""Image encoder (v3): wrist RGB-D frame -> 64 latent-query tokens.

Every pixel is used as it is. No segmentation, no point cloud, no crop rule, no depth completion:
pixels without a depth return keep depth 0 (the D405 gives 0 there), exactly as the camera outputs.

Per image patch (14 x 14 px; the 320 x 320 crop is resized to 322 x 322 -> 23 x 23 = 529 patches,
the native DINOv2 patch count for this crop, so nothing is upsampled for nothing):
    DINOv2 ViT-S/14 feature of the RGB patch      frozen, pretrained (DINOv3 is gated on Hugging
                                                  Face for this account; DINOv2 as agreed)
    raw RGB patch (588 values)                    exact colours, not only DINO's summary
    raw depth patch (196 values, metres / 0.2)    0 where the camera has no return, kept as 0
    viewing ray of the patch centre               from the intrinsics, rotated into port axes, so
                                                  the network can turn depth into geometry itself
    learned 2-D position embedding
Plus one geometry token: the camera orientation in port axes and the camera position relative to
the hand (from FK + hand-eye mount + the board-plane fit), all real-robot quantities.

All patch tokens and the geometry token go through a transformer (self-attention: the relations
between plug, hole and board are learned, not designed); 64 learned queries then read the result
(cross-attention) and form the latent's 64 tokens (an 8 x 8 grid for the world model).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TOKENS = 64            # latent tokens = CDiT's 8 x 8 grid
IMG = 322                # 23 x 14: the 320 px crop at DINOv2's patch size, no upsampling
PATCH = 14
GRID = IMG // PATCH      # 23
N_PATCH = GRID * GRID    # 529
DEPTH_UNIT_M = 1e-4      # stored depth unit (wm.data.schema.DEPTH_UNIT_M)
DEPTH_SCALE_M = 0.2      # depth / 0.2 m -> about [0, 1.5] in this scene


class Dinov2(nn.Module):
    """Frozen DINOv2 ViT-S/14 patch features (the checkpoint already cached for v1/v2)."""

    def __init__(self):
        super().__init__()
        hub = os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")), "hub")
        local = os.path.join(hub, "facebookresearch_dinov2_main")
        if os.path.isdir(local):
            self.net = torch.hub.load(local, "dinov2_vits14", source="local", verbose=False)
        else:
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        self.net.eval().requires_grad_(False)
        self.dim = 384

    def train(self, mode: bool = True):          # stays in eval mode (frozen)
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        """rgb01 (B, 3, 322, 322) in [0, 1], ImageNet-normalised inside -> (B, 529, 384)."""
        mean = rgb01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self.net.forward_features((rgb01 - mean) / std)["x_norm_patchtokens"]


class _NoDino(nn.Module):
    """Zero features of DINOv2's size: CPU unit tests only (no weights to load)."""
    dim = 384

    def forward(self, rgb01):
        return rgb01.new_zeros(rgb01.shape[0], N_PATCH, self.dim)


def patchify(x: torch.Tensor) -> torch.Tensor:
    """(B, C, 322, 322) -> (B, 529, C * 14 * 14), row-major patches."""
    return F.unfold(x, kernel_size=PATCH, stride=PATCH).transpose(1, 2)


def rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two columns (inverse of geometry.rot6d_to_matrix)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def prepare(rgb: torch.Tensor, depth: torch.Tensor) -> dict:
    """Resize the 320 x 320 crop to 322 x 322 and cut patches.

    rgb (B, 320, 320, 3) uint8, depth (B, 320, 320) int [0.1 mm] -> dict of
        rgb01 (B, 3, 322, 322), rgb_patch (B, 529, 588), depth_patch (B, 529, 196) [m / 0.2].
    Depth uses nearest-neighbour resizing so a missing return stays exactly 0 (no blending of 0 with
    valid depth). These patch tensors are also the VAE's reconstruction targets.
    """
    x = rgb.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    d = depth.float()[:, None] * (DEPTH_UNIT_M / DEPTH_SCALE_M)
    d = F.interpolate(d, size=(IMG, IMG), mode="nearest")
    return {"rgb01": x, "rgb_patch": patchify(x), "depth_patch": patchify(d)}


def patch_rays(K: torch.Tensor, cam_T_world: torch.Tensor, R_P: torch.Tensor, crop: int) -> torch.Tensor:
    """Unit viewing ray of every patch centre in port axes, (B, 529, 3).

    K is the crop's intrinsic matrix (crop x crop pixels); patch centres of the 322 grid are mapped
    back to crop pixels. cam_T_world maps world -> camera, so camera -> world rotation is its
    transpose; R_P's columns are the port axes in world, so world -> port is R_P^T."""
    B = K.shape[0]
    c = (torch.arange(GRID, device=K.device, dtype=torch.float32) + 0.5) * PATCH * crop / IMG - 0.5
    v, u = torch.meshgrid(c, c, indexing="ij")
    pix = torch.stack([u.flatten(), v.flatten(), torch.ones_like(u.flatten())], 0)        # (3, 529)
    rays = torch.linalg.solve(K.float(), pix[None].expand(B, -1, -1))                       # camera frame
    R_wc = cam_T_world[:, :3, :3].float().transpose(1, 2)
    rays = R_P.float().transpose(1, 2) @ R_wc @ rays
    return F.normalize(rays.transpose(1, 2), dim=-1)


def geometry_features(cam_T_world: torch.Tensor, hand: torch.Tensor, R_P: torch.Tensor) -> torch.Tensor:
    """(B, 9): camera orientation in port axes (6D) and camera position relative to the hand in port
    axes (cm). The board's position is not known to the robot and is not given."""
    R_wc = cam_T_world[:, :3, :3].float().transpose(1, 2)
    p_cam = -(R_wc @ cam_T_world[:, :3, 3:].float())[..., 0]                                 # camera centre, world
    Rt = R_P.float().transpose(1, 2)
    rel = (Rt @ (p_cam - hand[:, :3, 3].float())[..., None])[..., 0] * 100.0
    return torch.cat([rot6d(Rt @ R_wc), rel], -1)


class ImageEncoder(nn.Module):
    def __init__(self, d: int = 256, layers: int = 6, query_layers: int = 2, heads: int = 8, crop: int = 320,
                 dino: bool = True):
        super().__init__()
        self.d, self.crop = d, crop
        self.dino = Dinov2() if dino else _NoDino()          # dino=False: unit tests only
        self.in_dino = nn.Linear(self.dino.dim, d)
        self.in_rgb = nn.Linear(3 * PATCH * PATCH, d)
        self.in_depth = nn.Linear(PATCH * PATCH, d)
        self.in_ray = nn.Linear(3, d)
        self.patch_pos = nn.Parameter(torch.randn(N_PATCH, d) * 0.02)
        self.geom = nn.Sequential(nn.Linear(9, d), nn.SiLU(), nn.Linear(d, d))
        self.geom_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(), layers)
        self.queries = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        self.query_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True) for _ in range(query_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, rgb, depth, K, cam_T_world, hand, R_P) -> dict:
        """rgb (B,320,320,3) uint8, depth (B,320,320) int [0.1 mm], K (B,3,3), cam_T_world / hand
        (B,4,4), R_P (B,3,3) -> tokens (B,64,d) and the reconstruction targets (dino, rgb_patch,
        depth_patch), which are this frame's own pixels / frozen features (no labels)."""
        p = prepare(rgb, depth)
        dino = self.dino(p["rgb01"])
        x = (self.in_dino(dino) + self.in_rgb(p["rgb_patch"]) + self.in_depth(p["depth_patch"])
             + self.in_ray(patch_rays(K, cam_T_world, R_P, self.crop)) + self.patch_pos)
        g = self.geom(geometry_features(cam_T_world, hand, R_P))[:, None] + self.geom_type
        h = self.encoder(torch.cat([g, x], 1))
        q = self.queries[None].expand(rgb.shape[0], -1, -1)
        for layer in self.query_layers:
            q = layer(q, h)
        return {"tokens": self.norm(q), "dino": dino, "rgb_patch": p["rgb_patch"], "depth_patch": p["depth_patch"]}
