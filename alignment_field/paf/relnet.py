"""Relational plug-port network (plan "A"): what the camera sees of the plug and
of the port, as two token sets that attend to each other.

Inputs per view (all real-robot quantities): wrist RGB (256x256), aligned depth
(mm), intrinsics K, extrinsics cam_T_world (FK + hand-eye), hand pose H (FK),
port axes R_P (board-plane fit + gravity, once per episode).

Pipeline (all on the GPU, differentiable where it matters):
  1. back-project depth -> points in hand-centred port axes (x into the board);
  2. per-pixel features: frozen DINOv2 ViT-S/14 patch features (Oquab et al.
     2023) bilinearly sampled at each point's pixel, plus RGB and xyz;
  3. board depth x_b = robust mode of x over far points; port points are within
     4 mm of the board plane, plug points are in front of it (x < x_b - 4 mm)
     and inside the region in front of the hand;
  4. tokens: random centres per set, k-nearest-neighbour groups, a shared
     mini-PointNet over (relative xyz, point features) -> token feature, plus a
     learned embedding of the centre position (PointNet++ set abstraction);
  5. a transformer with self-attention inside each set and cross-attention
     plug<->port (TAX-Pose / RPDiff style relational reasoning);
  6. attention pooling -> relative state u (6, sigma units).
Outputs use the keys of AlignNet so training and evaluation code is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import mlp

IMG = 256
DINO_RES = 448          # 32 x 32 patches of 14 px
N_PORT, N_PLUG, K_NN = 256, 128, 16
N_CAND = 2048          # candidate points per set before grouping
BOARD_BAND = 0.004      # m
BOX_YZ = 0.045          # m, lateral half size kept around the hand axis
PLUG_X = (-0.02, None)  # plug region starts 2 cm behind the hand origin


def backproject_torch(depth_mm, K):
    """depth (B,H,W) mm -> camera points (B,H,W,3) m and validity (B,H,W)."""
    B, H, W = depth_mm.shape
    d = depth_mm.float() / 1000.0
    v, u = torch.meshgrid(torch.arange(H, device=d.device), torch.arange(W, device=d.device), indexing="ij")
    fx, fy, cx, cy = K[:, 0, 0, None, None], K[:, 1, 1, None, None], K[:, 0, 2, None, None], K[:, 1, 2, None, None]
    x = (u[None] - cx) * d / fx
    y = (v[None] - cy) * d / fy
    return torch.stack([x, y, d], -1), d > 0


def to_port_frame(pts_cam, cam_T_world, H, R_P):
    """camera points -> hand-centred port axes. All (B,...) batched."""
    world_T_cam = torch.linalg.inv(cam_T_world)
    p = pts_cam @ world_T_cam[:, None, :3, :3].transpose(-1, -2) + world_T_cam[:, None, None, :3, 3]
    p = p - H[:, None, None, :3, 3]
    return p @ R_P[:, None]  # (B,H,W,3), coordinates along the columns of R_P


class DinoFeatures(nn.Module):
    """Frozen DINOv2 ViT-S/14 patch features, upsampled to the image grid."""

    def __init__(self):
        super().__init__()
        import os
        local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.isdir(local):  # no network access needed once cached
            self.net = torch.hub.load(local, "dinov2_vits14", source="local", verbose=False)
        else:
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.dim = 384

    @torch.no_grad()
    def forward(self, rgb_uint8):  # (B,H,W,3) -> (B,384,32,32)
        x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(DINO_RES, DINO_RES), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tok = self.net.forward_features(x)["x_norm_patchtokens"]  # (B, 32*32, 384)
        g = DINO_RES // 14
        return tok.transpose(1, 2).reshape(x.shape[0], self.dim, g, g)


def _sample(mask, n, gen=None):
    """Random indices of True entries per batch row (with replacement if few)."""
    B, N = mask.shape
    w = mask.float() + 1e-6 * (~mask).float()
    return torch.multinomial(w, n, replacement=True, generator=gen)


class SetAbstraction(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.local = mlp((in_dim + 3, 128, out_dim), last_act=True)
        self.pos = mlp((3, 64, out_dim))

    def forward(self, xyz, feat, mask, n_centres):
        """xyz (B,N,3) m, feat (B,N,C), mask (B,N) -> tokens (B,n,D), centres (B,n,3)."""
        idx = _sample(mask, n_centres)
        centres = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))
        with torch.autocast("cuda", enabled=False):
            d = torch.cdist(centres.float(), xyz.float())   # (B,n,N), float32 distances
        d = d + (~mask)[:, None, :].float() * 1e3           # never group invalid points
        nn_idx = d.topk(K_NN, dim=-1, largest=False).indices  # (B,n,k)
        B, n, k = nn_idx.shape
        flat = nn_idx.reshape(B, n * k)
        g_xyz = torch.gather(xyz, 1, flat[..., None].expand(-1, -1, 3)).reshape(B, n, k, 3)
        g_feat = torch.gather(feat, 1, flat[..., None].expand(-1, -1, feat.shape[-1])).reshape(B, n, k, -1)
        rel = ((g_xyz - centres[:, :, None]) / 0.005).float()  # ~5 mm neighbourhoods, fp32 before the MLP
        tok = self.local(torch.cat([rel, g_feat], -1)).max(2).values
        return tok + self.pos(centres / 0.05), centres


class CrossBlock(nn.Module):
    def __init__(self, d=256, heads=8):
        super().__init__()
        self.sa_a = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, norm_first=True)
        self.sa_b = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, norm_first=True)
        self.ca_ab = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca_ba = nn.MultiheadAttention(d, heads, batch_first=True)
        self.na, self.nb = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, a, b):
        a, b = self.sa_a(a), self.sa_b(b)
        a = a + self.ca_ab(self.na(a), b, b, need_weights=False)[0]
        b = b + self.ca_ba(self.nb(b), a, a, need_weights=False)[0]
        return a, b


class RelNet(nn.Module):
    def __init__(self, d=256, layers=4, use_dino=True, focus=None, n_cand=N_CAND, n_plug=N_PLUG):
        """focus: if set (m), sample board and plug points only within +-focus (lateral) of the
        pin tips, located from the plug points themselves (the plug points closest to the board);
        the front 3 cm of the plug. Concentrates the sampled points on the pins and the holes."""
        super().__init__()
        self.focus, self.n_cand, self.n_plug = focus, n_cand, n_plug
        self.dino = DinoFeatures() if use_dino else None
        fdim = (384 if use_dino else 0) + 3 + 3          # dino + rgb + xyz
        self.point_in = mlp((fdim, 128, 128), last_act=True)
        self.sa_port = SetAbstraction(128, d)
        self.sa_plug = SetAbstraction(128, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.blocks = nn.ModuleList([CrossBlock(d) for _ in range(layers)])
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 8, batch_first=True)
        self.head = mlp((d, 256, 6))
        self.id_head = mlp((d, 128, 1))

    def tokens(self, rgb, depth_mm, K, cam_T_world, H, R_P):
        B = rgb.shape[0]
        # geometry in float32: bf16 would round 5 cm coordinates by ~0.2 mm
        with torch.autocast("cuda", enabled=False):
            pts_cam, valid = backproject_torch(depth_mm, K.float())
            p = to_port_frame(pts_cam, cam_T_world.float(), H.float(), R_P.float()).reshape(B, -1, 3)
        valid = valid.reshape(B, -1)
        rgbf = rgb.reshape(B, -1, 3).float() / 255.0
        fmap = self.dino(rgb) if self.dino is not None else None          # (B,384,32,32)
        lateral = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
        far = valid & lateral & (p[..., 0] > 0.0)
        # robust board depth: 80th percentile of x over points in front of the hand
        xs = torch.where(far, p[..., 0], torch.full_like(p[..., 0], float("nan")))
        x_b = torch.nanquantile(xs, 0.8, dim=1, keepdim=True).nan_to_num(0.05)
        port = valid & lateral & ((p[..., 0] - x_b).abs() < BOARD_BAND)
        plug = valid & lateral & (p[..., 0] < x_b - BOARD_BAND) & (p[..., 0] > PLUG_X[0])
        if self.focus is not None:
            port, plug = self._focus(p, port, plug)
        out = []
        for mask, sa, n in ((plug, self.sa_plug, self.n_plug), (port, self.sa_port, N_PORT)):
            idx = _sample(mask, self.n_cand)                                  # (B,M)
            xyz = torch.gather(p, 1, idx[..., None].expand(-1, -1, 3))
            col = torch.gather(rgbf, 1, idx[..., None].expand(-1, -1, 3))
            m = torch.gather(mask, 1, idx)
            feats = [col, xyz / 0.05]
            if fmap is not None:
                Hh, Ww = rgb.shape[1], rgb.shape[2]
                vv, uu = (idx // Ww).float(), (idx % Ww).float()
                grid = torch.stack([(uu + 0.5) / Ww * 2 - 1, (vv + 0.5) / Hh * 2 - 1], -1)[:, None]
                dino = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)  # (B,384,1,M)
                feats.insert(0, dino[:, :, 0].transpose(1, 2))
            h = self.point_in(torch.cat(feats, -1))
            tok, _ = sa(xyz, h, m, n)
            out.append(tok)
        return out[0] + self.type_emb[0], out[1] + self.type_emb[1]

    def _focus(self, p, port, plug):
        """Pin-tip centre from the plug points nearest the board; keep points near it."""
        x = torch.where(plug, p[..., 0], torch.full_like(p[..., 0], -1.0))
        front = x.max(1, keepdim=True).values                                # (B,1)
        tips = plug & (p[..., 0] > front - 0.004)
        yz = torch.where(tips[..., None], p[..., 1:], torch.full_like(p[..., 1:], float("nan")))
        c = yz.nanmedian(1).values.nan_to_num(0.0)                          # (B,2)
        near = ((p[..., 1:] - c[:, None]).abs() < self.focus).all(-1)
        port_f = port & near
        plug_f = plug & near & (p[..., 0] > front - 0.03)
        ok_port = port_f.sum(1, keepdim=True) >= 64
        ok_plug = plug_f.sum(1, keepdim=True) >= 32
        return torch.where(ok_port, port_f, port), torch.where(ok_plug, plug_f, plug)

    def forward(self, rgb, depth_mm, K, cam_T_world, H, R_P):
        a, b = self.tokens(rgb, depth_mm, K, cam_T_world, H, R_P)
        for blk in self.blocks:
            a, b = blk(a, b)
        z = self.pool(self.pool_q.expand(a.shape[0], -1, -1), torch.cat([a, b], 1),
                      torch.cat([a, b], 1), need_weights=False)[0][:, 0]
        u = self.head(z)
        return {"q_pc": u, "q_hat": u, "id_logit": self.id_head(z).squeeze(-1)}
