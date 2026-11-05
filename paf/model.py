"""Perception network: point cloud -> coarse 6D state, RGB -> sub-mm residual.

    f_pc  = PointNet(points in hand-centred port axes)          (DP3 / PointNet)
    q_pc  = MLP(f_pc)                       coarse relative state, all 6 dof
    f_rgb = ResNet18(native-resolution crop around tips/holes)  (RVT-2 zoom idea)
    q_hat = q_pc + MLP([f_pc, f_rgb, q_pc])  fine residual   (CFVS coarse->fine)
    p_id  = sigmoid(MLP([f_pc, f_rgb]))      in-distribution confidence (SID)

All state outputs are in sigma units (1 mm, 1 deg) of paf.config.SIGMA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def mlp(sizes, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class PointNet(nn.Module):
    def __init__(self, in_dim=6, width=(64, 128, 256)):
        super().__init__()
        self.point_mlp = mlp((in_dim,) + width, last_act=True)
        self.out_dim = 2 * width[-1]

    def forward(self, pts):                       # (B, N, 6)
        f = self.point_mlp(pts)                   # (B, N, C)
        return torch.cat([f.max(1).values, f.mean(1)], -1)


class RGBNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Identity()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_dim = 512

    def forward(self, rgb_uint8):                 # (B, H, W, 3) uint8
        x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0
        return self.net((x - self.mean) / self.std)


class AlignNet(nn.Module):
    def __init__(self, pretrained=True, use_rgb=True, rgb_drop=0.0):
        super().__init__()
        self.use_rgb = use_rgb
        self.rgb_drop = rgb_drop  # training only: drop the whole RGB feature per sample
        self.pc = PointNet()
        self.q_pc_head = mlp((self.pc.out_dim, 256, 6))
        self.rgb = RGBNet(pretrained) if use_rgb else None
        fdim = self.pc.out_dim + (self.rgb.out_dim if use_rgb else 0)
        self.fine_head = mlp((fdim + 6, 512, 256, 6))
        self.id_head = mlp((fdim, 256, 1))

    def forward(self, rgb, pts):
        f_pc = self.pc(pts)
        q_pc = self.q_pc_head(f_pc)
        feats = [f_pc]
        if self.use_rgb:
            f_rgb = self.rgb(rgb)
            if self.training and self.rgb_drop > 0:
                keep = (torch.rand(f_rgb.shape[0], 1, device=f_rgb.device) >= self.rgb_drop).to(f_rgb.dtype)
                f_rgb = f_rgb * keep
            feats.append(f_rgb)
        f = torch.cat(feats, -1)
        q_hat = q_pc + self.fine_head(torch.cat([f, q_pc.detach()], -1))
        return {"q_pc": q_pc, "q_hat": q_hat, "id_logit": self.id_head(f).squeeze(-1)}
