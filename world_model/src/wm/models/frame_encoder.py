"""Image encoder v6 (decided with the user 2026-11-12): a plain ResNet on the whole wrist image, no reconstruction.

Why: the v4 / v5 pipeline (UNet -> two crops -> crop autoencoder with a decoder and ten losses) kept less of what
decides insert / jam than a plain ResNet-18 trained directly (pin tip across the hole axis, held-out exploration,
pins at the face, tip within 1 mm: 0.39-0.77 mm against 0.22 mm; docs/2026-11-12_v5.md), and every repair added
structure. So: no UNet, no crops, no decoder, no reconstruction target.

    FrameEncoder   RGB (3) + depth (1) + depth validity (1) of the 320 x 320 image -> ResNet-34 (the user: "a little
                   larger" than ResNet-18; `arch`) (ImageNet start,
                   batch-norm statistics frozen) to stride 16 or 32 -> 1 x 1 convolution to `ch` channels -> every
                   position's vector normalised (zero mean, unit variance over its channels): a feature map
                   (ch, 20, 20) or (ch, 10, 10) whose scale is fixed, which is what the world model predicts and
                   what every policy reads.
    LabelHead      training only: from the feature map (+ the camera geometry, to speak in port axes) the remaining
                   rigid motion of the hand to its inserted pose (translation mm, rotation vector deg; label = FK of
                   the grasp's one insertion, geometry.insertion_motion) and "inserted now" (the contact switch).
                   It keeps the encoder from collapsing when it is trained together with the world model and makes
                   it look at where the plug is relative to the hole. It is not used by anything afterwards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..geometry import _exp_so3
from .encoder import DEPTH_SCALE_M, DEPTH_UNIT_M

IMG = 320


def rotvec_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> rotation vector (B, 3), rad."""
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1) / 2
    s = w.norm(dim=1).clamp(min=1e-9)
    ang = torch.atan2(s, ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1, 1))
    return w / s[:, None] * ang[:, None]


def matrix_from_rotvec(w: torch.Tensor) -> torch.Tensor:
    return _exp_so3(w)


class FrameEncoder(nn.Module):
    def __init__(self, ch: int = 16, stride: int = 16, pretrained: bool = True, arch: str = "resnet34"):
        super().__init__()
        assert stride in (16, 32) and arch in ("resnet18", "resnet34", "resnet50")
        r = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        width = (2048 if stride == 32 else 1024) if arch == "resnet50" else (512 if stride == 32 else 256)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(r.conv1.weight)
        r.conv1.weight.data[:, :3] = w
        layers = [r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3] + ([r.layer4] if stride == 32 else [])
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(width, ch, 1)
        self.ch, self.stride, self.grid, self.arch = ch, stride, IMG // stride, arch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Batch-norm layers always use their stored statistics (small, correlated batches; and a slowly following
        copy of this encoder gives the world model's targets -- the two must normalise alike)."""
        super().train(mode)
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (B, 320, 320, 3) uint8, depth (B, 320, 320) int [0.1 mm, 0 = no return] -> (B, ch, grid, grid)."""
        x = (rgb.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        d = depth.float()[:, None] * (DEPTH_UNIT_M / DEPTH_SCALE_M)
        f = self.proj(self.net(torch.cat([x, d, (d > 0).float()], 1)))
        return F.layer_norm(f.permute(0, 2, 3, 1), (self.ch,)).permute(0, 3, 1, 2)


class LabelHead(nn.Module):
    def __init__(self, ch: int, grid: int, d: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(ch, d)
        self.pos = nn.Parameter(torch.randn(grid * grid, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, norm_first=True), layers)
        self.geom = nn.Sequential(nn.Linear(9, 64), nn.SiLU())
        self.out = nn.Sequential(nn.LayerNorm(d + 64), nn.Linear(d + 64, 256), nn.SiLU(), nn.Linear(256, 7))

    def forward(self, feat: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """feat (B, ch, H, W), geom (B, 9) (encoder.geometry_features) -> (B, 7): translation mm (3), rotation vector
        deg (3), "inserted now" logit."""
        x = self.enc(self.inp(feat.flatten(2).transpose(1, 2)) + self.pos).mean(1)
        return self.out(torch.cat([x, self.geom(geom)], -1))


def label_targets(hand: torch.Tensor, h_star: torch.Tensor, R_P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (R*, t* [m], target (B, 6) = t* in mm and the rotation vector of R* in deg)."""
    from ..geometry import insertion_motion
    R, t = insertion_motion(hand.float(), h_star.float(), R_P.float())
    return R, t, torch.cat([t * 1000, torch.rad2deg(rotvec_from_matrix(R))], 1)


def label_loss(pred: torch.Tensor, target: torch.Tensor, inserted: torch.Tensor, w_rot: float = 0.5, w_ins: float = 1.0) -> dict:
    """Smooth L1 on the translation (mm) and the rotation (deg), cross-entropy on "inserted now"."""
    pred = pred.float()
    lt = F.smooth_l1_loss(pred[:, :3], target[:, :3])
    lr = F.smooth_l1_loss(pred[:, 3:6], target[:, 3:])
    li = F.binary_cross_entropy_with_logits(pred[:, 6], inserted.float())
    return {"label": lt + w_rot * lr + w_ins * li, "label_t_mm": lt, "label_rot_deg": lr, "label_inserted_bce": li}
