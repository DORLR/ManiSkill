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
                   rigid motion to the inserted pose (label = FK of the grasp's one insertion,
                   geometry.insertion_motion; how it is written: label_targets below) and "inserted now" (the contact
                   switch).
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
    def __init__(self, ch: int = 16, stride: int = 16, pretrained: bool = True, arch: str = "resnet34", pool: bool = True,
                 freeze_bn: bool = True):
        """`pool` False: the stem's max-pooling is left out, so every stage works at twice the resolution (the dedicated
        reader on the crops did; measured 2026-11-12: on the whole image with the pooling, the same target is read to
        0.34 / 0.48 mm against 0.22 / 0.27 on the crops). The stages are then taken one further for the same stride."""
        super().__init__()
        assert stride in (8, 16, 32) and arch in ("resnet18", "resnet34", "resnet50")
        r = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        self.freeze_bn = freeze_bn
        if not pool:
            assert stride in (8, 16)
            stride_in = stride * 2                                  # the stage whose usual stride is twice as coarse
        else:
            stride_in = stride
        width = {8: 128, 16: 256, 32: 512}[stride_in] * (4 if arch == "resnet50" else 1)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(r.conv1.weight)
        r.conv1.weight.data[:, :3] = w
        layers = ([r.conv1, r.bn1, r.relu] + ([r.maxpool] if pool else []) + [r.layer1, r.layer2] + ([r.layer3] if stride_in >= 16 else [])
                  + ([r.layer4] if stride_in == 32 else []))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv2d(width, ch, 1)
        self.ch, self.stride, self.grid, self.arch = ch, stride, IMG // stride, arch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Batch-norm layers always use their stored statistics (small, correlated batches; and a slowly following
        copy of this encoder gives the world model's targets -- the two must normalise alike)."""
        super().train(mode)
        if self.freeze_bn:
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
        self.out = nn.Sequential(nn.LayerNorm(d + 64), nn.Linear(d + 64, 256), nn.SiLU(), nn.Linear(256, LABEL_DIM + 1))

    def forward(self, feat: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """feat (B, ch, H, W), geom (B, 9) (encoder.geometry_features) -> (B, LABEL_DIM + 1): the remaining motion of
        the reference points (compressed, label_targets) and the "inserted now" logit."""
        x = self.enc(self.inp(feat.flatten(2).transpose(1, 2)) + self.pos).mean(1)
        return self.out(torch.cat([x, self.geom(geom)], -1))


# The label, written so that fractions of a millimetre near the hole count (measured 2026-11-12: with translation in mm
# and rotation in deg as two L1 targets over all frames, the far frames' large values ran the training and the tip
# across the axis was read to 0.53-0.67 mm within 1 mm, worse than blind 0.45):
#   reference points  the remaining rigid motion (R*, t*) is given as where 4 fixed points move: the centre of the
#                     reference cube where the plug sits (geometry.plug_region_points) and that centre + 1 cm along each
#                     port axis; the same for every shape and grasp -- a ruler, not a measured plug location. Translation
#                     and rotation are then one quantity, in millimetres, at the place the image shows.
#   compression       symlog with a 1 mm scale: sign(x) ln(1 + |x| / 1 mm). Near zero it is x itself (0.1 mm stays 0.1);
#                     30 mm becomes 3.4.
REF_POINTS = torch.tensor([[0.02, 0.0, 0.0], [0.03, 0.0, 0.0], [0.02, 0.01, 0.0], [0.02, 0.0, 0.01]])
LABEL_DIM = 12


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * torch.expm1(y.abs())


def label_targets(hand: torch.Tensor, h_star: torch.Tensor, R_P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (R*, t* [m], target (B, 12) = symlog of the 4 reference points' remaining motion in mm)."""
    from ..geometry import apply_rigid, insertion_motion
    R, t = insertion_motion(hand.float(), h_star.float(), R_P.float())
    p = REF_POINTS.to(R.device)[None].expand(R.shape[0], -1, -1)
    return R, t, symlog((apply_rigid(p, R, t) - p).flatten(1) * 1000)


def label_motion(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A label-head output (or a target) -> the rigid motion (R, t [m]) that moves the reference points as it says."""
    from ..geometry import weighted_rigid_fit
    p = REF_POINTS.to(pred.device)[None].expand(pred.shape[0], -1, -1)
    q = p + symexp(pred.float()[:, :LABEL_DIM]).unflatten(1, (4, 3)) / 1000
    return weighted_rigid_fit(p, q, torch.ones(pred.shape[0], 4, device=pred.device))


def centre_lateral_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Error (mm) across the insertion axis of the reference centre's remaining motion; target from label_targets."""
    return (symexp(pred.float()[:, 1:3]) - symexp(target[:, 1:3])).norm(dim=-1)


def label_loss(pred: torch.Tensor, target: torch.Tensor, inserted: torch.Tensor, w_ins: float = 1.0,
               close_weight: float = 0.0) -> dict:
    """Smooth L1 on the compressed point motions, cross-entropy on "inserted now". `close_weight`: frames whose reference
    centre is within 3 mm of where it ends across the axis count 1 + close_weight times."""
    pred = pred.float()
    per = F.smooth_l1_loss(pred[:, :LABEL_DIM], target, beta=0.05, reduction="none").mean(1)
    w = 1 + close_weight * (symexp(target[:, 1:3]).norm(dim=-1) < 3.0).float()
    lm = (per * w).sum() / w.sum()
    li = F.binary_cross_entropy_with_logits(pred[:, LABEL_DIM], inserted.float())
    return {"label": lm + w_ins * li, "label_motion": lm, "label_centre_lateral_mm": centre_lateral_mm(pred, target).mean().detach(),
            "label_inserted_bce": li}


# ------------------------------------------------------------------------------------------------ the label, per cell
# Measured 2026-11-12 (probe_encoder_grid.py, held-out exploration, pins at the face, tip within 1 / 3 mm): the same
# encoder reads the true tip's offset to 0.34 / 0.48 mm when trained on it, and to 0.45 / 0.72 with the label above. The
# label above says how the hand has to move; what the image shows is the plug's surface in front of the hole, and
# between the two lies how the plug sits in the hand. So the same FK label is also given where it is seen: for every
# cell of the feature map that shows the plug with depth, the remaining motion of that piece of surface,
# R* p + t* - p (p = the cell's mean 3-D point from the depth image, hand-centred port axes). No tip location, no CAD;
# the plug mask is the recorded class map (a loss mask, as for the v3 / v4 motion field), never an input.
PLUG_CLASS = 1


class DenseLabelHead(nn.Module):
    """Training only: per cell of the feature map, the remaining motion (symlog mm, 3 numbers) of the plug surface there."""

    def __init__(self, ch: int, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch + 9, d, 3, padding=1), nn.SiLU(), nn.Conv2d(d, d, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(d, 3, 1))

    def forward(self, feat: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        g = geom[:, :, None, None].expand(-1, -1, *feat.shape[-2:]).to(feat.dtype)
        return self.net(torch.cat([feat, g], 1))


def dense_targets(depth, seg, K, cam_T_world, hand, h_star, R_P, grid: int, min_px: int = 8):
    """-> target (B, 3, grid, grid) symlog mm; weight (B, grid, grid): 1 where the cell shows >= min_px plug pixels with
    depth; points (B, grid, grid, 3): those pixels' mean 3-D point (m, hand-centred port axes)."""
    from ..geometry import apply_rigid, insertion_motion, pixel_points
    B = depth.shape[0]
    P = pixel_points(depth.float() * DEPTH_UNIT_M, K, cam_T_world, hand, R_P, crop=IMG)                 # (B, N, 3)
    R, t = insertion_motion(hand.float(), h_star.float(), R_P.float())
    M = (apply_rigid(P, R, t) - P) * 1000
    m = ((seg == PLUG_CLASS) & (depth > 0)).float().view(B, 1, IMG, IMG)
    cell = IMG // grid
    cnt = F.avg_pool2d(m, cell) * cell * cell
    mean = lambda X: F.avg_pool2d(X.transpose(1, 2).reshape(B, 3, IMG, IMG) * m, cell) * cell * cell / cnt.clamp(min=1)
    return symlog(mean(M)), (cnt[:, 0] >= min_px).float(), mean(P).permute(0, 2, 3, 1)


def dense_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.smooth_l1_loss(pred.float(), target, beta=0.05, reduction="none").mean(1)
    return (per * weight).sum() / weight.sum().clamp(min=1)


def dense_motion(pred: torch.Tensor, weight: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The rigid motion (R, t [m]) that best moves the plug cells' points as the head says (weighted rigid fit)."""
    from ..geometry import weighted_rigid_fit
    p = points.flatten(1, 2)
    q = p + symexp(pred.float()).permute(0, 2, 3, 1).flatten(1, 2) / 1000
    return weighted_rigid_fit(p, q, weight.flatten(1))
