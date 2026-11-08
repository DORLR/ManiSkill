"""Losses (all distances reported in millimetres)."""

from __future__ import annotations

import torch

from ..geometry import apply_rigid, insertion_motion, plug_points_at_insertion, rotation_angle_deg


def chamfer_mm(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance (mm) between pred (B,P,3) and the valid target points (B,M,3)."""
    with torch.autocast("cuda", enabled=False):
        d = torch.cdist(pred.float() * 1000, target.float() * 1000)               # (B,P,M)
        d_pm = d + (~mask)[:, None, :].float() * 1e6
        a = d_pm.min(2).values.mean(1)                                             # pred -> target
        b = (d.min(1).values * mask).sum(1) / mask.sum(1).clamp(min=1)             # target -> pred
    has = mask.any(1).float()
    return ((a + b) * has).sum() / has.sum().clamp(min=1)


def plug_point_loss_mm(R: torch.Tensor, t: torch.Tensor, plug_xyz: torch.Tensor, plug_mask: torch.Tensor,
                       hand: torch.Tensor, h_star: torch.Tensor, R_P: torch.Tensor) -> torch.Tensor:
    """Mean distance (mm) between where the predicted rigid motion sends the observed plug points
    and where they are when inserted (from FK of the one insertion). Points far from the hand
    (the pins) carry the largest rotation errors, so the tip is weighted without being located."""
    with torch.autocast("cuda", enabled=False):
        target = plug_points_at_insertion(plug_xyz.float(), hand.float(), h_star.float(), R_P.float())
        pred = apply_rigid(plug_xyz.float(), R.float(), t.float())
        err = (pred - target).norm(dim=-1) * 1000
    return (err * plug_mask).sum() / plug_mask.sum().clamp(min=1)


def insertion_angle_loss_deg(R: torch.Tensor, hand: torch.Tensor, h_star: torch.Tensor, R_P: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between the predicted rotation still needed and the true one (FK of the one
    insertion). The plug-point loss is nearly blind to small angles (0.4 deg moves a 20 mm plug's
    points by ~0.14 mm); this term supervises the rotation directly."""
    with torch.autocast("cuda", enabled=False):
        R_star, _ = insertion_motion(hand.float(), h_star.float(), R_P.float())
        return rotation_angle_deg(R.float(), R_star).mean()
