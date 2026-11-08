"""Autoencoder losses (v3). All targets come from the frame itself or from FK of one insertion.

Reconstruction (the frame as the camera gave it):
    depth_mm     L1 (mm) of the depth where the camera returned a depth
    valid_bce    where the camera returned no depth (0): per-pixel logit, binary cross-entropy
    rgb_l1       L1 of the RGB patch, [0, 1] scale
    dino_cos     1 - cosine similarity to the frozen DINOv2 feature of each patch
Pose (what the plug still has to do; label = FK of this grasp's one insertion):
    grid_mm      reference-grid pose error (geometry.grid_pose_error_mm): mostly translation
    angle_deg    angle between predicted and true remaining rotation: small rotations, which the
                 2 cm grid barely sees (0.4 deg -> 0.14 mm)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..geometry import insertion_motion, pose_loss_terms
from ..models.encoder import DEPTH_SCALE_M


def reconstruction(pred: dict, target: dict) -> dict:
    with torch.autocast("cuda", enabled=False):
        d_t = target["depth_patch"].float()
        valid = d_t > 0
        d_err = (pred["depth_patch"].float() - d_t).abs() * DEPTH_SCALE_M * 1000      # mm
        depth_mm = (d_err * valid).sum() / valid.sum().clamp(min=1)
        valid_bce = F.binary_cross_entropy_with_logits(pred["valid_logit"].float(), valid.float())
        rgb_l1 = (pred["rgb_patch"].float() - target["rgb_patch"].float()).abs().mean()
        dino_cos = 1 - F.cosine_similarity(pred["dino"].float(), target["dino"].float(), dim=-1).mean()
    return {"depth_mm": depth_mm, "valid_bce": valid_bce, "rgb_l1": rgb_l1, "dino_cos": dino_cos}


def insertion_pose(R: torch.Tensor, t: torch.Tensor, hand: torch.Tensor, h_star: torch.Tensor,
                   R_P: torch.Tensor, ref: torch.Tensor) -> dict:
    """Pose head vs. the remaining motion to the inserted pose (FK; geometry.insertion_motion)."""
    with torch.autocast("cuda", enabled=False):
        R_star, t_star = insertion_motion(hand.float(), h_star.float(), R_P.float())
        grid, angle = pose_loss_terms(R, t, R_star, t_star, ref)
    return {"grid_mm": grid.mean(), "angle_deg": angle.mean()}
