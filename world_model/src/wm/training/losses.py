"""Autoencoder losses (v3). All targets come from the frame itself or from FK of one insertion.

Reconstruction (the frame as the camera gave it):
    depth_mm     L1 (mm) of the depth where the camera returned a depth
    valid_bce    where the camera returned no depth (0): per-pixel logit, binary cross-entropy
    rgb_l1       L1 of the RGB patch, [0, 1] scale
    dino_cos     1 - cosine similarity to the frozen DINOv2 feature of each patch
Pose (what the plug still has to do; label = FK of this grasp's one insertion, R*, t*):
    motion_mm    per pixel of the plug (segmentation label; only as a loss mask, never an input) with a
                 depth return: L1 between the predicted motion and R* p + t* - p (p = the pixel's
                 3-D point from the true depth), i.e. where that point is when inserted
    weight_bce   per pixel with a depth return: weight logit vs "is plug" (which pixels the fit uses)
    grid_mm      reference-grid error of the solved pose (geometry.grid_pose_error_mm): mostly translation
    angle_deg    angle between solved and true remaining rotation (0.4 deg moves the 2 cm grid 0.14 mm)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..geometry import apply_rigid, insertion_motion, pixel_points, pose_loss_terms
from ..models.encoder import DEPTH_SCALE_M, IMG
from ..models.vae import MOTION_SCALE_M, unpatchify

PLUG_CLASS = 1        # data.schema.SEG_CLASSES.index("plug")


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


def geometric_pose(pred: dict, target: dict, batch: dict, ref: torch.Tensor) -> dict:
    """Per-pixel motion / weight losses and the error of the solved pose (pred["R"], pred["t"])."""
    with torch.autocast("cuda", enabled=False):
        geom = {k: batch[k] for k in ("K", "cam_T_world", "hand", "R_P")}
        depth_m = unpatchify(target["depth_patch"].float(), 1)[:, 0] * DEPTH_SCALE_M
        valid = depth_m.flatten(1) > 0
        p = pixel_points(depth_m, geom["K"], geom["cam_T_world"], geom["hand"], geom["R_P"], crop=320)
        R_star, t_star = insertion_motion(batch["hand"].float(), batch["h_star"].float(), batch["R_P"].float())
        motion_star = apply_rigid(p, R_star, t_star) - p                                       # (B, N, 3) m
        seg = F.interpolate(batch["seg"].float()[:, None], size=(IMG, IMG), mode="nearest").flatten(1)
        plug = (seg == PLUG_CLASS) & valid
        motion = unpatchify(pred["motion_patch"].float(), 3).flatten(2).transpose(1, 2) * MOTION_SCALE_M
        err = (motion - motion_star).abs().sum(-1) * 1000                                       # mm per pixel
        motion_mm = (err * plug).sum() / plug.sum().clamp(min=1)
        logit = unpatchify(pred["weight_logit"].float(), 1).flatten(1)
        weight_bce = (F.binary_cross_entropy_with_logits(logit, plug.float(), reduction="none") * valid).sum() \
            / valid.sum().clamp(min=1)
        grid, angle = pose_loss_terms(pred["R"], pred["t"], R_star, t_star, ref)
    return {"motion_mm": motion_mm, "weight_bce": weight_bce, "grid_mm": grid.mean(), "angle_deg": angle.mean()}
