"""Torch SE(3) helpers used by the models and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation (Zhou et al. 2019) -> (.., 3, 3)."""
    a, b = x[..., :3], x[..., 3:6]
    e1 = F.normalize(a, dim=-1)
    e2 = F.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def insertion_motion(hand: torch.Tensor, h_star: torch.Tensor, R_P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The rigid motion (R*, t*) that takes the plug from now to inserted, in hand-centred port axes:
    p_inserted = R* p + t* for any plug point p. The plug is rigid in the hand, so this is the hand
    motion delta = h_star @ inv(hand) expressed in that frame (FK only; no CAD, no tip location)."""
    delta = h_star @ torch.linalg.inv(hand)
    Rt = R_P.transpose(1, 2)
    h = hand[:, :3, 3]
    R = Rt @ delta[:, :3, :3] @ R_P
    t = ((delta[:, :3, :3] @ h[..., None])[..., 0] + delta[:, :3, 3] - h)[:, None] @ R_P
    return R, t[:, 0]


def rotation_angle_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between rotations, (B,3,3) x2 -> (B,). Chordal form: smooth gradient at zero."""
    chord = (Ra - Rb).pow(2).sum((-2, -1)).add(1e-12).sqrt()
    return torch.rad2deg(2 * torch.asin((chord / (2 * 2 ** 0.5)).clamp(max=1.0)))


def plug_points_at_insertion(p: torch.Tensor, hand: torch.Tensor, h_star: torch.Tensor,
                             R_P: torch.Tensor) -> torch.Tensor:
    """Where observed plug points would be if the hand moved from `hand` to `h_star`.

    p       (B, N, 3) plug points in hand-centred port axes (origin at the hand, axes = columns of R_P)
    hand    (B, 4, 4) current hand pose (FK), world
    h_star  (B, 4, 4) inserted hand pose of this grasp (FK of the one insertion), world
    R_P     (B, 3, 3) port axes (board-plane fit + gravity)
    returns (B, N, 3) in the same frame as p. The plug is rigid in the hand, so every
    point moves by delta = h_star @ inv(hand); no CAD and no tip location needed.
    """
    R, t = insertion_motion(hand, h_star, R_P)
    return apply_rigid(p, R, t)


def plug_region_points(device=None) -> torch.Tensor:
    """The reference grid: a fixed 5x5x5 grid (2 cm cube, x 10-30 mm, y and z +-10 mm) where the plug
    sits in hand-centred port axes, the same for every shape and grasp. A rigid motion (R, t) is
    turned into millimetres by moving these points; it is a ruler, never a label or a measured
    plug location."""
    g = torch.linspace(-0.01, 0.01, 5, device=device)
    return torch.stack(torch.meshgrid(g + 0.02, g, g, indexing="ij"), -1).reshape(-1, 3)


def apply_rigid(p: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """p (B,N,3), R (B,3,3), t (B,3) -> R p + t."""
    return p @ R.transpose(1, 2) + t[:, None]


def grid_pose_error_mm(R_a: torch.Tensor, t_a: torch.Tensor, R_b: torch.Tensor, t_b: torch.Tensor,
                       ref: torch.Tensor) -> torch.Tensor:
    """Reference-grid pose error (B,): mean distance (mm) between the reference-grid points moved by
    (R_a, t_a) and by (R_b, t_b). Mostly measures translation: the grid is only 2 cm across, so a
    0.4 deg rotation moves its points by ~0.14 mm (rotation_angle_deg covers small angles)."""
    p = ref[None].expand(R_a.shape[0], -1, -1)
    return (apply_rigid(p, R_a, t_a) - apply_rigid(p, R_b, t_b)).norm(dim=-1).mean(1) * 1000


def pose_loss_terms(R_a, t_a, R_b, t_b, ref) -> tuple[torch.Tensor, torch.Tensor]:
    """(reference-grid error mm, angle deg), each (B,). Used for the VAE pose head against the FK
    label and for the world model's predicted pose against the pose read from the true latent."""
    return (grid_pose_error_mm(R_a.float(), t_a.float(), R_b.float(), t_b.float(), ref),
            rotation_angle_deg(R_a.float(), R_b.float()))
