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
    """A fixed 5x5x5 grid (2 cm cube, x 10-30 mm) where the plug sits in hand-centred port axes.
    Reading a pose (R, t) as the mean displacement of these points gives one shape-independent
    number in mm; used to compare two poses, never as a label."""
    g = torch.linspace(-0.01, 0.01, 5, device=device)
    return torch.stack(torch.meshgrid(g + 0.02, g, g, indexing="ij"), -1).reshape(-1, 3)


def apply_rigid(p: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """p (B,N,3), R (B,3,3), t (B,3) -> R p + t."""
    return p @ R.transpose(1, 2) + t[:, None]
