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
    w = p @ R_P.transpose(1, 2) + hand[:, None, :3, 3]                  # world
    delta = h_star @ torch.linalg.inv(hand)
    w_star = w @ delta[:, :3, :3].transpose(1, 2) + delta[:, None, :3, 3]
    return (w_star - hand[:, None, :3, 3]) @ R_P                          # back to hand-centred port axes


def apply_rigid(p: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """p (B,N,3), R (B,3,3), t (B,3) -> R p + t."""
    return p @ R.transpose(1, 2) + t[:, None]
