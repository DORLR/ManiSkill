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


# ---------------------------------------------------------------------------- geometric pose readout
def pixel_points(depth_m: torch.Tensor, K: torch.Tensor, cam_T_world: torch.Tensor, hand: torch.Tensor,
                 R_P: torch.Tensor, crop: int) -> torch.Tensor:
    """3-D point of every pixel of a G x G depth image (the crop resized to G x G), in hand-centred port
    axes: (B, G*G, 3). depth_m (B, G, G) metres along the optical axis (0 = no return; such points are
    meaningless and must be masked by the caller). K is the crop's intrinsics (crop x crop pixels)."""
    B, G = depth_m.shape[0], depth_m.shape[-1]
    c = (torch.arange(G, device=depth_m.device, dtype=torch.float32) + 0.5) * crop / G - 0.5
    v, u = torch.meshgrid(c, c, indexing="ij")
    pix = torch.stack([u.flatten(), v.flatten(), torch.ones_like(u.flatten())], 0)            # (3, G*G)
    rays = torch.linalg.solve(K.float(), pix[None].expand(B, -1, -1))                          # z = 1
    x_cam = rays * depth_m.reshape(B, 1, -1).float()                                            # (B, 3, N)
    R_wc = cam_T_world[:, :3, :3].float().transpose(1, 2)
    c_w = -(R_wc @ cam_T_world[:, :3, 3:].float())                                              # camera centre
    x_w = R_wc @ x_cam + c_w
    return (R_P.float().transpose(1, 2) @ (x_w - hand[:, :3, 3:].float())).transpose(1, 2)


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation vector (B, 3) -> (B, 3, 3)."""
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)[..., None]
    Kx = _skew(w) / th
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def weighted_rigid_fit(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor, iters: int = 4, damping: float = 1e-6):
    """Rigid motion (R, t) minimising sum_i w_i |R p_i + t - q_i|^2 (the Kabsch objective), solved by
    damped Gauss-Newton on (rotation vector, translation) from the identity. Used instead of the SVD
    solution because SVD gradients blow up for nearly planar point sets (the plug pins); for the
    rotations here (a few degrees, < 30 deg) 4 steps converge to the Kabsch optimum. Differentiable.
    p, q (B, N, 3) metres, w (B, N) >= 0; points are centred and scaled internally for conditioning."""
    w = w.float().clamp(min=0) + 1e-8
    wn = w / w.sum(1, keepdim=True)
    mp = (wn[..., None] * p).sum(1, keepdim=True)
    scale = ((wn * (p - mp).pow(2).sum(-1)).sum(1).sqrt().clamp(min=1e-4))[:, None, None]
    P, Qs = (p - mp) / scale, (q - mp) / scale
    R = torch.eye(3, device=p.device).expand(p.shape[0], 3, 3)
    t = (wn[..., None] * (Qs - P)).sum(1)
    I6 = torch.eye(6, device=p.device)
    for _ in range(iters):
        Pr = P @ R.transpose(1, 2)                                   # rotated points
        r = Pr + t[:, None] - Qs                                     # residuals (B, N, 3)
        J = torch.cat([-_skew(Pr), torch.eye(3, device=p.device).expand(*Pr.shape[:2], 3, 3)], -1)   # (B,N,3,6)
        A = torch.einsum("bn,bnij,bnik->bjk", wn, J, J) + damping * I6
        g = torch.einsum("bn,bnij,bni->bj", wn, J, r)
        d = -torch.linalg.solve(A, g)
        R = _exp_so3(d[:, :3]) @ R
        t = t + d[:, 3:]
    # back to the original frame: q ~ R (p - mp) + mp + scale * t
    t_out = mp[:, 0] + scale[:, 0, 0, None] * t - (R @ mp[:, 0, :, None])[..., 0]
    return R, t_out
