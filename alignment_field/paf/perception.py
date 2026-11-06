"""Observation preprocessing: wrist RGB-D -> (RGB crop, canonical point cloud).

Point cloud (literature: DP3 for sparse point-cloud encoders; SID and CFVS for
object/hole-centred canonicalisation):
  1. back-project the depth image with the intrinsics,
  2. move the points to the robot base frame with the camera extrinsics
     (= forward kinematics + hand-eye calibration on a real robot),
  3. estimate the board plane by RANSAC (Fischler & Bolles 1981) and build the
     port axes from its normal and gravity,
  4. express points relative to the hand position in those port axes,
     crop a box in front of the hand and sample a fixed number of points.

RGB: a fixed native-resolution crop around the region where the pin tips and
the hole openings appear (probe of 2026-11-05: a 5 mm hole shift changes only
~50-80 pixels, all next to the tips, so no downscaling).
"""

from __future__ import annotations

import numpy as np

from .config import CROP, N_POINTS, PC_BOX, REL_CROP
from .geometry import UP, port_axes


def backproject(depth_mm: np.ndarray, K: np.ndarray, stride: int = 1):
    d = depth_mm[::stride, ::stride].astype(np.float64) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    valid = d > 0
    z = d[valid]
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), valid


def to_world(pts_cam: np.ndarray, cam_T_world: np.ndarray) -> np.ndarray:
    world_T_cam = np.linalg.inv(cam_T_world)
    return pts_cam @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]


def ransac_board_normal(pts_w: np.ndarray, cam_pos: np.ndarray, rng: np.random.Generator,
                        iters: int = 2000, thresh: float = 0.001, max_up: float = 0.5):
    """Largest plane whose normal is not vertical (rejects the table).
    Returns (unit normal pointing away from the camera, inlier fraction)."""
    n_pts = len(pts_w)
    if n_pts < 50:
        return None, 0.0
    sub = pts_w[rng.choice(n_pts, size=min(n_pts, 8000), replace=False)]
    best, best_n = 0, None
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n @ UP) > max_up:
            continue
        cnt = int((np.abs((sub - a) @ n) < thresh).sum())
        if cnt > best:
            best, best_n, best_a = cnt, n, a
    if best_n is None:
        return None, 0.0
    # least-squares refit on inliers
    inl = sub[np.abs((sub - best_a) @ best_n) < thresh]
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
    n = vt[-1]
    if (c0 - cam_pos) @ n < 0:
        n = -n
    return n, best / len(sub)


def canonical_points(obs, R_P: np.ndarray, rng: np.random.Generator):
    """Point cloud (N_POINTS, 6): xyz in hand-centred port axes (/0.05 m) + rgb (0..1)."""
    pts_cam, valid = backproject(obs.depth_mm, obs.K)
    rgb = obs.rgb[..., :3][valid].astype(np.float32) / 255.0
    pts_w = to_world(pts_cam, obs.cam_T_world)
    p = (pts_w - obs.H[:3, 3]) @ R_P
    (x0, x1), (y0, y1), (z0, z1) = PC_BOX
    keep = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] > z0) & (p[:, 2] < z1)
    p, rgb = p[keep], rgb[keep]
    out = np.zeros((N_POINTS, 6), np.float32)
    if len(p) == 0:
        return out
    idx = rng.choice(len(p), N_POINTS, replace=len(p) < N_POINTS)
    out[:, :3] = p[idx] / 0.05
    out[:, 3:] = rgb[idx]
    return out


def rel_view(obs):
    """RGB, depth and intrinsics of the relational-net window (REL_CROP)."""
    r0, r1, c0, c1 = REL_CROP
    K = np.array(obs.K, dtype=np.float32, copy=True)
    K[0, 2] -= c0
    K[1, 2] -= r0
    return (np.ascontiguousarray(obs.rgb[r0:r1, c0:c1, :3]), np.ascontiguousarray(obs.depth_mm[r0:r1, c0:c1]), K)


def rgb_crop(obs) -> np.ndarray:
    r0, r1, c0, c1 = CROP
    return np.ascontiguousarray(obs.rgb[r0:r1, c0:c1, :3])


def port_axes_from_obs(obs, rng):
    """Port axes estimated from the image (used by the method at test time)."""
    pts_cam, _ = backproject(obs.depth_mm, obs.K, stride=2)
    pts_w = to_world(pts_cam, obs.cam_T_world)
    cam_pos = np.linalg.inv(obs.cam_T_world)[:3, 3]
    n, frac = ransac_board_normal(pts_w, cam_pos, rng)
    if n is None:
        return None, 0.0
    return port_axes(n), frac


def preprocess(obs, R_P_est, rng):
    return rgb_crop(obs), canonical_points(obs, R_P_est, rng)
