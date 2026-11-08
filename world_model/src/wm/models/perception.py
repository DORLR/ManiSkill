"""Port axes from one frame: board-plane RANSAC fit + gravity (alignment_field/paf/perception.py,
tested there). Estimated once per trajectory / episode from its first image; a real-robot quantity
(depth image, intrinsics, camera pose from FK + hand-eye mount)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .._paths import use_alignment_field

use_alignment_field()

from paf.perception import port_axes_from_obs  # noqa: E402

DEPTH_UNIT_MM = 0.1  # stored depth unit (wm.data.schema.DEPTH_UNIT_M) in millimetres


def port_axes(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_T_world: np.ndarray,
              seed: int = 0) -> np.ndarray | None:
    """(3, 3) port axes from one frame (columns e_x into the board, e_y, e_z up), or None."""
    obs = SimpleNamespace(rgb=rgb, depth_mm=depth.astype(np.float32) * DEPTH_UNIT_MM, K=K,
                          cam_T_world=cam_T_world)
    R, _ = port_axes_from_obs(obs, np.random.default_rng(seed))
    return R
