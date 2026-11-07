"""Perception primitives.

Reused from alignment_field/paf (tested there): back-projection, hand-centred port
axes, frozen DINOv2 ViT-S/14 features, PointNet++ set abstraction, plug<->port
cross-attention block. Port axes come from a board-plane RANSAC fit plus gravity
(paf.perception.port_axes_from_obs), estimated once per trajectory from its first image.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .._paths import use_alignment_field

use_alignment_field()

from paf.model import mlp  # noqa: E402,F401
from paf.perception import port_axes_from_obs  # noqa: E402
from paf.relnet import (BOARD_BAND, BOX_YZ, PLUG_X, CrossBlock, DinoFeatures,  # noqa: E402,F401
                        SetAbstraction, _sample, backproject_torch, to_port_frame)

DEPTH_UNIT_MM = 0.1  # stored depth unit (wm.data.schema.DEPTH_UNIT_M) in millimetres


def port_axes(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_T_world: np.ndarray,
              seed: int = 0) -> np.ndarray | None:
    """(3, 3) port axes from one frame (columns e_x into the board, e_y, e_z up), or None."""
    obs = SimpleNamespace(rgb=rgb, depth_mm=depth.astype(np.float32) * DEPTH_UNIT_MM, K=K,
                          cam_T_world=cam_T_world)
    R, _ = port_axes_from_obs(obs, np.random.default_rng(seed))
    return R
