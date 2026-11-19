"""Simulator-free helpers of the online attempts (tested on a CPU): the store's JPEG codec and the joint offset that
moves the hand."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

RANDOM_EDGES_MM = (0.0, 2.0, 6.0)               # attempt320's ablation "random": bins of the estimated gap g_t (4 bins)


def gap_bin(g: float) -> int:
    """Bin of the estimated pin-tip gap g [mm] for the ablation "random": 0 (g < 0), 1 (0-2), 2 (2-6), 3 (>= 6)."""
    return int(sum(g >= e for e in RANDOM_EDGES_MM))



def servo_move(o_x: float, o_y: float, axis: np.ndarray, max_mm: float) -> np.ndarray:
    """attempt320's ablation "servo": the hand translation [m, world] that cancels a lateral offset (o_x, o_y) [mm]
    given in the label frame of wmdata320._basis(axis) (rows: axis, e_1, e_2), shortened to at most max_mm."""
    from ..tdmpc.wmdata320 import _basis
    n = float(np.hypot(o_x, o_y))
    if n == 0.0:
        return np.zeros(3)
    B = _basis(np.asarray(axis, np.float64))
    return -(o_x * B[1] + o_y * B[2]) * min(1.0, max_mm / n) / 1000.0


def jpeg(rgb: np.ndarray) -> bytes:
    """The store's codec (wm/data/store.py: quality 95, no chroma subsampling)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue()


def unjpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def offset(J: np.ndarray, d_m: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Joint offset (7,) that moves the hand by d_m (3,) [m, world] with no rotation: damped least squares on the
    (6, 7) Jacobian (position [m] and rotation vector [rad] per joint [rad])."""
    tgt = np.r_[d_m, 0.0, 0.0, 0.0]
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), tgt)
