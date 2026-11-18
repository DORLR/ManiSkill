"""Simulator-free helpers of the online attempts (tested on a CPU): the store's JPEG codec and the joint offset that
moves the hand."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


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
