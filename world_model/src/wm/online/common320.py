"""Simulator-free helpers of the online attempts (tested on a CPU): the store's JPEG codec, the joint offset that
moves the hand, and the AutoMate check of a checkpoint's offset_mode."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

RANDOM_EDGES_MM = (0.0, 2.0, 6.0)               # attempt320's ablation "random": bins of the estimated gap g_t (4 bins)


def gap_bin(g: float) -> int:
    """Bin of the estimated pin-tip gap g [mm] for the ablation "random": 0 (g < 0), 1 (0-2), 2 (2-6), 3 (>= 6)."""
    return int(sum(g >= e for e in RANDOM_EDGES_MM))


def simulator_of_robot(robot) -> str:
    """"automate" for the AutoMate adapter (wm.sim_automate's Robot), "maniskill" otherwise. Reads the robot's class
    only (no port_spec, no pin length)."""
    root = __name__.split(".")[0]
    return "automate" if type(robot).__module__.startswith(f"{root}.sim_automate.") else "maniskill"


def check_offset_mode(offset_mode: str, simulator: str) -> None:
    """An attempt on AutoMate needs a checkpoint with offset_mode "gap" (decision 6 of 2026-11-23: the in-hole rule
    and every gap then read the prediction without L); ManiSkill takes either mode (the "along" formula unchanged).
    Anything else: ValueError."""
    if simulator == "automate" and offset_mode != "gap":
        raise ValueError(f"a {simulator} attempt needs a checkpoint trained with offset_mode 'gap', this one has "
                         f"{offset_mode!r} (its gap would need the pin length at test time; train a new one)")


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
