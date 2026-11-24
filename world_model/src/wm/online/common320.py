"""Simulator-free helpers of the online attempts (tested on a CPU): the store's JPEG codec, the joint offset that
moves the hand, and the estimated pin-tip gap from the offset read-out (with its checkpoint-label check)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

RANDOM_EDGES_MM = (0.0, 2.0, 6.0)               # attempt320's ablation "random": bins of the estimated gap g_t (4 bins)


def gap_bin(g: float) -> int:
    """Bin of the estimated pin-tip gap g [mm] for the ablation "random": 0 (g < 0), 1 (0-2), 2 (2-6), 3 (>= 6)."""
    return int(sum(g >= e for e in RANDOM_EDGES_MM))


def gap_estimate_mm(o: np.ndarray, offset_label: str, pin_length_mm) -> float:
    """The estimated pin-tip gap g_t [mm] (negative: the tips that deep past the entrance) from the offset read-out
    o (4,) of a checkpoint trained with the offset label `offset_label` (wmdata320 OFFSET_LABELS, module docstring):
      "gap_mm"   (AutoMate, the user's decision 6 of 2026-11-23): the read-out's first component is the gap itself,
                 g = o[0]; pin_length_mm is never called (no pin length, H* or simulator truth at test time)
      "along_mm" (ManiSkill; every checkpoint without the key): g = -o[0] - L, L = pin_length_mm() [mm] (the CAD pin
                 length; the formula of 2026-11-18, unchanged)
    pin_length_mm: a function without arguments returning L [mm], called for "along_mm" only."""
    from ..tdmpc.wmdata320 import OFFSET_ALONG, OFFSET_GAP
    if offset_label == OFFSET_GAP:
        return float(o[0])
    if offset_label == OFFSET_ALONG:
        return float(-o[0] - pin_length_mm())
    raise ValueError(f"unknown offset label {offset_label!r}")


def simulator_of_robot(robot) -> str:
    """"automate" for the AutoMate adapter (wm.sim_automate's Robot), "maniskill" otherwise. Reads the robot's class
    only (no port_spec, no pin length)."""
    root = __name__.split(".")[0]
    return "automate" if type(robot).__module__.startswith(f"{root}.sim_automate.") else "maniskill"


def check_offset_label(offset_label: str, simulator: str) -> None:
    """An attempt on AutoMate needs a checkpoint trained on the gap label ("gap_mm", decision 6 of 2026-11-23: the
    in-hole rule and every gap then read the prediction without L); one on ManiSkill needs "along_mm" (the old
    formula). Anything else: ValueError."""
    from ..tdmpc.wmdata320 import OFFSET_ALONG, OFFSET_GAP
    want = OFFSET_GAP if simulator == "automate" else OFFSET_ALONG
    if offset_label != want:
        raise ValueError(f"a {simulator} attempt needs a checkpoint trained with offset label {want!r}, this one has "
                         f"{offset_label!r}" + (" (trained before decision 6 of 2026-11-23: its gap would need the pin "
                                                "length at test time; train a new one)" if simulator == "automate" else ""))



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
