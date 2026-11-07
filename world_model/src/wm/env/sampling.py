"""Grasps, start poses and contact-trial offsets.

Start poses and offsets are written in the expert's socket-goal frame: x along
the insertion axis (into the board), tip gap = distance from the pin tips to the
hole face (= 3 mm - t_x of the alignment_field state), y/z lateral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._paths import use_alignment_field

use_alignment_field()
from paf import config as _C  # noqa: E402

CONDITIONS = {  # name: (pitch kind, position kind) as in alignment_field/paf/config.py
    "train": ("train", "train"),
    "pitch_interp": ("interp", "train"),
    "pitch_extrap": ("extrap", "train"),
    "pos_interp": ("train", "interp"),
    "pos_extrap": ("train", "extrap"),
    "combined": ("extrap", "extrap"),
}
BAND_CONDITIONS = [c for c in CONDITIONS if c != "train"]


def sample_grasp(rng: np.random.Generator, condition: str) -> tuple[float, float]:
    """(pitch deg, grasp position m) inside the bands of `condition`."""
    return _C.sample_grasp(rng, *CONDITIONS[condition])


@dataclass(frozen=True)
class StartBox:
    tip_gap_m: tuple[float, float] = (0.006, 0.018)  # t_x in [-15, -3] mm
    lateral_m: float = 0.010                         # |y|, |z| <= 10 mm, centre included
    rotation_deg: float = 6.0                        # each axis


@dataclass(frozen=True)
class Offset:
    """A start or an offset: tip gap (m), lateral y/z (m), roll/pitch/yaw (deg, static XYZ)."""
    tip_gap_m: float
    yz_m: tuple[float, float]
    rpy_deg: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {"tip_gap_mm": self.tip_gap_m * 1e3, "yz_mm": [v * 1e3 for v in self.yz_m],
                "rpy_deg": list(self.rpy_deg)}


def sample_start(rng: np.random.Generator, box: StartBox = StartBox()) -> Offset:
    return Offset(float(rng.uniform(*box.tip_gap_m)),
                  tuple(float(v) for v in rng.uniform(-box.lateral_m, box.lateral_m, 2)),
                  tuple(float(v) for v in rng.uniform(-box.rotation_deg, box.rotation_deg, 3)))


PUSH_START_GAP_M = 0.003  # contact trials start at the pre-insertion gap, as the expert's insertion does


def sample_contact_offset(rng: np.random.Generator, max_lateral_m: float = 0.003,
                          max_rotation_deg: float = 3.0, p_zero: float = 0.15) -> Offset:
    """Pose at the pre-insertion gap (pin tips 3 mm from the hole face) from which a straight push starts.

    Lateral and rotation magnitudes are uniform in [0, max] with random directions;
    with probability p_zero the push starts aligned (a positive example).
    """
    if rng.random() < p_zero:
        return Offset(PUSH_START_GAP_M, (0.0, 0.0), (0.0, 0.0, 0.0))
    d = rng.normal(size=2)
    yz = d / np.linalg.norm(d) * rng.uniform(0, max_lateral_m)
    a = rng.normal(size=3)
    rpy = a / np.linalg.norm(a) * rng.uniform(0, max_rotation_deg)
    return Offset(PUSH_START_GAP_M, tuple(float(v) for v in yz), tuple(float(v) for v in rpy))
