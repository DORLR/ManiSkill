"""Pose and camera helpers for the preheld insertion task."""

from __future__ import annotations
from typing import Any
import numpy as np
import sapien
from mani_skill.utils import common


def _sp(value: Any) -> sapien.Pose:
    """Read one CPU pose; ManiSkill quaternions and SAPIEN use wxyz."""
    if hasattr(value, "sp"):
        return value.sp
    if isinstance(value, sapien.Pose):
        return value
    array = np.asarray(common.to_numpy(value), dtype=np.float64).reshape(-1)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError("Expected one finite [x,y,z,qw,qx,qy,qz] pose")
    return sapien.Pose(p=array[:3], q=array[3:])


def _raw(value: Any) -> list[float]:
    value = _sp(value)
    return np.r_[value.p, value.q].astype(float).tolist()


def _pitch(reference: sapien.Pose, held: sapien.Pose) -> dict[str, Any]:
    rotation = (reference.inv() * held).to_transformation_matrix()[:3, :3]
    direction = rotation @ np.array([1.0, 0.0, 0.0])
    return {
        "pitch_deg": float(np.degrees(np.arctan2(direction[2], direction[0]))),
        "forward_axis_in_zero_reference": direction.astype(float).tolist(),
        "lateral_residual": float(direction[1]),
    }


def _pose_error(actual: sapien.Pose, desired: sapien.Pose) -> dict[str, float]:
    qa = np.asarray(actual.q, dtype=np.float64)
    qb = np.asarray(desired.q, dtype=np.float64)
    dot = float(np.dot(qa, qb) / (np.linalg.norm(qa) * np.linalg.norm(qb)))
    return {
        "translation_mm": float(np.linalg.norm(actual.p - desired.p) * 1000.0),
        "rotation_deg": float(np.degrees(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0)))),
    }


def _native_rgb256_config(config: dict | None) -> dict:
    """Allow only the resolution override, preserving native camera geometry."""
    for key, value in (config or {}).items():
        if key in ("width", "height"):
            if value != 256:
                raise ValueError("WMInsertionCharger-v2 uses 256x256 native RGB/depth")
        elif key in ("base_camera", "hand_camera") and isinstance(value, dict):
            if any(
                name not in ("width", "height") or size != 256
                for name, size in value.items()
            ):
                raise ValueError(
                    "Only native camera width/height=256 may be overridden"
                )
        else:
            raise ValueError(
                "Camera mount, FOV and other native camera parameters are fixed"
            )
    return {"width": 256, "height": 256}
