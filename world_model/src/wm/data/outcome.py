"""Success label from the simulator (v4): is the plug inserted at a given step?

The same criterion as the closed-loop evaluation (eval/rollout.py): within 1 mm of full depth AND
within 2 mm of the hole axis AND tilted at most 2 deg. Computed from the recorded charger pose in the
socket-goal frame (eval_rel). A label only -- on the real robot it is the connector's contact switch
(or a person watching), never a model input.
"""

from __future__ import annotations

import numpy as np

INSERTED_MARGIN_M = 0.001
INSERTED_LATERAL_M = 0.002
INSERTED_ANGLE_DEG = 2.0


def diagnosis(rel7: np.ndarray) -> dict:
    """rel7 (..., 7) = (x, y, z, qw, qx, qy, qz) charger in goal frame -> depth left, lateral (m), angle (deg)."""
    rel7 = np.asarray(rel7, np.float64)
    w = np.clip(np.abs(rel7[..., 3]) / np.linalg.norm(rel7[..., 3:7], axis=-1), 0, 1)
    return {"depth_left_m": -rel7[..., 0], "lateral_m": np.hypot(rel7[..., 1], rel7[..., 2]),
            "angle_deg": np.degrees(2 * np.arccos(w))}


def inserted(rel7: np.ndarray) -> np.ndarray:
    d = diagnosis(rel7)
    return (d["depth_left_m"] <= INSERTED_MARGIN_M) & (d["lateral_m"] <= INSERTED_LATERAL_M) & \
        (d["angle_deg"] <= INSERTED_ANGLE_DEG)
