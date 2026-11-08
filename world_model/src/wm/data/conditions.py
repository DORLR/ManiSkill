"""Robot-state and action conditions of the world model (v3), numpy, shared by the dataset, the
checks and the planner so that training and use build them identically.

Everything is a hand (TCP) pose from forward kinematics: measured hand poses (FK of measured
joints) and commanded hand poses (FK of the commanded joint targets), as in Ctrl-World, where the
world model is conditioned on end-effector poses computed from the actions.

All poses are expressed relative to the current measured hand pose and in port axes R_P (columns =
port axes in world, from the board-plane fit of the first frame): position difference in cm and
the relative rotation as 6D (first two columns). 9 values per pose.

condition y (216 values) =
    context  for each of the 4 context frames: measured hand pose and commanded hand pose at that
             frame's control step                                        4 x 2 x 9 = 72
             (blocked = the commanded pose kept moving, the measured one did not)
    path     the commanded hand trajectory from now to the target time, 16 waypoints evenly spaced
             in time (backwards in time for a past target, NWM-style negative time shifts)
                                                                         16 x 9 = 144
rel_t = target time - now, in seconds (NWM's time shift k), passed to CDiT separately.
"""

from __future__ import annotations

import numpy as np

CONTROL_HZ = 20
N_WAYPOINTS = 16
POSE_DIM = 9
CONTEXT_DIM = 4 * 2 * POSE_DIM
COND_DIM = CONTEXT_DIM + N_WAYPOINTS * POSE_DIM      # 216


def pose_feature(T: np.ndarray, T_ref: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """(..., 4, 4) poses relative to T_ref (4, 4), in port axes -> (..., 9)."""
    Rt = R_P.T
    dp = (T[..., :3, 3] - T_ref[:3, 3]) @ Rt.T * 100.0                    # cm, port axes
    R = Rt @ T[..., :3, :3] @ T_ref[:3, :3].T @ R_P                       # relative rotation, port axes
    return np.concatenate([dp, R[..., :, 0], R[..., :, 1]], -1).astype(np.float32)


def _at(poses: np.ndarray, s) -> np.ndarray:
    """poses[s] with s clipped to the recorded range (commands after the end hold the last one)."""
    return poses[np.clip(np.asarray(s), 0, len(poses) - 1)]


def wm_condition(hand: np.ndarray, hand_cmd: np.ndarray, ctx_steps, s_now: int, s_target: int,
                 R_P: np.ndarray) -> np.ndarray:
    """hand (T+1, 4, 4) measured, hand_cmd (T, 4, 4) commanded, ctx_steps (4,) control steps of the
    context frames (oldest first, the last = s_now), s_target the target frame's control step."""
    ref = hand[s_now]
    ctx = np.stack([pose_feature(_at(hand, ctx_steps), ref, R_P),
                    pose_feature(_at(hand_cmd, ctx_steps), ref, R_P)], 1).reshape(-1)
    s = np.rint(np.linspace(s_now, s_target, N_WAYPOINTS + 1)[1:]).astype(int)
    path = pose_feature(_at(hand_cmd, s), ref, R_P).reshape(-1)
    return np.concatenate([ctx, path]).astype(np.float32)


def rel_t_seconds(s_now: int, s_target: int) -> float:
    return (s_target - s_now) / CONTROL_HZ
