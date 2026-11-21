"""SE(3) helpers and the port-frame relative state q.

Everything here is built only from quantities a real robot also has:
hand (TCP) poses from forward kinematics, the insertion axis measured from
the hand motion during one insertion, and the gravity direction of the robot
base. No simulator object pose is used.

Notation (all lengths in metres inside the code, reported in mm):
    H      hand pose in the robot base frame, a 4x4 homogeneous matrix
    H_pre  pre-insertion hand pose: the inserted hand pose moved back along
           the insertion axis until the pin tips are `gap0` outside the board
    R_P    port-frame axes [e_x, e_y, e_z] as columns, e_x = insertion axis
           (pointing into the board), e_z = gravity "up" made orthogonal to
           e_x, e_y = e_z x e_x
    q      relative state of H with respect to H_pre, written in port axes:
           q = (t, r), t = R_P^T (p - p_pre)        (translation, metres)
                       r = Log(R_P^T R R_pre^T R_P) (rotation vector, rad)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

UP = np.array([0.0, 0.0, 1.0])


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def pose_to_T(p, q_wxyz) -> np.ndarray:
    """sapien/ManiSkill pose (p, q in wxyz order) -> 4x4."""
    w, x, y, z = q_wxyz
    return make_T(Rotation.from_quat([x, y, z, w]).as_matrix(), np.asarray(p, float))


def T_to_pose(T: np.ndarray):
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return T[:3, 3].copy(), np.array([w, x, y, z])


def port_axes(axis: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """Port axes from the insertion axis and gravity-up (columns e_x,e_y,e_z)."""
    ex = axis / np.linalg.norm(axis)
    ez = up - np.dot(up, ex) * ex
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez], axis=1)


def relative_state(H: np.ndarray, H_pre: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """q = (t[3], r[3]) of hand pose H relative to H_pre, in port axes."""
    t = R_P.T @ (H[:3, 3] - H_pre[:3, 3])
    dR = R_P.T @ H[:3, :3] @ H_pre[:3, :3].T @ R_P
    r = Rotation.from_matrix(dR).as_rotvec()
    return np.concatenate([t, r])


def pose_from_state(q: np.ndarray, H_pre: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """Inverse of relative_state: the hand pose whose state is q."""
    p = H_pre[:3, 3] + R_P @ q[:3]
    R = R_P @ Rotation.from_rotvec(q[3:]).as_matrix() @ R_P.T @ H_pre[:3, :3]
    return make_T(R, p)


def apply_port_motion(H: np.ndarray, dq: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """Move the hand by dq = (dt, dr) expressed in port axes, rotating about
    the hand position. This is how a field output is executed at test time:
    it needs only the current hand pose (FK) and the port axes, never H_pre."""
    p = H[:3, 3] + R_P @ dq[:3]
    R = R_P @ Rotation.from_rotvec(dq[3:]).as_matrix() @ R_P.T @ H[:3, :3]
    return make_T(R, p)


def compose_state(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """State after applying apply_port_motion(dq) to a hand in state q."""
    t = q[:3] + dq[:3]
    R = Rotation.from_rotvec(dq[3:]).as_matrix() @ Rotation.from_rotvec(q[3:]).as_matrix()
    return np.concatenate([t, Rotation.from_matrix(R).as_rotvec()])


def interpolate_T(A: np.ndarray, B: np.ndarray, s: float) -> np.ndarray:
    """Straight-line position, geodesic rotation interpolation."""
    p = (1 - s) * A[:3, 3] + s * B[:3, 3]
    dR = Rotation.from_matrix(B[:3, :3] @ A[:3, :3].T).as_rotvec()
    R = Rotation.from_rotvec(s * dR).as_matrix() @ A[:3, :3]
    return make_T(R, p)


def pose_distance(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(translation m, rotation rad) between two poses."""
    dt = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    dr = float(np.linalg.norm(Rotation.from_matrix(A[:3, :3] @ B[:3, :3].T).as_rotvec()))
    return dt, dr
