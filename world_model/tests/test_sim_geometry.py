"""Fast checks (no simulator) of the pose and frame algebra in wm.sim.geometry.
Kept from the retired alignment_field tests; its leak test covered only the retired method's modules
(the world model's own leak test is tests/test_no_leak.py)."""

import numpy as np
from scipy.spatial.transform import Rotation

from wm.sim.geometry import (apply_port_motion, compose_state, make_T, port_axes, pose_from_state,
                             relative_state)

def _rand_T(rng):
    return make_T(Rotation.random(random_state=rng.integers(1 << 31)).as_matrix(), rng.normal(size=3) * 0.3)


def test_state_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        H_pre = _rand_T(rng)
        R_P = port_axes(rng.normal(size=3))
        q = np.r_[rng.normal(size=3) * 0.01, rng.normal(size=3) * 0.1]
        assert np.allclose(relative_state(pose_from_state(q, H_pre, R_P), H_pre, R_P), q, atol=1e-9)


def test_motion_matches_state_composition():
    """Executing dq with apply_port_motion moves the state as compose_state says,
    without knowing H_pre (what the method does at test time)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        H_pre = _rand_T(rng)
        R_P = port_axes(rng.normal(size=3))
        q = np.r_[rng.normal(size=3) * 0.01, rng.normal(size=3) * 0.1]
        dq = np.r_[rng.normal(size=3) * 0.002, rng.normal(size=3) * 0.02]
        H = pose_from_state(q, H_pre, R_P)
        q2 = relative_state(apply_port_motion(H, dq, R_P), H_pre, R_P)
        assert np.allclose(q2, compose_state(q, dq), atol=1e-9)


def test_port_axes_orthonormal_and_up():
    R = port_axes(np.array([0.99, 0.13, 0.01]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) > 0
    assert R[:, 2] @ np.array([0, 0, 1.0]) > 0.99
