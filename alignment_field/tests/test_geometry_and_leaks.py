"""Fast checks (no simulator): frame algebra and that the method never reads
simulator truth. Run: python -m pytest -q tests"""

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from paf.geometry import (apply_port_motion, compose_state, make_T, port_axes, pose_from_state,
                          relative_state)

ROOT = Path(__file__).resolve().parents[1]


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


FORBIDDEN = [r"\.charger\b", r"goal_pose", r"receptacle", r"charger_v2", r"\.audit\(", r"H_pre",
             r"\bpaf\.truth\b", r"_episode_record", r"hole_offset"]


def test_method_modules_do_not_touch_truth():
    for mod in ("paf/policy.py", "paf/perception.py", "paf/model.py", "paf/field.py"):
        src = (ROOT / mod).read_text()
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        for pat in FORBIDDEN:
            assert not re.search(pat, code), f"{mod} references {pat}"


def test_hand_frame_action_matches_port_action():
    """A port-axis step a equals the hand-frame step R_hand^T R_P a (used to convert demos)."""
    from paf.fm_policy import apply_hand_motion
    from paf import config as C
    rng = np.random.default_rng(3)
    for _ in range(30):
        H = _rand_T(rng)
        R_P = port_axes(rng.normal(size=3))
        a = rng.normal(size=6) * 0.3
        M = H[:3, :3].T @ R_P
        a_h = np.r_[M @ a[:3], M @ a[3:]]
        assert np.allclose(apply_port_motion(H, a * C.SIGMA, R_P), apply_hand_motion(H, a_h), atol=1e-9)
