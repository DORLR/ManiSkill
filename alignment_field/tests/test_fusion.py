"""Multi-frame fusion in FieldPolicy returns the exact state when estimates are exact."""
import numpy as np

from paf import config as C
from paf.geometry import pose_from_state, port_axes, relative_state
from paf.policy import FieldPolicy


def test_fusion_is_exact_for_exact_estimates():
    rng = np.random.default_rng(0)
    fp = FieldPolicy.__new__(FieldPolicy)
    fp.fuse, fp.history = 5, []
    fp.R_P = port_axes(np.array([1.0, 0.2, -0.1]))
    ref = pose_from_state(np.zeros(6), np.eye(4), np.eye(3))
    ref[:3, 3] = [0.4, 0.1, 0.3]
    for _ in range(8):
        q = np.r_[rng.uniform(-0.01, 0.01, 3), rng.uniform(-0.1, 0.1, 3)]
        H = pose_from_state(q, ref, fp.R_P)
        u = relative_state(H, ref, fp.R_P) / C.SIGMA
        assert np.allclose(fp._fused(H, u), u, atol=1e-6)
