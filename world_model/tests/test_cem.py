"""CEM planner (online/actor.py plan_cem): it must find a better offset than the FM's own proposals when the
imagined score rewards one, include the unperturbed proposals in the first iteration, and return targets of
the right shape."""

import numpy as np
import torch

from wm.online.actor import plan_cem, smooth_offsets


class FakeImag:
    """Score = closeness of the executed targets' mean joint-0 offset to a goal (stands in for the world
    model + success classifier); the FM continuation is irrelevant here."""

    def __init__(self, goal):
        self.goal, self.q = goal, None

    def reset_from(self, starts):
        self.B = len(starts)
        self.q = np.zeros((self.B, 9))

    def step(self, targets):
        s = np.exp(-((targets[:, :8, 0].mean(1) - self.goal) / 0.002) ** 2)
        return {"p_success": s}

    def obs(self):
        return {"z": torch.zeros(self.B, 4, 64, 16), "q": torch.zeros(self.B, 7)}


class FakeModels:
    def fm(self, z, q):
        return torch.zeros(z.shape[0], 16, 7)


def test_smooth_offsets_shape_and_ramp():
    p = np.zeros((2, 14))
    p[0, 0], p[0, 7] = 0.001, 0.0005
    o = smooth_offsets(p, 16)
    assert o.shape == (2, 16, 7)
    assert np.isclose(o[0, 0, 0], 0.001) and np.isclose(o[0, 15, 0], 0.001 + 15 * 0.0005)
    assert np.allclose(o[1], 0)


def test_cem_improves_on_fm_proposals():
    cfg = dict(cem_population=32, cem_elites=8, cem_iterations=3, cem_sigma_a=0.002, cem_sigma_b=0.0003,
               cem_min_sigma_a=0.0003, cem_min_sigma_b=0.00005, horizon=2)
    proposals = np.zeros((8, 16, 7))                    # the FM proposes "no offset"
    goal = 0.003                                        # the imagined success needs +3 mrad on joint 0
    best, score, fm_scores = plan_cem(FakeImag(goal), FakeModels(), {}, proposals, np.zeros(9), cfg,
                                      np.random.default_rng(0))
    assert best.shape == (16, 7)
    assert fm_scores.shape == (8,) and fm_scores.max() < 0.2      # the FM's own proposals score badly
    assert score > 0.8                                            # CEM found the offset
    assert abs(best[:8, 0].mean() - goal) < 0.001
