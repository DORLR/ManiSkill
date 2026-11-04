"""The alignment field (SID-style motion field toward the in-distribution set).

State q in R^6 (port-axis translation and rotation vector of the hand relative
to the pre-insertion pose), measured in sigma units u = q / SIGMA
(SIGMA = 1 mm, 1 deg), so translation and rotation are comparable (SID's
d(x, x_j) with sigma_p, sigma_r).

In-distribution set  M = { u_j } : states from which the fixed straight push
                               inserted (training grasps only, pooled over the
                               three training shapes).

Energy (soft nearest-neighbour; Back to the Manifold's -log density):
    E_h(u) = -h^2 log( (1/|M|) sum_j exp( -||u - u_j||^2 / (2 h^2) ) )
    h -> 0 gives E = 1/2 min_j ||u - u_j||^2, SID's potential.

Field = negative gradient (mean shift):
    v(u) = -grad E_h(u) = sum_j w_j(u) (u_j - u),
    w_j  = softmax_j( -||u - u_j||^2 / (2 h^2) ).
v points to the weighted nearest ID states, its length is the distance to
them, and it vanishes at the modes of M (field "turns off" when in
distribution, as in SID).

Executed step (distance-scheduled with clipping, SID's eta(d)):
    du = v * min(1, s_max / ||v||)

Learned field (SID's f_theta): a small MLP regressing v(u) on states sampled
uniformly over the working box; this is what runs in the loop. It is fitted
from M alone and never sees images.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import mlp


def analytic_field(u: torch.Tensor, M: torch.Tensor, h: float) -> torch.Tensor:
    """v(u) for a batch u (B,6) given ID set M (N,6), both in sigma units."""
    d2 = torch.cdist(u, M) ** 2                      # (B, N)
    w = torch.softmax(-d2 / (2 * h * h), dim=1)
    return w @ M - u


def energy(u: torch.Tensor, M: torch.Tensor, h: float) -> torch.Tensor:
    d2 = torch.cdist(u, M) ** 2
    return -h * h * (torch.logsumexp(-d2 / (2 * h * h), 1) - np.log(M.shape[0]))


class FieldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp((6, 256, 256, 256, 6))

    def forward(self, u):
        return self.net(u)


def fit_field(M: np.ndarray, h: float, box: np.ndarray, steps: int = 6000, device="cuda", seed=0):
    """Regress v(u) over u ~ Uniform(-box, box) plus u ~ near M (SID)."""
    g = torch.Generator(device=device).manual_seed(seed)
    Mt = torch.as_tensor(M, dtype=torch.float32, device=device)
    boxt = torch.as_tensor(box, dtype=torch.float32, device=device)
    net = FieldNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        a = (torch.rand(2048, 6, device=device, generator=g) * 2 - 1) * boxt
        b = Mt[torch.randint(len(Mt), (1024,), device=device, generator=g)] + \
            torch.randn(1024, 6, device=device, generator=g) * 1.5
        u = torch.cat([a, b])
        with torch.no_grad():
            v = analytic_field(u, Mt, h)
        loss = ((net(u) - v) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    # held-out fit error
    with torch.no_grad():
        u = (torch.rand(4096, 6, device=device, generator=g) * 2 - 1) * boxt
        err = (net(u) - analytic_field(u, Mt, h)).norm(dim=1)
        ref = analytic_field(u, Mt, h).norm(dim=1)
    return net, {"rmse": float(err.pow(2).mean().sqrt()), "mean_norm": float(ref.mean())}


def step_from_field(v: np.ndarray, s_max: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v * min(1.0, s_max / max(n, 1e-9))
