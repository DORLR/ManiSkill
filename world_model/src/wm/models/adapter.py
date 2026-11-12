"""Joint adapter (Ctrl-World's adapter, for joint position targets).

Ctrl-World: a two-layer MLP maps the current joint angles and the policy's joint velocities to the
future joint configuration; FK of those gives the hand poses that condition the world model, which
itself never predicts joints. Here the policy commands joint position targets, tracked by the arm's
controller, so the adapter maps

    measured joints now | tracking history (command - measured, last 1.2 s) | next 8 targets
        -> the joints measured after each of the next 8 commands (relative to now)

Trained on all recorded trajectories, jams included: once a push is blocked the history shows the
gap and the adapter predicts that the joints stay. Inputs and outputs are standardised with
train-split statistics stored in the checkpoint.

VisionJointAdapter (2026-11-11, decided with the user). The adapter above cannot know, at the moment the pins
reach the face, whether they enter or stop: the tracking history shows the gap only afterwards. In imagination
the hand then follows the command and the world model, given a hand that advanced by the pin length, draws an
insertion (imagined 92.8 % vs 25.5 % in the simulator on the FM's states). Ctrl-World's setting does not have this
case (free-space motion; its authors exclude contact-rich tasks). So the adapter also gets what the camera saw:

    the 231 numbers above | the history the world model and the success classifier see (latents of the seen
    slots, proprioception of every slot; attention over the seen slots only) | the commanded hand path of the next
    0.4 s (16 waypoints, relative to the measured hand, port axes: the world model's own conditioning)
        -> the same 56 outputs

With `outcome` (2026-11-11 evening, decided with the user) it also gets the success classifier's decision
"inserted within its horizon" as a 0/1 input: the true label when training, the classifier's (calibrated) output
against one random threshold per imagined episode in imagination (imagination4.py). Without it a stuck plug is
not an equilibrium in imagination: the regression lets the hand advance about 60 % of every commanded lead (0.37 of 0.58 mm per 0.4 s), the
policy re-bases its targets on the new joints, and the hand creeps 16 mm in 16 s where the real one moves 0.05 mm.

It is called at image steps only (one decision = 0.4 s = 8 commands starts at an image). The history encoder is
the classifier's (models/success.py); the numeric input and the path enter it as two more tokens, and the numeric
input also goes straight to the output MLP, so the blind adapter is the special case of an ignored history.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.history import PATH_DIM

IN_DIM = 7 + 24 * 7 + 8 * 7     # 231
OUT_DIM = 8 * 7                 # 56


class JointAdapter(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, hidden), nn.ReLU(), nn.Linear(hidden, OUT_DIM))
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(name, torch.zeros(IN_DIM if name[0] == "x" else OUT_DIM))
        self.x_std.fill_(1.0)
        self.y_std.fill_(1.0)

    def set_stats(self, x_mean, x_std, y_mean, y_std):
        for name, v in zip(("x_mean", "x_std", "y_mean", "y_std"), (x_mean, x_std, y_mean, y_std)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def forward_normalised(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_mean) / self.x_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 231) as data.datasets.adapter_io -> (B, 56) next 8 measured joints minus joints now (rad)."""
        return self.forward_normalised(x) * self.y_std + self.y_mean


class VisionJointAdapter(nn.Module):
    takes_history = True

    def __init__(self, latent_ch: int = 16, d: int = 256, layers: int = 6, heads: int = 8, hidden: int = 512,
                 outcome: bool = False):
        super().__init__()
        self.takes_outcome = outcome
        from .success import SuccessHead
        self.hist = SuccessHead(latent_ch, d, layers, heads)
        self.hist.out = nn.Identity()                                   # the classifier's logit head is not used
        self.x_tok = nn.Sequential(nn.Linear(IN_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.path_tok = nn.Sequential(nn.Linear(PATH_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.extra_type = nn.Parameter(torch.randn(1, 3 if outcome else 2, d) * 0.02)
        if outcome:
            self.o_emb = nn.Embedding(2, d)
            self.register_buffer("outcome_prior", torch.tensor(float("nan")))    # share of positive training samples
        self.net = nn.Sequential(nn.Linear(IN_DIM + d + int(outcome), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, OUT_DIM))
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(name, torch.zeros(IN_DIM if name[0] == "x" else OUT_DIM))
        self.x_std.fill_(1.0)
        self.y_std.fill_(1.0)

    set_stats = JointAdapter.set_stats

    def forward_normalised(self, x, z_hist, seen, proprio, dt, avail, path, outcome=None) -> torch.Tensor:
        xn = (x - self.x_mean) / self.x_std
        assert (outcome is not None) == self.takes_outcome
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            toks = [self.x_tok(xn), self.path_tok(path)] + ([self.o_emb(outcome.long())] if self.takes_outcome else [])
            f = self.hist.features(z_hist, seen, proprio, dt, avail, torch.stack(toks, 1) + self.extra_type)
        tail = [outcome.float()[:, None]] if self.takes_outcome else []
        return self.net(torch.cat([xn, f.float()] + tail, 1))           # joints to 0.01 mrad: full precision

    def forward(self, x, z_hist, seen, proprio, dt, avail, path, outcome=None) -> torch.Tensor:
        """x (B, 231) as data.datasets.adapter_io; the history as data.history.history; path (B, 144); outcome (B,)
        0/1 if the adapter takes it -> (B, 56) next 8 measured joints minus joints now (rad)."""
        return self.forward_normalised(x, z_hist, seen, proprio, dt, avail, path, outcome) * self.y_std + self.y_mean
