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
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
