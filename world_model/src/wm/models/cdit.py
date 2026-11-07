"""World model: NWM's CDiT (vendored, unchanged) over the 8 x 8 latent grid.

The 64 latent tokens (63 relation + 1 joint) are laid out as an 8 x 8 "image"
with C channels and patch size 1, so CDiT's patch embedding, learned positional
embedding, blocks and final layer are used as they are. The only change is the
action embedder: NWM embeds a 3-D navigation action (x, y, yaw); here the action
is the chunk of commanded arm joint targets between two world-model steps,
relative to the measured joints (steps x 7 values), embedded by an MLP.
Diffusion: NWM's Gaussian diffusion (1000 steps, linear schedule, learned sigma).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..third_party.nwm.diffusion import create_diffusion  # noqa: F401  (re-exported)
from ..third_party.nwm.models import CDiT
from .tokenizer import N_TOKENS

GRID = 8
assert GRID * GRID == N_TOKENS
SIZES = {"S": dict(depth=12, hidden_size=384, num_heads=6),
         "B": dict(depth=12, hidden_size=768, num_heads=12)}


def tokens_to_grid(z: torch.Tensor) -> torch.Tensor:
    """(..., 64, C) -> (..., C, 8, 8)."""
    return z.unflatten(-2, (GRID, GRID)).movedim(-1, -3)


def grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(..., C, 8, 8) -> (..., 64, C)."""
    return x.movedim(-3, -1).flatten(-3, -2)


class ActionChunkEmbedder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(action_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.mlp(a)


class TokenCDiT(CDiT):
    """CDiT with the joint-target-chunk action embedder. forward(x, t, y, x_cond, rel_t) as in NWM:
    x (B,C,8,8) noisy target latent, t (B,) diffusion step, y (B, action_dim),
    x_cond (B, context, C, 8, 8) past latents, rel_t (B,) time offset of the target."""

    def __init__(self, action_dim: int, latent_ch: int = 8, context_size: int = 4, size: str = "S"):
        super().__init__(input_size=GRID, context_size=context_size, patch_size=1, in_channels=latent_ch,
                         learn_sigma=True, **SIZES[size])
        self.y_embedder = ActionChunkEmbedder(action_dim, SIZES[size]["hidden_size"])
