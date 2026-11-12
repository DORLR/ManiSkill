"""World model: NWM's CDiT (vendored unchanged in third_party/nwm) over the 8 x 8 latent grid.

The 64 latent tokens are laid out as an 8 x 8 "image" with C channels and patch size 1, so CDiT's
patch embedding, positional embedding, blocks and final layer are used as they are, including
NWM's time-shift input rel_t (how far in the future -- or past -- the target frame is).

Changes, all in this subclass:
* action embedder: NWM embeds a navigation action (x, y, yaw) summed over the time shift; here the
  condition is the robot state + commanded hand path (data/conditions.py, 216 values), embedded by
  an MLP.
* context-noise level (GameNGen-style augmentation, used when training): the context latents are
  noised with a random level sigma and sigma is embedded and added to the conditioning vector, so
  the model learns to work from imperfect (its own, predicted) context. At use time sigma = 0.
  The embedder's last layer starts at zero, so with sigma ignored the model is exactly NWM's.
Diffusion: NWM's Gaussian diffusion (1000 steps, linear schedule, learned sigma).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..third_party.nwm.diffusion import create_diffusion  # noqa: F401  (re-exported)
from ..third_party.nwm.models import CDiT, TimestepEmbedder
from .encoder import N_TOKENS

GRID = 8
assert GRID * GRID == N_TOKENS
SIZES = {"S": dict(depth=12, hidden_size=384, num_heads=6),
         "B": dict(depth=12, hidden_size=768, num_heads=12)}


def tokens_to_grid(z: torch.Tensor) -> torch.Tensor:
    """(..., 64, C) -> (..., C, 8, 8); v5 latents (..., 128, C) -> (..., C, 16, 8): the plug crop's 8 x 8 region
    tokens above the opening crop's 8 x 8."""
    return z.unflatten(-2, (z.shape[-2] // GRID, GRID)).movedim(-1, -3)


def grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(..., C, 8, 8) -> (..., 64, C); (..., C, 16, 8) -> (..., 128, C)."""
    return x.movedim(-3, -1).flatten(-3, -2)


class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(y)


class TokenCDiT(CDiT):
    """forward(x, t, y, x_cond, rel_t, ctx_noise=None):
    x (B,C,8,8) noisy target latent, t (B,) diffusion step, y (B, cond_dim) robot state + commanded
    path, x_cond (B, context, C, 8, 8) context latents, rel_t (B,) time shift in seconds,
    ctx_noise (B,) the noise level added to the context (0 or None at use time)."""

    def __init__(self, cond_dim: int, latent_ch: int = 8, context_size: int = 4, size: str = "S"):
        super().__init__(input_size=GRID, context_size=context_size, patch_size=1, in_channels=latent_ch,
                         learn_sigma=True, **SIZES[size])
        h = SIZES[size]["hidden_size"]
        self.y_embedder = ConditionEmbedder(cond_dim, h)
        self.noise_embedder = TimestepEmbedder(h)
        nn.init.normal_(self.noise_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.noise_embedder.mlp[2].weight)
        nn.init.zeros_(self.noise_embedder.mlp[2].bias)

    def forward(self, x, t, y, x_cond, rel_t, ctx_noise=None):
        # identical to CDiT.forward (third_party/nwm/models.py) except the added noise-level term
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) \
            + self.pos_embed[:self.context_size]
        x_cond = x_cond.flatten(1, 2)
        c = self.t_embedder(t[..., None]) + self.time_embedder(rel_t[..., None]) + self.y_embedder(y)
        if ctx_noise is None:
            ctx_noise = torch.zeros_like(rel_t)
        c = c + self.noise_embedder(ctx_noise[..., None] * 1000)       # sigma in [0, 0.1] -> [0, 100]
        for block in self.blocks:
            x = block(x, c, x_cond)
        x = self.final_layer(x, c)
        return self.unpatchify(x)
