"""World model v4: NWM's CDiT (vendored unchanged) with a 6.2 s history of irregularly seen frames.

What stays NWM's: the target latent as an 8 x 8 grid with patch size 1, the blocks (self-attention on
the target tokens, cross-attention to the context, adaLN-Zero conditioning on the diffusion step and
the time shift rel_t), the final layer, the Gaussian diffusion with learned sigma.

What changes (all in this subclass, the vendored code is not edited):
* context = up to HIST (32) slots of 0.2 s instead of a fixed 4-frame stack. Each slot that has a
  latent contributes its 64 tokens: CDiT's patch embedding of the latent + one spatial position
  embedding (CDiT's own, shared by all slots) + an embedding of the slot's time (seconds before now,
  NWM's sinusoidal timestep embedder). Slots without a latent (before the start, masked when training,
  in between imagined frames) contribute nothing: their keys are masked out in every block's
  cross-attention (key_padding_mask), so attention only sees the frames that were observed.
* every slot also gives one proprioception token (measured + commanded hand pose, data/history.py, +
  the slot's time) -- never masked: the robot always knows where its hand is and where it was told
  to go, also when the camera sees nothing useful. Blocked = commanded kept moving, measured did not.
* the conditioning vector: diffusion step + rel_t (as NWM) + the commanded hand path from now to the
  target time (16 waypoints, MLP; NWM sums navigation actions) + the context-noise level (GameNGen
  augmentation, 0 at use time; the embedder starts at zero).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.history import HIST, PATH_DIM, PROPRIO_DIM
from ..third_party.nwm.diffusion import create_diffusion  # noqa: F401  (re-exported)
from ..third_party.nwm.models import CDiT, TimestepEmbedder, modulate
from .cdit import GRID, SIZES, ConditionEmbedder, grid_to_tokens, tokens_to_grid  # noqa: F401

DT_SCALE = 20.0            # slot time in control steps (0.05 s) for the sinusoidal embedding


class HistoryCDiT(CDiT):
    """forward(x, t, path, z_hist, seen, proprio, dt, avail, rel_t, ctx_noise=None)
    x (B,C,8,8) noisy target, t (B,) diffusion step, path (B,144), z_hist (B,HIST,64,C) scaled latents,
    seen (B,HIST) bool slot has a latent, proprio (B,HIST,18), dt (B,HIST) seconds (<= 0),
    avail (B,HIST) bool slot exists (after the trajectory start), rel_t (B,) s, ctx_noise (B,)."""

    def __init__(self, latent_ch: int = 16, size: str = "S"):
        super().__init__(input_size=GRID, context_size=1, patch_size=1, in_channels=latent_ch, learn_sigma=True,
                         **SIZES[size])
        h = SIZES[size]["hidden_size"]
        self.hidden = h
        self.y_embedder = ConditionEmbedder(PATH_DIM, h)
        self.slot_time = TimestepEmbedder(h)
        self.proprio_embedder = nn.Sequential(nn.Linear(PROPRIO_DIM, h), nn.SiLU(), nn.Linear(h, h))
        self.proprio_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.noise_embedder = TimestepEmbedder(h)
        nn.init.zeros_(self.noise_embedder.mlp[2].weight)
        nn.init.zeros_(self.noise_embedder.mlp[2].bias)

    def context(self, z_hist, seen, proprio, dt, avail):
        """Keys / values for the cross-attention and their padding mask (True = ignore)."""
        B, S = seen.shape
        te = self.slot_time((-dt * DT_SCALE).reshape(-1, 1)).view(B, S, 1, -1)              # (B, S, 1, h)
        v = self.x_embedder(tokens_to_grid(z_hist).flatten(0, 1)).unflatten(0, (B, S))  # (B, S, 64, h)
        v = (v + self.pos_embed[0] + te).flatten(1, 2)                                   # (B, S*64, h)
        p = self.proprio_embedder(proprio) + te[:, :, 0] + self.proprio_type             # (B, S, h)
        keys = torch.cat([v, p], 1)
        ignore = torch.cat([(~seen).repeat_interleave(GRID * GRID, 1), ~avail], 1)
        return keys, ignore

    def forward(self, x, t, path, z_hist, seen, proprio, dt, avail, rel_t, ctx_noise=None):
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        keys, ignore = self.context(z_hist, seen, proprio, dt, avail)
        c = self.t_embedder(t[..., None]) + self.time_embedder(rel_t[..., None]) + self.y_embedder(path)
        if ctx_noise is None:
            ctx_noise = torch.zeros_like(rel_t)
        c = c + self.noise_embedder(ctx_noise[..., None] * 1000)
        for block in self.blocks:
            x = _block_forward(block, x, c, keys, ignore)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def masked_cross_attention(mha: nn.MultiheadAttention, query, key, ignore):
    """nn.MultiheadAttention(add_bias_kv=True, batch_first=True)(query, key, key, key_padding_mask=ignore),
    computed with the same weights by torch's fused scaled_dot_product_attention (memory-efficient
    kernels; nn.MultiheadAttention falls back to its slow path when a padding mask is given).
    tests/test_history_cdit.py checks that both give the same output."""
    B, Lq, E = query.shape
    H = mha.num_heads
    w, bias = mha.in_proj_weight, mha.in_proj_bias
    q = F.linear(query, w[:E], bias[:E])
    k = F.linear(key, w[E:2 * E], bias[E:2 * E])
    v = F.linear(key, w[2 * E:], bias[2 * E:])
    k = torch.cat([k, mha.bias_k.expand(B, 1, E)], 1)                  # add_bias_kv: one extra key / value,
    v = torch.cat([v, mha.bias_v.expand(B, 1, E)], 1)                  # never masked
    keep = torch.cat([~ignore, ignore.new_ones(B, 1)], 1)              # True = attend
    split = lambda t: t.view(B, -1, H, E // H).transpose(1, 2)
    o = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=keep[:, None, None, :])
    return mha.out_proj(o.transpose(1, 2).reshape(B, Lq, E))


def _block_forward(blk, x, c, x_cond, ignore):
    """CDiTBlock.forward (third_party/nwm/models.py) with a key padding mask in the cross-attention.
    The block's own bias key (add_bias_kv) is always attendable, so a sample with no key left is fine."""
    (shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x,
     shift_mlp, scale_mlp, gate_mlp) = blk.adaLN_modulation(c).chunk(11, dim=1)
    x = x + gate_msa.unsqueeze(1) * blk.attn(modulate(blk.norm1(x), shift_msa, scale_msa))
    x_cond_norm = modulate(blk.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
    x = x + gate_ca_x.unsqueeze(1) * masked_cross_attention(blk.cttn, modulate(blk.norm2(x), shift_ca_x, scale_ca_x),
                                                            x_cond_norm, ignore)
    x = x + gate_mlp.unsqueeze(1) * blk.mlp(modulate(blk.norm3(x), shift_mlp, scale_mlp))
    return x
