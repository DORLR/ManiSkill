"""Success classifier v4: is the plug inserted now? -- from the same history the world model sees.

Input: the HIST (32) slots of 0.2 s (data/history.py): latents of the slots that were seen (attention
only over those; no "not seen" flag), and one proprioception token per slot (never masked). Used on
real latents and on imagined ones (world-model predictions), so it is trained with the same vision
masking and with context noise.

Per seen slot the 64 latent tokens are summarised by attention pooling (4 learned queries, shared),
then a transformer over [CLS, 4 x HIST vision tokens, HIST proprio tokens] with key padding for the
unseen / unavailable slots; the CLS output -> one logit. Label: inserted at the current image
(data/outcome.py: the evaluation criterion; on the real robot, the connector's contact switch).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.history import HIST, PROPRIO_DIM
from ..third_party.nwm.models import TimestepEmbedder
from .history_cdit import DT_SCALE

POOL = 4


class SuccessHead(nn.Module):
    def __init__(self, latent_ch: int = 16, d: int = 256, layers: int = 6, heads: int = 8):
        super().__init__()
        self.inp = nn.Linear(latent_ch, d)
        self.token_pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.pool_q = nn.Parameter(torch.randn(POOL, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.slot_time = TimestepEmbedder(d)
        self.proprio = nn.Sequential(nn.Linear(PROPRIO_DIM, d), nn.SiLU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.randn(3, 1, d) * 0.02)          # cls, vision, proprio
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def features(self, z_hist, seen, proprio, dt, avail, extra=None) -> torch.Tensor:
        """The CLS output (B, d). `extra` (B, n, d): further tokens, always attended (the vision-conditioned joint
        adapter adds its numeric input and the commanded path, models/adapter.py)."""
        B, S = seen.shape
        te = self.slot_time((-dt * DT_SCALE).reshape(-1, 1)).view(B, S, -1)
        tok = self.inp(z_hist) + self.token_pos                                       # (B, S, 64, d)
        q = self.pool_q[None].expand(B * S, -1, -1)
        pooled = self.pool(q, tok.flatten(0, 1), tok.flatten(0, 1), need_weights=False)[0]
        v = self.pool_norm(pooled).view(B, S, POOL, -1) + te[:, :, None] + self.type_emb[1]
        p = self.proprio(proprio) + te + self.type_emb[2]
        x = torch.cat([self.cls.expand(B, -1, -1) + self.type_emb[0], v.flatten(1, 2), p] + ([extra] if extra is not None else []), 1)
        ignore = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seen.device),
                            (~seen).repeat_interleave(POOL, 1), ~avail]
                           + ([torch.zeros(B, extra.shape[1], dtype=torch.bool, device=seen.device)] if extra is not None else []), 1)
        return self.enc(x, src_key_padding_mask=ignore)[:, 0]

    def forward(self, z_hist, seen, proprio, dt, avail) -> torch.Tensor:
        """z_hist (B,HIST,64,C), seen / avail (B,HIST) bool, proprio (B,HIST,18), dt (B,HIST) -> logit (B,)."""
        return self.out(self.features(z_hist, seen, proprio, dt, avail))[:, 0]
