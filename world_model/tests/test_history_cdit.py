"""The fused masked cross-attention must equal nn.MultiheadAttention with a key padding mask."""

import torch
import torch.nn as nn

from wm.models.history_cdit import HistoryCDiT, masked_cross_attention


def test_masked_cross_attention_equals_mha():
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(64, 4, add_bias_kv=True, bias=True, batch_first=True).eval()
    q, k = torch.randn(3, 10, 64), torch.randn(3, 50, 64)
    ignore = torch.rand(3, 50) < 0.4
    ignore[2] = True                                       # a sample with every key masked (bias key remains)
    ref = mha(q, k, k, key_padding_mask=ignore, need_weights=False)[0]
    out = masked_cross_attention(mha, q, k, ignore)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()


def test_history_cdit_forward_runs():
    m = HistoryCDiT(latent_ch=16, size="S").eval()
    B, S = 2, 32
    out = m(torch.randn(B, 16, 8, 8), torch.randint(0, 1000, (B,)), torch.randn(B, 144), torch.randn(B, S, 64, 16),
            torch.rand(B, S) < 0.5, torch.randn(B, S, 18), -torch.rand(B, S) * 6, torch.ones(B, S, dtype=torch.bool),
            torch.full((B,), 0.4))
    assert out.shape == (B, 32, 8, 8) and torch.isfinite(out).all()


def test_history_cdit_takes_region_tokens():
    """v5 latents: 128 tokens as a (16, 8) grid; the square case still unpatchifies as NWM's CDiT does."""
    from wm.models.cdit import grid_to_tokens, tokens_to_grid
    from wm.third_party.nwm.models import CDiT
    torch.manual_seed(0)
    z = torch.randn(2, 128, 16)
    assert tokens_to_grid(z).shape == (2, 16, 16, 8) and torch.equal(grid_to_tokens(tokens_to_grid(z)), z)
    m = HistoryCDiT(latent_ch=16, size="S", n_tokens=128).eval()
    B, S = 2, 32
    out = m(tokens_to_grid(z), torch.randint(0, 1000, (B,)), torch.randn(B, 144), torch.randn(B, S, 128, 16),
            torch.rand(B, S) < 0.5, torch.randn(B, S, 18), -torch.rand(B, S) * 6, torch.ones(B, S, dtype=torch.bool),
            torch.full((B,), 0.4))
    assert out.shape == (B, 32, 16, 8) and torch.isfinite(out).all()
    sq = HistoryCDiT(latent_ch=16, size="S")
    x = torch.randn(2, 64, 32)
    assert torch.equal(sq.unpatchify(x), CDiT.unpatchify(sq, x))
