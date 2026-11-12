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


def test_v6_encoder_and_world_model_shapes():
    """v6: a plain ResNet gives a (16, 20, 20) feature map of fixed scale; the world model takes the maps of the seen
    slots, predicts the target map (2 x 2 patches -> 100 tokens) and gives the 19 extra outputs."""
    from wm.models.frame_encoder import FrameEncoder, LabelHead
    torch.manual_seed(0)
    enc = FrameEncoder(ch=16, stride=16, arch="resnet18", pretrained=False).train()
    assert not any(m.training for m in enc.net.modules() if isinstance(m, nn.BatchNorm2d))   # statistics stay frozen
    f = enc(torch.randint(0, 255, (2, 320, 320, 3), dtype=torch.uint8), torch.randint(0, 3000, (2, 320, 320), dtype=torch.int32))
    assert f.shape == (2, 16, 20, 20)
    assert torch.allclose(f.mean(1), torch.zeros(2, 20, 20), atol=1e-4) and torch.allclose(f.var(1, unbiased=False), torch.ones(2, 20, 20), atol=1e-2)
    assert LabelHead(16, 20)(f, torch.randn(2, 9)).shape == (2, 13)
    wm = HistoryCDiT(latent_ch=16, size="S", feature_map=20, patch=2, aux_dim=19).eval()
    B, S = 2, 32
    z = torch.zeros(B, S, 16, 20, 20)
    z[:, -1] = f.detach()
    seen = torch.zeros(B, S, dtype=torch.bool)
    seen[:, -1] = True
    out = wm(torch.randn(B, 16, 20, 20), torch.randint(0, 1000, (B,)), torch.randn(B, 144), z, seen, torch.randn(B, S, 18),
             -torch.rand(B, S) * 6, torch.ones(B, S, dtype=torch.bool), torch.full((B,), 0.4))
    assert out.shape == (B, 32, 20, 20) and wm.n_tokens == 100 and wm.aux.shape == (B, 19)


def test_label_points_round_trip():
    """The label as reference-point motions: compressing and solving back gives the rigid motion it came from."""
    from wm.geometry import _exp_so3
    from wm.models.frame_encoder import REF_POINTS, label_motion, symexp, symlog
    torch.manual_seed(0)
    R = _exp_so3(torch.randn(5, 3) * 0.05)
    t = torch.randn(5, 3) * torch.tensor([0.03, 0.002, 0.002])
    p = REF_POINTS[None].expand(5, -1, -1)
    target = symlog((p @ R.transpose(1, 2) + t[:, None] - p).flatten(1) * 1000)
    R2, t2 = label_motion(target)
    assert torch.allclose(R, R2, atol=1e-4) and torch.allclose(t, t2, atol=1e-6)
    assert torch.allclose(symexp(symlog(torch.tensor([-30.0, 0.1, 5.0]))), torch.tensor([-30.0, 0.1, 5.0]), atol=1e-5)
