"""CPU shape, gradient and geometry checks of the v3 models (no DINOv2 weights, no simulator)."""

import math

import numpy as np
import torch

from wm.data.conditions import COND_DIM, pose_feature, wm_condition
from wm.geometry import (insertion_motion, plug_points_at_insertion, plug_region_points, pose_loss_terms,
                         rot6d_to_matrix, rotation_angle_deg)
from wm.models.adapter import IN_DIM, OUT_DIM, JointAdapter
from wm.models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from wm.models.encoder import N_PATCH, prepare
from wm.models.vae import TokenVAE
from wm.training.losses import insertion_pose, reconstruction

ENC = dict(d=32, layers=1, query_layers=1, heads=4, dino=False)


def _batch(B=2):
    depth = np.zeros((B, 320, 320), np.int32)
    depth[:, 100:200, 100:200] = 1500                  # some returns, the rest has no depth (0)
    K = torch.tensor([[300.0, 0, 160], [0, 300.0, 160], [0, 0, 1]]).expand(B, 3, 3)
    eye = torch.eye(4).expand(B, 4, 4).clone()
    return {"rgb": torch.randint(0, 255, (B, 320, 320, 3), dtype=torch.uint8), "depth": torch.from_numpy(depth),
            "K": K, "cam_T_world": eye, "hand": eye.clone(), "R_P": torch.eye(3).expand(B, 3, 3), "h_star": eye.clone()}


def test_depth_kept_as_is():
    b = _batch()
    p = prepare(b["rgb"], b["depth"])
    assert p["depth_patch"].shape == (2, N_PATCH, 196)
    assert (p["depth_patch"] == 0).any() and (p["depth_patch"] > 0).any()   # missing depth stays exactly 0


def test_autoencoder_forward_backward():
    torch.manual_seed(0)
    m = TokenVAE(encoder=ENC, latent_ch=8, dec_d=32, dec_layers=1, patch_layers=1, heads=4)
    b = _batch()
    out = m(b)
    assert out["z"].shape == (2, 64, 8) and out["pred"]["depth_patch"].shape == (2, N_PATCH, 196)
    l = reconstruction(out["pred"], out["target"])
    l.update(insertion_pose(out["pred"]["R"], out["pred"]["t"], b["hand"], b["h_star"], b["R_P"], plug_region_points()))
    sum(l.values()).backward()
    assert all(torch.isfinite(v) for v in l.values())


def test_insertion_angle_and_grid_zero_for_true_motion():
    B, a = 2, 0.01
    hand = torch.eye(4).expand(B, 4, 4).clone()
    hand[:, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    h_star = hand.clone()
    h_star[:, :3, :3] = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    R_P = rot6d_to_matrix(torch.randn(B, 6))
    R_star, t_star = insertion_motion(hand, h_star, R_P)
    p = torch.randn(B, 10, 3) * 0.02
    assert torch.allclose(plug_points_at_insertion(p, hand, h_star, R_P), p @ R_star.transpose(1, 2) + t_star[:, None], atol=1e-6)
    g, ang = pose_loss_terms(R_star, t_star, R_star, t_star, plug_region_points())
    assert float(g.max()) < 1e-4 and float(ang.max()) < 2e-2
    assert abs(float(rotation_angle_deg(R_star, torch.eye(3).expand(B, 3, 3))[0]) - math.degrees(a)) < 1e-2


def test_rot6d_is_rotation():
    R = rot6d_to_matrix(torch.randn(5, 6))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(5), atol=1e-5)


def test_condition_shapes_and_frame():
    T = 50
    hand = np.tile(np.eye(4), (T + 1, 1, 1))
    hand[:, 0, 3] = np.arange(T + 1) * 1e-3                 # moving 1 mm per step along world x
    cmd = hand[:T].copy()
    cmd[:, 0, 3] += 2e-3                                      # commands 2 mm ahead
    R_P = np.eye(3)
    y = wm_condition(hand, cmd, np.array([0, 8, 16, 24]), 24, 40, R_P)
    assert y.shape == (COND_DIM,)
    f = pose_feature(hand[30], hand[24], R_P)
    assert np.allclose(f[:3], [0.6, 0, 0]) and np.allclose(f[3:], [1, 0, 0, 0, 1, 0])   # 6 mm = 0.6 cm


def test_cdit_diffusion_step_with_time_shift_and_context_noise():
    torch.manual_seed(0)
    z = torch.randn(2, 64, 8)
    assert torch.equal(grid_to_tokens(tokens_to_grid(z)), z)
    model = TokenCDiT(cond_dim=COND_DIM, latent_ch=8, context_size=4, size="S")
    diffusion = create_diffusion(timestep_respacing="")
    x = tokens_to_grid(z)
    kw = dict(y=torch.randn(2, COND_DIM), x_cond=tokens_to_grid(torch.randn(2, 4, 64, 8)),
              rel_t=torch.tensor([0.4, -8.0]), ctx_noise=torch.tensor([0.0, 0.05]))
    loss = diffusion.training_losses(model, x, torch.tensor([10, 500]), kw)["loss"].mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_adapter_shapes():
    m = JointAdapter(hidden=32)
    assert m(torch.randn(3, IN_DIM)).shape == (3, OUT_DIM)
