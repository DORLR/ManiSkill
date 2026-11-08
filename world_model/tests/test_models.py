"""CPU shape and gradient checks (no DINOv2, no simulator)."""

import math

import numpy as np
import torch

from wm.geometry import insertion_motion, plug_points_at_insertion, rot6d_to_matrix, rotation_angle_deg
from wm.models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from wm.models.vae import TokenVAE
from wm.training.losses import insertion_angle_loss_deg, plug_point_loss_mm


def _batch(B=2, H=48, W=48):
    depth = np.full((B, H, W), 1500, np.int32)       # board 150 mm in front of the camera
    depth[:, 30:, 18:30] = 900                        # something nearer: the plug
    K = torch.tensor([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1]]).expand(B, 3, 3)
    eye = torch.eye(4).expand(B, 4, 4)
    R_P = torch.tensor([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]).T.expand(B, 3, 3)   # port x = camera z
    return {"rgb": torch.randint(0, 255, (B, H, W, 3), dtype=torch.uint8), "depth": torch.from_numpy(depth),
            "K": K, "cam_T_world": eye, "hand": eye.clone(), "R_P": R_P, "qpos": torch.zeros(B, 9),
            "h_star": eye.clone()}


def test_vae_forward_backward():
    torch.manual_seed(0)
    m = TokenVAE(tokenizer=dict(d=64, cross_layers=1, query_layers=1, n_plug=16, n_port=32, n_cand=128,
                                use_dino=False), latent_ch=8, dec_d=64, dec_layers=1, n_dec_plug=32, n_dec_port=64)
    out = m(_batch())
    assert out["z"].shape == (2, 64, 8) and out["plug_points"].shape == (2, 32, 3)
    loss = out["plug_points"].sum() + out["z"].pow(2).mean() + out["t"].sum()
    loss.backward()


def test_plug_point_loss_zero_for_true_motion():
    B, N = 2, 20
    p = torch.randn(B, N, 3) * 0.02
    hand = torch.eye(4).expand(B, 4, 4).clone()
    a = 0.1
    h_star = torch.eye(4).expand(B, 4, 4).clone()
    h_star[:, :3, :3] = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    h_star[:, :3, 3] = torch.tensor([0.01, -0.002, 0.003])
    R_P = torch.eye(3).expand(B, 3, 3)
    target = plug_points_at_insertion(p, hand, h_star, R_P)
    mask = torch.ones(B, N, dtype=torch.bool)
    assert plug_point_loss_mm(h_star[:, :3, :3], h_star[:, :3, 3], p, mask, hand, h_star, R_P) < 1e-3
    assert torch.allclose(target, p @ h_star[:, :3, :3].transpose(1, 2) + h_star[:, None, :3, 3], atol=1e-6)


def test_insertion_angle_zero_for_true_motion():
    B = 2
    a = 0.01
    hand = torch.eye(4).expand(B, 4, 4).clone()
    hand[:, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    h_star = hand.clone()
    h_star[:, :3, :3] = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    R_P = rot6d_to_matrix(torch.randn(B, 6))
    R_star, t_star = insertion_motion(hand, h_star, R_P)
    p = torch.randn(B, 10, 3) * 0.02
    assert torch.allclose(plug_points_at_insertion(p, hand, h_star, R_P), p @ R_star.transpose(1, 2) + t_star[:, None], atol=1e-6)
    assert insertion_angle_loss_deg(R_star, hand, h_star, R_P) < 2e-2
    assert abs(float(rotation_angle_deg(R_star, torch.eye(3).expand(B, 3, 3))[0]) - math.degrees(a)) < 1e-2


def test_action_condition_history():
    from wm.data.datasets import HISTORY_STEPS, action_condition
    T = 40
    qpos = np.zeros((T + 1, 9), np.float32)
    action = np.ones((T, 7), np.float32)          # commands 1 rad ahead, joints never move: blocked
    y0 = action_condition(action, qpos, 0, history=True)
    y = action_condition(action, qpos, 30, history=True)
    assert y.shape == (56 + 7 * HISTORY_STEPS,) and np.allclose(action_condition(action, qpos, 30, False), y[:56])
    assert np.allclose(y0[56:], 0) and np.allclose(y[56:], 1)


def test_rot6d_is_rotation():
    R = rot6d_to_matrix(torch.randn(5, 6))
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(5), atol=1e-5)


def test_cdit_diffusion_step():
    torch.manual_seed(0)
    z = torch.randn(2, 64, 8)
    assert torch.equal(grid_to_tokens(tokens_to_grid(z)), z)
    model = TokenCDiT(action_dim=56, latent_ch=8, context_size=4, size="S")
    diffusion = create_diffusion(timestep_respacing="")
    x, x_cond = tokens_to_grid(z), tokens_to_grid(torch.randn(2, 4, 64, 8))
    t = torch.randint(0, diffusion.num_timesteps, (2,))
    loss = diffusion.training_losses(model, x, t, dict(y=torch.randn(2, 56), x_cond=x_cond,
                                                      rel_t=torch.ones(2)))["loss"].mean()
    loss.backward()
    assert torch.isfinite(loss)
