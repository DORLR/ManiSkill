"""Method side of the closed loop: observation -> estimated state -> field step.

Uses only: wrist RGB-D, camera intrinsics/extrinsics, hand pose (FK), the
trained networks and the ID set. It never receives H_pre or any simulator
object pose (tests/test_no_truth_leak.py checks this module's source).
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from .field import FieldNet, step_from_field
from .geometry import apply_port_motion
from .model import AlignNet
from .perception import port_axes_from_obs, preprocess


class FieldPolicy:
    def __init__(self, run_dir, device="cuda", estimate="hat"):
        self.dev = device
        self.estimate = estimate  # "hat" (point cloud + RGB) or "pc" (point cloud only)
        self.model = AlignNet(pretrained=False).to(device).eval()
        self.model.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location=device))
        self.field = FieldNet().to(device).eval()
        self.field.load_state_dict(torch.load(f"{run_dir}/field.pt", map_location=device))
        self.R_P = None

    def begin_episode(self, obs, rng):
        """Port axes from the first view (board-plane RANSAC + gravity)."""
        self.R_P, _ = port_axes_from_obs(obs, rng)
        return self.R_P is not None

    @torch.no_grad()
    def estimate_state(self, obs, rng):
        crop, pc = preprocess(obs, self.R_P, rng)
        rgb = torch.as_tensor(crop[None], device=self.dev)
        pts = torch.as_tensor(pc[None], device=self.dev, dtype=torch.float32)
        o = self.model(rgb, pts)
        u = o["q_pc" if self.estimate == "pc" else "q_hat"][0].float().cpu().numpy()
        return u, float(torch.sigmoid(o["id_logit"][0]).item())

    @torch.no_grad()
    def field_step(self, u):
        v = self.field(torch.as_tensor(u[None], device=self.dev, dtype=torch.float32))[0].cpu().numpy()
        return v, step_from_field(v, C.STEP_MAX)

    def target_pose(self, obs, du):
        return apply_port_motion(obs.H, du * C.SIGMA, self.R_P)
