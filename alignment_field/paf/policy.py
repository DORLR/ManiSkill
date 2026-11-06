"""Method side of the closed loop: observation -> estimated state -> field step.

Uses only: wrist RGB-D, camera intrinsics/extrinsics, hand pose (FK), the
trained networks and the ID set. It never receives H_pre or any simulator
object pose (tests/test_no_truth_leak.py checks this module's source).
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import noise as N
from .field import FieldNet, step_from_field
from .geometry import apply_port_motion
from .geometry import port_axes
from .model import AlignNet, RGBOnlyNet
from .relnet import RelNet
from .perception import port_axes_from_obs, preprocess, rel_view, rgb_crop


class FieldPolicy:
    def __init__(self, run_dir, device="cuda", estimate="hat", share_from=None, depth_noise=None, noise_seed=0):
        self.dev = device
        self.depth_noise = depth_noise          # None or a level of paf.noise (sensor model)
        self.noise_rng = np.random.default_rng(noise_seed)
        # "hat" (point cloud + RGB), "pc" (point cloud only) or "rgb" (RGB only, no depth)
        self.estimate = estimate
        if share_from is not None:  # same weights, one copy in GPU memory
            self.model, self.field = share_from.model, share_from.field
        else:
            if estimate == "rel":
                self.model = RelNet().to(device).eval()
            else:
                net = RGBOnlyNet if estimate == "rgb" else AlignNet
                self.model = net(pretrained=False).to(device).eval()
            self.model.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location=device))
            self.field = FieldNet().to(device).eval()
            self.field.load_state_dict(torch.load(f"{run_dir}/field.pt", map_location=device))
        self.R_P = None

    def begin_episode(self, obs, rng):
        """Port axes from the first view: board-plane RANSAC + gravity, or for
        the RGB-only variant the predicted insertion axis (hand frame) + gravity."""
        obs = N.apply(obs, self.noise_rng, self.depth_noise)
        if self.estimate == "rgb":
            with torch.no_grad():
                o = self.model(torch.as_tensor(rgb_crop(obs)[None], device=self.dev))
            axis_hand = o["axis_hand"][0].float().cpu().numpy()
            self.R_P = port_axes(obs.H[:3, :3] @ axis_hand)
            return True
        self.R_P, _ = port_axes_from_obs(obs, rng)
        return self.R_P is not None

    @torch.no_grad()
    def estimate_state(self, obs, rng):
        obs = N.apply(obs, self.noise_rng, self.depth_noise)
        if self.estimate == "rgb":
            o = self.model(torch.as_tensor(rgb_crop(obs)[None], device=self.dev))
        elif self.estimate == "rel":
            t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a)[None], device=self.dev, dtype=dt)
            rgb, depth, K = rel_view(obs)
            o = self.model(t(rgb, torch.uint8), t(depth, torch.int16), t(K),
                           t(obs.cam_T_world), t(obs.H), t(self.R_P))
        else:
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

    def point_step(self, u, u_target):
        """Control baseline: straight to one fixed relative state (no field)."""
        v = np.asarray(u_target, float) - u
        return v, step_from_field(v, C.STEP_MAX)

    def target_pose(self, obs, du):
        return apply_port_motion(obs.H, du * C.SIGMA, self.R_P)
