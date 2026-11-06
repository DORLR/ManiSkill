"""Method side of the closed loop: observation -> estimated state -> field step.

Uses only: wrist RGB-D, camera intrinsics/extrinsics, hand pose (FK), the
trained networks and the ID set. It never receives H_pre or any simulator
object pose (tests/test_no_truth_leak.py checks this module's source).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import noise as N
from .field import FieldNet, step_from_field
from scipy.spatial.transform import Rotation

from .geometry import apply_port_motion, relative_state
from .geometry import port_axes
from .model import AlignNet, RGBOnlyNet
from .relnet import RelNet
from .perception import port_axes_from_obs, preprocess, rel_view, rgb_crop


class FieldPolicy:
    def __init__(self, run_dir, device="cuda", estimate="hat", share_from=None, depth_noise=None, noise_seed=0,
                 fuse=1):
        self.dev = device
        # fuse > 1: each estimate implies an estimate of the (fixed) pre-insertion pose given the
        # current hand pose (FK); the median of the last `fuse` implied poses gives the state.
        self.fuse = fuse
        self.history = []
        self.depth_noise = depth_noise          # None or a level of paf.noise (sensor model)
        self.noise_rng = np.random.default_rng(noise_seed)
        # "hat" (point cloud + RGB), "pc" (point cloud only) or "rgb" (RGB only, no depth)
        self.estimate = estimate
        if share_from is not None:  # same weights, one copy in GPU memory
            self.model, self.field = share_from.model, share_from.field
        else:
            if estimate == "rel":
                kw_file = Path(f"{run_dir}/relnet_kw.json")
                kw = json.loads(kw_file.read_text()) if kw_file.exists() else {}
                self.model = RelNet(**kw).to(device).eval()
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
        p_id = float(torch.sigmoid(o["id_logit"][0]).item())
        if self.fuse > 1:
            u = self._fused(obs.H, u)
        return u, p_id

    def reset_history(self):
        self.history = []

    def _fused(self, H, u):
        """Median over recent frames of the pre-insertion pose implied by (H, u); FK only."""
        q = u * C.SIGMA
        R_P = self.R_P
        p_est = H[:3, 3] - R_P @ q[:3]
        R_est = R_P @ Rotation.from_rotvec(q[3:]).as_matrix().T @ R_P.T @ H[:3, :3]
        self.history = (self.history + [(p_est, R_est)])[-self.fuse:]
        p_med = np.median(np.array([h[0] for h in self.history]), 0)
        R_ref = self.history[-1][1]
        dv = np.median(np.array([Rotation.from_matrix(h[1] @ R_ref.T).as_rotvec() for h in self.history]), 0)
        T_est = np.eye(4)
        T_est[:3, :3] = Rotation.from_rotvec(dv).as_matrix() @ R_ref
        T_est[:3, 3] = p_med
        return relative_state(H, T_est, R_P) / C.SIGMA

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
