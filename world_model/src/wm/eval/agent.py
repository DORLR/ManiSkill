"""Policy agent on the shared latent: frame -> VAE latent -> FM policy -> joint targets.

Uses only real-robot inputs: wrist RGB-D, intrinsics, extrinsics (FK + hand-eye), hand pose
(FK), joint angles, and port axes fitted to the first image of the episode.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..data.datasets import CONTEXT
from ..env.robot import Frame
from ..models.perception import port_axes
from ..models.policy import ACTION_DIM, CHUNK, LatentFMPolicy
from ..models.vae import load_vae

EXECUTE = 8   # control steps executed per decision (one world-model step, 0.4 s)


class LatentPolicyAgent:
    def __init__(self, vae_ckpt: str, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        self.device = torch.device(device)
        self.vae = load_vae(vae_ckpt, self.device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)  # our own file; holds numpy action stats
        self.policy = LatentFMPolicy(**p["cfg"]["model"])
        self.policy.load_state_dict(p["ema"])
        self.policy.to(self.device).eval()
        self.mean = torch.tensor(p["mean"], device=self.device)
        self.std = torch.tensor(p["std"], device=self.device)
        self.scale, self.flow_steps = float(p["scale"]), flow_steps
        self.R_P, self.context = None, deque(maxlen=CONTEXT)

    def reset(self, first: Frame, seed: int = 0) -> bool:
        """Fit the port axes on the first image; False if no board plane is found."""
        self.R_P = port_axes(first.rgb, first.depth, first.K, first.cam_T_world, seed)
        self.context.clear()
        return self.R_P is not None

    @torch.no_grad()
    def encode(self, f: Frame) -> torch.Tensor:
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)[None]
        b = {"rgb": t(f.rgb, torch.uint8), "depth": t(f.depth.astype(np.int32), torch.int32), "K": t(f.K),
             "cam_T_world": t(f.cam_T_world), "hand": t(f.hand), "qpos": t(f.qpos), "R_P": t(self.R_P)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, _ = self.vae.encode(b)
        return mean.float()[0] * self.scale

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        """Joint targets (n_samples, CHUNK, 7), absolute; the caller executes the first EXECUTE."""
        z = self.encode(f)
        if not self.context:
            self.context.extend([z] * CONTEXT)       # repeat the first frame, as in training
        else:
            self.context.append(z)
        zc = torch.stack(list(self.context))[None]
        a = self.policy.sample(zc, steps=self.flow_steps, n=n_samples)[0] * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + f.qpos[:7]


class ImagePolicyAgent:
    """Latent-free baseline: the same encoder architecture as the VAE, trained end to end with the
    policy; same inputs as LatentPolicyAgent (port axes fitted on the first image, 4-frame context)."""

    def __init__(self, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        from ..models.image_policy import ImageFMPolicy
        self.device = torch.device(device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self.policy = ImageFMPolicy(**p["cfg"]["model"])
        self.policy.load_state_dict(p["ema"])
        self.policy.to(self.device).eval()
        self.mean = torch.tensor(p["mean"], device=self.device)
        self.std = torch.tensor(p["std"], device=self.device)
        self.flow_steps, self.R_P, self.context = flow_steps, None, deque(maxlen=CONTEXT)

    def reset(self, first: Frame, seed: int = 0) -> bool:
        self.R_P = port_axes(first.rgb, first.depth, first.K, first.cam_T_world, seed)
        self.context.clear()
        return self.R_P is not None

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        if not self.context:
            self.context.extend([f] * CONTEXT)
        else:
            self.context.append(f)
        fs = list(self.context)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.device)[None]
        obs = {"rgb": t(np.stack([x.rgb for x in fs]), torch.uint8),
               "depth": t(np.stack([x.depth for x in fs]).astype(np.int32), torch.int32),
               "K": t(f.K), "cam_T_world": t(np.stack([x.cam_T_world for x in fs])),
               "hand": t(np.stack([x.hand for x in fs])), "qpos": t(np.stack([x.qpos for x in fs])),
               "R_P": t(self.R_P)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.policy.sample(obs, steps=self.flow_steps, n=n_samples)[0].float()
        a = a * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + f.qpos[:7]
