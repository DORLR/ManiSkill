"""Policy agents for closed-loop evaluation (v3).

LatentPolicyAgent  frame -> autoencoder latent -> FM policy (4-frame latent context + joints)
ImagePolicyAgent   latent-free baseline: the same 4 frames as raw inputs + joints

Both use only real-robot inputs: wrist RGB-D, intrinsics, camera pose (FK + hand-eye mount), hand
pose (FK), joint angles, and the port axes fitted to the first image of the episode. Context frames
are 0.4 s apart (one decision each); early in an episode the first frame is repeated, as in training.
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

EXECUTE = 8   # control steps executed per decision (0.4 s), then the policy replans


def _t(x, dev, dt=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)[None]


def frame_batch(f: Frame, R_P: np.ndarray, dev) -> dict:
    """One frame as the encoder's input batch (B = 1)."""
    return {"rgb": _t(f.rgb, dev, torch.uint8), "depth": _t(f.depth.astype(np.int32), dev, torch.int32),
            "K": _t(f.K, dev), "cam_T_world": _t(f.cam_T_world, dev), "hand": _t(f.hand, dev), "R_P": _t(R_P, dev)}


class _Agent:
    def reset(self, first: Frame, seed: int = 0) -> bool:
        """Fit the port axes on the first image; False if no board plane is found."""
        self.R_P = port_axes(first.rgb, first.depth, first.K, first.cam_T_world, seed)
        self.context.clear()
        return self.R_P is not None

    def _push(self, item):
        if not self.context:
            self.context.extend([item] * CONTEXT)       # repeat the first frame, as in training
        else:
            self.context.append(item)


class LatentPolicyAgent(_Agent):
    def __init__(self, vae_ckpt: str, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        self.device = torch.device(device)
        self.vae = load_vae(vae_ckpt, self.device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)  # our own file; holds numpy stats
        self.policy = LatentFMPolicy(**p["cfg"]["model"])
        self.policy.load_state_dict(p["ema"])
        self.policy.to(self.device).eval()
        self.mean, self.std = torch.tensor(p["mean"], device=self.device), torch.tensor(p["std"], device=self.device)
        self.scale, self.flow_steps = float(p["scale"]), flow_steps
        self.R_P, self.context = None, deque(maxlen=CONTEXT)

    @torch.no_grad()
    def encode(self, f: Frame) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = self.vae.encode(frame_batch(f, self.R_P, self.device))
        return z.float()[0] * self.scale

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        """Joint targets (n_samples, CHUNK, 7), absolute; the caller executes the first EXECUTE."""
        self._push(self.encode(f))
        z = torch.stack(list(self.context))[None]
        q = _t(f.qpos[:7], self.device)
        a = self.policy.sample(z, q, steps=self.flow_steps, n=n_samples)[0] * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + f.qpos[:7]


class ImagePolicyAgent(_Agent):
    def __init__(self, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        from ..models.baseline_policy import ImageFMPolicy
        self.device = torch.device(device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self.policy = ImageFMPolicy(**p["cfg"]["model"])
        self.policy.load_state_dict(p["ema"])
        self.policy.to(self.device).eval()
        self.mean, self.std = torch.tensor(p["mean"], device=self.device), torch.tensor(p["std"], device=self.device)
        self.flow_steps, self.R_P, self.context = flow_steps, None, deque(maxlen=CONTEXT)

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        self._push(f)
        fs = list(self.context)
        d = self.device
        obs = {"rgb": _t(np.stack([x.rgb for x in fs]), d, torch.uint8),
               "depth": _t(np.stack([x.depth for x in fs]).astype(np.int32), d, torch.int32),
               "K": _t(f.K, d), "cam_T_world": _t(np.stack([x.cam_T_world for x in fs]), d),
               "hand": _t(np.stack([x.hand for x in fs]), d), "R_P": _t(self.R_P, d), "q": _t(f.qpos[:7], d)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.policy.sample(obs, steps=self.flow_steps, n=n_samples)[0].float()
        a = a * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + f.qpos[:7]


class ResidualAgent(LatentPolicyAgent):
    """Latent FM + the residual trained in imagination (wm.rl.residual): the FM proposes 16 targets, the
    residual policy's mean correction (deterministic at test time) is added to the 8 executed ones."""

    def __init__(self, vae_ckpt: str, policy_ckpt: str, residual_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        super().__init__(vae_ckpt, policy_ckpt, device, flow_steps)
        from ..rl.residual import ResidualActorCritic
        ck = torch.load(residual_ckpt, map_location="cpu", weights_only=False)
        self.residual = ResidualActorCritic()
        self.residual.load_state_dict(ck["model"])
        self.residual.to(self.device).eval()
        self.bound = ck["cfg"].get("residual_rad", 0.004)

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        from ..rl.residual import ResidualActorCritic
        self._push(self.encode(f))
        z = torch.stack(list(self.context))[None]
        q = _t(f.qpos[:7], self.device)
        base = (self.policy.sample(z, q, steps=self.flow_steps, n=1)[0] * self.std + self.mean).view(1, CHUNK, ACTION_DIM)
        a = self.residual.actor(ResidualActorCritic.features(z, q, base)).clamp(-1, 1)
        base[:, :EXECUTE] += (a * self.bound).view(1, EXECUTE, ACTION_DIM)
        return base.cpu().numpy() + f.qpos[:7]
