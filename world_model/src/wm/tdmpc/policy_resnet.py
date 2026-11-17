"""The flow-matching policy on the world model's ResNet (2026-11-16).

Encoder: the FrameNet of the trained branch model (wm/tdmpc/model.py), frozen; the policy reads its last feature map
(10 x 10 x 32 for the stem-stride-4 model = 100 tokens of 32 numbers per picture, 640 x 640 RGB + depth). Policy:
models/policy.py's LatentFMPolicy unchanged (4 pictures 0.4 s apart + the 7 measured joints -> 16 joint targets
relative to the measured joints, 8 executed).
Training data: the attempts of the branch set that end seated by the FK label (the old policy's own successes, at
640 px). The expert demonstrations exist only at 320 px, which this encoder cannot read; they would have to be
collected again at 640.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..models.policy import ACTION_DIM, CHUNK, LatentFMPolicy
from .model import LatentWM, make_cfg

CONTEXT, STRIDE_IMG = 4, 2          # context pictures 0.4 s apart = every second stored picture
IMAGE_EVERY = 4


def load_wm(ckpt: str, device="cuda") -> LatentWM:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = LatentWM(make_cfg(**ck["cfg"]), ck["stats"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def frame_tokens(wm: LatentWM, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """rgb (N, 3, 640, 640) uint8, depth (N, 640, 640) int16 [0.1 mm] -> (N, tokens, 32): FrameNet up to its feature map."""
    f = wm.frame
    x = torch.cat([rgb.float() / 255.0 - 0.5, (depth.float() * 1e-3).clamp(0, 3).unsqueeze(1)], 1)
    x = f.stem(x.contiguous(memory_format=torch.channels_last))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=f.amp):
        x = f.squeeze(f.trunk(x))
    return x.float().flatten(2).transpose(1, 2)


class TokenData:
    """All encoded branches in memory. One sample: the context pictures' tokens, the joints, the next 16 targets."""

    def __init__(self, files: list[str]):
        tok, self.q, self.a, self.first, self.where = [], [], [], [], []
        off = 0
        for f in files:
            with np.load(f) as z:
                t, qpos, act, img = z["tok"], z["qpos"], z["action"], z["img_step"]
            n = int((img < len(act)).sum())                       # pictures from which at least one target follows
            for i in range(n):
                s = int(img[i])
                a = act[s:s + CHUNK]
                if len(a) < CHUNK:
                    a = np.concatenate([a, np.repeat(a[-1:], CHUNK - len(a), 0)])
                self.q.append(qpos[s, :ACTION_DIM])
                self.a.append((a - qpos[s, :ACTION_DIM]).reshape(-1))
                self.first.append(off)
                self.where.append(off + i)
            tok.append(t)
            off += len(t)
        self.tok = torch.from_numpy(np.concatenate(tok))          # (pictures, tokens, 32) float16
        self.q, self.a = torch.from_numpy(np.asarray(self.q, np.float32)), torch.from_numpy(np.asarray(self.a, np.float32))
        self.first, self.where = torch.tensor(self.first), torch.tensor(self.where)
        self._back = torch.arange(CONTEXT - 1, -1, -1) * STRIDE_IMG

    def __len__(self):
        return len(self.where)

    def batch(self, idx: torch.Tensor, device="cuda"):
        ctx = torch.maximum(self.where[idx][:, None] - self._back[None], self.first[idx][:, None])      # (B, CONTEXT)
        z = self.tok[ctx.reshape(-1)].reshape(len(idx), CONTEXT, *self.tok.shape[1:])
        return z.to(device).float(), self.q[idx].to(device), self.a[idx].to(device)


class ResnetPolicyAgent:
    """Closed loop: reads the recorder's latest full picture (640 x 640) and the measured joints."""

    def __init__(self, wm_ckpt: str, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10):
        self.device = torch.device(device)
        self.wm = load_wm(wm_ckpt, device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self.policy = LatentFMPolicy(**p["model"])
        self.policy.load_state_dict(p["ema"])
        self.policy.to(self.device).eval()
        self.mean, self.std = torch.tensor(p["mean"], device=self.device), torch.tensor(p["std"], device=self.device)
        self.flow_steps, self.context = flow_steps, deque(maxlen=CONTEXT)

    def reset(self, first=None, seed: int = 0) -> bool:
        """`first`, `seed`: accepted and unused (the harness of collect/branch.py passes them)."""
        self.context.clear()
        return True

    @torch.no_grad()
    def act(self, obs, n_samples: int = 1) -> np.ndarray:
        """Joint targets (n_samples, CHUNK, 7), absolute. `obs`: collect.branch.FullRecorder, or its `frame`
        (which carries the full picture as `frame.full`)."""
        if hasattr(obs, "buf"):
            rgb_np, depth_np, qpos = obs.buf["rgb"][-1], obs.buf["depth"][-1], obs.buf["qpos"][-1]
        else:
            (rgb_np, depth_np), qpos = obs.full, obs.qpos
        rgb = torch.from_numpy(rgb_np)[None].to(self.device).permute(0, 3, 1, 2)
        depth = torch.from_numpy(np.minimum(depth_np, 32767).astype(np.int16))[None].to(self.device)
        t = frame_tokens(self.wm, rgb, depth)[0]
        if not self.context:
            self.context.extend([t] * CONTEXT)
        else:
            self.context.append(t)
        q = np.asarray(qpos[:ACTION_DIM], np.float32)
        z = torch.stack(list(self.context))[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.policy.sample(z, torch.from_numpy(q)[None].to(self.device), steps=self.flow_steps, n=n_samples)[0].float()
        a = a * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + q
