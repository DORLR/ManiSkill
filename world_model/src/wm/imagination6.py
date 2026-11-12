"""Imagination v6: the world model alone advances the scene, 0.4 s per step (8 commands, two history slots).

Everything is in hand poses, as the model's inputs are: the commands are hand poses (FK of joint targets, or a script's
Cartesian commands), and the model itself says where the hand really ends up (its extra output: measured minus commanded
pose features at +0.2 s and +0.4 s), so there is no joint adapter and no joint angle in the loop.

State per episode: z (HIST, C, H, W) feature maps (torch), seen / avail (HIST,), meas / cmd (HIST, 4, 4) measured and
commanded hand pose of every slot, R_P (3, 3).
    step(cmd9)   cmd9 (B, 9, 4, 4): the commanded hand poses at control steps s .. s+8 (index k = the command executed at
                 step s+k; the newest slot's command becomes cmd9[:, 0]). The model samples the features 0.4 s ahead and
                 gives the hand at +0.2 s / +0.4 s relative to the commands cmd9[:, 3] / cmd9[:, 7] that led there, and
                 "inserted" at +0.4 s. Two slots are appended: +0.2 s (proprioception only), +0.4 s (features too).
"""

from __future__ import annotations

import numpy as np
import torch

from .data.conditions import N_WAYPOINTS, pose_feature
from .data.frame_history import RES_SCALE
from .data.history import HIST
from .models.history_cdit import create_diffusion

WM_STEP, SLOT_STEPS = 8, 4


def pose_from_feature(f: np.ndarray, T_ref: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """Inverse of conditions.pose_feature for one pose: (9,) relative to T_ref in port axes -> 4x4, world."""
    a, b = f[3:6], f[6:9]
    a = a / np.linalg.norm(a)
    b = b - (b @ a) * a
    b = b / np.linalg.norm(b)
    R_rel = np.stack([a, b, np.cross(a, b)], 1)
    T = np.eye(4)
    T[:3, :3] = R_P @ R_rel @ R_P.T @ T_ref[:3, :3]
    T[:3, 3] = T_ref[:3, 3] + R_P @ (f[:3] / 100.0)
    return T


class FeatureImagination:
    def __init__(self, wm, device, sampling_steps: int = 20):
        self.wm, self.dev = wm, device
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))

    def reset(self, z: torch.Tensor, seen, avail, meas, cmd, R_P):
        self.z = z.to(self.dev).float()
        self.seen, self.avail = np.asarray(seen, bool).copy(), np.asarray(avail, bool).copy()
        self.meas, self.cmd, self.R_P = np.asarray(meas, np.float64).copy(), np.asarray(cmd, np.float64).copy(), np.asarray(R_P, np.float64)
        self.B = len(self.seen)

    def tensors(self) -> dict:
        B = self.B
        prop = np.zeros((B, HIST, 18), np.float32)
        for b in range(B):
            ref = self.meas[b, -1]
            prop[b] = np.concatenate([pose_feature(self.meas[b], ref, self.R_P[b]), pose_feature(self.cmd[b], ref, self.R_P[b])], -1)
        prop[~self.avail] = 0
        dt = np.broadcast_to((np.arange(HIST) - (HIST - 1)) * SLOT_STEPS / 20.0, (B, HIST)).astype(np.float32).copy()
        dt[~self.avail] = 0
        T = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.dev)
        seen = T(self.seen, torch.bool)
        return {"z_hist": self.z * seen[..., None, None, None], "seen": seen, "proprio": T(prop), "dt": T(dt), "avail": T(self.avail, torch.bool)}

    @torch.no_grad()
    def step(self, cmd9: np.ndarray, meas2: np.ndarray | None = None) -> dict:
        """`meas2` (B, 2, 4, 4): the real measured hand at +0.2 s / +0.4 s (checks: the model's own is not used)."""
        B = self.B
        self.cmd[:, -1] = cmd9[:, 0]
        h = self.tensors()
        ref = self.meas[:, -1]
        off = np.rint(np.linspace(0, WM_STEP, N_WAYPOINTS + 1)[1:]).astype(int)
        path = torch.tensor(np.stack([pose_feature(cmd9[b, off], ref[b], self.R_P[b]).reshape(-1) for b in range(B)]),
                            dtype=torch.float32, device=self.dev)
        kw = dict(path=path, rel_t=torch.full((B,), WM_STEP / 20.0, device=self.dev), **h)
        shape = self.z.shape[0:1] + self.z.shape[2:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.sampler.p_sample_loop(self.wm.forward, shape, torch.randn(shape, device=self.dev), clip_denoised=False,
                                           model_kwargs=kw, progress=False, device=self.dev)
        aux = self.wm.aux.float().cpu().numpy()
        res = aux[:, :18].reshape(B, 2, 9) / RES_SCALE
        own = np.stack([np.stack([pose_from_feature(pose_feature(cmd9[b, k], ref[b], self.R_P[b]) + res[b, i], ref[b], self.R_P[b])
                                  for i, k in enumerate((SLOT_STEPS - 1, WM_STEP - 1))]) for b in range(B)])
        meas = own if meas2 is None else np.asarray(meas2, np.float64)
        x = x.float()
        self.z = torch.cat([self.z[:, 2:], torch.zeros_like(x)[:, None], x[:, None]], 1)
        self.seen = np.concatenate([self.seen[:, 2:], np.zeros((B, 1), bool), np.ones((B, 1), bool)], 1)
        self.avail = np.concatenate([self.avail[:, 2:], np.ones((B, 2), bool)], 1)
        self.meas = np.concatenate([self.meas[:, 2:], meas], 1)
        self.cmd = np.concatenate([self.cmd[:, 2:], cmd9[:, [SLOT_STEPS, WM_STEP]]], 1)
        return {"features": x, "hand": meas[:, -1], "hand_own": own[:, -1], "p_inserted": 1 / (1 + np.exp(-aux[:, 18]))}
