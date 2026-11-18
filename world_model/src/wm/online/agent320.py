"""The world model's own agent (wm/tdmpc/wm320.py) for the online rounds and their tests: pictures and joints only.

Per decision (0.4 s):
  observe   the 4 pictures of the policy's context (0.4 s apart, through the store's JPEG codec as every training
            picture) and the 7 measured joints -> latent z
  propose   the built-in ACT head's next 16 joint targets relative to the measured joints; 8 are executed
  score     candidates = the proposal and corrected copies of it (built by the caller: the proposal plus a joint
            offset that moves the hand); each candidate is imagined `horizon` decisions ahead in the latent: its own
            action first, then the ACT head's action on every imagined latent; score per value head k =
            sum_t gamma^t c_t r_t + gamma^h c_h V_k(z_h), with c_t the imagined probability that the attempt has not
            ended yet (product of 1 - done) -- a seated state ends the sum, as a terminal state does in training
  done      the done head's probability that the plug is seated now
  value     the value heads' estimates V at the current latent (their mean is what the back-off decision compares)
The probe head of the model (a read-out for the training-time check of imagined hand positions) is not used here.
Nothing here reads H*, the insertion axis or the simulator's truth.
"""

from __future__ import annotations

import numpy as np
import torch

from ..tdmpc import demo320 as D
from ..tdmpc import wmdata320 as WD
from ..tdmpc.wm320 import WM320


class WMAgent:
    def __init__(self, ckpt: str, device: str = "cuda", horizon: int = 3):
        self.device = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)           # as wm320.load, read once
        if ck.get("labels") != WD.LABELS:
            raise ValueError(f"{ckpt} was trained with labels {ck.get('labels')}, this code has {WD.LABELS}")
        self.m = WM320(ck["stem"], ck["stats"], ck.get("shaping", 0.0), **ck["cfg"])
        self.m.load_state_dict(ck["state"])
        self.m.to(self.device).eval()
        self.horizon, self.gamma, self.step = horizon, float(self.m.cfg.gamma), int(ck.get("step", -1))
        del ck

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, depth: np.ndarray, q: np.ndarray) -> torch.Tensor:
        """rgb (4, 320, 320, 3) uint8 (already through the JPEG codec), depth (4, 320, 320) uint16 [0.1 mm], q (7,) -> z (1, L)."""
        r = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        d = torch.from_numpy(np.minimum(depth, 32767).astype(np.int16)).to(self.device)
        tok = self.m.enc.tokens(r, d)[None]                                   # (1, 4, 100, 32)
        return self.m.enc.fuse(tok, torch.from_numpy(np.asarray(q, np.float32)[:D.JOINTS])[None].to(self.device))

    @torch.no_grad()
    def propose(self, z: torch.Tensor) -> np.ndarray:
        """The ACT head's next 16 joint targets minus the measured joints [rad], (16, 7)."""
        return self.m.act_chunk(z)[0].double().cpu().numpy()

    @torch.no_grad()
    def done(self, z: torch.Tensor) -> float:
        return float(torch.sigmoid(self.m._done(z))[0, 0])

    @torch.no_grad()
    def values(self, z: torch.Tensor) -> np.ndarray:
        """V of every value head at z, (num_q,)."""
        return self.m.scalar(self.m.V(z))[:, 0, 0].double().cpu().numpy()

    @torch.no_grad()
    def score(self, z: torch.Tensor, deltas: np.ndarray) -> np.ndarray:
        """deltas (C, 8, 7): each candidate's 8 executed joint targets minus the measured joints [rad] -> (C, num_q)."""
        m, C = self.m, len(deltas)
        a = m.scale(torch.from_numpy(np.asarray(deltas, np.float32).reshape(C, WD.ACTION_DIM)).to(self.device))
        zc = z.expand(C, -1)
        G = torch.zeros(C, device=self.device)
        cont = torch.ones(C, device=self.device)
        for t in range(self.horizon):
            if t > 0:
                a = m.policy_action(zc)
            G = G + cont * self.gamma ** t * m.scalar(m.reward(zc, a))[:, 0]
            zc = m.next(zc, a)
            cont = cont * (1 - torch.sigmoid(m._done(zc))[:, 0])
        V = m.scalar(m.V(zc))[..., 0]                                         # (num_q, C)
        return (G[None] + (cont * self.gamma ** self.horizon)[None] * V).T.double().cpu().numpy()


__all__ = ["WMAgent"]
