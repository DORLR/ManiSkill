"""Imagination v4: the policy acts in the history world model; the success classifier judges.

Batched (B episodes), one step = one policy decision = 0.4 s = two history slots of 0.2 s:
    joints   the joint adapter predicts the measured joints after each of the 8 executed commands
             (models/adapter.py; the vision-conditioned one also gets the history and the commanded
             path), unless the real ones are given (checks)
    poses    FK: commanded hand poses of the targets, measured hand poses of the predicted joints
    slots    two new slots: +0.2 s (proprioception only: no image is imagined there) and +0.4 s
             (the world model's predicted latent + proprioception); the oldest two drop out
    world    the history world model predicts the latent 0.4 s ahead from the history (seen latents,
             proprioception of every slot) and the commanded hand path (16 waypoints)
    success  the success classifier on the new history: P(inserted now)

State per episode (numpy unless noted): z (HIST, 64, C) scaled latents (torch), seen (HIST,) bool,
avail (HIST,) bool, meas / cmd (HIST, 4, 4) measured / commanded hand pose of each slot, q (9,) joints
now, hist (24, 7) command - measured of the last 24 control steps, R_P (3, 3).
"""

from __future__ import annotations

import numpy as np
import torch

from .data.conditions import N_WAYPOINTS, pose_feature
from .data.history import HIST
from .models.cdit import grid_to_tokens, tokens_to_grid
from .models.history_cdit import create_diffusion

WM_STEP = 8
HISTORY_STEPS = 24
SLOT_STEPS = 4


def fm_context_slots() -> list[int]:
    """Slots of the FM policy's 4 context frames (0.4 s apart, newest last)."""
    return [HIST - 7, HIST - 5, HIST - 3, HIST - 1]


class HistoryImagination:
    def __init__(self, wm, adapter, fk, success, scale: float, device, sampling_steps: int = 20):
        self.wm, self.adapter, self.fk, self.success = wm, adapter, fk, success
        self.scale, self.dev = scale, device
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))

    # ------------------------------------------------------------------ state
    def reset_from(self, starts: list[dict]):
        st = {k: np.stack([s[k] for s in starts]) for k in starts[0] if k != "z"}
        self.B = len(starts)
        self.z = torch.stack([torch.as_tensor(s["z"], dtype=torch.float32) for s in starts]).to(self.dev)
        self.seen, self.avail = st["seen"].astype(bool), st["avail"].astype(bool)
        self.meas, self.cmd = st["meas"].astype(np.float64), st["cmd"].astype(np.float64)
        self.q, self.hist, self.R_P = st["q"].astype(np.float64), st["hist"].astype(np.float64), st["R_P"].astype(np.float64)
        self.p_success = self.score()
        return self.obs()

    def tensors(self) -> dict:
        """History relative to the current measured hand (the newest slot), as the models take it."""
        B = self.B
        prop = np.zeros((B, HIST, 18), np.float32)
        for b in range(B):
            ref = self.meas[b, -1]
            prop[b] = np.concatenate([pose_feature(self.meas[b], ref, self.R_P[b]), pose_feature(self.cmd[b], ref, self.R_P[b])], -1)
        prop[~self.avail] = 0
        dt = np.broadcast_to((np.arange(HIST) - (HIST - 1)) * SLOT_STEPS / 20.0, (B, HIST)).astype(np.float32).copy()
        dt[~self.avail] = 0
        T = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=self.dev)
        z = self.z * T(self.seen)[..., None, None]
        return {"z_hist": z, "seen": T(self.seen, torch.bool), "proprio": T(prop), "dt": T(dt), "avail": T(self.avail, torch.bool)}

    @torch.no_grad()
    def score(self) -> np.ndarray:
        h = self.tensors()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = self.success(h["z_hist"], h["seen"], h["proprio"], h["dt"], h["avail"]).float()
        return torch.sigmoid(logit).cpu().numpy()

    def obs(self) -> dict:
        """FM policy input: 4 context latents (0.4 s apart; an unseen slot takes the newest seen before it)."""
        idx = []
        for s in fm_context_slots():
            k = s
            while k > 0 and not self.seen[:, k].all():
                k -= 1
            idx.append(k)
        return {"z": self.z[:, idx], "q": torch.tensor(self.q[:, :7], dtype=torch.float32, device=self.dev)}

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, targets: np.ndarray, q_steps: np.ndarray | None = None) -> dict:
        """targets (B, 16, 7) absolute joint targets (first 8 executed); q_steps (B, 8, 7) real joints
        (checks) or None (adapter). Returns the predicted latent, P(success) and the new hand pose."""
        B = self.B
        fingers = self.q[:, 7:9]
        cmd = np.stack([self.fk(np.concatenate([targets[b, :WM_STEP + 1], np.repeat(fingers[b][None], WM_STEP + 1, 0)], 1))
                        for b in range(B)])                                         # commands at steps s .. s+8
        self.cmd[:, -1] = cmd[:, 0]                                                 # the newest slot's command = this chunk's first
        h = self.tensors()
        ref = self.meas[:, -1]
        off = np.rint(np.linspace(0, WM_STEP, N_WAYPOINTS + 1)[1:]).astype(int)
        path = torch.tensor(np.stack([pose_feature(cmd[b, off], ref[b], self.R_P[b]).reshape(-1) for b in range(B)]),
                            dtype=torch.float32, device=self.dev)
        if q_steps is None:
            x = np.concatenate([self.q[:, :7], self.hist.reshape(B, -1), (targets[:, :WM_STEP] - self.q[:, None, :7]).reshape(B, -1)], 1)
            x = torch.tensor(x, dtype=torch.float32, device=self.dev)
            if getattr(self.adapter, "takes_history", False):       # vision-conditioned: the history and the commanded path too
                dq = self.adapter(x, h["z_hist"], h["seen"], h["proprio"], h["dt"], h["avail"], path)
            else:
                dq = self.adapter(x)
            q_steps = self.q[:, None, :7] + dq.view(B, WM_STEP, 7).cpu().numpy().astype(np.float64)
        meas = np.stack([self.fk(np.concatenate([q_steps[b, [SLOT_STEPS - 1, WM_STEP - 1]], np.repeat(fingers[b][None], 2, 0)], 1))
                         for b in range(B)])                                        # measured at s+4, s+8
        C = self.z.shape[-1]
        kw = dict(path=path,
                  rel_t=torch.full((B,), WM_STEP / 20.0, device=self.dev), **h)
        x = self.sampler.p_sample_loop(self.wm.forward, (B, C, 8, 8), torch.randn(B, C, 8, 8, device=self.dev),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.dev)
        z_new = grid_to_tokens(x)
        # advance two slots: +0.2 s (no image), +0.4 s (the prediction)
        self.z = torch.cat([self.z[:, 2:], torch.zeros_like(z_new)[:, None], z_new[:, None]], 1)
        self.seen = np.concatenate([self.seen[:, 2:], np.zeros((B, 1), bool), np.ones((B, 1), bool)], 1)
        self.avail = np.concatenate([self.avail[:, 2:], np.ones((B, 2), bool)], 1)
        self.meas = np.concatenate([self.meas[:, 2:], meas], 1)
        self.cmd = np.concatenate([self.cmd[:, 2:], cmd[:, [SLOT_STEPS, WM_STEP]]], 1)   # +0.4 s: replaced by the next chunk's first
        new_hist = targets[:, :WM_STEP] - np.concatenate([self.q[:, None, :7], q_steps[:, :-1]], 1)
        self.hist = np.concatenate([self.hist, new_hist], 1)[:, -HISTORY_STEPS:]
        self.q = np.concatenate([q_steps[:, -1], fingers], 1)
        self.p_success = self.score()
        return {"z": z_new, "p_success": self.p_success.copy(), "hand": meas[:, -1],
                "push_gap_mm": np.linalg.norm(cmd[:, WM_STEP - 1, :3, 3] - meas[:, -1, :3, 3], axis=1) * 1000}


def start_from_record(lat: list, meas_img: np.ndarray, cmd_img: np.ndarray, qpos: np.ndarray, action: np.ndarray,
                      s_now: int, R_P: np.ndarray) -> dict:
    """Imagination start state from a live record (online actor / checks).
    lat: latents (scaled, (64, C) torch or numpy) of the recorded images so far, one per 0.2 s, the last
    one taken now; meas_img / cmd_img (n, 4, 4) measured / commanded hand pose at each image's step (the
    command of the newest image is replaced by the next chunk's first); qpos (s_now+1, 9), action (s_now, 7)."""
    n = len(lat)
    slots = n - 1 - np.arange(HIST - 1, -1, -1)
    avail = slots >= 0
    sl = np.clip(slots, 0, None)
    C = lat[0].shape[-1]
    z = torch.stack([torch.as_tensor(lat[i], dtype=torch.float32) for i in sl])
    z[torch.from_numpy(~avail)] = 0
    h = np.zeros((HISTORY_STEPS, 7))
    lo = max(0, s_now - HISTORY_STEPS)
    if s_now > lo:
        h[HISTORY_STEPS - (s_now - lo):] = action[lo:s_now] - qpos[lo:s_now, :7]
    assert z.shape[-1] == C
    return {"z": z, "seen": avail.copy(), "avail": avail, "meas": meas_img[sl].astype(np.float64),
            "cmd": cmd_img[sl].astype(np.float64), "q": qpos[s_now].astype(np.float64), "hist": h,
            "R_P": np.asarray(R_P, np.float64)}
