"""Imagined insertion environment: the policy interacts with the world model instead of the simulator.

Batched (B episodes in parallel), one step = one policy decision = 0.4 s:

    action  absolute joint targets (B, 16, 7) from the policy; the first 8 are executed
    joints  the joint adapter predicts the measured joints after each of the 8 commands from the
            current joints and the 1.2 s tracking history (models/adapter.py; Ctrl-World's adapter)
    poses   forward kinematics: commanded hand poses of the targets, measured hand poses of the
            predicted joints
    world   the world model predicts the latent 0.4 s later from the 4 context latents and the
            condition built exactly as in training (data/conditions.py: measured / commanded hand
            pose of each context frame + the commanded hand path)
    reward  read from the predicted latent, no simulator truth: the plug's remaining motion is solved
            from the decoded frame (models/vae.py solve_pose) with the imagined frame's geometry
            (camera pose = imagined hand pose x the hand-eye mount of the start frame)

Episodes start from real recorded states (train split): 4 context latents, joints, tracking history
and geometry of a recorded frame, anywhere from far in front of the board to pushing on it (contact
trials, jams included), so the policy practises what to do when blocked. Episodes last `horizon`
steps (2 s by default, where the world model was checked to be reliable) and are bootstrapped by the
critic after that.

Each prediction is sampled twice (two diffusion seeds); their disagreement on the distance to inserted
is returned as the world model's uncertainty, which the reward penalises (MOPO-style) so the policy is
not rewarded for moves the world model is unsure about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data import index as data_index
from ..data.conditions import N_WAYPOINTS, pose_feature
from ..data.datasets import CONTEXT, HISTORY_STEPS, WM_STEP, WM_STRIDE, port_axes_ok
from ..geometry import grid_pose_error_mm, plug_region_points
from ..models.cdit import create_diffusion, grid_to_tokens, tokens_to_grid
from ..models.policy import ACTION_DIM, CHUNK

START_KINDS = ("insert", "perturbed", "reference", "contact")


class StartPool:
    """Recorded start states: (latent file arrays, image index j >= 6, port axes)."""

    def __init__(self, data_root: str, latent_root: str, split: str = "train", max_files: int | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        entries = [e for e in data_index.load(data_root) if e["split"] == split and e["kind"] in START_KINDS
                   and e["n_images"] > (CONTEXT - 1) * WM_STRIDE + 1 and port_axes_ok(e)]
        if max_files:
            entries = [entries[i] for i in rng.choice(len(entries), size=min(max_files, len(entries)), replace=False)]
        self.files, self.items = [], []
        for e in entries:
            p = Path(latent_root) / Path(e["path"]).relative_to(data_root)
            if not p.exists():
                continue
            with np.load(p) as f:
                d = {k: f[k] for k in f.files}
            fi = len(self.files)
            self.files.append((d, np.asarray(e["R_P"], np.float64), e["kind"]))
            self.items += [(fi, j) for j in range((CONTEXT - 1) * WM_STRIDE, e["n_images"])]
        self.rng = rng

    def sample(self, n: int):
        return [self.items[i] for i in self.rng.choice(len(self.items), size=n, replace=True)]


class ImaginedInsertion:
    def __init__(self, vae, wm, adapter, fk, scale: float, pool: StartPool, device, n_envs: int = 64,
                 horizon: int = 5, sampling_steps: int = 20, success_mm: float = 1.5, fence_mm: float | None = None):
        self.vae, self.wm, self.adapter, self.fk = vae, wm, adapter, fk
        self.scale, self.pool, self.dev = scale, pool, device
        self.B, self.horizon, self.success_mm = n_envs, horizon, success_mm
        # v2: an imagined episode ends ("fenced") when the hand has moved more than fence_mm across the
        # insertion axis since the start (FK, port axes): the world model and the pose readout have never
        # seen the plug beside / beyond the board, where v1's residual found readings it could exploit.
        self.fence_mm = fence_mm
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))
        self.ref = plug_region_points(device)

    # ------------------------------------------------------------------ geometry / readout
    def _geom(self, hand: np.ndarray, mount: np.ndarray, K: np.ndarray, R_P: np.ndarray) -> dict:
        """Imagined frames' geometry: camera pose = hand @ mount (world -> camera = its inverse)."""
        cam = np.linalg.inv(hand @ mount)
        T = lambda x: torch.tensor(x, dtype=torch.float32, device=self.dev)
        return {"K": T(K), "cam_T_world": T(cam), "hand": T(hand), "R_P": T(R_P)}

    @torch.no_grad()
    def distance(self, z: torch.Tensor, geom: dict) -> torch.Tensor:
        """Distance to inserted (mm) solved from latents (B, 64, C), scaled."""
        R, t = self.vae.decode_pose(z / self.scale, geom)
        eye = torch.eye(3, device=z.device).expand(len(z), 3, 3)
        return grid_pose_error_mm(R.float(), t.float(), eye, torch.zeros_like(t).float(), self.ref)

    # ------------------------------------------------------------------ reset
    def reset(self) -> dict:
        items = self.pool.sample(self.B)
        ctx, q, hist, ctx_meas, ctx_cmd, K, R_P, mount, s_now = [], [], [], [], [], [], [], [], []
        for fi, j in items:
            d, rp, _ = self.pool.files[fi]
            steps = d["img_step"].astype(int)
            c = [j - WM_STRIDE * k for k in range(CONTEXT - 1, -1, -1)]
            s0 = int(steps[j])
            ctx.append(d["mean"][c].astype(np.float32) * self.scale)
            q.append(d["qpos"][s0].astype(np.float64))
            h = np.zeros((HISTORY_STEPS, 7))
            lo = max(0, s0 - HISTORY_STEPS)
            if s0 > lo:
                h[HISTORY_STEPS - (s0 - lo):] = d["action"][lo:s0] - d["qpos"][lo:s0, :7]
            hist.append(h)
            ctx_meas.append(d["hand"][steps[c]].astype(np.float64))
            cmd_idx = np.clip(steps[c], 0, len(d["hand_cmd"]) - 1)
            ctx_cmd.append(d["hand_cmd"][cmd_idx].astype(np.float64))
            K.append(d["K"].astype(np.float64))
            R_P.append(rp)
            mount.append(np.linalg.inv(d["hand"][s0].astype(np.float64)) @ np.linalg.inv(d["cam_T_world"][j].astype(np.float64)))
            s_now.append(s0)
        self.ctx = torch.tensor(np.stack(ctx), device=self.dev)
        self.q, self.hist = np.stack(q), np.stack(hist)
        self.ctx_meas, self.ctx_cmd = np.stack(ctx_meas), np.stack(ctx_cmd)
        self.K, self.R_P, self.mount, self.s_now = np.stack(K), np.stack(R_P), np.stack(mount), np.array(s_now)
        self.t = 0
        hand_now = self.ctx_meas[:, -1]
        self.d_now = self.distance(self.ctx[:, -1], self._geom(hand_now, self.mount, self.K, self.R_P)).cpu().numpy()
        self.d_start = self.d_now.copy()
        self.hand_start = hand_now.copy()
        # starts that already read as inserted carry no learning signal: they count as finished
        self.done = self.d_now < self.success_mm + 0.5
        return self.obs()

    def obs(self) -> dict:
        return {"z": self.ctx, "q": torch.tensor(self.q[:, :7], dtype=torch.float32, device=self.dev),
                "distance_mm": torch.tensor(self.d_now, dtype=torch.float32, device=self.dev)}

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, targets: np.ndarray) -> tuple[dict, dict]:
        """targets (B, 16, 7) absolute joint targets. Returns the next observation and step info:
        distance before / after (mm, mean of the two samples), uncertainty (mm), push gap (mm, commanded
        vs predicted measured hand position at the end of the executed chunk), done flags."""
        B = self.B
        # 1. joints after each of the 8 executed commands (adapter)
        x = np.concatenate([self.q[:, :7], self.hist.reshape(B, -1), (targets[:, :WM_STEP] - self.q[:, None, :7]).reshape(B, -1)], 1)
        dq = self.adapter(torch.tensor(x, dtype=torch.float32, device=self.dev)).view(B, WM_STEP, 7).cpu().numpy()
        q_steps = self.q[:, None, :7] + dq                                            # measured after command k
        # 2. forward kinematics (fingers as measured)
        fingers = self.q[:, 7:9]
        cmd_poses = np.stack([self.fk(np.concatenate([targets[b, :WM_STEP + 1], np.repeat(fingers[b][None], WM_STEP + 1, 0)], 1))
                              for b in range(B)])                                     # (B, 9, 4, 4) commands at s .. s+8
        meas_end = np.stack([self.fk(np.concatenate([q_steps[b, -1:], fingers[b][None]], 1))[0] for b in range(B)])
        # 3. condition, as data.conditions.wm_condition: newest context frame's command = this chunk's first
        self.ctx_cmd[:, -1] = cmd_poses[:, 0]
        ys = []
        for b in range(B):
            ref = self.ctx_meas[b, -1]
            ctx_feat = np.stack([pose_feature(self.ctx_meas[b], ref, self.R_P[b]),
                                 pose_feature(self.ctx_cmd[b], ref, self.R_P[b])], 1).reshape(-1)
            s = self.s_now[b]
            off = np.rint(np.linspace(s, s + WM_STEP, N_WAYPOINTS + 1)[1:]).astype(int) - s
            ys.append(np.concatenate([ctx_feat, pose_feature(cmd_poses[b, off], ref, self.R_P[b]).reshape(-1)]))
        y = torch.tensor(np.stack(ys), dtype=torch.float32, device=self.dev)
        # 4. world model, two samples
        C = self.ctx.shape[-1]
        kw = dict(y=y.repeat(2, 1), x_cond=tokens_to_grid(self.ctx).repeat(2, 1, 1, 1, 1),
                  rel_t=torch.full((2 * B,), WM_STEP / 20.0, device=self.dev))
        x = self.sampler.p_sample_loop(self.wm.forward, (2 * B, C, 8, 8), torch.randn(2 * B, C, 8, 8, device=self.dev),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.dev)
        z2 = grid_to_tokens(x)
        geom = self._geom(meas_end, self.mount, self.K, self.R_P)
        geom2 = {k: v.repeat(2, *([1] * (v.dim() - 1))) for k, v in geom.items()}
        d2 = self.distance(z2, geom2).view(2, B).cpu().numpy()
        z_new = z2[:B]
        # 5. advance the state
        d_before = self.d_now.copy()
        self.d_now = np.where(self.done, self.d_now, d2.mean(0))
        new_hist = targets[:, :WM_STEP] - np.concatenate([self.q[:, None, :7], q_steps[:, :-1]], 1)
        self.hist = np.concatenate([self.hist, new_hist], 1)[:, -HISTORY_STEPS:]
        self.q = np.concatenate([q_steps[:, -1], fingers], 1)
        self.ctx = torch.cat([self.ctx[:, 1:], z_new[:, None]], 1)
        self.ctx_meas = np.concatenate([self.ctx_meas[:, 1:], meas_end[:, None]], 1)
        self.ctx_cmd = np.concatenate([self.ctx_cmd[:, 1:], cmd_poses[:, -1:]], 1)   # replaced by the next chunk's first
        self.s_now = self.s_now + WM_STEP
        self.t += 1
        gap = np.linalg.norm(cmd_poses[:, WM_STEP - 1, :3, 3] - meas_end[:, :3, 3], axis=1) * 1000
        rel = np.einsum("bji,bj->bi", self.R_P, meas_end[:, :3, 3] - self.hand_start[:, :3, 3]) * 1000   # port axes, mm
        lateral = np.linalg.norm(rel[:, 1:], axis=1)
        step_lat = np.einsum("bji,bj->bi", self.R_P, meas_end[:, :3, 3] - self.ctx_meas[:, -2, :3, 3])[:, 1:]
        fenced = (lateral > self.fence_mm) & ~self.done if self.fence_mm else np.zeros(B, bool)
        success = (self.d_now < self.success_mm) & ~self.done & ~fenced
        info = {"lateral_travel_mm": lateral, "lateral_step_mm": np.linalg.norm(step_lat, axis=1) * 1000, "fenced": fenced,
                "d_before": d_before, "d_after": self.d_now.copy(), "uncertainty_mm": np.abs(d2[0] - d2[1]),
                "push_gap_mm": gap, "success": success, "was_done": self.done.copy()}
        self.done = self.done | success | fenced
        info["timeout"] = self.t >= self.horizon
        return self.obs(), info
