"""Imagined insertion: the policy interacts with the world model instead of the simulator.

Batched (B episodes in parallel), one step = one policy decision = 0.4 s:

    action  absolute joint targets (B, 16, 7) from the policy; the first 8 are executed
    joints  the joint adapter predicts the measured joints after each of the 8 commands from the
            current joints and the 1.2 s tracking history (models/adapter.py; Ctrl-World's adapter).
            For checks, the measured joints can be given instead (`q_steps`), to separate the
            adapter's error from the world model's.
    poses   forward kinematics: commanded hand poses of the targets, measured hand poses of the
            (predicted or given) joints
    world   the world model predicts the latent 0.4 s later from the 4 context latents and the
            condition built exactly as in training (data/conditions.py: measured / commanded hand
            pose of each context frame + the commanded hand path)
    readout the plug's remaining motion solved from the predicted latent (models/vae.py solve_pose)
            with the imagined frame's geometry (camera pose = imagined hand pose x the hand-eye mount
            of the start frame); no simulator truth

Start states are either recorded training trajectories (StartPool, for imitation in imagination) or
explicit states built from a simulator rollout (`reset_from`, for checking imagination against the
simulator; analysis/imagination_checks.py).

With `wm_samples=2` each prediction is drawn twice (two diffusion seeds) and their disagreement on
the distance to inserted is returned as the world model's uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import index as data_index
from .data.conditions import N_WAYPOINTS, pose_feature
from .data.datasets import CONTEXT, HISTORY_STEPS, WM_STEP, WM_STRIDE, port_axes_ok
from .geometry import grid_pose_error_mm, plug_region_points, rotation_angle_deg
from .models.cdit import create_diffusion, grid_to_tokens, tokens_to_grid
from .models.policy import ACTION_DIM, CHUNK, LatentFMPolicy

START_KINDS = ("insert", "perturbed", "reference", "contact")


class FMSampler:
    """The latent FM policy for batched use in imagination: scaled latents + joints -> targets."""

    def __init__(self, policy_ckpt: str, device, flow_steps: int = 10):
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)   # our own file
        self.model = LatentFMPolicy(**p["cfg"]["model"])
        self.model.load_state_dict(p["ema"])
        self.model.to(device).eval().requires_grad_(False)
        self.mean, self.std = torch.tensor(p["mean"], device=device), torch.tensor(p["std"], device=device)
        self.scale, self.steps = float(p["scale"]), flow_steps

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """z (B, 4, 64, C) scaled latents, q (B, 7) -> targets relative to q (B, 16, 7)."""
        a = self.model.sample(z, q, steps=self.steps, n=1)[:, 0] * self.std + self.mean
        return a.view(-1, CHUNK, ACTION_DIM)


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

    def state(self, fi: int, j: int, scale: float) -> dict:
        """The start state at image j of recorded trajectory fi (format of ImaginedInsertion.reset_from)."""
        d, rp, _ = self.files[fi]
        steps = d["img_step"].astype(int)
        c = [j - WM_STRIDE * k for k in range(CONTEXT - 1, -1, -1)]
        s0 = int(steps[j])
        h = np.zeros((HISTORY_STEPS, 7))
        lo = max(0, s0 - HISTORY_STEPS)
        if s0 > lo:
            h[HISTORY_STEPS - (s0 - lo):] = d["action"][lo:s0] - d["qpos"][lo:s0, :7]
        cmd_idx = np.clip(steps[c], 0, len(d["hand_cmd"]) - 1)
        return {"z": d["mean"][c].astype(np.float32) * scale, "q": d["qpos"][s0].astype(np.float64), "hist": h,
                "ctx_meas": d["hand"][steps[c]].astype(np.float64), "ctx_cmd": d["hand_cmd"][cmd_idx].astype(np.float64),
                "K": d["K"].astype(np.float64), "R_P": rp, "cam_T_world": d["cam_T_world"][j].astype(np.float64)}


class ImaginedInsertion:
    def __init__(self, vae, wm, adapter, fk, scale: float, device, sampling_steps: int = 20,
                 success_mm: float = 1.5, wm_samples: int = 1, pool: StartPool | None = None):
        self.vae, self.wm, self.adapter, self.fk = vae, wm, adapter, fk
        self.scale, self.pool, self.dev = scale, pool, device
        self.success_mm, self.wm_samples, self.read_batch = success_mm, wm_samples, 32
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))
        self.ref = plug_region_points(device)

    # ------------------------------------------------------------------ geometry / readout
    def _geom(self, hand: np.ndarray) -> dict:
        """Imagined frames' geometry: camera pose = hand @ mount (world -> camera = its inverse)."""
        cam = np.linalg.inv(hand @ self.mount)
        T = lambda x: torch.tensor(x, dtype=torch.float32, device=self.dev)
        return {"K": T(self.K), "cam_T_world": T(cam), "hand": T(hand), "R_P": T(self.R_P)}

    @torch.no_grad()
    def readout(self, z: torch.Tensor, geom: dict) -> dict:
        """Pose read from scaled latents (B, 64, C): distance to inserted (mm, reference-grid error),
        depth left along the insertion axis, offset across it (mm) and angle (deg), as numpy."""
        Rs, ts = [], []
        for a in range(0, len(z), self.read_batch):              # the pose solve needs memory: 32 frames at a time
            sl = slice(a, a + self.read_batch)
            R, t = self.vae.decode_pose(z[sl] / self.scale, {k: v[sl] for k, v in geom.items()})
            Rs.append(R.float()), ts.append(t.float())
        R, t = torch.cat(Rs), torch.cat(ts)
        eye = torch.eye(3, device=z.device).expand(len(z), 3, 3)
        dist = grid_pose_error_mm(R, t, eye, torch.zeros_like(t), self.ref)
        t = t.cpu().numpy() * 1000
        return {"distance_mm": dist.cpu().numpy(), "depth_mm": t[:, 0], "lateral_mm": np.hypot(t[:, 1], t[:, 2]),
                "angle_deg": rotation_angle_deg(R, eye).cpu().numpy()}

    # ------------------------------------------------------------------ reset
    def reset(self, n: int) -> dict:
        """n episodes from random recorded training states."""
        return self.reset_from([self.pool.state(fi, j, self.scale) for fi, j in self.pool.sample(n)])

    def reset_from(self, starts: list[dict]) -> dict:
        """starts: dicts with z (4, 64, C) scaled context latents (oldest first), q (9,) joints now,
        hist (24, 7) command - measured of the last 24 control steps, ctx_meas / ctx_cmd (4, 4, 4)
        measured / commanded hand pose at each context frame's control step, K (3, 3), R_P (3, 3),
        cam_T_world (4, 4) of the newest frame."""
        st = {k: np.stack([s[k] for s in starts]) for k in starts[0]}
        self.B = len(starts)
        self.ctx = torch.tensor(st["z"], dtype=torch.float32, device=self.dev)
        self.q, self.hist = st["q"].astype(np.float64), st["hist"].astype(np.float64)
        self.ctx_meas, self.ctx_cmd = st["ctx_meas"].astype(np.float64), st["ctx_cmd"].astype(np.float64)
        self.K, self.R_P = st["K"], st["R_P"]
        hand_now = self.ctx_meas[:, -1]
        self.mount = np.linalg.inv(hand_now) @ np.linalg.inv(st["cam_T_world"])       # hand -> camera
        self.t = 0
        self.read = self.readout(self.ctx[:, -1], self._geom(hand_now))
        self.d_now = self.read["distance_mm"].copy()
        self.hand_start = hand_now.copy()
        self.done = self.d_now < self.success_mm
        return self.obs()

    def obs(self) -> dict:
        return {"z": self.ctx, "q": torch.tensor(self.q[:, :7], dtype=torch.float32, device=self.dev)}

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, targets: np.ndarray, q_steps: np.ndarray | None = None) -> tuple[dict, dict]:
        """targets (B, 16, 7) absolute joint targets; q_steps (B, 8, 7) the measured joints after each
        executed command if known (checks only), else predicted by the adapter.
        Returns the next observation and step info (readout of the new latent, uncertainty, push gap,
        lateral hand travel since the start, success / done flags, the new latent)."""
        B = self.B
        # 1. joints after each of the 8 executed commands
        if q_steps is None:
            x = np.concatenate([self.q[:, :7], self.hist.reshape(B, -1), (targets[:, :WM_STEP] - self.q[:, None, :7]).reshape(B, -1)], 1)
            dq = self.adapter(torch.tensor(x, dtype=torch.float32, device=self.dev)).view(B, WM_STEP, 7).cpu().numpy()
            q_steps = self.q[:, None, :7] + dq
        # 2. forward kinematics (fingers as measured)
        fingers = self.q[:, 7:9]
        cmd_poses = np.stack([self.fk(np.concatenate([targets[b, :WM_STEP + 1], np.repeat(fingers[b][None], WM_STEP + 1, 0)], 1))
                              for b in range(B)])                                     # (B, 9, 4, 4) commands at s .. s+8
        meas_end = np.stack([self.fk(np.concatenate([q_steps[b, -1:], fingers[b][None]], 1))[0] for b in range(B)])
        # 3. condition, as data.conditions.wm_condition: newest context frame's command = this chunk's first
        self.ctx_cmd[:, -1] = cmd_poses[:, 0]
        off = np.rint(np.linspace(0, WM_STEP, N_WAYPOINTS + 1)[1:]).astype(int)
        ys = []
        for b in range(B):
            ref = self.ctx_meas[b, -1]
            ctx_feat = np.stack([pose_feature(self.ctx_meas[b], ref, self.R_P[b]),
                                 pose_feature(self.ctx_cmd[b], ref, self.R_P[b])], 1).reshape(-1)
            ys.append(np.concatenate([ctx_feat, pose_feature(cmd_poses[b, off], ref, self.R_P[b]).reshape(-1)]))
        y = torch.tensor(np.stack(ys), dtype=torch.float32, device=self.dev)
        # 4. world model, wm_samples draws
        S, C = self.wm_samples, self.ctx.shape[-1]
        kw = dict(y=y.repeat(S, 1), x_cond=tokens_to_grid(self.ctx).repeat(S, 1, 1, 1, 1),
                  rel_t=torch.full((S * B,), WM_STEP / 20.0, device=self.dev))
        x = self.sampler.p_sample_loop(self.wm.forward, (S * B, C, 8, 8), torch.randn(S * B, C, 8, 8, device=self.dev),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.dev)
        zs = grid_to_tokens(x)
        geom = self._geom(meas_end)
        geomS = {k: v.repeat(S, *([1] * (v.dim() - 1))) for k, v in geom.items()}
        rs = self.readout(zs, geomS)
        d_all = rs["distance_mm"].reshape(S, B)
        z_new = zs[:B]
        read = {k: v[:B] for k, v in rs.items()}
        # 5. advance the state
        d_before = self.d_now.copy()
        self.d_now = np.where(self.done, self.d_now, read["distance_mm"])
        new_hist = targets[:, :WM_STEP] - np.concatenate([self.q[:, None, :7], q_steps[:, :-1]], 1)
        self.hist = np.concatenate([self.hist, new_hist], 1)[:, -HISTORY_STEPS:]
        self.q = np.concatenate([q_steps[:, -1], fingers], 1)
        self.ctx = torch.cat([self.ctx[:, 1:], z_new[:, None]], 1)
        self.ctx_meas = np.concatenate([self.ctx_meas[:, 1:], meas_end[:, None]], 1)
        self.ctx_cmd = np.concatenate([self.ctx_cmd[:, 1:], cmd_poses[:, -1:]], 1)   # replaced by the next chunk's first
        self.t += 1
        gap = np.linalg.norm(cmd_poses[:, WM_STEP - 1, :3, 3] - meas_end[:, :3, 3], axis=1) * 1000
        rel = np.einsum("bji,bj->bi", self.R_P, meas_end[:, :3, 3] - self.hand_start[:, :3, 3]) * 1000   # port axes, mm
        success = (self.d_now < self.success_mm) & ~self.done
        info = {"read": read, "z": z_new, "hand": meas_end, "d_before": d_before, "d_after": self.d_now.copy(),
                "uncertainty_mm": np.abs(d_all[0] - d_all[1]) if S > 1 else np.zeros(B),
                "push_gap_mm": gap, "lateral_travel_mm": np.linalg.norm(rel[:, 1:], axis=1),
                "success": success, "was_done": self.done.copy()}
        self.done = self.done | success
        return self.obs(), info
