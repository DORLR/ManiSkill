"""FM policy guided by the world model (sampling-based selection, no extra training).

Every 0.4 s: the FM policy proposes N action chunks (16 joint targets = 2 world-model steps);
the CDiT world model rolls each candidate forward 2 steps in latent space; the VAE decoder reads,
from the predicted latent, the rigid motion the plug still needs to be fully inserted; the cost
is the mean distance that motion moves the plug points seen now (mm, same quantity as the
plug-point loss, shape-independent: zero when inserted). The cheapest candidate is executed.
Only real-robot inputs are used (the same as the plain agent).
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.datasets import CONTEXT
from ..env.robot import Frame
from ..geometry import apply_rigid
from ..models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from ..models.policy import ACTION_DIM, CHUNK
from .agent import EXECUTE, LatentPolicyAgent

WM_STEP = 8   # control steps per world-model step


class WorldModelGuidedAgent(LatentPolicyAgent):
    def __init__(self, vae_ckpt: str, policy_ckpt: str, wm_ckpt: str, n_candidates: int = 16,
                 sampling_steps: int = 20, device: str = "cuda"):
        super().__init__(vae_ckpt, policy_ckpt, device)
        ck = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if cfg.get("history"):
            raise NotImplementedError("this agent feeds only the next 8 targets; a world model trained with "
                                      "tracking history needs per-control-step joints from the rollout loop")
        self.wm = TokenCDiT(action_dim=cfg["action_dim"], latent_ch=self.vae.latent_ch, context_size=CONTEXT,
                            size=cfg["size"])
        self.wm.load_state_dict(ck["ema"])
        self.wm.to(self.device).eval()
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))
        self.n = n_candidates
        self.plug_points = None
        self.debug = {}

    @torch.no_grad()
    def encode(self, f: Frame) -> torch.Tensor:
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.device)[None]
        b = {"rgb": t(f.rgb, torch.uint8), "depth": t(f.depth.astype(np.int32), torch.int32), "K": t(f.K),
             "cam_T_world": t(f.cam_T_world), "hand": t(f.hand), "qpos": t(f.qpos), "R_P": t(self.R_P)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, aux = self.vae.encode(b)
        m = aux["plug_mask"][0]
        if m.any():
            self.plug_points = aux["plug_xyz"][0][m].float()      # (P, 3) hand-centred port axes
        return mean.float()[0] * self.scale

    @torch.no_grad()
    def remaining_mm(self, z: torch.Tensor) -> torch.Tensor:
        """Mean plug-point displacement (mm) still needed to be inserted, read from latents z (B, 64, C)."""
        d = self.vae.decode(z / self.scale)
        p = self.plug_points[None].expand(z.shape[0], -1, -1)
        return (apply_rigid(p, d["R"].float(), d["t"].float()) - p).norm(dim=-1).mean(1) * 1000

    @torch.no_grad()
    def _predict(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """context (B, 4, 64, C) scaled latents, actions (B, 8, 7) relative targets -> next latent (B, 64, C)."""
        B = context.shape[0]
        kw = dict(y=actions.reshape(B, -1), x_cond=tokens_to_grid(context), rel_t=torch.ones(B, device=self.device))
        x = self.sampler.p_sample_loop(self.wm.forward, (B, self.vae.latent_ch, 8, 8),
                                       torch.randn(B, self.vae.latent_ch, 8, 8, device=self.device),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.device)
        return grid_to_tokens(x)

    @torch.no_grad()
    def act(self, f: Frame, n_samples: int = 1) -> np.ndarray:
        z = self.encode(f)
        if not self.context:
            self.context.extend([z] * CONTEXT)
        else:
            self.context.append(z)
        ctx = torch.stack(list(self.context))[None]                                   # (1, 4, 64, C)
        a = self.policy.sample(ctx, steps=self.flow_steps, n=self.n)[0] * self.std + self.mean
        cand = a.view(self.n, CHUNK, ACTION_DIM)                                        # relative to q now
        q_now = torch.as_tensor(f.qpos[:7], dtype=torch.float32, device=self.device)
        c = ctx.expand(self.n, -1, -1, -1)
        z1 = self._predict(c, cand[:, :WM_STEP])
        q1 = self.vae.decode(z1 / self.scale)["joints"].float()                        # predicted joints after step 1
        c2 = torch.cat([c[:, 1:], z1[:, None]], 1)
        z2 = self._predict(c2, cand[:, WM_STEP:2 * WM_STEP] + q_now - q1[:, None])     # targets relative to q1
        cost = self.remaining_mm(z2) if self.plug_points is not None else torch.zeros(self.n, device=self.device)
        best = int(cost.argmin())
        self.debug = {"cost_mm": cost.cpu().numpy(), "chosen": best, "z_now": z, "z_pred": z2[best],
                      "now_mm": float(self.remaining_mm(z[None])[0]) if self.plug_points is not None else float("nan")}
        return (cand[best:best + 1] + q_now).cpu().numpy()
