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
  gap       the pin-tip gap from the offset read-out head (gap_mm; attempt320's in-hole rule): for a checkpoint with
            offset_mode "gap" the read-out itself, without the pin length (required for AutoMate, the user's decision 6
            of 2026-11-23); offset_mode "along" (legacy; every checkpoint without the key) minus the pin length
Nothing here reads H*, the insertion axis or the simulator's truth.
"""

from __future__ import annotations

import numpy as np
import torch

from ..tdmpc import demo320 as D
from ..tdmpc import wmdata320 as WD
from ..tdmpc.wm320 import WM320, encoder_settings


# MPPI (the user's decisions of 2026-11-26): TD-MPC2's planner (tdmpc2.py plan(), config.yaml at 332f19e) unchanged in
# its numbers and steps, in an action space of hand moves around the built-in head's proposal:
#   u_t in [-1, 1]^3 (t = 0..H-1): the hand moved by MPPI_MM * u_t along x, y, z of the HAND's frame (FK of the measured
#   joints, as the fixed candidates), as one joint offset dq_t = Wq u_t (Wq (7, 3) from the caller: damped least squares
#   on the FK Jacobian, rotation held, computed once from the measured joints) added to all 8 executed targets of
#   pi(z_t), the head's proposal at the imagined latent z_t. u = 0 is the head itself, so TD-MPC2's mean init of zero
#   is "centred on the head's proposal" (decision 1) and its pi trajectories (num_pi_trajs) are u = 0.
#   MPPI_MM = 3 mm per unit (the expert's largest per-decision hand move, measured 2026-11-26); init std = max_std = 2
#   (decision 2: TD-MPC2's default).
# Value of a sequence = TD-MPC2's _estimate_value (episodic): sum_t gamma^t (1 - term_t) r_t + gamma^H (1 - term_H)
#   Q_avg(z_H, pi(z_H)), term from the done head > 0.5, Q_avg = mean of 2 random Q heads (a new pair per call).
# Steps (per decision): mean = 0 (t0 of the attempt) else the previous decision's mean shifted by one; std = max_std;
#   6 iterations of: 488 Gaussian samples (mean + std * N(0, 1), clamped to [-1, 1]) + 24 pi trajectories -> value ->
#   64 elites -> weights exp(0.5 (v - v_max)) -> new mean / std (clamped to [0.05, 2]); then one elite drawn by
#   TD-MPC2's gumbel_softmax_sample, its first step u_0; outside eval_mode (collection) u_0 + std_0 * N(0, 1); clamp.
MPPI = {"horizon": 3, "iterations": 6, "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24, "min_std": 0.05,
        "max_std": 2.0, "temperature": 0.5, "mm_per_unit": 3.0}


class WMAgent:
    def __init__(self, ckpt: str, device: str = "cuda", horizon: int = 3):
        self.device = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)           # as wm320.load, read once
        if ck.get("labels") != WD.LABELS:
            raise ValueError(f"{ckpt} was trained with labels {ck.get('labels')}, this code has {WD.LABELS}")
        self.offset_mode = str(ck.get("offset_mode", "along"))
        if self.offset_mode not in WD.OFFSET_MODES:
            raise ValueError(f"{ckpt} has unknown offset_mode {self.offset_mode!r}; expected one of {WD.OFFSET_MODES}")
        self.m = WM320(ck["stem"], ck["stats"], ck.get("shaping", 0.0), imagenet=ck.get("imagenet", False),
                       offset_weight=ck.get("offset_weight", 0.0), offset_mode=self.offset_mode, **encoder_settings(ck), **ck["cfg"])
        self.m.load_state_dict(ck["state"])
        self.m.to(self.device).eval()
        self.horizon, self.gamma, self.step = horizon, float(self.m.cfg.gamma), int(ck.get("step", -1))
        self.shaping = float(ck.get("shaping", 0.0))                           # the reward's shaping weight (for true returns)
        if (ck.get("provenance") or {}).get("split") is not None and self.offset_mode != "gap":
            raise ValueError(f"{ckpt} was trained on AutoMate data with offset_mode {self.offset_mode!r}: its gap would "
                             "need the pin length at test time (decision 6 of 2026-11-23); train a new one")
        del ck

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, depth: np.ndarray, q: np.ndarray) -> torch.Tensor:
        """rgb (4, 320, 320, 3) uint8 (already through the JPEG codec), depth (4, 320, 320) uint16 [0.1 mm], q (7,) -> z (1, L)."""
        r = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        d = torch.from_numpy(np.minimum(depth, 32767).astype(np.int16)).to(self.device)
        tok = self.m.enc.tokens(r, d)[None]                                   # (1, 4, n_tokens, token_dim)
        q = np.asarray(q, np.float32)[:D.JOINTS]                              # the arm's 7 joints only (qpos[7:9]: fingers)
        assert q.shape[-1] == D.JOINTS == 7
        return self.m.enc.fuse(tok, torch.from_numpy(q)[None].to(self.device))

    @torch.no_grad()
    def propose(self, z: torch.Tensor) -> np.ndarray:
        """The ACT head's next 16 joint targets minus the measured joints [rad], (16, 7)."""
        return self.m.act_chunk(z)[0].double().cpu().numpy()

    @torch.no_grad()
    def done(self, z: torch.Tensor) -> float:
        return float(torch.sigmoid(self.m._done(z))[0, 0])

    @torch.no_grad()
    def offset_mm(self, z: torch.Tensor) -> np.ndarray | None:
        """The offset read-out head on z, (4,). None without the head.
        - offset_mode == 'gap': (gap_mm, lateral x [mm], lateral y [mm], tilt [deg]) — zero-privilege pin-tip gap.
        - offset_mode == 'along' (legacy): (along_mm, lateral x [mm], lateral y [mm], tilt [deg]) from H*."""
        if self.m._offset is None:
            return None
        return (self.m._offset(z) * self.m.off_scale)[0].double().cpu().numpy()

    @torch.no_grad()
    def gap_mm(self, z: torch.Tensor, pin_length_mm: float | None = None) -> float | None:
        """Estimated distance of the pin tips in front of the board face [mm] (negative: that deep in the hole), from
        the offset read-out head on z (pictures and joints only; no H* at test time). None without the head.
        - offset_mode == 'gap' (zero-privilege): reads float(o[0]) directly; raises ValueError if pin_length_mm is passed.
        - offset_mode == 'along' (legacy backwards-compat): requires pin_length_mm and computes -o[0] - pin_length_mm."""
        if self.offset_mode == "gap":
            if pin_length_mm is not None:
                raise ValueError("offset_mode='gap' is zero-privilege: do not pass pin_length_mm to gap_mm()")
            o = self.offset_mm(z)
            return None if o is None else float(o[0])
        if self.offset_mode == "along":
            if pin_length_mm is None:
                raise ValueError("legacy offset_mode='along' requires pin_length_mm; use an offset_mode='gap' checkpoint for zero-privilege inference")
            o = self.offset_mm(z)
            return None if o is None else float(-o[0] - float(pin_length_mm))
        raise ValueError(f"unknown offset_mode {self.offset_mode!r}")

    @torch.no_grad()
    def values(self, z: torch.Tensor) -> np.ndarray:
        """V of every value head at z, (num_q,)."""
        return self.m.scalar(self.m.V(z))[:, 0, 0].double().cpu().numpy()

    @torch.no_grad()
    def score(self, z: torch.Tensor, deltas: np.ndarray, parts: bool = False):
        """deltas (C, 8, 7): each candidate's 8 executed joint targets minus the measured joints [rad] -> (C, num_q).
        parts=True: (that, G) with G (C,) = the imagined reward sum alone (sum_t gamma^t c_t r_t, no value term)."""
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
        total = (G[None] + (cont * self.gamma ** self.horizon)[None] * V).T.double().cpu().numpy()
        return (total, G.double().cpu().numpy()) if parts else total

    @torch.no_grad()
    def _mppi_value(self, z: torch.Tensor, u: torch.Tensor, Wq: torch.Tensor) -> torch.Tensor:
        """TD-MPC2 _estimate_value for sequences u (N, H, 3) of hand moves around pi (module comment MPPI) -> (N,)."""
        m, N, H = self.m, u.shape[0], u.shape[1]
        zc = z.reshape(1, -1).expand(N, -1)
        G = torch.zeros(N, device=self.device)
        term = torch.zeros(N, device=self.device)
        disc = 1.0
        for t in range(H):
            base = m.act_chunk(zc)[:, :WD.EXECUTE]                                    # (N, 8, 7) rad
            a = m.scale((base + (u[:, t] @ Wq.T)[:, None, :]).reshape(N, WD.ACTION_DIM))
            r = m.scalar(m.reward(zc, a))[:, 0]
            zc = m.next(zc, a)
            G = G + disc * (1 - term) * r
            disc = disc * self.gamma
            term = torch.clamp(term + (torch.sigmoid(m._done(zc))[:, 0] > 0.5).float(), max=1.0)
        q = m.scalar(m.Q(zc, m.policy_action(zc)))[..., 0]                            # (num_q, N)
        i, j = torch.randperm(q.shape[0], device=self.device)[:2]                     # TD-MPC2 Q(..., 'avg')
        return G + disc * (1 - term) * (q[i] + q[j]) / 2

    @torch.no_grad()
    def plan_mppi(self, z: torch.Tensor, Wq: np.ndarray, prev_mean: torch.Tensor | None, eval_mode: bool) -> dict:
        """One MPPI plan (module comment MPPI). Wq (7, 3): joint offset [rad] per unit of u. prev_mean: the previous
        decision's final mean (H, 3) or None at the attempt's first decision (TD-MPC2's t0).
        -> {"u0": (3,) the executed unit move, "dq": (7,) its joint offset [rad], "move_hand_mm": (3,) hand-frame mm,
            "mean": (H, 3) to pass back next decision, "std0": (3,), "value_max": best elite value, "value_pi": value
            of the pi trajectory (u = 0) in the last iteration}"""
        c, H = MPPI, self.horizon
        if H != c["horizon"]:
            raise ValueError(f"MPPI uses TD-MPC2's horizon {c['horizon']}; the agent has {H}")
        dev = self.device
        Wt = torch.as_tensor(np.asarray(Wq, np.float32), device=dev)                  # (7, 3)
        mean = torch.zeros(H, 3, device=dev)
        if prev_mean is not None:
            mean[:-1] = prev_mean[1:]
        std = torch.full((H, 3), c["max_std"], device=dev)
        n_pi, n = c["num_pi_trajs"], c["num_samples"]
        u = torch.zeros(n, H, 3, device=dev)                                          # rows < n_pi: pi (u = 0)
        for _ in range(c["iterations"]):
            u[n_pi:] = (mean[None] + std[None] * torch.randn(n - n_pi, H, 3, device=dev)).clamp(-1, 1)
            value = self._mppi_value(z, u, Wt).nan_to_num(0)
            elite = torch.topk(value, c["num_elites"]).indices
            ev, eu = value[elite], u[elite]                                           # (E,), (E, H, 3)
            score = torch.exp(c["temperature"] * (ev - ev.max()))
            score = score / score.sum()
            mean = (score[:, None, None] * eu).sum(0) / (score.sum() + 1e-9)
            std = ((score[:, None, None] * (eu - mean[None]) ** 2).sum(0) / (score.sum() + 1e-9)).sqrt()
            std = std.clamp(c["min_std"], c["max_std"])
        k = self.m.math.gumbel_softmax_sample(score)                                  # TD-MPC2 common/math.py
        u0 = eu[int(k)][0]
        if not eval_mode:
            u0 = u0 + std[0] * torch.randn(3, device=dev)
        u0 = u0.clamp(-1, 1)
        dq = (Wt @ u0).double().cpu().numpy()
        return {"u0": u0.double().cpu().numpy(), "dq": dq, "move_hand_mm": (u0 * c["mm_per_unit"]).double().cpu().numpy(),
                "mean": mean, "std0": std[0].double().cpu().numpy(), "value_max": float(ev.max()), "value_pi": float(value[0])}


__all__ = ["WMAgent", "MPPI"]
