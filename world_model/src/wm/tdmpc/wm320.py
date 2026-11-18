"""The 320 px world model of the insertion project (2026-11-17, the design agreed with the researcher).

Trained once and then kept fixed; any external policy is used with it through a small "policy stand-in" on its
latent (see stand-in training, separate). Inside:
  encoder   one picture (RGB + depth, 320 x 320) -> 100 tokens (policy320.PictureTokens, ResNet-18, group norm, from
            scratch); a transformer over the tokens of the 4 pictures of an observation (0.4 s apart, exactly the
            policy's context) and a token of the 7 joints -> latent z (512, TD-MPC2's SimNorm)
  heads     TD-MPC2's own layers: dynamics z' = d(z, a); reward r(z, a); Q x5 (z, a); V x5 (z) (two-hot bins); plus
            done(z) = "seated now" and a built-in ACT head z -> the next 16 joint targets (8 executed)
  action    a = the 8 executed joint targets minus the measured joints (56 numbers, asinh-scaled)
Built-in ACT head: trained by imitation of the successful expert demonstrations (the old policy's samples: every
picture, independently drawn); its loss also trains the encoder (the expert's aligning motion needs the direction of
the misalignment); it gives the next action of the value targets; one forward pass per action, so imagination is
cheap. Values are bootstrapped as in TD-MPC2 (r + gamma * min of two target heads at the next latent and the
built-in head's next action), except during a warm-up where the data's own next action is used; the last decision
of a scripted push that ended at a jam takes the jam's absorbing value as its continuation (wmdata320 'boot').
Differences from TD-MPC2, all on purpose: picture encoder (ResNet + transformer over 4 pictures); a state value V
next to Q; a done head; the next action of the targets comes from an imitation head instead of a Q-maximising prior
(the value then means "continue as the expert would", no optimism of a maximising prior); value targets clamped to the
possible range of returns; a probe on the latent (no gradient into it) that reads the hand's offset from H*, for
diagnostics only (imagined hand motion against the real one, e.g. through a jammed contact).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import demo320 as D
from . import use_tdmpc2
from . import wmdata320 as WD
from .model import make_cfg
from .policy320 import TOKEN_DIM, TOKENS, PictureTokens, _amp

CHUNK = D.CHUNK
PHI_MAX = 150.0                        # largest |shaping potential| [mm + deg] of a start (gap <= 18 + pin 20 + lateral 14 + 6 deg x)


class Encoder(nn.Module):
    def __init__(self, stem: str, cfg, d: int = 256, layers: int = 4, heads: int = 8):
        super().__init__()
        from common import layers as L
        self.pic = PictureTokens(stem)
        self.inp = nn.Linear(TOKEN_DIM, d)
        self.token_pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(D.CONTEXT, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.joint_in = nn.Sequential(nn.Linear(D.JOINTS, d), nn.SiLU(), nn.Linear(d, d))
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), layers,
                                        enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.latent_dim), L.SimNorm(cfg))

    def tokens(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (N, 320, 320, 3) uint8, depth (N, 320, 320) -> (N, 100, 32)."""
        return self.pic(rgb.permute(0, 3, 1, 2), depth)

    def fuse(self, tok: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """tok (B, 4, 100, 32) of the pictures [t-1.2 s, ..., t], q (B, 7) -> z (B, latent)."""
        B = tok.shape[0]
        h = self.inp(tok) + self.token_pos[None, None] + self.frame_pos[None, :, None]
        with torch.autocast(q.device.type, enabled=False):
            j = self.joint_in(q.float())
        h = torch.cat([self.cls.expand(B, -1, -1), j[:, None], h.flatten(1, 2)], 1)
        with _amp(h):
            h = self.tf(h)[:, 0]
        return self.out(h.float())


class WM320(nn.Module):
    def __init__(self, stem: str, stats: dict, shaping: float = 0.0, **cfg_over):
        super().__init__()
        self.ret_max = 20.0 + 10.0 + shaping * PHI_MAX                    # value targets lie within +-ret_max
        self.shaping = shaping
        use_tdmpc2()
        from common import init, layers as L, math as M
        cfg = make_cfg(**cfg_over)
        self.cfg, self.math = cfg, M
        Ld, A, Md = cfg.latent_dim, WD.ACTION_DIM, cfg.mlp_dim
        self.enc = Encoder(stem, cfg)
        self._dynamics = L.mlp(Ld + A, 2 * [Md], Ld, act=L.SimNorm(cfg))
        self._reward = L.mlp(Ld + A, 2 * [Md], cfg.num_bins)
        self._Qs = L.Ensemble([L.mlp(Ld + A, 2 * [Md], cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self._Vs = L.Ensemble([L.mlp(Ld, 2 * [Md], cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self._done = L.mlp(Ld, [256], 1)
        self._act = L.mlp(Ld, 3 * [512], CHUNK * D.JOINTS)
        self._probe = nn.Linear(Ld, 4)
        for m in (self._dynamics, self._reward, self._Qs, self._Vs, self._done, self._act):
            m.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"], self._Vs.params["2", "weight"]])
        from copy import deepcopy
        self._Qt = deepcopy(self._Qs).requires_grad_(False)
        self._Qt.train(False)
        for k, v in stats.items():
            self.register_buffer(k, torch.tensor(np.asarray(v), dtype=torch.float32))

    def train(self, mode: bool = True):
        super().train(mode)
        self._Qt.train(False)                       # the target network never uses dropout (as TD-MPC2)
        return self

    # ---- parts
    def next(self, z, a):
        return self._dynamics(torch.cat([z, a], -1))

    def reward(self, z, a):
        return self._reward(torch.cat([z, a], -1))

    def Q(self, z, a):
        return self._Qs(torch.cat([z, a], -1))

    def V(self, z):
        return self._Vs(z)

    def scalar(self, logits):
        return self.math.two_hot_inv(logits, self.cfg)

    def phi_hat(self, z):
        """(B,) the shaping potential read by the probe: -(|along| + |lateral| [mm] + angle [deg]) (wmdata320's phi)."""
        p = self._probe(z) * 10.0
        return -(p[..., 0].abs() + torch.hypot(p[..., 1], p[..., 2]) + p[..., 3].clamp(min=0))

    def unshaped_value(self, z):
        """(B,) mean over the value heads of V(z) with the shaping term removed: the learned value is
        V_orig - shaping * phi(s) (potential-based shaping, potential 0 at terminal states), and -phi grows with the
        distance from the seated pose, so a far-away hand would look valuable; V_orig ~ V + shaping * phi_hat.
        A validation diagnostic only (train_wm320.validate); the online agent does not use it."""
        return self.scalar(self.V(z))[..., 0].mean(0) + float(self.shaping) * self.phi_hat(z)

    def act_chunk(self, z):
        """The built-in ACT head: the next 16 joint targets relative to the measured joints [rad], (B, 16, 7)."""
        return (self._act(z) * self.a_std + self.a_mean).view(-1, CHUNK, D.JOINTS)

    def scale(self, delta):                          # as wmdata320.scale_action
        return (torch.asinh(delta / self.s0) / torch.asinh(self.top / self.s0)).clamp(-1, 1)

    @torch.no_grad()
    def policy_action(self, z):
        """The built-in head's next decision in the model's action scale: z (..., L) -> (..., 56)."""
        was = self._act.training
        self._act.eval()
        a = self.act_chunk(z.reshape(-1, z.shape[-1]))[:, :WD.EXECUTE].reshape(*z.shape[:-1], WD.ACTION_DIM)
        self._act.train(was)
        return self.scale(a)

    @torch.no_grad()
    def target_value(self, z, a):
        v = self.scalar(self._Qt(torch.cat([z, a], -1)))[..., 0]
        i = torch.randperm(v.shape[0], device=v.device)[:2]
        return v[i].min(0).values

    @torch.no_grad()
    def update_target(self, tau: float = 0.01):
        for pt, p in zip(self._Qt.parameters(), self._Qs.parameters()):
            pt.lerp_(p.detach(), tau)

    def observe(self, rgb, depth, q):
        """Windows: rgb (B, H+4, 320, 320, 3), depth (B, H+4, ...), q (B, H+1, 7) -> z (B, H+1, L)."""
        B, S = depth.shape[:2]
        tok = self.enc.tokens(rgb.flatten(0, 1), depth.flatten(0, 1)).view(B, S, TOKENS, TOKEN_DIM)
        return torch.stack([self.enc.fuse(tok[:, j:j + D.CONTEXT], q[:, j]) for j in range(S - D.CONTEXT + 1)], 1)

    # ---- training
    def forward(self, *args, **kwargs):
        """= loss(): DistributedDataParallel synchronises gradients only through forward()."""
        return self.loss(*args, **kwargs)

    def loss(self, W: dict, Bc: dict | None, step: int, warmup: int, bc_weight: float, detail: bool = False):
        cfg, M = self.cfg, self.math
        B = W["q"].shape[0]
        z = self.observe(W["rgb"], W["depth"], W["q"])                        # (B, H+1, L)
        act, rew, term, valid = W["action"].transpose(0, 1), W["reward"].T, W["term"].T, W["valid"].T
        boot, bok = W["boot"].T, W["boot_ok"].T                              # a jam's continuation (wmdata320)
        ov, vv = W["obs_valid"].T, W["v_valid"].T
        zs, cons = [z[:, 0]], 0.0
        for t in range(WD.H):
            zt = self.next(zs[-1], act[t])
            cons = cons + ((zt - z[:, t + 1].detach()).pow(2).mean(-1) * valid[t]).sum() / B * cfg.rho ** t
            zs.append(zt)
        zs = torch.stack(zs)                                                  # (H+1, B, L)
        zd = z.detach().transpose(0, 1)
        with torch.no_grad():
            if step < warmup:                                                 # the data's own next action
                a_nx, a_ok = W["a_next"].transpose(0, 1), W["a_next_valid"].T
            else:
                a_nx, a_ok = self.policy_action(zd), torch.ones_like(ov)
            tv = self.target_value(zd, a_nx).clamp(-self.ret_max, self.ret_max)    # (H+1, B)
            y = (rew + cfg.gamma * (1 - term) * torch.where(bok > 0, boot, tv[1:])).clamp(-self.ret_max, self.ret_max)
            q_ok = valid * (term + bok + (1 - term - bok) * a_ok[1:])
            v_ok = vv * a_ok
        rp, qs, vs = self.reward(zs[:-1], act), self.Q(zs[:-1], act), self.V(zs)
        rl, ql, vl = 0.0, 0.0, 0.0
        for t in range(WD.H + 1):
            w = cfg.rho ** t
            if t < WD.H:
                rl = rl + (M.soft_ce(rp[t], rew[t, :, None], cfg)[:, 0] * valid[t]).sum() / B * w
                for h in range(cfg.num_q):
                    ql = ql + (M.soft_ce(qs[h, t], y[t, :, None], cfg)[:, 0] * q_ok[t]).sum() / B * w
            for h in range(cfg.num_q):
                vl = vl + (M.soft_ce(vs[h, t], tv[t, :, None], cfg)[:, 0] * v_ok[t]).sum() / B * w
        cons, rl, ql, vl = cons / WD.H, rl / WD.H, ql / (WD.H * cfg.num_q), vl / ((WD.H + 1) * cfg.num_q)
        dlog = self._done(z)[..., 0]
        dimg = self._done(zs[1:])[..., 0].T                                  # the imagined latents too (as TD-MPC2's termination)
        dl = ((F.binary_cross_entropy_with_logits(dlog, W["done"], reduction="none") * W["obs_valid"]).sum()
              + (F.binary_cross_entropy_with_logits(dimg, W["done"][:, 1:], reduction="none") * W["obs_valid"][:, 1:]).sum()) / (B * (2 * WD.H + 1))
        pl = (((self._probe(z.detach()) - W["probe"] / 10.0) ** 2).mean(-1) * W["obs_valid"]).sum() / (B * (WD.H + 1))
        bc = zs.sum() * 0
        if Bc is not None:
            tok = self.enc.tokens(Bc["rgb"].flatten(0, 1), Bc["depth"].flatten(0, 1)).view(*Bc["depth"].shape[:2], TOKENS, TOKEN_DIM)
            zb = self.enc.fuse(tok, Bc["q"])
            bc = (self._act(zb) - (Bc["a"] - self.a_mean) / self.a_std).abs().mean()
        total = cfg.consistency_coef * cons + cfg.reward_coef * rl + cfg.value_coef * (ql + vl) + 0.1 * dl + bc_weight * bc + pl
        info = {"total": total, "consistency": cons, "reward_loss": rl, "q_loss": ql, "v_loss": vl, "done_loss": dl, "probe_loss": pl,
                "bc_loss": bc, "z_spread": z[:, 0].std(0).mean(), "target_value_mean": (tv * v_ok).sum() / v_ok.sum().clamp(min=1)}
        info = {k: float(v) for k, v in info.items()}
        if detail:
            with torch.no_grad():
                info["_v0"] = self.scalar(self.V(z[:, 0]))[..., 0].mean(0).cpu().numpy()
                info["_v0u"] = self.unshaped_value(z[:, 0]).cpu().numpy()
                info["_phi0"] = self.phi_hat(z[:, 0]).cpu().numpy()
                info["_vpress_u"] = self.unshaped_value(z[:, -2]).cpu().numpy()   # observation H-1: of an end window, the last pressing state
                info["_done"] = torch.sigmoid(dlog).cpu().numpy()
                info["_done_imag"] = torch.sigmoid(dimg).cpu().numpy()
                info["_imagined_probe"] = (self._probe(zs[-1]) * 10.0).cpu().numpy()
        return total, info


def load(ckpt: str, device="cuda") -> "WM320":
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("labels") != WD.LABELS:
        raise ValueError(f"{ckpt} was trained with labels {ck.get('labels')}, this code has {WD.LABELS}")
    m = WM320(ck["stem"], ck["stats"], ck.get("shaping", 0.0), **ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


# ---------------------------------------------------------------- data (torch wrappers around wmdata320 / demo320)
class WindowData(torch.utils.data.Dataset):
    """One item = one window (wmdata320.window) of a fast-copy trajectory with its H+4 pictures. items: rows (entry, k)."""

    def __init__(self, root, entries, items, stats: dict, shaping: float, gamma: float):
        self.paths = [str(Path(root) / e["path"]) for e in entries]
        self.items, self.st, self.shaping, self.gamma = np.asarray(items, np.int64), stats, shaping, gamma
        self.kind = [WD_KINDS.get(e["kind"], 3) for e in entries]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, k = (int(x) for x in self.items[i])
        with np.load(self.paths[e]) as z:
            meta = json.loads(str(z["meta"]))
            tr = WD.trajectory(z, meta, self.shaping, self.gamma)
            jb, jo, pb, po = z["rgb_jpeg"], z["rgb_offsets"], z["depth_png"], z["depth_offsets"]
        w = WD.window(tr, k, self.st["s0"], self.st["top"])
        s = w.pop("slots")
        u = {int(j): (D.decode_rgb(jb, jo, int(j)), np.minimum(D.decode_depth(pb, po, int(j)), 32767).astype(np.int16)) for j in np.unique(s)}
        out = {k2: torch.from_numpy(np.asarray(v)) for k2, v in w.items()}
        out.update(rgb=torch.from_numpy(np.stack([u[int(j)][0] for j in s])), depth=torch.from_numpy(np.stack([u[int(j)][1] for j in s])),
                   kind=torch.tensor(self.kind[e]), end_seated=torch.tensor(float(tr["end_seated"])), item=torch.tensor([e, k]))
        return out


WD_KINDS = {"reference": 0, "insert": 0, "perturbed": 0, "contact": 1, "random": 2}


def window_items(entries: list[dict]) -> np.ndarray:
    """(entry, k) for every decision k of every trajectory (n = steps // 8)."""
    return np.array([(i, k) for i, e in enumerate(entries) for k in range(e["steps"] // WD.EXECUTE)], np.int64).reshape(-1, 2)


def stats_for(root, entries: list[dict], n_traj: int = 600, seed: int = 0) -> dict:
    """Action scale of the model (from n_traj training trajectories of every kind) and the built-in head's action
    normalisation (20,000 imitation samples, the policy's rule)."""
    rng = np.random.default_rng(seed)
    trs = []
    for i in rng.choice(len(entries), min(n_traj, len(entries)), replace=False):
        with np.load(Path(root) / entries[i]["path"]) as z:
            tr = WD.trajectory(z, json.loads(str(z["meta"])), 0.0, 0.99)
        if tr is not None:
            trs.append(tr)
    s0, top = WD.action_scale(trs)
    bc = [e for e in entries if e.get("bc")]
    am, asd = D.action_stats(root, bc, n=20000, seed=seed)
    return {"s0": s0, "top": top, "a_mean": am.astype(np.float32), "a_std": asd.astype(np.float32)}


def auc(score, label) -> float | None:
    """Area under the ROC curve (scikit-learn); None when one class is missing or there are fewer than 10 points."""
    from sklearn.metrics import roc_auc_score
    s, y = np.asarray(score, np.float64), np.asarray(label) > 0.5
    if y.all() or not y.any() or len(y) < 10:
        return None
    return float(roc_auc_score(y, s))


def face_items(entries: list[dict], root, shaping: float, gamma: float, kinds=("contact",)) -> np.ndarray:
    """(entry, k) of the decision just before the pin tips reach the board face: the first k whose next observation
    has the hand within half a millimetre of the face along the insertion axis (FK; also with a sideways offset)."""
    out = []
    for i, e in enumerate(entries):
        if e["kind"] not in kinds:
            continue
        with np.load(Path(root) / e["path"]) as z:
            tr = WD.trajectory(z, json.loads(str(z["meta"])), shaping, gamma)
        if tr is None:
            continue
        L_mm = 1000.0 * float(e["pin_length_m"])
        hit = np.nonzero(tr["probe"][1:, 0] >= -L_mm - 0.5)[0]
        if len(hit):
            out.append((i, int(hit[0])))
    return np.array(out, np.int64).reshape(-1, 2)


def end_items(entries: list[dict], kinds=("contact",)) -> np.ndarray:
    """(entry, max(0, n - H)) of every trajectory of the given kinds: its last window, which ends at its last state."""
    return np.array([(i, max(0, e["steps"] // WD.EXECUTE - WD.H)) for i, e in enumerate(entries)
                     if e["kind"] in kinds and e["steps"] >= WD.EXECUTE], np.int64).reshape(-1, 2)


__all__ = ["WM320", "Encoder", "WindowData", "window_items", "stats_for", "face_items", "end_items", "auc", "load"]
