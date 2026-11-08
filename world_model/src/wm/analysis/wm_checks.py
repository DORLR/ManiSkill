"""What does the world model predict? Offline checks on held-out shapes (ood_val), v3.

All poses are read from latents by the autoencoder's pose head: "distance to inserted" = the
reference-grid displacement of the remaining motion (mm), the same reading for real and predicted
latents. Simulator truth is not used.

horizon_accuracy   direct prediction at fixed time shifts (0.4 ... 16 s ahead, 4 s back): pose error
                   (grid mm, angle deg) of the prediction vs the true latent, next to "nothing
                   changes" (the current latent)
jam_separation     contact trials, context ending at the push start: direct prediction 2 s and 4 s
                   ahead with the recorded commanded path. Progress = distance now - distance
                   then. Reported per outcome (jammed / inserted) for real and predicted latents,
                   and the AUC with which progress separates the outcomes (real progress = what
                   is achievable; predicted = whether the world model foresees the jam)
rollout_stability  long imagined rollouts in 0.4 s steps (up to 16 s) on recorded trajectories with
                   the recorded commands, nothing real fed back: the latents are the world model's
                   own predictions and the measured joints / hand poses come from the joint
                   adapter + FK (Ctrl-World's arrangement). Errors vs the real trajectory over time,
                   and the RMS of the imagined latents relative to real ones (drift off the latent
                   distribution shows up there)
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.conditions import rel_t_seconds, wm_condition
from ..data.datasets import CONTEXT, WM_STRIDE, LatentWindowDataset, adapter_io
from ..geometry import grid_pose_error_mm, plug_region_points, pose_loss_terms
from ..models.cdit import create_diffusion, grid_to_tokens, tokens_to_grid

HORIZONS = {"+0.4s": 2, "+2s": 10, "+4s": 20, "+8s": 40, "+16s": 80, "-4s": -20}


class Reader:
    """Pose head readings and world-model sampling with fixed settings."""

    def __init__(self, vae, wm, scale: float, dev, sampling_steps: int = 50):
        self.vae, self.wm, self.scale, self.dev = vae, wm, scale, dev
        self.sampler = create_diffusion(timestep_respacing=str(sampling_steps))
        self.ref = plug_region_points(dev)

    @torch.no_grad()
    def pose(self, z):
        return self.vae.decode_pose(z / self.scale)

    @torch.no_grad()
    def distance(self, z) -> np.ndarray:
        """Distance to inserted (mm) read from latents (B, 64, C)."""
        R, t = self.pose(z)
        eye = torch.eye(3, device=z.device).expand(len(z), 3, 3)
        return grid_pose_error_mm(R.float(), t.float(), eye, torch.zeros_like(t).float(), self.ref).cpu().numpy()

    @torch.no_grad()
    def errors(self, z_a, z_b) -> tuple[np.ndarray, np.ndarray]:
        Ra, ta = self.pose(z_a)
        Rb, tb = self.pose(z_b)
        g, a = pose_loss_terms(Ra, ta, Rb, tb, self.ref)
        return g.cpu().numpy(), a.cpu().numpy()

    @torch.no_grad()
    def predict(self, ctx, y, rel_t) -> torch.Tensor:
        B = ctx.shape[0]
        kw = dict(y=y, x_cond=tokens_to_grid(ctx), rel_t=rel_t)
        C = ctx.shape[-1]
        x = self.sampler.p_sample_loop(self.wm.forward, (B, C, 8, 8), torch.randn(B, C, 8, 8, device=self.dev),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.dev)
        return grid_to_tokens(x)


def _med(x) -> float:
    return round(float(np.median(x)), 3) if len(x) else float("nan")


def horizon_accuracy(rd: Reader, latent_root, entries, data_root, n_per_shift: int = 1024, batch: int = 128) -> dict:
    out = {}
    for name, m in HORIZONS.items():
        ds = LatentWindowDataset(latent_root, entries, "ood_val", data_root, rd.scale, fixed_shift=m)
        idx = np.random.default_rng(0).choice(len(ds), size=min(n_per_shift, len(ds)), replace=False)
        g, a, gc, ac = [], [], [], []
        for b in range(0, len(idx), batch):
            items = [ds[int(i)] for i in idx[b:b + batch]]
            st = lambda k: torch.stack([it[k] for it in items]).to(rd.dev)
            pred = rd.predict(st("x_cond"), st("y"), st("rel_t"))
            e = rd.errors(pred, st("x"))
            c = rd.errors(st("x_cond")[:, -1], st("x"))
            g.append(e[0]), a.append(e[1]), gc.append(c[0]), ac.append(c[1])
        g, a, gc, ac = map(np.concatenate, (g, a, gc, ac))
        out[name] = {"windows": len(g), "pred_grid_mm": _med(g), "pred_angle_deg": _med(a),
                     "copy_grid_mm": _med(gc), "copy_angle_deg": _med(ac)}
    return out


def jam_separation(rd: Reader, trials: list[dict], shifts=(("+2s", 10), ("+4s", 20)), batch: int = 128) -> dict:
    from .probes import _auc
    out = {}
    for name, m in shifts:
        rows = []
        usable = [t for t in trials if t["j0"] >= (CONTEXT - 1) * WM_STRIDE and t["j0"] + m < len(t["lat"]["img_step"])]
        for b in range(0, len(usable), batch):
            chunk = usable[b:b + batch]
            ctx, y, rel, tgt = [], [], [], []
            for t in chunk:
                d, j = t["lat"], t["j0"]
                steps = d["img_step"].astype(int)
                c = [j - WM_STRIDE * k for k in range(CONTEXT - 1, -1, -1)]
                ctx.append(d["mean"][c] * rd.scale)
                tgt.append(d["mean"][j + m] * rd.scale)
                y.append(wm_condition(d["hand"], d["hand_cmd"], steps[c], int(steps[j]), int(steps[j + m]), t["R_P"]))
                rel.append(rel_t_seconds(int(steps[j]), int(steps[j + m])))
            T = lambda v: torch.tensor(np.stack(v), dtype=torch.float32, device=rd.dev)
            ctx_t = T(ctx)
            now = rd.distance(ctx_t[:, -1])
            real = rd.distance(T(tgt))
            pred = rd.distance(rd.predict(ctx_t, T(y), T(rel)))
            rows += [(t["y"], now[i] - real[i], now[i] - pred[i]) for i, t in enumerate(chunk)]
        r = np.array(rows)
        if not len(r):
            continue
        yv = r[:, 0].astype(int)
        res = {"trials": len(r)}
        for lab, nm in ((1, "jammed"), (0, "inserted")):
            res[f"{nm}_real_progress_mm"] = _med(r[yv == lab, 1])
            res[f"{nm}_wm_progress_mm"] = _med(r[yv == lab, 2])
        res["auc_real_progress"] = round(_auc(-r[:, 1], yv), 3)        # less progress -> jammed
        res["auc_wm_progress"] = round(_auc(-r[:, 2], yv), 3)
        out[name] = res
    return out


def rollout_stability(rd: Reader, adapter, fk, trajs: list[dict], steps_ahead: int = 40, batch: int = 32) -> dict:
    """trajs: dicts with 'lat' (latent file arrays) and 'R_P'; each needs >= 6 + 2 * steps_ahead images."""
    report_at = {1: "0.4s", 5: "2s", 10: "4s", 20: "8s", 40: "16s"}
    acc = {k: {"grid": [], "angle": [], "joint_mrad": [], "hand_mm": [], "rms_ratio": []} for k in report_at}
    j0 = (CONTEXT - 1) * WM_STRIDE
    usable = [t for t in trajs if len(t["lat"]["img_step"]) > j0 + WM_STRIDE * steps_ahead]
    for b in range(0, len(usable), batch):
        chunk = usable[b:b + batch]
        q_im = [t["lat"]["qpos"].astype(np.float64).copy() for t in chunk]
        h_im = [t["lat"]["hand"].astype(np.float64).copy() for t in chunk]
        ctx = torch.tensor(np.stack([t["lat"]["mean"][[j0 - WM_STRIDE * c for c in range(CONTEXT - 1, -1, -1)]]
                                     for t in chunk]) * rd.scale, dtype=torch.float32, device=rd.dev)
        for k in range(steps_ahead):
            ys, rels = [], []
            for i, t in enumerate(chunk):
                d = t["lat"]
                st = d["img_step"].astype(int)
                j = j0 + WM_STRIDE * k
                c = [j - WM_STRIDE * m for m in range(CONTEXT - 1, -1, -1)]
                ys.append(wm_condition(h_im[i], d["hand_cmd"], st[c], int(st[j]), int(st[j + WM_STRIDE]), t["R_P"]))
                rels.append(rel_t_seconds(int(st[j]), int(st[j + WM_STRIDE])))
                # joints for the next 0.4 s from the adapter (imagined history), hand poses by FK
                s = int(st[j])
                with torch.no_grad():
                    x, _ = adapter_io(q_im[i].astype(np.float32), d["action"].astype(np.float32), s)
                    dq = adapter(torch.tensor(x[None], device=rd.dev))[0].view(8, 7).cpu().numpy()
                n = min(8, len(q_im[i]) - 1 - s)
                q_im[i][s + 1:s + 1 + n, :7] = q_im[i][s, :7] + dq[:n]
                h_im[i][s + 1:s + 1 + n] = fk(q_im[i][s + 1:s + 1 + n])
            T = lambda v: torch.tensor(np.stack(v), dtype=torch.float32, device=rd.dev)
            z = rd.predict(ctx, T(ys), T(rels))
            ctx = torch.cat([ctx[:, 1:], z[:, None]], 1)
            if k + 1 in report_at:
                jt = j0 + WM_STRIDE * (k + 1)
                real = T([t["lat"]["mean"][jt] * rd.scale for t in chunk])
                g, a = rd.errors(z, real)
                st_t = [int(t["lat"]["img_step"][jt]) for t in chunk]
                jerr = [np.abs(q_im[i][s, :7] - t["lat"]["qpos"][s, :7]).mean() * 1000 for i, (t, s) in enumerate(zip(chunk, st_t))]
                herr = [np.linalg.norm(h_im[i][s][:3, 3] - t["lat"]["hand"][s][:3, 3]) * 1000 for i, (t, s) in enumerate(zip(chunk, st_t))]
                rms = (z.pow(2).mean((1, 2)).sqrt() / real.pow(2).mean((1, 2)).sqrt()).cpu().numpy()
                a_ = acc[k + 1]
                a_["grid"] += list(g)
                a_["angle"] += list(a)
                a_["joint_mrad"] += jerr
                a_["hand_mm"] += herr
                a_["rms_ratio"] += list(rms)
    return {report_at[k]: {"trajectories": len(v["grid"]), "pose_grid_mm": _med(v["grid"]), "pose_angle_deg": _med(v["angle"]),
                           "adapter_joint_err_mrad": _med(v["joint_mrad"]), "hand_pos_err_mm": _med(v["hand_mm"]),
                           "latent_rms_ratio": _med(v["rms_ratio"])} for k, v in acc.items()}


@torch.no_grad()
def adapter_on_contact(adapter, data_root: str, split: str = "ood_val", dev=None) -> dict:
    """Joint adapter on the contact trials (raw trajectories; no latents needed), per phase of the push:
    free (before the plug reaches the hole: first 0.8 s of the push) and late (after 1.2 s, where jammed
    trials are blocked), for jammed and inserted trials. Error of the adapter vs the rule "the joints
    reach the commanded targets" (mrad, mean over the 8 predicted steps x 7 joints)."""
    from ..data import index
    from ..data.datasets import port_axes_ok
    from .probes import PUSH_PHASE
    rows = {}
    for e in index.load(data_root):
        if e["split"] != split or e["kind"] != "contact" or e.get("outcome") not in ("inserted", "jammed") or not port_axes_ok(e):
            continue
        with np.load(e["path"]) as f:
            qpos, action, phase = f["qpos"].astype(np.float32), f["action"].astype(np.float32), f["phase"]
        push = np.flatnonzero(phase == PUSH_PHASE)
        if len(push) == 0:
            continue
        p0 = int(push[0])
        for s in push[:-8]:
            seg = "free" if s - p0 < 16 else ("late" if s - p0 >= 24 else None)
            if seg is None:
                continue
            x, y = adapter_io(qpos, action, int(s))
            rows.setdefault((e["outcome"], seg), []).append((x, y))
    out = {}
    for (outcome, seg), xy in sorted(rows.items()):
        x = torch.tensor(np.stack([a for a, _ in xy]), device=dev)
        y = torch.tensor(np.stack([b for _, b in xy]), device=dev)
        pred = adapter(x)
        cmd = x[:, -56:]
        out[f"{outcome}/{seg}"] = {"windows": len(xy),
                                   "adapter_err_mrad": round(float((pred - y).abs().mean() * 1000), 4),
                                   "follow_command_err_mrad": round(float((cmd - y).abs().mean() * 1000), 4),
                                   "joint_motion_mrad": round(float(y.abs().mean() * 1000), 4)}
    return out
