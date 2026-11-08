"""World model on contact trials, two offline checks (ood_val contact trials, no training).

1. Single step (0.4 s, real context), split by when in the push the window starts:
       push_start  the first 0.4 s of the push (the plug has not been stopped yet; the tracking
                   history holds no sign of blocking): does the WM foresee the stall?
       later       0.4 s or more into the push (a stall may already be visible in the history)
   per trial outcome (inserted / jammed): real vs predicted progress towards inserted, share of
   windows with < 0.2 mm progress ("stalled").
2. Five steps (2 s, the whole push) imagined from the push start: the recorded
   joint targets are replayed, nothing real is fed back. Where the action condition needs measured
   joints (relative targets, tracking history), the joints are imagined too: the real joints at
   the start plus the change of the decoded joints since the start (the decoder's joint error,
   ~25 mrad, is far larger than one step's motion; differences cancel its bias), held over each
   0.4 s step. Cumulative progress after 1..5 steps, real vs imagined, per outcome.

"Progress" = decrease of the distance to inserted read by the VAE's pose head (plug-region points,
mm), the same reading for real and predicted latents.

    python world_model/scripts/probe_wm_contact.py --latents $Q/data/wm_v2_latents \
        --vae $Q/runs/vae_v2/best.pt --wm $Q/runs/wm_v2/best.pt --out $Q/runs/probe_contact_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_contact import Q, trials
from wm.data.datasets import CONTEXT, WM_STRIDE, action_condition
from wm.geometry import apply_rigid, plug_region_points
from wm.models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from wm.models.vae import load_vae

STALL_MM = 0.2


class Models:
    def __init__(self, vae_ckpt, wm_ckpt, scale, dev):
        self.dev, self.scale = dev, scale
        self.vae = load_vae(vae_ckpt, dev)
        ck = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
        self.step, self.history = ck.get("step"), ck["cfg"].get("history", False)
        self.wm = TokenCDiT(action_dim=ck["cfg"]["action_dim"], latent_ch=self.vae.latent_ch, context_size=CONTEXT,
                            size=ck["cfg"]["size"])
        self.wm.load_state_dict(ck["ema"])
        self.wm.to(dev).eval()
        self.sampler = create_diffusion(timestep_respacing="20")
        self.ref = plug_region_points(dev)

    @torch.no_grad()
    def remaining(self, z):
        R, t = self.vae.decode_pose(z / self.scale)
        p = self.ref[None].expand(z.shape[0], -1, -1)
        return ((apply_rigid(p, R.float(), t.float()) - p).norm(dim=-1).mean(1) * 1000).cpu().numpy()

    @torch.no_grad()
    def joints(self, z):
        return self.vae.decode(z / self.scale)["joints"].float().cpu().numpy()

    @torch.no_grad()
    def predict(self, ctx, y):
        B = ctx.shape[0]
        kw = dict(y=y, x_cond=tokens_to_grid(ctx), rel_t=torch.ones(B, device=self.dev))
        x = self.sampler.p_sample_loop(self.wm.forward, (B, self.vae.latent_ch, 8, 8),
                                       torch.randn(B, self.vae.latent_ch, 8, 8, device=self.dev),
                                       clip_denoised=False, model_kwargs=kw, progress=False, device=self.dev)
        return grid_to_tokens(x)


def T(x, dev):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)


def single_step(m: Models, te, max_later: int) -> dict:
    windows = {"push_start": [], "later": []}
    for t in te:
        n = len(t["lat"]["img_step"])
        for j in range(max(t["j0"], (CONTEXT - 1) * WM_STRIDE), n - WM_STRIDE):
            windows["push_start" if j - t["j0"] < WM_STRIDE else "later"].append((t, j))
    rng = np.random.default_rng(0)
    if len(windows["later"]) > max_later:
        windows["later"] = [windows["later"][i] for i in rng.choice(len(windows["later"]), max_later, replace=False)]
    out = {}
    for name, ws in windows.items():
        rows = {0: [], 1: []}
        for b in range(0, len(ws), 64):
            chunk = ws[b:b + 64]
            ctx = T([t["lat"]["mean"][[j - WM_STRIDE * c for c in range(CONTEXT - 1, -1, -1)]] * m.scale for t, j in chunk], m.dev)
            nxt = T([t["lat"]["mean"][j + WM_STRIDE] * m.scale for t, j in chunk], m.dev)
            y = T([action_condition(t["lat"]["action"], t["lat"]["qpos"], int(t["lat"]["img_step"][j]), m.history)
                   for t, j in chunk], m.dev)
            now, real, pred = m.remaining(ctx[:, -1]), m.remaining(nxt), m.remaining(m.predict(ctx, y))
            for k, (t, j) in enumerate(chunk):
                rows[t["y"]].append((now[k] - real[k], now[k] - pred[k]))
        out[name] = {}
        for y, oname in ((0, "inserted_trials"), (1, "jammed_trials")):
            r = np.array(rows[y])
            if len(r):
                out[name][oname] = {
                    "windows": len(r),
                    "real_progress_mm": round(float(np.median(r[:, 0])), 3),
                    "wm_progress_mm": round(float(np.median(r[:, 1])), 3),
                    "abs_err_mm": round(float(np.median(np.abs(r[:, 0] - r[:, 1]))), 3),
                    "real_stalled": round(float((r[:, 0] < STALL_MM).mean()), 3),
                    "wm_stalled": round(float((r[:, 1] < STALL_MM).mean()), 3)}
    return out


def rollout(m: Models, te, offset: int, steps: int, max_trials: int) -> dict:
    """Imagine `steps` WM steps from image j0 + offset with the recorded targets, nothing real fed back."""
    starts = []
    for t in te:
        j = t["j0"] + offset
        if j >= (CONTEXT - 1) * WM_STRIDE and j + WM_STRIDE * steps < len(t["lat"]["img_step"]):
            starts.append((t, j))
    rng = np.random.default_rng(1)
    if len(starts) > max_trials:
        starts = [starts[i] for i in rng.choice(len(starts), max_trials, replace=False)]
    res = {0: [], 1: []}
    for b in range(0, len(starts), 64):
        chunk = starts[b:b + 64]
        ctx = T([t["lat"]["mean"][[j - WM_STRIDE * c for c in range(CONTEXT - 1, -1, -1)]] * m.scale for t, j in chunk], m.dev)
        rem0 = m.remaining(ctx[:, -1])
        j0dec = m.joints(ctx[:, -1])                                  # decoded joints of the real start latent
        qim = [t["lat"]["qpos"].copy() for t, _ in chunk]            # measured joints, overwritten by imagined ones
        real_cum, wm_cum = [], []
        z = ctx[:, -1]
        for k in range(steps):
            ys = []
            for i, (t, j) in enumerate(chunk):
                d = t["lat"]
                s_k = int(d["img_step"][j + WM_STRIDE * k])
                if k > 0:                                             # imagined joints from now on
                    s_prev = int(d["img_step"][j + WM_STRIDE * (k - 1)])
                    s_0 = int(d["img_step"][j])
                    qim[i][s_prev + 1:s_k + 1, :7] = d["qpos"][s_0, :7] + (jz[i] - j0dec[i])
                ys.append(action_condition(d["action"], qim[i], s_k, m.history))
            z = m.predict(ctx, T(ys, m.dev))
            jz = m.joints(z)
            ctx = torch.cat([ctx[:, 1:], z[:, None]], 1)
            real = m.remaining(T([t["lat"]["mean"][j + WM_STRIDE * (k + 1)] * m.scale for t, j in chunk], m.dev))
            real_cum.append(rem0 - real)
            wm_cum.append(rem0 - m.remaining(z))
        real_cum, wm_cum = np.stack(real_cum, 1), np.stack(wm_cum, 1)  # (B, steps)
        for i, (t, _) in enumerate(chunk):
            res[t["y"]].append((real_cum[i], wm_cum[i]))
    out = {}
    for y, oname in ((0, "inserted_trials"), (1, "jammed_trials")):
        if not res[y]:
            continue
        r = np.array([a for a, _ in res[y]])
        w = np.array([b for _, b in res[y]])
        out[oname] = {"trials": len(r),
                      "real_cum_progress_mm": [round(float(x), 2) for x in np.median(r, 0)],
                      "wm_cum_progress_mm": [round(float(x), 2) for x in np.median(w, 0)],
                      "abs_err_mm": [round(float(x), 2) for x in np.median(np.abs(r - w), 0)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{Q}/data/wm_v1")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-later", type=int, default=3000)
    ap.add_argument("--max-trials", type=int, default=1500)
    ap.add_argument("--skip-single", action="store_true")
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    torch.manual_seed(0)
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    te = trials(args.data, args.latents, "ood_val")
    m = Models(args.vae, args.wm, scale, torch.device("cuda"))
    rep = {"wm_checkpoint_step": m.step, "history": m.history}
    if not args.skip_single:
        rep["single_step"] = single_step(m, te, args.max_later)
    print(json.dumps(rep, indent=1), flush=True)
    for name, off in (("from_push_start", 0),):   # a push lasts ~2 s: 5 steps from its start cover it
        rep[f"rollout_5_steps_{name}"] = rollout(m, te, off, 5, args.max_trials)
        print(name, json.dumps(rep[f"rollout_5_steps_{name}"], indent=1), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"wm_contact{args.tag}{'_rollout' if args.skip_single else ''}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
