"""Which part of the plug-hole misalignment does the latent keep?  (offline check)

Contact trials, at push start (3 mm before the hole): regress the sampled offset (simulator truth,
judging only) -- lateral y, z (mm) and the three angles (deg) -- from
    latent   4 context latents (ridge regression and a small MLP)
    pose     the VAE pose head's reading of the current latent (rotation vector + translation),
             linearly mapped (frames / signs are fitted, nothing else)
    joints   measured joints only (no vision), to see how much comes from grasp/shape priors
Fit on train, report on ood_val (held-out shapes): R^2 and median absolute error per component,
next to the spread of the truth (median |offset|).

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/probe_offset.py --out $Q/runs/probe_contact_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from probe_contact import Q, trials
from wm.data.datasets import CONTEXT, WM_STRIDE

NAMES = ["lateral_y_mm", "lateral_z_mm", "roll_deg", "pitch_deg", "yaw_deg"]


def truth(t):
    return np.array([t["truth"][0], t["truth"][1], *t["truth"][3:6]], np.float32)


def ctx_latent(t, scale):
    j = t["j0"]
    c = [max(0, j - WM_STRIDE * k) for k in range(CONTEXT - 1, -1, -1)]
    return (t["lat"]["mean"][c] * scale).reshape(-1)


def fit(xtr, ytr, xte, kind):
    xtr, ytr, xte = (np.asarray(v, np.float32) for v in (xtr, ytr, xte))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    a, b = torch.tensor((xtr - mu) / sd), torch.tensor((ytr - ym) / ys)
    torch.manual_seed(0)
    m = nn.Linear(a.shape[1], b.shape[1]) if kind == "linear" else nn.Sequential(
        nn.Linear(a.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, b.shape[1]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(2000):
        i = torch.randint(0, len(a), (min(512, len(a)),))
        loss = (m(a[i]) - b[i]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor((xte - mu) / sd)).numpy() * ys + ym


def score(pred, y):
    out = {}
    for k, n in enumerate(NAMES):
        r2 = 1 - ((pred[:, k] - y[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum()
        out[n] = {"r2": round(float(r2), 3), "median_abs_err": round(float(np.median(np.abs(pred[:, k] - y[:, k]))), 3)}
    return out


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 3) axis * angle (rad); fine for the small angles here."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return v * (ang / (2 * torch.sin(ang).clamp_min(1e-6)))[:, None]


@torch.no_grad()
def pose_reading(ts, vae, scale, dev):
    out = []
    for b in range(0, len(ts), 256):
        z = torch.tensor(np.stack([t["lat"]["mean"][t["j0"]] for t in ts[b:b + 256]]), dtype=torch.float32, device=dev)
        d = vae.decode(z)
        out.append(torch.cat([matrix_to_rotvec(d["R"].float()) * 57.2958, d["t"].float() * 1000], -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{Q}/data/wm_v1")
    ap.add_argument("--latents", default=f"{Q}/data/wm_v1_latents")
    ap.add_argument("--vae", default=f"{Q}/runs/vae_v1/best.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    tr, te = trials(args.data, args.latents, "train"), trials(args.data, args.latents, "ood_val")
    ytr, yte = np.stack([truth(t) for t in tr]), np.stack([truth(t) for t in te])
    rep = {"truth_spread_median_abs": {n: round(float(np.median(np.abs(yte[:, k]))), 3) for k, n in enumerate(NAMES)}}
    xtr, xte = np.stack([ctx_latent(t, scale) for t in tr]), np.stack([ctx_latent(t, scale) for t in te])
    for kind in ("linear", "mlp"):
        rep[f"latent/{kind}"] = score(fit(xtr, ytr, xte, kind), yte)
    jtr = np.stack([t["lat"]["qpos"][int(t["lat"]["img_step"][t["j0"]])][:7] for t in tr]).astype(np.float32)
    jte = np.stack([t["lat"]["qpos"][int(t["lat"]["img_step"][t["j0"]])][:7] for t in te]).astype(np.float32)
    rep["joints/mlp"] = score(fit(jtr, ytr, jte, "mlp"), yte)
    try:
        from wm.models.vae import load_vae
        dev = torch.device("cuda")
        vae = load_vae(args.vae, dev)
        ptr, pte = pose_reading(tr, vae, scale, dev), pose_reading(te, vae, scale, dev)
        rep["pose_head/linear"] = score(fit(ptr.astype(np.float32), ytr, pte.astype(np.float32), "linear"), yte)
    except Exception as exc:                        # the latent probes above stand on their own
        rep["pose_head/linear"] = f"failed: {type(exc).__name__}: {exc}"
    for k, v in rep.items():
        print(k, json.dumps(v), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "offset_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
