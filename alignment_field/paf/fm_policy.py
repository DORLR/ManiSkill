"""Insertion policy: behaviour cloning with a conditional flow-matching action head
(as in SID's execution policy and pi0-style flow policies).

Observation encoder: PointNet over the coloured point cloud + ResNet-18 over
the RGB crop (ImageNet init) + proprio (hand motion since episode start).
Action head: velocity field v(A_tau, tau | obs) over a chunk of H=4 actions;
trained with  || v - (A - A0) ||^2,  A_tau = (1 - tau) A0 + tau A,  A0 ~ N(0, I)
(conditional flow matching, Lipman et al. 2023); sampled with 10 Euler steps.
Actions are hand steps in port axes, sigma units, scaled by 1/0.5.

python -m paf.fm_policy --data "data/v4_train/*.npz" --val-ports G46 G47 G48 --out runs_v4/policy
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .demo import proprio
from .model import PointNet, RGBNet, mlp
from .perception import port_axes_from_obs, preprocess

H_CHUNK, EXEC = 4, 2
ACT_SCALE = 0.5


def time_emb(tau, dim=64):
    f = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=tau.device))
    a = tau[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class FMPolicy(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pc = PointNet()
        self.rgb = RGBNet(pretrained)
        self.cond = mlp((self.pc.out_dim + self.rgb.out_dim + 6, 512, 512), last_act=True)
        self.vel = mlp((512 + H_CHUNK * 6 + 64, 512, 512, H_CHUNK * 6))

    def encode(self, rgb, pts, prop):
        return self.cond(torch.cat([self.pc(pts), self.rgb(rgb), prop], -1))

    def velocity(self, c, A, tau):
        return self.vel(torch.cat([c, A.flatten(1), time_emb(tau)], -1)).view(-1, H_CHUNK, 6)

    def loss(self, rgb, pts, prop, A):
        c = self.encode(rgb, pts, prop)
        tau = torch.rand(A.shape[0], device=A.device)
        A0 = torch.randn_like(A)
        At = (1 - tau)[:, None, None] * A0 + tau[:, None, None] * A
        return ((self.velocity(c, At, tau) - (A - A0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, rgb, pts, prop, steps=10):
        c = self.encode(rgb, pts, prop)
        A = torch.randn(rgb.shape[0], H_CHUNK, 6, device=rgb.device)
        for k in range(steps):
            tau = torch.full((rgb.shape[0],), k / steps, device=rgb.device)
            A = A + self.velocity(c, A, tau) / steps
        return A


class InsertionPolicy:
    """Closed-loop wrapper: port axes from the first view, then act every step."""

    def __init__(self, run_dir, device="cuda"):
        self.dev = device
        self.model = FMPolicy(pretrained=False).to(device).eval()
        self.model.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location=device))
        self.R_P, self.queue = None, []

    def reset(self):
        self.queue = []

    def set_axes_from(self, obs, rng):
        self.R_P, _ = port_axes_from_obs(obs, rng)
        return self.R_P is not None

    def act(self, obs, H0, rng):
        if not self.queue:
            crop, pc = preprocess(obs, self.R_P, rng)
            t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)[None], device=self.dev, dtype=dt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                A = self.model.sample(t(crop, torch.uint8), t(pc), t(proprio(obs.H, H0, self.R_P)))
            self.queue = list(A[0].float().cpu().numpy()[:EXEC] * ACT_SCALE)
        return self.queue.pop(0)


def load_demos(pattern, ports):
    rgb, pc, prop, chunks, grasp = [], [], [], [], []
    for gi, f in enumerate(sorted(glob.glob(pattern))):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if meta.get("port") not in ports or "demo_act" not in d.files or len(d["demo_act"]) == 0:
            continue
        act, did = d["demo_act"], d["demo_id"]
        for k in np.unique(did):
            idx = np.nonzero(did == k)[0]
            a = act[idx]
            pad = np.concatenate([a, np.zeros((H_CHUNK, 6), np.float32)])
            chunks.append(np.stack([pad[t:t + H_CHUNK] for t in range(len(idx))]))
            rgb.append(d["demo_rgb"][idx])
            pc.append(d["demo_pc"][idx])
            prop.append(d["demo_prop"][idx])
            grasp.append(np.full(len(idx), gi))
    return (np.concatenate(rgb), np.concatenate(pc), np.concatenate(prop),
            np.concatenate(chunks) / ACT_SCALE, np.concatenate(grasp))


def main():
    from .ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None, help="warm start from a previous model.pt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    gen = list(generated_specs())
    tr_ports = [p for p in gen if p not in args.val_ports]
    R, P, Q, A, _ = load_demos(args.data, tr_ports)
    try:
        Rv, Pv, Qv, Av, _ = load_demos(args.data, args.val_ports)
    except ValueError:  # no validation-shape demos (smoke tests): hold out 5 % of frames
        k = max(1, len(A) // 20)
        R, Rv, P, Pv, Q, Qv, A, Av = R[k:], R[:k], P[k:], P[:k], Q[k:], Q[:k], A[k:], A[:k]
    print(f"demo frames: train {len(A)}, val {len(Av)}", flush=True)
    T = [torch.as_tensor(x).pin_memory() for x in (R, P, Q, A)]
    V = [torch.as_tensor(x) for x in (Rv, Pv, Qv, Av)]
    model = FMPolicy().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    spe = len(A) // args.bs
    total = args.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    hist, t0 = [], time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(A))
        tot = 0.0
        for s in range(spe):
            b = perm[s * args.bs:(s + 1) * args.bs]
            rgb, pts, prop, act = [x[b].to(dev, non_blocking=True) for x in T]
            g = torch.empty(len(b), 1, 1, 3, device=dev).uniform_(0.8, 1.2)
            rgb = (rgb.float() * g).clamp(0, 255).to(torch.uint8)
            pts = pts.float()
            pts[..., :3] += torch.randn_like(pts[..., :3]) * (0.0005 / 0.05)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(rgb, pts, prop.float(), act.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            idx = torch.randperm(len(Av))[:2048]
            rgb, pts, prop, act = [x[idx].to(dev) for x in V]
            pred = model.sample(rgb, pts.float(), prop.float())[:, 0].float()
            err = ((pred - act[:, 0].float()) * ACT_SCALE).abs().mean(0).cpu().numpy()
        model.train()
        res = {"epoch": ep, "loss": tot / max(1, spe), "val_first_action_abs_err_sigma": err.round(4).tolist(),
               "minutes": (time.time() - t0) / 60}
        hist.append(res)
        torch.save(model.state_dict(), out / "model.pt")
        print(json.dumps(res), flush=True)
    (out / "summary.json").write_text(json.dumps({"history": hist, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
