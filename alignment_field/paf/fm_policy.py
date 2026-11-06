"""RGB-only insertion policy learned from demonstrations (behaviour cloning).

Inputs: the native-resolution RGB crop around the pin tips and holes (no depth,
no point cloud) and joint proprioception (the 7 arm joint angles, radians). Encoder: ResNet-18 (ImageNet init). Output: a chunk of H=4 hand
steps in the current hand frame (sigma units, scaled by 1/0.5); the first 2 are
executed, then the policy re-plans.
Heads:
  fm  conditional flow matching: || v(A_tau, tau | obs) - (A - A0) ||^2 with
      A_tau = (1 - tau) A0 + tau A, A0 ~ N(0, I) (Lipman et al. 2023); 10 Euler steps
  bc  direct regression of the chunk (MSE)
Demo actions were recorded in port axes and are converted to the hand frame
with R_hand^T R_P (R_hand from FK), so the policy needs no port-axis estimate.

python -m paf.fm_policy --data "data/v4_train/*.npz" --head fm --out runs_v4/policy
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
from scipy.spatial.transform import Rotation

from . import config as C  # noqa: F401
from .model import RGBNet, mlp
from .perception import rgb_crop

H_CHUNK, EXEC, ACT_DIM = 8, 4, 7     # 8 future joint targets, execute 4, re-plan
STRIDE = 2                            # control ticks per chunk element (paf.expert_demo)
ACT_SCALE = 0.01                      # rad; labels are divided by this


def time_emb(tau, dim=64):
    f = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=tau.device))
    a = tau[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def hand_proprio(H, H0):
    """Hand motion since the episode start, in the start hand frame, sigma units (FK only)."""
    t = H0[:3, :3].T @ (H[:3, 3] - H0[:3, 3])
    r = Rotation.from_matrix(H0[:3, :3].T @ H[:3, :3]).as_rotvec()
    return (np.r_[t, r] / C.SIGMA).astype(np.float32)


def apply_hand_motion(H, a):
    """Execute action a (sigma units) in the current hand frame."""
    T = H.copy()
    T[:3, 3] = H[:3, 3] + H[:3, :3] @ (a[:3] * C.SIGMA_T)
    T[:3, :3] = H[:3, :3] @ Rotation.from_rotvec(a[3:] * C.SIGMA_R).as_matrix()
    return T


class InsertPolicyNet(nn.Module):
    """RGB-only insertion policy: ResNet-18 on the RGB crop + proprio; action head
    'fm' (conditional flow matching over a chunk) or 'bc' (direct regression)."""

    def __init__(self, head="fm", pretrained=True):
        super().__init__()
        self.head = head
        self.rgb = RGBNet(pretrained)
        self.cond = mlp((self.rgb.out_dim + 7, 512, 512), last_act=True)
        if head == "fm":
            self.out = mlp((512 + H_CHUNK * ACT_DIM + 64, 512, 512, H_CHUNK * ACT_DIM))
        else:
            self.out = mlp((512, 512, 512, H_CHUNK * ACT_DIM))

    def encode(self, rgb, prop):
        return self.cond(torch.cat([self.rgb(rgb), prop], -1))

    def velocity(self, c, A, tau):
        return self.out(torch.cat([c, A.flatten(1), time_emb(tau)], -1)).view(-1, H_CHUNK, ACT_DIM)

    def loss(self, rgb, prop, A):
        c = self.encode(rgb, prop)
        if self.head == "bc":
            return ((self.out(c).view(-1, H_CHUNK, ACT_DIM) - A) ** 2).mean()
        tau = torch.rand(A.shape[0], device=A.device)
        A0 = torch.randn_like(A)
        At = (1 - tau)[:, None, None] * A0 + tau[:, None, None] * A
        return ((self.velocity(c, At, tau) - (A - A0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, rgb, prop, steps=10):
        c = self.encode(rgb, prop)
        if self.head == "bc":
            return self.out(c).view(-1, H_CHUNK, ACT_DIM)
        A = torch.randn(rgb.shape[0], H_CHUNK, ACT_DIM, device=rgb.device)
        for k in range(steps):
            tau = torch.full((rgb.shape[0],), k / steps, device=rgb.device)
            A = A + self.velocity(c, A, tau) / steps
        return A


class InsertionPolicy:
    """Closed-loop wrapper: sees the RGB crop and joint angles, outputs joint targets."""

    def __init__(self, run_dir, device="cuda"):
        self.dev = device
        p = Path(f"{run_dir}/summary.json")
        cfg = json.loads(p.read_text())["args"] if p.exists() else {}
        self.model = InsertPolicyNet(head=cfg.get("head", "fm"), pretrained=False).to(device).eval()
        self.model.load_state_dict(torch.load(f"{run_dir}/model.pt", map_location=device))

    def reset(self):
        pass

    def set_axes_from(self, obs, rng):  # evaluation API; the IL policy needs no port axes
        return True

    def plan(self, obs):
        """Chunk of absolute joint targets (H_CHUNK, 7) for the next H_CHUNK*STRIDE ticks."""
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)[None], device=self.dev, dtype=dt)
        q = obs.qpos[:7].astype(np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            A = self.model.sample(t(rgb_crop(obs), torch.uint8), t(q))
        return q[None] + A[0].float().cpu().numpy() * ACT_SCALE


def main():
    from .il_data import load_demos
    from .ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="expert demonstrations (scripts/collect_expert_demos.py)")
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None, help="warm start from a previous model.pt")
    ap.add_argument("--head", default="fm", choices=["fm", "bc"])
    ap.add_argument("--frame-stride", type=int, default=1, help="keep every k-th recorded frame (memory)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    gen = list(generated_specs())
    tr_ports = [p for p in gen if p not in args.val_ports]
    R, Q, A, _ = load_demos(args.data, tr_ports, args.frame_stride)
    try:
        Rv, Qv, Av, _ = load_demos(args.data, args.val_ports, args.frame_stride)
    except ValueError:  # no validation-shape demos (smoke tests): hold out 5 % of frames
        k = max(1, len(A) // 20)
        R, Rv, Q, Qv, A, Av = R[k:], R[:k], Q[k:], Q[:k], A[k:], A[:k]
    print(f"demo frames: train {len(A)}, val {len(Av)}", flush=True)
    T = [torch.from_numpy(x) for x in (R, Q, A)]  # no pinned copy: 62 GB container limit
    V = [torch.as_tensor(x) for x in (Rv, Qv, Av)]
    model = InsertPolicyNet(head=args.head).to(dev)
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
            rgb, prop, act = [x[b].to(dev, non_blocking=True) for x in T]
            g = torch.empty(len(b), 1, 1, 3, device=dev).uniform_(0.8, 1.2)
            rgb = (rgb.float() * g).clamp(0, 255).to(torch.uint8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(rgb, prop.float(), act.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            idx = torch.randperm(len(Av))[:2048]
            errs = []
            for c in idx.split(256):  # chunks: 256x256 crops would need >4 GB at once
                rgb, prop, act = [x[c].to(dev) for x in V]
                pred = model.sample(rgb, prop.float())[:, 0].float()
                errs.append(((pred - act[:, 0].float()) * ACT_SCALE * 1000).abs().cpu())
            err = torch.cat(errs).mean(0).numpy()  # mrad
        model.train()
        res = {"epoch": ep, "loss": tot / max(1, spe), "val_first_target_abs_err_mrad": err.round(4).tolist(),
               "minutes": (time.time() - t0) / 60}
        hist.append(res)
        torch.save(model.state_dict(), out / "model.pt")
        print(json.dumps(res), flush=True)
    (out / "summary.json").write_text(json.dumps({"history": hist, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
