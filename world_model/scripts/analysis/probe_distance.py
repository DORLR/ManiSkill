"""How far is the board? The distance along the insertion axis from the hand now to where it will stop, read from the
frozen encoder's features (2026-11-13, proposed by the review's dissenting role: axial precision was never measured,
and the walk through the board is an axial failure; probe_onset.py asked for the timing of contact within one 0.4 s
step, where every error is a hit or a miss -- a distance has graceful errors).

    python world_model/scripts/analysis/probe_distance.py --ckpt $Q/runs/checks_v6/joint_v6c_60000.pt --out ...
Approach-set trajectories that end blocked (not inserted). Stop = the hand's position at the first control step the
proprioceptive rule calls pressed. Samples: recorded images before the stop with 0.3-8 mm still to go (at most
`--per-traj` per trajectory, drawn at random). A small reader is trained on the train split and tested on held-out
shapes and grasps (ood_val). Readers:
    blind              the training median for everybody
    travelled+time     distance travelled since the trajectory start and time since the start (the prior over where
                       scripts meet the board; what an imagined rollout also has)
    proprio motion     lead, last advances, lag (no distance, no time)
    proprio all        both
    image              the newest frame's feature map only
    image + proprio all
Reported: mean absolute error in mm, overall and by the true distance (0.3-1, 1-2, 2-4, 4-8 mm), and the correlation.
Kill criterion fixed before the run: if the image reader's error is not below half the blind error, contact position
cannot be taken from vision with this encoder.
"""
import argparse
import io
import json
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.data.frame_history import pressed_at
from wm.training.common import limit_gpu_memory
from wm.training.joint import load_joint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1a")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--per-traj", type=int, default=10)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
BINS = ((0.3, 1), (1, 2), (2, 4), (4, 8))


def scan(e):
    with np.load(e["path"]) as f:
        steps, hand = f["img_step"].astype(int), f["hand"].astype(np.float64)
        meta = json.loads(str(f["meta"]))
    if meta.get("outcome") != "not_inserted":
        return []
    with np.load(str(e["path"]).replace(args.data, args.data + "_ee")) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    a = np.asarray(e["R_P"], np.float64)[:, 0]
    x, c = hand[:, :3, 3] @ a * 1000, cmd[:, :3, 3] @ a * 1000
    stop = next((s for s in range(8, len(cmd)) if pressed_at(hand, cmd, s, a)), None)
    if stop is None:
        return []
    out = []
    for j, s in enumerate(steps):
        if s < 16 or s + 7 >= len(cmd) or s >= stop:
            continue
        d = x[stop] - x[s]
        if not 0.3 <= d <= 8.0:
            continue
        prev, lead_prev = x[s] - x[s - 8], c[s - 1] - x[s - 8]
        out.append((e["path"], j, float(d), [c[s + 7] - x[s], c[s] - x[s], prev, prev / lead_prev if abs(lead_prev) > 0.05 else 0.0, x[s - 8] - x[s - 16],
                                             c[s - 1] - x[s], x[s] - x[steps[0]], s / 20.0], e["split"]))
    rng = np.random.default_rng(abs(hash(e["path"])) % (2 ** 32))
    if len(out) > args.per_traj:
        out = [out[i] for i in sorted(rng.choice(len(out), args.per_traj, replace=False))]
    return out


def load_frames(items):
    with np.load(items[0][0]) as f:
        off, jpeg, depth = f["rgb_offsets"], f["rgb_jpeg"], f["depth"]
        js = [it[1] for it in items]
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[off[t]:off[t + 1]].tobytes())).convert("RGB")) for t in js])
        return rgb, depth[js].astype(np.int32)


class Reader(nn.Module):
    def __init__(self, ch, image, cols):
        super().__init__()
        self.image, self.cols = image, list(cols)
        self.conv = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU()) if image else None
        self.mlp = nn.Sequential(nn.Linear((256 if image else 0) + len(self.cols), 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, z, p):
        v = []
        if self.image:
            h = self.conv(z)
            v += [h.mean((2, 3)), h.amax((2, 3))]
        if self.cols:
            v.append(p[:, self.cols])
        return self.mlp(torch.cat(v, 1))[:, 0]


if __name__ == "__main__":
    limit_gpu_memory()
    dev = torch.device("cuda")
    entries = [e for e in index.load(args.data) if e["split"] in ("train", "ood_val") and e["n_images"] > 1 and port_axes_ok(e)]
    enc, head, wm, ck = load_joint(args.ckpt, dev)
    del wm, head
    data = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        found = [it for group in pool.map(scan, entries, chunksize=16) for it in group]
        for split in ("train", "ood_val"):
            by_path = {}
            for it in [it for it in found if it[4] == split]:
                by_path.setdefault(it[0], []).append(it)
            groups = list(by_path.values())
            Z = []
            for rgb, dep in pool.imap(load_frames, groups, chunksize=4):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    Z.append(enc(torch.as_tensor(rgb, device=dev), torch.as_tensor(dep, device=dev)).float().cpu())
            items = [it for g in groups for it in g]
            data[split] = {"z": torch.cat(Z), "p": torch.tensor([it[3] for it in items], dtype=torch.float32), "y": torch.tensor([it[2] for it in items], dtype=torch.float32)}
            print(split, "trajectories", len(groups), "frames", len(items), flush=True)
    tr, va = data["train"], data["ood_val"]
    mu, sd = tr["p"].mean(0), tr["p"].std(0) + 1e-6
    yv = va["y"].numpy()

    def report(pred):
        err = np.abs(pred - yv)
        r = {"mae_mm": round(float(err.mean()), 3), "median_abs_mm": round(float(np.median(err)), 3),
             "corr": round(float(np.corrcoef(pred, yv)[0, 1]), 3) if np.std(pred) > 1e-6 else None}
        for lo, hi in BINS:
            s = (yv >= lo) & (yv < hi)
            r[f"mae_{lo}-{hi}mm"] = round(float(err[s].mean()), 3)
            r[f"bias_{lo}-{hi}mm"] = round(float((pred - yv)[s].mean()), 3)
        return r

    rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "train_frames": len(tr["y"]), "held_out_frames": len(yv),
           "held_out_by_distance": {f"{lo}-{hi}mm": int(((yv >= lo) & (yv < hi)).sum()) for lo, hi in BINS},
           "blind": report(np.full(len(yv), float(tr["y"].median())))}
    print("blind", rep["blind"], flush=True)
    arms = {"travelled+time": (False, (6, 7)), "proprio motion": (False, (0, 1, 2, 3, 4, 5)), "proprio all": (False, tuple(range(8))),
            "image": (True, ()), "image + proprio all": (True, tuple(range(8)))}
    for name, (im, cols) in arms.items():
        torch.manual_seed(1)
        m = Reader(tr["z"].shape[1], im, cols).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        g = torch.Generator().manual_seed(2)
        curve = {}
        for step in range(1, args.steps + 1):
            b = torch.randint(0, len(tr["y"]), (256,), generator=g)
            loss = nn.functional.smooth_l1_loss(m(tr["z"][b].to(dev), ((tr["p"][b] - mu) / sd).to(dev)), tr["y"][b].to(dev), beta=0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 1000 == 0 or step == args.steps:
                m.eval()
                with torch.no_grad():
                    s = torch.cat([m(va["z"][i:i + 512].to(dev), ((va["p"][i:i + 512] - mu) / sd).to(dev)).cpu() for i in range(0, len(yv), 512)]).numpy()
                m.train()
                curve[step] = round(float(np.abs(s - yv).mean()), 3)
        rep[name] = {**report(s), "mae_during_training": curve, "train_loss_last": round(float(loss), 4)}
        print(name, rep[name], flush=True)
        json.dump(rep, open(args.out, "w"), indent=1)
