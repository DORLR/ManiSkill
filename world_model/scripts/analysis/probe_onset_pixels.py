"""Upper bound for probe_onset.py: is "the hand meets the board in the next 0.4 s" in the 320 x 320 RGB-D image at all?
(2026-11-13; the user: measure first, repairs afterwards)

    python world_model/scripts/analysis/probe_onset_pixels.py --out $Q/runs/checks_v6/probe_onset_pixels.json
The same samples, classes and split as probe_onset.py (approach set; decision starts where the hand had been moving
freely; slows = covers < 0.3 of the command's lead over the next 0.4 s, goes on = > 0.45; trained on the train split,
tested on held-out shapes and grasps), but the encoder is trained on this label from the pixels (ImageNet start) together
with the small reader -- what the camera view allows when nothing else competes for the features. Arms:
    as_joint          the joint runs' encoder (ResNet-18, stride 16, 16 channels, stem max-pooling, stored batch-norm)
    no_pool_bn        the same without the stem's max-pooling and with trained batch-norm (measured 2026-11-12 on the
                      lateral read-out: 0.34 -> 0.23 mm; not in the joint runs)
    no_pool_bn+proprio   + the proprioception / command values of probe_onset.py
For comparison (probe_onset.py, frozen encoders): proprio only 0.87; features of joint_v6d at 10 000: 0.81;
of joint_v6c at 50 000: 0.89 (0.91 with proprio).
All held-out samples are scored (no sub-sampling); training batches are drawn 30 % slows.
"""
import argparse
import io
import json
import multiprocessing as mp
import subprocess

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.data.frame_history import pressed_at
from wm.models.frame_encoder import FrameEncoder
from wm.training.common import limit_gpu_memory

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1a")
ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--arms", nargs="+", default=["as_joint", "no_pool_bn", "no_pool_bn+proprio"])
ap.add_argument("--stop-when", default="", help="path of a training log: no new arm is started once its last step is >= --stop-step")
ap.add_argument("--stop-step", type=int, default=0)
args = ap.parse_args()
NAMES = ("lead_end_mm", "lead_start_mm", "advance_last_0.4s_mm", "fraction_last_0.4s", "advance_before_that_mm", "lag_mm", "travelled_mm", "time_s")


def scan(e):
    out = []
    with np.load(e["path"]) as f:
        steps, hand = f["img_step"].astype(int), f["hand"].astype(np.float64)
    with np.load(str(e["path"]).replace(args.data, args.data + "_ee")) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    a = np.asarray(e["R_P"], np.float64)[:, 0]
    x, c = hand[:, :3, 3] @ a * 1000, cmd[:, :3, 3] @ a * 1000
    for j, s in enumerate(steps):
        if s % 8 or s < 16 or s + 8 >= len(hand) or s + 7 >= len(cmd):
            continue
        prev, nxt, lead, lead_prev = x[s] - x[s - 8], x[s + 8] - x[s], c[s + 7] - x[s], c[s - 1] - x[s - 8]
        if lead <= 0.4 or lead_prev <= 0.4 or pressed_at(hand, cmd, int(s), a):
            continue
        fp, fn = prev / lead_prev, nxt / lead
        if fp < 0.45 or 0.3 <= fn <= 0.45:
            continue
        out.append((e["path"], j, int(fn < 0.3), [lead, c[s] - x[s], prev, fp, x[s - 8] - x[s - 16], c[s - 1] - x[s], x[s] - x[steps[0]], s / 20.0], e["split"]))
    return out


def load_frames(items):
    with np.load(items[0][0]) as f:
        off, jpeg, depth = f["rgb_offsets"], f["rgb_jpeg"], f["depth"]
        js = [it[1] for it in items]
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[off[t]:off[t + 1]].tobytes())).convert("RGB")) for t in js])
        return rgb, depth[js].astype(np.int32)


class Reader(nn.Module):
    def __init__(self, ch, proprio, n_p=8):
        super().__init__()
        self.proprio = proprio
        self.conv = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.mlp = nn.Sequential(nn.Linear(256 + (n_p if proprio else 0), 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, z, p):
        h = self.conv(z)
        v = [h.mean((2, 3)), h.amax((2, 3))] + ([p] if self.proprio else [])
        return self.mlp(torch.cat(v, 1))[:, 0]


def auc(score, pos):
    pos = pos.astype(bool)
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return round(float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())), 3)


if __name__ == "__main__":
    limit_gpu_memory()
    dev = torch.device("cuda")
    entries = [e for e in index.load(args.data) if e["split"] in ("train", "ood_val") and e["n_images"] > 1 and port_axes_ok(e)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        found = [it for group in pool.map(scan, entries, chunksize=16) for it in group]
        data = {}
        for split in ("train", "ood_val"):
            by_path = {}
            for it in [it for it in found if it[4] == split]:
                by_path.setdefault(it[0], []).append(it)
            groups = list(by_path.values())
            out = pool.map(load_frames, groups, chunksize=4)
            items = [it for g in groups for it in g]
            data[split] = {"rgb": torch.from_numpy(np.concatenate([o[0] for o in out])), "depth": torch.from_numpy(np.concatenate([o[1] for o in out])),
                           "p": torch.tensor([it[3] for it in items], dtype=torch.float32), "y": torch.tensor([it[2] for it in items], dtype=torch.float32)}
            print(split, "frames", tuple(data[split]["rgb"].shape), "slows", int(data[split]["y"].sum()), "goes on", int((1 - data[split]["y"]).sum()), flush=True)
    tr, va = data["train"], data["ood_val"]
    mu, sd = tr["p"].mean(0), tr["p"].std(0) + 1e-6
    yv = va["y"].numpy()
    pos_i, neg_i = torch.nonzero(tr["y"] == 1)[:, 0], torch.nonzero(tr["y"] == 0)[:, 0]
    rep = {"data": args.data, "train": {"slows": len(pos_i), "goes_on": len(neg_i)}, "ood_val": {"slows": int(yv.sum()), "goes_on": int((1 - yv).sum())},
           "steps": args.steps, "batch": args.batch}

    def score(enc, m):
        enc.eval(), m.eval()
        s = []
        with torch.no_grad():
            for i in range(0, len(yv), 128):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = enc(va["rgb"][i:i + 128].to(dev), va["depth"][i:i + 128].to(dev))
                s.append(m(z.float(), ((va["p"][i:i + 128] - mu) / sd).to(dev)).float().cpu())
        enc.train(), m.train()
        return torch.cat(s).numpy()

    for arm in args.arms:
        if args.stop_when:
            last = subprocess.run(f"grep '\"loss\"' {args.stop_when} | tail -n 1 | sed 's/.*\"step\": \\([0-9]*\\).*/\\1/'", shell=True, capture_output=True, text=True).stdout.strip()
            if last.isdigit() and int(last) >= args.stop_step:
                print("stopped before arm", arm, "(the watched training is at step", last + ")", flush=True)
                break
        torch.manual_seed(1)
        plain = arm == "as_joint"
        enc = FrameEncoder(ch=16, stride=16, arch="resnet18", pool=plain, freeze_bn=plain).to(dev)
        m = Reader(16, arm.endswith("+proprio")).to(dev)
        opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 1e-4}, {"params": m.parameters(), "lr": 1e-3}], weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        g = torch.Generator().manual_seed(2)
        n_pos = round(0.3 * args.batch)
        curve, enc_tr = {}, enc.train()
        for step in range(1, args.steps + 1):
            b = torch.cat([pos_i[torch.randint(0, len(pos_i), (n_pos,), generator=g)], neg_i[torch.randint(0, len(neg_i), (args.batch - n_pos,), generator=g)]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = enc(tr["rgb"][b].to(dev), tr["depth"][b].to(dev))
            loss = nn.functional.binary_cross_entropy_with_logits(m(z.float(), ((tr["p"][b] - mu) / sd).to(dev)), tr["y"][b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == args.steps:
                s = score(enc, m)
                curve[step] = auc(s, yv)
                print(arm, step, "loss %.4f" % float(loss), "held-out AUC", curve[step], flush=True)
        thr = np.quantile(s[yv < 0.5], 0.9)                       # the score that 10 % of "goes on" exceed
        rep[arm] = {"auc_slows_vs_goes_on": curve[args.steps], "auc_during_training": curve, "auc_best_during_training_optimistic": max(curve.values()),
                    "finds_of_the_slow_downs_at_10pct_false_alarms": round(float((s[yv > 0.5] > thr).mean()), 3),
                    "finds_at_2pct_false_alarms": round(float((s[yv > 0.5] > np.quantile(s[yv < 0.5], 0.98)).mean()), 3),
                    "train_loss_last": round(float(loss), 4)}
        print(arm, rep[arm], flush=True)
        json.dump(rep, open(args.out, "w"), indent=1)
        del enc, m, opt
        torch.cuda.empty_cache()
