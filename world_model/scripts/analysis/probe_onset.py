"""Is "the hand is about to meet the board" in the encoder's features? (2026-11-13)

    python world_model/scripts/analysis/probe_onset.py --ckpt $Q/runs/checks_v6/joint_v6d_10000.pt --out ...
The world model predicts the hand from commands and proprioception and hardly uses the images (check_wm6_image_use.py,
check_wm6_push.py arms): in closed loop the first contact is never predicted. Before anything is changed: is the
information there at all? Frozen encoder of a joint checkpoint; a small reader is trained on the approach set (train
split) and tested on its held-out shapes and grasps (ood_val).

Samples: control steps that are multiples of 8 (the approach script plans its ramps of 8 or 16 commands from the measured
hand at such steps, so the commands of the next 0.4 s were fixed before anything of that window happened), not
"pressed", and the hand was moving freely over the last 0.4 s: it covered > 0.45 of what the command then led by (free
motion covers 0.55-0.75). Classes by the same fraction over the next 0.4 s: slows (< 0.3) = it meets the board;
goes on (> 0.45). Readers, same training:
    proprio          lead at the end and at the start of the coming 0.4 s, advance over the last 0.4 s and its fraction,
                     advance over the 0.4 s before, lag behind the last command, distance travelled since the trajectory
                     start, time since the start
    image            the newest frame's feature map
    image + proprio  both
Reported on ood_val: AUC of slows against goes on, and the same on trajectories that end blocked / inserted.
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
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--neg-per-pos", type=int, default=4)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()


def scan(e):
    out = []
    with np.load(e["path"]) as f:
        steps, hand = f["img_step"].astype(int), f["hand"].astype(np.float64)
        meta = json.loads(str(f["meta"]))
    with np.load(str(e["path"]).replace(args.data, args.data + "_ee")) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    a = np.asarray(e["R_P"], np.float64)[:, 0]
    x = hand[:, :3, 3] @ a * 1000                                         # mm along the insertion axis
    c = cmd[:, :3, 3] @ a * 1000
    for j, s in enumerate(steps):
        if s % 8 or s < 16 or s + 8 >= len(hand) or s + 7 >= len(cmd):
            continue
        prev, nxt, lead, lead_prev = x[s] - x[s - 8], x[s + 8] - x[s], c[s + 7] - x[s], c[s - 1] - x[s - 8]
        if lead <= 0.4 or lead_prev <= 0.4 or pressed_at(hand, cmd, int(s), a):
            continue
        fp, fn = prev / lead_prev, nxt / lead
        if fp < 0.45 or 0.3 <= fn <= 0.45:
            continue
        out.append((e["path"], j, int(fn < 0.3), [lead, c[s] - x[s], prev, fp, x[s - 8] - x[s - 16], c[s - 1] - x[s], x[s] - x[steps[0]], s / 20.0],
                    e["split"], str(meta.get("stopped", meta.get("outcome")))))
    return out


def load_frames(items):
    path = items[0][0]
    with np.load(path) as f:
        off, jpeg, depth = f["rgb_offsets"], f["rgb_jpeg"], f["depth"]
        js = [it[1] for it in items]
        rgb = np.stack([np.asarray(Image.open(io.BytesIO(jpeg[off[t]:off[t + 1]].tobytes())).convert("RGB")) for t in js])
        return rgb, depth[js].astype(np.int32)


class Reader(nn.Module):
    def __init__(self, ch, image, proprio, n_p=8):
        super().__init__()
        self.image, self.proprio = image, proprio
        self.conv = nn.Sequential(nn.Conv2d(ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.SiLU()) if image else None
        self.mlp = nn.Sequential(nn.Linear((256 if image else 0) + (n_p if proprio else 0), 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, z, p):
        v = []
        if self.image:
            h = self.conv(z)
            v += [h.mean((2, 3)), h.amax((2, 3))]
        if self.proprio:
            v.append(p)
        return self.mlp(torch.cat(v, 1))[:, 0]


def auc(score, pos):
    pos = pos.astype(bool)
    if pos.sum() < 5 or (~pos).sum() < 5:
        return None
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return round(float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())), 3)


if __name__ == "__main__":
    limit_gpu_memory()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cuda")
    entries = [e for e in index.load(args.data) if e["split"] in ("train", "ood_val") and e["n_images"] > 1 and port_axes_ok(e)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        found = [it for group in pool.map(scan, entries, chunksize=16) for it in group]
    sets = {}
    for split in ("train", "ood_val"):
        pos = [it for it in found if it[4] == split and it[2] == 1]
        neg = [it for it in found if it[4] == split and it[2] == 0]
        keep = rng.permutation(len(neg))[:args.neg_per_pos * len(pos)]
        sets[split] = pos + [neg[i] for i in keep]
        print(split, "slows", len(pos), "goes on", len(neg), "kept", len(keep), flush=True)
    enc, head, wm, ck = load_joint(args.ckpt, dev)
    del wm, head
    data = {}
    for split, items in sets.items():
        by_path = {}
        for k, it in enumerate(items):
            by_path.setdefault(it[0], []).append((k, it))
        groups = [[it for _, it in g] for g in by_path.values()]
        order = [k for g in by_path.values() for k, _ in g]
        Z = []
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for rgb, dep in pool.imap(load_frames, groups, chunksize=4):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    Z.append(enc(torch.as_tensor(rgb, device=dev), torch.as_tensor(dep, device=dev)).float().cpu())
        Z = torch.cat(Z)
        items = [items[k] for k in order]
        data[split] = {"z": Z, "p": torch.tensor([it[3] for it in items], dtype=torch.float32), "y": torch.tensor([it[2] for it in items], dtype=torch.float32),
                       "end": np.array([it[5] for it in items])}
        print(split, "encoded", tuple(Z.shape), "ends", {k: int((data[split]["end"] == k).sum()) for k in set(data[split]["end"])}, flush=True)
    tr, va = data["train"], data["ood_val"]
    NAMES = ("lead_end_mm", "lead_start_mm", "advance_last_0.4s_mm", "fraction_last_0.4s", "advance_before_that_mm", "lag_mm", "travelled_mm", "time_s")
    mu, sd = tr["p"].mean(0), tr["p"].std(0) + 1e-6
    rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "data": args.data,
           "train": {"slows": int(tr["y"].sum()), "goes_on": int((1 - tr["y"]).sum())}, "ood_val": {"slows": int(va["y"].sum()), "goes_on": int((1 - va["y"]).sum())},
           "ood_val_proprio_means_slows_vs_goes_on": {n: (round(float(va["p"][va["y"] == 1][:, i].mean()), 3), round(float(va["p"][va["y"] == 0][:, i].mean()), 3))
                                                      for i, n in enumerate(NAMES)},
           "single_feature_auc": {n: auc(va["p"][:, i].numpy(), va["y"].numpy()) for i, n in enumerate(NAMES)}}
    yv = va["y"].numpy()
    for name, (im, pr) in {"proprio": (False, True), "image": (True, False), "image+proprio": (True, True)}.items():
        torch.manual_seed(1)
        m = Reader(tr["z"].shape[1], im, pr).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        g = torch.Generator().manual_seed(2)
        curve = {}
        for step in range(1, args.steps + 1):
            b = torch.randint(0, len(tr["y"]), (256,), generator=g)
            loss = nn.functional.binary_cross_entropy_with_logits(m(tr["z"][b].to(dev), ((tr["p"][b] - mu) / sd).to(dev)), tr["y"][b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0 or step == args.steps:
                m.eval()
                with torch.no_grad():
                    s = torch.cat([m(va["z"][i:i + 512].to(dev), ((va["p"][i:i + 512] - mu) / sd).to(dev)).cpu() for i in range(0, len(yv), 512)]).numpy()
                m.train()
                curve[step] = auc(s, yv)
        ins = va["end"] == "inserted"
        rep[name] = {"auc_slows_vs_goes_on": curve[args.steps], "auc_during_training": curve, "train_loss_last": round(float(loss), 4),
                     "accuracy_at_0": round(float(((s > 0) == (yv > 0.5)).mean()), 3),
                     "says_slows_when_slows": round(float((s[yv > 0.5] > 0).mean()), 3), "says_slows_when_goes_on": round(float((s[yv < 0.5] > 0).mean()), 3),
                     "auc_on_trajectories_not_ending_inserted": auc(s[~ins], yv[~ins])}
        print(name, rep[name], flush=True)
    json.dump(rep, open(args.out, "w"), indent=1)
