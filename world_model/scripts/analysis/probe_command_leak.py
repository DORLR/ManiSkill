"""Do the recorded commands alone tell where the hand is? (2026-11-13; before deciding to hide the measured hand)

    python world_model/scripts/analysis/probe_command_leak.py --out $Q/runs/checks_v6/probe_command_leak.json
The scripts that recorded the data set every command from the MEASURED hand (a lead of at most 1.5 mm, re-planned every
0.4-0.8 s). If the command history therefore gives the hand's motion away, hiding the measured hand from the world model
(in part or altogether) removes nothing: the model can read the same thing from the commands.
A small MLP per data set, trained on the train split, tested on held-out shapes and grasps (ood_val). At every decision
start s (multiples of 8 control steps), along the insertion axis, mm:
    inputs   commands   the commanded position at s-1-4k (k = 1..11, i.e. 2.2 s back) and at s+3, s+7, relative to the
                        command at s-1 -- no measured hand anywhere, no image
             + hand     the measured hand at s-4k (k = 0..11) relative to the command at s-1 (what the model gets now)
    targets  the hand's advance over the last 0.4 s; its lag behind the command now; "pressed" now (the rule);
             the hand's advance over the NEXT 0.4 s (the world model's hand output)
Reported: mean absolute error against a constant guess, and AUC for pressed. CPU only.
"""
import argparse
import json
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.data.frame_history import pressed_at

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", nargs="+", default=["/mnt/storage2/qingxin/data/wm_v1a", "/mnt/storage2/qingxin/data/wm_v1c", "/mnt/storage2/qingxin/data/wm_v1x"])
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()


def scan(job):
    root, e = job
    with np.load(e["path"]) as f:
        hand = f["hand"].astype(np.float64)
    with np.load(str(e["path"]).replace(root, root + "_ee")) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    a = np.asarray(e["R_P"], np.float64)[:, 0]
    x, c = hand[:, :3, 3] @ a * 1000, cmd[:, :3, 3] @ a * 1000
    rows = []
    for s in range(48, min(len(x) - 8, len(c) - 7), 8):
        ref = c[s - 1]
        rows.append([c[s - 1 - 4 * k] - ref for k in range(1, 12)] + [c[s + 3] - ref, c[s + 7] - ref] + [x[s - 4 * k] - ref for k in range(12)]
                    + [x[s] - x[s - 8], c[s - 1] - x[s], float(pressed_at(hand, cmd, s, a)), x[s + 8] - x[s]])
    return e["split"], np.asarray(rows, np.float32).reshape(-1, 29)


def auc(score, pos):
    pos = pos.astype(bool)
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return round(float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())), 3)


if __name__ == "__main__":
    torch.set_num_threads(4)
    rep = {}
    for root in args.data:
        name = root.rstrip("/").split("/")[-1]
        entries = [(root, e) for e in index.load(root) if e["split"] in ("train", "ood_val") and e["n_images"] > 1 and port_axes_ok(e)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            out = pool.map(scan, entries, chunksize=32)
        tr = torch.from_numpy(np.concatenate([r for s, r in out if s == "train"]))
        va = torch.from_numpy(np.concatenate([r for s, r in out if s == "ood_val"]))
        if len(tr) > 300000:
            tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:300000]]
        Y, Yv = tr[:, 25:], va[:, 25:]
        r = {"train_decisions": len(tr), "held_out_decisions": len(va), "held_out_pressed_share": round(float(Yv[:, 2].mean()), 3),
             "constant_guess_mae_mm": {"advance_last_0.4s": round(float((Yv[:, 0] - Y[:, 0].median()).abs().mean()), 4),
                                       "lag_now": round(float((Yv[:, 1] - Y[:, 1].median()).abs().mean()), 4),
                                       "advance_next_0.4s": round(float((Yv[:, 3] - Y[:, 3].median()).abs().mean()), 4)}}
        for arm, cols in (("commands only", list(range(13))), ("commands + measured hand", list(range(25)))):
            torch.manual_seed(1)
            X, Xv = tr[:, cols], va[:, cols]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            m = nn.Sequential(nn.Linear(len(cols), 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4))
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
            g = torch.Generator().manual_seed(2)
            for step in range(args.steps):
                b = torch.randint(0, len(X), (1024,), generator=g)
                o = m((X[b] - mu) / sd)
                loss = nn.functional.smooth_l1_loss(o[:, [0, 1, 3]], Y[b][:, [0, 1, 3]], beta=0.05) + 0.2 * nn.functional.binary_cross_entropy_with_logits(o[:, 2], Y[b][:, 2])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                o = m((Xv - mu) / sd)
            held = ((Yv[:, 3].abs() < 0.05) & (Xv[:, 12] > 0.1)).numpy()            # the hand really stays although the command goes on
            moves = (Yv[:, 3] > 0.3).numpy()
            r[arm] = {"advance_last_0.4s_mae_mm": round(float((o[:, 0] - Yv[:, 0]).abs().mean()), 4),
                      "lag_now_mae_mm": round(float((o[:, 1] - Yv[:, 1]).abs().mean()), 4),
                      "pressed_now_auc": auc(o[:, 2].numpy(), Yv[:, 2].numpy()),
                      "advance_next_0.4s_mae_mm": round(float((o[:, 3] - Yv[:, 3]).abs().mean()), 4),
                      "advance_next_corr": round(float(np.corrcoef(o[:, 3].numpy(), Yv[:, 3].numpy())[0, 1]), 3),
                      "predicted_next_advance_mm_where_the_hand_really_stays": round(float(o[:, 3][torch.from_numpy(held)].mean()), 4), "n_stays": int(held.sum()),
                      "predicted_next_advance_mm_where_it_really_moves_over_0.3": round(float(o[:, 3][torch.from_numpy(moves)].mean()), 4), "n_moves": int(moves.sum()),
                      "true_next_advance_mm_where_it_moves": round(float(Yv[:, 3][torch.from_numpy(moves)].mean()), 4)}
        rep[name] = r
        print(name, json.dumps(r), flush=True)
        json.dump(rep, open(args.out, "w"), indent=1)
