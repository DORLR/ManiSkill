"""Train the flow-matching policy on the world model's ResNet tokens (wm/tdmpc/policy_resnet.py). Budget and optimiser
as the v3 policy (configs/v3: batch 256, lr 3e-4, weight decay 0.01, grad clip 1, EMA 0.999).

    python world_model/scripts/train/train_branch_policy.py --tokens <wm_v1b_tok> --out <run> --steps 40000
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from wm.models.policy import LatentFMPolicy
from wm.tdmpc.policy_resnet import CONTEXT, TokenData
from wm.training.common import EMA, Logger, seed_all


@torch.no_grad()
def evaluate(model, ds, mean, std, n=2048, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    fl, err, k = 0.0, 0.0, 0
    for i in range(0, len(idx), 256):
        z, q, a = ds.batch(idx[i:i + 256])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fl += float(model.loss(z, q, (a - mean) / std))
            pred = model.sample(z, q)[:, 0].float() * std + mean
        err += float((pred - a).abs().mean() * 1000)
        k += 1
    model.train()
    return fl / k, err / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int)
    args = ap.parse_args()
    seed_all(args.seed)
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    files = {s: sorted(glob.glob(f"{args.tokens}/{s}/*/*/*.npz")) for s in ("train", "dev", "ood_val")}
    if args.max_files:
        files = {k: v[:args.max_files] for k, v in files.items()}
    train = TokenData(files["train"])
    val = {k: TokenData(files[k]) for k in ("dev", "ood_val") if files[k]}
    sub = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:20000]
    mean, std = train.a[sub].mean(0).cuda(), (train.a[sub].std(0) + 1e-4).cuda()
    cfg = {"latent_ch": int(train.tok.shape[2]), "context": CONTEXT, "d": 256, "layers": 4, "heads": 8, "hidden": 512, "n_tokens": int(train.tok.shape[1])}
    model = LatentFMPolicy(**cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
    ema = EMA(model, 0.999)
    log = Logger(run)
    log(event="start", train_attempts=len(files["train"]), train_samples=len(train), val={k: len(v) for k, v in val.items()},
        tokens_GB=train.tok.numel() * 2 / 1e9, model=cfg, args=vars(args))
    best, acc, t0 = None, [], time.time()
    for step in range(1, args.steps + 1):
        z, q, a = train.batch(torch.randint(len(train), (args.batch,)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(z, q, (a - mean) / std)
        if not torch.isfinite(loss):
            log(event="stopped: loss not finite", step=step)
            raise SystemExit(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        acc.append(float(loss))
        if step % 500 == 0:
            log(step=step, train_loss=float(np.mean(acc)), it_per_s=500 / (time.time() - t0))
            acc, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {}
            for k, ds in val.items():
                fl, err = evaluate(ema.model, ds, mean, std)
                rec[k] = {"flow_loss": fl, "action_err_mrad": err}
            score = np.mean([v["action_err_mrad"] for v in rec.values()]) if rec else 0.0
            ck = {"model": cfg, "ema": ema.model.state_dict(), "raw": model.state_dict(), "mean": mean.cpu().numpy(), "std": std.cpu().numpy(), "step": step}
            torch.save(ck, str(run / "last.pt") + ".tmp"); os.replace(str(run / "last.pt") + ".tmp", run / "last.pt")
            if best is None or score < best:
                best = score
                torch.save(ck, str(run / "best.pt") + ".tmp"); os.replace(str(run / "best.pt") + ".tmp", run / "best.pt")
            log(event="val", step=step, best=float(best), **rec)
            t0 = time.time()
    log(event="done", step=args.steps)


if __name__ == "__main__":
    main()
