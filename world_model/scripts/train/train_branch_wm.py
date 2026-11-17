"""Train the agreed model (wm/tdmpc/model.py) on the branch set. Resumes from <out>/last.pt.

    python world_model/scripts/train/train_branch_wm.py --data /mnt/storage3/qingxin/data/wm_v1b \
        --out /mnt/storage3/qingxin/runs/branch_wm_v1 --steps 40000
One batch = `--files` branch files x `--per-file` windows. Every `--dev-every` steps: the losses on windows of the
dev grasps and of the held-out shapes / grasps, and the report (wm/tdmpc/report.py) on a fixed sample of both.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from wm.tdmpc import data, evaluate, report
from wm.tdmpc.model import LatentWM, make_cfg, optimizer
from wm.training.common import Logger, seed_all, to_device


def sample(branches, n, seed):
    """n branches as whole starts (the report compares the branches of a start)."""
    starts = sorted({(b["grasp"], b["start"]) for b in branches})
    rng = np.random.default_rng(seed)
    keep = {starts[i] for i in rng.permutation(len(starts))[:max(n // 4, 1)]}
    return [b for b in branches if (b["grasp"], b["start"]) in keep]


def save(path, model, opt, step, stats):
    tmp = str(path) + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": vars(model.cfg), "stats": stats}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dev-every", type=int, default=4000)
    ap.add_argument("--dev-branches", type=int, default=160)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-split", default="train", help="smoke tests only: another split to draw the windows from")
    ap.add_argument("--max-branches", type=int)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--stem-stride", type=int, default=2, choices=[2, 4])
    args = ap.parse_args()
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    train_b = data.list_branches(args.data, args.train_split)
    if args.max_branches:
        train_b = train_b[:args.max_branches]
    dev = {"dev": sample(data.list_branches(args.data, "dev"), args.dev_branches, 5),
           "ood_val": sample(data.list_branches(args.data, "ood_val"), args.dev_branches, 5)}
    if (run / "stats.json").exists():
        stats = json.loads((run / "stats.json").read_text())
    else:
        stats = data.compute_stats(train_b)
        (run / "stats.json").write_text(json.dumps(stats))
    cfg = make_cfg(horizon=args.horizon, amp=not args.no_amp, stem_stride=args.stem_stride)
    model = LatentWM(cfg, stats).cuda()
    opt = optimizer(model)
    step = 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    log = Logger(run)
    log(event="start", step=step, train_branches=len(train_b), dev={k: len(v) for k, v in dev.items()},
        params_M=sum(p.numel() for p in model.parameters()) / 1e6, args=vars(args))
    if step >= args.steps:
        return
    ds = data.Windows(train_b, stats, cfg.horizon, cfg.gamma, args.per_file, (args.steps - step) * args.files,
                      seed=args.seed * 1_000_003 + step, heads=cfg.num_q)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.files, num_workers=args.workers, collate_fn=data.collate_cat,
                                         prefetch_factor=2, drop_last=True)

    def dev_check():
        model.eval()
        rec = {}
        for name, bs in dev.items():
            if not bs:
                continue
            dl = torch.utils.data.DataLoader(data.Windows(bs, stats, cfg.horizon, cfg.gamma, 2, 24, seed=777, heads=cfg.num_q),
                                             batch_size=6, num_workers=min(4, args.workers), collate_fn=data.collate_cat)
            with torch.no_grad():
                infos = [model.loss(to_device(b, "cuda"), augment=False)[1] for b in dl]
            rec[name] = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
            rep = report.build(evaluate.score_wm(model, bs, stats, min(6, args.workers)), bs)
            (run / f"report_{name}_step{step}.json").write_text(json.dumps(rep, indent=1))
            rec[name]["report"] = {tag: {s: {"auc": t[s]["auc_inserted"], "pairs": t[s]["same_start"]["pairs"],
                                             "pick": t[s]["all_starts"]["pick"], "random": t[s]["all_starts"]["random"]}
                                         for s in ("v0", "q0", "im2", "im4")}
                                   for tag, t in rep.items() if isinstance(t, dict) and tag in ("start", "face-5", "face-2", "face+0")}
        model.train()
        log(event="dev", step=step, **rec)

    model.train()
    acc, t_last, t_data, t0 = {}, time.time(), 0.0, time.time()
    for batch in loader:
        t_data += time.time() - t0
        total, info = model.loss(to_device(batch, "cuda"))
        if not torch.isfinite(total):
            log(event="stopped: loss not finite", step=step, **info)
            raise SystemExit(1)
        opt.zero_grad(set_to_none=True)
        total.backward()
        info["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm))
        opt.step()
        step += 1
        for k, v in info.items():
            acc.setdefault(k, []).append(v)
        if step % 50 == 0:
            dt = time.time() - t_last
            log(step=step, **{k: float(np.mean(v)) for k, v in acc.items()}, it_per_s=50 / dt, data_wait=t_data / dt,
                gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
            acc, t_last, t_data = {}, time.time(), 0.0
        if step % args.save_every == 0 or step == args.steps:
            save(run / "last.pt", model, opt, step, stats)
            if step % args.keep_every == 0:
                save(run / f"step_{step}.pt", model, opt, step, stats)
        if step % args.dev_every == 0 or step == args.steps:
            dev_check()
            t_last = time.time()
        t0 = time.time()
    log(event="done", step=step)


if __name__ == "__main__":
    main()
