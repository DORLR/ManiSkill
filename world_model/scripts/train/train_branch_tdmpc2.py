"""Train TD-MPC2 unchanged (wm/tdmpc/original.py) on the 64 x 64 copies of the branch set: the control run.
Resumes from <out>/last.pt.

    python world_model/scripts/train/train_branch_tdmpc2.py --data /mnt/storage3/qingxin/data/wm_v1b \
        --cache /mnt/storage3/qingxin/data/wm_v1b_c64 --out /mnt/storage3/qingxin/runs/branch_tdmpc2_orig --steps 100000
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from wm.tdmpc import data, evaluate, original, report
from wm.training.common import Logger, seed_all


def sample(branches, n, seed):
    starts = sorted({(b["grasp"], b["start"]) for b in branches})
    rng = np.random.default_rng(seed)
    keep = {starts[i] for i in rng.permutation(len(starts))[:max(n // 4, 1)]}
    return [b for b in branches if (b["grasp"], b["start"]) in keep]


def save(path, agent, step, stats):
    tmp = str(path) + ".tmp"
    torch.save({"model": agent.model.state_dict(), "optim": agent.optim.state_dict(), "pi_optim": agent.pi_optim.state_dict(),
                "scale": float(agent.scale.value), "step": step, "cfg": vars(agent.cfg), "stats": stats}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--dev-every", type=int, default=10000)
    ap.add_argument("--dev-branches", type=int, default=800)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--max-branches", type=int)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    train_b = data.list_branches(args.data, args.train_split)
    if args.max_branches:
        train_b = train_b[:args.max_branches]
    if (run / "stats.json").exists():
        stats = json.loads((run / "stats.json").read_text())
    else:
        stats = data.compute_stats(train_b)
        (run / "stats.json").write_text(json.dumps(stats))
    cfg = original.make_cfg(steps=args.steps, seed=args.seed, compile=not args.no_compile)
    buffer = original.RamBuffer(original.load_cache(args.cache, train_b), stats, cfg)
    dev = {}
    for name in ("dev", "ood_val"):
        bs = sample(data.list_branches(args.data, name), args.dev_branches, 5)
        dev[name] = (bs, original.load_cache(args.cache, bs))
    agent = original.make_agent(cfg)
    step = 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location="cuda:0", weights_only=False)
        agent.model.load_state_dict(ck["model"])
        agent.optim.load_state_dict(ck["optim"])
        agent.pi_optim.load_state_dict(ck["pi_optim"])
        agent.scale.value.fill_(ck["scale"])
        step = ck["step"]
    log = Logger(run)
    log(event="start", step=step, episodes=buffer.num_eps, windows=buffer.windows, frames_GB=buffer.frames.numel() / 1e9,
        discount=float(agent.discount), args=vars(args))

    def dev_check():
        rec = {}
        for name, (bs, cache) in dev.items():
            if not any(c is not None for c in cache):
                continue
            rep = report.build(evaluate.score_original(agent, cache, stats), bs)
            (run / f"report_{name}_step{step}.json").write_text(json.dumps(rep, indent=1))
            rec[name] = {tag: {s: {"auc": t[s]["auc_inserted"], "pairs": t[s]["same_start"]["pairs"],
                                   "pick": t[s]["all_starts"]["pick"], "random": t[s]["all_starts"]["random"]}
                               for s in ("q0", "qpi", "im3")}
                         for tag, t in rep.items() if isinstance(t, dict) and tag in ("start", "face-5", "face-2", "face+0")}
        log(event="dev", step=step, **rec)

    acc, t_last = {}, time.time()
    while step < args.steps:
        info = agent.update(buffer)
        step += 1
        if step % 100 == 0:
            for k, v in info.items():
                acc.setdefault(k, []).append(float(v))
        if step % 1000 == 0:
            if not np.isfinite(acc["total_loss"]).all():
                log(event="stopped: loss not finite", step=step)
                raise SystemExit(1)
            log(step=step, **{k: float(np.mean(v)) for k, v in acc.items()}, it_per_s=1000 / (time.time() - t_last),
                gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
            acc, t_last = {}, time.time()
        if step % args.save_every == 0 or step == args.steps:
            save(run / "last.pt", agent, step, stats)
        if step % args.dev_every == 0 or step == args.steps:
            dev_check()
            t_last = time.time()
    log(event="done", step=step)


if __name__ == "__main__":
    main()
