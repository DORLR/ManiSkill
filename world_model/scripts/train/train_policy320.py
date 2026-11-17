"""Train the 320 px ResNet flow-matching policy (wm/tdmpc/policy320.py) on the fast copy of the expert demonstrations.

Kept from the old policy's training (git 4a3a6d3: training/policy.py, configs/v3/policy.json): behaviour cloning of
the successful expert trajectories; every picture a sample, drawn independently without replacement; effective batch
256; AdamW, lr 3e-4, weight decay 0.01; gradient clip 1; EMA 0.999 (with its warm-up); up to 100,000 steps;
validation every 5,000 steps with the EMA weights; stopping only for harm (5 validations without a new best AND the
error more than 5 % above the best); a non-finite step is skipped, the run stops after 50 of them or on a non-finite
validation; action scales from 20,000 samples.

What differs from the old policy's training, all on purpose:
  encoder      ResNet-18 from scratch on the 320 px pictures, trained end to end (old: frozen DINOv2 latents); hence
               the clip and the weight decay also cover the encoder, and lr 3e-4 without warm-up is untested for it
  augmentation a random shift of --shift-px pixels per sample (old: none, its encoder was frozen); the value is a
               required argument and is logged
  split        new held-out set (interior strips + 13 shapes); the outer ends of the ranges now train
  selection    best.pt and the stopping rule use the SELECTION split (1 in 20 training-side grasps, not trained on),
               never the held-out set; the old run used its held-out validation set
  validation   a fixed list of samples, a fixed batch size (--eval-batch, independent of --batch) and fixed random
               numbers, so that two validations, and the two stems, are comparable
  batch        --batch x --accum = 256 by gradient accumulation (group norm, no batch statistics: the same gradient)
  joints       embedded in float32 (models/policy.py)
Loss values are therefore not comparable with the old run's log (other action scales, other split).

Resume: --resume continues from last.pt (weights, EMA and its counter, optimiser, step, best / bad / skipped); the
sample stream restarts from a seed derived from the step. A run that was stopped before its first validation has no
last.pt: --resume then starts it again from step 0. A regular end (the last step, or stopping for harm)
writes DONE.json and returns 0; anything else returns 1 and leaves no DONE.json.

    python world_model/scripts/train/train_policy320.py --data /mnt/storage3/qingxin/data/wm_v1_fast --out <run> \
        --stem standard --shift-px 4
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from wm.tdmpc import demo320 as D
from wm.tdmpc.policy320 import ResnetFMPolicy, evaluate_policy, policy_loader, shift
from wm.training.common import EMA, Logger, cycle, seed_all


def save(obj: dict, path: Path):
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True, choices=["standard", "nopool"])
    ap.add_argument("--shift-px", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--effective-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=5120)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--keep-every", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--harm-margin", type=float, default=0.05)
    ap.add_argument("--max-skipped", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.batch * args.accum != args.effective_batch:
        raise SystemExit(f"--batch x --accum = {args.batch * args.accum}, not the effective batch {args.effective_batch}")
    run, dev = Path(args.out), torch.device(args.device)
    resume = args.resume and (run / "last.pt").exists()        # --resume on a run that never reached a validation starts it again
    if not args.resume and (run / "log.jsonl").exists():
        raise SystemExit(f"{run} already holds a run; use --resume to continue it")
    if (run / "DONE.json").exists():
        raise SystemExit(f"{run} is finished")
    run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    ents = [e for e in D.load_entries(args.data) if e["bc"]]
    sets = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}
    if not sets["train"] or not sets["select"]:
        raise SystemExit("no trajectories in the training or the selection split")
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if resume else None
    m, s = (ck["mean"], ck["std"]) if ck else D.action_stats(args.data, sets["train"], n=20000, seed=args.seed)
    (run / "stats.json").write_text(json.dumps({"mean": np.asarray(m).tolist(), "std": np.asarray(s).tolist()}))
    mean, std = torch.tensor(m, device=dev), torch.tensor(s, device=dev)
    val = {}                                        # fixed samples: the selection split decides, the others are logged
    for name, n in (("select", args.eval_samples), ("heldout", args.eval_samples), ("train", args.eval_samples // 5)):
        if sets[name]:
            items = D.policy_items(sets[name])
            val[name] = items[np.sort(np.random.default_rng(1).choice(len(items), size=min(n, len(items)), replace=False))]

    def validate(model):
        return {k: evaluate_policy(model, policy_loader(args.data, sets[k], args.eval_batch, min(args.workers, 6), shuffle=False, items=v), mean, std, dev)
                for k, v in val.items()}

    torch.manual_seed(args.seed)                    # the model's initial weights depend on the seed only
    model = ResnetFMPolicy(args.stem).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, 0.999)
    step, best, bad, skipped = 0, None, 0, 0
    if ck:
        model.load_state_dict(ck["raw"]); ema.model.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        ema.n, step, best, bad, skipped = ck["ema_n"], ck["step"], ck["best"], ck["bad"], ck["skipped"]
    info = {"source": "stand-alone", "stem": args.stem, "shift_px": args.shift_px, "effective_batch": args.effective_batch, "lr": args.lr,
            "trajectories": {k: len(v) for k, v in sets.items()}, "grasps": {k: len({e["grasp"] for e in v}) for k, v in sets.items()}}

    def policy_ck():
        return {"format": "policy320-2", "model": {"stem": args.stem}, "ema": ema.model.state_dict(), "mean": np.asarray(m), "std": np.asarray(s),
                "step": step, "info": info}

    if ck and bad == 0:                             # a kill between the checkpoint files: rewrite what the last validation owed
        save(policy_ck(), run / "best.pt")
    if ck and step % args.keep_every == 0 and not (run / f"step{step}.pt").exists():
        save(policy_ck(), run / f"step{step}.pt")
    log = Logger(run)
    log(event="resume" if ck else "start", step=step, **info, kinds=dict(Counter(e["kind"] for e in sets["train"])),
        samples={k: int(sum(e["n_images"] for e in v)) for k, v in sets.items()}, validation_samples={k: len(v) for k, v in val.items()},
        params_m=sum(p.numel() for p in model.parameters()) / 1e6, args=vars(args))
    batches = cycle(policy_loader(args.data, sets["train"], args.batch, args.workers, seed=args.seed + step))
    params = list(model.parameters())
    losses, norms, t0, event = [], [], time.time(), "done"
    while step < args.steps:
        step += 1
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            b = next(batches)
            rgb, depth = shift(b["rgb"].to(dev, non_blocking=True), b["depth"].to(dev, non_blocking=True), args.shift_px)
            loss = model.loss(rgb, depth, b["q"].to(dev), (b["a"].to(dev) - mean) / std) / args.accum
            loss.backward()
            total += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if math.isfinite(total) and math.isfinite(gn):
            opt.step()
            ema.update(model)
            losses.append(total); norms.append(gn)
        else:
            skipped += 1
            log(event="skipped_nonfinite_step", step=step, skipped=skipped)
            if skipped >= args.max_skipped:
                event = "stop_nonfinite"
                break
        if step % 250 == 0:
            log(step=step, train_loss=float(np.mean(losses)) if losses else None, grad_norm=float(np.mean(norms)) if norms else None,
                it_per_s=250 / (time.time() - t0), gpu_GB=torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0, skipped=skipped)
            losses, norms, t0 = [], [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            rec = validate(ema.model)
            score = rec["select"]["action_err_mrad"]
            if not math.isfinite(score):
                event = "stop_nonfinite_validation"
                break
            improved = best is None or score < best
            best, bad = (score, 0) if improved else (best, bad + 1)
            save({**policy_ck(), "raw": model.state_dict(), "opt": opt.state_dict(), "ema_n": ema.n, "best": best, "bad": bad, "skipped": skipped,
                  "args": vars(args)}, run / "last.pt")
            if improved:
                save(policy_ck(), run / "best.pt")
            if step % args.keep_every == 0:
                save(policy_ck(), run / f"step{step}.pt")
            log(event="val", step=step, best=float(best), **rec)
            t0 = time.time()
            if bad >= args.patience and score > best * (1 + args.harm_margin):
                event = "early_stop_harm"
                break
    log(event=event, step=step)
    if event not in ("done", "early_stop_harm"):
        raise SystemExit(1)
    if not (run / f"step{step}.pt").exists():
        save(policy_ck(), run / f"step{step}.pt")
    (run / "DONE.json").write_text(json.dumps({"event": event, "step": step}))


if __name__ == "__main__":
    main()
