"""Train the 320 px world model (wm/tdmpc/wm320.py) offline on the fast copy: every trajectory of the training split,
of every kind and outcome (expert demonstrations, offset pushes that jam or insert, random motion), plus the
imitation stream of the built-in ACT head (the successful expert trajectories, every picture, independent samples).

One optimiser step = --accum micro-batches of --wm-batch windows (4 decisions each) and --bc-batch imitation samples
(all three required: on one 24 GB A5000 the standard stem fits 16 / 32 / 8, the stem without max-pool 4 / 8 / 32;
group norm, so accumulating gives the same gradient, except that the 2 of 5 target heads are drawn per micro-batch).
The pictures' depth enters the stem in float32, but cuDNN's default TF32 rounds it to ~0.1 mm below 0.2 m and
~0.2 mm beyond (as in the stand-alone policy). last.pt is written every --save-every steps; the random generators
are saved with it.
Adam, lr 3e-4 with the encoder at --enc-lr-scale (TD-MPC2's 0.3), gradient clip 20, target Q rate 0.01, one target
update per step; random shift --shift-px per window / sample (required: stated, not a silent default).
Validation (select split decides nothing yet; held-out logged): losses on fixed windows; the done head's AUC on
observations with the pins in the hole or at the face (FK depth > 0); V just before the pins reach the face of a
contact push against how it ended; the hand's offset read by the probe from the imagined latent at the last state of a contact push (its last window)
against the real one, separately for pushes that jammed and that went in (does the model let a jammed hand walk on?);
the built-in head's action error on held imitation samples. Fixed random numbers, restored afterwards.
Resume and DONE.json as train_policy320.py.

    python world_model/scripts/train/train_wm320.py --data $Q/data/wm_v1_fast --out <run> --stem standard --shift-px 4 --shaping 0.3
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wm.tdmpc import demo320 as D
from wm.tdmpc import wmdata320 as WD
from wm.tdmpc.policy320 import PolicyData, collate_cat, policy_loader, shift
from wm.tdmpc.wm320 import WM320, WindowData, auc, end_items, face_items, stats_for, window_items
from wm.training.common import Logger, cycle, seed_all


def save(obj, path: Path):
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def loader(ds, batch, workers, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g if shuffle else None, drop_last=shuffle,
                                       num_workers=workers, persistent_workers=shuffle and workers > 0, prefetch_factor=2 if workers else None,
                                       pin_memory=torch.cuda.is_available())


def to(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def validate(model, sets, args, dev):
    model.eval()
    cpu, gpu = torch.random.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.manual_seed(1234)
    out = {}
    for name, (wl, fl, el, bl) in sets.items():
        num, done_s, done_y, di_s, di_y, v0, end, imag, real, outc = defaultdict(list), [], [], [], [], [], [], [], [], []
        for W in wl:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            for k, v in info.items():
                if not k.startswith("_"):
                    num[k].append(v)
            m = ((W["obs_valid"] > 0) & (W["fk_depth"] > 0)).cpu().numpy()
            done_s.append(info["_done"][m]); done_y.append(W["done"].cpu().numpy()[m])
            di_s.append(info["_done_imag"][m[:, 1:]]); di_y.append(W["done"][:, 1:].cpu().numpy()[m[:, 1:]])
        for W in fl:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            v0.append(info["_v0"]); end.append(W["end_seated"].cpu().numpy())
        for W in el:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            ok = W["obs_valid"][:, -1].cpu().numpy() > 0
            imag.append(info["_imagined_probe"][ok, 0]); real.append(W["probe"][:, -1, 0].cpu().numpy()[ok])
            outc.append(W["end_seated"].cpu().numpy()[ok])
        err = []
        for b in bl:
            b = to(b, dev)
            tok = model.enc.tokens(b["rgb"].flatten(0, 1), b["depth"].flatten(0, 1)).view(*b["depth"].shape[:2], 100, 32)
            pred = model.act_chunk(model.enc.fuse(tok, b["q"])).flatten(1)
            err.append(float((pred - b["a"]).abs().mean()) * 1000)
        rec = {k: float(np.mean(v)) for k, v in num.items()}
        if done_y:
            rec["auc_done_depth_pos"] = auc(np.concatenate(done_s), np.concatenate(done_y))
            rec["auc_done_imagined_depth_pos"] = auc(np.concatenate(di_s), np.concatenate(di_y))
        if v0:
            rec["auc_V_before_face_contact"] = auc(np.concatenate(v0), np.concatenate(end))
        if imag:
            im, re, oc = np.concatenate(imag), np.concatenate(real), np.concatenate(outc) > 0.5
            for tag, sel in (("jammed", ~oc), ("inserted", oc)):
                if sel.any():
                    rec[f"imagined_minus_real_along_mm_{tag}"] = float(np.mean(im[sel] - re[sel]))
                    rec[f"abs_imagined_minus_real_along_mm_{tag}"] = float(np.mean(np.abs(im[sel] - re[sel])))
                    rec[f"windows_{tag}"] = int(sel.sum())
        if err:
            rec["act_err_mrad"] = float(np.mean(err))
        out[name] = rec
    torch.random.set_rng_state(cpu)
    if gpu is not None:
        torch.cuda.set_rng_state_all(gpu)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True, choices=["standard", "nopool"])
    ap.add_argument("--shift-px", type=int, required=True)
    ap.add_argument("--shaping", type=float, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--wm-batch", type=int, required=True)
    ap.add_argument("--bc-batch", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-scale", type=float, default=0.3)
    ap.add_argument("--td-warmup", type=int, default=5000)
    ap.add_argument("--bc-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1 - 1 / 30)
    ap.add_argument("--workers-wm", type=int, default=8)
    ap.add_argument("--workers-bc", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--keep-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run, dev = Path(args.out), torch.device(args.device)
    resume = args.resume and (run / "last.pt").exists()
    if not args.resume and (run / "log.jsonl").exists():
        raise SystemExit(f"{run} already holds a run; use --resume to continue it")
    if (run / "DONE.json").exists():
        raise SystemExit(f"{run} is finished")
    run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    ents = [e for e in D.load_entries(args.data) if e["split"] != "test" and e["steps"] >= WD.EXECUTE]
    by = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if resume else None
    stats = ck["stats"] if ck else stats_for(args.data, by["train"], seed=args.seed)
    rng = np.random.default_rng(1)
    sets = {}
    for name in ("select", "heldout"):
        if not by[name]:
            continue
        wi = window_items(by[name])
        wi = wi[np.sort(rng.choice(len(wi), min(args.eval_windows, len(wi)), replace=False))]
        fi = face_items(by[name], args.data, args.shaping, args.gamma)
        ei = end_items(by[name])
        bce = [e for e in by[name] if e["bc"]]
        bi = D.policy_items(bce)
        bi = bi[np.sort(rng.choice(len(bi), min(1024, len(bi)), replace=False))]
        mk = lambda items: [{k: v.clone() for k, v in b.items()} for b in loader(WindowData(args.data, by[name], items, stats, args.shaping, args.gamma), 16, 4, 0, False)]  # noqa: E731
        sets[name] = (mk(wi), mk(fi) if len(fi) else [], mk(ei) if len(ei) else [], [{k: v.clone() for k, v in b.items()} for b in policy_loader(args.data, bce, 32, 4, shuffle=False, items=bi)])
    torch.manual_seed(args.seed)
    model = WM320(args.stem, stats, args.shaping, gamma=args.gamma).to(dev)
    enc = list(model.enc.parameters())
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt = torch.optim.Adam([{"params": enc, "lr": args.lr * args.enc_lr_scale}, {"params": rest, "lr": args.lr}])
    step, skipped = 0, 0
    if ck:
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); step, skipped = ck["step"], ck["skipped"]
        torch.random.set_rng_state(ck["rng_cpu"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
    info0 = {"format": "wm320-1", "stem": args.stem, "shaping": args.shaping, "cfg": {"gamma": args.gamma}, "stats": stats}

    def checkpoint(full: bool):
        c = {**info0, "state": model.state_dict(), "step": step, "skipped": skipped, "args": vars(args)}
        if full:
            c.update(opt=opt.state_dict(), rng_cpu=torch.random.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        return c

    log = Logger(run)
    log(event="resume" if ck else "start", step=step, trajectories={k: len(v) for k, v in by.items()},
        windows=int(len(window_items(by["train"]))), imitation_samples=int(sum(e["n_images"] for e in by["train"] if e["bc"])),
        validation={k: [len(v[0]) * 16, len(v[1]) * 16, len(v[2]) * 16, len(v[3]) * 32] for k, v in sets.items()},
        params_m=sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, args=vars(args))
    wtrain = cycle(loader(WindowData(args.data, by["train"], window_items(by["train"]), stats, args.shaping, args.gamma), args.wm_batch,
                          args.workers_wm, args.seed + step))
    btrain = cycle(loader(PolicyData(args.data, [e for e in by["train"] if e["bc"]]), args.bc_batch, args.workers_bc, args.seed + 1 + step))
    params = [p for p in model.parameters() if p.requires_grad]
    acc, t0, event = defaultdict(list), time.time(), "done"
    early = {500, 1000, 2500}
    while step < args.steps:
        step += 1
        opt.zero_grad(set_to_none=True)
        finite = True
        for _ in range(args.accum):
            W, Bc = to(next(wtrain), dev), to(next(btrain), dev)
            W["rgb"], W["depth"] = shift(W["rgb"], W["depth"], args.shift_px)
            Bc["rgb"], Bc["depth"] = shift(Bc["rgb"], Bc["depth"], args.shift_px)
            loss, info = model.loss(W, Bc, step, args.td_warmup, args.bc_weight)
            (loss / args.accum).backward()
            finite &= math.isfinite(info["total"])
            for k, v in info.items():
                acc[k].append(v)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 20.0))
        if finite and math.isfinite(gn):
            opt.step()
            model.update_target()
        else:
            skipped += 1
            log(event="skipped_nonfinite_step", step=step, skipped=skipped)
            if skipped >= 50:
                event = "stop_nonfinite"
                break
        acc["grad_norm"].append(gn)
        if step % 250 == 0:
            log(step=step, it_per_s=250 / (time.time() - t0), gpu_GB=torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0,
                **{k: float(np.mean(v)) for k, v in acc.items()})
            acc, t0 = defaultdict(list), time.time()
        if step % args.keep_every == 0:
            save(checkpoint(False), run / f"step{step}.pt")
        if step % args.save_every == 0:
            save(checkpoint(True), run / "last.pt")
        if step in early or step % args.eval_every == 0 or step == args.steps:
            rec = validate(model, sets, args, dev)
            log(event="val", step=step, **rec)
            t0 = time.time()
    log(event=event, step=step)
    if event != "done":
        raise SystemExit(1)
    save(checkpoint(False), run / f"step{step}.pt")
    (run / "DONE.json").write_text(json.dumps({"event": event, "step": step}))


if __name__ == "__main__":
    main()
