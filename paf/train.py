"""Train one leave-one-shape-out fold: perception net + alignment field.

python -m paf.train --held-out A --out runs/fold_A
Training uses only the three other shapes and only training-band grasps (all
of data/train is training-band by construction). 10% of training grasps are a
validation split for model selection; the held-out shape is never loaded.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .field import fit_field
from .model import AlignNet


def load_shards(pattern, ports):
    rgb, pc, q, push, kind, cfg = [], [], [], [], [], []
    metas = []
    for i, f in enumerate(sorted(glob.glob(pattern))):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if meta.get("port") not in ports or "q" not in d.files or len(d["q"]) == 0:
            continue
        metas.append(meta)
        n = len(d["q"])
        rgb.append(d["rgb"])
        pc.append(d["pc"])
        q.append(d["q"])
        push.append(d["push"])
        kind.append(d["kind"])
        cfg.append(np.full(n, len(metas) - 1))
    return dict(rgb=np.concatenate(rgb), pc=np.concatenate(pc), q=np.concatenate(q),
                push=np.concatenate(push), kind=np.concatenate(kind), cfg=np.concatenate(cfg)), metas


def id_labels(u, push, kind):
    """1/0 from push trials; far-away views (> 4 sigma) are certain failures."""
    lab = np.full(len(u), -1.0, np.float32)
    lab[push == 1] = 1.0
    lab[push == 0] = 0.0
    far = (np.linalg.norm(u, axis=1) > 4.0) & (push < 0)
    lab[far] = 0.0
    return lab


def evaluate(model, T, idx, bs=512, dev="cuda"):
    model.eval()
    out_pc, out_hat = [], []
    idx = torch.as_tensor(np.asarray(idx))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            o = model(T["rgb"][b].to(dev), T["pc"][b].to(dev).float())
            out_pc.append(o["q_pc"].float().cpu())
            out_hat.append(o["q_hat"].float().cpu())
    model.train()
    u = T["u"][idx]
    e_pc = (torch.cat(out_pc) - u).abs()
    e_hat = (torch.cat(out_hat) - u).abs()
    near = u.norm(dim=1) < 3.0
    res = {}
    for name, e in (("pc", e_pc), ("hat", e_hat)):
        for reg, m in (("all", torch.ones_like(near)), ("near", near)):
            ee = e[m]
            res[f"{name}_{reg}_t_med_mm"] = float(ee[:, :3].median())
            res[f"{name}_{reg}_r_med_deg"] = float(ee[:, 3:].median())
            res[f"{name}_{reg}_t_p90_mm"] = float(ee[:, :3].quantile(0.9))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--data", default="data/train/*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--h", type=float, default=0.5, help="field bandwidth, sigma units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-port", default=None,
                    help="nested leave-one-shape-out: validate on this training shape (not trained on)")
    ap.add_argument("--mldg", action="store_true",
                    help="episodic domain generalisation (Li et al. 2018, first-order MLDG): each step, "
                         "meta-train on two training shapes, meta-test on the third")
    ap.add_argument("--mldg-alpha", type=float, default=1e-3)
    ap.add_argument("--mldg-beta", type=float, default=1.0)
    ap.add_argument("--monitor-every", type=int, default=0,
                    help="log errors on the held-out test shape every N epochs (record only, never used "
                         "for selection; the final epoch is the model)")
    ap.add_argument("--strong-aug", action="store_true", help="colour jitter per channel + crop shift +-8 px")
    ap.add_argument("--reg", action="store_true",
                    help="shape-generalisation regulariser: RGB feature dropout 0.5, crop shift, stronger colour jitter")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ports = [p for p in C.PORTS if p != args.held_out]
    D, metas = load_shards(args.data, ports)
    u = (D["q"] / C.SIGMA).astype(np.float32)
    lab = id_labels(u, D["push"], D["kind"])
    n_cfg = len(metas)
    if args.mldg:
        val_cfg = set()  # all three training shapes are used; the final epoch is the model
    elif args.val_port:
        val_cfg = {i for i, m in enumerate(metas) if m["port"] == args.val_port}
    else:
        val_cfg = set(rng.choice(n_cfg, max(1, n_cfg // 10), replace=False).tolist())
    is_val = np.array([c in val_cfg for c in D["cfg"]])
    dev = "cuda"
    # data stay in (pinned) CPU memory; batches are moved to the GPU per step
    T = {"rgb": torch.as_tensor(D["rgb"]).pin_memory(), "pc": torch.as_tensor(D["pc"]).pin_memory(),
         "u": torch.as_tensor(u), "lab": torch.as_tensor(lab)}
    port_of = np.array([metas[c]["port"] for c in D["cfg"]])
    tr_idx = np.nonzero(~is_val)[0]
    va_idx = np.nonzero(is_val)[0]
    by_port = {p: tr_idx[port_of[tr_idx] == p] for p in ports if p != args.val_port}
    print(f"held-out {args.held_out}: {n_cfg} grasps, {len(tr_idx)} train / {len(va_idx)} val views"
          f"{' (MLDG)' if args.mldg else ''}", flush=True)
    mon = None
    if args.monitor_every:
        Dm, _ = load_shards(args.data, [args.held_out])
        mon = {"rgb": torch.as_tensor(Dm["rgb"]), "pc": torch.as_tensor(Dm["pc"]),
               "u": torch.as_tensor((Dm["q"] / C.SIGMA).astype(np.float32))}

    def batch(idx):
        idx_t = torch.as_tensor(idx)
        rgb = T["rgb"][idx_t].to(dev, non_blocking=True)
        pts = T["pc"][idx_t].to(dev, non_blocking=True).float()
        ub = T["u"][idx_t].to(dev, non_blocking=True)
        lb = T["lab"][idx_t].to(dev, non_blocking=True)
        if args.reg or args.strong_aug:
            g = torch.empty(len(idx), 1, 1, 3, device=dev).uniform_(0.7, 1.3)
            rgb = (rgb.float() * g + torch.randn(len(idx), 1, 1, 1, device=dev) * 10).clamp(0, 255)
            sx, sy = np.random.randint(-8, 9, 2)
            rgb = torch.roll(rgb, shifts=(int(sy), int(sx)), dims=(1, 2)).to(torch.uint8)
        else:
            g = torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(0.85, 1.15)
            rgb = (rgb.float() * g).clamp(0, 255).to(torch.uint8)
        pts[..., :3] += torch.randn_like(pts[..., :3]) * (0.0005 / 0.05)
        return rgb, pts, ub, lb

    def loss_fn(o, ub, lb):
        loss = F.smooth_l1_loss(o["q_pc"].float(), ub) + F.smooth_l1_loss(o["q_hat"].float(), ub)
        m = lb >= 0
        if m.any():
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(o["id_logit"].float()[m], lb[m])
        return loss

    model = AlignNet(pretrained=True, rgb_drop=0.5 if args.reg else 0.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(tr_idx) // args.bs
    total = args.epochs * steps_per_epoch
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    names = [n for n, _ in model.named_parameters()]
    history, best = [], None
    t0 = time.time()
    for ep in range(args.epochs):
        perm = np.random.permutation(tr_idx)
        tot, tot_te = 0.0, 0.0
        for s in range(steps_per_epoch):
            if args.mldg:
                # first-order MLDG: theta' = theta - alpha * grad L_tr(theta);
                # minimise L_tr(theta) + beta * L_te(theta'), gradient of
                # theta' w.r.t. theta taken as identity (first order)
                doms = list(by_port)
                te_dom = doms[np.random.randint(len(doms))]
                tr_pool = np.concatenate([by_port[d] for d in doms if d != te_dom])
                rgb, pts, ub, lb = batch(np.random.choice(tr_pool, args.bs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_tr = loss_fn(model(rgb, pts), ub, lb)
                params = [p for p in model.parameters()]
                grads = torch.autograd.grad(l_tr, params, retain_graph=True, allow_unused=True)
                virt = {n: (p - args.mldg_alpha * g.detach()) if g is not None else p
                        for n, p, g in zip(names, params, grads)}
                bufs = dict(model.named_buffers())
                rgb2, pts2, ub2, lb2 = batch(np.random.choice(by_port[te_dom], args.bs))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o2 = torch.func.functional_call(model, (virt, bufs), (rgb2, pts2))
                    l_te = loss_fn(o2, ub2, lb2)
                loss = l_tr + args.mldg_beta * l_te
                tot_te += l_te.detach()
            else:
                rgb, pts, ub, lb = batch(perm[s * args.bs:(s + 1) * args.bs])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(rgb, pts), ub, lb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach()
        res = {"epoch": ep, "loss": float(tot) / steps_per_epoch, "minutes": (time.time() - t0) / 60}
        if args.mldg:
            res["meta_test_loss"] = float(tot_te) / steps_per_epoch
        if len(va_idx):
            res.update(evaluate(model, T, va_idx))
            score = min(res["hat_near_t_med_mm"] + res["hat_near_r_med_deg"],
                        res["pc_near_t_med_mm"] + res["pc_near_r_med_deg"])
            if best is None or score < best:
                best = score
                torch.save(model.state_dict(), out / "model.pt")
        else:
            torch.save(model.state_dict(), out / "model.pt")  # final epoch is the model
        if mon is not None and ((ep + 1) % args.monitor_every == 0 or ep + 1 == args.epochs):
            res.update({f"MONITOR_test_shape_{k}": v for k, v in
                        evaluate(model, mon, np.arange(len(mon["u"]))).items()})
            torch.save(model.state_dict(), out / f"model_ep{ep + 1}.pt")
        history.append(res)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)

    # ---------------------------------------------------------------- field
    M = u[(D["push"] == 1) & ~is_val]
    box = np.array([15.0, 10.0, 10.0, 8.0, 8.0, 8.0])
    field, fit = fit_field(M, args.h, box, device=dev, seed=args.seed)
    torch.save(field.state_dict(), out / "field.pt")
    np.save(out / "id_set.npy", M)
    summary = {"held_out": args.held_out, "train_ports": ports, "n_grasps": n_cfg, "n_val_grasps": len(val_cfg),
               "n_id": int(len(M)), "id_rate_push": float((D["push"][D["push"] >= 0] == 1).mean()),
               "id_set_mean_sigma": M.mean(0).tolist(), "id_set_std_sigma": M.std(0).tolist(),
               "field_h": args.h, "field_fit": fit, "best_val_score": best, "history": history,
               "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done", json.dumps({k: summary[k] for k in ("n_id", "id_rate_push", "field_fit")}), flush=True)


if __name__ == "__main__":
    main()
