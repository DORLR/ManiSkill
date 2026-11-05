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


def evaluate(model, T, idx, bs=512):
    model.eval()
    out_pc, out_hat = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            o = model(T["rgb"][b], T["pc"][b].float())
            out_pc.append(o["q_pc"].float())
            out_hat.append(o["q_hat"].float())
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
    if args.val_port:
        val_cfg = {i for i, m in enumerate(metas) if m["port"] == args.val_port}
    else:
        val_cfg = set(rng.choice(n_cfg, max(1, n_cfg // 10), replace=False).tolist())
    is_val = np.array([c in val_cfg for c in D["cfg"]])
    dev = "cuda"
    T = {"rgb": torch.as_tensor(D["rgb"], device=dev), "pc": torch.as_tensor(D["pc"], device=dev),
         "u": torch.as_tensor(u, device=dev), "lab": torch.as_tensor(lab, device=dev)}
    tr_idx = torch.as_tensor(np.nonzero(~is_val)[0], device=dev)
    va_idx = torch.as_tensor(np.nonzero(is_val)[0], device=dev)
    print(f"held-out {args.held_out}: {n_cfg} grasps, {len(tr_idx)} train / {len(va_idx)} val views", flush=True)

    model = AlignNet(pretrained=True, rgb_drop=0.5 if args.reg else 0.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(tr_idx) // args.bs
    total = args.epochs * steps_per_epoch
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    history, best = [], None
    t0 = time.time()
    for ep in range(args.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for s in range(steps_per_epoch):
            b = perm[s * args.bs:(s + 1) * args.bs]
            rgb = T["rgb"][b]
            # photometric jitter (brightness/contrast) and point jitter (0.5 mm)
            if args.reg:
                g = torch.empty(len(b), 1, 1, 3, device=dev).uniform_(0.7, 1.3)
                rgb = (rgb.float() * g + torch.randn(len(b), 1, 1, 1, device=dev) * 10).clamp(0, 255)
                sx, sy = np.random.randint(-8, 9, 2)
                rgb = torch.roll(rgb, shifts=(int(sy), int(sx)), dims=(1, 2)).to(torch.uint8)
            else:
                g = torch.empty(len(b), 1, 1, 1, device=dev).uniform_(0.85, 1.15)
                rgb = (rgb.float() * g).clamp(0, 255).to(torch.uint8)
            pts = T["pc"][b].float()
            pts[..., :3] += torch.randn_like(pts[..., :3]) * (0.0005 / 0.05)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(rgb, pts)
            ub, lb = T["u"][b], T["lab"][b]
            loss = F.smooth_l1_loss(o["q_pc"].float(), ub) + F.smooth_l1_loss(o["q_hat"].float(), ub)
            m = lb >= 0
            if m.any():
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(o["id_logit"].float()[m], lb[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach()
        res = evaluate(model, T, va_idx)
        res.update(epoch=ep, loss=float(tot) / steps_per_epoch, minutes=(time.time() - t0) / 60)
        history.append(res)
        score = min(res["hat_near_t_med_mm"] + res["hat_near_r_med_deg"],
                    res["pc_near_t_med_mm"] + res["pc_near_r_med_deg"])
        if best is None or score < best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")
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
