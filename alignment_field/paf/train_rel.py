"""Train the relational plug-port network (paf/relnet.py) on packed raw RGB-D.

python -m paf.train_rel --packed data/packed_v4 --val-ports G46 G47 G48 \
    --monitor-packed data/packed_abcd --out runs_rel/main --epochs 30
Training shapes = all shapes in --packed except --val-ports (shape-held-out
model selection). --monitor-packed (the A-D test shapes) is evaluated every
--monitor-every epochs for the record only; it never influences the model.

Reported every epoch (val shapes, and monitor shapes when logged):
  near error      median |u_hat - u| over views whose TRUE state is within 3 sigma
  all error       median over all views
  false arrival   among views where the network believes it is within 1 sigma
                  (1 mm) of the target, the fraction that are actually more than
                  1.5 mm away -- the failure that broke v3 in closed loop
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .field import fit_field
from .relnet import RelNet


class Packed(Dataset):
    def __init__(self, root, keep_ports=None, drop_ports=None):
        self.root = Path(root)
        meta = json.loads((self.root / "ports.json").read_text())
        self.grasp_port = meta["grasp_port"]
        self.rgb = np.load(self.root / "rgb.npy", mmap_mode="r")
        self.depth = np.load(self.root / "depth.npy", mmap_mode="r")
        self.small = {k: np.load(self.root / f"{k}.npy") for k in
                      ("K", "cam_T_world", "H", "R_P", "u", "push", "kind", "grasp")}
        port = np.array(self.grasp_port)[self.small["grasp"]]
        m = np.ones(len(port), bool)
        if keep_ports is not None:
            m &= np.isin(port, keep_ports)
        if drop_ports:
            m &= ~np.isin(port, drop_ports)
        self.idx = np.nonzero(m)[0]
        self.port = port

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        s = self.small
        return (np.asarray(self.rgb[j]), np.asarray(self.depth[j]), s["K"][j], s["cam_T_world"][j],
                s["H"][j], s["R_P"][j], s["u"][j], np.float32(s["push"][j]))


def to_dev(batch, dev):
    return [torch.as_tensor(x).to(dev, non_blocking=True) for x in batch]


def metrics(model, loader, dev, max_batches=None):
    model.eval()
    est, tru = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            rgb, depth, K, T, H, R, u, _ = to_dev(batch, dev)
            est.append(model(rgb, depth, K.float(), T.float(), H.float(), R.float())["q_hat"].float().cpu())
            tru.append(u.float().cpu())
    model.train()
    e, t = torch.cat(est), torch.cat(tru)
    err = (e - t).abs()
    near = t.norm(dim=1) < 3.0
    believe = e[:, :3].norm(dim=1) < 1.0
    res = {"near_t_med_mm": float(err[near][:, :3].median()) if near.any() else float("nan"),
           "near_r_med_deg": float(err[near][:, 3:].median()) if near.any() else float("nan"),
           "all_t_med_mm": float(err[:, :3].median()), "all_t_p90_mm": float(err[:, :3].quantile(0.9)),
           "believed_arrivals": int(believe.sum()),
           "false_arrival": float((t[believe][:, :3].norm(dim=1) > 1.5).float().mean()) if believe.any()
           else float("nan")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--val-ports", nargs="+", default=[])
    ap.add_argument("--train-ports", nargs="+", default=None)
    ap.add_argument("--monitor-packed", default=None)
    ap.add_argument("--monitor-every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after this many steps")
    ap.add_argument("--h", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    tr = Packed(args.packed, keep_ports=args.train_ports, drop_ports=args.val_ports)
    va = Packed(args.packed, keep_ports=args.val_ports) if args.val_ports else None
    mon = Packed(args.monitor_packed) if args.monitor_packed else None
    mk = lambda ds, sh: DataLoader(ds, args.bs, shuffle=sh, num_workers=args.workers, pin_memory=True,
                                   drop_last=sh, persistent_workers=args.workers > 0)
    tl = mk(tr, True)
    vl = mk(va, False) if va is not None and len(va) else None
    ml = mk(mon, False) if mon is not None else None
    print(f"train views {len(tr)}, val views {len(va) if va else 0}, monitor views {len(mon) if mon else 0}",
          flush=True)
    model = RelNet(use_dino=not args.no_dino).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    hist, best, step, t0 = [], None, 0, time.time()
    for ep in range(args.epochs):
        tot = 0.0
        for batch in tl:
            rgb, depth, K, T, H, R, u, push = to_dev(batch, dev)
            # photometric jitter on the RGB (augmentation of recorded images)
            g = torch.empty(len(rgb), 1, 1, 3, device=dev).uniform_(0.8, 1.2)
            rgb = (rgb.float() * g).clamp(0, 255).to(torch.uint8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(rgb, depth, K.float(), T.float(), H.float(), R.float())
            loss = F.smooth_l1_loss(o["q_hat"].float(), u.float())
            # in-distribution label: push outcome; far views (> 4 sigma) count as failures; others unknown
            lab = torch.where((push < 0) & (u.norm(dim=1) > 4), torch.zeros_like(push), push)
            m = lab >= 0
            if m.any():
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(o["id_logit"].float()[m], lab[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        res = {"epoch": ep, "loss": tot / max(1, len(tl)), "minutes": (time.time() - t0) / 60}
        if vl is not None:
            res.update({f"val_{k}": v for k, v in metrics(model, vl, dev).items()})
            score = res["val_near_t_med_mm"] + res["val_near_r_med_deg"] + 5 * res["val_false_arrival"]
            if best is None or score < best:
                best = score
                torch.save(model.state_dict(), out / "model.pt")
        else:
            torch.save(model.state_dict(), out / "model.pt")
        if ml is not None and ((ep + 1) % args.monitor_every == 0 or ep + 1 == args.epochs or args.max_steps):
            res.update({f"MONITOR_{k}": v for k, v in metrics(model, ml, dev).items()})
        hist.append(res)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    # the field is unchanged: fit to the ID set (push successes of training shapes)
    s = tr.small
    sel = tr.idx[s["push"][tr.idx] == 1]
    M = s["u"][sel]
    field, fit = fit_field(M, args.h, np.array([15.0, 10.0, 10.0, 8.0, 8.0, 8.0]), device=dev)
    torch.save(field.state_dict(), out / "field.pt")
    np.save(out / "id_set.npy", M)
    (out / "summary.json").write_text(json.dumps({"history": hist, "n_id": int(len(M)), "field_fit": fit,
                                                  "args": vars(args)}, indent=1))
    print("done", json.dumps({"n_id": int(len(M)), "field_fit": fit}), flush=True)


if __name__ == "__main__":
    main()
