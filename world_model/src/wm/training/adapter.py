"""Joint adapter training (models/adapter.py).

L1 on standardised outputs; validation on ood_val reports the joint error in mrad, next to the
"joints reach the commanded targets" rule (the assumption of Ctrl-World's setting). The "blocked"
counters use a 5 mrad gap that random validation windows never reach (logged 0 in the v3 run); the
blocked case is checked on the contact trials by analysis/wm_checks.adapter_on_contact
(scripts/analysis/check_adapter.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import AdapterDataset
from ..models.adapter import IN_DIM, JointAdapter
from . import common

BLOCKED_GAP_RAD = 0.005


def stats(ds: AdapterDataset, n: int = 200_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs, ys = zip(*[(ds[int(i)]["x"].numpy(), ds[int(i)]["y"].numpy()) for i in idx])
    x, y = np.stack(xs), np.stack(ys)
    return x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6


@torch.no_grad()
def evaluate(model, loader, device, max_batches) -> dict:
    model.eval()
    err, err_b, rule, rule_b, n, nb = 0.0, 0.0, 0.0, 0.0, 0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        x, y = b["x"].to(device), b["y"].to(device)
        pred = model(x)
        cmd = x[:, IN_DIM - 56:]                                     # next 8 targets relative to now
        e = (pred - y).abs().view(-1, 8, 7).mean((1, 2)) * 1000      # mrad per window
        r = (cmd - y).abs().view(-1, 8, 7).mean((1, 2)) * 1000       # "joints = commands"
        blocked = (cmd - y).abs().view(-1, 8, 7).mean((1, 2)) > BLOCKED_GAP_RAD
        err, rule, n = err + float(e.sum()), rule + float(r.sum()), n + len(e)
        err_b, rule_b, nb = err_b + float(e[blocked].sum()), rule_b + float(r[blocked].sum()), nb + int(blocked.sum())
    model.train()
    return {"joint_err_mrad": err / max(n, 1), "follow_cmd_err_mrad": rule / max(n, 1),
            "blocked_windows": nb, "blocked_joint_err_mrad": err_b / max(nb, 1),
            "blocked_follow_cmd_err_mrad": rule_b / max(nb, 1)}


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    if entries is None:                      # one data root or several (demonstrations, contact exploration)
        roots = [cfg["data"]] if isinstance(cfg["data"], str) else cfg["data"]
        entries = [e for r in roots for e in data_index.load(r)]
    train_ds, val_ds = AdapterDataset(entries, "train"), AdapterDataset(entries, "ood_val")
    mk = lambda ds: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                               num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0)
    train_loader, val_loader = mk(train_ds), mk(val_ds)
    model = JointAdapter(**cfg["model"]).to(device)
    model.set_stats(*stats(train_ds))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log = common.Logger(run_dir)
    log(event="start", train_windows=len(train_ds), val_windows=len(val_ds))
    best, bad = float("inf"), 0
    steps = max_steps or cfg["steps"]
    for step, b in zip(range(1, steps + 1), common.cycle(train_loader)):
        x, y = b["x"].to(device), b["y"].to(device)
        loss = (model.forward_normalised(x) - (y - model.y_mean) / model.y_std).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg["log_every"] == 0:
            log(step=step, train_loss=float(loss))
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(model, val_loader, device, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), step=step, cfg=cfg)
            if not math.isfinite(v["joint_err_mrad"]):
                log(event="stop_nonfinite_validation", step=step)
                break
            if v["joint_err_mrad"] < best:
                best, bad = v["joint_err_mrad"], 0
                common.save_checkpoint(run_dir / "best.pt", model=model.state_dict(), step=step, cfg=cfg)
            else:
                bad += 1
                if bad >= cfg["patience"] and v["joint_err_mrad"] > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=v["joint_err_mrad"])
                    break
    return run_dir / "best.pt"


def load_adapter(path: str, device=None) -> JointAdapter:
    ck = torch.load(path, map_location="cpu")
    m = JointAdapter(**ck["cfg"]["model"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m.to(device) if device is not None else m
