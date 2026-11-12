"""Training of the vision-conditioned joint adapter (models/adapter.py VisionJointAdapter), from scratch.

Samples: data/history.py AdapterHistoryDataset, demonstrations and contact exploration mixed by
`source_weights`, the world model's vision masking and context noise (in imagination the latents are the world
model's, one per 0.4 s; on the robot the last frames may be missing). Loss: L1 on the standardised outputs, as
for the blind adapter (training/adapter.py).

Validation on ood_val per source, with the full history and with the last 2 s of vision masked: joint error
(mrad) next to "the joints reach the commands"; the same on the `held` windows, where the arm did not follow
(a final target more than 0.5 mrad away, of which less than half was covered). Selection: mean joint error over
sources and both histories; harm-only early stop. Whether the image is used at the moment of contact is checked
on the hand's advance by scripts/analysis/check_adapter_decisions.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.history import AdapterHistoryDataset
from ..models.adapter import IN_DIM, VisionJointAdapter
from . import common
from .wm4 import latent_scale, noisy_history, offline_files, sources, train_batches

HISTORY_KEYS = ("z", "seen", "proprio", "dt", "avail", "path")


def model_args(b: dict) -> list:
    """The adapter's inputs after x: the history, the path and, if the batch has it, the outcome."""
    return [b[k] for k in HISTORY_KEYS] + ([b["outcome"]] if "outcome" in b else [])


def outcome_horizon(cfg: dict) -> int | None:
    """Images of the classifier's label horizon if the adapter takes the outcome, else None."""
    return int(round(cfg["label_horizon_s"] / 0.2)) if cfg["model"].get("outcome") else None
HELD_MIN_RAD = 0.0005


class TailMaskedAdapter(AdapterHistoryDataset):
    """Validation: the last `tail` seen slots without vision."""

    def __init__(self, *a, tail: int, **k):
        super().__init__(*a, mask=None, **k)
        self.tail = tail

    def __getitem__(self, k):
        out = super().__getitem__(k)
        seen = out["seen"].clone()
        idx = torch.nonzero(seen)[:, 0]
        if len(idx) > 1:
            seen[idx[-min(self.tail, len(idx) - 1):]] = False
            out["z"][~seen] = 0
            out["seen"] = seen
        return out


def io_stats(datasets: list[AdapterHistoryDataset], n: int = 100_000, seed: int = 0):
    """Standardisation statistics of the numeric input and the output (n windows per source)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for ds in datasets:
        for k in rng.choice(len(ds), size=min(n, len(ds)), replace=False):
            x, y = ds.io(int(k))
            xs.append(x), ys.append(y)
    x, y = np.stack(xs), np.stack(ys)
    return x.mean(0), x.std(0) + 1e-6, y.mean(0), y.std(0) + 1e-6


def window_errors(pred: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Per window (B,): joint error of the prediction and of "joints = commands" (mrad), and the `held` flag."""
    cmd = x[:, IN_DIM - 56:].view(-1, 8, 7)
    yv = y.view(-1, 8, 7)
    e = (pred.view(-1, 8, 7) - yv).abs().mean((1, 2)) * 1000
    r = (cmd - yv).abs().mean((1, 2)) * 1000
    size = cmd[:, -1].abs().max(1).values                                  # the final target's largest joint offset
    short = (cmd[:, -1] - yv[:, -1]).abs().max(1).values
    return {"err": e, "rule": r, "held": (size > HELD_MIN_RAD) & (short > 0.5 * size)}


@torch.no_grad()
def evaluate(model, loaders: dict, device, max_batches: int) -> dict:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        acc = {"err": [], "rule": [], "held": []}
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            b = common.to_device(b, device)
            w = window_errors(model(b["x"], *model_args(b)), b["x"], b["y"])
            for k in acc:
                acc[k].append(w[k].cpu())
        e, r, h = (torch.cat(acc[k]) for k in ("err", "rule", "held"))
        out[f"{name}/joint_err_mrad"] = float(e.mean())
        out[f"{name}/follow_cmd_err_mrad"] = float(r.mean())
        out[f"{name}/held_windows"] = int(h.sum())
        out[f"{name}/held_joint_err_mrad"] = float(e[h].mean()) if h.any() else float("nan")
        out[f"{name}/held_follow_cmd_err_mrad"] = float(r[h].mean()) if h.any() else float("nan")
    model.train()
    return out


def loader(ds, cfg: dict, workers: int | None = None) -> DataLoader:
    w = cfg["workers"] if workers is None else workers
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=w,
                      persistent_workers=w > 0, pin_memory=True)


def val_loaders(cfg: dict, scale: float) -> dict:
    out, h = {}, outcome_horizon(cfg)
    for src, _, _ in sources(cfg):
        files = offline_files(cfg, "ood_val", src)
        items = AdapterHistoryDataset.index(files)
        out[f"{src}:full"] = loader(AdapterHistoryDataset(files, scale, None, items=items, horizon=h), cfg, 2)
        out[f"{src}:tail2s"] = loader(TailMaskedAdapter(files, scale, items=items, tail=10, horizon=h), cfg, 2)
    return out


def train(cfg: dict, run_dir: Path, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    scale = latent_scale(cfg)
    train_sets = []

    def make(files):
        train_sets.append(AdapterHistoryDataset(files, scale, cfg["mask"], horizon=outcome_horizon(cfg)))
        return loader(train_sets[-1], cfg)

    batches = train_batches(cfg, make)
    model = VisionJointAdapter(**{k: v for k, v in cfg["model"].items() if k != "kind"}).to(device)
    model.set_stats(*io_stats(train_sets, seed=cfg["seed"]))
    if model.takes_outcome:                  # share of positive samples as drawn in training (sources mixed by weight)
        rng = np.random.default_rng(cfg["seed"])
        w = np.asarray(cfg.get("source_weights") or [1.0] * len(train_sets), float)
        pri = [np.mean([ds.outcome(int(k)) for k in rng.choice(len(ds), size=min(20000, len(ds)), replace=False)]) for ds in train_sets]
        model.outcome_prior.fill_(float((w / w.sum() * np.asarray(pri)).sum()))
    ema = common.EMA(model, cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    vals = val_loaders(cfg, scale)
    log = common.Logger(run_dir)
    log(event="start", train_windows=[len(d) for d in train_sets], params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        outcome_prior=float(model.outcome_prior) if model.takes_outcome else None)
    best, bad = float("inf"), 0
    steps = max_steps or cfg["steps"]
    for step in range(1, steps + 1):
        b = common.to_device(next(batches), device)
        z, _ = noisy_history(b["z"], b["seen"], cfg["p_ctx_noise"], cfg["ctx_noise_max"])
        pred = model.forward_normalised(b["x"], z, b["seen"], b["proprio"], b["dt"], b["avail"], b["path"], b.get("outcome"))
        loss = (pred - (b["y"] - model.y_mean) / model.y_std).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            log(step=step, event="skipped_nonfinite_step")
            continue
        opt.step()
        ema.update(model)
        if step % cfg["log_every"] == 0:
            log(step=step, train_loss=float(loss), grad_norm=float(gn))
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, vals, device, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", ema=ema.model.state_dict(), step=step, cfg=cfg)
            errs = [x for k, x in v.items() if k.endswith("/joint_err_mrad")]
            key = sum(errs) / len(errs)
            log(step=step, val_selection_joint_err_mrad=key)
            if not math.isfinite(key):
                log(event="stop_nonfinite_validation", step=step)
                break
            if key < best:
                best, bad = key, 0
                common.save_checkpoint(run_dir / "best.pt", ema=ema.model.state_dict(), step=step, cfg=cfg)
            else:
                bad += 1
                if bad >= cfg["patience"] and key > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=key)
                    break
    return run_dir / "best.pt"
