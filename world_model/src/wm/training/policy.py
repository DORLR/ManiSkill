"""FM policy (behaviour cloning of successful expert trajectories), on the shared latent or, as a
baseline, latent-free: the VAE's encoder architecture trained end to end by the policy loss
(cfg["input"] = "latent" | "image"; same data, context, actions and head).

Actions are normalised per dimension with statistics of the train split (stored in the
checkpoint). Validation (OOD split): flow-matching loss, and the error of a sampled chunk
against the expert's chunk (mrad, mean over the 16 x 7 targets). Closed-loop success is
measured separately (wm.eval.rollout).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import ImagePolicyDataset, LatentPolicyDataset
from ..models.image_policy import OBS_KEYS, ImageFMPolicy
from ..models.policy import LatentFMPolicy
from . import common


def action_stats(ds, n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    get = getattr(ds, "action", None) or (lambda k: ds[k]["a"].numpy())   # image dataset: skip decoding images
    a = np.stack([get(int(i)) for i in idx])
    return a.mean(0), a.std(0) + 1e-4


def _obs(b: dict, device):
    """Policy input of a batch: latents (B, 4, 64, C), or a dict of images and joints."""
    if "z" in b:
        return b["z"].to(device, non_blocking=True)
    return {k: b[k].to(device, non_blocking=True) for k in OBS_KEYS}


@torch.no_grad()
def evaluate(model, loader, mean, std, device, max_batches) -> dict:
    model.eval()
    fl, err, n = 0.0, 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        z, a = _obs(b, device), b["a"].to(device)
        an = (a - mean) / std
        fl += float(model.loss(z, an))
        pred = model.sample(z)[:, 0] * std + mean
        err += float((pred - a).abs().mean() * 1000)
        n += 1
    model.train()
    return {"flow_loss": fl / max(n, 1), "action_err_mrad": err / max(n, 1)}


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = entries if entries is not None else data_index.load(cfg["data"])
    if cfg.get("input", "latent") == "image":
        scale = 1.0
        train_ds, val_ds = ImagePolicyDataset(entries, "train"), ImagePolicyDataset(entries, "ood_val")
        model = ImageFMPolicy(**cfg["model"]).to(device)
    else:
        scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
        train_ds = LatentPolicyDataset(cfg["latents"], entries, "train", cfg["data"], scale)
        val_ds = LatentPolicyDataset(cfg["latents"], entries, "ood_val", cfg["data"], scale)
        model = LatentFMPolicy(**cfg["model"]).to(device)
    mk = lambda ds: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                               num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0,
                               pin_memory=True)
    train_loader, val_loader = mk(train_ds), mk(val_ds)
    m, s = action_stats(train_ds)
    mean, std = torch.tensor(m, device=device), torch.tensor(s, device=device)
    ema = common.EMA(model, cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log = common.Logger(run_dir)
    log(event="start", train_samples=len(train_ds), val_samples=len(val_ds),
        params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, b in zip(range(1, steps + 1), common.cycle(train_loader)):
        z, a = _obs(b, device), b["a"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(z, (a - mean) / std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            skipped += 1
            log(step=step, event="skipped_nonfinite_step", skipped=skipped)
            if skipped >= cfg.get("max_skipped", 50):
                log(event="stop_nonfinite", step=step)
                break
            continue
        opt.step()
        ema.update(model)
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, train_loss=float(loss))
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, val_loader, mean, std, device, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            state = dict(ema=ema.model.state_dict(), mean=m, std=s, scale=scale, step=step, cfg=cfg)
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), opt=opt.state_dict(), **state)
            if not math.isfinite(v["action_err_mrad"]):
                log(event="stop_nonfinite_validation", step=step)
                break
            if v["action_err_mrad"] < best:
                best, bad = v["action_err_mrad"], 0
                common.save_checkpoint(run_dir / "best.pt", **state)
            else:
                bad += 1
                if bad >= cfg["patience"] and v["action_err_mrad"] > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=v["action_err_mrad"])
                    break
    return run_dir / "best.pt"
