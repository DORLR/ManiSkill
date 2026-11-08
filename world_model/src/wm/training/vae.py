"""Stage 1: autoencoder (v3), trained from scratch.

Loss = w_depth * depth L1 [mm] + w_valid * validity BCE + w_rgb * RGB L1 + w_dino * (1 - cos DINO)
       + w_motion * per-pixel motion L1 [mm] + w_weight * per-pixel weight BCE
       + w_grid * reference-grid error of the solved pose [mm] + w_angle * its angle error [deg]
(losses.py; weights in the config). No KL (deterministic latent). v3g: the pose is solved from per-pixel
motions by a weighted rigid fit (models/vae.py), not regressed.

Validation on ood_val (held-out shapes) every eval_every steps; model selection and the harm-only
early stop use the pose score = grid + w_angle * angle, the quantity the world model is read by.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import FrameDataset
from ..geometry import plug_region_points
from ..models.vae import TokenVAE
from . import common
from .losses import geometric_pose, reconstruction


def losses(out: dict, batch: dict, w: dict, ref: torch.Tensor) -> dict:
    l = reconstruction(out["pred"], out["target"])
    l.update(geometric_pose(out["pred"], out["target"], batch, ref))
    l["pose_score"] = l["grid_mm"] + w["angle"] * l["angle_deg"]
    l["total"] = (w["depth"] * l["depth_mm"] + w["valid"] * l["valid_bce"] + w["rgb"] * l["rgb_l1"]
                  + w["dino"] * l["dino_cos"] + w["motion"] * l["motion_mm"] + w["weight"] * l["weight_bce"]
                  + w["grid"] * l["grid_mm"] + w["angle"] * l["angle_deg"])
    return l


@torch.no_grad()
def evaluate(model, loader, w, device, max_batches: int, ref) -> dict:
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model(batch), batch, w, ref)
        for k, v in l.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = entries if entries is not None else data_index.load(cfg["data"])
    mk = lambda ds: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                               num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0, pin_memory=True)
    train_loader, val_loader = mk(FrameDataset(entries, "train")), mk(FrameDataset(entries, "ood_val"))
    model = TokenVAE(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(common.trainable(model), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    log, w, ref = common.Logger(run_dir), cfg["loss_weights"], plug_region_points(device)
    log(event="start", train_frames=len(train_loader.dataset), val_frames=len(val_loader.dataset),
        params_m=sum(p.numel() for p in common.trainable(model)) / 1e6)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, batch in zip(range(1, steps + 1), common.cycle(train_loader)):
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model(batch), batch, w, ref)
        opt.zero_grad(set_to_none=True)
        l["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(common.trainable(model), cfg["grad_clip"])
        if not (torch.isfinite(l["total"]) and torch.isfinite(gn)):
            skipped += 1                       # never apply a non-finite update
            log(step=step, event="skipped_nonfinite_step", skipped=skipped)
            if skipped >= cfg.get("max_skipped", 50):
                log(event="stop_nonfinite", step=step)
                break
            continue
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, grad_norm=float(gn), **{f"train_{k}": float(v) for k, v in l.items()})
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(model, val_loader, w, device, cfg["eval_batches"], ref)
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            if not math.isfinite(v["pose_score"]):
                log(event="stop_nonfinite_validation", step=step)
                break
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg)
            if v["pose_score"] < best:
                best, bad = v["pose_score"], 0
                common.save_checkpoint(run_dir / "best.pt", model=model.state_dict(), step=step, cfg=cfg)
            else:
                bad += 1
                if bad >= cfg["patience"] and v["pose_score"] > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=v["pose_score"])
                    break
    return run_dir / "best.pt"
