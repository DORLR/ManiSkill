"""Stage 1: token autoencoder.

Loss = Chamfer(plug) + Chamfer(port) [mm] + w_pose * plug-point loss [mm]
       + w_angle * insertion-angle error [deg] + w_joint * joint L1 [mrad]
(v1 had + beta * KL and no angle term; see docs/worklog.md, 2026-11-07.)
Model selection and the harm-only early stop use the OOD validation split, on
plug-point loss + w_angle * angle (the two pose terms).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import FrameDataset
from ..models.vae import TokenVAE
from . import common
from .losses import chamfer_mm, insertion_angle_loss_deg, plug_point_loss_mm


def losses(out: dict, batch: dict, w: dict) -> dict:
    l = {"chamfer_plug": chamfer_mm(out["plug_points"], out["plug_xyz"], out["plug_mask"]),
         "chamfer_port": chamfer_mm(out["port_points"], out["port_xyz"], out["port_mask"]),
         "plug_point": plug_point_loss_mm(out["R"], out["t"], out["plug_xyz"], out["plug_mask"],
                                          batch["hand"], batch["h_star"], batch["R_P"]),
         "angle_deg": insertion_angle_loss_deg(out["R"], batch["hand"], batch["h_star"], batch["R_P"]),
         "joint_mrad": (out["joints"].float() - batch["qpos"][:, :7].float()).abs().mean() * 1000}
    l["pose_score"] = l["plug_point"] + w.get("angle", 0.0) * l["angle_deg"]
    l["total"] = (l["chamfer_plug"] + l["chamfer_port"] + w["pose"] * l["plug_point"]
                  + w.get("angle", 0.0) * l["angle_deg"] + w["joint"] * l["joint_mrad"])
    return l


@torch.no_grad()
def evaluate(model, loader, w, device, max_batches: int) -> dict:
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model(batch), batch, w)
        for k, v in l.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def build(cfg: dict, entries: list[dict] | None = None):
    entries = entries if entries is not None else data_index.load(cfg["data"])
    train_ds = FrameDataset(entries, "train")
    val_ds = FrameDataset(entries, "ood_val")
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=shuffle,
                                        num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0)
    return TokenVAE(**cfg["model"]), mk(train_ds, True), mk(val_ds, True)


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    model, train_loader, val_loader = build(cfg, entries)
    model.to(device)
    opt = torch.optim.AdamW(common.trainable(model), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log, w = common.Logger(run_dir), cfg["loss_weights"]
    log(event="start", train_frames=len(train_loader.dataset), val_frames=len(val_loader.dataset),
        params_m=sum(p.numel() for p in common.trainable(model)) / 1e6)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, batch in zip(range(1, steps + 1), common.cycle(train_loader)):
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model(batch), batch, w)
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
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, grad_norm=float(gn), **{f"train_{k}": float(v) for k, v in l.items()})
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(model, val_loader, w, device, cfg["eval_batches"])
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
