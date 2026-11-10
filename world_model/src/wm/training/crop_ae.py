"""Stage 1 (v4): crop autoencoder (models/crop_ae.py), trained from scratch.

Loss, summed over the two crops:
    depth_mm    L1 (mm) where the camera returned a depth
    valid_bce   per-pixel "camera returned a depth" logit
    rgb_l1      RGB L1 ([0, 1])
    dino_cos    1 - cosine to the frozen DINOv2 feature of each token
    motion_mm   per plug pixel with depth (segmentation = loss mask only): L1 between the predicted
                motion and R* p + t* - p, (R*, t*) = FK of the grasp's one insertion
    weight_bce  per pixel with depth: weight logit vs "is plug"
and for the pose solved from both crops: reference-grid error (mm) + angle error (deg).
Validation on ood_val; model selection / harm-only early stop on pose score = grid + w_angle * angle.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import CropFrameDataset
from ..geometry import apply_rigid, insertion_motion, plug_region_points, pose_loss_terms
from ..models.crop_ae import MOTION_SCALE_M, CropAE, unpatch
from ..models.crop_encoder import cut
from ..models.encoder import DEPTH_SCALE_M
from . import common

PLUG_CLASS = 1


def losses(model: CropAE, out: dict, batch: dict, w: dict, ref: torch.Tensor) -> dict:
    l = {k: 0.0 for k in ("depth_mm", "valid_bce", "rgb_l1", "dino_cos", "motion_mm", "weight_bce")}
    with torch.autocast("cuda", enabled=False):
        R_star, t_star = insertion_motion(batch["hand"].float(), batch["h_star"].float(), batch["R_P"].float())
        geom = {k: batch[k] for k in ("K", "cam_T_world", "hand", "R_P", "plug_uv0", "hole_uv0")}
        for k in ("plug", "hole"):
            pred, tgt = out["pred"][k], out["target"][k]
            valid = tgt["valid_patch"] > 0
            d_err = (pred["depth_patch"].float() - tgt["depth_patch"].float()).abs() * DEPTH_SCALE_M * 1000
            l["depth_mm"] = l["depth_mm"] + (d_err * valid).sum() / valid.sum().clamp(min=1)
            l["valid_bce"] = l["valid_bce"] + F.binary_cross_entropy_with_logits(pred["valid_logit"].float(), valid.float())
            l["rgb_l1"] = l["rgb_l1"] + (pred["rgb_patch"].float() - tgt["rgb_patch"].float()).abs().mean()
            l["dino_cos"] = l["dino_cos"] + 1 - F.cosine_similarity(pred["dino"].float(), tgt["dino"].float(), dim=-1).mean()
            p = CropAE.crop_points(tgt["depth_patch"], geom, k)
            motion_star = apply_rigid(p, R_star, t_star) - p
            seg = cut(batch["seg"], batch[f"{k}_uv0"]).flatten(1)
            vflat = unpatch(tgt["valid_patch"].float(), 1).flatten(1) > 0
            plug = (seg == PLUG_CLASS) & vflat
            motion = unpatch(pred["motion_patch"].float(), 3).flatten(2).transpose(1, 2) * MOTION_SCALE_M
            err = (motion - motion_star).abs().sum(-1) * 1000
            l["motion_mm"] = l["motion_mm"] + (err * plug).sum() / plug.sum().clamp(min=1)
            logit = unpatch(pred["weight_logit"].float(), 1).flatten(1)
            l["weight_bce"] = l["weight_bce"] + (F.binary_cross_entropy_with_logits(logit, plug.float(), reduction="none")
                                                 * vflat).sum() / vflat.sum().clamp(min=1)
        grid, angle = pose_loss_terms(out["R"], out["t"], R_star, t_star, ref)
        # near the hole (remaining translation < 5 mm): the case v3 could not read; lateral = across the axis
        near = t_star.norm(dim=-1) < 0.005
        lat_err = (out["t"].float()[:, 1:] - t_star[:, 1:]).norm(dim=-1) * 1000
        l["near_frac"] = near.float().mean()
        l["near_grid_mm"] = (grid * near).sum() / near.sum().clamp(min=1)
        l["near_lateral_err_mm"] = (lat_err * near).sum() / near.sum().clamp(min=1)
        l["near_angle_deg"] = (angle * near).sum() / near.sum().clamp(min=1)
        l["lateral_err_mm"] = lat_err.mean()
    l["grid_mm"], l["angle_deg"] = grid.mean(), angle.mean()
    l["pose_score"] = l["grid_mm"] + w["angle"] * l["angle_deg"]
    l["total"] = (w["depth"] * l["depth_mm"] + w["valid"] * l["valid_bce"] + w["rgb"] * l["rgb_l1"]
                  + w["dino"] * l["dino_cos"] + w["motion"] * l["motion_mm"] + w["weight"] * l["weight_bce"]
                  + w["grid"] * l["grid_mm"] + w["angle"] * l["angle_deg"])
    return l


@torch.no_grad()
def evaluate(model, loader, w, device, max_batches, ref) -> dict:
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model, model(batch), batch, w, ref)
        nf = float(l["near_frac"])
        for k, v in l.items():
            wk = nf if (k.startswith("near_") and k != "near_frac") else 1.0     # near-hole metrics: per near sample
            acc[k] = acc.get(k, 0.0) + float(v) * wk
            acc["_w_" + k] = acc.get("_w_" + k, 0.0) + wk
        n += 1
    model.train()
    return {k: v / max(acc["_w_" + k], 1e-9) for k, v in acc.items() if not k.startswith("_w_")}


def train(cfg: dict, run_dir: Path, max_steps: int | None = None, resume: str | None = None) -> Path:
    """`resume`: a last.pt of the same design (model + optimizer + step); training continues from its step."""
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = data_index.load(cfg["data"])
    mk = lambda split: DataLoader(CropFrameDataset(entries, split, cfg["data"], cfg["crops"]), batch_size=cfg["batch_size"],
                                  shuffle=True, drop_last=True, num_workers=cfg["workers"],
                                  persistent_workers=cfg["workers"] > 0, pin_memory=True)
    train_loader, val_loader = mk("train"), mk("ood_val")
    model = CropAE(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(common.trainable(model), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    start = 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        for _ in range(start):                 # the warm-up schedule continues where it was
            sched.step()
    log, w, ref = common.Logger(run_dir), cfg["loss_weights"], plug_region_points(device)
    log(event="start", resume=resume, start_step=start, train_frames=len(train_loader.dataset),
        val_frames=len(val_loader.dataset), params_m=sum(p.numel() for p in common.trainable(model)) / 1e6)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, batch in zip(range(start + 1, steps + 1), common.cycle(train_loader)):
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model, model(batch), batch, w, ref)
        opt.zero_grad(set_to_none=True)
        l["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(common.trainable(model), cfg["grad_clip"])
        if not (torch.isfinite(l["total"]) and torch.isfinite(gn)):
            skipped += 1
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
