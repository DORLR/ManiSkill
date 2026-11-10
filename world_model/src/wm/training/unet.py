"""Stage 0 (v4): segmentation UNet (perception/unet.py) on the wrist RGB.

Loss = weighted cross-entropy over the 7 classes (pins and openings up-weighted: they are a few
hundred pixels of 102 400) + soft Dice of pins and openings.

Validation on ood_val (held-out shapes), every eval_every steps:
    IoU per class
    for pins and openings, per image: present in the label (>= MIN_PX pixels) vs predicted (>= MIN_PX),
    and when both, the distance between the label's and the prediction's mask centres (px) -- the
    crops are centred there, so this is what matters downstream
Model selection and the harm-only early stop: mean of the pins and opening IoU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import SegDataset
from ..perception.labels import CLASSES, OPENING, PINS
from ..perception.unet import UNet
from . import common

MIN_PX = 10


def dice(logits: torch.Tensor, label: torch.Tensor, cls: int) -> torch.Tensor:
    p = logits.float().softmax(1)[:, cls]
    t = (label == cls).float()
    inter = (p * t).sum((1, 2))
    den = p.sum((1, 2)) + t.sum((1, 2))
    present = t.sum((1, 2)) > 0
    d = 1 - (2 * inter + 1) / (den + 1)
    return d[present].mean() if present.any() else d.mean() * 0


def centres(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """mask (B, H, W) bool -> centre (B, 2) in (u, v) pixels and pixel count (B,)."""
    B, H, W = mask.shape
    v, u = torch.meshgrid(torch.arange(H, device=mask.device), torch.arange(W, device=mask.device), indexing="ij")
    m = mask.float()
    n = m.sum((1, 2))
    c = torch.stack([(m * u).sum((1, 2)), (m * v).sum((1, 2))], 1) / n.clamp(min=1)[:, None]
    return c, n


@torch.no_grad()
def evaluate(model, loader, device, max_batches) -> dict:
    model.eval()
    inter = torch.zeros(len(CLASSES), device=device)
    union = torch.zeros(len(CLASSES), device=device)
    stats = {c: {"err": [], "tp": 0, "fn": 0, "fp": 0, "tn": 0} for c in ("pins", "opening")}
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = common.to_device(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(b["rgb"]).argmax(1)
        lab = b["label"]
        for c in range(len(CLASSES)):
            inter[c] += ((pred == c) & (lab == c)).sum()
            union[c] += ((pred == c) | (lab == c)).sum()
        for name, c in (("pins", PINS), ("opening", OPENING)):
            ct, nt = centres(lab == c)
            cp, npred = centres(pred == c)
            has_t, has_p = nt >= MIN_PX, npred >= MIN_PX
            s = stats[name]
            s["tp"] += int((has_t & has_p).sum())
            s["fn"] += int((has_t & ~has_p).sum())
            s["fp"] += int((~has_t & has_p).sum())
            s["tn"] += int((~has_t & ~has_p).sum())
            both = has_t & has_p
            s["err"] += (ct[both] - cp[both]).norm(dim=1).tolist()
    model.train()
    iou = (inter / union.clamp(min=1)).tolist()
    out = {f"iou_{c}": iou[k] for k, c in enumerate(CLASSES)}
    for name, s in stats.items():
        e = np.asarray(s["err"]) if s["err"] else np.array([np.nan])
        out.update({f"{name}_centre_px_median": float(np.median(e)), f"{name}_centre_px_p90": float(np.percentile(e, 90)),
                    f"{name}_recall": s["tp"] / max(1, s["tp"] + s["fn"]),
                    f"{name}_false_pos_rate": s["fp"] / max(1, s["fp"] + s["tn"])})
    out["score"] = (out["iou_pins"] + out["iou_opening"]) / 2
    return out


def train(cfg: dict, run_dir: Path, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = data_index.load(cfg["data"])
    mk = lambda split, aug: DataLoader(SegDataset(entries, split, cfg["data"], cfg["label_geometry"], aug),
                                       batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                       num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0, pin_memory=True)
    train_loader, val_loader = mk("train", True), mk("ood_val", False)
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = cfg["warmup_steps"]
    steps = max_steps or cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    cw = torch.ones(len(CLASSES), device=device)
    cw[PINS], cw[OPENING] = cfg["w_pins"], cfg["w_opening"]
    log = common.Logger(run_dir)
    log(event="start", train_images=len(train_loader.dataset), val_images=len(val_loader.dataset))
    best, bad = -1.0, 0
    for step, b in zip(range(1, steps + 1), common.cycle(train_loader)):
        b = common.to_device(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["rgb"])
        ce = F.cross_entropy(logits.float(), b["label"], weight=cw)
        dl = dice(logits, b["label"], PINS) + dice(logits, b["label"], OPENING)
        loss = ce + cfg["w_dice"] * dl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            log(step=step, event="skipped_nonfinite_step")
            continue
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            log(step=step, ce=float(ce), dice=float(dl), lr=sched.get_last_lr()[0])
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(model, val_loader, device, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), step=step, cfg=cfg)
            if v["score"] > best:
                best, bad = v["score"], 0
                common.save_checkpoint(run_dir / "best.pt", model=model.state_dict(), step=step, cfg=cfg, val=v)
            else:
                bad += 1
                if bad >= cfg["patience"] and v["score"] < best * (1 - cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=v["score"])
                    break
    return run_dir / "best.pt"
