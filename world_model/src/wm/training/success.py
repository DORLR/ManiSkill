"""Stage 2 (v4): success classifier (models/success.py), from scratch, then online.

Samples: data/history.py SuccessDataset (the same vision masking as the world model: the classifier
must decide "inserted" when the last seconds show only the plug in front of the opening), label =
inserted at the current image. Positives are a few % of the images, so batches are drawn with
replacement at a fixed positive fraction (`pos_fraction`). Context noise as for the world model, so
imagined latents (world-model samples) are not out of distribution; the online learner also feeds
world-model predictions with the label of the real frame they predict.

Loss: binary cross-entropy. Validation on ood_val with the full history and with the last 2 s of
vision masked: AUC, and precision / recall at 0.5. Model selection: AUC with the last 2 s masked (the
hard case), harm-only early stop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.history import SuccessDataset
from ..models.success import SuccessHead
from . import common
from .wm4 import noisy_history, offline_files


def balanced_loader(ds: SuccessDataset, cfg: dict) -> DataLoader:
    lab = np.array([it[2] for it in ds.items])
    p = cfg["pos_fraction"]
    w = np.where(lab, p / max(1, lab.sum()), (1 - p) / max(1, (~lab).sum()))
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["workers"],
                      persistent_workers=cfg["workers"] > 0, pin_memory=True, drop_last=True)


class TailMaskedSuccess(SuccessDataset):
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


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model, loaders: dict, device, max_batches: int) -> dict:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        s, l = [], []
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            b = common.to_device(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s.append(model(b["z"], b["seen"], b["proprio"], b["dt"], b["avail"]).float().cpu())
            l.append(b["label"].cpu() > 0.5)
        s, l = torch.cat(s).numpy(), torch.cat(l).numpy()
        pred = s > 0
        out[f"{name}/auc"] = auc(s, l)
        out[f"{name}/precision"] = float((pred & l).sum() / max(1, pred.sum()))
        out[f"{name}/recall"] = float((pred & l).sum() / max(1, l.sum()))
        out[f"{name}/positives"] = int(l.sum())
    model.train()
    return out


def horizon_images(cfg: dict) -> int:
    return int(round(cfg.get("label_horizon_s", 0.0) / 0.2))


def val_loaders(cfg: dict, scale: float) -> dict:
    """Held-out samples per source: full history and the last 2 s of vision masked."""
    from .wm4 import sources
    out, h = {}, horizon_images(cfg)
    vcfg = {**cfg, "workers": 2}
    for src, _, _ in sources(cfg):
        files = offline_files(cfg, "ood_val", src)
        items = SuccessDataset.index(files, h)
        out[f"{src}:full"] = balanced_loader(SuccessDataset(files, scale, None, items=items, horizon=h), vcfg)
        out[f"{src}:tail2s"] = balanced_loader(TailMaskedSuccess(files, scale, items=items, tail=10, horizon=h), vcfg)
    return out


def train(cfg: dict, run_dir: Path, batches, max_steps: int | None = None, resume: str | None = None,
          extra=None) -> Path:
    """`batches`: iterator of training batches. `extra(model)`: optional, returns an additional batch
    (online: world-model predictions labelled with the real frame's outcome) or None."""
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    from .wm4 import latent_scale
    scale = latent_scale(cfg)
    model = SuccessHead(**cfg["model"]).to(device)
    ema = common.EMA(model, cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    start = 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.n = ck.get("ema_n", 10 ** 6)
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
    vals = val_loaders(cfg, scale)
    log = common.Logger(run_dir)
    log(event="start", resume=resume, start_step=start, params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    best, bad = -1.0, 0
    steps = max_steps or cfg["steps"]
    for step in range(start + 1, steps + 1):
        b = common.to_device(next(batches), device)
        bs = [b]
        if extra is not None:
            e = extra(ema.model)
            if e is not None:
                bs.append(common.to_device(e, device))
        loss = 0.0
        for bb in bs:
            z, _ = noisy_history(bb["z"], bb["seen"], cfg["p_ctx_noise"], cfg["ctx_noise_max"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(z, bb["seen"], bb["proprio"], bb["dt"], bb["avail"]).float()
            loss = loss + F.binary_cross_entropy_with_logits(logit, bb["label"].float()) / len(bs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            log(step=step, event="skipped_nonfinite_step")
            continue
        opt.step()
        ema.update(model)
        if step % cfg["log_every"] == 0:
            log(step=step, train_bce=float(loss), grad_norm=float(gn))
        if step % cfg["save_every"] == 0:
            common.save_checkpoint(run_dir / "current.pt", ema=ema.model.state_dict(), step=step, cfg=cfg)
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, vals, device, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(), ema_n=ema.n,
                                   opt=opt.state_dict(), step=step, cfg=cfg)
            aucs = [x for k, x in v.items() if k.endswith(":tail2s/auc") and math.isfinite(x)]
            key = sum(aucs) / len(aucs) if aucs else float("nan")        # mean over the sources, vision masked
            log(step=step, val_selection_auc=key)
            if math.isfinite(key) and key > best:
                best, bad = key, 0
                common.save_checkpoint(run_dir / "best.pt", ema=ema.model.state_dict(), step=step, cfg=cfg)
            else:
                bad += 1
                if cfg.get("patience") and bad >= cfg["patience"] and key < best * (1 - cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=key)
                    break
    return run_dir / "best.pt"


def load_success(path: str, device=None) -> SuccessHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SuccessHead(**ck["cfg"]["model"])
    m.load_state_dict(ck["ema"])
    m.eval()
    m.pos_fraction = ck["cfg"].get("pos_fraction")       # share of positives it was trained with (for calibration)
    return m.to(device) if device is not None else m
