"""v6: the image encoder and the history world model trained together from scratch (decided with the user 2026-11-12).

    encoder (models/frame_encoder.py)   image -> feature map; trained by
        label        its label head: remaining motion to inserted (FK of the grasp's one insertion) and "inserted now",
                     on the newest seen frame of every sample -- the target a collapsed encoder cannot meet
        diffusion    through that frame's features in the world model's context
    target encoder                       a slowly following copy (EMA) of the encoder, no gradient: it gives the features of
                     the other context frames and of the target frame (what the world model has to predict). Without the
                     copy and the labels, "predict your own next features" is solved by making all features equal.
    world model (models/history_cdit.py, NWM's CDiT unchanged, feature map cut in 2 x 2 patches)
        diffusion    NWM's loss on the target frame's feature map, given the history (features of the seen slots,
                     proprioception of every slot), the commanded hand path and the time shift
        aux          from its final tokens: how the hand really moves from now to +0.2 s / +0.4 s (pose features;
                     replaces the joint adapter) and "inserted" at the target time (the judge of imagined
                     outcomes; an output, not a condition, so that imagined actions can change it)
    Vision masking, random time shifts, context noise: as wm4.py. Every kept frame is encoded at every step, so at most
    `max_frames` seen frames per sample (data/frame_history.py).

Validation on ood_val (held-out shapes and grasps), per source, +0.4 s: diffusion loss; the label head on real features
(error across the axis on frames within 3 mm of it, next to an image-blind guess); the label head on the world model's
sampled prediction, next to "nothing changes" (the newest seen features); the aux outputs. Model selection on the sum of
the two read-out errors; harm-only early stop.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.frame_history import RES_SCALE, FrameHistoryDataset
from ..data.history import HIST
from ..models.encoder import geometry_features
from ..models.frame_encoder import FrameEncoder, LabelHead, centre_lateral_mm, label_loss, label_targets, symexp
from ..models.history_cdit import HistoryCDiT, create_diffusion
from . import common
from .wm4 import mixed_batches

AUX_DIM = 19          # 2 x 9 pose features of the hand's motion (to +0.2 s, +0.4 s) and the "inserted at the target time" logit


def build(cfg: dict, device):
    m = cfg["encoder"]
    enc = FrameEncoder(ch=m["ch"], stride=m["stride"], arch=m["arch"]).to(device)
    head = LabelHead(m["ch"], enc.grid).to(device)
    wm = HistoryCDiT(latent_ch=m["ch"], size=cfg["size"], feature_map=enc.grid, patch=cfg["patch"], aux_dim=AUX_DIM).to(device)
    return enc, head, wm


def encode_history(enc, enc_t, b: dict, grad: bool = True):
    """-> z_hist (B, HIST, C, H, W) (zeros in unseen slots), the newest seen frame's features (B, C, H, W; with gradient
    through `enc` when grad), the target frame's features (target encoder, no gradient)."""
    B, Fm = b["frame_ok"].shape
    with torch.no_grad():
        ok = b["frame_ok"].clone()
        ok[:, 0] = False                                               # the newest frame goes through the trained encoder
        f_old = enc_t(b["frames_rgb"][ok], b["frames_depth"][ok])
        x = enc_t(b["tgt_rgb"], b["tgt_depth"])
    with torch.set_grad_enabled(grad):
        f_new = enc(b["frames_rgb"][:, 0], b["frames_depth"][:, 0])
    z = f_new.new_zeros(B, HIST, *f_new.shape[1:])
    bi = torch.arange(B, device=f_new.device)[:, None].expand(B, Fm)
    z[bi[ok], b["frame_slot"][ok]] = f_old.to(z.dtype)
    z[bi[:, 0], b["frame_slot"][:, 0]] = f_new
    return z, f_new, x.to(z.dtype)


def aux_losses(aux: torch.Tensor, b: dict) -> dict:
    aux = aux.float()
    ok = b["res_ok"]
    res = F.smooth_l1_loss(aux[ok, :18], b["res"][ok], beta=0.05) if ok.any() else aux.sum() * 0
    ins = F.binary_cross_entropy_with_logits(aux[:, 18], b["t_inserted"].float())
    return {"aux_res": res, "aux_inserted_bce": ins}


@torch.no_grad()
def evaluate(enc_t, head, wm, diffusion, sampler, loaders, device, max_batches) -> dict:
    out = {}
    for name, loader in loaders.items():
        acc, n = {}, 0
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            b = common.to_device(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, f_new, x = encode_history(enc_t, enc_t, b, grad=False)
            kw = dict(path=b["path"], rel_t=b["rel_t"], z_hist=z.float(), seen=b["seen"], proprio=b["proprio"], dt=b["dt"], avail=b["avail"])
            x = x.float()
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            vl = diffusion.training_losses(wm, x, t, kw)["loss"].mean()
            pred = sampler.p_sample_loop(wm.forward, x.shape, torch.randn_like(x), clip_denoised=False, model_kwargs=kw,
                                         progress=False, device=device)
            aux = wm.aux.float()
            _, _, tgt = label_targets(b["t_hand"], b["h_star"], b["R_P"])
            geom_t = geometry_features(b["t_cam"].float(), b["t_hand"].float(), b["R_P"].float())
            lat_t = symexp(tgt[:, 1:3])                                     # mm across the axis, reference centre
            close = lat_t.norm(dim=-1) < 3.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e_real, e_pred, e_copy = (centre_lateral_mm(head(v, geom_t), tgt) for v in (x, pred, f_new.float()))
            ok = b["res_ok"]
            adv_true = b["res"][ok][:, 9] / RES_SCALE * 10                  # mm along the axis: the hand's motion in 0.4 s
            adv_pred = aux[ok][:, 9] / RES_SCALE * 10
            lead = b["path"][ok][:, 13 * 9] * 10                            # mm: the decision's last command ahead of the hand now
            held = (adv_true.abs() < 0.05) & (lead > 0.1)                   # pressed: the command leads, the hand stays
            ins = b["t_inserted"]
            m = {"diffusion_loss": float(vl), "close_frac": float(close.float().mean()),
                 "real_close_lateral_mm": float((e_real * close).sum() / close.sum().clamp(min=1)),
                 "pred_close_lateral_mm": float((e_pred * close).sum() / close.sum().clamp(min=1)),
                 "copy_close_lateral_mm": float((e_copy * close).sum() / close.sum().clamp(min=1)),
                 "blind_close_lateral_mm": float(((lat_t - lat_t[close].mean(0)).norm(dim=-1) * close).sum() / close.sum().clamp(min=1)),
                 "aux_lag_err_mm": float((adv_pred - adv_true).abs().mean()) if ok.any() else float("nan"),
                 "aux_lag_true_abs_mm": float(adv_true.abs().mean()) if ok.any() else float("nan"),
                 "aux_held_frac": float(held.float().mean()) if ok.any() else float("nan"),
                 "aux_held_pred_advance_mm": float(adv_pred[held].mean()) if held.any() else float("nan"),
                 "aux_held_pred_abs_mm": float(adv_pred[held].abs().mean()) if held.any() else float("nan"),
                 "aux_inserted_acc": float(((aux[:, 18] > 0) == ins).float().mean()), "inserted_frac": float(ins.float().mean()),
                 "feature_std_over_samples": float(x.std(0).mean())}
            for k, v in m.items():
                if math.isfinite(v):
                    acc[k] = acc.get(k, 0.0) + v
                    acc["_n_" + k] = acc.get("_n_" + k, 0) + 1
            n += 1
        for k, v in acc.items():
            if not k.startswith("_n_"):
                out[f"{name}/{k}"] = v / acc["_n_" + k]
    sel = [out[k] + out[k.replace("real_", "pred_")] for k in out if k.endswith("/real_close_lateral_mm")]
    out["selection"] = sum(sel) / len(sel) if sel else float("nan")
    return out


def loaders(cfg: dict):
    mk = lambda d, split, workers, **kw: DataLoader(
        FrameHistoryDataset(d, d + "_ee", split, max_frames=cfg["max_frames"], **kw), batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    train = [mk(d, "train", w, mask=cfg["mask"], p_step=cfg["p_step"]) for d, w in zip(cfg["data"], cfg["workers"])]
    val = {Path(d).name: mk(d, "ood_val", 3, mask=None, fixed_shift=2) for d in cfg["data"]}
    return mixed_batches(train, cfg["source_weights"], cfg["seed"] + 5), val


def train(cfg: dict, run_dir: Path, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    enc, head, wm = build(cfg, device)
    enc_t = common.EMA(enc, cfg["encoder_ema"])
    wm_ema = common.EMA(wm, cfg["ema_decay"])
    params = list(enc.parameters()) + list(head.parameters()) + list(wm.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    diffusion = create_diffusion(timestep_respacing="")
    sampler = create_diffusion(timestep_respacing=str(cfg["eval_sampling_steps"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    batches, vals = loaders(cfg)
    log, w = common.Logger(run_dir), cfg["loss_weights"]
    log(event="start", encoder=cfg["encoder"], feature_map=enc.grid, tokens=wm.n_tokens, size=cfg["size"], history_slots=HIST,
        params_m={k: round(sum(p.numel() for p in m.parameters()) / 1e6, 2) for k, m in (("encoder", enc), ("label_head", head), ("world_model", wm))})
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    enc.train(), head.train(), wm.train()
    for step in range(1, steps + 1):
        b = common.to_device(next(batches), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, f_new, x = encode_history(enc, enc_t.model, b)
            pred = head(f_new, geometry_features(b["g_cam"].float(), b["g_hand"].float(), b["R_P"].float()))
        _, _, tgt = label_targets(b["g_hand"], b["h_star"], b["R_P"])
        ll = label_loss(pred, tgt, b["g_inserted"], close_weight=cfg.get("close_weight", 0.0))
        z, x = z.float(), x.float()
        B = x.shape[0]
        sigma = torch.rand(B, device=device) * cfg["ctx_noise_max"]
        sigma = torch.where(torch.rand(B, device=device) < cfg["p_ctx_noise"], sigma, torch.zeros_like(sigma))
        z = z + sigma.view(B, 1, 1, 1, 1) * torch.randn_like(z) * b["seen"][..., None, None, None]
        t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
        kw = dict(path=b["path"], rel_t=b["rel_t"], z_hist=z, seen=b["seen"], proprio=b["proprio"], dt=b["dt"], avail=b["avail"], ctx_noise=sigma)
        diff = diffusion.training_losses(wm, x, t, kw)["loss"].mean()
        al = aux_losses(wm.aux, b)
        loss = diff + w["label"] * ll["label"] + w["aux_res"] * al["aux_res"] + w["aux_inserted"] * al["aux_inserted_bce"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            skipped += 1
            log(step=step, event="skipped_nonfinite_step", skipped=skipped)
            if skipped >= cfg.get("max_skipped", 50):
                log(event="stop_nonfinite", step=step)
                break
            continue
        opt.step()
        sched.step()
        enc_t.update(enc)
        wm_ema.update(wm)
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, loss=float(loss), diffusion=float(diff), grad_norm=float(gn), lr=sched.get_last_lr()[0],
                feature_std_over_samples=float(x.std(0).mean()), **{k: float(v) for k, v in {**ll, **al}.items()})
        state = lambda: dict(encoder=enc_t.model.state_dict(), label_head=head.state_dict(), wm=wm_ema.model.state_dict(), step=step, cfg=cfg)
        if step % cfg["save_every"] == 0:
            common.save_checkpoint(run_dir / "current.pt", **state())
        if step % cfg["eval_every"] == 0 or step == steps:
            head.eval()
            v = evaluate(enc_t.model, head, wm_ema.model, diffusion, sampler, vals, device, cfg["eval_batches"])
            head.train()
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", **state(), encoder_online=enc.state_dict(), wm_online=wm.state_dict(),
                                   opt=opt.state_dict())
            key = v["selection"]
            if not math.isfinite(key):
                log(event="stop_nonfinite_validation", step=step)
                break
            if key < best:
                best, bad = key, 0
                common.save_checkpoint(run_dir / "best.pt", **state())
            else:
                bad += 1
                if cfg.get("patience") and bad >= cfg["patience"] and key > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=key)
                    break
    return run_dir / "best.pt"


def load_joint(path: str, device=None):
    """-> (encoder, label head, world model) of a checkpoint (the slowly following copies), in evaluation mode."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, head, wm = build(ck["cfg"], torch.device("cpu"))
    enc.load_state_dict(ck["encoder"]), head.load_state_dict(ck["label_head"]), wm.load_state_dict(ck["wm"])
    out = [m.eval().to(device) if device is not None else m.eval() for m in (enc, head, wm)]
    return (*out, ck)
