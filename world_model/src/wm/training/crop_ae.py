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
Validation on ood_val per data source; model selection / harm-only early stop on the pose score
(grid + w_angle * angle) averaged over the sources.

v5 (config keys `unet`, `key_weight`, loss weights `edge` / `lpips`, `selection`), why: measured on crop_ae_v4e,
the reconstructions lose the pins' edges and the openings. The RGB term was a plain pixel mean (the visible openings
are 120-190 of the crop's 16 384 pixels at the median, about 1 %; UNet masks on held-out frames) and about 4 % of the
total loss, whose gradient is clipped all the time.
    key pixels  the frozen UNet's predicted pins and openings (dilated), the same real-robot quantity that places
                the crops; their RGB error counts `1 + key_weight` times (rgb_w; rgb_l1 stays the plain mean)
    edge_l1     L1 between the horizontal / vertical pixel differences of the reconstruction and of the image
                (also across patch borders), with the same weights: edges must be where they are
    lpips       perceptual distance (LPIPS) between reconstruction and crop. AlexNet features (`lpips_net`): measured
                on the shared GPU, 64 crops forward + backward take 0.010 s, with VGG 0.19 s (more than the rest of
                the training step)
    selection   pose score + near_lateral * (error across the axis within 5 mm of inserted) + key_rgb * (RGB error
                on the key pixels): the pose alone is one global quantity and says nothing about kept detail
v5b (decided with the user 2026-11-12; model `rgb: false`, `seg_classes: 7`; loss weights `seg`, `lateral`, `close_weight`):
v5 at 10 000 steps had tokens that differ (token-specific share 36-55 %) and twice v4e's reconstruction, but the pose
terms had not moved since step 4000 (v4e's fell steadily): RGB + edge weights, 8 times v4e's RGB weight in all, own
the clipped gradient. And an RGB reconstruction is blocky at the 8 px patch size -- no way to a one-pixel edge.
    no RGB target   colour is an input (the edges are read from it: without RGB a dedicated reader is blind within
                    1 mm) but not something the latent has to give back
    seg_ce          instead the decoder gives every pixel's class (the frozen UNet's prediction on the image, a
                    real-robot quantity): cross-entropy, key pixels weighted as before. An edge off by a pixel costs.
                    Logged: IoU of the decoded pins / openings with the UNet's.
    depth_w         depth L1 weighted towards the pins, the openings and the board around them (15 px): where depth
                    is needed (the gap to the board, the tilt; without depth the reader loses the vertical direction)
    close samples   frames whose remaining motion across the axis is under 3 mm (FK label) count `1 + close_weight`
                    times in the pose terms, and `lateral` weighs their error across the axis (mm) directly
    weights         otherwise v4e's, under which the pose was learned
Logged at validation: latent_specific_share = the part of the latent's frame-to-frame variation that differs
between the tokens of a frame (v4e: under 1 %: the tokens were copies).
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
from ..models.crop_encoder import cut, patches
from ..perception.labels import OPENING, PINS
from ..models.encoder import DEPTH_SCALE_M
from . import common

PLUG_CLASS = 1


class KeyPixels:
    """Loss weights from the frozen UNet (pins / openings) and the perceptual distance; nothing of it is a model input."""

    def __init__(self, cfg: dict, device):
        self.unet, self.lpips = None, None
        self.weight, self.dilate = float(cfg.get("key_weight", 0.0)), int(cfg.get("key_dilate", 5))
        if cfg.get("unet"):
            from ..perception.unet import load_unet
            self.unet = load_unet(cfg["unet"], device).requires_grad_(False)
        if cfg["loss_weights"].get("lpips"):
            import lpips
            self.lpips = lpips.LPIPS(net=cfg.get("lpips_net", "alex"), verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict | None:
        """Per image pixel (B, 320, 320): `w` = 1, and 1 + key_weight on the predicted pins and openings (dilated);
        `wd` the same with a wide margin (the board around them; for the depth); `cls` the predicted class."""
        if self.unet is None:
            return None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = self.unet(batch["rgb"]).argmax(1)
        m = ((cls == PINS) | (cls == OPENING)).float()[:, None]
        return {"w": 1 + self.weight * F.max_pool2d(m, self.dilate, 1, self.dilate // 2)[:, 0],
                "wd": 1 + self.weight * F.max_pool2d(m, 31, 1, 15)[:, 0], "cls": cls}


def losses(model: CropAE, out: dict, batch: dict, w: dict, ref: torch.Tensor, key: dict | None = None,
           perceptual=None) -> dict:
    l = {k: 0.0 for k in ("depth_mm", "depth_w", "key_depth_mm", "valid_bce", "rgb_l1", "rgb_w", "key_rgb_l1", "edge_l1",
                          "lpips", "seg_ce", "seg_iou_pins", "seg_iou_opening", "dino_cos", "motion_mm", "weight_bce")}
    with torch.autocast("cuda", enabled=False):
        R_star, t_star = insertion_motion(batch["hand"].float(), batch["h_star"].float(), batch["R_P"].float())
        geom = {k: batch[k] for k in ("K", "cam_T_world", "hand", "R_P", "plug_uv0", "hole_uv0")}
        for k in ("plug", "hole"):
            pred, tgt = out["pred"][k], out["target"][k]
            valid = tgt["valid_patch"] > 0
            d_err = (pred["depth_patch"].float() - tgt["depth_patch"].float()).abs() * DEPTH_SCALE_M * 1000
            l["depth_mm"] = l["depth_mm"] + (d_err * valid).sum() / valid.sum().clamp(min=1)
            l["valid_bce"] = l["valid_bce"] + F.binary_cross_entropy_with_logits(pred["valid_logit"].float(), valid.float())
            Wk = cut(key["w"], batch[f"{k}_uv0"]).float() if key is not None else None      # (B, 128, 128)
            if key is None:
                l["depth_w"] = l["depth_w"] + (d_err * valid).sum() / valid.sum().clamp(min=1)
            else:
                wdp = patches(cut(key["wd"], batch[f"{k}_uv0"]).float()[:, None]) * valid   # (B, 256, 64), valid pixels only
                l["depth_w"] = l["depth_w"] + (d_err * wdp).sum() / wdp.sum().clamp(min=1)
                kd = (wdp > 1).float()
                l["key_depth_mm"] = l["key_depth_mm"] + (d_err * kd).sum() / kd.sum().clamp(min=1)
            if "rgb_patch" in pred:
                rgb_err = (pred["rgb_patch"].float() - tgt["rgb_patch"].float()).abs()
                l["rgb_l1"] = l["rgb_l1"] + rgb_err.mean()
                if key is None:
                    l["rgb_w"] = l["rgb_w"] + rgb_err.mean()
                else:
                    wp = patches(Wk[:, None])                                     # (B, 256, 64), as the patches' pixels
                    e_pix = rgb_err.unflatten(-1, (3, -1)).mean(2)                # per pixel, mean over the 3 colours
                    l["rgb_w"] = l["rgb_w"] + (e_pix * wp).sum() / wp.sum()
                    is_key = (wp > 1).float()
                    l["key_rgb_l1"] = l["key_rgb_l1"] + (e_pix * is_key).sum() / is_key.sum().clamp(min=1)
                    P, T = unpatch(pred["rgb_patch"].float(), 3), unpatch(tgt["rgb_patch"].float(), 3)
                    for ax in (-1, -2):
                        d = (P.diff(dim=ax) - T.diff(dim=ax)).abs().mean(1)
                        wd = torch.maximum(Wk.narrow(ax, 1, Wk.shape[ax] - 1), Wk.narrow(ax, 0, Wk.shape[ax] - 1))
                        l["edge_l1"] = l["edge_l1"] + (d * wd).sum() / wd.sum()
                    if perceptual is not None:
                        l["lpips"] = l["lpips"] + perceptual(P * 2 - 1, T * 2 - 1).mean()
            if "seg_patch" in pred and key is not None:
                cls_t = cut(key["cls"], batch[f"{k}_uv0"])                         # (B, 128, 128) the UNet's classes
                logits = unpatch(pred["seg_patch"].float(), model.seg_classes)     # (B, classes, 128, 128)
                ce = F.cross_entropy(logits, cls_t, reduction="none")
                l["seg_ce"] = l["seg_ce"] + (ce * Wk).sum() / Wk.sum()
                cls_p = logits.argmax(1)
                for name, c in (("pins", PINS), ("opening", OPENING)):
                    a, b = cls_p == c, cls_t == c
                    l[f"seg_iou_{name}"] = l[f"seg_iou_{name}"] + 0.5 * (a & b).sum() / (a | b).sum().clamp(min=1)
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
        # close to the hole axis (under 3 mm across it left to go, at any depth): where insert / jam is decided
        close = (t_star[:, 1:].norm(dim=-1) < 0.003).float()
        sw = 1 + w.get("close_weight", 0.0) * close
        l["close_frac"] = close.mean()
        l["close_lateral_err_mm"] = (lat_err * close).sum() / close.sum().clamp(min=1)
        grid_w, angle_w = (grid * sw).sum() / sw.sum(), (angle * sw).sum() / sw.sum()
    l["grid_mm"], l["angle_deg"] = grid.mean(), angle.mean()
    l["pose_score"] = l["grid_mm"] + w["angle"] * l["angle_deg"]
    l["total"] = (w["depth"] * l["depth_w"] + w["valid"] * l["valid_bce"] + w.get("rgb", 0.0) * l["rgb_w"]
                  + w.get("edge", 0.0) * l["edge_l1"] + w.get("lpips", 0.0) * l["lpips"] + w.get("seg", 0.0) * l["seg_ce"]
                  + w["dino"] * l["dino_cos"] + w["motion"] * l["motion_mm"] + w["weight"] * l["weight_bce"]
                  + w["grid"] * grid_w + w["angle"] * angle_w + w.get("lateral", 0.0) * l["close_lateral_err_mm"])
    return l


@torch.no_grad()
def evaluate(model, loader, w, device, max_batches, ref, key=None) -> dict:
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            l = losses(model, out, batch, w, ref, key(batch) if key else None, key.lpips if key else None)
        r = out["z"].float() - out["z"].float().mean(0, keepdim=True)         # frame-to-frame variation of every token
        l["latent_specific_share"] = (r - r.mean(1, keepdim=True)).pow(2).sum() / r.pow(2).sum().clamp(min=1e-12)
        nf, cf = float(l["near_frac"]), float(l["close_frac"])
        for k, v in l.items():
            wk = nf if (k.startswith("near_") and k != "near_frac") else cf if k == "close_lateral_err_mm" else 1.0
            acc[k] = acc.get(k, 0.0) + float(v) * wk
            acc["_w_" + k] = acc.get("_w_" + k, 0.0) + wk
        n += 1
    model.train()
    return {k: v / max(acc["_w_" + k], 1e-9) for k, v in acc.items() if not k.startswith("_w_")}


def train(cfg: dict, run_dir: Path, max_steps: int | None = None, resume: str | None = None) -> Path:
    """`resume`: a last.pt of the same design (model + optimizer + step); training continues from its step."""
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    # one or several data sources (demonstrations, contact exploration): `data` / `crops` are a path or parallel
    # lists; training batches are drawn from the sources with `source_weights`, validation is per source
    srcs = list(zip(*[[cfg[k]] if isinstance(cfg[k], str) else cfg[k] for k in ("data", "crops")]))
    names = [Path(d).name for d, _ in srcs]
    mk = lambda d, c, split, workers: DataLoader(CropFrameDataset(data_index.load(d), split, d, c), batch_size=cfg["batch_size"],
                                                 shuffle=True, drop_last=True, num_workers=workers,
                                                 persistent_workers=workers > 0, pin_memory=True)
    train_loaders = [mk(d, c, "train", cfg["workers"]) for d, c in srcs]
    val_loaders = {n: mk(d, c, "ood_val", 6) for n, (d, c) in zip(names, srcs)}
    weights = cfg.get("source_weights") or [1.0] * len(srcs)
    from .wm4 import mixed_batches
    batches = mixed_batches(train_loaders, weights, cfg["seed"] + 3)
    model = CropAE(**cfg["model"]).to(device)
    key = KeyPixels(cfg, device)
    sel = cfg.get("selection") or {}
    score = lambda x: (x["pose_score"] + sel.get("near_lateral", 0.0) * x["near_lateral_err_mm"] + sel.get("key_rgb", 0.0) * x["key_rgb_l1"]
                       + sel.get("close_lateral", 0.0) * x["close_lateral_err_mm"]
                       + sel.get("seg_iou", 0.0) * (2 - x["seg_iou_pins"] - x["seg_iou_opening"]))
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
    log(event="start", resume=resume, start_step=start, sources=names, source_weights=weights,
        train_frames=[len(l.dataset) for l in train_loaders], val_frames=[len(l.dataset) for l in val_loaders.values()],
        params_m=sum(p.numel() for p in common.trainable(model)) / 1e6)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, batch in zip(range(start + 1, steps + 1), batches):
        batch = common.to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = losses(model, model(batch), batch, w, ref, key(batch), key.lpips)
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
            per = {n: evaluate(model, l, w, device, cfg["eval_batches"], ref, key) for n, l in val_loaders.items()}
            v = {"pose_score": sum(score(x) for x in per.values()) / len(per)}            # selection: mean over the sources
            log(step=step, val_pose_score=v["pose_score"],
                **{f"val_{n}/{k}": x for n, m in per.items() for k, x in m.items()})
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
