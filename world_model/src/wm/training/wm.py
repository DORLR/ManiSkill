"""Stage 2: CDiT world model (v3), trained from scratch on the v3 latents.

Each sample (data/datasets.py LatentWindowDataset): 4 context latents (0.4 s apart), a target latent
at a time shift drawn per sample (half the samples +0.4 s, the step used with the policy; the other
half uniform over +-16 s, NWM's range), the robot-state / commanded-hand-path condition
(data/conditions.py) and rel_t = the shift in seconds.

Loss = NWM's diffusion loss (random diffusion step, learned sigma)
     + w_pose * predicted-pose loss: the model's clean-latent estimate at the target time and the
       true target latent are both read by the frozen autoencoder's pose head; their difference
       (reference-grid mm + w_angle * angle deg) is penalised, on diffusion steps t < pose_max_t only
       (at large t the clean estimate is still mostly noise).
Context-noise augmentation (GameNGen): with probability p_ctx_noise the context latents get
Gaussian noise of a random level sigma in [0, ctx_noise_max]; sigma is given to the model (0 at
use time), so it learns to predict from imperfect context such as its own predictions.

fp32 with TF32 matmuls (bf16 made the learned-variance loss diverge in v1), LR warm-up, EMA with
warm-up, gradient accumulation (`accum`) to keep the effective batch when memory is short.

Validation on ood_val at fixed time shifts (0.4, 2, 4, 8, 16 s ahead and 4 s back): diffusion loss,
and the pose read from a sampled prediction vs the pose read from the true latent, next to the
"nothing changes" prediction (the current latent). Model selection: the mean pose score over the
shifts (the world model has to be right at every horizon, not only one step).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import CONTEXT, LatentWindowDataset
from ..geometry import plug_region_points, pose_loss_terms
from ..models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from ..models.vae import TokenVAE, load_vae
from . import common

VAL_SHIFTS = {"+0.4s": 2, "+2s": 10, "+4s": 20, "+8s": 40, "+16s": 80, "-4s": -20}   # images (0.2 s each)


def pose_error(vae: TokenVAE, z_a, z_b, scale: float, ref, grad_a: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(grid mm, angle deg) per sample between the poses read from latents z_a and z_b (B, 64, C).
    Gradient flows into z_a only when grad_a."""
    with torch.set_grad_enabled(grad_a and torch.is_grad_enabled()):
        Ra, ta = vae.decode_pose(z_a / scale)
    with torch.no_grad():
        Rb, tb = vae.decode_pose(z_b / scale)
    return pose_loss_terms(Ra, ta, Rb, tb, ref)


def noisy_context(x_cond: torch.Tensor, p: float, max_sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    B = x_cond.shape[0]
    sigma = torch.rand(B, device=x_cond.device) * max_sigma
    sigma = torch.where(torch.rand(B, device=x_cond.device) < p, sigma, torch.zeros_like(sigma))
    return x_cond + sigma.view(B, 1, 1, 1, 1) * torch.randn_like(x_cond), sigma


@torch.no_grad()
def evaluate(model, diffusion, sampler, vae, loaders: dict, device, scale, max_batches, ref, w_angle) -> dict:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        acc, n = {}, 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = common.to_device(batch, device)
            x, x_cond = tokens_to_grid(batch["x"]), tokens_to_grid(batch["x_cond"])
            kw = dict(y=batch["y"], x_cond=x_cond, rel_t=batch["rel_t"])
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            vl = diffusion.training_losses(model, x, t, kw)["loss"].mean()
            pred = sampler.p_sample_loop(model.forward, x.shape, torch.randn_like(x), clip_denoised=False,
                                         model_kwargs=kw, progress=False, device=device)
            g, a = pose_error(vae, grid_to_tokens(pred), batch["x"], scale, ref, grad_a=False)
            gc, ac = pose_error(vae, batch["x_cond"][:, -1], batch["x"], scale, ref, grad_a=False)
            m = {"diffusion_loss": float(vl), "pred_grid_mm": float(g.mean()), "pred_angle_deg": float(a.mean()),
                 "copy_grid_mm": float(gc.mean()), "copy_angle_deg": float(ac.mean())}
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        if n == 0:                       # no held-out window this long (cannot happen on the full data)
            continue
        for k, v in acc.items():
            out[f"{name}/{k}"] = v / n
        out[f"{name}/pred_score"] = out[f"{name}/pred_grid_mm"] + w_angle * out[f"{name}/pred_angle_deg"]
    scores = [v for k, v in out.items() if k.endswith("/pred_score")]
    out["mean_pred_score"] = sum(scores) / len(scores) if scores else float("nan")
    model.train()
    return out


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = entries if entries is not None else data_index.load(cfg["data"])
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    args = (cfg["latents"], entries)
    train_ds = LatentWindowDataset(*args, "train", cfg["data"], scale, p_step=cfg["p_step"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0, pin_memory=True)
    val_loaders = {name: DataLoader(LatentWindowDataset(*args, "ood_val", cfg["data"], scale, fixed_shift=m),
                                    batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=2)
                   for name, m in VAL_SHIFTS.items()}
    vae = load_vae(cfg["vae_checkpoint"], device).requires_grad_(False)
    model = TokenCDiT(cond_dim=cfg["cond_dim"], latent_ch=vae.latent_ch, context_size=CONTEXT, size=cfg["size"]).to(device)
    ema = common.EMA(model, cfg["ema_decay"])
    diffusion = create_diffusion(timestep_respacing="")
    sampler = create_diffusion(timestep_respacing=str(cfg["eval_sampling_steps"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ref, accum = plug_region_points(device), cfg.get("accum", 1)
    w_pose, w_angle, pose_max_t = cfg["pose_loss"], cfg["pose_angle_weight"], cfg["pose_loss_max_t"]
    log = common.Logger(run_dir)
    log(event="start", train_windows=len(train_ds), params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        scale=scale, size=cfg["size"], effective_batch=cfg["batch_size"] * accum,
        val_windows={k: len(v.dataset) for k, v in val_loaders.items()})
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    batches = common.cycle(train_loader)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "diffusion": 0.0, "pose_grid": 0.0, "pose_angle": 0.0}
        for _ in range(accum):
            batch = common.to_device(next(batches), device)
            x, x_cond = tokens_to_grid(batch["x"]), tokens_to_grid(batch["x_cond"])
            x_cond, sigma = noisy_context(x_cond, cfg["p_ctx_noise"], cfg["ctx_noise_max"])
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            cache = {}

            def net(x_t, tt, **kw):              # keeps the model output for the pose loss
                out = model(x_t, tt, **kw)
                cache["x_t"], cache["out"] = x_t, out
                return out

            kw = dict(y=batch["y"], x_cond=x_cond, rel_t=batch["rel_t"], ctx_noise=sigma)
            diff = diffusion.training_losses(net, x, t, kw)["loss"].mean()
            grid = angle = torch.zeros((), device=device)
            low = t < pose_max_t
            if w_pose > 0 and low.any():
                C = x.shape[1]
                x0 = diffusion._predict_xstart_from_eps(cache["x_t"][low], t[low], cache["out"][low, :C])
                g, a = pose_error(vae, grid_to_tokens(x0), batch["x"][low], scale, ref)
                grid, angle = g.mean(), a.mean()
            loss = (diff + w_pose * (grid + w_angle * angle)) / accum
            loss.backward()
            for k, v in zip(tot, (loss * accum, diff, grid, angle)):
                tot[k] += float(v) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (math.isfinite(tot["loss"]) and torch.isfinite(gn)):   # never apply a non-finite update
            skipped += 1
            log(step=step, event="skipped_nonfinite_step", skipped=skipped)
            if skipped >= cfg.get("max_skipped", 50):
                log(event="stop_nonfinite", step=step)
                break
            continue
        opt.step()
        sched.step()
        ema.update(model)
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, **{f"train_{k}": v for k, v in tot.items()}, grad_norm=float(gn), lr=sched.get_last_lr()[0])
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, diffusion, sampler, vae, val_loaders, device, scale, cfg["eval_batches"], ref, w_angle)
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(),
                                   opt=opt.state_dict(), step=step, cfg=cfg)
            key = v["mean_pred_score"]
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


def load_world_model(path: str, device=None) -> TokenCDiT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = TokenCDiT(cond_dim=cfg["cond_dim"], latent_ch=cfg.get("latent_ch", 8), context_size=CONTEXT, size=cfg["size"])
    m.load_state_dict(ck["ema"])
    m.eval()
    return m.to(device) if device is not None else m
