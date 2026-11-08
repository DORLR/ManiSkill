"""Stage 2: CDiT diffusion world model on VAE latents.

Training follows NWM: random diffusion step, `diffusion.training_losses` with
model_kwargs (y = action condition, x_cond = 4 context latents, rel_t), AdamW, EMA.
Since v2 (docs/worklog.md, 2026-11-07), optional:
    cfg["history"]          y also holds the 1.2 s tracking history (data.datasets.action_condition)
    cfg["pose_loss"]        weight of the predicted-pose loss: the model's clean-latent estimate
                            and the true next latent are both read by the frozen pose head; the
                            loss is their difference (plug-region displacement mm + angle deg),
                            on diffusion steps t < cfg["pose_loss_max_t"] only (noisy estimates
                            at large t are not meaningful)
Evaluation on the OOD validation split:
    val_diffusion_loss    the training loss on held-out data
    predicted vs true next step, both decoded by the frozen VAE decoder:
        pose_rot_deg / pose_trans_mm   plug-pose difference
        joint_mrad                     joint difference (the control-signal prediction)
    and the same numbers for "copy the last context latent" (no-change baseline)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import index as data_index
from ..data.datasets import CONTEXT, LatentWindowDataset
from ..geometry import apply_rigid, plug_region_points, rotation_angle_deg
from ..models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from ..models.vae import TokenVAE, load_vae
from . import common


def _rot_deg(Ra, Rb):
    c = ((Ra.transpose(1, 2) @ Rb).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


@torch.no_grad()
def compare_decoded(vae: TokenVAE, z_pred, z_true, scale: float) -> dict:
    a, b = vae.decode(z_pred / scale), vae.decode(z_true / scale)
    return {"pose_rot_deg": float(_rot_deg(a["R"].float(), b["R"].float()).mean()),
            "pose_trans_mm": float((a["t"] - b["t"]).norm(dim=-1).mean() * 1000),
            "joint_mrad": float((a["joints"] - b["joints"]).abs().mean() * 1000)}


def predicted_pose_error(vae: TokenVAE, z_pred, z_true, scale: float, ref) -> torch.Tensor:
    """Per sample: plug-region displacement difference (mm) + rotation difference (deg) between the
    poses read from predicted and true latents (B, 64, C). Gradient flows into z_pred only."""
    Rp, tp = vae.decode_pose(z_pred / scale)
    with torch.no_grad():
        Rt, tt = vae.decode_pose(z_true / scale)
    p = ref[None].expand(z_pred.shape[0], -1, -1)
    disp = (apply_rigid(p, Rp.float(), tp.float()) - apply_rigid(p, Rt.float(), tt.float())).norm(dim=-1).mean(1) * 1000
    return disp + rotation_angle_deg(Rp.float(), Rt.float())


@torch.no_grad()
def evaluate(model, diffusion, sampler, vae, loader, device, scale, max_batches) -> dict:
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = common.to_device(batch, device)
        x, x_cond = tokens_to_grid(batch["x"]), tokens_to_grid(batch["x_cond"])
        kw = dict(y=batch["y"], x_cond=x_cond, rel_t=batch["rel_t"])
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vl = diffusion.training_losses(model, x, t, kw)["loss"].mean()
            pred = sampler.p_sample_loop(model.forward, x.shape, torch.randn_like(x), clip_denoised=False,
                                         model_kwargs=kw, progress=False, device=device)
        m = {"diffusion_loss": float(vl)}
        m.update({f"pred_{k}": v for k, v in compare_decoded(vae, grid_to_tokens(pred.float()), batch["x"], scale).items()})
        m.update({f"copy_{k}": v for k, v in compare_decoded(vae, batch["x_cond"][:, -1], batch["x"], scale).items()})
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def train(cfg: dict, run_dir: Path, entries: list[dict] | None = None, max_steps: int | None = None) -> Path:
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    entries = entries if entries is not None else data_index.load(cfg["data"])
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    history = cfg.get("history", False)
    mk = lambda split: DataLoader(LatentWindowDataset(cfg["latents"], entries, split, cfg["data"], scale, history),
                                  batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                                  num_workers=cfg["workers"], persistent_workers=cfg["workers"] > 0)
    train_loader, val_loader = mk("train"), mk("ood_val")
    vae = load_vae(cfg["vae_checkpoint"], device).requires_grad_(False)
    model = TokenCDiT(action_dim=cfg["action_dim"], latent_ch=vae.latent_ch, context_size=CONTEXT,
                      size=cfg["size"]).to(device)
    ema = common.EMA(model, cfg["ema_decay"])
    pose_w, pose_max_t = cfg.get("pose_loss", 0.0), cfg.get("pose_loss_max_t", 200)
    ref = plug_region_points(device)
    diffusion = create_diffusion(timestep_respacing="")
    sampler = create_diffusion(timestep_respacing=str(cfg["eval_sampling_steps"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) if warmup else 1.0)
    # bf16 autocast made the learned-variance diffusion loss spike (run of 2026-11-07: 0.008 -> 0.88
    # between steps 4100 and 5100); the 49 M-parameter model trains in fp32 unless "amp" is set.
    amp = cfg.get("amp", False)
    # TF32 only for matmuls/convolutions: fp32 range everywhere (the loss's log/exp stay fp32),
    # about twice as fast as plain fp32 on this GPU.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log = common.Logger(run_dir)
    log(event="start", train_windows=len(train_loader.dataset), val_windows=len(val_loader.dataset),
        params_m=sum(p.numel() for p in model.parameters()) / 1e6, scale=scale,
        action_dim=cfg["action_dim"], history=history, pose_loss=pose_w)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step, batch in zip(range(1, steps + 1), common.cycle(train_loader)):
        batch = common.to_device(batch, device)
        x, x_cond = tokens_to_grid(batch["x"]), tokens_to_grid(batch["x_cond"])
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        cache = {}

        def net(x_t, tt, **kw):                       # keeps the model output for the pose loss
            out = model(x_t, tt, **kw)
            cache["x_t"], cache["out"] = x_t, out
            return out

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            diff_loss = diffusion.training_losses(net, x, t, dict(y=batch["y"], x_cond=x_cond, rel_t=batch["rel_t"]))["loss"].mean()
            pose_loss = torch.zeros((), device=device)
            low = t < pose_max_t
            if pose_w > 0 and low.any():
                C = x.shape[1]
                x0 = diffusion._predict_xstart_from_eps(cache["x_t"][low], t[low], cache["out"][low, :C])
                pose_loss = predicted_pose_error(vae, grid_to_tokens(x0), batch["x"][low], scale, ref).mean()
            loss = diff_loss + pose_w * pose_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (torch.isfinite(loss) and torch.isfinite(gn)):   # never apply a non-finite update
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
            log(step=step, train_loss=float(loss), train_diffusion_loss=float(diff_loss),
                train_pose_err=float(pose_loss), grad_norm=float(gn), lr=sched.get_last_lr()[0])
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, diffusion, sampler, vae, val_loader, device, scale, cfg["eval_batches"])
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(),
                                   opt=opt.state_dict(), step=step, cfg=cfg)
            key = v["pred_joint_mrad"] + v["pred_pose_trans_mm"]
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
