"""Stage 2 (v4): history world model (models/history_cdit.py), trained from scratch, then online.

Each sample (data/history.py HistoryWindowDataset): the 6.2 s history ending now (vision masked at
random: mostly the last slots, as when the plug covers the opening; plus random blocks and dropped
slots), a target latent at a time shift (half +0.4 s, the rest uniform over +-16 s), the commanded hand
path to the target, rel_t.

Loss = NWM's diffusion loss + w_pose x (grid mm + w_angle x angle deg) between the pose solved from the
model's clean-latent estimate and from the true target latent (frozen crop autoencoder, target-frame
geometry), on diffusion steps t < pose_max_t. Context-noise augmentation (GameNGen) on the seen latents.

The training loop runs over a `Sources` object, so the online stage (online/learner.py) can keep
adding files collected by the actors (mixed with the offline data at a fixed ratio).

Validation (ood_val, fixed shifts 0.4 / 2 / 4 / 16 s ahead): with the full history and with the last
2 s of vision masked; pose error of a sampled prediction vs the true latent, next to "nothing changes"
(the last seen latent).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.history import HIST, HistoryWindowDataset, LatentFiles
from ..geometry import plug_region_points, pose_loss_terms
from ..models.crop_ae import CropAE, load_crop_ae
from ..models.history_cdit import HistoryCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
from . import common

VAL_SHIFTS = {"+0.4s": 2, "+2s": 10, "+4s": 20, "+16s": 80}
HIST_KEYS = ("z_hist", "seen", "proprio", "dt", "avail")


def model_inputs(batch: dict) -> dict:
    return {"z_hist": batch["z"], "seen": batch["seen"], "proprio": batch["proprio"], "dt": batch["dt"],
            "avail": batch["avail"]}


def pose_error(ae: CropAE, z_a, z_b, scale, ref, geom, grad_a=True):
    with torch.set_grad_enabled(grad_a and torch.is_grad_enabled()):
        Ra, ta = ae.decode_pose(z_a / scale, geom)
    with torch.no_grad():
        Rb, tb = ae.decode_pose(z_b / scale, geom)
    return pose_loss_terms(Ra, ta, Rb, tb, ref)


def noisy_history(z, seen, p, max_sigma):
    B = z.shape[0]
    sigma = torch.rand(B, device=z.device) * max_sigma
    sigma = torch.where(torch.rand(B, device=z.device) < p, sigma, torch.zeros_like(sigma))
    z = z + sigma.view(B, 1, 1, 1) * torch.randn_like(z) * seen[..., None, None]
    return z, sigma


def last_seen(z, seen):
    """The latest seen latent of each sample (B, 64, C): the 'nothing changes' prediction."""
    idx = (seen.float() * torch.arange(1, seen.shape[1] + 1, device=seen.device)).argmax(1)
    return z[torch.arange(z.shape[0], device=z.device), idx]


@torch.no_grad()
def evaluate(model, diffusion, sampler, ae, loaders, device, scale, max_batches, ref, w_angle) -> dict:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        acc, n = {}, 0
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            b = common.to_device(b, device)
            x = tokens_to_grid(b["x"])
            kw = dict(path=b["path"], rel_t=b["rel_t"], **model_inputs(b))
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            vl = diffusion.training_losses(model, x, t, kw)["loss"].mean()
            pred = sampler.p_sample_loop(model.forward, x.shape, torch.randn_like(x), clip_denoised=False,
                                         model_kwargs=kw, progress=False, device=device)
            g, a = pose_error(ae, grid_to_tokens(pred), b["x"], scale, ref, b["geom"], grad_a=False)
            gc, ac = pose_error(ae, last_seen(b["z"], b["seen"]), b["x"], scale, ref, b["geom"], grad_a=False)
            m = {"diffusion_loss": float(vl), "pred_grid_mm": float(g.mean()), "pred_angle_deg": float(a.mean()),
                 "copy_grid_mm": float(gc.mean()), "copy_angle_deg": float(ac.mean())}
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
        if n:
            for k, v in acc.items():
                out[f"{name}/{k}"] = v / n
            out[f"{name}/pred_score"] = out[f"{name}/pred_grid_mm"] + w_angle * out[f"{name}/pred_angle_deg"]
    scores = [v for k, v in out.items() if k.endswith("/pred_score")]
    out["mean_pred_score"] = sum(scores) / len(scores) if scores else float("nan")
    model.train()
    return out


class FixedTailDataset(HistoryWindowDataset):
    """Validation: the last `tail` slots of vision masked (the plug covers the opening)."""

    def __init__(self, *a, tail: int, **k):
        super().__init__(*a, mask=None, **k)
        self.tail = tail

    def __getitem__(self, k):
        out = super().__getitem__(k)
        seen = out["seen"].clone()
        idx = torch.nonzero(seen)[:, 0]
        if len(idx) > 1:
            hide = idx[-min(self.tail, len(idx) - 1):]
            seen[hide] = False
            out["z"][~seen] = 0
            out["seen"] = seen
        return out


def sources(cfg: dict) -> list[tuple[str, str, str]]:
    """(name, data root, latent root) of every data source. `data` / `latents` in the config are one path or
    two parallel lists (demonstrations first, then contact exploration); the first source defines the
    latent scale (the autoencoder and the FM were trained with it)."""
    d, l = cfg["data"], cfg["latents"]
    if isinstance(d, str):
        return [(Path(d).name, d, l)]
    return [(Path(x).name, x, y) for x, y in zip(d, l)]


def latent_scale(cfg: dict) -> float:
    return json.loads((Path(sources(cfg)[0][2]) / "stats.json").read_text())["scale"]


def offline_files(cfg: dict, split: str, source: str | None = None) -> LatentFiles:
    """Latent files of a split, of all sources or of one (by name)."""
    from ..data import index as data_index
    paths = []
    for name, data, root in sources(cfg):
        if source is not None and name != source:
            continue
        for e in data_index.load(data):
            if e["split"] == split:
                q = Path(root) / Path(e["path"]).relative_to(data)
                if q.exists():
                    paths.append(str(q))
    return LatentFiles(paths)


def mixed_batches(loaders: list, weights: list[float], seed: int):
    """Endless batches; each batch comes from one source, drawn with the given probabilities."""
    its = [common.cycle(l) for l in loaders]
    rng = __import__("numpy").random.default_rng(seed)
    w = [x / sum(weights) for x in weights]
    while True:
        yield next(its[int(rng.choice(len(its), p=w))])


def train_batches(cfg: dict, make_dataset):
    """One loader per source (make_dataset(files) -> dataset, loader), mixed by cfg["source_weights"]."""
    loaders = [make_dataset(offline_files(cfg, "train", name)) for name, _, _ in sources(cfg)]
    weights = cfg.get("source_weights") or [1.0] * len(loaders)
    return mixed_batches(loaders, weights, cfg["seed"] + 5)


def build(cfg: dict, device):
    scale = latent_scale(cfg)
    ae = load_crop_ae(cfg["ae_checkpoint"], device).requires_grad_(False)
    cfg["n_tokens"] = ae.n_tokens                           # 64 (v4 latents) or 128 (v5); saved with the checkpoint
    model = HistoryCDiT(latent_ch=ae.latent_ch, size=cfg["size"], n_tokens=ae.n_tokens).to(device)
    return scale, ae, model


def val_loaders(cfg: dict, scale: float) -> dict:
    """Held-out (ood_val) windows per source and fixed shift, with the full history and with the last 2 s of
    vision masked. Shifts a source has too few windows for (short exploration trajectories) are left out."""
    out = {}
    for src, _, _ in sources(cfg):
        files = offline_files(cfg, "ood_val", src)
        for name, m in VAL_SHIFTS.items():
            items = HistoryWindowDataset.index(files, fixed_shift=m)
            if len(items) < 4 * cfg["batch_size"]:
                continue
            for tail_name, ds in (("full", HistoryWindowDataset(files, scale, None, fixed_shift=m, items=items)),
                                  ("tail2s", FixedTailDataset(files, scale, fixed_shift=m, items=items, tail=10))):
                out[f"{src}:{name}_{tail_name}"] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                                                             drop_last=True, num_workers=2)
    return out


def train(cfg: dict, run_dir: Path, batches, max_steps: int | None = None, resume: str | None = None,
          on_step=None) -> Path:
    """`batches`: an iterator of training batches (offline loader, or the online mixture). `on_step(step)`
    is called after every optimiser step (the online learner refreshes its data there)."""
    common.seed_all(cfg["seed"])
    device = torch.device("cuda")
    scale, ae, model = build(cfg, device)
    ema = common.EMA(model, cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    start = 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.n = ck.get("ema_n", 10 ** 6)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
    diffusion = create_diffusion(timestep_respacing="")
    sampler = create_diffusion(timestep_respacing=str(cfg["eval_sampling_steps"]))
    warmup = cfg.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1 + start) / warmup) if warmup else 1.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ref, accum = plug_region_points(device), cfg.get("accum", 1)
    w_pose, w_angle, pose_max_t = cfg["pose_loss"], cfg["pose_angle_weight"], cfg["pose_loss_max_t"]
    vals = val_loaders(cfg, scale)
    log = common.Logger(run_dir)
    log(event="start", resume=resume, start_step=start, params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        scale=scale, size=cfg["size"], effective_batch=cfg["batch_size"] * accum, history_slots=HIST)
    best, bad, skipped = float("inf"), 0, 0
    steps = max_steps or cfg["steps"]
    for step in range(start + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "diffusion": 0.0, "pose_grid": 0.0, "pose_angle": 0.0}
        for _ in range(accum):
            b = common.to_device(next(batches), device)
            x = tokens_to_grid(b["x"])
            z, sigma = noisy_history(b["z"], b["seen"], cfg["p_ctx_noise"], cfg["ctx_noise_max"])
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            cache = {}

            def net(x_t, tt, **kw):
                out = model(x_t, tt, **kw)
                cache["x_t"], cache["out"] = x_t, out
                return out

            kw = dict(path=b["path"], rel_t=b["rel_t"], z_hist=z, seen=b["seen"], proprio=b["proprio"], dt=b["dt"],
                      avail=b["avail"], ctx_noise=sigma)
            diff = diffusion.training_losses(net, x, t, kw)["loss"].mean()
            grid = angle = torch.zeros((), device=device)
            low = t < pose_max_t
            if w_pose > 0 and low.any():
                C = x.shape[1]
                x0 = diffusion._predict_xstart_from_eps(cache["x_t"][low], t[low], cache["out"][low, :C])
                g, a = pose_error(ae, grid_to_tokens(x0), b["x"][low], scale, ref, {k: v[low] for k, v in b["geom"].items()})
                grid, angle = g.mean(), a.mean()
            loss = (diff + w_pose * (grid + w_angle * angle)) / accum
            loss.backward()
            for k, v in zip(tot, (loss * accum, diff, grid, angle)):
                tot[k] += float(v) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not (math.isfinite(tot["loss"]) and torch.isfinite(gn)):
            skipped += 1
            log(step=step, event="skipped_nonfinite_step", skipped=skipped)
            if skipped >= cfg.get("max_skipped", 50):
                log(event="stop_nonfinite", step=step)
                break
            continue
        opt.step()
        sched.step()
        ema.update(model)
        if on_step is not None:
            on_step(step)
        if step % cfg["log_every"] == 0 or step == steps:
            log(step=step, **{f"train_{k}": v for k, v in tot.items()}, grad_norm=float(gn), lr=sched.get_last_lr()[0])
        if step % cfg["save_every"] == 0:
            common.save_checkpoint(run_dir / "current.pt", ema=ema.model.state_dict(), step=step, cfg=cfg)
        if step % cfg["eval_every"] == 0 or step == steps:
            v = evaluate(ema.model, diffusion, sampler, ae, vals, device, scale, cfg["eval_batches"], ref, w_angle)
            log(step=step, **{f"val_{k}": x for k, x in v.items()})
            common.save_checkpoint(run_dir / "last.pt", model=model.state_dict(), ema=ema.model.state_dict(), ema_n=ema.n,
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
                if cfg.get("patience") and bad >= cfg["patience"] and key > best * (1 + cfg["harm_margin"]):
                    log(event="early_stop_harm", step=step, best=best, current=key)
                    break
    return run_dir / "best.pt"


def load_history_wm(path: str, device=None) -> HistoryCDiT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = HistoryCDiT(latent_ch=ck["cfg"].get("latent_ch", 16), size=ck["cfg"]["size"], n_tokens=ck["cfg"].get("n_tokens", 64))
    m.load_state_dict(ck["ema"])
    m.eval()
    return m.to(device) if device is not None else m
