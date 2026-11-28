"""Fast zero-privilege gap-head conversion / fine-tuning for an existing WM320 checkpoint (e.g. Changhao's r8/model.pt).

Takes an existing checkpoint (typically trained with legacy offset_mode='along' where dim 0 of _offset predicted
along_mm relative to H*, requiring privileged pin_length_m at test time via gap = -along - L) and converts/retrains
the _offset head so dim 0 directly predicts the zero-privilege visual pin-tip gap:
    gap_mm = -along_mm - L_mm   (positive in front of board face, negative inside hole).

At test time, WMAgent reads float(o[0]) directly with NO pin_length_mm and NO self.L access.
The source checkpoint (--ckpt) is NEVER modified.

Default mode (head-only, frozen encoder/dynamics/policy/value/done):
  - Keeps enc, _dynamics, _reward, _Qs, _Vs, _Qt, _done, _act BIT-FOR-BIT IDENTICAL to --ckpt (verified before save).
  - Computes z and H-step imagined zs under torch.no_grad(), training only _offset (~132k params) for ~2k steps (<5 min on 1 GPU).
  - Logs select & heldout gap MAE, near-face (-3..+3 mm) gap MAE, in-hole (< -0.5 mm) accuracy/AUC, unseen-shape vs
    interior-strip breakdown, and gap_bias_slope_vs_pin_length (OLS slope of (pred_gap - true_gap) vs L_mm; ~0 means
    genuine visual gap readout, ~+1 means frozen z lacks pin-length visual cues and --unfreeze-encoder is needed).

Optional --unfreeze-encoder mode:
  - Jointly fine-tunes enc + _offset (and optionally remaining heads at low LR) using full WM320.loss with offset_mode='gap'.

Example (5-minute head-only proof on fir):
    python world_model/scripts/train/finetune_r8_gap_head.py \
        --ckpt $Q/data/online320_wm320_standard_imnet_off_s0/r8/model.pt \
        --data $Q/data/wm_v1_fast \
        --out  $OUT/r8_gap_head \
        --steps 2000 --batch 32 --shift-px 0 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_wm320 as T  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import PolicyData, shift  # noqa: E402
from wm.tdmpc.wm320 import WM320, WindowData, auc, encoder_settings, window_items  # noqa: E402
from wm.training.common import Logger, cycle, seed_all  # noqa: E402


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Ordinary least-squares slope of y vs x; None if fewer than 10 points or zero x variance."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 10:
        return None
    vx = float(np.var(x))
    if vx < 1e-12:
        return None
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


def init_gap_head_from_along(offset_head: nn.Module, off_scale_0: float, mean_pin_length_mm: float) -> bool:
    """Warm-start dim 0 of an 'along' _offset head to 'gap' = -along - mean_L while keeping dims 1..3 intact.
    Returns True if the final Linear layer was warm-started, False otherwise."""
    if not isinstance(offset_head, nn.Sequential) or not len(offset_head):
        return False
    last = offset_head[-1]
    if not isinstance(last, nn.Linear) or last.weight.shape[0] < 1:
        return False
    with torch.no_grad():
        last.weight[0].neg_()
        if last.bias is not None:
            last.bias[0].neg_().sub_(float(mean_pin_length_mm) / float(off_scale_0))
    return True


def _gap_metrics_from_arrays(pred_enc: np.ndarray, true_enc: np.ndarray, pred_img_last: np.ndarray,
                             true_img_last: np.ndarray, ov_last: np.ndarray, pin_length_mm: np.ndarray,
                             inside_mm: float = 0.5) -> dict:
    """Compute zero-privilege gap-head diagnostics from encoded (N, 4) and last-imagined (N, 4) arrays."""
    err_enc = pred_enc - true_enc
    gap_err_enc = np.abs(err_enc[:, 0])
    lat_err_enc = np.linalg.norm(err_enc[:, 1:3], axis=1)
    tilt_err_enc = np.abs(err_enc[:, 3])
    out = {
        "n": int(len(true_enc)),
        "gap_mae_mm_encoded": float(gap_err_enc.mean()) if len(gap_err_enc) else None,
        "gap_median_err_mm_encoded": float(np.median(gap_err_enc)) if len(gap_err_enc) else None,
        "gap_p90_err_mm_encoded": float(np.percentile(gap_err_enc, 90)) if len(gap_err_enc) else None,
        "lat_err_mm_encoded": float(lat_err_enc.mean()) if len(lat_err_enc) else None,
        "tilt_err_deg_encoded": float(tilt_err_enc.mean()) if len(tilt_err_enc) else None,
        "gap_bias_slope_vs_pin_length": _ols_slope(pin_length_mm, err_enc[:, 0]),
    }
    near = np.abs(true_enc[:, 0]) <= 3.0
    out["near_face_n"] = int(near.sum())
    out["gap_mae_near_face_mm_encoded"] = float(gap_err_enc[near].mean()) if near.any() else None
    y_in = true_enc[:, 0] < -inside_mm
    p_in = pred_enc[:, 0] < -inside_mm
    out["in_hole_acc_0p5mm"] = float((y_in == p_in).mean()) if len(y_in) else None
    out["in_hole_auc_0p5mm"] = auc(-pred_enc[:, 0], y_in.astype(np.float64)) if len(y_in) else None
    m = ov_last > 0.5
    if m.any():
        err_last = pred_img_last[m] - true_img_last[m]
        out["gap_mae_mm_imagined_last"] = float(np.abs(err_last[:, 0]).mean())
        out["lat_err_mm_imagined_last"] = float(np.linalg.norm(err_last[:, 1:3], axis=1).mean())
    else:
        out["gap_mae_mm_imagined_last"] = None
        out["lat_err_mm_imagined_last"] = None
    return out


@torch.no_grad()
def evaluate_gap_head(model: WM320, eval_sets: dict, dev: torch.device, inside_mm: float = 0.5) -> dict:
    """Evaluate encoded + imagined zero-privilege gap head on select / heldout sets (overall + by group)."""
    was_training = model.training
    was_off_training = model._offset.training if model._offset is not None else False
    model.eval()
    out = {}
    for name, (batches, entries, split_mode) in eval_sets.items():
        pe, te, pi, ti, ovi, lm, grps = [], [], [], [], [], [], []
        for W in batches:
            e_idx = W["item"][:, 0].numpy()
            W_dev = T.to(W, dev)
            z = model.observe(W_dev["rgb"], W_dev["depth"], W_dev["q"])        # (B, H+1, L)
            act = W_dev["action"].transpose(0, 1)                               # (H, B, A)
            zs = [z[:, 0]]
            for t in range(WD.H):
                zs.append(model.next(zs[-1], act[t]))
            zs = torch.stack(zs)                                                # (H+1, B, L)
            pr = (model._offset(zs) * model.off_scale).double().cpu().numpy()   # (H+1, B, 4)
            tgt = W["probe"].transpose(0, 1).double().numpy()                   # (H+1, B, 4)
            ov = W["obs_valid"].T.double().numpy()                              # (H+1, B)
            m0 = ov[0] > 0.5
            if not m0.any():
                continue
            pe.append(pr[0][m0])
            te.append(tgt[0][m0])
            pi.append(pr[-1][m0])
            ti.append(tgt[-1][m0])
            ovi.append(ov[-1][m0])
            lm.append(np.array([1000.0 * float(entries[int(e)]["pin_length_m"]) for e in e_idx[m0]], np.float64))
            grps += [D.group320(entries[int(e)]["port"], entries[int(e)]["pitch_deg"],
                                entries[int(e)]["grasp_pos_m"], entries[int(e)]["grasp"], split_mode=split_mode)
                     for e in e_idx[m0]]
        if not pe:
            continue
        pe_a, te_a = np.concatenate(pe, 0), np.concatenate(te, 0)
        pi_a, ti_a = np.concatenate(pi, 0), np.concatenate(ti, 0)
        ovi_a, lm_a = np.concatenate(ovi, 0), np.concatenate(lm, 0)
        grps_a = np.asarray(grps)
        rec = _gap_metrics_from_arrays(pe_a, te_a, pi_a, ti_a, ovi_a, lm_a, inside_mm=inside_mm)
        by_grp = {}
        for g in sorted(set(grps)):
            sel = grps_a == g
            by_grp[g] = _gap_metrics_from_arrays(pe_a[sel], te_a[sel], pi_a[sel], ti_a[sel], ovi_a[sel], lm_a[sel], inside_mm=inside_mm)
        rec["by_group"] = by_grp
        out[name] = rec
    model.train(was_training)
    if model._offset is not None:
        model._offset.train(was_off_training)
    return out


def verify_zero_privilege_checkpoint(ckpt_path: Path, device: str = "cpu") -> dict:
    """Load the saved checkpoint with WMAgent and verify zero-privilege gap_mm() contract."""
    agent = WMAgent(str(ckpt_path), device=device)
    if agent.offset_mode != "gap":
        raise RuntimeError(f"saved checkpoint {ckpt_path} loaded with offset_mode={agent.offset_mode!r}, expected 'gap'")
    if agent.m._offset is None:
        raise RuntimeError(f"saved checkpoint {ckpt_path} has _offset=None")
    z = torch.zeros(1, agent.m.cfg.latent_dim, device=agent.device)
    g = agent.gap_mm(z)
    o = agent.offset_mm(z)
    if g is None or o is None or abs(g - float(o[0])) > 1e-6:
        raise RuntimeError(f"gap_mm(z)={g} did not match float(offset_mm(z)[0])={None if o is None else float(o[0])}")
    try:
        agent.gap_mm(z, pin_length_mm=16.0)
    except ValueError:
        pass
    else:
        raise RuntimeError("gap_mm(z, pin_length_mm=16.0) did NOT raise ValueError in offset_mode='gap'")
    return {"offset_mode": agent.offset_mode, "dummy_gap_mm": round(g, 4), "rejects_pin_length_mm": True}


def main():
    ap = argparse.ArgumentParser(description="Convert / fine-tune WM320 checkpoint _offset head to zero-privilege offset_mode='gap'.")
    ap.add_argument("--ckpt", required=True, help="source WM320 checkpoint (e.g. r8/model.pt); never modified")
    ap.add_argument("--data", required=True, help="wm_v1_fast directory")
    ap.add_argument("--out", required=True, help="output directory for converted checkpoint and report")
    ap.add_argument("--online", help="optional online pool directory (training grasps only) to mix in")
    ap.add_argument("--online-frac", type=float, default=0.5, help="share of windows per micro-batch from --online")
    ap.add_argument("--online-min", type=int, default=100, help="minimum online trajectories before mixing")
    ap.add_argument("--split-mode", choices=list(D.SPLIT_MODES), default="split320")
    ap.add_argument("--init", choices=["from_ckpt", "scratch"], default="from_ckpt",
                    help="from_ckpt = warm-start dim 0 (-along - mean_L) keeping dims 1..3; scratch = fresh _offset MLP")
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="also unfreeze encoder with joint WM320.loss (default: freeze everything except _offset)")
    ap.add_argument("--enc-lr", type=float, default=3e-5, help="encoder LR when --unfreeze-encoder is set")
    ap.add_argument("--bc-batch", type=int, default=32, help="BC micro-batch size when --unfreeze-encoder is set")
    ap.add_argument("--bc-weight", type=float, default=1.0, help="BC weight when --unfreeze-encoder is set")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--shift-px", type=int, default=0, help="random pixel shift (0 = recommended clean setting)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inside-mm", type=float, default=0.5, help="reference threshold for in-hole classification diagnostics")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    src_ckpt = Path(args.ckpt).resolve()
    out = Path(args.out).resolve()
    if out == src_ckpt.parent or (out / "model.pt").resolve() == src_ckpt:
        raise SystemExit(f"refusing to write into source checkpoint directory {src_ckpt.parent}; choose a fresh --out directory")
    if args.steps < 1 or args.batch < 1 or args.accum < 1:
        raise SystemExit("--steps, --batch and --accum must be >= 1")

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    seed_all(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / "model.pt").exists():
        raise SystemExit(f"{out}/model.pt already exists; use a fresh --out or pass --resume")

    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    if ck.get("labels") != WD.LABELS:
        raise SystemExit(f"{src_ckpt} has labels={ck.get('labels')!r}, expected {WD.LABELS!r}")
    src_offset_mode = str(ck.get("offset_mode", "along"))
    src_offset_weight = float(ck.get("offset_weight", 0.0))
    shaping = float(ck.get("shaping", 0.0))
    gamma = float(ck["cfg"]["gamma"])
    stats = ck["stats"]
    out_offset_weight = src_offset_weight if src_offset_weight > 0 else 1.0

    ents = [e for e in D.load_entries(args.data, split_mode=args.split_mode) if e["split"] != "test" and e["steps"] >= WD.EXECUTE]
    by = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}
    if not by["train"]:
        raise SystemExit(f"no training trajectories found in {args.data} for split_mode={args.split_mode}")
    mean_L_mm = float(np.mean([1000.0 * float(e["pin_length_m"]) for e in by["train"]]))

    torch.manual_seed(args.seed)
    model = WM320(ck["stem"], stats, shaping, imagenet=ck.get("imagenet", False),
                  offset_weight=out_offset_weight, offset_mode="gap",
                  max_pin_mm=ck.get("max_pin_mm") or 20.0, **encoder_settings(ck), **ck["cfg"])
    if src_offset_weight > 0:
        model.load_state_dict(ck["state"], strict=True)
    else:
        res = model.load_state_dict(ck["state"], strict=False)
        bad_missing = [k for k in res.missing_keys if not k.startswith("_offset.")]
        if bad_missing or res.unexpected_keys:
            raise RuntimeError(f"unexpected checkpoint mismatch: missing={bad_missing}, unexpected={res.unexpected_keys}")

    warm_started = False
    if args.init == "scratch" and src_offset_weight > 0:
        from common import init as td_init
        model._offset.apply(td_init.weight_init)
    elif args.init == "from_ckpt" and src_offset_weight > 0 and src_offset_mode == "along":
        warm_started = init_gap_head_from_along(model._offset, float(model.off_scale[0]), mean_L_mm)

    frozen_ref = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("_offset.") and torch.is_tensor(v)}
    model = model.to(dev)

    if args.unfreeze_encoder:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model._offset.parameters():
            p.requires_grad_(True)
        for p in model.enc.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([
            {"params": list(model._offset.parameters()), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": list(model.enc.parameters()), "lr": args.enc_lr, "weight_decay": args.weight_decay},
        ])
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model._offset.parameters():
            p.requires_grad_(True)
        model.eval()
        model._offset.train()
        opt = torch.optim.AdamW(list(model._offset.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)
    step, best_select_mae = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        rck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(rck["state"])
        opt.load_state_dict(rck["opt"])
        if "sched" in rck:
            sched.load_state_dict(rck["sched"])
        step = int(rck.get("finetune_step", 0))
        best_select_mae = float(rck.get("best_select_mae", float("inf")))

    # Build fixed validation sets (select + heldout)
    rng = np.random.default_rng(1)
    eval_sets = {}
    for name in ("select", "heldout"):
        if not by[name]:
            continue
        wi = window_items(by[name])
        wi = wi[np.sort(rng.choice(len(wi), min(args.eval_windows, len(wi)), replace=False))]
        ds = WindowData(args.data, by[name], wi, stats, shaping, gamma, offset_mode="gap")
        dl = T.loader(ds, min(16, args.batch), min(4, args.workers), 0, shuffle=False)
        eval_sets[name] = ([{k: v.clone() for k, v in b.items()} for b in dl], by[name], args.split_mode)

    # Training loaders
    onl, mix = [], False
    wo = int(round(args.online_frac * args.batch)) if args.online else 0
    if args.online:
        onl, _ = T.online_entries(args.online, split_mode=args.split_mode)
        mix = len(onl) >= args.online_min and len(window_items(onl)) >= wo and 0 < wo < args.batch
    ws = WindowData(args.data, by["train"], window_items(by["train"]), stats, shaping, gamma, offset_mode="gap")
    wtrain = [cycle(T.loader(ws, args.batch - (wo if mix else 0), args.workers, args.seed + step))]
    if mix:
        oset = WindowData(args.data, onl, window_items(onl), stats, shaping, gamma, offset_mode="gap")
        wtrain.append(cycle(T.loader(oset, wo, max(1, args.workers // 2) if args.workers else 0, args.seed + step + 3)))
    btrain = None
    if args.unfreeze_encoder:
        bce = [e for e in by["train"] if e["bc"]] + ([e for e in onl if e["bc"]] if mix else [])
        btrain = cycle(T.loader(PolicyData(args.data, bce, T.bc_items(bce)), args.bc_batch, args.workers, args.seed + step + 1))

    info0 = {
        "format": ck.get("format", "wm320-1"),
        "labels": WD.LABELS,
        "stem": ck["stem"],
        "imagenet": ck.get("imagenet", False),
        "offset_weight": out_offset_weight,
        **encoder_settings(ck),
        "offset_mode": "gap",
        "split_mode": args.split_mode,
        "tilt_definition": ck.get("tilt_definition", "full_angle"),
        "term_mode": ck.get("term_mode", "last_step"),
        "max_pin_mm": float(ck.get("max_pin_mm") or 20.0),
        "shaping": shaping,
        "cfg": ck["cfg"],
        "stats": stats,
        "source_ckpt": str(src_ckpt),
        "source_step": int(ck.get("step", -1)),
        "source_offset_mode": src_offset_mode,
        "finetune_mode": "unfreeze_encoder" if args.unfreeze_encoder else "head_only",
    }

    def build_ckpt(full: bool) -> dict:
        c = {
            **info0,
            "state": model.state_dict(),
            "step": int(ck.get("step", -1)),
            "finetune_step": step,
            "best_select_mae": best_select_mae,
            "args": vars(args),
        }
        if full:
            c.update(opt=opt.state_dict(), sched=sched.state_dict())
        return c

    log = Logger(out)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log(event="start", source_ckpt=str(src_ckpt), source_step=int(ck.get("step", -1)),
        source_offset_mode=src_offset_mode, warm_started=warm_started, mean_pin_length_mm=round(mean_L_mm, 3),
        unfreeze_encoder=args.unfreeze_encoder, trainable_params=sum(p.numel() for p in trainable_params),
        online_mixed=mix, online_trajectories=len(onl), args=vars(args))

    eval_init = evaluate_gap_head(model, eval_sets, dev, inside_mm=args.inside_mm) if step == 0 else None
    if eval_init is not None:
        log(event="val", step=0, **eval_init)

    acc, t0, event = defaultdict(list), time.time(), "done"
    stopper = T.GracefulStopper().install()
    try:
        while step < args.steps:
            if stopper.requested:
                T.save(build_ckpt(True), out / "last.pt")
                event = stopper.reason or "signal_stop"
                break
            step += 1
            opt.zero_grad(set_to_none=True)
            finite = True
            for _ in range(args.accum):
                W = T.to(next(wtrain[0]), dev) if len(wtrain) == 1 else T.cat(T.to(next(wtrain[0]), dev), T.to(next(wtrain[1]), dev))
                W["rgb"], W["depth"] = shift(W["rgb"], W["depth"], args.shift_px)
                if args.unfreeze_encoder:
                    Bc = T.to(next(btrain), dev)
                    Bc["rgb"], Bc["depth"] = shift(Bc["rgb"], Bc["depth"], args.shift_px)
                    loss, info = model.loss(W, Bc, step=10**9, warmup=0, bc_weight=args.bc_weight)
                    (loss / args.accum).backward()
                    finite &= math.isfinite(info["total"])
                    for k, v in info.items():
                        acc[k].append(v)
                else:
                    B = W["q"].shape[0]
                    with torch.no_grad():
                        z = model.observe(W["rgb"], W["depth"], W["q"])            # (B, H+1, L)
                        act = W["action"].transpose(0, 1)                           # (H, B, A)
                        zs = [z[:, 0]]
                        for t in range(WD.H):
                            zs.append(model.next(zs[-1], act[t]))
                        zs = torch.stack(zs)                                        # (H+1, B, L)
                    tgt = W["probe"].transpose(0, 1) / model.off_scale              # (H+1, B, 4)
                    ov = W["obs_valid"].T                                           # (H+1, B)
                    pr = model._offset(zs)
                    per = F.smooth_l1_loss(pr, tgt, reduction="none").mean(-1)
                    ol = sum((per[t] * ov[t]).sum() / B * model.cfg.rho ** t for t in range(WD.H + 1)) / (WD.H + 1)
                    (ol / args.accum).backward()
                    finite &= math.isfinite(float(ol))
                    with torch.no_grad():
                        err = (pr - tgt) * model.off_scale
                        lat = err[..., 1:3].norm(dim=-1)
                        acc["offset_loss"].append(float(ol))
                        acc["offset_gap_err_mm_encoded"].append(float((err[0, :, 0].abs() * ov[0]).sum() / ov[0].sum().clamp(min=1)))
                        acc["offset_gap_err_mm_imagined_last"].append(float((err[-1, :, 0].abs() * ov[-1]).sum() / ov[-1].sum().clamp(min=1)))
                        acc["offset_lat_err_mm_encoded"].append(float((lat[0] * ov[0]).sum() / ov[0].sum().clamp(min=1)))
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_params, 20.0))
            if finite and math.isfinite(gn):
                opt.step()
                sched.step()
            acc["grad_norm"].append(gn)
            if step % 100 == 0 or step == args.steps:
                log(step=step, lr=float(opt.param_groups[0]["lr"]), it_per_s=len(acc["grad_norm"]) / max(time.time() - t0, 1e-6),
                    **{k: float(np.mean(v)) for k, v in sorted(acc.items())})
                acc, t0 = defaultdict(list), time.time()
            if stopper.requested:
                T.save(build_ckpt(True), out / "last.pt")
                event = stopper.reason or "signal_stop"
                break
            if step % args.save_every == 0:
                T.save(build_ckpt(True), out / "last.pt")
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate_gap_head(model, eval_sets, dev, inside_mm=args.inside_mm)
                log(event="val", step=step, **ev)
                sel_rec = ev.get("select") or ev.get("heldout") or {}
                score = sel_rec.get("gap_mae_near_face_mm_encoded")
                if score is None:
                    score = sel_rec.get("gap_mae_mm_encoded")
                if score is not None and float(score) < best_select_mae:
                    best_select_mae = float(score)
                    T.save(build_ckpt(False), out / "best_select.pt")
                t0 = time.time()
    finally:
        stopper.restore()

    if event.startswith("signal_"):
        log(event=event, step=step)
        return

    # Verify frozen backbone integrity when head_only
    frozen_unchanged = None
    if not args.unfreeze_encoder:
        cur = model.state_dict()
        changed = [k for k, v in frozen_ref.items() if not torch.equal(v, cur[k].detach().cpu())]
        if changed:
            raise RuntimeError(f"head_only violation: frozen parameters changed: {changed[:5]}")
        frozen_unchanged = True

    T.save(build_ckpt(True), out / "last.pt")
    T.save(build_ckpt(False), out / "model.pt")
    eval_final = evaluate_gap_head(model, eval_sets, dev, inside_mm=args.inside_mm)
    zp_check = verify_zero_privilege_checkpoint(out / "model.pt", device="cpu")

    report = {
        "source_ckpt": str(src_ckpt),
        "out_ckpt": str(out / "model.pt"),
        "best_select_ckpt": str(out / "best_select.pt") if (out / "best_select.pt").exists() else None,
        "finetune_mode": info0["finetune_mode"],
        "steps": step,
        "split_mode": args.split_mode,
        "shift_px": args.shift_px,
        "warm_started": warm_started,
        "mean_pin_length_mm": mean_L_mm,
        "frozen_backbone_unchanged": frozen_unchanged,
        "zero_privilege_check": zp_check,
        "eval_before": eval_init,
        "eval_after": eval_final,
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2))
    log(event="done", step=step, zero_privilege_check=zp_check)
    print(f"Saved zero-privilege gap checkpoint: {out / 'model.pt'}", flush=True)
    for split_name, rec in eval_final.items():
        print(f"  [{split_name}] gap MAE={rec['gap_mae_mm_encoded']:.3f} mm "
              f"(near-face={rec['gap_mae_near_face_mm_encoded']}, img_last={rec['gap_mae_mm_imagined_last']}), "
              f"in-hole acc={rec['in_hole_acc_0p5mm']}, AUC={rec['in_hole_auc_0p5mm']}, "
              f"lat={rec['lat_err_mm_encoded']:.3f} mm, slope_vs_L={rec['gap_bias_slope_vs_pin_length']}", flush=True)


if __name__ == "__main__":
    main()
