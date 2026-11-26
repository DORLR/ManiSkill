"""Train `VideoWM320` (`D2a` / `D2b` / `D2c`) offline or with online round mixing on `fir` / multi-GPU DDP.

Features:
  - Unified CLI across `D2a` (`d2a_segformer`, `d2a_resnet34`, `d2a_resnet18`), `D2b` (`d2b_segformer`,
    `d2b_resnet18`, `d2b_resnet34`), and `D2c` (`d2c_vjepa2_vitl`, `d2c_vjepa2_ac_vitg`).
  - Zero-privilege `--offset-mode gap` default (1D visual pin-tip-to-board gap head + strictly detached
    stop-gradient 4D B1 diagnostic probe), plus `--offset-mode probe` and `--offset-mode none`.
  - `--shift-px 0` default (configurable for shift ablations).
  - Exact compatibility with Changhao's `WindowData`, `PolicyData`, fast copy (`wm_v1_fast`), and
    streaming online pools (`--online <pool>`, complete `r<R>/DONE` rounds + partial finished grasps).
  - Slurm job chaining (`--max-steps-this-job N`) and DDP-synchronized `SIGTERM` / `SIGUSR1` clean
    atomic checkpointing (`last.pt` saved cleanly without writing `DONE.json` prematurely).
  - Logs `B1`-ready factual fidelity metrics (face/inside lateral & along errors at `k=1,3`,
    through-board penetration share from both `_offset` and zero-privilege `_gap`, reward MAE,
    done AUC, V-before-face AUC, and ACT action error) at every validation step!

Example (single `3g.40gb` H100 MIG on `fir`):
    python experiments/fir_wm320_20261123/d2_video_wm/train_video_wm320.py \\
        --data $SCRATCH/data/wm_v1_fast --out $SCRATCH/runs/d2b_segformer_gap \\
        --arch d2b_segformer --offset-mode gap --shift-px 0 --shaping 0.3 \\
        --wm-batch 16 --bc-batch 32 --accum 8 --max-steps-this-job 25000 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def _ensure_wm_src(wm_src_arg: Optional[str] = None) -> None:
    """Add `world_model/src` to `sys.path` if not already importable."""
    candidates: List[Path] = []
    if wm_src_arg:
        candidates.append(Path(wm_src_arg))
    if os.environ.get("WM_SRC"):
        candidates.append(Path(os.environ["WM_SRC"]))
    candidates.append(_REPO_ROOT / "task_release" / "ManiSkill-WMInsertion-v2" / "world_model" / "src")
    for p in candidates:
        if (p / "wm" / "tdmpc" / "wm320.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


# Bootstrap `wm` path early if standard layout is present
_ensure_wm_src()

from video_wm320 import (  # noqa: E402
    ARCH_PRESETS,
    DEFAULT_SEGFORMER_ID,
    DEFAULT_VJEPA2_VITL_ID,
    VIDEO_WM_FORMAT,
    VideoWM320,
    VideoWindowData,
)

RS = 1_000_003       # seed stride between DDP ranks
IMITATE = (0, 1)     # online decisions imitated by built-in ACT head (proposal, correction)
BACKOFF = 4          # attempt320.KINDS["backoff"]
BLOCKED_MM = 0.2
BLOCKED3_MM = 0.5
THROUGH_MM = 1.0


def save_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_loader(ds, batch: int, workers: int, seed: int, shuffle: bool = True):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=g if shuffle else None,
        drop_last=shuffle,
        num_workers=workers,
        persistent_workers=shuffle and workers > 0,
        prefetch_factor=2 if workers else None,
        pin_memory=torch.cuda.is_available(),
    )


def _finished_grasps(r: Path) -> List[dict]:
    out: List[dict] = []
    for f in sorted(r.glob("results/*/*.json")):
        try:
            s = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if not s.get("error"):
            out += s.get("entries", [])
    return out


def online_entries(pool: str | Path, D, WD) -> Tuple[List[dict], List[str]]:
    out: List[dict] = []
    rounds: List[str] = []
    for r in sorted(Path(pool).glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not r.name[1:].isdigit():
            continue
        if (r / "DONE").exists() and (r / "manifest.json").exists():
            rounds.append(r.name)
            ents = json.loads((r / "manifest.json").read_text())["entries"]
            partial = False
        else:
            ents, partial = _finished_grasps(r), True
        for e in ents:
            sp = D.split320(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"])
            if sp != "train":
                raise SystemExit(
                    f"online trajectory {e['path']} is of split {sp}: the pool must hold training grasps only"
                )
            if e["steps"] >= WD.EXECUTE and (not partial or (Path(pool) / e["path"]).exists()):
                out.append(
                    {
                        **e,
                        "path": str(Path(pool) / e["path"]),
                        "split": sp,
                        "bc": e.get("outcome") == "seated_fk",
                        "partial": partial,
                    }
                )
    return out, rounds


def bc_items(entries: List[dict], D, WD) -> np.ndarray:
    rows = []
    for i, e in enumerate(entries):
        kinds = e.get("dec_kind")
        if kinds is None:
            rows += [(i, j) for j in range(e["n_images"])]
            continue
        if not kinds:
            continue
        first = max((d for d, k in enumerate(kinds) if k == BACKOFF), default=-1) + 1
        for j in range(e["n_images"]):
            d0 = min(j // 2, len(kinds) - 1)
            if d0 >= first and all(
                kinds[d] in IMITATE for d in range(d0, min(len(kinds), (4 * j + D.CHUNK - 1) // WD.EXECUTE + 1))
            ):
                rows.append((i, j))
    return np.array(rows, np.int64).reshape(-1, 2)


def cat_batches(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


def to_device(b: dict, dev: torch.device) -> dict:
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def _phase_of(gap_mm: np.ndarray, seated: np.ndarray) -> np.ndarray:
    """Exact B1 phase definition from `fidelity_factual320.py`: 0=approach, 1=face, 2=inside, 3=seated."""
    p = np.where(gap_mm > 2.0, 0, np.where(gap_mm > -1.0, 1, 2))
    return np.where(seated > 0.5, 3, p)


@torch.no_grad()
def validate(model: VideoWM320, sets: dict, args, dev: torch.device, auc_fn) -> dict:
    """Run validation on `select` and `heldout`, logging both legacy `wm320` AND live `B1`-ready metrics."""
    model.eval()
    cpu = torch.random.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(1234)
    out: Dict[str, dict] = {}

    for name, (wl, fl, bl) in sets.items():
        num = defaultdict(list)
        done_s, done_y, di_s, di_y, v0, end = [], [], [], [], [], []
        # B1-ready accumulators across validation windows
        b1_rows: Dict[str, List[np.ndarray]] = defaultdict(list)

        for W in wl:
            W = to_device(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            for k, v in info.items():
                if not k.startswith("_"):
                    num[k].append(v)

            obs_v = W["obs_valid"].cpu().numpy()
            fk_d = W["fk_depth"].cpu().numpy()
            done_gt = W["done"].cpu().numpy()
            m = (obs_v > 0) & (fk_d > 0)
            done_s.append(info["_done"][m])
            done_y.append(done_gt[m])
            di_s.append(info["_done_imag"][m[:, 1:]])
            di_y.append(done_gt[:, 1:][m[:, 1:]])

            # B1 factual fidelity diagnostics (face & inside phases, k=1 and k=3, through-board share)
            # Use unambiguous `probe_along` + `gap_mm` from VideoWindowData and align sign with D1 fidelity_factual320!
            pr_along = (W["probe_along"] if "probe_along" in W else W["probe"]).transpose(0, 1).double().cpu().numpy()
            valid = W["valid"].T.double().cpu().numpy()                  # (H, B)
            ov_t = obs_v.T                                               # (H+1, B)
            rew_gt = W["reward"].T.double().cpu().numpy()                # (H, B)
            if "gap_mm" in W:
                gap0 = W["gap_mm"][:, 0].double().cpu().numpy()
            else:
                L_mm = W["pin_length_mm"].double().cpu().numpy()
                gap0 = -pr_along[0, :, 0] - L_mm
            ph = _phase_of(gap0, done_gt[:, 0])
            chain = np.cumprod(valid, axis=0)                            # (H, B)

            b1_rows["ph"].append(ph)
            b1_rows["ok1"].append((chain[0] > 0) & (ov_t[1] > 0))
            b1_rows["ok3"].append((chain[2] > 0) & (ov_t[3] > 0))
            b1_rows["prog_true_k1"].append(pr_along[1, :, 0] - pr_along[0, :, 0])
            b1_rows["prog_true_k3"].append(pr_along[3, :, 0] - pr_along[0, :, 0])
            b1_rows["rew_err_k1"].append(np.abs(info["_rew_img"][0] - rew_gt[0]))
            b1_rows["rew_err_k3"].append(np.abs(info["_rew_img"][2] - rew_gt[2]))

            if "_off_img" in info and "_off_enc" in info:
                off_img = info["_off_img"].astype(np.float64)            # (H+1, B, 4)
                off_enc = info["_off_enc"].astype(np.float64)            # (H+1, B, 4)
                # Sign alignment matching D1 fidelity_factual320.along_progress_from_delta:
                # In 'gap' mode, dim 0 is gap_mm so along_progress = -(g(t+k) - g(t)); in 'along' mode, +(a(t+k) - a(t)).
                dim0_sign = -1.0 if model.offset_mode == "gap" else 1.0
                for j in (1, 3):
                    dt = pr_along[j] - pr_along[0]
                    dw = off_img[j] - off_enc[0]
                    de = off_enc[j] - off_enc[0]
                    prog_wm = dim0_sign * dw[:, 0]
                    b1_rows[f"lat_wm_k{j}"].append(np.linalg.norm((dw - dt)[:, 1:3], axis=1))
                    b1_rows[f"lat_enc_k{j}"].append(np.linalg.norm((de - dt)[:, 1:3], axis=1))
                    b1_rows[f"lat_zero_k{j}"].append(np.linalg.norm(dt[:, 1:3], axis=1))
                    b1_rows[f"al_wm_k{j}"].append(np.abs(prog_wm - dt[:, 0]))
                    b1_rows[f"prog_wm_k{j}"].append(prog_wm)

            if "_gap_img" in info and "_gap_enc" in info:
                # Zero-privilege axial progress directly from 1D `_gap` head: delta_along = -(gap(t+k) - gap(t))
                gap_img = info["_gap_img"].astype(np.float64)            # (H+1, B)
                gap_enc = info["_gap_enc"].astype(np.float64)            # (H+1, B)
                for j in (1, 3):
                    b1_rows[f"gap_prog_wm_k{j}"].append(-(gap_img[j] - gap_enc[0]))

        for W in fl:
            W = to_device(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            v0.append(info["_v0"])
            end.append(W["end_seated"].cpu().numpy())

        err = []
        for b in bl:
            b = to_device(b, dev)
            zb = model.encode_obs(b["rgb"], b["depth"], b["q"])
            pred = model.act_chunk(zb).flatten(1)
            err.append(float((pred - b["a"]).abs().mean()) * 1000.0)

        rec: Dict[str, Any] = {k: float(np.mean(v)) for k, v in num.items()}
        if done_y:
            rec["auc_done_depth_pos"] = auc_fn(np.concatenate(done_s), np.concatenate(done_y))
            rec["auc_done_imagined_depth_pos"] = auc_fn(np.concatenate(di_s), np.concatenate(di_y))
        if v0:
            rec["auc_V_before_face_contact"] = auc_fn(np.concatenate(v0), np.concatenate(end) > 0.5)
        if err:
            rec["act_err_mrad"] = float(np.mean(err))

        # Summarise live B1 factual fidelity metrics
        if b1_rows["ph"]:
            R = {k: np.concatenate(v, axis=0) for k, v in b1_rows.items()}
            face1 = R["ok1"] & (R["ph"] == 1)
            face3 = R["ok3"] & (R["ph"] == 1)
            ins1 = R["ok1"] & (R["ph"] == 2)
            ins3 = R["ok3"] & (R["ph"] == 2)

            if face1.any():
                rec["b1_face_rew_mae_k1"] = float(R["rew_err_k1"][face1].mean())
            if face3.any():
                rec["b1_face_rew_mae_k3"] = float(R["rew_err_k3"][face3].mean())

            if "lat_wm_k1" in R:
                for p_name, sel1, sel3 in (("face", face1, face3), ("inside", ins1, ins3)):
                    if sel1.any():
                        rec[f"b1_{p_name}_k1_lat_wm_mm"] = float(R["lat_wm_k1"][sel1].mean())
                        rec[f"b1_{p_name}_k1_lat_enc_mm"] = float(R["lat_enc_k1"][sel1].mean())
                        rec[f"b1_{p_name}_k1_lat_zero_mm"] = float(R["lat_zero_k1"][sel1].mean())
                    if sel3.any():
                        rec[f"b1_{p_name}_k3_lat_wm_mm"] = float(R["lat_wm_k3"][sel3].mean())
                        rec[f"b1_{p_name}_k3_lat_enc_mm"] = float(R["lat_enc_k3"][sel3].mean())
                        rec[f"b1_{p_name}_k3_lat_zero_mm"] = float(R["lat_zero_k3"][sel3].mean())
                if face1.any():
                    rec["b1_face_al_wm_k1_mm"] = float(R["al_wm_k1"][face1].mean())
                    b1 = face1 & (np.abs(R["prog_true_k1"]) < BLOCKED_MM)
                    if b1.any():
                        rec["b1_face_through_board_k1"] = float((R["prog_wm_k1"][b1] > THROUGH_MM).mean())
                if face3.any():
                    rec["b1_face_al_wm_k3_mm"] = float(R["al_wm_k3"][face3].mean())
                    b3 = face3 & (np.abs(R["prog_true_k3"]) < BLOCKED3_MM)
                    if b3.any():
                        rec["b1_face_through_board_k3"] = float((R["prog_wm_k3"][b3] > THROUGH_MM).mean())

            if "gap_prog_wm_k1" in R:
                b1 = face1 & (np.abs(R["prog_true_k1"]) < BLOCKED_MM)
                if b1.any():
                    rec["b1_face_gap_through_board_k1"] = float((R["gap_prog_wm_k1"][b1] > THROUGH_MM).mean())
                b3 = face3 & (np.abs(R["prog_true_k3"]) < BLOCKED3_MM)
                if b3.any():
                    rec["b1_face_gap_through_board_k3"] = float((R["gap_prog_wm_k3"][b3] > THROUGH_MM).mean())

        out[name] = rec

    torch.random.set_rng_state(cpu)
    if gpu is not None:
        torch.cuda.set_rng_state_all(gpu)
    model.train()
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Train VideoWM320 (D2a / D2b / D2c) with zero-privilege gap & B1 metrics")
    ap.add_argument("--data", required=True, help="Fast-copy dataset root (wm_v1_fast)")
    ap.add_argument("--out", required=True, help="Output run directory")
    ap.add_argument("--wm-src", default=None, help="Optional path to world_model/src if not auto-detected")
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_PRESETS.keys()))

    # Architecture overrides
    ap.add_argument("--latent-grid", type=int, default=None, help="Spatial latent grid G (K=G*G) for D2b/D2c (e.g. 5, 8, 10)")
    ap.add_argument("--token-dim", type=int, default=None, help="Per-token dim D_z for D2b/D2c (must be divisible by 8)")
    ap.add_argument("--dit-dim", type=int, default=None, help="ST-DiT hidden width (default 384)")
    ap.add_argument("--dit-layers", type=int, default=None, help="ST-DiT block depth (default 4)")
    ap.add_argument("--dit-heads", type=int, default=None, help="ST-DiT attention heads (default 6)")
    ap.add_argument("--vjepa2-unfreeze-blocks", type=int, default=None, help="Top N V-JEPA 2 blocks to unfreeze (0 = frozen)")
    ap.add_argument("--detach-target-obs", type=int, choices=[0, 1], default=None, help="Override detach_target_obs (1=save VRAM)")
    ap.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing on SegFormer backbone")
    ap.add_argument("--no-pretrained", action="store_true", help="Initialize encoder from scratch (smoke tests / ablations)")
    ap.add_argument("--local-files-only", action="store_true", help="Never contact HF Hub (for offline fir compute nodes)")
    ap.add_argument("--segformer-model-id", default=DEFAULT_SEGFORMER_ID)
    ap.add_argument("--vjepa2-model-id", default=DEFAULT_VJEPA2_VITL_ID)

    # Zero-privilege gap head & loss weights (100% harmonized with D1 overlay)
    ap.add_argument(
        "--offset-mode",
        default="gap",
        choices=["gap", "gap_1d", "along", "probe", "none"],
        help=(
            "'gap' (default, D1b parity: 4D (gap_mm, lx, ly, tilt) with grads, zero self.L at test time), "
            "'gap_1d' (ablation: 1D gap head with grads + detached 4D B1 probe), "
            "'along' (alias 'probe': legacy 4D along_mm head), or 'none'"
        ),
    )
    ap.add_argument("--offset-weight", type=float, default=1.0, help="Weight on 4D/1D offset/gap head")
    ap.add_argument("--no-sg-probe", action="store_true",
                    help="Disable the strictly stop-gradient 4D B1 diagnostic probe in gap_1d/none mode")

    # Data & optimization (mirrors train_wm320.py)
    ap.add_argument("--shift-px", type=int, default=0, help="Random shift augmentation pixels (default 0)")
    ap.add_argument("--shaping", type=float, default=0.3, help="Potential-based shaping weight")
    ap.add_argument("--steps", type=int, default=100000, help="Total target optimizer steps")
    ap.add_argument("--max-steps-this-job", type=int, default=0,
                    help=">0: stop cleanly after this many optimizer steps in this Slurm job (saves last.pt, no DONE.json)")
    ap.add_argument("--wm-batch", type=int, default=16)
    ap.add_argument("--bc-batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-scale", type=float, default=0.3)
    ap.add_argument("--pretrained-lr-scale", type=float, default=None,
                    help="Override pretrained trunk LR scale (default: 0.1 for V-JEPA 2, else enc-lr-scale)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=20.0)
    ap.add_argument("--td-warmup", type=int, default=5000)
    ap.add_argument(
        "--cons-warmup-steps",
        type=int,
        default=None,
        help="Linear consistency_coef warmup steps (0.5 -> 20.0); default 1000 for spatial_latent (D2b/D2c), 0 for D2a",
    )
    ap.add_argument("--bc-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1 - 1 / 30)
    ap.add_argument("--workers-wm", type=int, default=8)
    ap.add_argument("--workers-bc", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-windows", type=int, default=1024)
    ap.add_argument("--keep-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")

    # Online round mixing (exact Changhao online320 pool schema)
    ap.add_argument("--online", default=None, help="Online pool directory; complete rounds + partial finished grasps mixed in")
    ap.add_argument("--online-frac", type=float, default=0.5)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--online-min", type=int, default=100)
    ap.add_argument("--online-reload", type=int, default=150)
    return ap.parse_args()


def main():
    args = parse_args()
    _ensure_wm_src(args.wm_src)

    from wm.tdmpc import demo320 as D
    from wm.tdmpc import wmdata320 as WD
    from wm.tdmpc.policy320 import PolicyData, policy_loader, shift
    from wm.tdmpc.wm320 import auc, face_items, stats_for, window_items
    from wm.training.common import Logger, cycle, seed_all

    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass

    if args.offset_weight < 0:
        raise SystemExit("--offset-weight must be >= 0")
    if args.offset_mode == "probe":
        args.offset_mode = "along"
    wire_offset_mode = "along" if args.offset_mode == "along" else "gap"

    # Register Slurm SIGTERM / SIGUSR1 clean checkpoint handlers
    sig_state = {"received": 0}

    def _on_signal(signum, _frame):
        sig_state["received"] = int(signum)

    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_signal)

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=torch.device("cuda", local))
        rank, world, dev = dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)
        frac = os.environ.get("WM_GPU_MEM_FRACTION")
        if frac:
            torch.cuda.set_per_process_memory_fraction(float(frac), local)
        random.seed(args.seed + RS * rank)
        np.random.seed(args.seed + RS * rank)
        torch.manual_seed(args.seed + RS * rank)
    else:
        rank, world, dev = 0, 1, torch.device(args.device)
        seed_all(args.seed)
    main_rank = rank == 0

    run = Path(args.out)
    resume = args.resume and (run / "last.pt").exists()
    refuse = (
        f"{run} is finished"
        if (run / "DONE.json").exists()
        else f"{run} already holds a run; use --resume to continue it"
        if (not args.resume and (run / "log.jsonl").exists())
        else None
    )
    if ddp:
        dist.barrier()
    if refuse:
        raise SystemExit(refuse)
    if main_rank:
        run.mkdir(parents=True, exist_ok=True)

    ents = [e for e in D.load_entries(args.data) if e["split"] != "test" and e["steps"] >= WD.EXECUTE]
    by = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}

    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if resume else None
    if ck is not None:
        if ck.get("format") != VIDEO_WM_FORMAT:
            raise SystemExit(f"{run}/last.pt has format={ck.get('format')!r}, expected {VIDEO_WM_FORMAT!r}")
        if ck.get("labels") != WD.LABELS:
            raise SystemExit(f"{run}/last.pt was trained with labels {ck.get('labels')!r}, code has {WD.LABELS!r}")
        if ck.get("arch") != args.arch:
            raise SystemExit(f"{run}/last.pt has arch={ck.get('arch')!r}, --arch is {args.arch!r}")
        ck_vmode = ck.get("video_offset_mode", ck.get("offset_mode", "along"))
        if ck_vmode == "probe":
            ck_vmode = "along"
        if ck_vmode != args.offset_mode:
            raise SystemExit(
                f"{run}/last.pt has video_offset_mode={ck_vmode!r}, --offset-mode is {args.offset_mode!r}"
            )
        if float(ck.get("offset_weight_train", ck.get("offset_weight", 0.0))) != float(args.offset_weight):
            raise SystemExit(f"{run}/last.pt offset_weight mismatch vs --offset-weight {args.offset_weight}")
        stats = ck["stats"]
    else:
        box = [stats_for(args.data, by["train"], seed=args.seed) if main_rank else None]
        if ddp:
            dist.broadcast_object_list(box, src=0)
        stats = box[0]

    # Fixed validation subsets (rank 0 only)
    rng = np.random.default_rng(1)
    sets = {}
    for name in ("select", "heldout") if main_rank else ():
        if not by[name]:
            continue
        wi = window_items(by[name])
        wi = wi[np.sort(rng.choice(len(wi), min(args.eval_windows, len(wi)), replace=False))]
        fi = face_items(by[name], args.data, args.shaping, args.gamma)
        bce = [e for e in by[name] if e["bc"]]
        bi = D.policy_items(bce)
        bi = bi[np.sort(rng.choice(len(bi), min(1024, len(bi)), replace=False))]

        def _mk(items):
            ds = VideoWindowData(
                args.data, by[name], items, stats, args.shaping, args.gamma, offset_mode=wire_offset_mode
            )
            return [{k: v.clone() for k, v in b.items()} for b in make_loader(ds, 16, 4, 0, shuffle=False)]

        sets[name] = (
            _mk(wi),
            _mk(fi) if len(fi) else [],
            [{k: v.clone() for k, v in b.items()} for b in policy_loader(args.data, bce, 32, 4, shuffle=False, items=bi)],
        )

    torch.manual_seed(args.seed)
    model_kwargs = dict(
        arch=args.arch,
        stats=stats,
        shaping=args.shaping,
        offset_mode=args.offset_mode,
        offset_weight=args.offset_weight,
        sg_probe=not args.no_sg_probe,
        pretrained=(not args.no_pretrained) if ck is None else False,
        local_files_only=args.local_files_only,
        segformer_model_id=args.segformer_model_id,
        vjepa2_model_id=args.vjepa2_model_id,
        latent_grid=args.latent_grid,
        token_dim=args.token_dim,
        dit_dim=args.dit_dim,
        dit_layers=args.dit_layers,
        dit_heads=args.dit_heads,
        vjepa2_unfreeze_blocks=args.vjepa2_unfreeze_blocks,
        detach_target_obs=(bool(args.detach_target_obs) if args.detach_target_obs is not None else None),
        grad_ckpt=args.grad_ckpt,
        cons_warmup_steps=args.cons_warmup_steps,
        gamma=args.gamma,
    )
    if ck is not None and ck.get("model_kwargs", {}).get("vjepa2_hf_config") is not None:
        model_kwargs["vjepa2_hf_config"] = ck["model_kwargs"]["vjepa2_hf_config"]

    try:
        model = VideoWM320(**model_kwargs)
        if ck is not None:
            model.load_state_dict(ck["state"], strict=True)
    except Exception:
        if ck is None:
            raise
        # Fallback for torch.hub models needing cached hub structure before state reload
        model_kwargs["pretrained"] = True
        model = VideoWM320(**model_kwargs)
        model.load_state_dict(ck["state"], strict=True)

    model = model.to(dev)
    net = (
        DDP(
            model,
            device_ids=[dev.index],
            broadcast_buffers=False,
            find_unused_parameters=(args.arch == "d2c_vjepa2_ac_vitg"),
        )
        if ddp
        else model
    )

    pgroups = model.param_groups(
        lr=args.lr,
        enc_lr_scale=args.enc_lr_scale,
        pretrained_lr_scale=args.pretrained_lr_scale,
        weight_decay=args.weight_decay,
    )
    opt_cls = torch.optim.AdamW if args.weight_decay > 0 else torch.optim.Adam
    opt = opt_cls(pgroups)

    step, skipped = 0, 0
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        step, skipped = int(ck["step"]), int(ck["skipped"])
        if not ddp:
            torch.random.set_rng_state(ck["rng_cpu"])
            if dev.type == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"][0], dev)
    if ddp:
        torch.manual_seed(args.seed + RS * rank + step)

    info0 = model.checkpoint_meta(stats)

    def checkpoint(full: bool) -> dict:
        c = {**info0, "state": model.state_dict(), "step": step, "skipped": skipped, "args": vars(args)}
        if full:
            c.update(
                opt=opt.state_dict(),
                rng_cpu=torch.random.get_rng_state(),
                rng_cuda=[torch.cuda.get_rng_state(dev)] if dev.type == "cuda" else None,
            )
        return c

    log = Logger(run) if main_rank else (lambda **kw: None)
    summary = model.architecture_summary()
    log(
        event="resume" if ck is not None else "start",
        step=step,
        processes=world,
        windows_per_step=world * args.wm_batch * args.accum,
        imitation_samples_per_step=world * args.bc_batch * args.accum,
        trajectories={k: len(v) for k, v in by.items()},
        windows=int(len(window_items(by["train"]))),
        imitation_samples=int(sum(e["n_images"] for e in by["train"] if e["bc"])),
        validation={k: [len(v[0]) * 16, len(v[1]) * 16, len(v[2]) * 32] for k, v in sets.items()},
        arch_summary=summary,
        param_groups=[
            {"group": g.get("group_name", "unnamed"), "lr": g["lr"], "n_params_m": round(sum(p.numel() for p in g["params"]) / 1e6, 3)}
            for g in pgroups
        ],
        args=vars(args),
    )

    wo = int(round(args.online_frac * args.wm_batch)) if args.online else 0
    if args.online and (args.refresh_every < 1 or args.online_min < 1 or args.online_reload < 1):
        raise SystemExit("--refresh-every, --online-min and --online-reload must be at least 1")
    if args.online and not 0 < wo < args.wm_batch:
        raise SystemExit(f"--online-frac {args.online_frac} of {args.wm_batch} leaves no window to one source")
    rounds_in, onl, mix = None, [], False

    def mixes(entries: List[dict]) -> bool:
        return len(entries) >= args.online_min and len(window_items(entries)) >= wo

    def make_loaders():
        sd = args.seed + RS * rank + step
        ws = VideoWindowData(
            args.data, by["train"], window_items(by["train"]), stats, args.shaping, args.gamma, offset_mode=wire_offset_mode
        )
        wl = [cycle(make_loader(ws, args.wm_batch - (wo if mix else 0), args.workers_wm, sd))]
        if mix:
            os_ds = VideoWindowData(
                args.data, onl, window_items(onl), stats, args.shaping, args.gamma, offset_mode=wire_offset_mode
            )
            wl.append(cycle(make_loader(os_ds, wo, max(1, args.workers_wm // 2), sd + 3)))
        bce = [e for e in by["train"] if e["bc"]] + ([e for e in onl if e["bc"]] if mix else [])
        return wl, cycle(make_loader(PolicyData(args.data, bce, bc_items(bce, D, WD)), args.bc_batch, args.workers_bc, sd + 1))

    def data_record():
        log(
            event="data",
            step=step,
            online_rounds=rounds_in,
            online_trajectories=len(onl),
            online_seated=sum(e["bc"] for e in onl),
            online_partial_trajectories=sum(bool(e.get("partial")) for e in onl),
            online_windows_per_micro_batch=wo if mix else 0,
        )

    if args.online:
        onl, rounds_in = online_entries(args.online, D, WD)
        mix = mixes(onl)
    wtrain, btrain = make_loaders()
    data_record()

    params = [p for p in model.parameters() if p.requires_grad]
    acc = defaultdict(list)
    t0 = time.time()
    event = "done"
    early = {500, 1000, 2500}
    start_step = step
    job_target = min(args.steps, start_step + args.max_steps_this_job) if args.max_steps_this_job > 0 else args.steps

    while step < job_target:
        step += 1
        opt.zero_grad(set_to_none=True)
        finite = True

        if args.online and step % args.refresh_every == 0:
            new, names = online_entries(args.online, D, WD)
            if names != rounds_in or len(new) - len(onl) >= args.online_reload or mixes(new) != mix:
                del wtrain, btrain
                onl, rounds_in = new, names
                mix = mixes(onl)
                wtrain, btrain = make_loaders()
                data_record()

        for i in range(args.accum):
            W = to_device(next(wtrain[0]), dev) if len(wtrain) == 1 else cat_batches(to_device(next(wtrain[0]), dev), to_device(next(wtrain[1]), dev))
            Bc = to_device(next(btrain), dev)
            if args.shift_px > 0:
                W["rgb"], W["depth"] = shift(W["rgb"], W["depth"], args.shift_px)
                Bc["rgb"], Bc["depth"] = shift(Bc["rgb"], Bc["depth"], args.shift_px)
            with net.no_sync() if (ddp and i < args.accum - 1) else nullcontext():
                loss, info = net(W, Bc, step, args.td_warmup, args.bc_weight)
                (loss / args.accum).backward()
            finite &= math.isfinite(info["total"])
            for k, v in info.items():
                acc[k].append(v)

        gn = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        if ddp:
            flag = torch.tensor([float(finite)], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            finite = bool(flag.item() > 0)
        if finite and math.isfinite(gn):
            opt.step()
            model.update_target()
        else:
            skipped += 1
            log(event="skipped_nonfinite_step", step=step, skipped=skipped, grad_norm=gn)
            if skipped >= 50:
                event = "stop_nonfinite"
                break
        acc["grad_norm"].append(gn)

        if step % 250 == 0:
            keys = sorted(acc)
            means = torch.tensor([float(np.mean(acc[k])) for k in keys], dtype=torch.float64, device=dev)
            gpu = torch.tensor(
                [torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0],
                dtype=torch.float64,
                device=dev,
            )
            if ddp:
                dist.all_reduce(means)
                means /= world
                dist.all_reduce(gpu, op=dist.ReduceOp.MAX)
            log(
                step=step,
                it_per_s=250.0 / max(time.time() - t0, 1e-6),
                gpu_GB=float(gpu[0]),
                **{k: float(v) for k, v in zip(keys, means.tolist())},
            )
            acc, t0 = defaultdict(list), time.time()

        if main_rank and step % args.keep_every == 0:
            save_atomic(checkpoint(False), run / f"step{step}.pt")
        if main_rank and step % args.save_every == 0:
            save_atomic(checkpoint(True), run / "last.pt")
        if main_rank and (step in early or step % args.eval_every == 0 or step == args.steps):
            rec = validate(model, sets, args, dev, auc)
            log(event="val", step=step, **rec)
            t0 = time.time()

        # Synchronize Slurm SIGTERM / SIGUSR1 clean stop across DDP ranks
        sig_num = sig_state["received"]
        if ddp:
            sig_t = torch.tensor([sig_num], dtype=torch.int64, device=dev)
            dist.all_reduce(sig_t, op=dist.ReduceOp.MAX)
            sig_num = int(sig_t.item())
        if sig_num != 0:
            event = "signal_checkpoint"
            if main_rank:
                save_atomic(checkpoint(True), run / "last.pt")
            log(event=event, step=step, signal=sig_num)
            break

    if event == "done" and step < args.steps:
        # Cleanly hit `--max-steps-this-job` limit (for Slurm job chaining)
        event = "job_step_limit"
        if main_rank:
            save_atomic(checkpoint(True), run / "last.pt")
            if step % args.eval_every != 0:
                rec = validate(model, sets, args, dev, auc)
                log(event="val", step=step, **rec)

    log(event=event, step=step)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if event == "stop_nonfinite":
        raise SystemExit(1)
    if event == "done" and main_rank:
        save_atomic(checkpoint(True), run / "last.pt")
        save_atomic(checkpoint(False), run / f"step{step}.pt")
        (run / "DONE.json").write_text(json.dumps({"event": event, "step": step, "arch": args.arch}))


if __name__ == "__main__":
    main()
