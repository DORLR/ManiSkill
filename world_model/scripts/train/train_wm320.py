"""Train the 320 px world model (wm/tdmpc/wm320.py) offline on the fast copy: every trajectory of the training split,
of every kind and outcome (expert demonstrations, offset pushes that jam or insert, random motion), plus the
imitation stream of the built-in ACT head (the successful expert trajectories, every picture, independent samples).

One optimiser step = --accum micro-batches of --wm-batch windows (4 decisions each) and --bc-batch imitation samples
(all three required: on one 24 GB A5000 the standard stem fits 16 / 32 / 8, the stem without max-pool 4 / 8 / 32;
group norm, so accumulating gives the same gradient, except that the 2 of 5 target heads are drawn per micro-batch).
The pictures' depth enters the stem in float32, but cuDNN's default TF32 rounds it to ~0.1 mm below 0.2 m and
~0.2 mm beyond (as in the stand-alone policy). last.pt is written every --save-every steps; the random generators
are saved with it.
Adam, lr 3e-4 with the encoder at --enc-lr-scale (TD-MPC2's 0.3), gradient clip 20, target Q rate 0.01, one target
update per step; random shift --shift-px per window / sample (required: stated, not a silent default).
Validation (select split decides nothing yet; held-out logged): losses on fixed windows; the done head's AUC on
observations with the pins in the hole or at the face (FK depth > 0); V just before the pins reach the face of a
contact push against how it ended; the built-in head's action error on held imitation samples. Fixed random numbers, restored afterwards.
Resume and DONE.json as train_policy320.py.
Online rounds (--online <pool>, scripts/online/run_wm320_attempts.py --mode collect): every --refresh-every steps the
complete rounds of the pool (r<R>/DONE written) and the finished grasps of a round still being collected are read
(2026-11-18); once the pool holds --online-min trajectories, --online-frac of every micro-batch's windows come from it (the rest from the fast copy; its
loader has --workers-wm // 2 extra workers per process), and the online attempts that ended seated by FK (1.5 mm,
4 deg of H*: labels, never the simulator's truth) join the ACT head's imitation set: only the pictures of their last
try (after the last back-off) whose next 16 targets lie in decisions where the proposal or the model's correction was
executed (no exploration shift or random candidate; the final hold counts as part of the last decision) -- the world
model's corrections are distilled into the head. Online trajectories are training grasps only (checked).
Several GPUs: start with torchrun (PyTorch's DistributedDataParallel, NCCL): every process draws its own windows and
samples, gradients are averaged over the processes; --accum is per process, so one optimiser step holds
processes x --accum micro-batches (processes x accum = the single-GPU accum keeps the single-GPU recipe). Only
process 0 validates, logs and writes checkpoints; the logged training losses are averaged over all processes, gpu_GB
is the largest of them. A run of any size can be resumed with any number of processes (the checkpoint holds the
model, not the wrapper, and the random state of the training GPU only). Process r's loaders draw from seeds
seed + 1_000_003 r + step; its torch stream is reseeded after the model is built (so a multi-GPU run does not
repeat the single-GPU stream).

    python world_model/scripts/train/train_wm320.py --data $Q/data/wm_v1_fast --out <run> --stem standard --shift-px 4 --shaping 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from wm.tdmpc import demo320 as D
from wm.tdmpc import wmdata320 as WD
from wm.tdmpc.policy320 import PolicyData, collate_cat, policy_loader, shift
from wm.tdmpc.wm320 import WM320, WindowData, auc, face_items, stats_for, window_items
from wm.training.common import Logger, cycle, seed_all


RS = 1_000_003                         # seed stride between processes (no (process, step) pair repeats)


class GracefulStopper:
    """Catch SIGTERM / SIGUSR1 (e.g., Slurm walltime warning / preemption) and request a clean stop after the current step."""

    def __init__(self, signals=(signal.SIGTERM, signal.SIGUSR1)):
        self.requested = False
        self.signum: int | None = None
        self._prev: dict[int, object] = {}
        self._signals = tuple(signals)

    def _handler(self, signum, _frame):
        self.requested = True
        self.signum = int(signum)

    def install(self) -> "GracefulStopper":
        for sig in self._signals:
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except (OSError, ValueError):
                pass
        return self

    def restore(self) -> None:
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)
            except (OSError, ValueError):
                pass
        self._prev.clear()

    @property
    def reason(self) -> str | None:
        if not self.requested:
            return None
        try:
            name = signal.Signals(self.signum).name
        except Exception:
            name = str(self.signum)
        return f"signal_{name}"


def save(obj, path: Path):
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def loader(ds, batch, workers, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g if shuffle else None, drop_last=shuffle,
                                       num_workers=workers, persistent_workers=shuffle and workers > 0, prefetch_factor=2 if workers else None,
                                       pin_memory=torch.cuda.is_available())


def _finished_grasps(r: Path) -> list[dict]:
    """Manifest entries of the grasps a round in progress has finished: its result files (written whole, by rename)
    without an error. A grasp that crashed is run again later and is not read before that."""
    out = []
    for f in sorted(r.glob("results/*/*.json")):
        try:
            s = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        if not s.get("error"):
            out += s.get("entries", [])
    return out


def online_entries(pool, split_mode: str = "split320") -> tuple[list[dict], list[str]]:
    """Trajectories of the online pool (absolute paths), with split and bc: the complete rounds (DONE + manifest) and,
    of a round still being collected, its finished grasps (2026-11-18, the user's decision: the learner uses a round
    while it is collected; those entries carry partial=True). Returns them and the names of the complete rounds."""
    out, rounds = [], []
    for r in sorted(Path(pool).glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not r.name[1:].isdigit():
            continue
        if (r / "DONE").exists() and (r / "manifest.json").exists():
            rounds.append(r.name)
            ents, partial = json.loads((r / "manifest.json").read_text())["entries"], False
        else:
            ents, partial = _finished_grasps(r), True
        for e in ents:
            sp = D._call_split(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"], split_mode)
            if sp != "train":
                raise SystemExit(f"online trajectory {e['path']} is of split {sp} (split_mode={split_mode}): the pool must hold training grasps only")
            if e["steps"] >= WD.EXECUTE and (not partial or (Path(pool) / e["path"]).exists()):
                out.append({**e, "path": str(Path(pool) / e["path"]), "split": sp, "bc": e.get("outcome") == "seated_fk", "partial": partial})
    return out, rounds


IMITATE = (0, 1)                       # online decisions the ACT head imitates: the proposal and the model's corrections


BACKOFF = 4


def bc_items(entries: list[dict]) -> np.ndarray:
    """(entry, picture) imitation samples: every picture of a demonstration; of an online attempt only the pictures
    after its last back-off whose next 16 targets lie in proposal / correction decisions (no exploration shift, random
    candidate or back-off; the 4-step hold at the end belongs to the last decision)."""
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
            if d0 >= first and all(kinds[d] in IMITATE for d in range(d0, min(len(kinds), (4 * j + D.CHUNK - 1) // WD.EXECUTE + 1))):
                rows.append((i, j))
    return np.array(rows, np.int64).reshape(-1, 2)


def cat(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


def to(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def validate(model, sets, args, dev):
    model.eval()
    cpu, gpu = torch.random.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.manual_seed(1234)
    out = {}
    for name, (wl, fl, el, bl) in sets.items():
        num, done_s, done_y, di_s, di_y, v0, end = defaultdict(list), [], [], [], [], [], []
        for W in wl:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            for k, v in info.items():
                if not k.startswith("_"):
                    num[k].append(v)
            m = ((W["obs_valid"] > 0) & (W["fk_depth"] > 0)).cpu().numpy()
            done_s.append(info["_done"][m]); done_y.append(W["done"].cpu().numpy()[m])
            di_s.append(info["_done_imag"][m[:, 1:]]); di_y.append(W["done"][:, 1:].cpu().numpy()[m[:, 1:]])
        for W in fl:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            v0.append(info["_v0"]); end.append(W["end_seated"].cpu().numpy())
        err = []
        for b in bl:
            b = to(b, dev)
            tok = model.enc.tokens(b["rgb"].flatten(0, 1), b["depth"].flatten(0, 1)).view(*b["depth"].shape[:2], 100, 32)
            pred = model.act_chunk(model.enc.fuse(tok, b["q"])).flatten(1)
            err.append(float((pred - b["a"]).abs().mean()) * 1000)
        rec = {k: float(np.mean(v)) for k, v in num.items()}
        if done_y:
            rec["auc_done_depth_pos"] = auc(np.concatenate(done_s), np.concatenate(done_y))
            rec["auc_done_imagined_depth_pos"] = auc(np.concatenate(di_s), np.concatenate(di_y))
        if v0:
            rec["auc_V_before_face_contact"] = auc(np.concatenate(v0), np.concatenate(end) > 0.5)
        if err:
            rec["act_err_mrad"] = float(np.mean(err))
        out[name] = rec
    torch.random.set_rng_state(cpu)
    if gpu is not None:
        torch.cuda.set_rng_state_all(gpu)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", required=True, choices=["standard", "nopool"])
    ap.add_argument("--imagenet", action="store_true", help="the ResNet starts from ImageNet weights (policy320.PictureTokens.init_imagenet)")
    ap.add_argument("--offset-weight", type=float, default=0.0, help="> 0: offset read-out head on z (wm320.WM320)")
    ap.add_argument("--offset-mode", choices=list(WD.OFFSET_MODES), default="gap",
                    help="dim 0 target of _offset head: 'gap' = zero-privilege pin-tip gap (-along - L) [recommended default]; 'along' = legacy")
    ap.add_argument("--split-mode", choices=list(D.SPLIT_MODES), default="split320",
                    help="split320 = Changhao's default split; strict_extrap = also hold out outer extrap grasp bands")
    ap.add_argument("--shift-px", type=int, default=0, help="random pixel shift per window/sample (0 = recommended clean setting)")
    ap.add_argument("--shaping", type=float, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--max-steps-this-job", "--stop-after-steps", dest="max_steps_this_job", type=int, default=None,
                    help="optional step cap for this job (Slurm segmented runs): save last.pt and exit 0 cleanly (no DONE.json) before --steps")
    ap.add_argument("--wm-batch", type=int, required=True)
    ap.add_argument("--bc-batch", type=int, required=True)
    ap.add_argument("--accum", type=int, required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-scale", type=float, default=0.3)
    ap.add_argument("--td-warmup", type=int, default=5000)
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
    ap.add_argument("--online", help="online pool; its complete rounds are mixed in")
    ap.add_argument("--online-frac", type=float, default=0.5, help="share of every micro-batch's windows from the online pool")
    ap.add_argument("--refresh-every", type=int, default=250, help="steps between reads of the online pool")
    ap.add_argument("--online-min", type=int, default=100, help="online trajectories needed before they are mixed in")
    ap.add_argument("--online-reload", type=int, default=150, help="new online trajectories that make a read rebuild the loaders")
    args = ap.parse_args()
    import resource                        # 2026-11-18: the loaders, rebuilt as an online round streams in, pass tensors
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)          # by file descriptor; a soft limit of 1024 ran out
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))          # ("received 0 items of ancdata") at step 15,000
    if args.offset_weight < 0:
        raise SystemExit("--offset-weight must be >= 0")
    if args.max_steps_this_job is not None and args.max_steps_this_job < 1:
        raise SystemExit("--max-steps-this-job must be >= 1")
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(minutes=60), device_id=torch.device("cuda", local))
        rank, world, dev = dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)
        frac = os.environ.get("WM_GPU_MEM_FRACTION")               # this process's own GPU only (seed_all would touch GPU 0)
        if frac:
            torch.cuda.set_per_process_memory_fraction(float(frac), local)
        random.seed(args.seed + RS * rank); np.random.seed(args.seed + RS * rank); torch.manual_seed(args.seed + RS * rank)
    else:
        rank, world, dev = 0, 1, torch.device(args.device)
        seed_all(args.seed)
    main_rank = rank == 0
    run = Path(args.out)
    resume = args.resume and (run / "last.pt").exists()
    refuse = (f"{run} is finished" if (run / "DONE.json").exists() else
              f"{run} already holds a run; use --resume to continue it" if not args.resume and (run / "log.jsonl").exists() else None)
    if ddp:
        dist.barrier()                                              # every process decided before process 0 writes anything
    if refuse:
        raise SystemExit(refuse)
    if main_rank:
        run.mkdir(parents=True, exist_ok=True)
    ents = [e for e in D.load_entries(args.data, split_mode=args.split_mode) if e["split"] != "test" and e["steps"] >= WD.EXECUTE]
    by = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if resume else None
    if ck and ck.get("labels") != WD.LABELS:                       # a changed design is trained from scratch, never resumed
        raise SystemExit(f"{run}/last.pt was trained with labels {ck.get('labels')}, this code has {WD.LABELS}: start a new run")
    if ck and ck.get("imagenet", False) != args.imagenet:
        raise SystemExit(f"{run}/last.pt has imagenet={ck.get('imagenet', False)}, --imagenet is {args.imagenet}: start a new run")
    if ck and ck.get("offset_weight", 0.0) != args.offset_weight:
        raise SystemExit(f"{run}/last.pt has offset_weight={ck.get('offset_weight', 0.0)}, --offset-weight is {args.offset_weight}: start a new run")
    if ck and (args.offset_weight > 0 or ck.get("offset_weight", 0.0) > 0) and ck.get("offset_mode", "along") != args.offset_mode:
        raise SystemExit(f"{run}/last.pt has offset_mode={ck.get('offset_mode', 'along')}, --offset-mode is {args.offset_mode}: start a new run")
    if ck and ck.get("split_mode", "split320") != args.split_mode:
        raise SystemExit(f"{run}/last.pt has split_mode={ck.get('split_mode', 'split320')}, --split-mode is {args.split_mode}: start a new run")
    if ck:
        stats = ck["stats"]
    else:
        box = [stats_for(args.data, by["train"], seed=args.seed) if main_rank else None]
        if ddp:
            dist.broadcast_object_list(box, src=0)
        stats = box[0]
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
        mk = lambda items: [{k: v.clone() for k, v in b.items()} for b in loader(WindowData(args.data, by[name], items, stats, args.shaping, args.gamma, offset_mode=args.offset_mode), 16, 4, 0, False)]  # noqa: E731
        sets[name] = (mk(wi), mk(fi) if len(fi) else [], [], [{k: v.clone() for k, v in b.items()} for b in policy_loader(args.data, bce, 32, 4, shuffle=False, items=bi)])
    torch.manual_seed(args.seed)
    model = WM320(args.stem, stats, args.shaping, imagenet=args.imagenet, offset_weight=args.offset_weight, offset_mode=args.offset_mode, gamma=args.gamma)
    copied = None if ck else model.enc.pic.init_imagenet() if args.imagenet else 0   # identical on every process
    model = model.to(dev)
    net = DDP(model, device_ids=[dev.index], broadcast_buffers=False) if ddp else model     # broadcasts process 0's weights
    enc = list(model.enc.parameters())
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    opt = torch.optim.Adam([{"params": enc, "lr": args.lr * args.enc_lr_scale}, {"params": rest, "lr": args.lr}])
    step, skipped = 0, 0
    if ck:
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); step, skipped = ck["step"], ck["skipped"]
        if not ddp:
            torch.random.set_rng_state(ck["rng_cpu"])
            if dev.type == "cuda" and ck.get("rng_cuda") is not None:       # the training GPU's state (older files: one GPU)
                torch.cuda.set_rng_state(ck["rng_cuda"][0], dev)
    if ddp:          # the model was built identically everywhere (same seed); from here each process draws its own
        torch.manual_seed(args.seed + RS * rank + step)     # shifts, noise and dropout (a resumed run: a fresh stream from the step on)
    info0 = {"format": "wm320-1", "labels": WD.LABELS, "stem": args.stem, "imagenet": args.imagenet, "offset_weight": args.offset_weight,
             "offset_mode": args.offset_mode, "split_mode": args.split_mode, "shaping": args.shaping, "cfg": {"gamma": args.gamma}, "stats": stats}

    def checkpoint(full: bool):
        c = {**info0, "state": model.state_dict(), "step": step, "skipped": skipped, "args": vars(args)}
        if full:
            c.update(opt=opt.state_dict(), rng_cpu=torch.random.get_rng_state(),
                     rng_cuda=[torch.cuda.get_rng_state(dev)] if dev.type == "cuda" else None)
        return c

    log = Logger(run) if main_rank else (lambda **kw: None)
    log(event="resume" if ck else "start", step=step, processes=world, windows_per_step=world * args.wm_batch * args.accum,
        imitation_samples_per_step=world * args.bc_batch * args.accum, trajectories={k: len(v) for k, v in by.items()},
        windows=int(len(window_items(by["train"]))), imitation_samples=int(sum(e["n_images"] for e in by["train"] if e["bc"])),
        validation={k: [len(v[0]) * 16, len(v[1]) * 16, len(v[2]) * 16, len(v[3]) * 32] for k, v in sets.items()},
        params_m=sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, imagenet=args.imagenet, imagenet_tensors_copied=copied,
        args=vars(args))
    wo = int(round(args.online_frac * args.wm_batch)) if args.online else 0
    if args.online and (args.refresh_every < 1 or args.online_min < 1 or args.online_reload < 1):
        raise SystemExit("--refresh-every, --online-min and --online-reload must be at least 1")
    if args.online and not 0 < wo < args.wm_batch:
        raise SystemExit(f"--online-frac {args.online_frac} of {args.wm_batch} windows leaves no window to one of the two sources")
    rounds_in, onl, mix = None, [], False

    def make_loaders():
        """Windows: the fast copy alone, or wm_batch - wo from it and wo from the online pool; imitation: the
        successful demonstrations and the online attempts that ended seated (FK)."""
        sd = args.seed + RS * rank + step
        ws = WindowData(args.data, by["train"], window_items(by["train"]), stats, args.shaping, args.gamma, offset_mode=args.offset_mode)
        wl = [cycle(loader(ws, args.wm_batch - (wo if mix else 0), args.workers_wm, sd))]
        if mix:                                                # (fewer windows than wo: a drop_last loader would never yield)
            wl.append(cycle(loader(WindowData(args.data, onl, window_items(onl), stats, args.shaping, args.gamma, offset_mode=args.offset_mode), wo,
                                   args.workers_wm // 2, sd + 3)))
        bce = [e for e in by["train"] if e["bc"]] + ([e for e in onl if e["bc"]] if mix else [])     # online data only once mixed in
        return wl, cycle(loader(PolicyData(args.data, bce, bc_items(bce)), args.bc_batch, args.workers_bc, sd + 1))

    def mixes(entries):
        return len(entries) >= args.online_min and len(window_items(entries)) >= wo

    def data_record():
        log(event="data", step=step, online_rounds=rounds_in, online_trajectories=len(onl), online_seated=sum(e["bc"] for e in onl),
            online_partial_trajectories=sum(bool(e.get("partial")) for e in onl), online_windows_per_micro_batch=wo if mix else 0)

    if args.online:
        onl, rounds_in = online_entries(args.online, split_mode=args.split_mode)
        mix = mixes(onl)
    wtrain, btrain = make_loaders()
    data_record()
    params = [p for p in model.parameters() if p.requires_grad]
    acc, t0, event, steps_this_job = defaultdict(list), time.time(), "done", 0
    early = {500, 1000, 2500}
    stopper = GracefulStopper().install()
    try:
        while step < args.steps:
            sig_stop = stopper.requested
            if ddp:
                sflag = torch.tensor([1.0 if sig_stop else 0.0], device=dev)
                dist.all_reduce(sflag, op=dist.ReduceOp.MAX)
                sig_stop = bool(sflag.item() > 0)
            if sig_stop:
                if main_rank and (step > 0 or ck is not None):
                    save(checkpoint(True), run / "last.pt")
                event = stopper.reason or "signal_stop"
                break
            step += 1
            opt.zero_grad(set_to_none=True)
            finite = True
            if args.online and step % args.refresh_every == 0:
                new, names = online_entries(args.online, split_mode=args.split_mode)
                if names != rounds_in or len(new) - len(onl) >= args.online_reload or mixes(new) != mix:
                    del wtrain, btrain                                     # the old loaders' workers end with them
                    onl, rounds_in = new, names
                    mix = mixes(onl)
                    wtrain, btrain = make_loaders()
                    data_record()
            for i in range(args.accum):
                W = to(next(wtrain[0]), dev) if len(wtrain) == 1 else cat(to(next(wtrain[0]), dev), to(next(wtrain[1]), dev))
                Bc = to(next(btrain), dev)
                W["rgb"], W["depth"] = shift(W["rgb"], W["depth"], args.shift_px)
                Bc["rgb"], Bc["depth"] = shift(Bc["rgb"], Bc["depth"], args.shift_px)
                with net.no_sync() if ddp and i < args.accum - 1 else nullcontext():   # gradients averaged once, at the last micro-batch
                    loss, info = net(W, Bc, step, args.td_warmup, args.bc_weight)
                    (loss / args.accum).backward()
                finite &= math.isfinite(info["total"])
                for k, v in info.items():
                    acc[k].append(v)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 20.0))
            if ddp:                                                      # every process takes the same decision
                flag = torch.tensor([float(finite)], device=dev)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                finite = bool(flag.item() > 0)
            if finite and math.isfinite(gn):
                opt.step()
                model.update_target()
            else:
                skipped += 1
                log(event="skipped_nonfinite_step", step=step, skipped=skipped)
                if skipped >= 50:
                    event = "stop_nonfinite"
                    break
            steps_this_job += 1
            acc["grad_norm"].append(gn)
            if step % 250 == 0:
                keys = sorted(acc)
                means = torch.tensor([float(np.mean(acc[k])) for k in keys], dtype=torch.float64, device=dev)
                gpu = torch.tensor([torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0], dtype=torch.float64, device=dev)
                if ddp:                                                  # every process: the same keys, the same steps
                    dist.all_reduce(means)
                    means /= world
                    dist.all_reduce(gpu, op=dist.ReduceOp.MAX)
                log(step=step, it_per_s=250 / (time.time() - t0), gpu_GB=float(gpu[0]), **{k: float(v) for k, v in zip(keys, means.tolist())})
                acc, t0 = defaultdict(list), time.time()
            sig_stop = stopper.requested
            if ddp:
                sflag = torch.tensor([1.0 if sig_stop else 0.0], device=dev)
                dist.all_reduce(sflag, op=dist.ReduceOp.MAX)
                sig_stop = bool(sflag.item() > 0)
            if sig_stop:
                if main_rank:
                    save(checkpoint(True), run / "last.pt")
                event = stopper.reason or "signal_stop"
                break
            if main_rank and step % args.keep_every == 0:
                save(checkpoint(False), run / f"step{step}.pt")
            if main_rank and step % args.save_every == 0:
                save(checkpoint(True), run / "last.pt")
            if main_rank and (step in early or step % args.eval_every == 0 or step == args.steps):
                rec = validate(model, sets, args, dev)
                log(event="val", step=step, **rec)
                t0 = time.time()
            if args.max_steps_this_job is not None and steps_this_job >= args.max_steps_this_job and step < args.steps:
                if main_rank and step % args.save_every != 0:
                    save(checkpoint(True), run / "last.pt")
                event = "segment_done"
                break
    finally:
        stopper.restore()
    log(event=event, step=step, steps_this_job=steps_this_job)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if event == "segment_done" or event.startswith("signal_"):
        return
    if event != "done":
        raise SystemExit(1)
    if main_rank:
        save(checkpoint(False), run / f"step{step}.pt")
        (run / "DONE.json").write_text(json.dumps({"event": event, "step": step}))


if __name__ == "__main__":
    main()
