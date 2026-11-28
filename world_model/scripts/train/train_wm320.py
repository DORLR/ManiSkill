"""Train the 320 px world model (wm/tdmpc/wm320.py) offline on the fast copy: every trajectory of the training split,
of every kind and outcome (expert demonstrations, offset pushes that jam or insert, random motion), plus the
imitation stream of the built-in ACT head (the successful expert trajectories, every picture, independent samples).

One optimiser step = --accum micro-batches of --wm-batch windows (4 decisions each) and --bc-batch imitation samples
(all three required: on one 24 GB A5000 the standard stem fits 16 / 32 / 8, the stem without max-pool 4 / 8 / 32;
group norm, so accumulating gives the same gradient, except that the 2 of 5 target heads are drawn per micro-batch;
with --encoder segformer_b1 too: its decoder's batch norm uses fixed statistics, wm/tdmpc/encoders320.py, BATCH NORM).
Picture encoder (2026-11-21): --encoder resnet18 (the legacy ResNet-18; old checkpoints have no encoder key and mean
it), resnet34 or segformer_b1 (wm/tdmpc/encoders320.py; these need --imagenet and start from their pretrained
weights). Without --encoder: resnet18 for ManiSkill data; AutoMate data (manifest entries with simulator "automate")
refuse a missing --encoder and refuse resnet18 (dropped for AutoMate; its depth input saturates at 0.3 m).
--token-dim: 32 for resnet18 when omitted; required for resnet34 / segformer_b1 (agreed: 128). Depth normalisation of
the new encoders: the json of scripts/automate/depth_stats.py (--depth-stats, default <data>/depth_stats.json; its
training-split hash, demo320.train_files_sha256, must match this run's training split), or (ManiSkill data only) all
of --depth-max-m, --depth-mean-m, --depth-std-m, which must agree with a statistics file if one exists (its hash is
then checked too); AutoMate data refuses the flags and needs the file. --depth-invalid defaults to (0 - mean) / std
and must lie below the smallest valid z (policy320.check_depth_settings); the placeholders are refused. The new
encoders need $HF_HOME and $TORCH_HOME set, outside the account's home directory (a shared account); the Hub is set
offline (HF_HUB_OFFLINE=1); every weight file must already be cached (segformer_b1: encoders320.SEGFORMER_B1_FILES at
the pinned revision, download once with `python -m wm.tdmpc.encoders320 --download`; resnet34: torchvision's
IMAGENET1K_V1 file), checked before anything is written. A checkpoint with other encoder settings is not resumed.
The start record and every checkpoint hold `provenance`: torch / transformers / huggingface_hub / torchvision /
scikit-learn / joblib / threadpoolctl versions and the imported scikit-learn's file, the md5 of
the model and data code, the git commit if any, the depth statistics file with its sha256, the file of the imported
wm package (always this tree's ROOT/src: the script puts it first on sys.path and refuses wm modules from anywhere
else) and the fast copy's split identity (manifest "split": demo320.split_identity, sha256 of the collection's
split_automate_v1.json; None for ManiSkill data). AutoMate data needs that identity in the manifest, and a depth
statistics file must carry the same one. scikit-learn (wm320.auc, the validation AUCs) is checked before the model is
built, so that a missing package does not end a run at its first validation (step 500).
Splits (demo320.split_of_entry): ManiSkill data by split320; AutoMate data by each grasp's stored condition, refused
while it is "unassigned"; AutoMate data also needs at least MIN_SELECT_GRASPS selection grasps and a non-empty
held-out set.
Completeness (AutoMate): the fast copy's manifest "completeness" (scripts/data/cache_wm_v1_fast.py, from
wm.data.index.build_automate) must exist and say partial = false (when it was indexed the collection had ended, every
planned grasp of an assembly that is not skipped had its grasp.json, and the missing grasps of skipped assemblies were
accepted with --accept-skipped); a missing record or partial = true is refused unless --allow-partial. The record and the switch are stored in provenance ("completeness", "allow_partial"),
with the manifest's "code" (the md5 of the converter that wrote the fast copy, and of earlier converters whose files a
resumed copy may keep; scripts/data/cache_wm_v1_fast.py) as "fast_copy_code" (recorded, not checked).
Offset mode (--offset-mode, wmdata320 OFFSET_MODES; default "gap"): the offset read-out head's first component is
trained on the pin-tip gap g = -along - L ("gap") or on along ("along", legacy); the checkpoint holds "offset_mode"
(a checkpoint without the key means "along"); a checkpoint with an offset head and another mode is not resumed.
AutoMate data (the user's decision 6 of 2026-11-23) refuses any mode but "gap"; the online agent requires "gap" on
AutoMate (agent320.WMAgent, online.common320.check_offset_mode). --split-mode (demo320.SPLIT_MODES) applies to
ManiSkill data; it is stored in the checkpoint as "split_mode".
AutoMate: the in-hole rule of the agent reads the offset head's first output (gap_mm); a checkpoint trained with
--offset-weight 0 has no offset head.
The pictures' depth enters the stem in float32, but cuDNN's default TF32 rounds it to ~0.1 mm below 0.2 m and
~0.2 mm beyond (as in the stand-alone policy). last.pt is written every --save-every steps; the random generators
are saved with it.
Adam, lr 3e-4 with the encoder at --enc-lr-scale (TD-MPC2's 0.3), gradient clip 20, target Q rate 0.01, one target
update per step; random shift --shift-px per window / sample (default 0 since the d1-gap-fix merge; the round-8
recipe and every AutoMate run pass --shift-px 4 explicitly).
Validation (select split decides nothing yet; held-out logged; AutoMate also logs heldout_shape and heldout_grasp,
the two stored conditions inside "heldout", separately): losses on fixed windows; the done head's AUC on
observations with the pins in the hole or at the face (FK depth > 0); V just before the pins reach the face of a
contact push against how it ended (wm320.face_items; "face_items_kept" = [contact pushes in this AUC, contact pushes
in the split], AutoMate runs only); the built-in head's action error on held imitation samples. Fixed random
numbers, restored afterwards.
AutoMate: the face is the tight-fit entrance (s = L); contact pushes that stop more than 0.5 mm above it (on the
socket top or the lead-in, assemblies with g_top > 0.5 mm) are not in this AUC.
Resume and DONE.json as train_policy320.py; an AutoMate checkpoint whose provenance split differs from the fast
copy's is not resumed.
GPU memory: with QX_GPU_GB in the environment (the scope's declared GPU memory, exported by qx_run) the process caps
itself at QX_GPU_GB - 1 GB (the CUDA context lies outside the allocator cap; cap_gpu_memory); AutoMate data on CUDA
refuses to start without it; ManiSkill runs without it are unchanged.
Online rounds (--online <pool>, scripts/online/run_wm320_attempts.py --mode collect): every --refresh-every steps the
complete rounds of the pool (r<R>/DONE written) and the finished grasps of a round still being collected are read
(2026-11-18); once the pool holds --online-min trajectories, --online-frac of every micro-batch's windows come from it (the rest from the fast copy; its
loader has --workers-wm // 2 extra workers per process), and the online attempts that ended seated by FK (1.5 mm,
4 deg of H*: labels, never the simulator's truth) join the ACT head's imitation set: only the pictures of their last
try (after the last back-off) whose next 16 targets lie in decisions where the proposal or the model's correction was
executed (no exploration shift or random candidate; the final hold counts as part of the last decision) -- the world
model's corrections are distilled into the head. Online trajectories are training grasps only (checked); an AutoMate
pool entry must also carry split_sha256 equal to the fast copy's split identity (sha256), and every pool entry must
come from the fast copy's simulator (no ManiSkill pool for an AutoMate run, and the reverse), else the run is refused.
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
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]                       # world_model/
sys.path.insert(0, str(ROOT / "src"))                            # this tree's wm package, before any other
import wm  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import (ENCODER_DEFAULTS, ENCODERS, PolicyData, check_depth_settings, collate_cat,  # noqa: E402
                                 default_depth_invalid, policy_loader, shift)
from wm.tdmpc.wm320 import (WM320, WindowData, auc, effective_decisions_for, encoder_settings, face_items,  # noqa: E402
                             stats_for, window_items)
from wm.training.common import Logger, cycle, seed_all  # noqa: E402


RS = 1_000_003                         # seed stride between processes (no (process, step) pair repeats)
MIN_SELECT_GRASPS = D.MIN_SELECT_GRASPS   # AutoMate: checkpoints are not chosen on fewer selection grasps (one source)
AUTOMATE_HELDOUT_CONDITIONS = ("heldout_shape", "heldout_grasp")   # logged separately inside "heldout" (AutoMate)
PROVENANCE_FILES = ("src/wm/tdmpc/policy320.py", "src/wm/tdmpc/encoders320.py", "src/wm/tdmpc/wm320.py",
                    "src/wm/tdmpc/wmdata320.py", "src/wm/tdmpc/demo320.py", "scripts/train/train_wm320.py")


def foreign_wm_modules() -> list[str]:
    """Loaded wm.* modules whose file lies outside this code tree (as scripts/automate/collect_expert.py)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT.resolve()):
            bad.append(f"{name} ({f})")
    return bad


def entries_or_exit(load, *a):
    """demo320.load_entries / online_entries with an undecided AutoMate split turned into SystemExit."""
    try:
        return load(*a)
    except D.SplitUndecided as exc:
        raise SystemExit(f"{exc} (demo320.split_of_entry)")


def provenance(depth_stats_file: Path | None, split: dict | None = None) -> dict:
    """What a run was built from: library versions, md5 of the model / data code, git commit, depth statistics, the
    imported wm package's file and the fast copy's split identity (demo320.split_identity)."""
    import hashlib
    import subprocess
    vers = {"torch": torch.__version__}
    for name in ("transformers", "huggingface_hub", "torchvision", "sklearn", "joblib", "threadpoolctl"):
        try:
            vers[name] = getattr(__import__(name), "__version__", None)
        except ImportError:
            vers[name] = None
    try:                                     # which scikit-learn copy the validation AUCs use (environment or extra site)
        import sklearn
        f = getattr(sklearn, "__file__", None)
        sklearn_file = str(Path(f).resolve()) if f else None
    except ImportError:
        sklearn_file = None
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    md5 = {f: hashlib.md5((ROOT / f).read_bytes()).hexdigest() for f in PROVENANCE_FILES}
    ds = None
    if depth_stats_file is not None:
        ds = {"path": str(depth_stats_file.resolve()), "sha256": hashlib.sha256(depth_stats_file.read_bytes()).hexdigest()}
    return {"versions": vers, "git_commit": commit, "md5": md5, "depth_stats": ds,
            "wm_file": str(Path(wm.__file__).resolve()), "split": split, "sklearn_file": sklearn_file}


def automate_provenance(manifest: dict, allow_partial: bool) -> dict:
    """The AutoMate fields added to provenance (module docstring, completeness): the manifest's completeness record,
    the --allow-partial switch and the converter's code record (manifest "code", None for an older fast copy)."""
    return {"completeness": manifest.get("completeness"), "allow_partial": bool(allow_partial),
            "fast_copy_code": manifest.get("code")}


GPU_CONTEXT_GB = 1.0                   # the CUDA context, outside the allocator cap


def cap_gpu_memory(dev: torch.device, automate: bool, ddp: bool) -> float | None:
    """The GPU memory the launcher declared for this job's scope (caps.sh / qx_run exports QX_GPU_GB) becomes the
    process's own cap: torch.cuda.set_per_process_memory_fraction((QX_GPU_GB - GPU_CONTEXT_GB) * 1e9 / total), so a
    job never allocates beyond its declaration (the watcher only reacts after GPU 1 is over budget). AutoMate data on
    CUDA refuses to run without QX_GPU_GB unless WM_DEDICATED_GPUS=1 (a machine whose GPUs nobody else uses,
    e.g. AutoDL: no cap, also under torchrun); ManiSkill runs without it are unchanged (no cap). QX_GPU_GB is one GPU's
    budget: refused with several processes (torchrun). -> the cap in GB, or None."""
    v = os.environ.get("QX_GPU_GB")
    if dev.type != "cuda":
        return None
    if not v:
        if automate and os.environ.get("WM_DEDICATED_GPUS") != "1":
            raise SystemExit("AutoMate data on CUDA: QX_GPU_GB (the declared GPU memory of the job, exported by qx_run) "
                             "is not set; start it through the capped launcher (scripts/automate/train_automate.sh), "
                             "or set WM_DEDICATED_GPUS=1 on a machine whose GPUs nobody else uses (AutoDL)")
        return None             # no cap: the GPUs are not shared (WM_DEDICATED_GPUS=1) or the data is ManiSkill
    if ddp:
        raise SystemExit("QX_GPU_GB is the budget of one GPU: not for a torchrun run of several processes")
    try:
        gb = float(v)
    except ValueError:
        raise SystemExit(f"QX_GPU_GB={v!r} is not a number")
    cap = gb - GPU_CONTEXT_GB
    if not cap > 0:
        raise SystemExit(f"QX_GPU_GB={v}: nothing left after {GPU_CONTEXT_GB} GB for the CUDA context")
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1e9 / total), dev)
    return cap


def check_encoder_env(encoder: str) -> None:
    """The new encoders read pretrained weights from local caches only: $HF_HOME / $TORCH_HOME set, outside the
    account's home directory, the Hub offline, and every weight file present. SystemExit otherwise."""
    import pwd
    home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    for var in ("HF_HOME", "TORCH_HOME"):
        v = os.environ.get(var)
        if not v:
            raise SystemExit(f"--encoder {encoder}: set ${var} (the weights are read from a local cache, never the account's home)")
        if Path(v).resolve().is_relative_to(home) and os.stat(Path(v).resolve()).st_dev == os.stat(home).st_dev:
            # (a cache on another file system mounted under home, e.g. AutoDL's data disk /root/autodl-tmp, is allowed)
            raise SystemExit(f"--encoder {encoder}: ${var}={v} lies in the account's home {home}: use our own storage")
    os.environ["HF_HUB_OFFLINE"] = "1"
    if encoder == "segformer_b1":
        from wm.tdmpc import encoders320 as E
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise SystemExit(f"--encoder segformer_b1 needs huggingface_hub ({exc})")
        missing = [f for f in E.SEGFORMER_B1_FILES
                   if not isinstance(try_to_load_from_cache(E.SEGFORMER_B1, f, revision=E.SEGFORMER_B1_REVISION), str)]
        if missing:
            raise SystemExit(f"{E.SEGFORMER_B1} at {E.SEGFORMER_B1_REVISION}: {missing} not in the local cache "
                             f"($HF_HOME={os.environ['HF_HOME']}): python -m wm.tdmpc.encoders320 --download")
    if encoder == "resnet34":
        from torchvision.models import ResNet34_Weights
        f = Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet34_Weights.IMAGENET1K_V1.url).name
        if not f.exists():
            raise SystemExit(f"--encoder resnet34: {f} is missing: python -c 'import torchvision; "
                             "torchvision.models.resnet34(weights=\"IMAGENET1K_V1\")' with $TORCH_HOME set")


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


def online_entries(pool, split_sha: str | None = None, split_mode: str = "split320") -> tuple[list[dict], list[str]]:
    """Trajectories of the online pool (absolute paths), with split and bc: the complete rounds (DONE + manifest) and,
    of a round still being collected, its finished grasps (2026-11-18, the user's decision: the learner uses a round
    while it is collected; those entries carry partial=True). Returns them and the names of the complete rounds.
    An AutoMate entry must carry split_sha256 equal to split_sha, the sha256 of the fast copy's split file (manifest
    "split"): its stored condition means "train" only under the split it was collected with, so a pool of another
    collection's split is refused (port plan item 10). split_sha is None exactly for a ManiSkill fast copy (only an
    AutoMate copy's manifest has "split"), so an entry whose simulator differs from the fast copy's is refused: a
    ManiSkill pool given to an AutoMate run, or the reverse (r7-data-01)."""
    pool = Path(pool).resolve()
    out, rounds = [], []
    for r in sorted(pool.glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not r.name[1:].isdigit():
            continue
        if (r / "DONE").exists() and (r / "manifest.json").exists():
            rounds.append(r.name)
            ents, partial = json.loads((r / "manifest.json").read_text())["entries"], False
        else:
            ents, partial = _finished_grasps(r), True
        for e in ents:
            if (split_sha is not None) != D.is_automate(e):          # the pool's simulator must be the fast copy's
                raise SystemExit(f"online trajectory {e['path']} is {'AutoMate' if D.is_automate(e) else 'ManiSkill'}, the "
                                 f"fast copy is {'AutoMate' if split_sha is not None else 'ManiSkill'}: the pool must come "
                                 "from the fast copy's simulator")
            sp = entries_or_exit(D.split_of_entry, e, False, split_mode)
            if D.is_automate(e) and (split_sha is None or e.get("split_sha256") != split_sha):
                raise SystemExit(f"online trajectory {e['path']} carries the split {e.get('split_sha256')}, the fast copy "
                                 f"{split_sha}: the pool must come from the fast copy's collection and split")
            if sp != "train":
                raise SystemExit(f"online trajectory {e['path']} is of split {sp} (split_mode={split_mode}): the pool must hold training grasps only")
            if e["steps"] >= WD.EXECUTE and (not partial or (pool / e["path"]).exists()):
                out.append({**e, "path": str(pool / e["path"]), "split": sp, "bc": e.get("outcome") == "seated_fk", "partial": partial})
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
            md = ((W.get("done_valid", W["obs_valid"]) > 0) & (W["fk_depth"] > 0)).cpu().numpy()
            mi = ((W["obs_valid"][:, 1:] > 0) & (W["fk_depth"][:, 1:] > 0)).cpu().numpy()
            done_s.append(info["_done"][md]); done_y.append(W["done"].cpu().numpy()[md])
            di_s.append(info["_done_imag"][mi]); di_y.append(W["done"][:, 1:].cpu().numpy()[mi])
        for W in fl:
            W = to(W, dev)
            _, info = model.loss(W, None, args.td_warmup, args.td_warmup, 0.0, detail=True)
            v0.append(info["_v0"]); end.append(W["end_seated"].cpu().numpy())
        err = []
        for b in bl:
            b = to(b, dev)
            tok = model.enc.tokens(b["rgb"].flatten(0, 1), b["depth"].flatten(0, 1)).view(*b["depth"].shape[:2], model.enc.pic.n_tokens,
                                                                                          model.enc.pic.token_dim)
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
    ap.add_argument("--imagenet", action="store_true", help="the ResNet starts from ImageNet weights (policy320.PictureTokens.init_pretrained)")
    ap.add_argument("--encoder", default=None, choices=list(ENCODERS),
                    help="picture encoder (policy320.PictureTokens); omitted: resnet18 for ManiSkill data, refused for AutoMate data")
    ap.add_argument("--token-dim", type=int, default=None, help="numbers per picture token (omitted: 32 for resnet18; required otherwise)")
    ap.add_argument("--depth-stats", help="resnet34 / segformer_b1: json of scripts/automate/depth_stats.py (default <data>/depth_stats.json)")
    ap.add_argument("--depth-max-m", type=float, help="resnet34 / segformer_b1: depth clipped to [0, this] m (encoders320.depth_z)")
    ap.add_argument("--depth-mean-m", type=float, help="resnet34 / segformer_b1: the data's fixed depth mean [m]")
    ap.add_argument("--depth-std-m", type=float, help="resnet34 / segformer_b1: the data's fixed depth standard deviation [m]")
    ap.add_argument("--depth-invalid", type=float, default=None,
                    help="resnet34 / segformer_b1: value of invalid pixels (depth 0) after normalisation (default (0 - mean) / std)")
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
    ap.add_argument("--allow-partial", action="store_true",
                    help="AutoMate: train on a fast copy of a running or unfinished collection (recorded in provenance)")
    ap.add_argument("--online", help="online pool; its complete rounds are mixed in")
    ap.add_argument("--online-frac", type=float, default=0.5, help="share of every micro-batch's windows from the online pool")
    ap.add_argument("--refresh-every", type=int, default=250, help="steps between reads of the online pool")
    ap.add_argument("--online-min", type=int, default=100, help="online trajectories needed before they are mixed in")
    ap.add_argument("--online-reload", type=int, default=150, help="new online trajectories that make a read rebuild the loaders")
    args = ap.parse_args()
    foreign = foreign_wm_modules()
    if foreign:
        raise SystemExit(f"wm modules from outside {ROOT}: {foreign[:3]} (unset PYTHONPATH or run this tree's copy)")
    import resource                        # 2026-11-18: the loaders, rebuilt as an online round streams in, pass tensors
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)          # by file descriptor; a soft limit of 1024 ran out
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))          # ("received 0 items of ancdata") at step 15,000
    if args.offset_weight < 0:
        raise SystemExit("--offset-weight must be >= 0")
    if args.max_steps_this_job is not None and args.max_steps_this_job < 1:
        raise SystemExit("--max-steps-this-job must be >= 1")
    encoder_given = args.encoder is not None
    encoder = args.encoder or ENCODER_DEFAULTS["encoder"]
    ents = [e for e in entries_or_exit(D.load_entries, args.data, False, args.split_mode) if e["split"] != "test" and e["steps"] >= WD.EXECUTE]
    by = {s: [e for e in ents if e["split"] == s] for s in ("train", "select", "heldout")}
    automate = any(D.is_automate(e) for e in ents)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    split_id = manifest.get("split")
    if automate:                                                     # AutoMate: no silent legacy encoder
        if not all(D.is_automate(e) for e in ents):
            raise SystemExit("the fast copy mixes AutoMate and ManiSkill trajectories")
        for c in AUTOMATE_HELDOUT_CONDITIONS:                         # logged separately, same windows as in "heldout"
            by[c] = [e for e in by["heldout"] if D.automate_condition(e) == c]
        if not encoder_given:
            raise SystemExit("AutoMate data: state --encoder (resnet34 or segformer_b1) and --token-dim explicitly")
        if encoder == "resnet18":
            raise SystemExit("AutoMate data: --encoder resnet18 was dropped for AutoMate (its depth input saturates at 0.3 m)")
        if args.offset_mode != "gap":                                # the user's decision 6 of 2026-11-23
            raise SystemExit("AutoMate data: --offset-mode gap only (the agent reads the pin-tip gap without the pin length)")
    depth_flags = {"depth_max_m": args.depth_max_m, "depth_mean_m": args.depth_mean_m, "depth_std_m": args.depth_std_m}
    depth_stats, depth_stats_file = None, None
    if encoder == "resnet18":
        if any(v is not None for v in depth_flags.values()) or args.depth_invalid is not None or args.depth_stats:
            raise SystemExit("--depth-* settings are for --encoder resnet34 / segformer_b1; the legacy ResNet-18 has its own depth input")
        depth_flags = {k: ENCODER_DEFAULTS[k] for k in (*depth_flags, "depth_invalid")}
        token_dim = ENCODER_DEFAULTS["token_dim"] if args.token_dim is None else args.token_dim
    else:
        if not args.imagenet:
            raise SystemExit(f"--encoder {encoder} starts from pretrained weights with ImageNet's RGB normalisation: add --imagenet")
        if args.stem != "standard":
            raise SystemExit(f"--encoder {encoder} is built with --stem standard only")
        if args.token_dim is None:
            raise SystemExit(f"--encoder {encoder} needs --token-dim (the agreed value for AutoMate is 128)")
        token_dim = args.token_dim
        given = [v is not None for v in depth_flags.values()]
        stats_file = Path(args.depth_stats) if args.depth_stats else Path(args.data) / "depth_stats.json"
        if automate and any(given):
            raise SystemExit("AutoMate data: the depth normalisation comes from the statistics file only (its training-split "
                             "hash is checked): drop --depth-max-m, --depth-mean-m and --depth-std-m, run scripts/automate/depth_stats.py")
        if any(given) and not all(given):
            raise SystemExit("--depth-max-m, --depth-mean-m and --depth-std-m go together (or use --depth-stats)")
        if not any(given):
            if not stats_file.exists():
                raise SystemExit(f"--encoder {encoder} needs the depth statistics: {stats_file} (scripts/automate/depth_stats.py) "
                                 "or --depth-max-m, --depth-mean-m and --depth-std-m (the data's fixed values)")
            depth_stats, depth_stats_file = json.loads(stats_file.read_text()), stats_file
            depth_flags = {k: float(depth_stats[k]) for k in depth_flags}
            if args.depth_invalid is not None and args.depth_invalid != depth_stats["depth_invalid"]:
                raise SystemExit(f"--depth-invalid {args.depth_invalid} differs from {stats_file}: {depth_stats['depth_invalid']}")
            depth_flags["depth_invalid"] = float(depth_stats["depth_invalid"])
        else:
            if not (args.depth_max_m > 0 and args.depth_std_m > 0):
                raise SystemExit("--depth-max-m and --depth-std-m must be > 0")
            if stats_file.exists():                                   # flags next to a statistics file must agree with it
                st = json.loads(stats_file.read_text())
                diff = [k for k in depth_flags if abs(float(st[k]) - depth_flags[k]) > 1e-9]
                if diff:
                    raise SystemExit(f"--{diff[0].replace('_', '-')} differs from {stats_file}: drop the flags or recompute the file")
                depth_stats, depth_stats_file = st, stats_file           # its training-split hash is checked below
            depth_flags["depth_invalid"] = (default_depth_invalid(args.depth_mean_m, args.depth_std_m)
                                            if args.depth_invalid is None else args.depth_invalid)
        why = check_depth_settings(*(depth_flags[k] for k in ("depth_max_m", "depth_mean_m", "depth_std_m", "depth_invalid")))
        if why:
            raise SystemExit(f"--encoder {encoder}: {why}")
        check_encoder_env(encoder)
    if token_dim < 1:
        raise SystemExit("--token-dim must be >= 1")
    enc_set = {"encoder": encoder, "token_dim": token_dim, **depth_flags}
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
    gpu_cap_gb = cap_gpu_memory(dev, automate, ddp)                  # before the first allocation on the GPU
    main_rank = rank == 0
    run = Path(args.out)
    resume = args.resume and (run / "last.pt").exists()
    refuse = (f"{run} is finished" if (run / "DONE.json").exists() else
              f"{run} already holds a run; use --resume to continue it" if not args.resume and (run / "log.jsonl").exists() else None)
    if ddp:
        dist.barrier()                                              # every process decided before process 0 writes anything
    if refuse:
        raise SystemExit(refuse)
    if depth_stats is not None and depth_stats.get("train_files_sha256") != D.train_files_sha256(by["train"], args.data):
        raise SystemExit("the depth statistics were computed on another training split than this run's: run depth_stats.py again")
    if depth_stats is not None and depth_stats.get("split") != split_id:
        raise SystemExit(f"the depth statistics carry the split {depth_stats.get('split')}, the fast copy {split_id}: "
                         "run depth_stats.py again")
    if automate and not (split_id or {}).get("sha256"):
        raise SystemExit("AutoMate data: the fast copy's manifest has no split identity (written by "
                         "scripts/data/cache_wm_v1_fast.py from the collection's plan.json): convert it again")
    if automate:                                                     # checkpoint selection and held-out validation exist
        n_sel = len({e["grasp"] for e in by["select"]})
        if n_sel < MIN_SELECT_GRASPS or not by["heldout"]:
            raise SystemExit(f"AutoMate data: {n_sel} selection grasps (at least {MIN_SELECT_GRASPS}) and "
                             f"{len(by['heldout'])} held-out trajectories (at least 1): the split rule must provide both")
    if automate:                                                     # the collection was complete when indexed
        comp = manifest.get("completeness")
        why = ("its manifest has no completeness record (convert it again after scripts/data/build_index.py --automate)"
               if not isinstance(comp, dict) else
               f"it was indexed from a running or unfinished collection (completeness partial: {comp.get('problems')})"
               if comp.get("partial") is not False else None)
        if why and not args.allow_partial:
            raise SystemExit(f"AutoMate fast copy {args.data}: {why}; --allow-partial trains on it anyway (recorded)")
    prov = provenance(depth_stats_file, split_id)
    if automate:
        prov.update(automate_provenance(manifest, args.allow_partial))
    if main_rank:
        run.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if resume else None
    if ck and automate and (ck.get("provenance") or {}).get("split") != split_id:
        raise SystemExit(f"{run}/last.pt was trained on the split {(ck.get('provenance') or {}).get('split')}, the fast copy "
                         f"carries {split_id}: start a new run")
    if ck and ck.get("labels") != WD.LABELS:                       # a changed design is trained from scratch, never resumed
        raise SystemExit(f"{run}/last.pt was trained with labels {ck.get('labels')}, this code has {WD.LABELS}: start a new run")
    if ck and ck.get("imagenet", False) != args.imagenet:
        raise SystemExit(f"{run}/last.pt has imagenet={ck.get('imagenet', False)}, --imagenet is {args.imagenet}: start a new run")
    if ck and ck.get("offset_weight", 0.0) != args.offset_weight:
        raise SystemExit(f"{run}/last.pt has offset_weight={ck.get('offset_weight', 0.0)}, --offset-weight is {args.offset_weight}: start a new run")
    for k, v in (encoder_settings(ck) if ck else {}).items():       # an old checkpoint: the legacy ResNet-18's settings
        if v != enc_set[k]:
            raise SystemExit(f"{run}/last.pt has {k}={v}, --{k.replace('_', '-')} is {enc_set[k]}: start a new run")
    if ck and (args.offset_weight > 0 or ck.get("offset_weight", 0.0) > 0) and ck.get("offset_mode", "along") != args.offset_mode:
        raise SystemExit(f"{run}/last.pt has offset_mode={ck.get('offset_mode', 'along')}, --offset-mode is {args.offset_mode}: start a new run")
    tilt_def = "symmetric_angle" if automate else "full_angle"
    t_mode = "first_seated" if automate else "last_step"
    if ck and ck.get("split_mode", "split320") != args.split_mode:
        raise SystemExit(f"{run}/last.pt has split_mode={ck.get('split_mode', 'split320')}, --split-mode is {args.split_mode}: start a new run")
    if ck and ck.get("tilt_definition", "full_angle") != tilt_def:
        raise SystemExit(f"{run}/last.pt has tilt_definition={ck.get('tilt_definition', 'full_angle')}, this code uses {tilt_def}: start a new run")
    if ck and ck.get("term_mode", "last_step") != t_mode:
        raise SystemExit(f"{run}/last.pt has term_mode={ck.get('term_mode', 'last_step')}, this code uses {t_mode}: start a new run")
    data_max_pin_mm = max([20.0] + [float(math.ceil(float(e["pin_length_m"]) * 1000.0))
                                    for e in by["train"] if e.get("pin_length_m") is not None])
    if ck:
        max_pin_mm = float(ck.get("max_pin_mm") or 20.0)
        if data_max_pin_mm > max_pin_mm:
            raise SystemExit(f"{run}/last.pt has max_pin_mm={max_pin_mm}, training data needs {data_max_pin_mm}: start a new run")
    else:
        max_pin_mm = data_max_pin_mm
    try:                                                              # wm320.auc at the first validation: refuse now,
        import sklearn.metrics  # noqa: F401                          # not at step 500
    except ImportError as exc:
        raise SystemExit(f"scikit-learn is missing ({exc}): the validation AUCs (wm320.auc) need it")
    if ck:
        stats = ck["stats"]
    else:
        box = [stats_for(args.data, by["train"], seed=args.seed) if main_rank else None]
        if ddp:
            dist.broadcast_object_list(box, src=0)
        stats = box[0]
    neff_cache: dict[str, int] = {}
    rng = np.random.default_rng(1)
    sets, face_kept = {}, {}
    for name in ("select", "heldout", *(AUTOMATE_HELDOUT_CONDITIONS if automate else ())) if main_rank else ():
        if not by[name]:
            continue
        wi = (window_items(by[name], effective_decisions_for(args.data, by[name], neff_cache, workers=max(1, min(4, args.workers_wm))))
              if automate else window_items(by[name]))
        wi = wi[np.sort(rng.choice(len(wi), min(args.eval_windows, len(wi)), replace=False))]
        fi = face_items(by[name], args.data, args.shaping, args.gamma)
        if automate:                                                # coverage of the face AUC (AutoMate only)
            face_kept[name] = [int(len(fi)), int(sum(e["kind"] == "contact" for e in by[name]))]
        bce = [e for e in by[name] if e["bc"]]
        bi = D.policy_items(bce)
        bi = bi[np.sort(rng.choice(len(bi), min(1024, len(bi)), replace=False))]
        mk = lambda items: [{k: v.clone() for k, v in b.items()} for b in loader(WindowData(args.data, by[name], items, stats, args.shaping, args.gamma, offset_mode=args.offset_mode), 16, 4, 0, False)]  # noqa: E731
        sets[name] = (mk(wi), mk(fi) if len(fi) else [], [], [{k: v.clone() for k, v in b.items()} for b in policy_loader(args.data, bce, 32, 4, shuffle=False, items=bi)])
    if automate:
        tbox = [effective_decisions_for(args.data, by["train"], neff_cache, workers=max(1, args.workers_wm)) if main_rank else None]
        if ddp:
            dist.broadcast_object_list(tbox, src=0)
        train_wi = window_items(by["train"], tbox[0])
    else:
        train_wi = window_items(by["train"])
    torch.manual_seed(args.seed)
    model = WM320(args.stem, stats, args.shaping, imagenet=args.imagenet, offset_weight=args.offset_weight, offset_mode=args.offset_mode,
                  max_pin_mm=max_pin_mm, **enc_set, gamma=args.gamma)
    copied = None if ck else model.enc.pic.init_pretrained() if args.imagenet else 0   # identical on every process
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
    info0 = {"format": "wm320-1", "labels": WD.LABELS, "stem": args.stem, "imagenet": args.imagenet, "offset_weight": args.offset_weight, **enc_set,
             "offset_mode": args.offset_mode, "split_mode": args.split_mode, "tilt_definition": tilt_def, "term_mode": t_mode,
             "max_pin_mm": max_pin_mm, "shaping": args.shaping, "cfg": {"gamma": args.gamma}, "stats": stats, "provenance": prov}

    def checkpoint(full: bool):
        c = {**info0, "state": model.state_dict(), "step": step, "skipped": skipped, "args": vars(args)}
        if full:
            c.update(opt=opt.state_dict(), rng_cpu=torch.random.get_rng_state(),
                     rng_cuda=[torch.cuda.get_rng_state(dev)] if dev.type == "cuda" else None)
        return c

    log = Logger(run) if main_rank else (lambda **kw: None)
    log(event="resume" if ck else "start", step=step, processes=world, windows_per_step=world * args.wm_batch * args.accum,
        imitation_samples_per_step=world * args.bc_batch * args.accum, trajectories={k: len(v) for k, v in by.items()},
        windows=int(len(train_wi)), imitation_samples=int(sum(e["n_images"] for e in by["train"] if e["bc"])),
        validation={k: [len(v[0]) * 16, len(v[1]) * 16, len(v[2]) * 16, len(v[3]) * 32] for k, v in sets.items()},
        params_m=sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, imagenet=args.imagenet, imagenet_tensors_copied=copied,
        tokens_per_picture=model.enc.pic.n_tokens, **enc_set, max_pin_mm=max_pin_mm, ret_max=float(model.ret_max),
        tilt_definition=tilt_def, term_mode=t_mode, provenance=prov,
        select_grasps=len({e["grasp"] for e in by["select"]}), heldout_grasps=len({e["grasp"] for e in by["heldout"]}),
        **({"gpu_cap_GB": gpu_cap_gb} if gpu_cap_gb is not None else {}), args=vars(args))
    wo = int(round(args.online_frac * args.wm_batch)) if args.online else 0
    if args.online and (args.refresh_every < 1 or args.online_min < 1 or args.online_reload < 1):
        raise SystemExit("--refresh-every, --online-min and --online-reload must be at least 1")
    if args.online and not 0 < wo < args.wm_batch:
        raise SystemExit(f"--online-frac {args.online_frac} of {args.wm_batch} windows leaves no window to one of the two sources")
    rounds_in, onl, onl_wi, mix = None, [], np.zeros((0, 2), np.int64), False

    def poll_online(force: bool = False):
        """Read --online pool on rank 0 only, validate pin lengths, compute AutoMate n_eff, and broadcast in lockstep
        so DDP ranks never diverge or hang on a live pool refresh / error."""
        if main_rank:
            try:
                new, names = online_entries(args.online, (split_id or {}).get("sha256"), args.split_mode)
                bad_pin = [e for e in new if e.get("pin_length_m") is not None and float(e["pin_length_m"]) * 1000.0 > max_pin_mm]
                if bad_pin:
                    raise SystemExit(f"online entry {bad_pin[0].get('grasp')} has pin_length_m={bad_pin[0]['pin_length_m']}, "
                                     f"exceeding model max_pin_mm={max_pin_mm}")
                neffs = (effective_decisions_for(args.data, new, neff_cache, workers=max(1, args.workers_wm // 2))
                         if automate else None)
                n_win = sum(neffs) if neffs is not None else sum(e["steps"] // WD.EXECUTE for e in new)
                new_mix = len(new) >= args.online_min and n_win >= wo
                changed = force or names != rounds_in or len(new) - len(onl) >= args.online_reload or new_mix != mix
                obox = [(changed, new, names, neffs, new_mix, None) if changed else (False, None, None, None, None, None)]
            except SystemExit as exc:
                obox = [(False, None, None, None, None, ("exit", exc.code))]
            except Exception as exc:
                obox = [(False, None, None, None, None, ("err", f"{type(exc).__name__}: {exc}"))]
        else:
            obox = [None]
        if ddp:
            dist.broadcast_object_list(obox, src=0)
        changed, new, names, neffs, new_mix, err = obox[0]
        if err is not None:
            if err[0] == "exit":
                raise SystemExit(err[1])
            raise RuntimeError(err[1])
        return changed, new, names, (window_items(new, neffs) if changed else None), new_mix

    def make_loaders():
        """Windows: the fast copy alone, or wm_batch - wo from it and wo from the online pool; imitation: the
        successful demonstrations and the online attempts that ended seated (FK)."""
        sd = args.seed + RS * rank + step
        ws = WindowData(args.data, by["train"], train_wi, stats, args.shaping, args.gamma, offset_mode=args.offset_mode)
        wl = [cycle(loader(ws, args.wm_batch - (wo if mix else 0), args.workers_wm, sd))]
        if mix:                                                # (fewer windows than wo: a drop_last loader would never yield)
            wl.append(cycle(loader(WindowData(args.data, onl, onl_wi, stats, args.shaping, args.gamma, offset_mode=args.offset_mode), wo,
                                   args.workers_wm // 2, sd + 3)))
        bce = [e for e in by["train"] if e["bc"]] + ([e for e in onl if e["bc"]] if mix else [])     # online data only once mixed in
        return wl, cycle(loader(PolicyData(args.data, bce, bc_items(bce)), args.bc_batch, args.workers_bc, sd + 1))

    def data_record():
        log(event="data", step=step, online_rounds=rounds_in, online_trajectories=len(onl), online_seated=sum(e["bc"] for e in onl),
            online_partial_trajectories=sum(bool(e.get("partial")) for e in onl), online_windows_per_micro_batch=wo if mix else 0)

    if args.online:
        _, onl, rounds_in, onl_wi, mix = poll_online(force=True)
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
                changed, new, names, new_wi, new_mix = poll_online()
                if changed:
                    del wtrain, btrain                                     # the old loaders' workers end with them
                    onl, rounds_in, onl_wi, mix = new, names, new_wi, new_mix
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
                for name, kept in face_kept.items():                    # contact pushes in auc_V_before_face_contact / all
                    rec[name]["face_items_kept"] = kept
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
