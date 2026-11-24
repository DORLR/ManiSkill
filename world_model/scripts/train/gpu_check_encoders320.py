"""GPU check of the picture encoders of the 320 px world model (policy320.PictureTokens: resnet18 / resnet34 /
segformer_b1) before an AutoMate training run, on ONE GPU, with the training step of train_wm320.py (WM320.loss,
backward, gradient clip 20, Adam, target update) on random windows and imitation samples. Per encoder it reports:
  dtypes      the output dtype of the first layer (must stay float32 inside the bfloat16 autocast: the resnet stem /
              encoders320.Float32Conv2d), of a later layer (bfloat16 expected) and of the picture tokens
  frozen BN   segformer_b1: the decoder's batch-norm running mean / var / num_batches_tracked bit-identical after the
              training steps in train(), train() output == eval() output (encoders320.FrozenStatsBatchNorm2d), weight
              and bias receive gradients
  memory      torch.cuda.max_memory_allocated / max_memory_reserved over all steps [GB], per (--wm-batch, --bc-batch)
  step time   median wall time of steps 3..--steps [s] (synchronised), per (--wm-batch, --bc-batch); the model step
              only: random windows in GPU memory, no data loading (train_wm320's loaders decode JPEG / PNG on the CPU)
An out-of-memory step is reported as such for that batch pair (the next pair still runs). The default pairs end with
the legacy recipe's micro-batch (16 windows, 32 imitation samples: train_wm320.py's docstring, ResNet-18 on a 24 GB
A5000), so the measurement says whether the agreed recipe fits; if it does not, the numbers go to the user (no smaller
recipe is chosen silently; a smaller micro-batch with a larger --accum gives the same optimiser step).
Results go to --out (json; required for a GPU run, written after every encoder, so a later failure keeps the earlier
numbers). Its folder must exist and be writable (checked before CUDA is touched): the launcher runs this script from
the read-only frozen code copy, so give an absolute path in the run folder.
Depth normalisation: --depth-stats (scripts/automate/depth_stats.py output), else --depth-max-m / --depth-mean-m /
--depth-std-m with depth_invalid = (0 - mean) / std (policy320.default_depth_invalid), checked by
policy320.check_depth_settings, as train_wm320.py does. Before any AutoMate fast copy exists the flags carry stand-in
values (e.g. 0.5 / 0.3 / 0.06): memory and step time do not depend on them; info.depth_source records which was used.

GPU rules: refuses unless exactly one GPU is visible, CUDA_VISIBLE_DEVICES equals --expect-visible (default "1": GPU 0
is never used) and CUDA_DEVICE_ORDER=PCI_BUS_ID, and unless QX_GPU_GB (the scope's declared GPU memory, exported by
qx_run) is set; the process's GPU memory is capped with torch.cuda.set_per_process_memory_fraction at --mem-gb
(default QX_GPU_GB - 1: the CUDA context, about 1 GB, lies outside the allocator cap; a larger --mem-gb is refused).
The free memory of the GPU at the start is recorded (info free_GB_at_start): the measurement means something only
when the declared memory was free. Run it through the capped launcher (memory-capped scope). It imports the code
next to it (world_model/src), never another copy.
--cpu-dry-run: one tiny step per encoder on the CPU (float32, batch 1, no timing or memory), only to show that the
script runs against the current code; the GPU run is the measurement.

    bash world_model/scripts/automate/train_automate.sh <run dir> <MemoryMax> <CPUQuota> <GPU GB> \\
      scripts/train/gpu_check_encoders320.py --depth-max-m 0.5 --depth-mean-m 0.3 --depth-std-m 0.06 \\
      --out <run dir>/gpu_check.json
(resources are stated by the caller, e.g. 40G 800% 36; train_automate.sh: frozen code, GPU 1 only, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1, QX_GPU_GB = the
declared GPU GB, HF_HOME / TORCH_HOME / offline Hub; the declared memory must be free on GPU 1, i.e. before the
collection's workers take their 6 x 7 GB, or after they end.)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import check_depth_settings, default_depth_invalid  # noqa: E402
from wm.tdmpc.wm320 import WM320  # noqa: E402

ENCODERS = ("resnet18", "resnet34", "segformer_b1")
GPU_CONTEXT_GB = 1.0                   # the CUDA context, outside the allocator cap (as train_wm320.GPU_CONTEXT_GB)
STATS = {"s0": np.full(WD.ACTION_DIM, 0.05, np.float32), "top": np.full(WD.ACTION_DIM, 0.5, np.float32),
         "a_mean": np.zeros(112, np.float32), "a_std": np.full(112, 0.1, np.float32)}


def depth_settings(a) -> dict:
    if a.depth_stats:
        st = json.loads(Path(a.depth_stats).read_text())
        d = {k: float(st[k]) for k in ("depth_max_m", "depth_mean_m", "depth_std_m", "depth_invalid")}
        a.depth_source = f"statistics file {Path(a.depth_stats).resolve()}"
    else:
        a.depth_source = "flags (stand-in values unless taken from the data; memory and step time do not depend on them)"
        if None in (a.depth_max_m, a.depth_mean_m, a.depth_std_m):
            raise SystemExit("give --depth-stats or all of --depth-max-m, --depth-mean-m, --depth-std-m")
        d = {"depth_max_m": a.depth_max_m, "depth_mean_m": a.depth_mean_m, "depth_std_m": a.depth_std_m,
             "depth_invalid": default_depth_invalid(a.depth_mean_m, a.depth_std_m)}
    why = check_depth_settings(d["depth_max_m"], d["depth_mean_m"], d["depth_std_m"], d["depth_invalid"])
    if why:
        raise SystemExit(f"depth settings refused: {why}")
    return d


def batch(B: int, Bb: int, g: torch.Generator, dev) -> tuple[dict, dict]:
    S = WD.H + 4
    W = {"rgb": torch.randint(0, 256, (B, S, 320, 320, 3), dtype=torch.uint8, generator=g),
         "depth": torch.randint(700, 5000, (B, S, 320, 320), dtype=torch.int16, generator=g),
         "q": torch.randn(B, WD.H + 1, 7, generator=g), "action": torch.rand(B, WD.H, WD.ACTION_DIM, generator=g) * 2 - 1,
         "reward": torch.randn(B, WD.H, generator=g), "term": torch.zeros(B, WD.H), "valid": torch.ones(B, WD.H),
         "obs_valid": torch.ones(B, WD.H + 1), "v_valid": torch.ones(B, WD.H + 1),
         "a_next": torch.rand(B, WD.H + 1, WD.ACTION_DIM, generator=g) * 2 - 1, "a_next_valid": torch.ones(B, WD.H + 1),
         "done": torch.zeros(B, WD.H + 1), "fk_depth": torch.zeros(B, WD.H + 1), "probe": torch.zeros(B, WD.H + 1, 4),
         "offset": torch.zeros(B, WD.H + 1, 4)}
    W["depth"][:, :, :32, :32] = 0                                                   # some pixels without a return
    Bc = {"rgb": torch.randint(0, 256, (Bb, 4, 320, 320, 3), dtype=torch.uint8, generator=g),
          "depth": torch.randint(700, 5000, (Bb, 4, 320, 320), dtype=torch.int16, generator=g),
          "q": torch.randn(Bb, 7, generator=g), "a": torch.randn(Bb, 112, generator=g) * 0.1}
    return {k: v.to(dev) for k, v in W.items()}, {k: v.to(dev) for k, v in Bc.items()}


def layers(pic, enc: str):
    """(first layer, a later layer) of PictureTokens `pic`."""
    if enc == "segformer_b1":
        return pic.seg.segformer.encoder.patch_embeddings[0].proj, pic.seg.segformer.encoder.block[0][0].attention.self.query
    return pic.stem[0], [m for m in pic.trunk if isinstance(m, torch.nn.Sequential)][0][0].conv1


def build(enc: str, depth: dict, token_dim: int, dev):
    kw = {} if enc == "resnet18" else dict(imagenet=True, encoder=enc, token_dim=token_dim, **depth)
    torch.manual_seed(0)
    m = WM320("standard", STATS, 0.3, gamma=0.967, **kw)
    if enc != "resnet18":
        m.enc.pic.init_pretrained()
    return m.to(dev).train()


def run_steps(m, W, Bc, steps: int, dev, timed: bool) -> list[float]:
    enc = list(m.enc.parameters())
    ids = {id(p) for p in enc}
    rest = [p for p in m.parameters() if p.requires_grad and id(p) not in ids]
    opt = torch.optim.Adam([{"params": enc, "lr": 3e-4 * 0.3}, {"params": rest, "lr": 3e-4}])
    params = [p for p in m.parameters() if p.requires_grad]
    times = []
    for step in range(1, steps + 1):
        if timed:
            torch.cuda.synchronize(dev)
        t = time.time()
        opt.zero_grad(set_to_none=True)
        total, _ = m.loss(W, Bc, step, 5, 1.0)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 20.0)
        opt.step()
        m.update_target()
        if timed:
            torch.cuda.synchronize(dev)
        times.append(time.time() - t)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite loss at step {step}")
    return times


def check_encoder(enc: str, depth: dict, a, dev, cpu: bool) -> dict:
    res = {"encoder": enc}
    g = torch.Generator().manual_seed(0)
    m = build(enc, depth, a.token_dim, dev)
    first, later = layers(m.enc.pic, enc)
    box = {}

    def note(name):
        def hook(mod, i, o):                    # returns None: the output is left unchanged
            box.setdefault(name, str(o.dtype))
        return hook
    hooks = [first.register_forward_hook(note("first")), later.register_forward_hook(note("later")),
             m.enc.pic.register_forward_hook(note("tokens"))]
    bn = m.enc.pic.seg.decode_head.batch_norm if enc == "segformer_b1" else None
    keys = ("running_mean", "running_var", "num_batches_tracked")
    before = {k: getattr(bn, k).clone() for k in keys} if bn is not None else None
    W, Bc = batch(1, 1, g, dev)
    run_steps(m, W, Bc, 1 if cpu else 3, dev, timed=False)
    for h in hooks:
        h.remove()
    res["dtypes"] = dict(box, expected={"first": "torch.float32", "later": "torch.float32" if cpu else "torch.bfloat16"})
    res["dtypes_ok"] = box.get("first") == "torch.float32" and box.get("later") == res["dtypes"]["expected"]["later"]
    if bn is not None:
        x = torch.randn(2, bn.num_features, 80, 80, device=dev, generator=torch.Generator(dev).manual_seed(1)) * 2
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=not cpu):
            yt = bn(x)
            bn.eval()
            ye = bn(x)
            bn.train()
        res["frozen_bn"] = {"buffers_unchanged": all(torch.equal(getattr(bn, k), before[k]) for k in keys),
                            "train_equals_eval": bool(torch.equal(yt, ye)),
                            "affine_grads": bool(bn.weight.grad is not None and float(bn.weight.grad.abs().sum()) > 0
                                                 and bn.bias.grad is not None and float(bn.bias.grad.abs().sum()) > 0)}
    del m, W, Bc
    if cpu:
        return res
    res["runs"] = []
    for B, Bb in zip(a.wm_batch, a.bc_batch):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        r = {"wm_batch": B, "bc_batch": Bb}
        try:
            m = build(enc, depth, a.token_dim, dev)
            W, Bc = batch(B, Bb, g, dev)
            times = run_steps(m, W, Bc, a.steps, dev, timed=True)
            r.update(peak_allocated_GB=torch.cuda.max_memory_allocated(dev) / 1e9,
                     peak_reserved_GB=torch.cuda.max_memory_reserved(dev) / 1e9,
                     step_s_median_3_to_n=statistics.median(times[2:]), step_s_first=times[0])
        except torch.cuda.OutOfMemoryError as exc:
            r.update(out_of_memory=True, error=str(exc)[:300], peak_allocated_GB=torch.cuda.max_memory_allocated(dev) / 1e9)
        finally:
            m = W = Bc = None
        res["runs"].append(r)
        print(json.dumps({enc: r}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="resnet34,segformer_b1", help=f"comma list of {ENCODERS}")
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--wm-batch", default="2,4,8,16", help="comma list of micro-batch window counts (16: the legacy recipe)")
    ap.add_argument("--bc-batch", default="4,8,16,32",
                    help="comma list of imitation micro-batch sizes (paired with --wm-batch; 32: the legacy recipe)")
    ap.add_argument("--steps", type=int, default=10, help="training steps per batch pair (median over steps 3..n)")
    ap.add_argument("--depth-stats")
    ap.add_argument("--depth-max-m", type=float)
    ap.add_argument("--depth-mean-m", type=float)
    ap.add_argument("--depth-std-m", type=float)
    ap.add_argument("--mem-gb", type=float, default=None, help="cap on this process's GPU memory (default QX_GPU_GB - 1)")
    ap.add_argument("--expect-visible", default="1", help="the required CUDA_VISIBLE_DEVICES (GPU 0 is never used)")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.encoders = a.encoders.split(",")
    a.wm_batch = [int(x) for x in a.wm_batch.split(",")]
    a.bc_batch = [int(x) for x in a.bc_batch.split(",")]
    if set(a.encoders) - set(ENCODERS) or len(a.wm_batch) != len(a.bc_batch) or a.steps < 3:
        raise SystemExit("unknown encoder, unpaired --wm-batch / --bc-batch, or --steps < 3")
    depth = depth_settings(a)
    if not a.cpu_dry_run and not a.out:
        raise SystemExit("--out is required for a GPU run (the measurement must be kept)")
    if a.out:
        a.out = str(Path(a.out).resolve())
        if not Path(a.out).parent.is_dir() or not os.access(Path(a.out).parent, os.W_OK):
            raise SystemExit(f"--out {a.out}: its folder does not exist or is not writable (the frozen code copy is read-only)")
    if a.cpu_dry_run:
        dev = torch.device("cpu")
        info = {"device": "cpu (dry run)"}
    else:
        qx = os.environ.get("QX_GPU_GB")                        # checked first, before CUDA is touched
        try:
            budget = float(qx) - GPU_CONTEXT_GB
        except (TypeError, ValueError):
            raise SystemExit(f"QX_GPU_GB={qx!r}: start this check through the capped launcher (it exports the declared GPU memory)")
        if not budget > 0:
            raise SystemExit(f"QX_GPU_GB={qx}: nothing left after {GPU_CONTEXT_GB} GB for the CUDA context")
        if a.mem_gb is None:
            a.mem_gb = budget
        elif not 0 < a.mem_gb <= budget:
            raise SystemExit(f"--mem-gb {a.mem_gb} must lie in (0, QX_GPU_GB - {GPU_CONTEXT_GB} = {budget}]")
        cvd, order = os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER")
        if cvd != a.expect_visible or order != "PCI_BUS_ID":
            raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r}, CUDA_DEVICE_ORDER={order!r}: need {a.expect_visible!r} and 'PCI_BUS_ID'")
        if a.expect_visible.strip() == "0" or torch.cuda.device_count() != 1:
            raise SystemExit(f"{torch.cuda.device_count()} GPUs visible (need exactly one, never GPU 0)")
        dev = torch.device("cuda", 0)
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 1e9 / total), dev)
        p = torch.cuda.get_device_properties(dev)
        info = {"device": torch.cuda.get_device_name(dev), "uuid": str(getattr(p, "uuid", None)), "total_GB": total / 1e9,
                "free_GB_at_start": torch.cuda.mem_get_info(dev)[0] / 1e9, "cap_GB": a.mem_gb, "QX_GPU_GB": float(qx)}
    try:
        import transformers
        tv = transformers.__version__
    except ImportError:
        tv = None
    out = {"info": {**info, "torch": torch.__version__, "transformers": tv, "code": str(ROOT), "depth": depth,
                    "depth_source": a.depth_source, "token_dim": a.token_dim, "steps": a.steps,
                    "pairs": list(zip(a.wm_batch, a.bc_batch)), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, "encoders": []}
    for enc in a.encoders:
        r = check_encoder(enc, depth, a, dev, a.cpu_dry_run)
        out["encoders"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "runs"}), flush=True)
        if a.out:                                            # after every encoder: a later failure keeps these numbers
            tmp = Path(a.out + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(out, indent=1, default=str))
            os.replace(tmp, a.out)
    bad = [r["encoder"] for r in out["encoders"] if not r["dtypes_ok"] or not all((r.get("frozen_bn") or {"ok": True}).values())]
    if bad:
        raise SystemExit(f"dtype or frozen-BN check failed for {bad}")


if __name__ == "__main__":
    main()
