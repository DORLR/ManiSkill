"""Collect the branch set (wm/collect/branch.py); resumable.

    python world_model/scripts/data/collect_branch.py --grasps $Q/data/wm_v1 --out /mnt/storage3/qingxin/data/wm_v1b \
        --split train --starts 6 --branches 4 --workers 6 \
        --wm <wm run>/last.pt --policy <policy run>/best.pt
`--split`: grasps of this split of wm_v1 (train: for training; ood_val: for checking only; test is refused).
Each worker holds one simulator and one copy of the policy (GPU memory capped by WM_GPU_MEM_FRACTION).
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

from wm.data.splits import split_of

_STATE = {"port": None, "robot": None, "agent": None, "init_error": None}


def _init(wm: str, policy: str):
    """Never raise here: an exception in a Pool initializer makes the pool respawn workers forever."""
    try:
        from wm.tdmpc.policy_resnet import ResnetPolicyAgent
        from wm.training.common import limit_gpu_memory
        limit_gpu_memory()
        _STATE["agent"] = ResnetPolicyAgent(wm, policy)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def work(args):
    g, out, starts, branches, seed, max_decisions = args
    if _STATE["init_error"]:
        return {"grasp": g["grasp"], "starts": [], "error": "agent init: " + _STATE["init_error"], "seconds": 0}
    from wm.collect.branch import collect_grasp
    from wm.env.robot import Robot
    try:
        if _STATE["port"] != g["port"]:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(g["port"]), g["port"]
    except Exception as exc:
        return {"grasp": g["grasp"], "starts": [], "error": f"environment: {type(exc).__name__}: {exc}", "seconds": 0}
    return collect_grasp(_STATE["robot"], _STATE["agent"], g, Path(out), starts, branches, seed, max_decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True, help="wm_v1 root (its grasp.json files give the grasps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "ood_val"])
    ap.add_argument("--wm", required=True, help="world-model checkpoint whose ResNet the policy reads")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-grasps", type=int)
    ap.add_argument("--every", type=int, default=1, help="smoke tests: every n-th grasp only")
    ap.add_argument("--max-decisions", type=int, default=150, help="0.4 s each")
    ap.add_argument("--seed", type=int, default=8_500_000)
    args = ap.parse_args()
    out = Path(args.out)
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    gs = [g for g in gs if g.get("reference_ok") and split_of(g["port"], g["condition"]) == args.split
          and not (out / g["port"] / g["grasp"] / "grasp.json").exists()]
    gs = gs[::args.every]
    if args.max_grasps:
        gs = gs[:args.max_grasps]
    gs.sort(key=lambda g: g["port"])
    jobs = [(g, str(out), args.starts, args.branches, args.seed + 100_003 * k, args.max_decisions) for k, g in enumerate(gs)]
    print(f"{len(jobs)} grasps x {args.starts} starts x {args.branches} branches, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.wm, args.policy),
                                      maxtasksperchild=20) as pool:
        for k, s in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            o = " | ".join(" ".join(f"{b['stopped'][:4]}/{b['outcome'][:4]}" for b in st["branches"]) for st in s["starts"])
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s['grasp']} {o} {s.get('error', '')[:100]} "
                  f"{s.get('seconds')}s {s.get('bytes', 0) / 1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
