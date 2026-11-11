"""Collect policy-independent contact exploration (wm/collect/explore.py); resumable.

    python world_model/scripts/data/collect_explore.py --grasps $Q/data/wm_v1 --out $Q/data/wm_v1x \
        --split train --per-grasp 10 --workers 16
`--split` : grasps of this split of wm_v1 (train: for training; ood_val: for checking only; test is refused).
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

from wm.data.splits import split_of

_STATE = {"port": None, "robot": None}


def work(args):
    g, out, n, seed = args
    from wm.collect.explore import collect_grasp
    from wm.env.robot import Robot
    try:
        if _STATE["port"] != g["port"]:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(g["port"]), g["port"]
    except Exception as exc:
        return {"grasp": g["grasp"], "trajectories": [], "error": f"environment: {type(exc).__name__}: {exc}", "seconds": 0}
    return collect_grasp(_STATE["robot"], g, Path(out), n, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True, help="wm_v1 root (its grasp.json files give the grasps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "ood_val"])
    ap.add_argument("--per-grasp", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="+", help="smoke tests: these shapes")
    ap.add_argument("--max-grasps", type=int)
    ap.add_argument("--seed", type=int, default=6_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    gs = [g for g in gs if g.get("reference_ok") and split_of(g["port"], g["condition"]) == args.split
          and (not args.only or g["port"] in args.only) and not (out / g["port"] / g["grasp"] / "grasp.json").exists()]
    if args.max_grasps:
        gs = gs[:args.max_grasps]
    gs.sort(key=lambda g: g["port"])
    jobs = [(g, str(out), args.per_grasp, args.seed + 7919 * k) for k, g in enumerate(gs)]
    print(f"{len(jobs)} grasps x {args.per_grasp} trajectories, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=40) as pool:
        for k, s in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            o = " ".join(f"{t.get('outcome', '?')[:6]}" for t in s["trajectories"])
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s['grasp']} {o} {s.get('error', '')[:100]} {s.get('seconds')}s", flush=True)


if __name__ == "__main__":
    main()
