"""Closed-loop test of a branch model (wm/tdmpc/closed_loop.py); resumable.

    WM_CAM_SCALE=2 python world_model/scripts/eval/eval_branch_closed_loop.py --grasps $Q/data/wm_v1 --split ood_val \
        --ckpt <run>/last.pt --abort-below <v0 threshold> --out <run>/closed_loop_ood_val --grasps-n 150 --starts 2 \
        --policy <policy run>/best.pt
Starts are drawn afresh (seed 8.9 M; the branch set used 8.5 M and 8.7 M). `test` grasps are refused.
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

from wm.data.splits import split_of

_STATE = {"port": None, "robot": None, "agent": None, "scorer": None, "init_error": None}


def _init(policy, ckpt):
    try:
        from wm.tdmpc.policy_resnet import ResnetPolicyAgent
        from wm.tdmpc.closed_loop import Scorer
        from wm.training.common import limit_gpu_memory
        limit_gpu_memory()
        _STATE["agent"] = ResnetPolicyAgent(ckpt, policy)
        _STATE["scorer"] = Scorer(ckpt)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def work(args):
    g, out_file, starts, seed, modes, abort_below, max_decisions = args
    if _STATE["init_error"]:
        return {"grasp": g["grasp"], "starts": [], "error": "init: " + _STATE["init_error"], "seconds": 0}
    from wm.env.robot import Robot
    from wm.tdmpc.closed_loop import test_grasp
    try:
        if _STATE["port"] != g["port"]:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(g["port"]), g["port"]
    except Exception as exc:
        return {"grasp": g["grasp"], "starts": [], "error": f"environment: {type(exc).__name__}: {exc}", "seconds": 0}
    return test_grasp(_STATE["robot"], _STATE["agent"], _STATE["scorer"], g, Path(out_file), starts, seed, modes, abort_below, max_decisions)


def main():
    from wm.tdmpc.closed_loop import MODES, summarise
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--split", required=True, choices=["train", "ood_val"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--abort-below", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--grasps-n", type=int, default=150)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=8_900_000)
    ap.add_argument("--only", help="comma-separated grasp names (checks; seeds as in the full run)")
    args = ap.parse_args()
    out = Path(args.out)
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    gs = [g for g in gs if g.get("reference_ok") and split_of(g["port"], g["condition"]) == args.split]
    step = max(len(gs) // args.grasps_n, 1)
    gs = gs[::step][:args.grasps_n]
    todo = [(k, g) for k, g in enumerate(gs) if not (out / g["port"] / f"{g['grasp']}.json").exists()]
    if args.only:
        todo = [(k, g) for k, g in todo if g["grasp"] in args.only.split(",")]
    todo.sort(key=lambda kg: kg[1]["port"])
    modes = args.modes.split(",")
    jobs = [(g, str(out / g["port"] / f"{g['grasp']}.json"), args.starts, args.seed + 100_003 * k, modes, args.abort_below,
             args.max_decisions) for k, g in todo]
    print(f"{len(gs)} grasps ({len(jobs)} to do) x {args.starts} starts x modes {modes}, {args.workers} workers", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.policy, args.ckpt),
                                          maxtasksperchild=10) as pool:
            for k, s in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
                o = " | ".join(" ".join(f"{m[:6]}={'Y' if r['success'] else 'n'}{len(r['attempts'])}" for m, r in st.get("modes", {}).items())
                               for st in s["starts"])
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s['grasp']} {o} {s.get('error', '')[:150]} {s.get('seconds')}s", flush=True)
    rep = summarise(sorted(glob.glob(f"{out}/*/*.json")))
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
