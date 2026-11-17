"""Closed-loop test of the ResNet-token policy alone (wm/tdmpc/closed_loop_policy.py); resumable. Same grasps, starts
and seeds as eval_branch_closed_loop.py; `--old` names that run's folder to print the old policy on the same starts.

    WM_CAM_SCALE=2 python world_model/scripts/eval/eval_branch_policy.py --grasps $Q/data/wm_v1 --split ood_val \
        --wm <wm run>/last.pt --policy <policy run>/best.pt --out <policy run>/closed_loop_ood_val --old <wm run>/closed_loop_ood_val
"""

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from wm.data.splits import split_of

_STATE = {"port": None, "robot": None, "agent": None, "init_error": None}


def _init(wm, policy):
    try:
        from wm.tdmpc.policy_resnet import ResnetPolicyAgent
        from wm.training.common import limit_gpu_memory
        limit_gpu_memory()
        _STATE["agent"] = ResnetPolicyAgent(wm, policy)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def work(args):
    g, out_file, starts, seed, max_decisions = args
    if _STATE["init_error"]:
        return {"grasp": g["grasp"], "starts": [], "error": "init: " + _STATE["init_error"], "seconds": 0}
    from wm.env.robot import Robot
    from wm.tdmpc.closed_loop_policy import test_grasp_policy
    try:
        if _STATE["port"] != g["port"]:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(g["port"]), g["port"]
    except Exception as exc:
        return {"grasp": g["grasp"], "starts": [], "error": f"environment: {type(exc).__name__}: {exc}", "seconds": 0}
    return test_grasp_policy(_STATE["robot"], _STATE["agent"], g, Path(out_file), starts, seed, max_decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--split", required=True, choices=["train", "ood_val"])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old")
    ap.add_argument("--grasps-n", type=int, default=150)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=8_900_000)
    ap.add_argument("--only")
    args = ap.parse_args()
    out = Path(args.out)
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    gs = [g for g in gs if g.get("reference_ok") and split_of(g["port"], g["condition"]) == args.split]
    gs = gs[::max(len(gs) // args.grasps_n, 1)][:args.grasps_n]
    todo = [(k, g) for k, g in enumerate(gs) if not (out / g["port"] / f"{g['grasp']}.json").exists()]
    if args.only:
        todo = [(k, g) for k, g in todo if g["grasp"] in args.only.split(",")]
    todo.sort(key=lambda kg: kg[1]["port"])
    jobs = [(g, str(out / g["port"] / f"{g['grasp']}.json"), args.starts, args.seed + 100_003 * k, args.max_decisions) for k, g in todo]
    print(f"{len(gs)} grasps ({len(jobs)} to do) x {args.starts} starts, {args.workers} workers", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.wm, args.policy), maxtasksperchild=10) as pool:
            for k, s in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
                o = " | ".join(f"{a['stopped'][:5]}/{a['outcome'][:4]}" for st in s["starts"] for a in st.get("modes", {}).get("resnet_policy", {}).get("attempts", []))
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {s['grasp']} {o} {s.get('error', '')[:150]} {s.get('seconds')}s", flush=True)
    new = {}
    for f in sorted(glob.glob(f"{out}/*/*.json")):
        s = json.loads(open(f).read())
        for st in s["starts"]:
            if st.get("modes"):
                new[(s["grasp"], st["idx"])] = st["modes"]["resnet_policy"]
    x = np.array([v["success"] for v in new.values()], float)
    rep = {"starts": len(x), "resnet_policy": {"success": float(x.mean()), "se": float(x.std() / max(len(x), 1) ** 0.5), "stopped": {}}}
    for v in new.values():
        a = v["attempts"][0]
        key = f"{a['stopped']}/{a['outcome']}"
        rep["resnet_policy"]["stopped"][key] = rep["resnet_policy"]["stopped"].get(key, 0) + 1
    if args.old:
        old = {}
        for f in sorted(glob.glob(f"{args.old}/*/*.json")):
            s = json.loads(open(f).read())
            for st in s["starts"]:
                if st.get("modes"):
                    old[(s["grasp"], st["idx"])] = st["modes"]["policy"]["success"]
        both = [k for k in new if k in old]
        a, b = np.array([new[k]["success"] for k in both], float), np.array([old[k] for k in both], float)
        d = a - b
        rep["against_old_policy_same_starts"] = {"starts": len(both), "resnet": float(a.mean()), "old": float(b.mean()), "diff": float(d.mean()),
                                                 "se": float(d.std() / max(len(d), 1) ** 0.5), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
