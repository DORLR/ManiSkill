"""Closed-loop test of a 320 px policy alone (wm/tdmpc/closed_loop_policy.py); resumable.

One success criterion: depth left < 1.25 mm, tilt < 4 deg, lateral error < 2 mm on the simulator's truth at the end
of the attempt (demo320.SUCCESS). Pictures are rendered at the demonstrations' size (WM_CAM_SCALE must be unset).
Splits (demo320.split320): train, select (the grasps set aside for choosing checkpoints), heldout (interior strips
and unseen shapes), test (ports A-D: only with --final-test). --grasps-n picks evenly over the whole sorted list.
Every grasp gets a file. A crash (exception, dead worker, failed start of a worker) is run again --retries times;
if crashes remain, the result goes to summary_incomplete.json and the exit code is 2: summary.json exists only for
a complete test. Placement failures (a start pose the arm cannot reach) and unusable grasps (reference insertion or
stand-off failed) are not attempts: they are counted and reported, not hidden.
Standard error by clusters (two starts of a grasp, and all grasps of an unseen shape, are not independent):
cluster = the port shape for unseen shapes, the grasp otherwise.

    python world_model/scripts/eval/eval_policy320.py --grasps $Q/data/wm_v1 --split heldout --policy <run>/step40000.pt --out <dir>
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from wm.tdmpc import demo320 as D

_STATE = {"port": None, "robot": None, "agent": None, "init_error": None}
MAX_POOL_REBUILDS = 3


def _init(policy: str, jpeg: bool):
    try:
        from wm.env import robot as _robot
        if _robot.CAM_SCALE != 1:
            raise RuntimeError(f"WM_CAM_SCALE={_robot.CAM_SCALE}: the policy reads the native 320 px render, unset it")
        from wm.tdmpc.policy320 import Policy320Agent
        from wm.training.common import limit_gpu_memory
        limit_gpu_memory()
        _STATE["agent"] = Policy320Agent(policy, jpeg_roundtrip=jpeg)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def work(job):
    g, out_file, starts, seed, max_decisions, stop = job

    def crash(msg):
        s = {"grasp": g["grasp"], "port": g["port"], "starts": [], "error": msg, "error_kind": "crash", "seconds": 0}
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(s))
        return s

    if _STATE["init_error"]:
        return crash("init: " + _STATE["init_error"])
    from wm.env.robot import Robot
    from wm.tdmpc.closed_loop_policy import test_grasp_policy
    try:
        if _STATE["port"] != g["port"]:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(g["port"]), g["port"]
    except Exception as exc:
        return crash(f"environment: {type(exc).__name__}: {exc}"[:500])
    return test_grasp_policy(_STATE["robot"], _STATE["agent"], g, Path(out_file), starts, seed, max_decisions, stop)


# ---------------------------------------------------------------- no simulator below this line (tested in tests/test_eval_stats.py)
def pick(n_total: int, n: int | None) -> list[int]:
    """Indices of n grasps spread evenly over the whole sorted list (all of them if n is None or not smaller)."""
    if not n or n >= n_total:
        return list(range(n_total))
    return sorted(set(np.linspace(0, n_total - 1, n).round().astype(int).tolist()))


def cluster_rate(success, cluster) -> dict:
    """Success rate over attempts and its standard error when attempts of one cluster are not independent:
    p = sum(s_c) / sum(n_c), se = sqrt(G / (G - 1) * sum((s_c - p n_c)^2)) / sum(n_c) over the G clusters."""
    n, s = defaultdict(int), defaultdict(float)
    for ok, c in zip(success, cluster):
        n[c] += 1
        s[c] += float(ok)
    N, G = sum(n.values()), len(n)
    if not N:
        return {"starts": 0, "clusters": 0, "success": None, "se": None, "ci95": None}
    p = sum(s.values()) / N
    se = math.sqrt(G / (G - 1) * sum((s[c] - p * n[c]) ** 2 for c in n)) / N if G > 1 else None
    return {"starts": N, "clusters": G, "success": p, "se": se,
            "ci95": None if se is None else [max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)]}


def grasp_state(f: Path) -> str:
    """done | setup (unusable grasp) | crash | missing."""
    if not f.exists():
        return "missing"
    try:
        s = json.loads(f.read_text())
    except ValueError:
        return "crash"
    return "done" if not s.get("error") else "setup" if s.get("error_kind") == "setup" else "crash"


def summarise(out: Path, gs: list[dict], starts: int, extra: dict) -> tuple[dict, bool]:
    rows, count = [], Counter()
    for g in gs:
        f = out / g["port"] / f"{g['grasp']}.json"
        state = grasp_state(f)
        count[state] += 1
        if state != "done":
            continue
        grp = D.group320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"])
        for st in json.loads(f.read_text())["starts"]:
            if "attempt" in st:
                rows.append((grp, g["port"] if grp == "unseen_shape" else g["grasp"], bool(st["attempt"]["success"]), st["attempt"]["stopped"], g["grasp"]))
            else:
                count["placement_failed_starts"] += 1
    groups = {}
    for name in ["all"] + sorted({r[0] for r in rows}):
        sel = [r for r in rows if name == "all" or r[0] == name]
        groups[name] = {**cluster_rate([r[2] for r in sel], [r[1] for r in sel]), "grasps": len({r[4] for r in sel}),
                        "ended": dict(Counter(f"{r[3]}/{'success' if r[2] else 'failure'}" for r in sel))}
    complete = count["crash"] == 0 and count["missing"] == 0
    summary = {"criterion": "depth left < 1.25 mm and tilt < 4 deg and lateral < 2 mm (simulator truth at the end)", "complete": complete,
               "planned_grasps": len(gs), "planned_starts": len(gs) * starts, "judged_starts": len(rows),
               "placement_failed_starts": count["placement_failed_starts"], "unusable_grasps": count["setup"],
               "crashed_grasps": count["crash"] + count["missing"], "groups": groups, **extra}
    return summary, complete


def show(summary: dict) -> str:
    lines = [f"{summary['judged_starts']} of {summary['planned_starts']} planned starts judged; placement failed {summary['placement_failed_starts']}, "
             f"unusable grasps {summary['unusable_grasps']}, crashed grasps {summary['crashed_grasps']}"]
    for name, g in summary["groups"].items():
        if g["success"] is None:
            continue
        se = "" if g["se"] is None else f" +- {100 * g['se']:.1f}"
        lines.append(f"  {name:15s} success {100 * g['success']:.1f} %{se}  ({g['starts']} starts, {g['grasps']} grasps, {g['clusters']} clusters)")
    return "\n".join(lines)


def run_pool(jobs: list, args) -> None:
    left, rebuilds = list(jobs), 0
    t0 = time.time()
    while left and rebuilds <= MAX_POOL_REBUILDS:
        # A fresh pool per slice of 10 grasps per worker, so that simulator processes are replaced regularly.
        # (max_tasks_per_child does this in principle, but on Python 3.11 a worker that leaves is not replaced when
        # all jobs were submitted at once: the test then waits forever.)
        ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(args.policy, args.jpeg_roundtrip == "on"))
        futs = {ex.submit(work, j): j for j in left[:10 * args.workers]}
        broken = False
        for f in as_completed(futs):
            try:
                s = f.result()
            except BrokenProcessPool:
                broken = True
                break
            except Exception as exc:               # the grasp has no file: the caller counts it as a crash and may retry it
                print(f"{futs[f][0]['grasp']} failed: {type(exc).__name__}: {exc}"[:300], flush=True)
                continue
            o = " | ".join(f"{st['attempt']['stopped'][:5]}/{'ok' if st['attempt']['success'] else 'no'}" if "attempt" in st else "unplaced"
                           for st in s["starts"])
            print(f"[{time.time() - t0:.0f}s] {s['grasp']} {o} {str(s.get('error', ''))[:150]} {s.get('seconds')}s", flush=True)
        ex.shutdown(wait=not broken, cancel_futures=True)
        if broken:
            rebuilds += 1
            print(f"a worker died; pool restart {rebuilds} of {MAX_POOL_REBUILDS}", flush=True)
        ran = {id(j) for f, j in futs.items() if f.done() and not f.cancelled() and not isinstance(f.exception(), BrokenProcessPool)}
        left = [j for j in left if id(j) not in ran]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--split", required=True, choices=["train", "select", "heldout", "test"])
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grasps-n", type=int)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--stop", default="free", choices=["free", "reference"])
    ap.add_argument("--jpeg-roundtrip", default="on", choices=["on", "off"])
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        raise SystemExit("ports A-D are the final test and stay untouched: --split test needs --final-test")
    out = Path(args.out)
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    every = [g for g in every if g.get("reference_ok") and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == args.split]
    chosen = pick(len(every), args.grasps_n)                     # k = index in the whole list: a grasp keeps its seed
    gs = [every[k] for k in chosen]
    path = {g["grasp"]: out / g["port"] / f"{g['grasp']}.json" for g in gs}
    print(f"{len(gs)} of {len(every)} grasps of split {args.split} x {args.starts} starts, {args.workers} workers, stop={args.stop}", flush=True)
    for _ in range(1 + args.retries):
        todo = [(k, g) for k, g in zip(chosen, gs) if grasp_state(path[g["grasp"]]) in ("missing", "crash")]
        if not todo:
            break
        for _, g in todo:
            path[g["grasp"]].unlink(missing_ok=True)
        todo.sort(key=lambda kg: kg[1]["port"])
        run_pool([(g, str(path[g["grasp"]]), args.starts, args.seed + 100_003 * k, args.max_decisions, args.stop) for k, g in todo], args)
    summary, complete = summarise(out, gs, args.starts, {"policy": args.policy, "split": args.split, "stop": args.stop,
                                                        "jpeg_roundtrip": args.jpeg_roundtrip, "starts_per_grasp": args.starts})
    out.mkdir(parents=True, exist_ok=True)
    (out / ("summary.json" if complete else "summary_incomplete.json")).write_text(json.dumps(summary, indent=1))
    print(show(summary), flush=True)
    sys.exit(0 if complete else 2)


if __name__ == "__main__":
    main()
