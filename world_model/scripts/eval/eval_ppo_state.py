"""Closed-loop test of the state-based PPO baseline on the SAME grasps, starts and seeds as eval_policy320.py (the
grasp list, the even selection over it, the per-grasp start generator and the placement retries), and the same
result files and summary (its summarise / show, clustered standard errors, the one success criterion), and the same
episode: placement then snapshot/restore, the global numpy/torch generators seeded with seed + 1000 * start, the
"free" stops (no stop on "seated"), the 4-step final hold. The policy acts deterministically; observations are
normalised with the run's VecNormalize statistics. Crashed grasps are retried as in eval_policy320.py (fresh worker
pool per slice, pool rebuilt when a worker dies). eval_policy320.py's own pool loop is not reused because that file
must not change while the flow-matching tests import it.

    python world_model/scripts/eval/eval_ppo_state.py --grasps $Q/data/wm_v1 --split heldout --model <run>/final.zip \
        --vecnorm <run>/final_vecnormalize.pkl --scale <run>/action_scale.json --out <dir>
"""

import argparse
import glob
import json
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_policy320 as E  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

_S = {"init_error": None}
MAX_POOL_REBUILDS = E.MAX_POOL_REBUILDS


def _init(model, vecnorm, scale, shaping, gamma):
    try:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from wm.baselines.state_env import StateInsertEnv
        _S["model"] = PPO.load(model, device="cpu")
        with open(vecnorm, "rb") as f:
            _S["vn"] = pickle.load(f)
        _S["env"] = StateInsertEnv([], json.loads(Path(scale).read_text()), 0, shaping, gamma, stop_on_seated=False,
                                   catch_errors=False)
    except Exception as exc:
        _S["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def work(job):
    import torch
    from wm.collect import branch as B
    g, out_file, starts, seed = job
    s = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    s.update(stop="free", starts=[], error="interrupted", error_kind="crash")    # stays if anything ends this function early
    t0 = time.time()
    try:
        if _S["init_error"]:
            raise RuntimeError("init: " + _S["init_error"])
        env, model, vn = _S["env"], _S["model"], _S["vn"]
        if not env.use_grasp(g):
            s.update(error="reference insertion or stand-off failed", error_kind="setup")
            return s                                                            # (finally still writes the file)
        rng = np.random.default_rng(seed)
        for i in range(starts):
            ok, fails = False, []
            for _ in range(B.PLACEMENT_ATTEMPTS):
                start = B.sample_start(rng)
                stage = env.ex.place(start)
                if stage == "ok":
                    ok = True
                    break
                fails.append(stage)
            entry = {"idx": i, "start": start, "placement_failures": fails}
            s["starts"].append(entry)
            if not ok:
                entry["placement_failed"] = True
                continue
            env.ex.snapshot()
            env.ex.restore()
            torch.manual_seed(seed + 1000 * i)
            np.random.seed((seed + 1000 * i) % (2 ** 32))
            obs, done, info = env.start_episode(), False, {}
            while not done:
                a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
                obs, _, term, trunc, info = env.step(a[0])
                done = term or trunc
            entry["attempt"] = {"success": bool(info["is_success"]), **{k: info[k] for k in ("stopped", "decisions", "seated_fk", "first_seated_decision",
                                                                                            "depth_left_mm", "angle_deg", "lateral_mm")}}
        del s["error"], s["error_kind"]
    except Exception as exc:
        s.update(error=f"{type(exc).__name__}: {exc}", error_kind="crash", trace=traceback.format_exc()[-2000:])
    finally:
        s["seconds"] = round(time.time() - t0, 1)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(s))
    return s


def run_pool(jobs: list, args) -> None:
    """eval_policy320.run_pool with this script's worker setup."""
    left, rebuilds = list(jobs), 0
    t0 = time.time()
    while left and rebuilds <= MAX_POOL_REBUILDS:
        ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(args.model, args.vecnorm, args.scale, args.shaping, args.gamma))
        futs = {ex.submit(work, j): j for j in left[:10 * args.workers]}
        broken = False
        for f in as_completed(futs):
            try:
                s = f.result()
            except BrokenProcessPool:
                broken = True
                break
            except Exception as exc:
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--shaping", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1 - 1 / 30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grasps-n", type=int)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        raise SystemExit("ports A-D are the final test and stay untouched: --split test needs --final-test")
    out = Path(args.out)
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    every = [g for g in every if g.get("reference_ok") and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == args.split]
    chosen = E.pick(len(every), args.grasps_n)                     # k = index in the whole list: a grasp keeps its seed
    gs = [every[k] for k in chosen]
    path = {g["grasp"]: out / g["port"] / f"{g['grasp']}.json" for g in gs}
    print(f"{len(gs)} of {len(every)} grasps of split {args.split} x {args.starts} starts, {args.workers} workers, stop=free", flush=True)
    for _ in range(1 + args.retries):
        todo = [(k, g) for k, g in zip(chosen, gs) if E.grasp_state(path[g["grasp"]]) in ("missing", "crash")]
        if not todo:
            break
        for _, g in todo:
            path[g["grasp"]].unlink(missing_ok=True)
        todo.sort(key=lambda kg: kg[1]["port"])
        run_pool([(g, str(path[g["grasp"]]), args.starts, args.seed + 100_003 * k) for k, g in todo], args)
    out.mkdir(parents=True, exist_ok=True)
    summary, complete = E.summarise(out, gs, args.starts, {"policy": "ppo_state (privileged state)", "model": args.model, "vecnorm": args.vecnorm,
                                                          "scale": args.scale, "split": args.split,
                                                          "stop": "free", "starts_per_grasp": args.starts})
    (out / ("summary.json" if complete else "summary_incomplete.json")).write_text(json.dumps(summary, indent=1))
    print(E.show(summary), flush=True)
    sys.exit(0 if complete else 2)


if __name__ == "__main__":
    main()
