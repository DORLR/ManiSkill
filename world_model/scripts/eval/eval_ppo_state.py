"""Closed-loop test of the state-based PPO baseline on the SAME grasps, starts and seeds as eval_policy320.py (the
grasp list, the even selection over it, the per-grasp start generator and the placement retries), and the same
result files and summary (its summarise / show, clustered standard errors, the one success criterion). The policy
acts deterministically; observations are normalised with the run's VecNormalize statistics.

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
import multiprocessing as mp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_policy320 as E  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

_S = {}


def _init(model, vecnorm, scale, shaping, gamma):
    from stable_baselines3 import PPO
    from wm.baselines.state_env import StateInsertEnv
    _S["model"] = PPO.load(model, device="cpu")
    with open(vecnorm, "rb") as f:
        _S["vn"] = pickle.load(f)
    _S["env"] = StateInsertEnv([], np.array(json.loads(Path(scale).read_text())), 0, shaping, gamma)


def work(job):
    from wm.collect import branch as B
    g, out_file, starts, seed = job
    env, model, vn = _S["env"], _S["model"], _S["vn"]
    s = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    s.update(stop="free", starts=[])
    t0 = time.time()
    try:
        if not env.use_grasp(g):
            s.update(error="reference insertion or stand-off failed", error_kind="setup")
            return s
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
            obs, done, info = env.start_episode(), False, {}
            while not done:
                a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
                obs, _, term, trunc, info = env.step(a[0])
                done = term or trunc
            entry["attempt"] = {"success": bool(info["is_success"]), **{k: info[k] for k in ("stopped", "decisions", "seated_fk", "depth_left_mm", "angle_deg", "lateral_mm")}}
    except Exception as exc:
        s.update(error=f"{type(exc).__name__}: {exc}", error_kind="crash", trace=traceback.format_exc()[-2000:])
    finally:
        s["seconds"] = round(time.time() - t0, 1)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(s))
    return s


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
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        raise SystemExit("ports A-D are the final test and stay untouched: --split test needs --final-test")
    out = Path(args.out)
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    every = [g for g in every if g.get("reference_ok") and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == args.split]
    chosen = E.pick(len(every), args.grasps_n)
    gs = [every[k] for k in chosen]
    jobs = [(g, str(out / g["port"] / f"{g['grasp']}.json"), args.starts, args.seed + 100_003 * k) for k, g in zip(chosen, gs)
            if E.grasp_state(out / g["port"] / f"{g['grasp']}.json") in ("missing", "crash")]
    jobs.sort(key=lambda j: j[0]["port"])
    print(f"{len(gs)} grasps of split {args.split}, {len(jobs)} to run, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(args.model, args.vecnorm, args.scale, args.shaping, args.gamma)) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            s = f.result()
            print(s["grasp"], [("ok" if st["attempt"]["success"] else "no") if "attempt" in st else "unplaced" for st in s["starts"]], s.get("error", ""), flush=True)
    summary, complete = E.summarise(out, gs, args.starts, {"policy": "ppo_state (privileged state)", "model": args.model, "split": args.split})
    (out / ("summary.json" if complete else "summary_incomplete.json")).write_text(json.dumps(summary, indent=1))
    print(E.show(summary), flush=True)
    sys.exit(0 if complete else 2)


if __name__ == "__main__":
    main()
