"""PPO baseline on the privileged state (wm/baselines/state_env.py), with Stable-Baselines3: PPO, VecNormalize
(observations and rewards), checkpoints with the normalisation statistics. The n_envs slots PPO steps in lock-step are
served by a pool of `--engines` simulator processes (wm/baselines/pool_vec_env.py, 2026-11-18: a slot whose episode
ends takes over an engine whose next start is already placed; with SubprocVecEnv every round waited for some
worker's 3-13 s reset). Training grasps only (the training split of demo320.split320: no selection, held-out or test
grasp); each engine gets whole ports (its own share of them, balanced by grasp count, so its simulator is rebuilt only
when it moves to another port); the engines' first resets (where their simulators are built) are 20 s apart. A resume refuses changed settings; every start or
resume writes its own log folder and args file.
Actions use the world model's asinh scaling (state_env.py): per joint the median (s0) and the 99.9th percentile (top,
at least 2 s0) over the training demonstrations' decisions of the largest |commanded target - joints measured at the
decision's start| among its 8 targets. The policy's initial exploration noise is std exp(-1) ~ 0.37
(SB3's default 1 makes random first moves leave the start box at once).
Resumable from the newest checkpoint; DONE.json at the end.

    python world_model/scripts/train/train_ppo_state.py --grasps $Q/data/wm_v1 --data $Q/data/wm_v1_fast --out <run>
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from wm.baselines.pool_vec_env import PoolVecEnv

from wm.tdmpc import demo320 as D
from wm.tdmpc import wmdata320 as WD


def train_grasps(root: str) -> list[dict]:
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{root}/*/*/grasp.json"))]
    if any(D.is_automate(g) for g in gs):                 # the state environment is ManiSkill's (SAPIEN)
        raise SystemExit("AutoMate grasps: the PPO state baseline runs in the ManiSkill simulator only")
    return [g for g in gs if g.get("reference_ok") and D.split_of_entry(g) == "train"]


def action_scale(data: str, n: int = 600, seed: int = 0) -> dict:
    ents = [e for e in D.load_entries(data) if e["split"] == "train" and e["bc"]]
    trs = []
    for i in np.random.default_rng(seed).choice(len(ents), min(n, len(ents)), replace=False):
        with np.load(Path(data) / ents[i]["path"]) as z:
            tr = WD.trajectory(z, json.loads(str(z["meta"])), 0.0, 0.99)
        if tr is not None:
            trs.append(tr)
    d = np.abs(np.concatenate([t["delta"] for t in trs])).reshape(-1, WD.EXECUTE, D.JOINTS).max(1)
    s0 = np.maximum(np.percentile(d, 50, axis=0), 1e-5)
    return {"s0": s0.tolist(), "top": np.maximum(np.percentile(d, 99.9, axis=0), 2 * s0).tolist()}


def make_env(grasps, scale, seed, shaping, gamma, per_grasp, delay_s):
    def f():
        from wm.baselines.state_env import StateInsertEnv
        return StateInsertEnv(grasps, scale, seed, shaping, gamma, per_grasp, start_delay_s=delay_s)
    return f


def port_shares(gs: list[dict], n: int) -> list[list[dict]]:
    """Whole ports to n workers, largest port first to the worker with the fewest grasps so far."""
    by = {}
    for g in gs:
        by.setdefault(g["port"], []).append(g)
    share = [[] for _ in range(n)]
    for p in sorted(by, key=lambda p: (-len(by[p]), p)):
        min(share, key=len).extend(by[p])
    return share


SAME = ("shaping", "gamma", "n_envs", "engines", "episodes_per_grasp", "seed", "n_steps", "batch", "epochs", "lr", "grasps", "data", "action_map")
ACTION_MAP = "asinh"                   # 2026-11-17: the linear map of the first smoke run is not resumable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=24)                # slots PPO steps in lock-step
    ap.add_argument("--engines", type=int, default=None)              # simulator processes serving them; default 2 x n_envs
    ap.add_argument("--total-steps", type=int, default=20_000_000)
    ap.add_argument("--shaping", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=1 - 1 / 30)
    ap.add_argument("--episodes-per-grasp", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=3072)          # divides n_steps x n_envs = 6144
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.action_map = ACTION_MAP                                  # stored in args.json; a resume must match it
    args.engines = 2 * args.n_envs if args.engines is None else args.engines
    if args.engines < args.n_envs:
        raise SystemExit(f"{args.engines} engines for {args.n_envs} slots")
    out = Path(args.out)
    if (out / "DONE.json").exists():
        raise SystemExit(f"{out} is finished")
    if not args.resume and (out / "args.json").exists():
        raise SystemExit(f"{out} already holds a run; use --resume to continue it")
    if args.resume and (out / "args.json").exists():
        old = json.loads((out / "args.json").read_text())
        diff = {k: (old.get(k), getattr(args, k)) for k in SAME if old.get(k) != getattr(args, k)}
        if diff:
            raise SystemExit(f"--resume with changed settings {diff}: start a new run instead")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    gs = train_grasps(args.grasps)
    if len({g["port"] for g in gs}) < args.engines:
        raise SystemExit(f"{len({g['port'] for g in gs})} training ports for {args.engines} engines")
    scale = action_scale(args.data, seed=args.seed) if not (out / "action_scale.json").exists() else json.loads((out / "action_scale.json").read_text())
    if not isinstance(scale, dict):
        raise SystemExit("action_scale.json is in the old linear format: start a new run")
    (out / "action_scale.json").write_text(json.dumps(scale))
    if not (out / "args.json").exists():
        (out / "args.json").write_text(json.dumps({**vars(args), "train_grasps": len(gs)}))
    nstep = lambda p: int(re.search(r"ppo_(\d+)_steps\.zip$", p).group(1))  # noqa: E731
    ck = sorted((p for p in glob.glob(str(out / "ckpt" / "ppo_*_steps.zip"))      # only checkpoints whose statistics were written too
                 if (out / "ckpt" / f"ppo_vecnormalize_{nstep(p)}_steps.pkl").exists()), key=nstep) if args.resume else []
    done = nstep(ck[-1]) if ck else 0
    tag = f"log_from_{done}" if done else "log"
    n = 1
    while (out / (tag if n == 1 else f"{tag}_{n}")).exists():                       # never overwrite an earlier log
        n += 1
    tag = tag if n == 1 else f"{tag}_{n}"
    share = port_shares(gs, args.engines)
    print(f"{len(gs)} training grasps over {args.engines} engines; grasps per engine {sorted(len(s) for s in share)}", flush=True)
    fns = [make_env(share[i], scale, args.seed * 1000 + i + done, args.shaping, args.gamma, args.episodes_per_grasp, 20 * i)
           for i in range(args.engines)]
    venv = VecMonitor(PoolVecEnv(fns, args.n_envs, start_method="spawn"), filename=str(out / f"monitor_{tag}"),
                      info_keywords=("is_success", "stopped", "decisions"))
    if ck:
        venv = VecNormalize.load(str(out / "ckpt" / f"ppo_vecnormalize_{done}_steps.pkl"), venv)
        model = PPO.load(ck[-1], env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs, gamma=args.gamma, gae_lambda=0.95,
                    learning_rate=args.lr, clip_range=0.2, ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0))
    (out / f"args_{tag}.json").write_text(json.dumps({**vars(args), "train_grasps": len(gs), "from_timesteps": done}))
    model.set_logger(configure(str(out / tag), ["csv", "stdout"]))
    cb = CheckpointCallback(save_freq=max(1, args.save_every // args.n_envs), save_path=str(out / "ckpt"), name_prefix="ppo", save_vecnormalize=True)
    model.learn(total_timesteps=max(0, args.total_steps - done), callback=cb, reset_num_timesteps=not ck)   # SB3 counts on from `done`
    model.save(str(out / "final.zip"))
    venv.save(str(out / "final_vecnormalize.pkl"))
    (out / "DONE.json").write_text(json.dumps({"event": "done", "timesteps": int(model.num_timesteps)}))
    venv.close()


if __name__ == "__main__":
    main()
