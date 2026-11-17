"""PPO baseline on the privileged state (wm/baselines/state_env.py), with Stable-Baselines3: PPO, SubprocVecEnv,
VecNormalize (observations and rewards), checkpoints with the normalisation statistics. Training grasps only (the
training split of demo320.split320: no selection, held-out or test grasp); each worker gets its own share of them.
The action scale per joint is the 99.9th percentile of the demonstrations' per-decision joint motion.
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from wm.tdmpc import demo320 as D
from wm.tdmpc import wmdata320 as WD


def train_grasps(root: str) -> list[dict]:
    gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{root}/*/*/grasp.json"))]
    return [g for g in gs if g.get("reference_ok") and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == "train"]


def action_scale(data: str, n: int = 600, seed: int = 0) -> np.ndarray:
    ents = [e for e in D.load_entries(data) if e["split"] == "train" and e["bc"]]
    trs = []
    for i in np.random.default_rng(seed).choice(len(ents), min(n, len(ents)), replace=False):
        with np.load(Path(data) / ents[i]["path"]) as z:
            tr = WD.trajectory(z, json.loads(str(z["meta"])), 0.0, 0.99)
        if tr is not None:
            trs.append(tr)
    d = np.abs(np.concatenate([t["delta"] for t in trs])).reshape(-1, WD.EXECUTE, D.JOINTS).max(1)
    return np.percentile(d, 99.9, axis=0)


def make_env(grasps, scale, seed, shaping, gamma, per_grasp):
    def f():
        from wm.baselines.state_env import StateInsertEnv
        return StateInsertEnv(grasps, scale, seed, shaping, gamma, per_grasp)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=24)
    ap.add_argument("--total-steps", type=int, default=20_000_000)
    ap.add_argument("--shaping", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=1 - 1 / 30)
    ap.add_argument("--episodes-per-grasp", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "DONE.json").exists():
        raise SystemExit(f"{out} is finished")
    if not args.resume and (out / "args.json").exists():
        raise SystemExit(f"{out} already holds a run; use --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    gs = train_grasps(args.grasps)
    scale = action_scale(args.data, seed=args.seed) if not (out / "action_scale.json").exists() else np.array(json.loads((out / "action_scale.json").read_text()))
    (out / "action_scale.json").write_text(json.dumps(scale.tolist()))
    (out / "args.json").write_text(json.dumps({**vars(args), "train_grasps": len(gs)}))
    fns = [make_env(gs[i::args.n_envs], scale, args.seed * 1000 + i, args.shaping, args.gamma, args.episodes_per_grasp) for i in range(args.n_envs)]
    venv = VecMonitor(SubprocVecEnv(fns, start_method="spawn"), info_keywords=("is_success",))
    ck = sorted(glob.glob(str(out / "ckpt" / "ppo_*_steps.zip")), key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    if args.resume and ck:
        steps = re.search(r"_(\d+)_steps", ck[-1]).group(1)
        venv = VecNormalize.load(str(out / "ckpt" / f"ppo_vecnormalize_{steps}_steps.pkl"), venv)
        model = PPO.load(ck[-1], env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs, gamma=args.gamma, gae_lambda=0.95,
                    learning_rate=args.lr, clip_range=0.2, ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    model.set_logger(configure(str(out), ["csv", "stdout"]))
    cb = CheckpointCallback(save_freq=max(1, args.save_every // args.n_envs), save_path=str(out / "ckpt"), name_prefix="ppo", save_vecnormalize=True)
    model.learn(total_timesteps=args.total_steps, callback=cb, reset_num_timesteps=not (args.resume and ck))
    model.save(str(out / "final.zip"))
    venv.save(str(out / "final_vecnormalize.pkl"))
    (out / "DONE.json").write_text(json.dumps({"event": "done", "timesteps": int(model.num_timesteps)}))
    venv.close()


if __name__ == "__main__":
    main()
