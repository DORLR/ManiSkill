"""Train the residual policy on top of the frozen latent FM inside the world model (wm.rl).

    python world_model/scripts/train/train_residual.py --config world_model/configs/v3/residual.json \
        --run $Q/runs/residual_v3g
"""

import argparse
import json
from pathlib import Path

import torch

from wm.data.kinematics import build_fk
from wm.models.vae import load_vae
from wm.rl import residual
from wm.rl.imagine import ImaginedInsertion, StartPool
from wm.training import common
from wm.training.adapter import load_adapter
from wm.training.wm import load_world_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--iterations", type=int, help="smoke tests only")
    ap.add_argument("--max-files", type=int, help="smoke tests only: start states from this many trajectories")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    if args.iterations:
        cfg["iterations"] = args.iterations
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    common.seed_all(cfg["seed"])
    dev = torch.device("cuda")
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    fk, robot = build_fk()
    log = common.Logger(run)
    pool = StartPool(cfg["data"], cfg["latents"], "train", max_files=args.max_files or cfg.get("max_files"), seed=cfg["seed"])
    log(event="start", start_states=len(pool.items), trajectories=len(pool.files))
    env = ImaginedInsertion(load_vae(cfg["vae"], dev).requires_grad_(False), load_world_model(cfg["wm"], dev).requires_grad_(False),
                            load_adapter(cfg["adapter"], dev).requires_grad_(False), fk, scale, pool, dev,
                            n_envs=cfg["n_envs"], horizon=cfg["horizon"], sampling_steps=cfg["sampling_steps"],
                            success_mm=cfg["success_mm"])
    base = residual.FrozenFM(cfg["policy"], dev)
    residual.train(env, base, cfg, run, log, dev)
    robot.env.close()


if __name__ == "__main__":
    main()
