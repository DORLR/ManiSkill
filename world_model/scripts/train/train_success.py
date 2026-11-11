"""Offline stage of the success classifier (v4, wm/training/success.py), from scratch.

    python world_model/scripts/train/train_success.py --config world_model/configs/v4/success.json --run $Q/runs/success_v4
"""

import argparse
import json
from pathlib import Path

from wm.data.history import SuccessDataset
from wm.training import common, success
from wm.training.wm4 import latent_scale, train_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    scale = latent_scale(cfg)
    h = success.horizon_images(cfg)
    mk = lambda files: success.balanced_loader(SuccessDataset(files, scale, cfg["mask"], horizon=h), cfg)
    batches = train_batches(cfg, mk)
    resume = str(run / "last.pt") if (run / "last.pt").exists() else None
    success.train(cfg, run, batches, max_steps=args.steps, resume=resume)


if __name__ == "__main__":
    main()
