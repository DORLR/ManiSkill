"""Offline stage of the success classifier (v4, wm/training/success.py), from scratch.

    python world_model/scripts/train/train_success.py --config world_model/configs/v4/success.json --run $Q/runs/success_v4
"""

import argparse
import json
from pathlib import Path

from wm.data.history import SuccessDataset
from wm.training import common, success
from wm.training.wm4 import offline_files


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
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    ds = SuccessDataset(offline_files(cfg, "train"), scale, cfg["mask"])
    resume = str(run / "last.pt") if (run / "last.pt").exists() else None
    success.train(cfg, run, common.cycle(success.balanced_loader(ds, cfg)), max_steps=args.steps, resume=resume)


if __name__ == "__main__":
    main()
