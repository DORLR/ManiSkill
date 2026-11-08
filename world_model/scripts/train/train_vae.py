"""Stage 1: train the autoencoder from scratch (wm.training.vae).

    python world_model/scripts/train/train_vae.py --config world_model/configs/v3/vae.json --run $Q/runs/vae_v3
"""

import argparse
import json
from pathlib import Path

from wm.training import common, vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    vae.train(cfg, run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
