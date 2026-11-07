"""Stage 1: train the token VAE.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/train_vae.py --config world_model/configs/vae_v1.json \
        --run $Q/runs/vae_v1
"""

import argparse
import json
from pathlib import Path

from wm.training import common, vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    vae.train(cfg, run)


if __name__ == "__main__":
    main()
