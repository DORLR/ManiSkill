"""Train the crop autoencoder (v4, wm/training/crop_ae.py).

    python world_model/scripts/train/train_crop_ae.py --config world_model/configs/v4/crop_ae.json --run $Q/runs/crop_ae_v4
"""

import argparse
import json
from pathlib import Path

from wm.training import common, crop_ae


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
    crop_ae.train(cfg, run, max_steps=args.steps)


if __name__ == "__main__":
    main()
