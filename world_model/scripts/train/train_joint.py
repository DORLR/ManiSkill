"""Train the v6 encoder and world model together (wm/training/joint.py).

    python world_model/scripts/train/train_joint.py --config world_model/configs/v6/joint.json --run $Q/runs/joint_v6
"""

import argparse
import json
from pathlib import Path

from wm.training import common, joint


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
    joint.train(cfg, run, max_steps=args.steps)


if __name__ == "__main__":
    main()
