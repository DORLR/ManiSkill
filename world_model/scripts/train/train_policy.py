"""Train an FM policy: latent (configs/v3/policy.json) or latent-free baseline (configs/v3/baseline.json).

    python world_model/scripts/train/train_policy.py --config world_model/configs/v3/policy.json --run $Q/runs/policy_v3
"""

import argparse
import json
from pathlib import Path

from wm.training import common, policy


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
    policy.train(cfg, run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
