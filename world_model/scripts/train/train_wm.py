"""Stage 2: train the CDiT world model from scratch on the latents (wm.training.wm).

    python world_model/scripts/train/train_wm.py --config world_model/configs/v3/wm_s.json --run $Q/runs/wm_v3_s
"""

import argparse
import json
from pathlib import Path

from wm.training import common, wm


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
    wm.train(cfg, run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
