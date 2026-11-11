"""Train the vision-conditioned joint adapter (wm.training.adapter_vision).

    python world_model/scripts/train/train_adapter_vision.py --config world_model/configs/v4/adapter_vision.json \
        --run $Q/runs/adapter_vision_v4e
"""

import argparse
import json
from pathlib import Path

from wm.training import adapter_vision, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int, help="smoke tests only")
    args = ap.parse_args()
    common.limit_gpu_memory()
    cfg = common.load_config(args.config)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    adapter_vision.train(cfg, run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
