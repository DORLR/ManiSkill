"""Collect the world-model dataset (resumable).

    python world_model/scripts/collect.py --config world_model/configs/collect_v1.json \
        --out $Q/data/wm_v1 --workers 16

Audit mode (no images, a few shapes, to check the expert before collecting):
    python world_model/scripts/collect.py --config ... --out $Q/data/wm_audit --no-images \
        --shapes A B C D G00 G05 G50 G100 --workers 12
"""

import argparse
from dataclasses import replace
from pathlib import Path

from wm.collect.plan import CollectConfig
from wm.collect.runner import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shapes", nargs="+", help="restrict to these ports/shapes")
    ap.add_argument("--max-grasps", type=int)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--max-gb", type=float, default=100.0)
    args = ap.parse_args()
    cfg = CollectConfig.load(args.config)
    if args.no_images:
        cfg = replace(cfg, images=False)
    run(cfg, Path(args.out), args.workers, args.shapes, args.max_grasps, max_gb=args.max_gb)


if __name__ == "__main__":
    main()
