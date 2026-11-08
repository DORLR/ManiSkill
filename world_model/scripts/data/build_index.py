"""Index a collected dataset (splits + per-trajectory port axes).

    python world_model/scripts/data/build_index.py $Q/data/wm_v1 --workers 16
"""

import argparse
from collections import Counter

from wm.data import index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    entries = index.build(args.root, args.workers)
    print(len(entries), "trajectories;", dict(Counter(e["split"] for e in entries)),
          "; port axes missing:", sum(e["R_P"] is None for e in entries))


if __name__ == "__main__":
    main()
