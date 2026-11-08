"""Commanded hand poses: FK of every recorded step's commanded joint targets (data/kinematics.py),
written to <out>/<relative trajectory path> (key hand_cmd). Also checks that the same FK of the
measured joints reproduces the recorded hand poses (prints the largest difference).

    python world_model/scripts/data/compute_ee.py $Q/data/wm_v1 --out $Q/data/wm_v1_ee --workers 6
"""

import argparse
import json
from pathlib import Path

from wm.data import index
from wm.data.kinematics import compute_commanded_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="smoke tests only: first N trajectories")
    args = ap.parse_args()
    entries = index.load(args.data)
    if args.limit:
        entries = entries[:args.limit]
    res = compute_commanded_hands(entries, Path(args.data), Path(args.out), args.workers)
    print(json.dumps(res))
    (Path(args.out) / "summary.json").write_text(json.dumps({**res, "trajectories": len(entries)}))


if __name__ == "__main__":
    main()
