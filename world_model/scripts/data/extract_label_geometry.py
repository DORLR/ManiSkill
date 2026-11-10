"""Per grasp: goal pose + port geometry for the segmentation labels (wm/perception/labels.py).

    python world_model/scripts/data/extract_label_geometry.py --data $Q/data/wm_v1 \
        --out $Q/data/wm_v1_labelgeom --workers 16
Writes <out>/<port>/<grasp>.json and <out>/summary.json (rigidity check of every grasp).
"""

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np

from wm.data import index, store
from wm.perception.labels import extract_grasp_geometry


def work(args):
    port, grasps, out = args
    from wm.env.robot import Robot
    robot = Robot(port)
    res = []
    for gkey, path in grasps:
        dst = Path(out) / port / f"{gkey}.json"
        if dst.exists():
            res.append(json.loads(dst.read_text()))
            continue
        arrays, meta = store.load(path, decode_rgb=False)
        g = extract_grasp_geometry(robot, meta, check=arrays)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(g))
        res.append(g)
    robot.env.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    by_port = defaultdict(dict)
    for e in index.load(args.data):
        if e["n_images"]:
            by_port[e["port"]].setdefault(e["grasp"], e["path"])        # any trajectory of the grasp
    jobs = []
    for port, g in by_port.items():
        items = sorted(g.items())
        for k in range(0, len(items), 8):                                 # 8 grasps per job (one env)
            jobs.append((port, items[k:k + 8], args.out))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        geoms = [g for r in pool.imap_unordered(work, jobs) for g in r]
    spread = np.array([g["rigid_spread_mm"] for g in geoms])
    summary = {"grasps": len(geoms), "rigid_spread_mm_max": float(spread.max()),
               "rigid_spread_mm_p99": float(np.percentile(spread, 99)),
               "bad": [g["grasp"] for g in geoms if g["rigid_spread_mm"] > 0.05]}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)[:500])


if __name__ == "__main__":
    main()
