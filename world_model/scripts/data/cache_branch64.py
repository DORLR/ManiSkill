"""64 x 64 copies of the branch set's pictures (10 x 10 pixel means of the 640 x 640 ones) with the decision-level
arrays, for the unchanged TD-MPC2 (its encoder takes 64 x 64 pictures). Resumable.

    python world_model/scripts/data/cache_branch64.py --data /mnt/storage3/qingxin/data/wm_v1b \
        --out /mnt/storage3/qingxin/data/wm_v1b_c64 --workers 8
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np

from wm.tdmpc.data import decision_arrays, decode_rgb, list_branches
from wm.tdmpc.original import SIZE


def work(job):
    src, dst = job
    dst = Path(dst)
    if dst.exists():
        return "kept"
    cv2.setNumThreads(1)
    with np.load(src) as z:
        a = decision_arrays(z)
        if a is None:
            return "irregular"
        blob, off = z["rgb_jpeg"], z["rgb_offsets"]
    rgb = np.stack([cv2.resize(decode_rgb(blob, off, i), (SIZE, SIZE), interpolation=cv2.INTER_AREA) for i in range(2 * a["n"] + 1)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, rgb=rgb, delta=a["delta"], reward=a["reward"], terminated=a["terminated"], q_img=a["q_img"], face_k=a["face_k"])
    os.replace(tmp, dst)
    return "written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,dev,ood_val")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(b["path"], f"{args.out}/{b['rel']}.npz") for s in args.splits.split(",") for b in list_branches(args.data, s)]
    count = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=8)):
            count[r] = count.get(r, 0) + 1
            if (k + 1) % 1000 == 0:
                print(k + 1, len(jobs), count, flush=True)
    print("done", len(jobs), count, flush=True)


if __name__ == "__main__":
    main()
