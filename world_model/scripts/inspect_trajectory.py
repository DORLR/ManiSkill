"""Print the arrays of one trajectory file and save an image strip (RGB above depth).

    python world_model/scripts/inspect_trajectory.py <file.npz> [--png out.png] [--frames 6]
"""

import argparse

import numpy as np
from PIL import Image

from wm.data import schema, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--png")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    arrays, meta = store.load(args.path)
    print({k: v for k, v in meta.items() if k not in ("H_star",)})
    for k, v in arrays.items():
        print(f"  {k:12s} {str(v.shape):22s} {v.dtype}  min {v.min():.4g} max {v.max():.4g}")
    phases = [schema.PHASES[i] for i in np.unique(arrays["phase"])]
    print("  phases:", phases)
    if args.png and "rgb" in arrays:
        idx = np.linspace(0, len(arrays["rgb"]) - 1, args.frames).astype(int)
        rgb = np.concatenate([arrays["rgb"][i] for i in idx], axis=1)
        d = np.concatenate([arrays["depth"][i] for i in idx], axis=1).astype(np.float32) * schema.DEPTH_UNIT_M
        valid = d > 0
        lo, hi = (np.percentile(d[valid], [1, 99]) if valid.any() else (0, 1))
        dn = np.where(valid, 1 - np.clip((d - lo) / (hi - lo + 1e-9), 0, 1), 0)
        depth_img = (np.stack([dn] * 3, -1) * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([rgb, depth_img], axis=0)).save(args.png)
        print("saved", args.png, "frames", idx.tolist())


if __name__ == "__main__":
    main()
