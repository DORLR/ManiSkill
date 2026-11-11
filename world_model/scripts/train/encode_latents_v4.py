"""Encode every recorded image with the frozen crop autoencoder (v4, wm/training/encode4.py).

    python world_model/scripts/train/encode_latents_v4.py --ae $Q/runs/crop_ae_v4b/best.pt --data $Q/data/wm_v1 \
        --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v4_latents
"""

import argparse
from pathlib import Path

import torch

from wm.data import index
from wm.models.crop_ae import load_crop_ae
from wm.training import encode4
from wm.training.common import limit_gpu_memory


def main():
    ap = argparse.ArgumentParser()
    for a in ("--ae", "--data", "--ee", "--crops", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, help="smoke tests only: this many trajectories per split")
    args = ap.parse_args()
    limit_gpu_memory()
    dev = torch.device("cuda")
    model = load_crop_ae(args.ae, dev).requires_grad_(False)
    entries = index.load(args.data)
    if args.limit:
        entries = [e for sp in ("train", "ood_val") for e in [x for x in entries if x["split"] == sp and x["n_images"]][:args.limit]]
    scale = encode4.run(model, entries, Path(args.data), Path(args.ee), Path(args.crops), Path(args.out),
                        dev, args.workers, args.batch)
    print("latent scale", scale)


if __name__ == "__main__":
    main()
