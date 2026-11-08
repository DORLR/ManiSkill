"""Encode every recorded image with the trained autoencoder (input of the world model and policy).

    python world_model/scripts/train/encode_latents.py --vae $Q/runs/vae_v3/best.pt \
        --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --out $Q/data/wm_v3_latents
"""

import argparse
from pathlib import Path

import torch

from wm.data import index
from wm.models.vae import load_vae
from wm.training import common, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ee", required=True, help="commanded hand poses (scripts/data/compute_ee.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="+", help="only these trajectory kinds (smoke tests)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    common.limit_gpu_memory()
    model = load_vae(args.vae)
    entries = [e for e in index.load(args.data) if not args.kinds or e["kind"] in args.kinds]
    scale = encode.run(model, entries, Path(args.data), Path(args.ee), Path(args.out), torch.device("cuda"), args.workers)
    print("latent scale", scale)


if __name__ == "__main__":
    main()
