"""Encode every recorded image with the trained VAE (input of stage 2).

    python world_model/scripts/encode_latents.py --vae $Q/runs/vae_v1/best.pt \
        --data $Q/data/wm_v1 --out $Q/data/wm_v1_latents
"""

import argparse
from pathlib import Path

import torch

from wm.data import index
from wm.models.vae import load_vae
from wm.training import encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="+", help="only these trajectory kinds (e.g. contact, for the probes)")
    args = ap.parse_args()
    model = load_vae(args.vae)
    entries = [e for e in index.load(args.data) if not args.kinds or e["kind"] in args.kinds]
    scale = encode.run(model, entries, Path(args.data), Path(args.out), torch.device("cuda"))
    print("latent scale", scale)


if __name__ == "__main__":
    main()
