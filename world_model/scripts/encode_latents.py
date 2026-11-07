"""Encode every recorded image with the trained VAE (input of stage 2).

    python world_model/scripts/encode_latents.py --vae $Q/runs/vae_v1/best.pt \
        --data $Q/data/wm_v1 --out $Q/data/wm_v1_latents
"""

import argparse
from pathlib import Path

import torch

from wm.data import index
from wm.models.vae import TokenVAE
from wm.training import encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.vae, map_location="cpu")
    model = TokenVAE(**ckpt["cfg"]["model"])
    model.load_state_dict(ckpt["model"])
    scale = encode.run(model, index.load(args.data), Path(args.data), Path(args.out), torch.device("cuda"))
    print("latent scale", scale)


if __name__ == "__main__":
    main()
