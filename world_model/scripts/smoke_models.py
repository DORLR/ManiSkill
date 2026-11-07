"""End-to-end smoke test of the model code on a few recorded trajectories. It does not train.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/smoke_models.py --data $Q/data/wm_v1 --out $Q/runs/smoke

VAE: a few steps, evaluation, checkpoint -> encode latents -> CDiT: a few steps and one
sampled prediction decoded by the VAE. Prints losses, timings and peak GPU memory.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch

from wm.data import index
from wm.models.vae import TokenVAE
from wm.training import common, encode, vae, wm

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="trajectories (from completed grasps)")
    args = ap.parse_args()
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    done = sorted(p.parent for p in Path(args.data).rglob("grasp.json"))
    paths = [str(p) for g in done for p in sorted(g.glob("t*.npz"))][: args.n]
    entries = [index._entry(p) for p in paths]
    for i, e in enumerate(entries):        # smoke only: alternate the two splits
        e["split"] = "train" if i % 2 == 0 else "ood_val"
    print(f"{len(entries)} trajectories, {sum(e['n_images'] for e in entries)} images, "
          f"port axes ok {sum(e['R_P'] is not None for e in entries)}")

    t0 = time.time()
    cfg_v = common.load_config(CONFIGS / "vae_v1.json", batch_size=4, workers=0, log_every=1, eval_every=3,
                               eval_batches=1, data=args.data)
    best = vae.train(cfg_v, out / "vae", entries, max_steps=3)
    print(f"VAE ok ({time.time() - t0:.0f}s), peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    ckpt = torch.load(best, map_location="cpu")
    model = TokenVAE(**ckpt["cfg"]["model"])
    model.load_state_dict(ckpt["model"])
    scale = encode.run(model, entries, Path(args.data), out / "latents", torch.device("cuda"))
    print(f"latents ok, scale {scale:.3f}")

    t0 = time.time()
    cfg_w = common.load_config(CONFIGS / "wm_v1.json", batch_size=8, workers=0, log_every=1, eval_every=3,
                               eval_batches=1, eval_sampling_steps=10, data=args.data,
                               latents=str(out / "latents"), vae_checkpoint=str(best))
    wm.train(cfg_w, out / "wm", entries, max_steps=3)
    print(f"CDiT ok ({time.time() - t0:.0f}s), peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
