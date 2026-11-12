"""How different are the latent tokens of one frame?

    python world_model/scripts/analysis/check_latent_tokens.py --ae $Q/runs/crop_ae_v5/best.pt \
        --data $Q/data/wm_v1 $Q/data/wm_v1x --crops $Q/data/wm_v1_crops $Q/data/wm_v1x_crops --out $Q/runs/checks_v5/latent_tokens_v5.json
Held-out frames (ood_val) of every source are encoded and the latent's variance is split into
    constant offsets   each token's mean over the frames (says nothing about a frame)
    shared             what all tokens of a frame have in common
    token-specific     the rest: what a token says that the other tokens of the same frame do not
and the number of principal components that hold 90 % / 99 % of the frame-to-frame variation is counted.
crop_ae_v4e (64 learned queries): token-specific 0.4-0.5 %, 99 % in 15 components -- the tokens were copies.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data import index
from wm.data.datasets import CropFrameDataset
from wm.models.crop_ae import load_crop_ae
from wm.training.common import limit_gpu_memory, to_device

ap = argparse.ArgumentParser()
ap.add_argument("--ae", required=True)
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--crops", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batches", type=int, default=60)
args = ap.parse_args()
limit_gpu_memory()
dev = torch.device("cuda")
model = load_crop_ae(args.ae, dev)
rep = {"autoencoder": args.ae}
for data, crops in zip(args.data, args.crops):
    ds = CropFrameDataset(index.load(data), "ood_val", data, crops)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=8, generator=torch.Generator().manual_seed(7))
    zs = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(model.encode(to_device(b, dev))[0].float().cpu())
    z = torch.cat(zs).double()                                           # (N, tokens, C)
    N, T, C = z.shape
    grand = z.mean((0, 1))
    m = z.mean(0)                                                        # each token's mean over the frames
    r = z - m
    s = r.mean(1)                                                        # shared by the tokens of a frame
    parts = {"constant_offsets": N * ((m - grand) ** 2).sum(), "shared_by_all_tokens": T * (s ** 2).sum(),
             "token_specific": ((r - s[:, None]) ** 2).sum()}
    total = sum(parts.values())
    ev = torch.linalg.svdvals(r.reshape(N, T * C)) ** 2
    cum = (ev / ev.sum()).cumsum(0)
    name = data.rstrip("/").split("/")[-1]
    rep[name] = {"frames": N, "tokens": T, "channels": C,
                 **{f"{k}_pct": round(float(100 * v / total), 2) for k, v in parts.items()},
                 "token_specific_pct_of_frame_to_frame_variation": round(float(100 * parts["token_specific"] / (parts["token_specific"] + parts["shared_by_all_tokens"])), 2),
                 "components_for_90pct": int((cum < 0.90).sum()) + 1, "components_for_99pct": int((cum < 0.99).sum()) + 1}
    print(name, rep[name], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
