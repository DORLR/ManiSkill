"""Encode the pictures of the seated attempts of the branch set with the world model's ResNet (wm/tdmpc/policy_resnet.py)
-> one small file per attempt (tokens of every picture, joints, executed targets). Resumable.

    python world_model/scripts/data/encode_branch_tokens.py --ckpt <run>/last.pt --data <wm_v1b> --out <wm_v1b_tok> --splits train,dev,ood_val
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from wm.tdmpc import data
from wm.tdmpc.policy_resnet import frame_tokens, load_wm
from wm.training.common import limit_gpu_memory


class Branches(torch.utils.data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        src, dst = self.jobs[i]
        with np.load(src) as z:
            blob, off = z["rgb_jpeg"], z["rgb_offsets"]
            rgb = np.stack([data.decode_rgb(blob, off, k) for k in range(len(off) - 1)])
            depth = np.minimum(z["depth"], 32767).astype(np.int16)
            small = {"qpos": z["qpos"][:, :7].astype(np.float32), "action": z["action"].astype(np.float32), "img_step": z["img_step"]}
        return {"rgb": torch.from_numpy(rgb), "depth": torch.from_numpy(depth), "small": small, "dst": dst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,dev,ood_val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-branches", type=int)
    args = ap.parse_args()
    limit_gpu_memory()
    wm = load_wm(args.ckpt)
    jobs = []
    for split in args.splits.split(","):
        bs = [b for b in data.list_branches(args.data, split) if b["seated_fk"]]
        if args.max_branches:
            bs = bs[:args.max_branches]
        jobs += [(b["path"], f"{args.out}/{split}/{b['rel']}.npz") for b in bs]
    jobs = [j for j in jobs if not Path(j[1]).exists()]
    print(len(jobs), "attempts to encode", flush=True)
    dl = torch.utils.data.DataLoader(Branches(jobs), batch_size=1, num_workers=args.workers, collate_fn=lambda x: x[0])
    for n, b in enumerate(dl):
        rgb, depth = b["rgb"].cuda().permute(0, 3, 1, 2), b["depth"].cuda()
        tok = torch.cat([frame_tokens(wm, rgb[i:i + 48], depth[i:i + 48]) for i in range(0, len(rgb), 48)]).cpu().numpy().astype(np.float16)
        dst = Path(b["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez(tmp, tok=tok, **{k: np.array(v) for k, v in b["small"].items()})
        os.replace(tmp, dst)
        if (n + 1) % 500 == 0:
            print(n + 1, len(jobs), flush=True)
    print("done", len(jobs), flush=True)


if __name__ == "__main__":
    main()
