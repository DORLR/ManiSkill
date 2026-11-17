"""Features for the two "goal" checks (2026-11-17, asked for by the user after the fourth review).

One pass of the finished model over recorded branches. Per branch, five moments (pictures every 4 control steps):
  face-2   the decision 0.8 s before the pins reach the board face
  face+0   the decision at which they reach it
  half     the picture whose FK depth left is closest to 6 mm inside [4, 8) mm
  near     the picture whose FK depth left is closest to 2.5 mm inside [1.25, 4) mm
  end      the picture after the last decision
Saved per moment: the latent z (3 pictures + joints), the feature of the last picture alone (no joints), a 40x40 copy
of its depth picture [mm], V, the FK depth left and the simulator's depth left / lateral error (judging only).
Shards of 1500 branches, so an interrupted run resumes. The analysis is analyse_goal_reuse.py.

    python world_model/scripts/analysis/check_goal_reuse.py --ckpt <run>/last.pt --data <wm_v1b> --out <dir>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from wm.tdmpc import data
from wm.tdmpc.model import LatentWM, make_cfg
from wm.training.common import limit_gpu_memory

POINTS = ("face-2", "face+0", "half", "near", "end")
BANDS = ((2, 4.0, 8.0, 6.0), (3, 1.25, 4.0, 2.5))            # slot, from, to, wanted [mm of FK depth left]
SHARD, THUMB = 1500, 40


class Frames(torch.utils.data.Dataset):
    def __init__(self, branches):
        self.b = branches

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        with np.load(self.b[i]["path"]) as z:
            a = data.decision_arrays(z)
            if a is None:
                return None
            n, f, L = a["n"], a["face_k"], a["meta"]["pin_length_m"]
            step = z["img_step"][:2 * n + 1]
            left = (L - z["depth_fk_m"][step]) * 1000.0
            rel = z["eval_rel"][step]
            pi = np.full(len(POINTS), -1)
            if 2 <= f < n + 2:
                pi[0] = 2 * (f - 2)
            if 0 <= f < n:
                pi[1] = 2 * f
            for slot, lo, hi, want in BANDS:
                c = np.nonzero((left >= lo) & (left < hi))[0]
                c = c[c >= 2]
                if len(c):
                    pi[slot] = c[np.argmin(np.abs(left[c] - want))]
            pi[4] = 2 * n
            ok = pi >= 0
            p = np.where(ok, pi, 2 * n)
            pics = data._Pictures(z)
            rgb, depth, q, thumb = [], [], [], []
            for j in p:
                idx = np.maximum(np.arange(j - 2, j + 1), 0)
                r_, d_ = pics.get(idx)
                rgb.append(r_); depth.append(d_); q.append(a["q_img"][idx])
                mm = np.minimum(d_[2].astype(np.float32) / 10.0, 400.0)
                thumb.append(cv2.resize(mm, (THUMB, THUMB), interpolation=cv2.INTER_AREA))
            truth = np.stack([-rel[p, 0] * 1000.0, np.hypot(rel[p, 1], rel[p, 2]) * 1000.0], 1).astype(np.float32)
        return {"rgb": torch.from_numpy(np.stack(rgb)), "depth": torch.from_numpy(np.stack(depth)), "q": torch.from_numpy(np.stack(q)),
                "ok": ok.copy(), "left_fk": left[p].astype(np.float32), "truth": truth, "thumb": np.stack(thumb).astype(np.float16), "i": i}


@torch.no_grad()
def run(model, branches, workers, device="cuda"):
    dl = torch.utils.data.DataLoader(Frames(branches), batch_size=1, num_workers=workers, collate_fn=lambda x: x[0])
    out = {k: [] for k in ("rel", "ok", "left_fk", "truth", "thumb", "z", "p", "v")}
    for b in dl:
        if b is None:
            continue
        feat = model.frames(b["rgb"].to(device).permute(0, 1, 4, 2, 3), b["depth"].to(device), chunk=24)
        z = model.fuse(feat, b["q"].to(device))
        out["z"].append(z.float().cpu().numpy().astype(np.float16))
        out["p"].append(feat[:, 2].float().cpu().numpy().astype(np.float16))
        out["v"].append(model.scalar(model.V(z))[..., 0].mean(0).float().cpu().numpy())
        out["rel"].append(branches[int(b["i"])]["rel"])
        for k in ("ok", "left_fk", "truth", "thumb"):
            out[k].append(np.array(b[k]))                      # copies: views would keep the workers' shared memory open
    return {k: np.stack(v) if k != "rel" else np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-branches", type=int)
    args = ap.parse_args()
    limit_gpu_memory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = LatentWM(make_cfg(**ck["cfg"]), ck["stats"]).cuda().eval()
    model.load_state_dict(ck["model"])
    for split in ("ood_val", "dev", "train"):
        bs = data.list_branches(args.data, split)
        if args.max_branches:
            bs = bs[::max(1, len(bs) // args.max_branches)]
        for s in range(0, len(bs), SHARD):
            f = out / f"points_{split}_{s // SHARD:03d}.npz"
            if f.exists():
                continue
            r = run(model, bs[s:s + SHARD], args.workers)
            tmp = out / f"tmp_{split}_{s // SHARD:03d}.npz"
            np.savez(tmp, **r)
            tmp.rename(f)
            print(split, s // SHARD, len(r["rel"]), "branches", flush=True)
    print("features done", flush=True)


if __name__ == "__main__":
    main()
