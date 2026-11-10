"""Run the trained UNet on every recorded image and place the two crops (wm/perception/crops.py).

    python world_model/scripts/data/place_crops.py --data $Q/data/wm_v1 --unet $Q/runs/unet_v4/best.pt \
        --out $Q/data/wm_v1_crops --workers 12
Writes <out>/<relative trajectory path>.npz with plug_uv0, hole_uv0 (N, 2) int32 crop origins and,
for bookkeeping only, how each centre was found; <out>/summary.json counts the cases.
DataLoader workers decode the images; the GPU runs the UNet; placement is sequential per trajectory.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wm.data import index, store
from wm.perception.crops import CropPlacer
from wm.perception.unet import load_unet

HOW = {"pins": 0, "plug": 1, "memory": 2, "opening": 0, "memory_3d": 1, "memory_2d": 2}


class Trajs(Dataset):
    def __init__(self, entries, data_root, out):
        self.items = [(e["path"], Path(out) / Path(e["path"]).relative_to(data_root)) for e in entries if e["n_images"]]
        self.items = [(p, d) for p, d in self.items if not d.exists()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, dst = self.items[i]
        arrays, _ = store.load(src)
        return {"dst": str(dst), "rgb": arrays["rgb"], "depth": arrays["depth"], "K": arrays["K"],
                "cam_T_world": arrays["cam_T_world"]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, help="smoke tests only: this many trajectories per split")
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    dev = torch.device("cuda")
    net = load_unet(args.unet, dev)
    entries = index.load(args.data)
    if args.limit:
        entries = [e for sp in ("train", "ood_val") for e in [x for x in entries if x["split"] == sp and x["n_images"]][:args.limit]]
    ds = Trajs(entries, args.data, args.out)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers, prefetch_factor=2, collate_fn=lambda x: x)
    counts = Counter()
    for k, t in enumerate(loader):
        rgb = torch.from_numpy(t["rgb"])
        cls = []
        for a in range(0, len(rgb), args.batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls.append(net(rgb[a:a + args.batch].to(dev)).argmax(1).to(torch.uint8).cpu().numpy())
        cls = np.concatenate(cls)
        placer = CropPlacer()
        out = {"plug_uv0": [], "hole_uv0": [], "how_plug": [], "how_hole": []}
        for j in range(len(cls)):
            r = placer(cls[j], t["depth"][j], t["K"], t["cam_T_world"][j])
            for key in out:
                out[key].append(HOW[r[key]] if key.startswith("how") else r[key])
            counts[f"plug_{r['how_plug']}"] += 1
            counts[f"hole_{r['how_hole']}"] += 1
        dst = Path(t["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp.npz")
        np.savez(tmp, plug_uv0=np.asarray(out["plug_uv0"], np.int32), hole_uv0=np.asarray(out["hole_uv0"], np.int32),
                 how_plug=np.asarray(out["how_plug"], np.uint8), how_hole=np.asarray(out["how_hole"], np.uint8))
        tmp.rename(dst)
        if k % 500 == 0:
            print(k, len(ds), dict(counts), flush=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(dict(counts), indent=1))
    print(dict(counts))


if __name__ == "__main__":
    main()
