"""Does the crop autoencoder (trained on demonstrations only) read the exploration frames as well?

    python world_model/scripts/analysis/check_ae_on_exploration.py --ae $Q/runs/crop_ae_v4b/best.pt \
        --demos $Q/data/wm_v1 --demo-crops $Q/data/wm_v1_crops --explore $Q/data/wm_v1x \
        --explore-crops $Q/data/wm_v1x_crops --out $Q/runs/checks_v4/explore/ae_readout.json
Held-out shapes (ood_val) of both datasets, the same number of frames each. Per distance still to move
(label: FK of the grasp's insertion) and for frames within 3 mm of the hole axis -- where insert / jam is
decided -- the error of the solved remaining motion: across the axis (mm), reference grid (mm), angle (deg).
Exit code 1 if, in a bin with >= 200 frames of each, the exploration error across the axis is more than twice
the demonstrations' and above 0.3 mm: the world model would then be trained on poorly encoded frames.
"""
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data import index
from wm.data.datasets import CropFrameDataset
from wm.geometry import insertion_motion, plug_region_points, pose_loss_terms
from wm.models.crop_ae import load_crop_ae
from wm.training.common import limit_gpu_memory, to_device

ap = argparse.ArgumentParser()
for a in ("--ae", "--demos", "--demo-crops", "--explore", "--explore-crops", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--batches", type=int, default=400)
args = ap.parse_args()
limit_gpu_memory()
dev = torch.device("cuda")
model = load_crop_ae(args.ae, dev)
ref = plug_region_points(dev)
BINS = [(0, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "13-17 mm (pins at the board face)"), (17, 1e9, ">17 mm")]


@torch.no_grad()
def run(data, crops):
    ds = CropFrameDataset(index.load(data), "ood_val", data, crops)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=8, generator=torch.Generator().manual_seed(7))
    rows = []
    for i, b in enumerate(loader):
        if i >= args.batches:
            break
        b = to_device(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        R_star, t_star = insertion_motion(b["hand"].float(), b["h_star"].float(), b["R_P"].float())
        g, a = pose_loss_terms(out["R"], out["t"], R_star, t_star, ref)
        lat = (out["t"].float()[:, 1:] - t_star[:, 1:]).norm(dim=-1) * 1000
        rows.append(torch.stack([t_star.norm(dim=-1) * 1000, t_star[:, 1:].norm(dim=-1) * 1000, lat, g, a], 1).cpu().numpy())
    return np.concatenate(rows), len(ds)


rep, fail = {}, []
res = {}
for name, data, crops in (("demonstrations", args.demos, args.demo_crops), ("exploration", args.explore, args.explore_crops)):
    R, n = run(data, crops)
    res[name] = R
    rep[name] = {"frames_available": n, "frames_used": int(len(R))}
    for lo, hi, lab in BINS:
        for sub, m in (("all offsets", np.ones(len(R), bool)), ("within 3 mm of the axis", R[:, 1] <= 3.0)):
            x = R[(R[:, 0] >= lo) & (R[:, 0] < hi) & m]
            rep[name][f"{lab} | {sub}"] = {"frames": int(len(x))} if not len(x) else {
                "frames": int(len(x)), "across_axis_err_mm": round(float(np.median(x[:, 2])), 3),
                "across_axis_err_mm_p90": round(float(np.percentile(x[:, 2], 90)), 3),
                "grid_err_mm": round(float(np.median(x[:, 3])), 3), "angle_err_deg": round(float(np.median(x[:, 4])), 3)}
for lo, hi, lab in BINS:
    k = f"{lab} | within 3 mm of the axis"
    d, e = rep["demonstrations"][k], rep["exploration"][k]
    if d["frames"] >= 200 and e["frames"] >= 200 and e["across_axis_err_mm"] > max(2 * d["across_axis_err_mm"], 0.3):
        fail.append([lab, d["across_axis_err_mm"], e["across_axis_err_mm"]])
rep["gate"], rep["failed"] = ("FAIL" if fail else "PASS"), fail
json.dump(rep, open(args.out, "w"), indent=1)
for name in ("demonstrations", "exploration"):
    print(name)
    for k, v in rep[name].items():
        if isinstance(v, dict) and "within 3 mm" in k:
            print("   %-60s %s" % (k, v))
print("GATE", rep["gate"], fail)
sys.exit(1 if fail else 0)
