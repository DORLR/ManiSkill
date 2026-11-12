"""Is the sub-millimetre tip offset in the pixels of the two crops at all (320 x 320 image, 128 x 128 crops)?

    python world_model/scripts/analysis/probe_pixels.py --data $Q/data/wm_v1x --crops $Q/data/wm_v1x_crops \
        --geometry $Q/data/wm_v1_labelgeom --out $Q/runs/checks_v5/probe_pixels.json
Step 0 (probe_before_readout.py) found that crop_ae_v4e's tokens before the read-out give the pin tip across the hole
axis to 0.30 mm when it is within 1 mm (blind 0.45 mm), hardly better than the latent (0.36). That leaves two
explanations: the encoder never learned the detail, or the image does not hold it at this resolution. This is the
upper bound: a network of its own (ResNet-18, no autoencoder, no latent) is trained on exactly what the encoder gets
-- both crops' RGB, depth and depth validity, every pixel's image position, the camera geometry -- to give that
offset directly (label: simulator truth; a diagnosis, nothing here is used by any model). Scored on held-out shapes
in the bins of check_ae_tips.py. If it reads the offset much better than 0.30 mm, the pixels hold it and the encoder
/ its losses are the limit; if not, the resolution (or what the wrist camera can see) is.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from wm.data import index
from wm.data.datasets import CropFrameDataset
from wm.geometry import apply_rigid, insertion_motion
from wm.models.crop_encoder import CROP, cut
from wm.models.encoder import DEPTH_SCALE_M, DEPTH_UNIT_M, geometry_features
from wm.perception.labels import load_geometry, pose7_to_T
from wm.training.common import limit_gpu_memory, to_device

ap = argparse.ArgumentParser()
for a in ("--data", "--crops", "--geometry", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--eval-batches", type=int, default=150)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")


class TipFrames(CropFrameDataset):
    _geo = {}

    def __getitem__(self, k):
        out = super().__getitem__(k)
        i, j = self.items[k]
        e = self.entries[i]
        if e["path"] not in self._geo:
            if len(self._geo) > 2000:
                self._geo.clear()
            g = load_geometry(args.geometry, args.data, e["path"])
            with np.load(e["path"]) as f:
                self._geo[e["path"]] = (np.asarray(g["goal_world"]), 2 * g["pin_half_length_m"], g["pins"][0]["center_yz_m"],
                                        f["eval_rel"][f["img_step"]], f["hand"][f["img_step"]])
        goal, L2, pin, rel, hand = self._geo[e["path"]]
        ch = goal @ pose7_to_T(rel[j])
        pw = ch[:3, :3] @ np.array([L2, pin[0], pin[1]]) + ch[:3, 3]
        out["tip"] = torch.tensor(np.asarray(e["R_P"]).T @ (pw - hand[j][:3, 3].astype(np.float64)), dtype=torch.float32)
        return out


def loader(split, seed):
    ds = TipFrames(index.load(args.data), split, args.data, args.crops)
    return DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=10, persistent_workers=True,
                      generator=torch.Generator().manual_seed(seed))


def inputs(b):
    """-> image input (B, 14, 128, 128): per crop RGB, depth, validity, pixel position (u, v); geometry (B, 9); label."""
    chans = []
    grid = torch.arange(CROP, device=dev, dtype=torch.float32)
    for k in ("plug_uv0", "hole_uv0"):
        uv0 = b[k]
        rgb = cut(b["rgb"], uv0).permute(0, 3, 1, 2).float() / 255.0
        d = cut(b["depth"], uv0).float()[:, None] * (DEPTH_UNIT_M / DEPTH_SCALE_M)
        u = ((uv0[:, 0, None].float() + grid[None]) / 160.0 - 1)[:, None, None, :].expand(-1, 1, CROP, CROP)
        v = ((uv0[:, 1, None].float() + grid[None]) / 160.0 - 1)[:, None, :, None].expand(-1, 1, CROP, CROP)
        chans += [rgb, d, (d > 0).float(), u, v]
    R_s, t_s = insertion_motion(b["hand"].float(), b["h_star"].float(), b["R_P"].float())
    p = b["tip"][:, None]
    true = (apply_rigid(p, R_s, t_s) - p)[:, 0] * 1000
    return torch.cat(chans, 1), geometry_features(b["cam_T_world"].float(), b["hand"].float(), b["R_P"].float()), true[:, 1:], t_s.norm(dim=-1) * 1000


class PixelReader(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(14, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(r.conv1.weight)
        r.conv1.weight.data[:, 0:3], r.conv1.weight.data[:, 7:10] = w, w        # the two crops' RGB start from ImageNet
        r.maxpool, r.fc = nn.Identity(), nn.Identity()                         # stride 16 at the end: 8 x 8 map
        self.net = r
        self.geom = nn.Sequential(nn.Linear(9, 64), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(512 + 64, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, x, g):
        return self.out(torch.cat([self.net(x), self.geom(g)], -1))


model = PixelReader().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.1)
it = iter(loader("train", 1))
for step in range(1, args.steps + 1):
    try:
        b = next(it)
    except StopIteration:
        it = iter(loader("train", step))
        b = next(it)
    x, g, true, _ = inputs(to_device(b, dev))
    keep = true.norm(dim=-1) <= 6.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x, g)
    loss = F.smooth_l1_loss(pred.float()[keep], true[keep])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 200 == 0:
        print(json.dumps({"step": step, "loss": round(float(loss), 4)}), flush=True)

model.eval()
rows = []
with torch.no_grad():
    for i, b in enumerate(loader("ood_val", 7)):
        if i >= args.eval_batches:
            break
        x, g, true, dist = inputs(to_device(b, dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, g).float()
        rows.append(torch.cat([dist[:, None], true, pred], 1).cpu().numpy())
X = np.concatenate(rows)
dist, true, pred = X[:, 0], X[:, 1:3], X[:, 3:5]
lat = np.linalg.norm(true, axis=1)
rep = {"frames": int(len(X)), "steps": args.steps}
for lo, hi, lab in ((0, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "pins at the face"), (17, 1e9, ">17 mm")):
    for sub, sel in (("tip within 0.5 mm", lat <= 0.5), ("tip within 1 mm", lat <= 1.0), ("tip within 3 mm", lat <= 3.0)):
        m = (dist >= lo) & (dist < hi) & sel
        if m.sum() < 80:
            continue
        rep[f"{lab} | {sub}"] = {"n": int(m.sum()), "true_tip_offset_mm": round(float(np.median(lat[m])), 3),
                                 "blind_error_mm": round(float(np.median(np.linalg.norm(true[m] - true[m].mean(0), axis=1))), 3),
                                 "pixel_reader_error_mm": round(float(np.median(np.linalg.norm(pred[m] - true[m], axis=1))), 3),
                                 "corr_y": round(float(np.corrcoef(pred[m][:, 0], true[m][:, 0])[0, 1]), 3),
                                 "corr_z": round(float(np.corrcoef(pred[m][:, 1], true[m][:, 1])[0, 1]), 3)}
        print(f"{lab} | {sub}", rep[f"{lab} | {sub}"], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
