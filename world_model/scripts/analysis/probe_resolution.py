"""Would twice the image resolution let the pin tip's offset from its hole be read finer?

    WM_CAM_SCALE=2 python world_model/scripts/data/collect_explore.py ... --out $Q/data/wm_hr_test      (640 x 640 frames)
    python world_model/scripts/analysis/probe_resolution.py --data $Q/data/wm_hr_test --out $Q/runs/checks_v5/probe_resolution.json
probe_pixels.py: a network of its own reads the tip across the hole axis to 0.22 mm from the 320 x 320 image's crops when
the pins are at the board face and the tip within 1 mm -- about one pixel (0.2-0.27 mm). Here the same frames exist at
640 x 640 (the same camera and field of view at twice the pixel density, as the D405's 1280-wide stream). Two networks of
the same design (ResNet-18 on the whole image: RGB, depth, depth validity; + the camera's orientation) are trained on the
same batches to give the tip's offset from its hole (socket axes, mm; simulator truth -- a diagnosis only):
    low    the frame reduced to 320 x 320 (RGB averaged 2 x 2, depth every second pixel)
    high   the frame as rendered, 640 x 640
Rows the real 1280 x 720 stream does not have (the lowest 120 of the 640) are blanked for both. Scored on held-out shapes.
If `high` is clearly finer than `low`, resolution is the limit and worth its cost; if not, it is not.
"""
import argparse
import glob
import io
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wm.data.outcome import diagnosis
from wm.data.splits import split_of
from wm.models.encoder import DEPTH_SCALE_M, DEPTH_UNIT_M
from wm.perception.labels import pose7_to_T
from wm.training.common import limit_gpu_memory

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--geometry", default="/mnt/storage2/qingxin/data/wm_v1_labelgeom")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
REAL_ROWS = 520                                    # rows of the 640 crop that the real 1280 x 720 stream has


class Frames(Dataset):
    def __init__(self, split):
        self.items = []
        for f in sorted(glob.glob(f"{args.data}/*/*/t*.npz")):
            with np.load(f) as d:
                meta = json.loads(str(d["meta"]))
                if split_of(meta["port"], meta["condition"]) != split:
                    continue
                g = json.load(open(f"{args.geometry}/{meta['port']}/{meta['grasp']}.json"))
                pin, L2 = g["pins"][0]["center_yz_m"], 2 * g["pin_half_length_m"]
                rel = d["eval_rel"][d["img_step"]]
            dep = diagnosis(rel)["depth_left_m"] * 1000
            for j in range(len(rel)):
                tip = pose7_to_T(rel[j]) @ np.array([L2, pin[0], pin[1], 1.0])
                self.items.append((f, j, ((tip[1:3] - np.asarray(pin)) * 1000).astype(np.float32), np.float32(dep[j])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, j, off, dep = self.items[k]
        with np.load(f) as d:
            o = d["rgb_offsets"]
            rgb = np.asarray(Image.open(io.BytesIO(d["rgb_jpeg"][o[j]:o[j + 1]].tobytes())).convert("RGB"))
            depth = d["depth"][j].astype(np.float32) * (DEPTH_UNIT_M / DEPTH_SCALE_M)
            R = d["cam_T_world"][j][:3, :3].astype(np.float32)
        return {"rgb": torch.from_numpy(rgb.copy()), "depth": torch.from_numpy(depth), "R": torch.from_numpy(R.reshape(-1).copy()),
                "off": torch.from_numpy(off), "dep": torch.tensor(dep)}


def views(b):
    """-> low (B, 5, 320, 320), high (B, 5, 640, 640): RGB, depth, validity; the same field of view."""
    rgb = b["rgb"].to(dev).permute(0, 3, 1, 2).float() / 255.0
    d = b["depth"].to(dev)[:, None]
    rgb[:, :, REAL_ROWS:], d[:, :, REAL_ROWS:] = 0, 0
    high = torch.cat([rgb, d, (d > 0).float()], 1)
    dl = d[:, :, ::2, ::2]
    low = torch.cat([F.avg_pool2d(rgb, 2), dl, (dl > 0).float()], 1)
    return low, high


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(r.conv1.weight)
        r.conv1.weight.data[:, :3] = w
        r.fc = nn.Identity()
        self.net = r
        self.geom = nn.Sequential(nn.Linear(9, 64), nn.SiLU())
        self.out = nn.Sequential(nn.Linear(512 + 64, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, x, R):
        return self.out(torch.cat([self.net(x), self.geom(R)], -1))


def loader(split, shuffle):
    ds = Frames(split)
    print(split, "frames", len(ds), flush=True)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, drop_last=shuffle, num_workers=12, persistent_workers=True)


models = {"low": Reader().to(dev), "high": Reader().to(dev)}
opt = torch.optim.AdamW([p for m in models.values() for p in m.parameters()], lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.1)
train = loader("train", True)
it = iter(train)
for step in range(1, args.steps + 1):
    try:
        b = next(it)
    except StopIteration:
        it = iter(train)
        b = next(it)
    low, high = views(b)
    off, R = b["off"].to(dev), b["R"].to(dev)
    keep = off.norm(dim=-1) <= 6.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = {k: F.smooth_l1_loss(m(x, R).float()[keep], off[keep]) for (k, m), x in zip(models.items(), (low, high))}
    opt.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    opt.step()
    sched.step()
    if step % 200 == 0:
        print(json.dumps({"step": step, **{k: round(float(v), 4) for k, v in losses.items()}}), flush=True)

for m in models.values():
    m.eval()
rows = []
with torch.no_grad():
    for b in loader("ood_val", False):
        low, high = views(b)
        R = b["R"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows.append(torch.cat([b["dep"][:, None].to(dev), b["off"].to(dev), models["low"](low, R).float(), models["high"](high, R).float()], 1).cpu().numpy())
X = np.concatenate(rows)
dep, true = X[:, 0], X[:, 1:3]
lat = np.linalg.norm(true, axis=1)
rep = {"frames": int(len(X)), "steps": args.steps}
for lo, hi, lab in ((-9, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "pins at the face"), (17, 1e9, ">17 mm")):
    for sub, sel in (("tip within 0.5 mm", lat <= 0.5), ("tip within 1 mm", lat <= 1.0), ("tip within 3 mm", lat <= 3.0)):
        m = (dep >= lo) & (dep < hi) & sel
        if m.sum() < 60:
            continue
        e = lambda k: round(float(np.median(np.linalg.norm(X[m][:, k:k + 2] - true[m], axis=1))), 3)
        c = lambda k: [round(float(np.corrcoef(X[m][:, k + a], true[m][:, a])[0, 1]), 2) for a in (0, 1)]
        rep[f"{lab} | {sub}"] = {"n": int(m.sum()), "true_offset_mm": round(float(np.median(lat[m])), 3),
                                 "blind_error_mm": round(float(np.median(np.linalg.norm(true[m] - true[m].mean(0), axis=1))), 3),
                                 "low_320_error_mm": e(3), "high_640_error_mm": e(5), "low_corr_yz": c(3), "high_corr_yz": c(5)}
        print(f"{lab} | {sub}", rep[f"{lab} | {sub}"], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
