"""How large must the encoder's feature map be, and is the whole image enough (no crops)?

    python world_model/scripts/analysis/probe_encoder_grid.py --stride 16 --ch 16 --out $Q/runs/checks_v6/grid_s16_c16.json
The v6 encoder (models/frame_encoder.py) with its label head is trained alone for a few thousand steps on the three
scripted data sources (labels: FK of each grasp's one insertion, and "inserted now"), then scored on held-out
exploration frames exactly as the autoencoders were (check_ae_tips.py): the pin tip's remaining motion across the hole
axis, by distance left and offset. References measured 2026-11-12, pins at the board face, tip within 1 mm / 3 mm:
blind 0.41-0.45 / 1.16-1.18, crop_ae_v4e decoder 0.39 / 0.53, a ResNet-18 of its own on the two crops 0.22 / 0.27.
A measurement to choose the feature-map size; the trained weights are not kept for anything.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data import index
from wm.data.datasets import FrameDataset
from wm.data.outcome import inserted
from wm.geometry import apply_rigid
from wm.models.encoder import geometry_features
from wm.models.frame_encoder import LABEL_DIM, FrameEncoder, LabelHead, label_loss, label_motion, label_targets
from wm.perception.labels import load_geometry, pose7_to_T
from wm.training.common import limit_gpu_memory, to_device
from wm.training.wm4 import mixed_batches

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=16)
ap.add_argument("--ch", type=int, default=16)
ap.add_argument("--arch", default="resnet34")
ap.add_argument("--close-weight", type=float, default=0.0)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", required=True)
ap.add_argument("--data", nargs="+", default=[f"/mnt/storage2/qingxin/data/{n}" for n in ("wm_v1", "wm_v1x", "wm_v1c")])
ap.add_argument("--weights", nargs="+", type=float, default=[0.4, 0.25, 0.35])
ap.add_argument("--geometry", default="/mnt/storage2/qingxin/data/wm_v1_labelgeom")
ap.add_argument("--eval-data", default="/mnt/storage2/qingxin/data/wm_v1x")
ap.add_argument("--eval-batches", type=int, default=150)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")


class LabelFrames(FrameDataset):
    """+ `inserted` (now; simulator truth standing for the contact switch); `tip` (scoring only) when asked."""

    def __init__(self, root, split, tip=False):
        super().__init__(index.load(root), split)
        self.root, self.tip, self._geo = root, tip, {}

    def __getitem__(self, k):
        out = super().__getitem__(k)
        out.pop("seg")
        i, j = self.items[k]
        e = self.entries[i]
        with np.load(e["path"]) as f:
            rel = f["eval_rel"][int(f["img_step"][j])]
        out["inserted"] = torch.tensor(bool(inserted(rel)))
        if self.tip:
            if e["path"] not in self._geo:
                g = load_geometry(args.geometry, self.root, e["path"])
                self._geo[e["path"]] = (np.asarray(g["goal_world"]), 2 * g["pin_half_length_m"], g["pins"][0]["center_yz_m"])
            goal, L2, pin = self._geo[e["path"]]
            ch = goal @ pose7_to_T(rel)
            pw = ch[:3, :3] @ np.array([L2, pin[0], pin[1]]) + ch[:3, 3]
            out["tip"] = torch.tensor(np.asarray(e["R_P"]).T @ (pw - out["hand"][:3, 3].numpy().astype(np.float64)), dtype=torch.float32)
        return out


enc = FrameEncoder(ch=args.ch, stride=args.stride, arch=args.arch).to(dev)
head = LabelHead(args.ch, enc.grid).to(dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.1)
loaders = [DataLoader(LabelFrames(d, "train"), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)
           for d in args.data]
batches = mixed_batches(loaders, args.weights, 0)
enc.train(), head.train()
for step in range(1, args.steps + 1):
    b = to_device(next(batches), dev)
    _, _, target = label_targets(b["hand"], b["h_star"], b["R_P"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = head(enc(b["rgb"], b["depth"]), geometry_features(b["cam_T_world"].float(), b["hand"].float(), b["R_P"].float()))
    l = label_loss(pred, target, b["inserted"], close_weight=args.close_weight)
    opt.zero_grad(set_to_none=True)
    l["label"].backward()
    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 5.0)
    opt.step()
    sched.step()
    if step % 200 == 0:
        print(json.dumps({"step": step, **{k: round(float(v), 4) for k, v in l.items()}}), flush=True)

enc.eval(), head.eval()
rows, feats = [], []
ev = DataLoader(LabelFrames(args.eval_data, "ood_val", tip=True), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=8,
                generator=torch.Generator().manual_seed(7))
with torch.no_grad():
    for i, b in enumerate(ev):
        if i >= args.eval_batches:
            break
        b = to_device(b, dev)
        R_s, t_s, _ = label_targets(b["hand"], b["h_star"], b["R_P"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = enc(b["rgb"], b["depth"])
            pred = head(f, geometry_features(b["cam_T_world"].float(), b["hand"].float(), b["R_P"].float())).float()
        p = b["tip"][:, None]
        true = (apply_rigid(p, R_s, t_s) - p)[:, 0] * 1000
        got = (apply_rigid(p, *label_motion(pred)) - p)[:, 0] * 1000
        rows.append(torch.cat([t_s.norm(dim=-1, keepdim=True) * 1000, true, got, pred[:, LABEL_DIM:LABEL_DIM + 1], b["inserted"].float()[:, None]], 1).cpu().numpy())
        if i < 20:
            feats.append(f.float().flatten(2).transpose(1, 2).cpu())
X = np.concatenate(rows)
dist, mt, mp, logit, ins = X[:, 0], X[:, 1:4], X[:, 4:7], X[:, 7], X[:, 8] > 0.5
lat = np.linalg.norm(mt[:, 1:], axis=1)
z = torch.cat(feats).double()                                              # (N, tokens, C)
r = z - z.mean(0)
pos_, neg_ = logit[ins], logit[~ins]
rep = {"arch": args.arch, "stride": args.stride, "ch": args.ch, "close_weight": args.close_weight, "grid": enc.grid, "values_per_frame": enc.grid ** 2 * args.ch, "steps": args.steps,
       "frames": int(len(X)),
       "inserted_auc": round(float((pos_[:, None] > neg_[None]).mean()), 4) if len(pos_) and len(neg_) else None,
       "feature_variation_specific_to_a_position_pct": round(float(100 * (r - r.mean(1, keepdim=True)).pow(2).sum() / r.pow(2).sum()), 1)}
for lo, hi, lab in ((0, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "pins at the face"), (17, 1e9, ">17 mm")):
    for sub, sel in (("tip within 0.5 mm", lat <= 0.5), ("tip within 1 mm", lat <= 1.0), ("tip within 3 mm", lat <= 3.0)):
        m = (dist >= lo) & (dist < hi) & sel
        if m.sum() < 80:
            continue
        rep[f"{lab} | {sub}"] = {"n": int(m.sum()), "blind_error_mm": round(float(np.median(np.linalg.norm(mt[m][:, 1:] - mt[m][:, 1:].mean(0), axis=1))), 3),
                                 "error_mm": round(float(np.median(np.linalg.norm(mp[m][:, 1:] - mt[m][:, 1:], axis=1))), 3),
                                 "corr_y": round(float(np.corrcoef(mp[m][:, 1], mt[m][:, 1])[0, 1]), 2),
                                 "corr_z": round(float(np.corrcoef(mp[m][:, 2], mt[m][:, 2])[0, 1]), 2)}
        print(f"{lab} | {sub}", rep[f"{lab} | {sub}"], flush=True)
print({k: v for k, v in rep.items() if not isinstance(v, dict)}, flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
