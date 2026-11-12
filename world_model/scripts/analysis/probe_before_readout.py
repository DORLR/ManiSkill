"""Step 0 of the v5 plan: is the sub-millimetre offset still in the encoder's tokens BEFORE the read-out?

    python world_model/scripts/analysis/probe_before_readout.py --ae $Q/runs/crop_ae_v4e/best.pt --data $Q/data/wm_v1x \
        --crops $Q/data/wm_v1x_crops --geometry $Q/data/wm_v1_labelgeom --out $Q/runs/checks_v5/probe_before_readout_v4e.json
The autoencoder is frozen. Two small readers of the same design (linear layer, 2 self-attention layers, mean, linear)
are trained on the exploration frames of the train split to give where the first pin's tip is across the hole axis
(mm; the label of check_ae_tips.py, simulator truth, used for this diagnosis only):
    before   from the 512 crop tokens after the joint attention (what the read-out reads)
    latent   from the latent tokens (what every later model gets)
Scored on held-out shapes (ood_val), in the bins of check_ae_tips.py, next to the autoencoder's own decoder and to an
image-blind reader (the bin's mean). If `before` reads the tip well and `latent` does not, the detail is lost in the
read-out. A reader that does not read it from `before` proves nothing (those layers were trained with this read-out).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wm.data import index
from wm.data.datasets import CropFrameDataset
from wm.geometry import apply_rigid, insertion_motion
from wm.models.crop_ae import load_crop_ae
from wm.perception.labels import load_geometry, pose7_to_T
from wm.training.common import limit_gpu_memory, to_device

ap = argparse.ArgumentParser()
for a in ("--ae", "--data", "--crops", "--geometry", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--eval-batches", type=int, default=150)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
model = load_crop_ae(args.ae, dev)


class TipFrames(CropFrameDataset):
    """+ `tip`: the first pin's tip in hand-centred port axes (truth; the probe's label and the scoring only)."""
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


@torch.no_grad()
def features(b):
    """-> crop tokens before the read-out (B, 512, d), latent (B, tokens, C), decoder's and true tip motion (mm), distance left (mm)."""
    got = {}
    h = model.encoder.joint.register_forward_hook(lambda m, i, o: got.__setitem__("x", o))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    h.remove()
    R_s, t_s = insertion_motion(b["hand"].float(), b["h_star"].float(), b["R_P"].float())
    p = b["tip"][:, None]
    true = (apply_rigid(p, R_s, t_s) - p)[:, 0] * 1000
    dec = (apply_rigid(p, out["R"].float(), out["t"].float()) - p)[:, 0] * 1000
    return got["x"][:, 3:].float(), out["z"].float(), dec[:, 1:], true[:, 1:], t_s.norm(dim=-1) * 1000


class Reader(nn.Module):
    def __init__(self, d_in, n_tok, d=128):
        super().__init__()
        self.norm, self.inp = nn.LayerNorm(d_in), nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, x):
        return self.out(self.enc(self.inp(self.norm(x)) + self.pos).mean(1))


readers, opt = None, None
it = iter(loader("train", 1))
for step in range(1, args.steps + 1):
    try:
        b = next(it)
    except StopIteration:
        it = iter(loader("train", step))
        b = next(it)
    x, z, _, true, _ = features(to_device(b, dev))
    if readers is None:
        readers = {"before": Reader(x.shape[-1], x.shape[1]).to(dev), "latent": Reader(z.shape[-1], z.shape[1]).to(dev)}
        opt = torch.optim.AdamW([p for r in readers.values() for p in r.parameters()], lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.1)
    keep = true.norm(dim=-1) <= 6.0                                      # the far ones are easy and not the question
    loss = sum(F.smooth_l1_loss(r(v)[keep], true[keep]) for r, v in ((readers["before"], x), (readers["latent"], z)))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 200 == 0:
        print(json.dumps({"step": step, "loss": round(float(loss), 4)}), flush=True)

rows = []
for r in readers.values():
    r.eval()
with torch.no_grad():
    for i, b in enumerate(loader("ood_val", 7)):
        if i >= args.eval_batches:
            break
        x, z, dec, true, dist = features(to_device(b, dev))
        rows.append(torch.cat([dist[:, None], true, dec, readers["before"](x), readers["latent"](z)], 1).cpu().numpy())
X = np.concatenate(rows)
dist, true = X[:, 0], X[:, 1:3]
lat = np.linalg.norm(true, axis=1)
rep = {"autoencoder": args.ae, "frames": int(len(X)), "probe_steps": args.steps}
for lo, hi, lab in ((0, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "pins at the face"), (17, 1e9, ">17 mm")):
    for sub, sel in (("tip within 1 mm", lat <= 1.0), ("tip within 3 mm", lat <= 3.0)):
        m = (dist >= lo) & (dist < hi) & sel
        if m.sum() < 80:
            continue
        err = lambda k: round(float(np.median(np.linalg.norm(X[m][:, k:k + 2] - true[m], axis=1))), 3)
        rep[f"{lab} | {sub}"] = {"n": int(m.sum()), "true_tip_offset_mm": round(float(np.median(lat[m])), 3),
                                 "blind_error_mm": round(float(np.median(np.linalg.norm(true[m] - true[m].mean(0), axis=1))), 3),
                                 "decoder_error_mm": err(3), "reader_before_readout_error_mm": err(5), "reader_on_latent_error_mm": err(7)}
        print(f"{lab} | {sub}", rep[f"{lab} | {sub}"], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
