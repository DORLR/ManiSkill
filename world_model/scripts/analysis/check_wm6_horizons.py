"""v6: one direct prediction at +0.4 / +2 / +4 / +8 s from the real history (no step-by-step rollout): is the predicted
feature map better than "nothing changes" (the newest seen frame's features)?

    python world_model/scripts/analysis/check_wm6_horizons.py --ckpt $Q/runs/checks_v6/joint_v6c_10000.pt --out ...
Held-out windows (ood_val) of each data set. For every horizon: the label head's read-out across the axis (frames within
3 mm) from the predicted, the real and the newest seen features, and the distance in feature space (mean squared
difference to the real target's features) of the prediction and of "nothing changes". Step by step, the read-out from
imagined features went from 0.97 mm (2 s) to 5.25 mm (12 s) on the contact set (blind 0.74-1.03).
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from wm.data.frame_history import FrameHistoryDataset
from wm.models.encoder import geometry_features
from wm.models.frame_encoder import centre_lateral_mm, label_targets, symexp
from wm.models.history_cdit import create_diffusion
from wm.training.common import limit_gpu_memory, to_device
from wm.training.joint import encode_history, load_joint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", nargs="+", default=["/mnt/storage2/qingxin/data/wm_v1c", "/mnt/storage2/qingxin/data/wm_v1x"])
ap.add_argument("--shifts", nargs="+", type=int, default=[2, 10, 20, 40])
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--samplers", nargs="+", default=["ddpm20"],
                help="ddpmN: ancestral sampling with N respaced steps (NWM's own inference uses ddpm250; ours used ddpm20); "
                     "ddimN: DDIM, eta 0, N steps. Every sample is also scored after re-normalising each position's vector "
                     "(the encoder's features are normalised that way by construction; NWM clips and re-encodes its predictions)")
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
enc, head, wm, ck = load_joint(args.ckpt, dev)
import torch.nn.functional as F


def sample(kind, shape, kw):
    n = kind[4:]
    if kind.startswith("ddim"):
        d = create_diffusion(timestep_respacing="ddim" + n)
        return d.ddim_sample_loop(wm.forward, shape, torch.randn(shape, device=dev), clip_denoised=False, model_kwargs=kw, progress=False, device=dev, eta=0.0)
    d = create_diffusion(timestep_respacing=n)
    return d.p_sample_loop(wm.forward, shape, torch.randn(shape, device=dev), clip_denoised=False, model_kwargs=kw, progress=False, device=dev)


renorm = lambda x: F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)
rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"])}
for d in args.data:
    name = d.rstrip("/").split("/")[-1]
    for m in args.shifts:
        ds = FrameHistoryDataset(d, d + "_ee", "ood_val", mask=None, max_frames=ck["cfg"]["max_frames"], fixed_shift=m)
        if len(ds) < args.batch * args.batches:
            continue
        acc = {}
        for i, b in enumerate(DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=6,
                                         generator=torch.Generator().manual_seed(7))):
            if i >= args.batches:
                break
            b = to_device(b, dev)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, f_new, x = encode_history(enc, enc, b, grad=False)
                x, f_new = x.float(), f_new.float()
                kw = dict(path=b["path"], rel_t=b["rel_t"], z_hist=z.float(), seen=b["seen"], proprio=b["proprio"], dt=b["dt"], avail=b["avail"],
                          state=b["state"])
                preds = {}
                for kind in args.samplers:
                    preds[kind] = sample(kind, x.shape, kw).float()
                    preds[kind + "+renorm"] = renorm(preds[kind])
                _, _, tgt = label_targets(b["t_hand"], b["h_star"], b["R_P"])
                geom = geometry_features(b["t_cam"].float(), b["t_hand"].float(), b["R_P"].float())
                lat = symexp(tgt[:, 1:3])
                close = lat.norm(dim=-1) < 3.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = {k: centre_lateral_mm(head(v, geom), tgt) for k, v in [("real", x), ("nothing_changes", f_new)] + list(preds.items())}
            r = {f"read_out_{k}_mm": float((v * close).sum() / close.sum().clamp(min=1)) for k, v in e.items()}
            r["read_out_blind_mm"] = float(((lat - lat[close].mean(0)).norm(dim=-1) * close).sum() / close.sum().clamp(min=1))
            r["feature_mse_nothing_changes"] = float((f_new - x).pow(2).mean())
            for kind, pr in preds.items():
                r[f"feature_mse_{kind}"] = float((pr - x).pow(2).mean())
            for k, v in r.items():
                acc[k] = acc.get(k, 0.0) + v / args.batches
        rep[f"{name} +{m * 0.2:g}s"] = {k: round(v, 3) for k, v in acc.items()}
        print(f"{name} +{m * 0.2:g}s", rep[f"{name} +{m * 0.2:g}s"], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
