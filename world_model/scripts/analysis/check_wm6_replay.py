"""v6 world model against the simulator: recorded script trajectories are replayed in imagination.

    python world_model/scripts/analysis/check_wm6_replay.py --ckpt $Q/runs/joint_v6/current.pt --data $Q/data/wm_v1c \
        --out $Q/runs/checks_v6/replay_contact.json
Held-out trajectories (ood_val: unseen shapes and grasps) of a scripted data set. From image `start` on, the world model
gets only the commands that the script really gave (hand poses) and advances by itself, 0.4 s per step, for `steps`
steps: its own features, its own hand position. Nothing real is fed back. Compared with what the simulator did:
  hand      where the hand is along the insertion axis, imagined minus real (mm), after 2 / 4 / 8 / 12 s
  held      decisions in which the real hand stayed (< 0.05 mm in 0.4 s) although the command led it by > 0.1 mm: how far the
            imagined hand moved in that decision; and trajectories whose real hand moved < 0.3 mm over the whole replay: how
            far the imagined hand went (the "creep": v4e imagined 16 mm in 16 s where the real hand moved 0.05 mm)
  inserted  the model's "inserted" at the end against the simulator's
  read-out  the label head on the imagined features against the FK label of the real frame at that time (error across the
            axis, frames within 3 mm), next to the same from the real frame's features and to an image-blind guess
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.data.frame_history import pressed_at
from wm.data.history import HIST
from wm.data.outcome import inserted as inserted_now
from wm.data.outcome import inserted
from wm.imagination6 import FeatureImagination
from wm.models.encoder import geometry_features
from wm.models.frame_encoder import centre_lateral_mm, label_targets, symexp
from wm.training.common import limit_gpu_memory
from wm.training.joint import load_joint

ap = argparse.ArgumentParser()
for a in ("--ckpt", "--data", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--start", type=int, default=6, help="image at which imagination takes over (0.2 s each)")
ap.add_argument("--steps", type=int, default=30, help="imagined steps of 0.4 s")
ap.add_argument("--max-traj", type=int, default=192)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--sampling-steps", type=int, default=20)
ap.add_argument("--real-features", action="store_true",
                help="diagnosis: after every step the imagined features are replaced by the real frame's (the hand stays the model's own): "
                     "what the hand and the judgement would be if the features were predicted perfectly")
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
enc, head, wm, ck = load_joint(args.ckpt, dev)
im = FeatureImagination(wm, dev, args.sampling_steps)
need = args.start + 2 * args.steps + 1
es = [e for e in index.load(args.data) if e["split"] == "ood_val" and port_axes_ok(e) and e["n_images"] >= need]
es = es[:: max(1, len(es) // args.max_traj)][:args.max_traj]
print("trajectories", len(es), "of", args.data, "| checkpoint step", ck["step"], flush=True)
T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)


def load(e):
    with np.load(e["path"]) as f:
        meta = json.loads(str(f["meta"]))
        d = {k: f[k] for k in ("img_step", "hand", "eval_rel", "cam_T_world", "depth", "rgb_jpeg", "rgb_offsets")}
    with np.load(str(e["path"]).replace(args.data, args.data + "_ee")) as g:
        d["hand_cmd"] = g["hand_cmd"].astype(np.float64)
    d["H_star"], d["R_P"] = np.asarray(meta["H_star"], np.float64), np.asarray(e["R_P"], np.float64)
    return d


def frames(d, idx):
    o = d["rgb_offsets"]
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(d["rgb_jpeg"][o[t]:o[t + 1]].tobytes())).convert("RGB")) for t in idx])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(T(rgb, torch.uint8), T(d["depth"][idx].astype(np.int32), torch.int32)).float()


rows, held_dec, per_traj = [], [], []
for a in range(0, len(es), args.batch):
    D = [load(e) for e in es[a:a + args.batch]]
    B, j0 = len(D), args.start
    slots = j0 - np.arange(HIST - 1, -1, -1)
    avail = slots >= 0
    sl = np.clip(slots, 0, None)
    z = torch.stack([frames(d, sl) for d in D])
    z[:, torch.from_numpy(~avail)] = 0
    st = [d["img_step"].astype(int) for d in D]
    cmd_at = lambda d, s: d["hand_cmd"][np.clip(s, 0, len(d["hand_cmd"]) - 1)]
    im.reset(z, np.tile(avail, (B, 1)), np.tile(avail, (B, 1)), np.stack([d["hand"][s[sl]].astype(np.float64) for d, s in zip(D, st)]),
             np.stack([cmd_at(d, s[sl]) for d, s in zip(D, st)]), np.stack([d["R_P"] for d in D]),
             np.array([[pressed_at(d["hand"].astype(np.float64), d["hand_cmd"], int(s[j0]), d["R_P"][:, 0]), bool(inserted_now(d["eval_rel"][s[j0]]))]
                       for d, s in zip(D, st)], np.float32))
    axis = np.stack([d["R_P"][:, 0] for d in D])                                   # the insertion axis, world
    along = lambda P: np.einsum("bi,bi->b", P[:, :3, 3], axis) * 1000              # mm
    start_real = np.stack([d["hand"][s[j0]].astype(np.float64) for d, s in zip(D, st)])
    prev_real, prev_im = start_real, start_real
    for k in range(1, args.steps + 1):
        j = j0 + 2 * k
        s0 = np.array([s[j - 2] for s in st])
        cmd9 = np.stack([cmd_at(d, s + np.arange(9)) for d, s in zip(D, s0)])
        out = im.step(cmd9)
        real = np.stack([d["hand"][s[j]].astype(np.float64) for d, s in zip(D, st)])
        lead = np.einsum("bi,bi->b", cmd9[:, 7, :3, 3] - prev_real[:, :3, 3], axis) * 1000
        d_real, d_im = along(real) - along(prev_real), along(out["hand"]) - along(prev_im)
        held = (np.abs(d_real) < 0.05) & (lead > 0.1)
        held_dec += list(zip(d_im[held], lead[held]))
        x_real = torch.cat([frames(d, [j]) for d in D])
        if args.real_features:
            im.z[:, -1] = x_real
        hand_t, cam_t = T(real), T(np.stack([d["cam_T_world"][j] for d in D]))
        _, _, tgt = label_targets(hand_t, T(np.stack([d["H_star"] for d in D])), T(np.stack([d["R_P"] for d in D])))
        geom = geometry_features(cam_t, hand_t, T(np.stack([d["R_P"] for d in D])))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e_im, e_re = centre_lateral_mm(head(out["features"], geom), tgt), centre_lateral_mm(head(x_real, geom), tgt)
        lat = symexp(tgt[:, 1:3]).cpu().numpy()
        ins = np.array([bool(inserted(d["eval_rel"][s[j]])) for d, s in zip(D, st)])
        rows.append(np.c_[np.full(B, k), along(out["hand"]) - along(real), e_im.cpu().numpy(), e_re.cpu().numpy(), lat, out["p_inserted"], ins,
                          along(real) - along(start_real), along(out["hand"]) - along(start_real)])
        prev_real, prev_im = real, out["hand"]
    print(f"replayed {a + B}/{len(es)}", flush=True)
R = np.concatenate(rows)
k, dh, e_im, e_re, lat, p, ins, mv_real, mv_im = R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4:6], R[:, 6], R[:, 7] > 0.5, R[:, 8], R[:, 9]
close = np.linalg.norm(lat, axis=1) < 3.0
rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "data": args.data, "trajectories": len(es), "start_s": args.start * 0.2,
       "sampling_steps": args.sampling_steps, "real_features_fed_back": args.real_features}
for sec in (2, 4, 8, 12, 16):
    m = k == round(sec / 0.4)
    if not m.any():
        continue
    c = m & close
    blind = np.linalg.norm(lat[c] - lat[c].mean(0), axis=1).mean() if c.any() else float("nan")
    still = m & (np.abs(mv_real) < 0.3)
    pos, neg = p[m & ins], p[m & ~ins]
    rep[f"+{sec}s"] = {
        "hand_along_axis_imagined_minus_real_mm": {"median_abs": round(float(np.median(np.abs(dh[m]))), 3), "mean": round(float(dh[m].mean()), 3), "p90_abs": round(float(np.percentile(np.abs(dh[m]), 90)), 3)},
        "real_hand_moved_under_0.3mm": {"trajectories": int(still.sum()), "imagined_moved_mm_mean": round(float(mv_im[still].mean()), 3) if still.any() else None,
                                        "imagined_moved_mm_p90_abs": round(float(np.percentile(np.abs(mv_im[still]), 90)), 3) if still.any() else None,
                                        "real_moved_mm_mean": round(float(mv_real[still].mean()), 3) if still.any() else None},
        "inserted": {"really_pct": round(100 * float(ins[m].mean()), 1), "called_pct": round(100 * float((p[m] > 0.5).mean()), 1),
                     "auc": round(float((pos[:, None] > neg[None]).mean()), 3) if len(pos) and len(neg) else None},
        "read_out_across_axis_mm_frames_within_3mm": {"n": int(c.sum()), "imagined_features": round(float(e_im[c].mean()), 3) if c.any() else None,
                                                      "real_features": round(float(e_re[c].mean()), 3) if c.any() else None, "blind": round(float(blind), 3)}}
    print(f"+{sec}s", rep[f"+{sec}s"], flush=True)
h = np.array(held_dec) if held_dec else np.zeros((0, 2))
rep["held_decisions"] = {"n": int(len(h)), "imagined_advance_mm_mean": round(float(h[:, 0].mean()), 4) if len(h) else None,
                         "imagined_advance_mm_p90_abs": round(float(np.percentile(np.abs(h[:, 0]), 90)), 4) if len(h) else None,
                         "command_lead_mm_mean": round(float(h[:, 1].mean()), 3) if len(h) else None}
print("held decisions", rep["held_decisions"], flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
