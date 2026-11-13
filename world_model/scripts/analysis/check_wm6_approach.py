"""v6 world model in closed loop from free space: a script pushes straight ahead; does the imagined hand stop where the
real one did?

    python world_model/scripts/analysis/check_wm6_approach.py --ckpt $Q/runs/checks_v6/joint_v6c_10000.pt --out ...
check_wm6_push.py starts in pressed states. With the contact state as a condition (v6c) the imagined hand stays there
(0.13 mm in 16 s) -- when the state says pressed. The states it did not call pressed crept all the way: the model does not
turn "free" into "pressed" by itself. That turn is the moment a policy coming from free space depends on.

Start states: held-out trajectories (ood_val) of the exploration sets that consist of the straight final push only (no
maneuver before it), at their first image: the plug 2-6 mm in front of the board, offset and tilted at random. In the
simulator the script then pushed straight along the axis until blocked or inserted, so the real hand's travel along the
axis up to that moment is known, and so is the outcome.
In imagination a script pushes the same way (decisions of 8 commands ramping from the imagined hand to `lead` ahead of
it along the axis; the hand's place across the axis and its orientation as commanded at the start) for `steps` decisions.
Compared: how far the imagined hand went against how far the real one went before it was stopped (blocked trajectories) or
until it was inserted; how often it went more than 1 mm / 3 mm past the real stop ("through the board"); what the model
says about pressed / inserted at the end.
"""
import argparse
import io
import json

import numpy as np
import torch
from PIL import Image

from wm.data import index, schema
from wm.data.datasets import port_axes_ok
from wm.data.frame_history import pressed_at
from wm.data.history import HIST
from wm.imagination6 import FeatureImagination
from wm.training.common import limit_gpu_memory
from wm.training.joint import load_joint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", nargs="+", default=["/mnt/storage2/qingxin/data/wm_v1c", "/mnt/storage2/qingxin/data/wm_v1x"])
ap.add_argument("--lead-mm", type=float, default=1.2)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--start", type=int, default=2, help="image at which imagination takes over")
ap.add_argument("--max-traj", type=int, default=192)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--sampling-steps", type=int, default=20)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
enc, head, wm, ck = load_joint(args.ckpt, dev)
im = FeatureImagination(wm, dev, args.sampling_steps)
PUSH = schema.PHASES.index("explore_push")
T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)

S = []
for D in args.data:
    es = [e for e in index.load(D) if e["split"] == "ood_val" and port_axes_ok(e) and e["n_images"] > args.start + 2]
    for e in es:
        with np.load(e["path"]) as f:
            meta = json.loads(str(f["meta"]))
            if len(meta.get("segments", [])) != 1 or meta["segments"][0]["kind"] != "final_push" or meta["segments"][0]["end"] not in ("blocked", "done"):
                continue
            ph, steps, hand = f["phase"], f["img_step"].astype(int), f["hand"].astype(np.float64)
            push = np.flatnonzero(ph == PUSH)
            if not len(push) or steps[args.start] >= push[-1]:
                continue
        with np.load(e["path"].replace(D, D + "_ee")) as g:
            cmd = g["hand_cmd"].astype(np.float64)
        a = np.asarray(e["R_P"], np.float64)[:, 0]
        s0, s1 = int(steps[args.start]), int(push[-1]) + 1                       # the push ends at s1 (blocked or at depth)
        S.append({"path": e["path"], "steps": steps, "hand": hand, "cmd": cmd, "R_P": np.asarray(e["R_P"], np.float64), "a": a, "s0": s0,
                  "real_travel": float((hand[s1, :3, 3] - hand[s0, :3, 3]) @ a) * 1000, "end": meta["segments"][0]["end"],
                  "outcome": meta["outcome"], "state": [pressed_at(hand, cmd, s0, a), False]})
S = S[:: max(1, len(S) // args.max_traj)][:args.max_traj]
print("push-only start states:", len(S), {k: sum(s["end"] == k for s in S) for k in ("blocked", "done")}, "| checkpoint step", ck["step"], flush=True)


def frames(s, idx):
    if "d" not in s:                                                        # images are read only for the states that are used
        with np.load(s["path"]) as f:
            s["d"] = {k: f[k] for k in ("depth", "rgb_jpeg", "rgb_offsets")}
    d, o = s["d"], s["d"]["rgb_offsets"]
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(d["rgb_jpeg"][o[t]:o[t + 1]].tobytes())).convert("RGB")) for t in idx])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(T(rgb, torch.uint8), T(d["depth"][idx].astype(np.int32), torch.int32)).float()


travel, p_ins, p_prs = [], [], []
for a0 in range(0, len(S), args.batch):
    B_ = S[a0:a0 + args.batch]
    B = len(B_)
    zs, seens, meas, cmds = [], [], [], []
    for s in B_:
        slots = args.start - np.arange(HIST - 1, -1, -1)
        avail = slots >= 0
        sl = np.clip(slots, 0, None)
        z = frames(s, sl)
        z[torch.from_numpy(~avail)] = 0
        st = s["steps"][sl]
        zs.append(z), seens.append(avail), meas.append(s["hand"][st]), cmds.append(s["cmd"][np.clip(st, 0, len(s["cmd"]) - 1)])
    im.reset(torch.stack(zs), np.stack(seens), np.stack(seens), np.stack(meas), np.stack(cmds), np.stack([s["R_P"] for s in B_]),
             np.array([s["state"] for s in B_], np.float32))
    axis = np.stack([s["a"] for s in B_])
    start = im.meas[:, -1].copy()
    line = im.cmd[:, -1].copy()                                             # the commanded pose at the start: its orientation and its
    tr, pi, pp = [], [], []                                                 # place across the axis are kept (a straight push)
    for k in range(args.steps):
        hand = im.meas[:, -1]
        along = np.einsum("bi,bi->b", hand[:, :3, 3] - line[:, :3, 3], axis)
        cmd9 = np.repeat(line[:, None], 9, 1).copy()
        ramp = np.minimum(np.arange(1, 10), 8) / 8.0
        cmd9[:, :, :3, 3] += (along[:, None] + ramp[None] * args.lead_mm / 1000)[..., None] * axis[:, None, :]
        out = im.step(cmd9)
        tr.append(np.einsum("bi,bi->b", out["hand"][:, :3, 3] - start[:, :3, 3], axis) * 1000)
        pi.append(out["p_inserted"].copy()), pp.append(out["p_pressed"].copy())
    travel.append(np.stack(tr, 1)), p_ins.append(np.stack(pi, 1)), p_prs.append(np.stack(pp, 1))
    print(f"pushed {a0 + B}/{len(S)}", flush=True)
M, PI, PP = np.concatenate(travel), np.concatenate(p_ins), np.concatenate(p_prs)
real = np.array([s["real_travel"] for s in S])
ins = np.array([s["outcome"] == "inserted" for s in S])
blk = np.array([s["end"] == "blocked" for s in S]) & ~ins
end = M[:, -1]
rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "start_states": len(S), "lead_mm": args.lead_mm, "seconds": args.steps * 0.4,
       "imagined_free_speed_mm_per_decision_first_2": round(float(np.median(M[:, 1] / 2)), 3)}
q = lambda v: {"median": round(float(np.median(v)), 2), "p10": round(float(np.percentile(v, 10)), 2), "p90": round(float(np.percentile(v, 90)), 2)}
for name, m in (("really_blocked", blk), ("really_inserted", ins)):
    if m.sum() < 5:
        continue
    over = end[m] - real[m]
    rep[name] = {"n": int(m.sum()), "real_travel_mm": q(real[m]), "imagined_travel_mm": q(end[m]), "imagined_minus_real_mm": q(over),
                 "went_more_than_1mm_past_pct": round(100 * float((over > 1).mean()), 1), "went_more_than_3mm_past_pct": round(100 * float((over > 3).mean()), 1),
                 "stopped_more_than_3mm_short_pct": round(100 * float((over < -3).mean()), 1),
                 "says_pressed_at_end_pct": round(100 * float((PP[m, -1] > 0.5).mean()), 1), "says_inserted_at_end_pct": round(100 * float((PI[m, -1] > 0.5).mean()), 1)}
    print(name, rep[name], flush=True)
pos, neg = PI[ins, -1], PI[blk, -1]
rep["auc_inserted_vs_blocked_by_p_inserted"] = round(float((pos[:, None] > neg[None]).mean()), 3) if len(pos) and len(neg) else None
pos, neg = end[ins], end[blk]
rep["auc_inserted_vs_blocked_by_imagined_travel"] = round(float((pos[:, None] > neg[None]).mean()), 3) if len(pos) and len(neg) else None
print({k: v for k, v in rep.items() if k.startswith("auc") or k.startswith("imagined_free")}, flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
