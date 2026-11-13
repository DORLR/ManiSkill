"""v6 world model in closed loop: a script keeps pushing from where the IMAGINED hand is.

    python world_model/scripts/analysis/check_wm6_push.py --ckpt $Q/runs/checks_v6/joint_v6_10000.pt --data $Q/data/wm_v1c \
        --out $Q/runs/checks_v6/push_joint_v6_10000.json
The v4e imagination crept: with a policy that re-plans from the imagined hand, the hand went 16 mm in 16 s in states
where the real one moved 0.05 mm (every decision the hand took about 0.36 mm of a 0.58 mm lead, and the next decision
started from there). The replay check cannot show this, its commands are recorded. Here the commands follow the
imagination: every 0.4 s a decision of 8 commands ramps from the imagined hand to that position + `lead` along the
insertion axis -- the contact set's "lead" primitive, with the imagined hand in place of the measured one.

Start states: held-out contact-set trajectories whose final push was blocked and whose real hand then stayed within
0.3 mm for the next `hold` seconds under the script's own pushing (so: pressed on the board, and pushing on did nothing
in the simulator). Imagination takes over 1 s into that phase with the real history up to there.
Reported: how far the imagined hand goes along the axis after 2 / 4 / 8 / 16 s (the real hand under the same kind of
commands: under 0.3 mm, by selection), per-decision advance, and how often the model says "inserted".
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
from wm.data.outcome import inserted as inserted_now
from wm.imagination6 import FeatureImagination
from wm.training.common import limit_gpu_memory
from wm.training.joint import load_joint

ap = argparse.ArgumentParser()
for a in ("--ckpt", "--data", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--lead-mm", type=float, default=0.6)
ap.add_argument("--orientation", default="measured", choices=["measured", "commanded"],
                help="measured: the commanded orientation is the imagined hand's (a policy that re-plans joint targets from "
                     "the measured joints); commanded: the orientation commanded when imagination took over is kept (the "
                     "contact set's lead primitive)")
ap.add_argument("--steps", type=int, default=40, help="decisions of 0.4 s")
ap.add_argument("--next-first", default="continues", choices=["continues", "from_hand"],
                help="the 9th command given to the model (the first command of the NEXT decision, which a re-planning script only "
                     "knows after it has seen where the hand went): continues = the ramp's end; from_hand = as if the next decision "
                     "started again from the hand's present place (what it is in the recorded data whenever the hand was held)")
ap.add_argument("--hold", type=float, default=6.0)
ap.add_argument("--max-traj", type=int, default=128)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--sampling-steps", type=int, default=20)
ap.add_argument("--force-pressed", type=int, default=None, choices=[0, 1],
                help="control arm (2026-11-13): the pressed bit given to the model is set to this value at every step instead of "
                     "carrying the model's own output -- does the bit alone decide whether the hand holds?")
ap.add_argument("--freeze-command", action="store_true",
                help="control arm: the command stays at the START position + lead instead of following the imagined hand -- "
                     "separates the feedback of a command that follows the imagined hand from what the model does under a "
                     "constant lead")
ap.add_argument("--real-features", action="store_true",
                help="control arm: after every step the newest image slot gets the real start frame's features again (the real "
                     "hand does not move in these states, so that is what the camera would show); the hand stays imagined")
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
enc, head, wm, ck = load_joint(args.ckpt, dev)
im = FeatureImagination(wm, dev, args.sampling_steps)
C_PHASE = schema.PHASES.index("explore_contact")
T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)

starts = []
for e in [e for e in index.load(args.data) if e["split"] == "ood_val" and port_axes_ok(e)]:
    with np.load(e["path"]) as f:
        meta = json.loads(str(f["meta"]))
        if meta.get("stopped") != "blocked":
            continue
        ph, steps, hand = f["phase"], f["img_step"].astype(int), f["hand"].astype(np.float64)
        c = np.flatnonzero(ph == C_PHASE)
        if not len(c):
            continue
        j0 = int(np.searchsorted(steps, c[0] + 20))                      # 1 s into the contact phase
        j1 = int(np.searchsorted(steps, c[0] + 20 + int(args.hold * 20)))
        if j1 >= len(steps):
            continue
        a = np.asarray(e["R_P"], np.float64)[:, 0]
        if np.abs((hand[steps[j0]:steps[j1] + 1, :3, 3] - hand[steps[j0], :3, 3]) @ a).max() * 1000 > 0.3:
            continue
        d = {k: f[k] for k in ("depth", "rgb_jpeg", "rgb_offsets")}
    with np.load(str(e["path"]).replace(args.data, args.data + "_ee")) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    starts.append({"d": d, "steps": steps, "hand": hand, "cmd": cmd, "j0": j0, "R_P": np.asarray(e["R_P"], np.float64),
                   "state": [pressed_at(hand, cmd, int(steps[j0]), a), False]})
    if len(starts) >= args.max_traj:
        break
print("pressed start states:", len(starts), "| checkpoint step", ck["step"], flush=True)


def frames(s, idx):
    d, o = s["d"], s["d"]["rgb_offsets"]
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(d["rgb_jpeg"][o[t]:o[t + 1]].tobytes())).convert("RGB")) for t in idx])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(T(rgb, torch.uint8), T(d["depth"][idx].astype(np.int32), torch.int32)).float()


moved, per_dec, p_ins, prs_steps = [], [], [], []
for a0 in range(0, len(starts), args.batch):
    S = starts[a0:a0 + args.batch]
    B = len(S)
    zs, seens, meas, cmds = [], [], [], []
    for s in S:
        slots = s["j0"] - np.arange(HIST - 1, -1, -1)
        avail = slots >= 0
        sl = np.clip(slots, 0, None)
        z = frames(s, sl)
        z[torch.from_numpy(~avail)] = 0
        st = s["steps"][sl]
        zs.append(z), seens.append(avail), meas.append(s["hand"][st]), cmds.append(s["cmd"][np.clip(st, 0, len(s["cmd"]) - 1)])
    im.reset(torch.stack(zs), np.stack(seens), np.stack(seens), np.stack(meas), np.stack(cmds), np.stack([s["R_P"] for s in S]),
             np.array([s["state"] for s in S], np.float32))
    axis = np.stack([s["R_P"][:, 0] for s in S])
    start = im.meas[:, -1].copy()
    R_cmd = im.cmd[:, -1, :3, :3].copy()
    z_real = im.z[:, -1].clone()
    along = lambda P: np.einsum("bi,bi->b", P[:, :3, 3] - start[:, :3, 3], axis) * 1000
    prev = np.zeros(B)
    for k in range(1, args.steps + 1):
        hand = im.meas[:, -1]
        cmd9 = np.repeat(hand[:, None], 9, 1).copy()                      # position ramps from the imagined hand to hand + lead
        if args.orientation == "commanded":
            cmd9[:, :, :3, :3] = R_cmd[:, None]
        ramp = np.minimum(np.arange(1, 10), 8) / 8.0
        if args.next_first == "from_hand":
            ramp[8] = 1 / 8.0
        cmd9[:, :, :3, 3] += ramp[None, :, None] * (args.lead_mm / 1000) * axis[:, None, :]
        if args.freeze_command:
            cmd9[:, :, :3, 3] = start[:, None, :3, 3] + (args.lead_mm / 1000) * axis[:, None, :]
        if args.force_pressed is not None:
            im.state[:, 0] = args.force_pressed
        out = im.step(cmd9)
        if args.real_features:
            im.z[:, -1] = z_real
        now = along(out["hand"])
        per_dec.append(now - prev)
        prev = now
        moved.append(now.copy()), p_ins.append(out["p_inserted"].copy()), prs_steps.append(out["p_pressed"].copy())
    print(f"pushed {a0 + B}/{len(starts)}", flush=True)
n_b = int(np.ceil(len(starts) / args.batch))
M = np.concatenate([np.stack(moved[i * args.steps:(i + 1) * args.steps], 1) for i in range(n_b)])       # (N, steps)
P = np.concatenate([np.stack(p_ins[i * args.steps:(i + 1) * args.steps], 1) for i in range(n_b)])
D = np.concatenate([np.stack(per_dec[i * args.steps:(i + 1) * args.steps], 1) for i in range(n_b)])
PR = np.concatenate([np.stack(prs_steps[i * args.steps:(i + 1) * args.steps], 1) for i in range(n_b)])
rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "start_states": len(starts), "lead_mm": args.lead_mm,
       "start_states_pressed_by_the_rule_pct": round(100 * float(np.mean([s["state"][0] for s in starts])), 1),
       "orientation": args.orientation, "next_first": args.next_first,
       "force_pressed": args.force_pressed, "freeze_command": args.freeze_command, "real_features": args.real_features,
       "real_hand_in_these_states": "moved under 0.3 mm in the next %.0f s (selection)" % args.hold,
       "advance_per_decision_mm": {"mean": round(float(D.mean()), 4), "median": round(float(np.median(D)), 4), "p90_abs": round(float(np.percentile(np.abs(D), 90)), 4)}}
for sec in (2, 4, 8, 16):
    k = round(sec / 0.4) - 1
    if k < args.steps:
        rep[f"+{sec}s"] = {"imagined_hand_moved_mm": {"mean": round(float(M[:, k].mean()), 3), "median": round(float(np.median(M[:, k])), 3),
                                                      "p90": round(float(np.percentile(M[:, k], 90)), 3), "max": round(float(M[:, k].max()), 3)},
                           "says_inserted_pct": round(100 * float((P[:, k] > 0.5).mean()), 1), "mean_p_inserted": round(float(P[:, k].mean()), 3),
                           "says_pressed_pct": round(100 * float((PR[:, k] > 0.5).mean()), 1)}
        print(f"+{sec}s", rep[f"+{sec}s"], flush=True)
print("per decision", rep["advance_per_decision_mm"], flush=True)
bit = np.array([bool(s["state"][0]) for s in starts])
q = lambda v: {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
               "p90": round(float(np.percentile(v, 90)), 3)} if len(v) else {"n": 0}
flip = np.where((PR < 0.5).any(1), (PR < 0.5).argmax(1), args.steps)          # first decision at which the model says "not pressed"
before = np.array([M[i, flip[i] - 1] if flip[i] > 0 else 0.0 for i in range(len(M))])
rep["by_start_bit"] = {"pressed_by_the_rule": {"travel_end_mm": q(M[bit, -1]), "says_pressed_end_pct": round(100 * float((PR[bit, -1] > 0.5).mean()), 1) if bit.any() else None,
                                               "first_not_pressed_decision": q(flip[bit]), "travel_before_first_not_pressed_mm": q(before[bit]),
                                               "advance_per_decision_while_says_pressed_mm": round(float(D[bit][np.concatenate([np.ones((bit.sum(), 1), bool), PR[bit, :-1] > 0.5], 1)].mean()), 4) if bit.any() else None},
                       "not_pressed_by_the_rule": {"travel_end_mm": q(M[~bit, -1]), "says_pressed_end_pct": round(100 * float((PR[~bit, -1] > 0.5).mean()), 1) if (~bit).any() else None,
                                                   "says_pressed_ever_pct": round(100 * float((PR[~bit] > 0.5).any(1).mean()), 1) if (~bit).any() else None}}
print("by start bit", rep["by_start_bit"], flush=True)
np.savez_compressed(args.out.replace(".json", "_cases.npz"), moved_mm=M, per_decision_mm=D, p_pressed=PR, p_inserted=P, start_pressed=bit)
json.dump(rep, open(args.out, "w"), indent=1)
