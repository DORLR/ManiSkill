"""Which part makes imagination optimistic on the FM's states: the classifier, the world model or the joints?

    python world_model/scripts/analysis/check_imagination_parts.py --config world_model/configs/v4/online.json \
        --episodes $Q/runs/checks_v4/imagination_v4e/episodes --wm <wm.pt> --success <success.pt> \
        --adapters <adapter.pt> [...] --out <report.json>

The closed-loop check (check_imagination4.py compare, section 3) lets the FM act inside imagination and calls a
continuation inserted when the classifier says > 0.5 at some decision. Here every recorded simulator continuation
(from its branch point to its end) is judged the same way, with more and more of it imagined:

    real            the classifier on the real histories (no world model): the classifier's own error under the
                    "at some decision" rule
    wm_real_joints  the commands really executed are replayed in the world model from the branch point, the
                    measured joints given; the latents are the world model's own (nothing real fed back)
    wm_<adapter>    the same with the joints of a joint adapter

and compared with the simulator outcome of that continuation. The executed commands are the FM's targets around
the measured joints, so they stop advancing when the arm is blocked: the replays show whether the world model
draws an insertion that did not happen when the hand motion it is given is (nearly) the true one. What the replays
cannot show is the closed loop, where the FM's targets follow the imagined joints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wm.analysis import imagination_checks4 as ic
from wm.data.kinematics import build_fk
from wm.imagination4 import HistoryImagination
from wm.training.adapter import load_adapter
from wm.training.common import limit_gpu_memory
from wm.training.success import load_success
from wm.training.wm4 import load_history_wm

ap = argparse.ArgumentParser()
for a in ("--config", "--episodes", "--wm", "--success", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--adapters", nargs="+", required=True)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--sampling-steps", type=int, default=20)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
np.random.seed(0)
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
fk, robot = build_fk()
wm = load_history_wm(args.wm, dev).requires_grad_(False)
succ = load_success(args.success, dev).requires_grad_(False)
adapters = {Path(a).parent.name: load_adapter(a, dev).requires_grad_(False) for a in args.adapters}
imag = HistoryImagination(wm, None, fk, succ, scale, dev, args.sampling_steps)

eps = [ic.load(p) for p in sorted(Path(args.episodes).glob("*.npz"))]
for ep in eps:
    ep["cmd_img"] = ic.commanded_hands(fk, ep)
    ep["n"] = len(ep["dec_img"]) - ep["k_branch"]          # decisions executed after the branch point
    ep["p"] = {}
print(len(eps), "continuations", flush=True)
B = args.batch

# ---------------------------------------------------------------- real histories
items = [(i, int(ep["dec_img"][k])) for i, ep in enumerate(eps) for k in range(ep["k_branch"], len(ep["dec_img"]))]
items += [(i, len(ep["img_step"]) - 1) for i, ep in enumerate(eps)]
best = np.zeros(len(eps))
last = np.zeros(len(eps))
for s0 in range(0, len(items), B):
    chunk = items[s0:s0 + B]
    imag.reset_from([ic.start_at_image(eps[i], j, eps[i]["cmd_img"], scale) for i, j in chunk])
    for (i, j), p in zip(chunk, imag.p_success):
        best[i] = max(best[i], p)
        if j == len(eps[i]["img_step"]) - 1:
            last[i] = p
for i, ep in enumerate(eps):
    ep["p"]["real"] = (float(best[i]), float(last[i]))
print("real histories done", flush=True)

# ---------------------------------------------------------------- replays of the executed commands
order = sorted(range(len(eps)), key=lambda i: eps[i]["n"])
for name, adapter in [("wm_real_joints", None)] + [(f"wm_{k}", a) for k, a in adapters.items()]:
    imag.adapter = adapter
    for s0 in range(0, len(order), B):
        chunk = [eps[i] for i in order[s0:s0 + B]]
        imag.reset_from([ic.start_at(ep, ep["k_branch"], ep["cmd_img"], scale) for ep in chunk])
        best, last = imag.p_success.copy(), imag.p_success.copy()
        for t in range(max(ep["n"] for ep in chunk)):
            targets, qs = [], []
            for ep in chunk:
                kk = min(ep["k_branch"] + t, len(ep["dec_img"]) - 1)
                s = int(ep["img_step"][ep["dec_img"][kk]])
                act = ep["action"][s:s + 9]
                targets.append(np.concatenate([act, np.repeat(ep["action"][-1:], 16 - len(act), 0)]))
                q = ep["qpos"][s + 1:s + 9, :7]
                qs.append(np.concatenate([q, np.repeat(ep["qpos"][-1:, :7], 8 - len(q), 0)]))
            info = imag.step(np.stack(targets), np.stack(qs) if adapter is None else None)
            for b, ep in enumerate(chunk):
                if t < ep["n"]:                                   # the continuation is still running
                    best[b] = max(best[b], info["p_success"][b])
                    last[b] = info["p_success"][b]
        for b, ep in enumerate(chunk):
            ep["p"][name] = (float(best[b]), float(last[b]))
    print(name, "done", flush=True)


def table(rs, name):
    out = {}
    for o in sorted({ep["outcome"] for ep in rs}):
        g = [ep for ep in rs if ep["outcome"] == o]
        out[o] = {"continuations": len(g),
                  "called_inserted_at_some_decision_pct": round(100 * float(np.mean([ep["p"][name][0] > 0.5 for ep in g])), 1),
                  "called_inserted_at_the_end_pct": round(100 * float(np.mean([ep["p"][name][1] > 0.5 for ep in g])), 1)}
    pos = [ep["p"][name][0] for ep in rs if ep["outcome"] == "inserted"]
    neg = [ep["p"][name][0] for ep in rs if ep["outcome"] != "inserted"]
    out["all"] = {"continuations": len(rs), "sim_inserted_pct": round(100 * len(pos) / len(rs), 1),
                  "called_inserted_at_some_decision_pct": round(100 * float(np.mean([ep["p"][name][0] > 0.5 for ep in rs])), 1),
                  "called_inserted_at_the_end_pct": round(100 * float(np.mean([ep["p"][name][1] > 0.5 for ep in rs])), 1),
                  "auc_for_sim_outcome": round(ic.auc(pos, neg), 3)}
    return out


names = list(eps[0]["p"])
rep = {"world_model": args.wm, "world_model_step": torch.load(args.wm, map_location="cpu", weights_only=False).get("step"),
       "success": args.success, "adapters": args.adapters,
       **{n: table(eps, n) for n in names},
       "by_branch": {f"k{k}": {n: table([ep for ep in eps if ep["k_branch"] == k], n)["all"] for n in names} for k in ic.BRANCHES}}
Path(args.out).write_text(json.dumps(rep, indent=1))
Path(args.out).with_suffix(".episodes.json").write_text(json.dumps(
    [{"key": ep["key"], "start": ep["start"], "k": int(ep["k_branch"]), "rep": int(ep["rep"]), "outcome": ep["outcome"],
      "condition": ep["condition"], "decisions": int(ep["n"]), **{n: ep["p"][n] for n in names}} for ep in eps]))
print(json.dumps({n: rep[n] for n in names}, indent=1))
robot.env.close()
