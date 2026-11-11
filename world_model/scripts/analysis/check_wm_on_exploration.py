"""Does the world model foresee what an action sequence leads to? (held-out contact exploration)

    python world_model/scripts/analysis/check_wm_on_exploration.py --config world_model/configs/v4/online.json \
        --wm $Q/runs/wm_v4e/current.pt --success $Q/runs/success_v4e/best.pt --data $Q/data/wm_v1x \
        --latents $Q/data/wm_v4ex_latents --out $Q/runs/checks_v4/explore/wm_on_exploration_<tag>.json
Every held-out exploration trajectory (random start, random maneuver, straight push) is replayed in imagination:
the world model sees only the first frame, then gets the commands that were really executed, 0.4 s at a time,
and its own predictions as history (nothing real fed back). At the end the success classifier judges the imagined
history. Compared with what really happened (inserted or not), next to
    first frame   the classifier on the first real frame alone (no world model: it cannot know the maneuver)
    real end      the classifier on the real history at the end (the ceiling: a perfect world model)
AUCs over all trajectories, over the push-only ones and over the ones with a maneuver (where the outcome
depends on the actions), with the measured joints given and with the joint adapter (as in real imagination).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wm.data import index, store
from wm.data.history import history, to_tensors
from wm.data.kinematics import build_fk
from wm.imagination4 import HistoryImagination, start_from_record
from wm.training.adapter import load_adapter
from wm.training.common import limit_gpu_memory
from wm.training.success import load_success
from wm.training.wm4 import load_history_wm

ap = argparse.ArgumentParser()
for a in ("--config", "--wm", "--success", "--data", "--latents", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--adapter")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--sampling-steps", type=int, default=20)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
fk, robot = build_fk()
succ = load_success(args.success, dev).requires_grad_(False)
wm = load_history_wm(args.wm, dev).requires_grad_(False)
adapter = load_adapter(args.adapter or cfg["actor"]["adapter"], dev).requires_grad_(False)
imag = HistoryImagination(wm, adapter, fk, succ, scale, dev, args.sampling_steps)

T = []
for e in index.load(args.data):
    if e["split"] != "ood_val" or e["n_images"] < 5:
        continue
    with np.load(e["path"].replace(args.data, args.latents)) as f:
        d = {k: f[k] for k in f.files}
    meta = store.read_meta(e["path"])
    T.append({"d": d, "inserted": e["outcome"] == "inserted", "push_only": len(meta["segments"]) == 1,
              "n_steps": (len(d["img_step"]) - 1) // 2})


def classify_real(d, j):
    h = to_tensors(history(d, j, d["R_P"].astype(np.float64), scale))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return float(torch.sigmoid(succ(*(h[k][None].to(dev) for k in ("z", "seen", "proprio", "dt", "avail"))).float())[0])


for t in T:
    t["p_first"], t["p_real_end"] = classify_real(t["d"], 0), classify_real(t["d"], 2 * t["n_steps"])

for given in (True, False):
    key = "p_wm_real_joints" if given else "p_wm_adapter"
    order = sorted(range(len(T)), key=lambda i: T[i]["n_steps"])
    for a in range(0, len(order), args.batch):
        chunk = [T[i] for i in order[a:a + args.batch]]
        starts = []
        for t in chunk:
            d = t["d"]
            cmd0 = d["hand_cmd"][[0]].astype(np.float64)
            starts.append(start_from_record([torch.from_numpy(d["mean"][0].astype(np.float32) * scale)], d["hand"][[0]].astype(np.float64),
                                            cmd0, d["qpos"][:1].astype(np.float64), d["action"][:0].astype(np.float64), 0, d["R_P"]))
        imag.reset_from(starts)
        n_max = max(t["n_steps"] for t in chunk)
        last = np.zeros(len(chunk))
        for k in range(n_max):
            targets, qs = [], []
            for t in chunk:
                d, s = t["d"], 8 * min(k, t["n_steps"] - 1)
                act = d["action"][s:s + 9].astype(np.float64)
                targets.append(np.concatenate([act, np.repeat(d["action"][-1:].astype(np.float64), 16 - len(act), 0)]))
                q = d["qpos"][s + 1:s + 9, :7].astype(np.float64)
                qs.append(np.concatenate([q, np.repeat(d["qpos"][-1:, :7].astype(np.float64), 8 - len(q), 0)]))
            info = imag.step(np.stack(targets), np.stack(qs) if given else None)
            for b, t in enumerate(chunk):
                if k == t["n_steps"] - 1:
                    last[b] = info["p_success"][b]
        for b, t in enumerate(chunk):
            t[key] = float(last[b])


def auc(rs, key):
    pos, neg = np.array([r[key] for r in rs if r["inserted"]]), np.array([r[key] for r in rs if not r["inserted"]])
    if not len(pos) or not len(neg):
        return None
    return round(float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()), 3)


rep = {"world_model": args.wm, "world_model_step": torch.load(args.wm, map_location="cpu", weights_only=False).get("step")}
for name, rs in (("all", T), ("push_only", [t for t in T if t["push_only"]]), ("with_maneuver", [t for t in T if not t["push_only"]])):
    rep[name] = {"trajectories": len(rs), "really_inserted": sum(t["inserted"] for t in rs),
                 "auc": {k: auc(rs, k) for k in ("p_first", "p_wm_real_joints", "p_wm_adapter", "p_real_end")},
                 "called_inserted_pct": {k: round(100 * float(np.mean([t[k] > 0.5 for t in rs])), 1) for k in ("p_first", "p_wm_real_joints", "p_wm_adapter", "p_real_end")},
                 "really_inserted_pct": round(100 * float(np.mean([t["inserted"] for t in rs])), 1)}
Path(args.out).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
robot.env.close()
