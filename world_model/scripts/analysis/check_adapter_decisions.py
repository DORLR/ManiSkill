"""Does a joint adapter know, before the tracking history shows it, whether the hand will advance or be held?

    python world_model/scripts/analysis/check_adapter_decisions.py --config world_model/configs/v4/adapter_vision.json \
        --adapters $Q/runs/adapter_v4e/best.pt $Q/runs/adapter_vision_v4e/best.pt --out <report.json>

Held-out contact exploration (ood_val of the config's second source), every image step with a full window of 8
commands after it. For each window, with hand positions from FK:
    commanded advance   c = command of the window's last step - command in force now      (kept if |c| >= 0.6 mm)
    real fraction       how far the measured hand really moved along c, as a fraction of |c|
    predicted fraction  the same from the adapter's predicted joints
    lag now             how far the command in force already leads the measured hand along c (mm): what the
                        tracking history can show
Windows are split by the lag now (< 0.3 mm: nothing to see in the history yet; 0.3-1 mm; >= 1 mm) and by what
really happened (advanced: fraction > 0.7; held: < 0.3). Reported per lag bin and adapter: how well the predicted
fraction separates advanced from held (AUC; 0.5 = cannot tell), the mean predicted fraction in each group and the
mean error of the fraction. The vision-conditioned adapter is run with the full history and with the last 2 s of
vision masked. Also: the imagination step and the training samples must give the adapter the same inputs
(checked on 20 windows before anything else).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data.history import AdapterHistoryDataset
from wm.data.kinematics import build_fk
from wm.imagination4 import HistoryImagination, start_from_record
from wm.training import common
from wm.training.adapter import load_adapter
from wm.training.adapter_vision import HISTORY_KEYS, TailMaskedAdapter
from wm.training.wm4 import latent_scale, offline_files, sources

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--adapters", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--source", type=int, default=1, help="index of the data source (1 = contact exploration)")
ap.add_argument("--split", default="ood_val")
args = ap.parse_args()
common.limit_gpu_memory()
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
scale = latent_scale(cfg)
fk, robot = build_fk()
files = offline_files(cfg, args.split, sources(cfg)[args.source][0])
items = []
for fi in range(len(files)):
    d = files.load(fi)
    items += [(fi, j) for j, s in enumerate(d["img_step"].astype(int)) if 1 <= s and s + 8 <= len(d["action"])]
ds = AdapterHistoryDataset(files, scale, None, items=items)
print(len(files), "trajectories,", len(items), "windows", flush=True)


# ---------------------------------------------------------------- same inputs in imagination and in training
class Capture(Exception):
    pass


class Recorder:
    takes_history = True

    def __call__(self, *a):
        self.args = a
        raise Capture


rec = Recorder()
imag = HistoryImagination(None, rec, fk, lambda *a: torch.zeros(len(a[0]), device=dev), scale, dev)
worst = {}
for k in np.random.default_rng(0).choice(len(items), size=20, replace=False):
    fi, j = items[int(k)]
    d = files.load(fi)
    st = d["img_step"][:j + 1].astype(int)
    s = int(st[-1])
    start = start_from_record(d["mean"][:j + 1].astype(np.float32) * scale, d["hand"][st].astype(np.float64),
                              d["hand_cmd"][np.clip(st, 0, len(d["hand_cmd"]) - 1)].astype(np.float64),
                              d["qpos"][:s + 1].astype(np.float64), d["action"][:s].astype(np.float64), s, d["R_P"])
    imag.reset_from([start])
    act = d["action"][s:s + 9].astype(np.float64)
    try:
        imag.step(np.concatenate([act, np.repeat(d["action"][-1:].astype(np.float64), 16 - len(act), 0)])[None])
    except Capture:
        pass
    ref = ds[int(k)]
    for name, got in zip(("x",) + HISTORY_KEYS, rec.args):
        diff = float((got[0].float().cpu() - ref[name].float()).abs().max())
        worst[name] = max(worst.get(name, 0.0), diff)
print("imagination vs training inputs, largest difference:", {k: round(v, 6) for k, v in worst.items()}, flush=True)
assert worst["x"] < 1e-5 and worst["z"] < 1e-5 and worst["proprio"] < 1e-3 and worst["path"] < 1e-3 and worst["dt"] < 1e-6, worst
assert worst["seen"] == 0 and worst["avail"] == 0, worst

# ---------------------------------------------------------------- geometry of every window
geo = {k: [] for k in ("p_now", "cmd_prev", "cmd_end", "real_end", "q_now", "fingers")}
for fi, j in items:
    d = files.load(fi)
    s = int(d["img_step"][j])
    geo["p_now"].append(d["hand"][s][:3, 3]), geo["cmd_prev"].append(d["hand_cmd"][s - 1][:3, 3])
    geo["cmd_end"].append(d["hand_cmd"][s + 7][:3, 3]), geo["real_end"].append(d["hand"][s + 8][:3, 3])
    geo["q_now"].append(d["qpos"][s, :7]), geo["fingers"].append(d["qpos"][s, 7:9])
geo = {k: np.asarray(v, np.float64) for k, v in geo.items()}
c = geo["cmd_end"] - geo["cmd_prev"]
cn = np.linalg.norm(c, axis=1)
keep = cn >= 0.0006
u = c / np.maximum(cn, 1e-9)[:, None]
f_real = ((geo["real_end"] - geo["p_now"]) * u).sum(1) / np.maximum(cn, 1e-9)
lag = ((geo["cmd_prev"] - geo["p_now"]) * u).sum(1) * 1000


def predict(adapter, dataset) -> np.ndarray:
    """Predicted joints after the 8th command, (N, 7), in the order of `items`."""
    out = []
    for b in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=6):
        b = common.to_device(b, dev)
        with torch.no_grad():
            dq = adapter(b["x"], *(b[k] for k in HISTORY_KEYS)) if getattr(adapter, "takes_history", False) else adapter(b["x"])
        out.append(dq.view(-1, 8, 7)[:, -1].double().cpu().numpy())
    return geo["q_now"] + np.concatenate(out)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)


BINS = (("lag<0.3mm", -1e9, 0.3), ("lag0.3-1mm", 0.3, 1.0), ("lag>=1mm", 1.0, 1e9))


def table(f_pred) -> dict:
    out = {}
    for name, lo, hi in BINS:
        m = keep & (lag >= lo) & (lag < hi)
        adv, held = m & (f_real > 0.7), m & (f_real < 0.3)
        out[name] = {"windows": int(m.sum()), "advanced": int(adv.sum()), "held": int(held.sum()),
                     "auc_advanced_vs_held": auc(f_pred[adv], f_pred[held]),
                     "mean_predicted_fraction_advanced": round(float(f_pred[adv].mean()), 3) if adv.any() else None,
                     "mean_predicted_fraction_held": round(float(f_pred[held].mean()), 3) if held.any() else None,
                     "mean_abs_fraction_error": round(float(np.abs(f_pred - f_real)[m].mean()), 3) if m.any() else None,
                     "mean_abs_fraction_error_held": round(float(np.abs(f_pred - f_real)[held].mean()), 3) if held.any() else None}
    return out


rep = {"windows": len(items), "with_commanded_advance": int(keep.sum()), "inputs_match_imagination": worst,
       "follow_the_command": table(1.0 + lag / np.maximum(cn * 1000, 1e-6))}      # the hand reaches the last command
for a in args.adapters:
    adapter = load_adapter(a, dev).requires_grad_(False)
    modes = {"": ds}
    if getattr(adapter, "takes_history", False):
        modes = {":full": ds, ":tail2s": TailMaskedAdapter(files, scale, items=items, tail=10)}
    for mode, dataset in modes.items():
        q_end = predict(adapter, dataset)
        p_end = fk(np.concatenate([q_end, geo["fingers"]], 1))[:, :3, 3]
        f_pred = ((p_end - geo["p_now"]) * u).sum(1) / np.maximum(cn, 1e-9)
        rep[Path(a).parent.name + mode] = table(f_pred)
        print(Path(a).parent.name + mode, json.dumps(rep[Path(a).parent.name + mode]), flush=True)
print("follow_the_command", json.dumps(rep["follow_the_command"]))
Path(args.out).write_text(json.dumps(rep, indent=1))
robot.env.close()
