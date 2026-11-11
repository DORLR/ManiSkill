"""Can the success classifier tell, before the push, whether the plug will go in? (held-out exploration)

    python world_model/scripts/analysis/check_success_on_exploration.py --success $Q/runs/success_v4e/best.pt \
        --data $Q/data/wm_v1x --latents $Q/data/wm_v4ex_latents --scale-from $Q/data/wm_v4e_latents \
        --geometry $Q/data/wm_v1_labelgeom --out $Q/runs/checks_v4/explore/success_on_exploration.json
Push-only exploration trajectories on held-out shapes (random start in front of the board, then a straight
push; nothing else): the classifier sees only the first frame (before any motion). Its P("inserted within the
horizon") is compared with what really happened, by the true offset of the pin tip from its hole at the start:
    per offset bin  how many really inserted, the mean P, and the share called "inserted" (P > 0.5)
    AUC             over all push-only starts, and inside the ambiguous band (tip offset 0.3 - 1.2 mm)
This is the direct test of whether perception suffices to foresee a jam. Also: P along the push (first frame,
at 1 s, at 2 s) -- contact gives proprioceptive evidence -- and the time real insertions took (label horizon).
"""
import argparse
import json

import numpy as np
import torch

from wm.data import index, store
from wm.data.history import history, to_tensors
from wm.perception.labels import load_geometry, pose7_to_T
from wm.training.common import limit_gpu_memory
from wm.training.success import load_success

ap = argparse.ArgumentParser()
for a in ("--success", "--data", "--latents", "--scale-from", "--geometry", "--out"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
limit_gpu_memory()
dev = torch.device("cuda")
model = load_success(args.success, dev)
scale = json.load(open(args.scale_from + "/stats.json"))["scale"]
rows = []
for e in index.load(args.data):
    if e["split"] != "ood_val" or not e["n_images"]:
        continue
    meta = store.read_meta(e["path"])
    if len(meta.get("segments", [])) != 1:                       # push-only trajectories
        continue
    lat_path = e["path"].replace(args.data, args.latents)
    with np.load(lat_path) as f:
        d = {k: f[k] for k in f.files}
    with np.load(e["path"]) as f:
        rel0 = f["eval_rel"][0]
    g = load_geometry(args.geometry, args.data, e["path"])
    tip = pose7_to_T(rel0) @ np.array([2 * g["pin_half_length_m"], *g["pins"][0]["center_yz_m"], 1.0])
    off = float(np.linalg.norm(tip[1:3] - np.asarray(g["pins"][0]["center_yz_m"])) * 1000)
    ps = []
    for j in (0, 5, 10):                                         # first frame, 1 s, 2 s into the push
        if j >= len(d["img_step"]):
            ps.append(np.nan)
            continue
        h = to_tensors(history(d, j, d["R_P"].astype(np.float64), scale))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(*(h[k][None].to(dev) for k in ("z", "seen", "proprio", "dt", "avail"))).float())[0]
        ps.append(float(p))
    ins = np.nonzero(d["inserted"])[0]
    rows.append({"offset_mm": off, "inserted": e["outcome"] == "inserted", "p0": ps[0], "p1s": ps[1], "p2s": ps[2],
                 "t_insert_s": float(d["img_step"][ins[0]] / 20) if len(ins) else None})


def auc(rs, key):
    pos = np.array([r[key] for r in rs if r["inserted"] and r[key] == r[key]])
    neg = np.array([r[key] for r in rs if not r["inserted"] and r[key] == r[key]])
    if not len(pos) or not len(neg):
        return None
    return round(float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()), 3)


rep = {"push_only_trajectories": len(rows), "inserted": sum(r["inserted"] for r in rows), "by_start_tip_offset": {}}
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.2), (1.2, 2.5), (2.5, 99)):
    g = [r for r in rows if lo <= r["offset_mm"] < hi]
    if g:
        rep["by_start_tip_offset"][f"{lo}-{hi} mm"] = {"n": len(g), "really_inserted_pct": round(100 * np.mean([r["inserted"] for r in g]), 1),
                                                        "mean_P_first_frame": round(float(np.mean([r["p0"] for r in g])), 3),
                                                        "called_inserted_pct": round(100 * np.mean([r["p0"] > 0.5 for r in g]), 1)}
band = [r for r in rows if 0.3 <= r["offset_mm"] < 1.2]
rep["auc_first_frame"] = {"all": auc(rows, "p0"), "tip_offset_0.3-1.2mm": auc(band, "p0"), "n_band": len(band)}
rep["auc_after_1s"] = {"all": auc(rows, "p1s"), "tip_offset_0.3-1.2mm": auc(band, "p1s")}
rep["auc_after_2s"] = {"all": auc(rows, "p2s"), "tip_offset_0.3-1.2mm": auc(band, "p2s")}
t = np.array([r["t_insert_s"] for r in rows if r["t_insert_s"] is not None])
rep["time_to_insert_s"] = {"n": int(len(t)), "median": float(np.median(t)), "p95": float(np.percentile(t, 95)), "max": float(t.max())} if len(t) else {}
json.dump(rep, open(args.out, "w"), indent=1)
print(json.dumps(rep, indent=1))
