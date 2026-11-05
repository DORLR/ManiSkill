"""Cheap check before the closed loop: state-estimation error on the shape each
fold never saw (its collected views), next to the error on the fold's own
validation grasps. No simulation, a few minutes of GPU.

python scripts/offline_check.py --runs runs --out results/offline_check.json
"""

import argparse
import json

import numpy as np
import torch

from paf import config as C
from paf.model import AlignNet, RGBOnlyNet
from paf.train import load_shards


def errors(model, D, dev="cuda", bs=512):
    u = D["q"] / C.SIGMA
    out = {"pc": [], "hat": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(u), bs):
            o = model(torch.as_tensor(D["rgb"][s:s + bs], device=dev),
                      torch.as_tensor(D["pc"][s:s + bs], device=dev).float())
            out["pc"].append(o["q_pc"].float().cpu().numpy())
            out["hat"].append(o["q_hat"].float().cpu().numpy())
    res = {}
    near = np.linalg.norm(u, axis=1) < 3.0
    for k in ("pc", "hat"):
        e = np.abs(np.concatenate(out[k]) - u)
        for reg, m in (("near", near), ("all", np.ones_like(near))):
            res[f"{k}_{reg}_t_med_mm"] = float(np.median(e[m][:, :3]))
            res[f"{k}_{reg}_r_med_deg"] = float(np.median(e[m][:, 3:]))
            res[f"{k}_{reg}_t_p90_mm"] = float(np.percentile(e[m][:, :3], 90))
            res[f"{k}_{reg}_r_p90_deg"] = float(np.percentile(e[m][:, 3:], 90))
    res["n_views"] = int(len(u))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data/train/*.npz")
    ap.add_argument("--out", default="results/offline_check.json")
    ap.add_argument("--rgb-only", action="store_true")
    args = ap.parse_args()
    report = {}
    for fold in C.PORTS:
        model = (RGBOnlyNet if args.rgb_only else AlignNet)(pretrained=False).cuda().eval()
        model.load_state_dict(torch.load(f"{args.runs}/fold_{fold}/model.pt", map_location="cuda"))
        D, _ = load_shards(args.data, [fold])
        summ = json.load(open(f"{args.runs}/fold_{fold}/summary.json"))
        best = min(summ["history"], key=lambda h: h["hat_near_t_med_mm"] + h["hat_near_r_med_deg"])
        report[fold] = {"unseen_shape": errors(model, D),
                        "seen_val_best_epoch": {k: v for k, v in best.items() if k.startswith(("pc_", "hat_"))},
                        "id_set": {"n": summ["n_id"], "std_sigma": summ["id_set_std_sigma"],
                                   "mean_sigma": summ["id_set_mean_sigma"]}}
        u = report[fold]["unseen_shape"]
        v = report[fold]["seen_val_best_epoch"]
        print(f"fold {fold}: near-ID median error  seen-val {v['hat_near_t_med_mm']:.2f} mm/"
              f"{v['hat_near_r_med_deg']:.2f} deg | unseen shape {u['hat_near_t_med_mm']:.2f} mm/"
              f"{u['hat_near_r_med_deg']:.2f} deg (pc-only {u['pc_near_t_med_mm']:.2f} mm/"
              f"{u['pc_near_r_med_deg']:.2f} deg)", flush=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
