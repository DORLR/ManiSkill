"""    python world_model/scripts/analysis/check_ae_tips.py --ae $Q/runs/crop_ae_v4e/best.pt --data $Q/data/wm_v1x \
        --crops $Q/data/wm_v1x_crops --geometry $Q/data/wm_v1_labelgeom --out $Q/runs/checks_v4/explore/ae_tips_v4e.json

Autoencoder readout on held-out exploration frames, in the quantities that decide insert / jam:
(1) where the pin tips are across the hole axis (predicted remaining motion of the true tip points vs the label),
(2) the remaining rotation split into roll (about the insertion axis), pitch and yaw.
Next to each: the error of an image-blind reader (always the mean of the bin). Truth (tip points) only for scoring."""
import json, sys
import numpy as np, torch
from wm.data import index
from wm.data.datasets import CropFrameDataset
from wm.geometry import apply_rigid, insertion_motion
from wm.models.crop_ae import load_crop_ae
from wm.perception.labels import load_geometry, pose7_to_T
from wm.training.common import limit_gpu_memory, to_device

limit_gpu_memory()
import argparse
ap = argparse.ArgumentParser()
for a in ("--ae", "--data", "--crops", "--geometry", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--batches", type=int, default=250)
args = ap.parse_args()
ae, n_batches = args.ae, args.batches
dev = torch.device("cuda")
model = load_crop_ae(ae, dev)
ds = CropFrameDataset(index.load(args.data), "ood_val", args.data, args.crops)
rng = np.random.default_rng(7)
order = rng.permutation(len(ds))[:n_batches * 32]
geo_cache, rows = {}, []


def rotvec(R):
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1) / 2
    s = w.norm(dim=1).clamp(min=1e-9)
    ang = torch.atan2(s, ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1, 1))
    return torch.rad2deg(w / s[:, None] * ang[:, None])


for a in range(0, len(order), 32):
    ks = order[a:a + 32]
    items = [ds[int(k)] for k in ks]
    b = to_device({k: torch.stack([it[k] for it in items]) for k in items[0]}, dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    R_s, t_s = insertion_motion(b["hand"].float(), b["h_star"].float(), b["R_P"].float())
    R, t = out["R"].float(), out["t"].float()
    tips = []
    for k in ks:
        i, j = ds.items[int(k)]
        e = ds.entries[i]
        if e["path"] not in geo_cache:
            g = load_geometry(args.geometry, args.data, e["path"])
            with np.load(e["path"]) as f:
                geo_cache[e["path"]] = (np.asarray(g["goal_world"]), 2 * g["pin_half_length_m"], [p["center_yz_m"] for p in g["pins"]],
                                        f["eval_rel"][f["img_step"]], f["hand"][f["img_step"]])
        goal, L2, pins, rel, hand = geo_cache[e["path"]]
        ch = goal @ pose7_to_T(rel[j])
        pw = np.array([[L2, p[0], p[1]] for p in pins[:1]]) @ ch[:3, :3].T + ch[:3, 3]            # first pin's tip, world
        tips.append((np.asarray(e["R_P"]).T @ (pw[0] - hand[j][:3, 3].astype(np.float64))))        # hand-centred port axes
    p = torch.tensor(np.array(tips), dtype=torch.float32, device=dev)[:, None]
    m_true = (apply_rigid(p, R_s, t_s) - p)[:, 0] * 1000                                           # remaining motion of the tip, mm
    m_pred = (apply_rigid(p, R, t) - p)[:, 0] * 1000
    rv_t, rv_p = rotvec(R_s), rotvec(R)
    rows.append(torch.cat([t_s.norm(dim=-1, keepdim=True) * 1000, m_true, m_pred, rv_t, rv_p], 1).cpu().numpy())
X = np.concatenate(rows)
dist, mt, mp, rt, rp = X[:, 0], X[:, 1:4], X[:, 4:7], X[:, 7:10], X[:, 10:13]
lat_true = np.linalg.norm(mt[:, 1:], axis=1)
med = lambda v: float(np.median(v)) if len(v) else float("nan")
print("frames", len(X))
rep = {"autoencoder": ae, "frames": int(len(X))}
for lo, hi, lab in ((0, 5, "<5 mm left"), (5, 13, "5-13 mm"), (13, 17, "pins at the face"), (17, 1e9, ">17 mm")):
    for sub, sel in (("tip within 1 mm", lat_true <= 1.0), ("tip within 3 mm", lat_true <= 3.0), ("all", lat_true < 1e9)):
        m = (dist >= lo) & (dist < hi) & sel
        if m.sum() < 80:
            continue
        err = np.linalg.norm(mp[m][:, 1:] - mt[m][:, 1:], axis=1)
        blind = np.linalg.norm(mt[m][:, 1:] - mt[m][:, 1:].mean(0), axis=1)
        corr = [float(np.corrcoef(mp[m][:, k], mt[m][:, k])[0, 1]) for k in (1, 2)]
        ax = " ".join("%s %.2f (blind %.2f)" % (n, med(np.abs(rp[m][:, k] - rt[m][:, k])), med(np.abs(rt[m][:, k] - rt[m][:, k].mean())))
                      for k, n in enumerate(("roll", "pitch", "yaw")))
        rep[f"{lab} | {sub}"] = {"n": int(m.sum()), "true_tip_offset_mm": round(med(lat_true[m]), 3), "error_mm": round(med(err), 3),
                                 "blind_error_mm": round(med(blind), 3), "corr_horizontal": round(corr[0], 3), "corr_vertical": round(corr[1], 3),
                                 **{f"{n}_err_deg": round(med(np.abs(rp[m][:, k] - rt[m][:, k])), 3) for k, n in enumerate(("roll", "pitch", "yaw"))},
                                 **{f"{n}_blind_deg": round(med(np.abs(rt[m][:, k] - rt[m][:, k].mean())), 3) for k, n in enumerate(("roll", "pitch", "yaw"))}}
        print("%-17s %-16s n %5d | tip across axis: true %.2f mm, error %.2f, blind %.2f, corr y %.2f z %.2f | deg: %s"
              % (lab, sub, int(m.sum()), med(lat_true[m]), med(err), med(blind), corr[0], corr[1], ax))

json.dump(rep, open(args.out, "w"), indent=1)
