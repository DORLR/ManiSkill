"""Why can the pose head not read pitch? Two checks on the pose label itself (no model involved).

1. label vs truth: at the push start of every contact trial, the pose head's training label (the
   remaining motion to the inserted pose, from FK: geometry.insertion_motion with H_star and the fitted
   port axes R_P) is compared with the sampled offset (simulator truth): R^2 of a linear map from the
   label's rotation vector / translation to each offset component. If the label carried the offset,
   each component would be near 1.
2. label noise: for every grasp, the hand pose at the end of each successful insertion (reference,
   insert, perturbed, inserted contact trials) relative to H_star (the reference insertion), expressed
   in port axes: how much do "inserted" poses of the same grasp differ? Spread comparable to the
   offsets (~0.4 deg, ~0.6 mm) would make the label noisy at exactly the scale to be learned.

    python world_model/scripts/analysis/check_pose_labels.py --out $Q/runs/checks_v3
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wm.analysis.probes import PUSH_PHASE
from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.geometry import insertion_motion

NAMES = ["lateral_y_mm", "lateral_z_mm", "roll_deg", "pitch_deg", "yaw_deg"]


def rotvec_deg(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1) / 2, -1 + 1e-9, 1 - 1e-9)
    ang = np.arccos(cos)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.degrees(v * ang / (2 * np.sin(ang) + 1e-12))


def r2_linear(xtr, ytr, xte, yte) -> dict:
    A = np.c_[xtr, np.ones(len(xtr))]
    W, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.c_[xte, np.ones(len(xte))] @ W
    out = {}
    for k, n in enumerate(NAMES):
        out[n] = round(float(1 - ((pred[:, k] - yte[:, k]) ** 2).sum() / ((yte[:, k] - yte[:, k].mean()) ** 2).sum()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    entries = [e for e in index.load(args.data) if port_axes_ok(e)]
    label, truth, split = [], [], []
    ends = defaultdict(list)
    for e in entries:
        if e["kind"] == "contact" and e.get("outcome") in ("inserted", "jammed"):
            with np.load(e["path"]) as f:
                phase, hand, meta = f["phase"], f["hand"], json.loads(str(f["meta"]))
            push = np.flatnonzero(phase == PUSH_PHASE)
            if len(push) == 0:
                continue
            R, t = insertion_motion(torch.tensor(hand[push[0]][None], dtype=torch.float64),
                                    torch.tensor(meta["H_star"], dtype=torch.float64)[None],
                                    torch.tensor(e["R_P"], dtype=torch.float64)[None])
            label.append(np.r_[rotvec_deg(R[0].numpy()), t[0].numpy() * 1000])
            off = meta["offset"]
            truth.append([*off["yz_mm"], *off["rpy_deg"]])
            split.append(e["split"])
        if e.get("outcome") == "inserted" and e["kind"] in ("reference", "insert", "perturbed", "contact"):
            ends[e["grasp"]].append(e)
    label, truth, split = np.array(label), np.array(truth, np.float64), np.array(split)
    tr, te = split == "train", split == "ood_val"
    rep = {"contact_trials": {"train": int(tr.sum()), "ood_val": int(te.sum())},
           "label_vs_truth_r2_linear": r2_linear(label[tr], truth[tr], label[te], truth[te]),
           "label_rotation_deg_std": dict(zip(("x_roll", "y_pitch", "z_yaw"), np.round(label[:, :3].std(0), 3).tolist())),
           "truth_rpy_deg_std": dict(zip(("roll", "pitch", "yaw"), np.round(truth[:, 2:].std(0), 3).tolist()))}

    # 2. spread of the inserted pose within a grasp (port axes of each trajectory)
    rot, trans = [], []
    for grasp, es in ends.items():
        if len(es) < 3:
            continue
        for e in es:
            with np.load(e["path"]) as f:
                hand_end, meta = f["hand"][-1], json.loads(str(f["meta"]))
            Hs = np.asarray(meta["H_star"])
            Rp = np.asarray(e["R_P"])
            dR = Rp.T @ hand_end[:3, :3] @ Hs[:3, :3].T @ Rp
            rot.append(rotvec_deg(dR))
            trans.append(Rp.T @ (hand_end[:3, 3] - Hs[:3, 3]) * 1000)
    rot, trans = np.array(rot), np.array(trans)
    q = lambda x: {"median_abs": np.round(np.median(np.abs(x), 0), 3).tolist(), "p90_abs": np.round(np.percentile(np.abs(x), 90, 0), 3).tolist()}
    rep["inserted_pose_vs_H_star"] = {"trajectories": len(rot), "rotation_deg_xyz(roll,pitch,yaw)": q(rot),
                                      "translation_mm_xyz(depth,lateral_y,lateral_z)": q(trans)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pose_labels.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
