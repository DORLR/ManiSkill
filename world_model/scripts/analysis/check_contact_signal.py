"""Does the contact signal tell *which way* the plug is misaligned?  (no vision, no learning of models)

The world model sees, for each context frame, the commanded and the measured hand pose (FK). When a
push is blocked the two separate. Here, for the contact trials, the features are that gap (commanded
minus measured hand pose, relative, in port axes: position mm + rotation vector deg) at several times
into the push, plus the commanded and measured motion since the push start. Linear and small-MLP
regressions fitted on train, scored on ood_val per grasp condition, of the sampled offset (simulator
truth, only the target of this check): lateral y/z, roll, pitch, yaw. Separately for trials that
jammed (the case that needs a correction) and all trials.

    python world_model/scripts/analysis/check_contact_signal.py --out $Q/runs/checks_v3g
"""

import argparse
import json
from pathlib import Path

import numpy as np

from wm.analysis.probes import PUSH_PHASE, _regressor
from wm.data import index
from wm.data.datasets import port_axes_ok

NAMES = ["lateral_y_mm", "lateral_z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
DELAYS = (8, 16, 24, 32)          # control steps after the push start: 0.4, 0.8, 1.2, 1.6 s


def rotvec_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1 + 1e-12, 1 - 1e-12)
    a = np.arccos(cos)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.degrees(v * a / (2 * np.sin(a) + 1e-15))


def rel(Ta, Tb, R_P):
    """Pose of Ta relative to Tb in port axes: position mm (3) + rotation vector deg (3)."""
    dp = R_P.T @ (Ta[:3, 3] - Tb[:3, 3]) * 1000
    dR = R_P.T @ Ta[:3, :3] @ Tb[:3, :3].T @ R_P
    return np.r_[dp, rotvec_deg(dR)]


def features(e, ee_root, data_root):
    with np.load(e["path"]) as f:
        phase, hand, meta = f["phase"], f["hand"].astype(np.float64), json.loads(str(f["meta"]))
    push = np.flatnonzero(phase == PUSH_PHASE)
    if len(push) < max(DELAYS) + 1:
        return None
    with np.load(Path(ee_root) / Path(e["path"]).relative_to(data_root)) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    R_P, s0 = np.asarray(e["R_P"], float), int(push[0])
    feats = []
    for d in DELAYS:
        s = s0 + d
        feats += [rel(cmd[s - 1], hand[s], R_P),           # commanded vs measured now (the gap)
                  rel(hand[s], hand[s0], R_P),              # measured motion since the push start
                  rel(cmd[s - 1], cmd[s0], R_P)]            # commanded motion since the push start
    off = meta["offset"]
    return np.concatenate(feats), np.array([*off["yz_mm"], *off["rpy_deg"]]), e["outcome"] == "jammed"


def r2(pred, y):
    return {n: round(float(1 - ((pred[:, k] - y[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum()), 3)
            for k, n in enumerate(NAMES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--ee", default="/mnt/storage2/qingxin/data/wm_v1_ee")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = {"train": [], "ood_val": []}
    for e in index.load(args.data):
        if e["kind"] == "contact" and e.get("outcome") in ("inserted", "jammed") and port_axes_ok(e) \
                and e["split"] in rows:
            r = features(e, args.ee, args.data)
            if r is not None:
                rows[e["split"]].append((*r, e["condition"]))
    rep = {"trials": {k: len(v) for k, v in rows.items()}, "delays_steps": DELAYS}
    for subset in ("jammed", "all"):
        keep = lambda rs: [r for r in rs if subset == "all" or r[2]]
        tr, te = keep(rows["train"]), keep(rows["ood_val"])
        xtr, ytr = np.stack([r[0] for r in tr]), np.stack([r[1] for r in tr])
        xte, yte = np.stack([r[0] for r in te]), np.stack([r[1] for r in te])
        cond = np.array([r[3] for r in te])
        for kind in ("linear", "mlp"):
            pred = _regressor(xtr, ytr, xte, kind)
            res = {"all_conditions": r2(pred, yte), "trials": len(te)}
            for c in sorted(set(cond)):
                m = cond == c
                res[c] = {"trials": int(m.sum()), **r2(pred[m], yte[m])}
            rep[f"{subset}/{kind}"] = res
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "contact_signal.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
