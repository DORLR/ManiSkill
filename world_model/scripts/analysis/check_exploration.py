"""Checks of the contact-exploration data (collect/explore.py) before it is used for training.

    python world_model/scripts/analysis/check_exploration.py --data $Q/data/wm_v1x --ee $Q/data/wm_v1x_ee \
        --policy-episodes $Q/runs/checks_v4/imagination/episodes --out $Q/runs/checks_v4/explore/report.json
A  collection: outcomes, placement redraws, how segments ended, lengths, start error
B  physics sanity: push-only trajectories by start offset -- aligned starts must mostly insert, far ones never
C  coverage: where the plug is (true depth left x offset from the hole axis; share of frames) in the
   exploration data vs in the FM policy's own rollouts on held-out shapes; cells the policy visits (>= 1 % of
   its frames) that exploration hardly contains (< 0.2 %) are listed
D  no truth in the commands: the true start offset and the final outcome predicted from the commanded hand
   path of the first 0.5 s alone (before contact can change the commands) (ridge regression, grasps held out). R^2 must be ~0 and the outcome AUC ~0.5
   (the expert data failed this: its commands point at the hole)
Exit code 1 if B or D fails.
"""
import argparse
import glob
import json
import math
import sys
from collections import Counter

import numpy as np

from wm.data import index
from wm.data.conditions import pose_feature
from wm.data.outcome import diagnosis

ap = argparse.ArgumentParser()
for a in ("--data", "--ee", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--policy-episodes")
args = ap.parse_args()
rep, fail = {}, []

# ------------------------------------------------------------------ A, B from the grasp summaries
T, errors = [], []
for f in sorted(glob.glob(f"{args.data}/*/*/grasp.json")):
    s = json.load(open(f))
    if s.get("error"):
        errors.append([s["grasp"], s["error"][:120]])
    T += s["trajectories"]
ok = [t for t in T if "depth_left_mm" in t and t.get("outcome") not in ("empty", "placement_failed")]
lat0 = lambda t: math.hypot(*t["start"]["lateral_m"]) * 1000
rot0 = lambda t: math.sqrt(sum(x * x for x in t["start"]["rotvec_deg"]))
rep["A_collection"] = {
    "grasps": len(glob.glob(f"{args.data}/*/*/grasp.json")), "grasp_errors": errors[:10], "n_grasp_errors": len(errors),
    "trajectories_planned": len(T), "trajectories_saved": len(ok),
    "outcomes": dict(Counter(t.get("outcome") for t in T)),
    "placement_redraws": dict(sorted(Counter(len(t.get("placement_failures", [])) for t in T).items())),
    "stopped_by": dict(Counter(t.get("stopped") for t in ok)),
    "segments": {f"{k[0]}:{k[1]}": v for k, v in sorted(Counter((x["kind"], x["end"]) for t in ok for x in t["segments"]).items())},
    "steps_median": float(np.median([t["steps"] for t in ok])), "steps_max": int(max(t["steps"] for t in ok)),
    "start_error_mm_median": float(np.median([t["start_error_mm"] for t in ok])),
    "start_error_mm_max": float(max(t["start_error_mm"] for t in ok))}
push = [t for t in ok if len(t["segments"]) == 1]
B = {}
for lo, hi in ((0, 1e-9), (1e-9, 0.5), (0.5, 1), (1, 3), (3, 11)):
    g = [t for t in push if lo <= lat0(t) < hi and rot0(t) < 1.0] if hi > 1e-9 else [t for t in push if lat0(t) == 0]
    name = "aligned" if hi <= 1e-9 else f"{lo:g}-{hi:g} mm, tilt < 1 deg"
    B[name] = {"n": len(g), "inserted": sum(t["outcome"] == "inserted" for t in g)}
rep["B_push_only_by_start_offset"] = B
al, far = B["aligned"], B["3-11 mm, tilt < 1 deg"]
if al["n"] >= 10 and al["inserted"] / al["n"] < 0.6:
    fail.append("aligned push-only starts insert less than 60 %")
if far["n"] >= 10 and far["inserted"] / far["n"] > 0.05:
    fail.append("far-offset push-only starts insert")

# ------------------------------------------------------------------ C coverage
DB = [(-1e9, 1, "<1"), (1, 5, "1-5"), (5, 13.5, "5-13.5"), (13.5, 16.5, "13.5-16.5 (pins at the face)"), (16.5, 25, "16.5-25"), (25, 1e9, ">25")]
LB = [(0, 0.5, "<0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, 8, "4-8"), (8, 1e9, ">8")]


def hist(rel):
    d = diagnosis(rel)
    dep, lat = d["depth_left_m"] * 1000, d["lateral_m"] * 1000
    h = np.zeros((len(DB), len(LB)))
    for i, (a, b, _) in enumerate(DB):
        for j, (c, e, _) in enumerate(LB):
            h[i, j] = ((dep >= a) & (dep < b) & (lat >= c) & (lat < e)).sum()
    return h


entries = [e for e in index.load(args.data) if e["n_images"]]
hx = np.zeros((len(DB), len(LB)))
for e in entries:
    if e["split"] != "train":
        continue
    with np.load(e["path"]) as f:
        hx += hist(f["eval_rel"][f["img_step"]])
C = {"depth_bins_mm": [b[2] for b in DB], "lateral_bins_mm": [b[2] for b in LB],
     "exploration_train_pct": (100 * hx / hx.sum()).round(2).tolist()}
if args.policy_episodes:
    hp = np.zeros_like(hx)
    for f in glob.glob(f"{args.policy_episodes}/*.npz"):
        with np.load(f) as z:
            hp += hist(z["eval_rel"][z["img_step"]])
    P, X = 100 * hp / hp.sum(), 100 * hx / hx.sum()
    C["policy_rollouts_pct"] = P.round(2).tolist()
    C["policy_cells_hardly_explored"] = [{"depth_left_mm": DB[i][2], "lateral_mm": LB[j][2], "policy_pct": round(float(P[i, j]), 2),
                                          "exploration_pct": round(float(X[i, j]), 3)}
                                         for i in range(len(DB)) for j in range(len(LB)) if P[i, j] >= 1.0 and X[i, j] < 0.2]
rep["C_coverage"] = C

# ------------------------------------------------------------------ D leakage
X, Y, G, O = [], [], [], []
rep["A_collection"]["trajectories_without_port_axes"] = sum(e["R_P"] is None for e in entries)
for e in entries:
    if e["R_P"] is None:
        continue
    with np.load(e["path"]) as f, np.load(e["path"].replace(args.data, args.ee)) as g:
        hand, cmd, rel = f["hand"].astype(np.float64), g["hand_cmd"].astype(np.float64), f["eval_rel"]
    if len(cmd) < 10:
        continue
    idx = np.arange(10)                    # the first 0.5 s of commands: before any contact can stop them
    X.append(pose_feature(cmd[idx], hand[0], np.asarray(e["R_P"], np.float64)).reshape(-1))
    q = rel[0, 3:7].astype(np.float64)
    w = np.clip(abs(q[0]) / np.linalg.norm(q), 0, 1)
    Y.append([rel[0, 1] * 1000, rel[0, 2] * 1000, math.degrees(2 * math.acos(w))])
    G.append(e["grasp"])
    O.append(e.get("outcome") == "inserted")
X, Y, O = np.array(X), np.array(Y), np.array(O)
grasps = sorted(set(G))
rng = np.random.default_rng(0)
test_g = set(rng.choice(grasps, max(1, len(grasps) // 5), replace=False))
te = np.array([g in test_g for g in G])
mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
Xn = np.c_[(X - mu) / sd, np.ones(len(X))]


def ridge(y, lam=10.0):
    A = Xn[~te]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[~te])
    return Xn[te] @ w


Dr = {"trajectories": int(len(X)), "held_out_trajectories": int(te.sum())}
for k, name in enumerate(("start_lateral_y_mm", "start_lateral_z_mm", "start_tilt_deg")):
    pred = ridge(Y[:, k])
    r2 = 1 - ((Y[te, k] - pred) ** 2).sum() / ((Y[te, k] - Y[te, k].mean()) ** 2).sum()
    Dr[f"r2_{name}_from_commands"] = round(float(r2), 4)
    if r2 > 0.1:
        fail.append(f"commands predict {name} (R2 {r2:.2f})")
sc = ridge(O.astype(float))
pos, neg = sc[O[te]], sc[~O[te]]
auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else float("nan")
Dr["auc_outcome_from_first_commands"] = round(auc, 4)
if auc == auc and abs(auc - 0.5) > 0.15:
    fail.append(f"the first commands predict the outcome (AUC {auc:.2f})")
rep["D_no_truth_in_commands"] = Dr
rep["gate"] = "FAIL" if fail else "PASS"
rep["failed"] = fail
json.dump(rep, open(args.out, "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("A_collection", "B_push_only_by_start_offset", "D_no_truth_in_commands", "gate", "failed")}, indent=1))
if "policy_cells_hardly_explored" in C:
    print("policy cells hardly explored:", json.dumps(C["policy_cells_hardly_explored"]))
sys.exit(1 if fail else 0)
