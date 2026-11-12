"""Checks of the contact set (collect/explore.py, --contact) before it is used for training.

    python world_model/scripts/analysis/check_contact_set.py --data $Q/data/wm_v1c --ee $Q/data/wm_v1c_ee --out $Q/runs/checks_v5/contact/report.json
A  collection: outcomes, how the push and the contact phase ended, lengths, inverse-kinematics misses
B  the contact phase: how long; how far the hand went across the axis from where the phase began; where the plug
   truly was (offset from the hole axis; share of the phase's control steps); how often the plug went in or came
   out during the phase
C  what the phase adds: decisions (8 commands, 0.4 s, starting on the decision clock) by the lead of the last
   command over the measured hand along the axis, and whether the hand then really advanced (> 0.1 mm) or stayed
   (< 0.05 mm): the windows the first exploration set hardly has (held hand with a command that keeps leading)
D  no truth in the commands: the true start offset and the outcome predicted from the commanded hand path of the
   first 0.5 s (ridge regression, grasps held out), as for the first set. R^2 must be ~0 and the AUC ~0.5.
Exit code 1 if D fails or the phase wanders (90th percentile of the hand's excursion above 6 mm).
"""
import argparse
import collections
import json
import math
import sys

import numpy as np

from wm.data import index, schema
from wm.data.conditions import pose_feature
from wm.data.outcome import diagnosis, inserted

ap = argparse.ArgumentParser()
for a in ("--data", "--ee", "--out"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
C_PHASE = schema.PHASES.index("explore_contact")
rep, fail = {}, []
entries = [e for e in index.load(args.data) if e["n_images"]]
A = collections.defaultdict(collections.Counter)
steps, csteps, exc, lat0, lat1, trans, missed = [], [], [], [], [], collections.Counter(), 0
lat_bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
by_lat = np.zeros(len(lat_bins))
lead_bins = [(-9, 0), (0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9)]
dec = np.zeros((len(lead_bins), 3))                      # stayed / in between / advanced
X, Y, G, O = [], [], [], []
for e in entries:
    with np.load(e["path"]) as f, np.load(e["path"].replace(args.data, args.ee)) as g:
        meta = json.loads(str(f["meta"]))
        ph, hand, rel, cmd = f["phase"], f["hand"].astype(np.float64), f["eval_rel"], g["hand_cmd"].astype(np.float64)
    A["outcome"][meta["outcome"]] += 1
    A["final_push"][meta["stopped"]] += 1
    A["contact_phase"][meta.get("contact_stopped")] += 1
    steps.append(len(ph))
    c = np.flatnonzero(ph == C_PHASE)
    if len(c):
        a = np.asarray(meta["axis"], np.float64)
        a /= np.linalg.norm(a)
        pos = hand[:, :3, 3]
        d = pos[c[0]:] - pos[c[0]]
        exc.append(np.linalg.norm(d - np.outer(d @ a, a), axis=1).max() * 1000)
        dg, ins = diagnosis(rel), inserted(rel)
        lat = dg["lateral_m"] * 1000
        lat0.append(lat[c[0]]), lat1.append(lat[-1]), csteps.append(len(c))
        trans[f"{'in' if ins[c[0]] else 'out'} -> {'in' if ins[-1] else 'out'}"] += 1
        missed += sum(x.get("ik_missed_steps", 0) for x in meta["contact"])
        for k, (lo, hi) in enumerate(lat_bins):
            by_lat[k] += ((lat[c] >= lo) & (lat[c] < hi)).sum()
        dep = pos @ a
        for s in range(int(np.ceil(c[0] / 8) * 8), len(ph) - 8 + 1, 8):
            lead = (cmd[s + 7, :3, 3] - pos[s]) @ a * 1000
            moved = (dep[s + 8] - dep[s]) * 1000
            i = next(k for k, (lo, hi) in enumerate(lead_bins) if lo <= lead < hi)
            dec[i, 0 if abs(moved) < 0.05 else 2 if moved > 0.1 else 1] += 1
    if e["R_P"] is not None and len(cmd) >= 10:
        X.append(pose_feature(cmd[np.arange(10)], hand[0], np.asarray(e["R_P"], np.float64)).reshape(-1))
        q = rel[0, 3:7].astype(np.float64)
        Y.append([rel[0, 1] * 1000, rel[0, 2] * 1000, math.degrees(2 * math.acos(np.clip(abs(q[0]) / np.linalg.norm(q), 0, 1)))])
        G.append(e["grasp"]), O.append(e.get("outcome") == "inserted")
pct = lambda v, q: round(float(np.percentile(v, q)), 2)
rep["A_collection"] = {"trajectories": len(entries), "by_split": dict(collections.Counter(e["split"] for e in entries)),
                       **{k: dict(v) for k, v in A.items()}, "steps_median": pct(steps, 50), "steps_max": int(max(steps)),
                       "ik_missed_steps_pct_of_contact_steps": round(100 * missed / max(sum(csteps), 1), 2)}
rep["B_contact_phase"] = {
    "trajectories_with_phase": len(csteps), "seconds_median": round(pct(csteps, 50) / 20, 1), "seconds_total_hours": round(sum(csteps) / 20 / 3600, 2),
    "hand_excursion_across_axis_mm": {"median": pct(exc, 50), "p90": pct(exc, 90), "max": pct(exc, 100)},
    "true_offset_from_axis_mm": {"at_phase_start_median": pct(lat0, 50), "at_the_end_median": pct(lat1, 50), "at_the_end_p90": pct(lat1, 90)},
    "phase_steps_by_true_offset_pct": {f"{lo:g}-{hi:g} mm" if hi < 1e9 else f">{lo:g} mm": round(float(100 * v / by_lat.sum()), 1) for (lo, hi), v in zip(lat_bins, by_lat)},
    "inserted_at_phase_start_to_end": dict(trans)}
rep["C_decisions_by_lead_along_axis"] = {
    (f"{lo:g}-{hi:g} mm" if lo >= 0 and hi < 9 else ("backing off" if lo < 0 else f">{lo:g} mm")):
        {"decisions": int(r.sum()), "hand_stayed": int(r[0]), "hand_advanced": int(r[2])} for (lo, hi), r in zip(lead_bins, dec)}
if rep["B_contact_phase"]["hand_excursion_across_axis_mm"]["p90"] > 6.0:
    fail.append("the contact phase wanders: 90th percentile of the hand's excursion across the axis above 6 mm")
X, Y, O = np.array(X), np.array(Y), np.array(O)
grasps = sorted(set(G))
test_g = set(np.random.default_rng(0).choice(grasps, max(1, len(grasps) // 5), replace=False))
te = np.array([g in test_g for g in G])
mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
Xn = np.c_[(X - mu) / sd, np.ones(len(X))]


def ridge(y, lam=10.0):
    Atr = Xn[~te]
    return Xn[te] @ np.linalg.solve(Atr.T @ Atr + lam * np.eye(Atr.shape[1]), Atr.T @ y[~te])


D = {"trajectories": int(len(X)), "held_out_trajectories": int(te.sum())}
for k, name in enumerate(("start_lateral_y_mm", "start_lateral_z_mm", "start_tilt_deg")):
    r2 = 1 - ((Y[te, k] - ridge(Y[:, k])) ** 2).sum() / ((Y[te, k] - Y[te, k].mean()) ** 2).sum()
    D[f"r2_{name}_from_commands"] = round(float(r2), 4)
    if r2 > 0.1:
        fail.append(f"commands predict {name} (R2 {r2:.2f})")
sc = ridge(O.astype(float))
pos_, neg_ = sc[O[te]], sc[~O[te]]
auc = float((pos_[:, None] > neg_[None]).mean() + 0.5 * (pos_[:, None] == neg_[None]).mean()) if len(pos_) and len(neg_) else float("nan")
D["auc_outcome_from_first_commands"] = round(auc, 4)
if auc == auc and abs(auc - 0.5) > 0.15:
    fail.append(f"the first commands predict the outcome (AUC {auc:.2f})")
rep["D_no_truth_in_commands"], rep["gate"], rep["failed"] = D, ("FAIL" if fail else "PASS"), fail
json.dump(rep, open(args.out, "w"), indent=1)
print(json.dumps(rep, indent=1))
sys.exit(1 if fail else 0)
