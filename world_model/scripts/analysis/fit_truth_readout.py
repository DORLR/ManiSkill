"""The same readout as fit_lateral_readout.py, but fitted on the simulator's lateral error of the plug (socket frame),
for comparison only: the joint-only label is the hand's offset from H*, which is not the pins' offset whenever the
hand is rotated against H* (lever arm of the plug)."""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/storage2/qingxin/code/ManiSkill/world_model/scripts/analysis")
from fit_lateral_readout import branch_list, direction
D = "/mnt/storage3/qingxin/runs/branch_wm_v1s4/"
tr, te = dict(np.load(D + "lateral_probe/points_train.npz")), dict(np.load(D + "lateral_probe/points_ood_val.npz"))
btr = branch_list("/mnt/storage3/qingxin/data/wm_v1b", "train", 4)
TJ = (2, 3, 4)
def xy(pts, tags):
    X, Y, G = [], [], []
    for s in tags:
        y = pts["truth"][:, TJ[s], 1:].astype(np.float64)
        good = pts["ok"][:, s] & np.isfinite(y).all(1) & (np.linalg.norm(y, axis=1) < 15)
        X.append(pts["z"][good, s].astype(np.float64)); Y.append(y[good]); G.append(pts["branch"][good])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)
Xa, Ya, Ga = xy(tr, (1, 2))
grasp = np.array([btr[i]["grasp"] for i in Ga])
import zlib
fold = np.array([zlib.crc32(g.encode()) % 5 for g in grasp])
mu, sd, ym = Xa.mean(0), Xa.std(0) + 1e-6, Ya.mean(0)
A = (Xa - mu) / sd
cv = {}
for lam in (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0):
    e = []
    for f in range(5):
        m = fold != f
        W = np.linalg.solve(A[m].T @ A[m] + lam * np.eye(A.shape[1]), A[m].T @ (Ya[m] - Ya[m].mean(0)))
        e.append(direction(A[~m] @ W + Ya[m].mean(0), Ya[~m])["within_45deg"])
    cv[lam] = float(np.mean(e))
lam = max(cv, key=cv.get)
W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ya - ym))
rep = {"cv_within_45deg_by_lambda": cv, "lambda": lam, "train_points": int(len(Xa))}
for s, name in enumerate(("face-5", "face-2", "face+0")):
    Xb, Yb, _ = xy(te, (s,))
    rep["held_out_" + name] = {"against_truth": direction(((Xb - mu) / sd) @ W + ym, Yb), "constant_guess": direction(np.tile(ym, (len(Yb), 1)), Yb)}
np.savez(D + "lateral_readout_truth.npz", W=W, mu=mu, sd=sd, ymean=ym, lam=lam)
open(D + "lateral_readout_truth.json", "w").write(json.dumps(rep, indent=1))
print(json.dumps(rep))
