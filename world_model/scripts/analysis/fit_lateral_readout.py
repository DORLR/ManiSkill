"""Fit the direction readout: latent z -> lateral offset of the hand from the reference inserted pose H*, in the frame
of the port axis (mm). The label comes from joints only (FK hand position minus H*, across the axis), as the depth
label does; the simulator's truth is used only to report how well the label agrees with it. Ridge regression on the
latents saved by probe_lateral_direction.py (training split), the penalty chosen by cross-validation over training
grasps; evaluated on the held-out split.

    python world_model/scripts/analysis/fit_lateral_readout.py --points <run>/lateral_probe --data <wm_v1b> --out <run>/lateral_readout.npz
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from wm.tdmpc import data

OFFS = (-5, -2, 0)            # the scored moments of probe_lateral_direction.py: face-5, face-2, face+0
USE = (1, 2)                  # fit on face-2 and face+0 (the readout is used right at the face)


def fk_lateral(path):
    from wm.collect.explore import frame
    with np.load(path) as z:
        a = data.decision_arrays(z)
        if a is None or a["face_k"] < 0:
            return None
        meta = a["meta"]
        F = frame(np.asarray(meta["axis"], np.float64))
        H = np.asarray(meta["H_star"], np.float64)
        hand, dec = z["hand"], z["dec_step"]
        out = np.full((len(OFFS), 2), np.nan)
        for j, o in enumerate(OFFS):
            k = a["face_k"] + o
            if 0 <= k < a["n"]:
                out[j] = (F.T @ (hand[dec[k]][:3, 3].astype(np.float64) - H[:3, 3]))[1:] * 1000.0
    return out


def branch_list(root, split, every):
    bs = data.list_branches(root, split)
    starts = set(sorted({(b["grasp"], b["start"]) for b in bs})[::every])
    return [b for b in bs if (b["grasp"], b["start"]) in starts]


def labels(points, bs, pool):
    lab = pool.map(fk_lateral, [bs[i]["path"] for i in points["branch"]], chunksize=16)
    return np.stack([x if x is not None else np.full((len(OFFS), 2), np.nan) for x in lab])


def direction(P, Y, min_mm=0.5):
    big = np.linalg.norm(Y, axis=1) > min_mm
    cos = (P[big] * Y[big]).sum(1) / (np.linalg.norm(P[big], axis=1) * np.linalg.norm(Y[big], axis=1) + 1e-9)
    return {"n": int(big.sum()), "within_90deg": float((cos > 0).mean()), "within_45deg": float((cos > 0.707).mean()),
            "within_30deg": float((cos > 0.866).mean()), "error_mm_median": float(np.median(np.linalg.norm(P - Y, axis=1)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-every", type=int, default=4)
    args = ap.parse_args()
    tr, te = dict(np.load(Path(args.points) / "points_train.npz")), dict(np.load(Path(args.points) / "points_ood_val.npz"))
    btr, bte = branch_list(args.data, "train", args.train_every), branch_list(args.data, "ood_val", 1)
    with mp.get_context("spawn").Pool(8) as pool:
        ytr, yte = labels(tr, btr, pool), labels(te, bte, pool)
    rep = {}
    truth = te["truth"][:, [2, 3, 4], 1:]                            # face-5, face-2, face+0 of the truth array
    ok = np.isfinite(yte).all(2) & np.isfinite(truth).all(2)
    dn = np.abs(np.linalg.norm(yte, axis=2) - np.linalg.norm(truth, axis=2))[ok]
    rep["label_fk_vs_truth_size_difference_mm_pct_50_90"] = np.percentile(dn, [50, 90]).tolist()

    def xy(pts, y, tags):
        X, Y, G = [], [], []
        for s in tags:
            good = pts["ok"][:, s] & np.isfinite(y[:, s]).all(1) & (np.linalg.norm(y[:, s], axis=1) < 15)
            X.append(pts["z"][good, s].astype(np.float64)); Y.append(y[good, s]); G.append(pts["branch"][good])
        return np.concatenate(X), np.concatenate(Y), np.concatenate(G)

    Xa, Ya, Ga = xy(tr, ytr, USE)
    grasp = np.array([btr[i]["grasp"] for i in Ga])
    mu, sd, ym = Xa.mean(0), Xa.std(0) + 1e-6, Ya.mean(0)
    A = (Xa - mu) / sd
    fold = np.array([hash(g) % 5 for g in grasp])
    cv = {}
    for lam in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
        errs = []
        for f in range(5):
            m = fold != f
            W = np.linalg.solve(A[m].T @ A[m] + lam * np.eye(A.shape[1]), A[m].T @ (Ya[m] - Ya[m].mean(0)))
            errs.append(direction(A[~m] @ W + Ya[m].mean(0), Ya[~m])["within_45deg"])
        cv[lam] = float(np.mean(errs))
    lam = max(cv, key=cv.get)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ya - ym))
    rep["cv_within_45deg_by_lambda"] = cv
    rep["lambda"] = lam
    rep["train_points"] = int(len(Xa))
    rep["train_label_mean_mm"] = ym.tolist()
    for s, name in enumerate(("face-5", "face-2", "face+0")):
        Xb, Yb, _ = xy(te, yte, (s,))
        P = ((Xb - mu) / sd) @ W + ym
        rep[f"held_out_{name}"] = {"against_fk_label": direction(P, Yb), "constant_guess": direction(np.tile(ym, (len(Yb), 1)), Yb)}
    np.savez(args.out, W=W, mu=mu, sd=sd, ymean=ym, lam=lam)
    Path(str(args.out).replace(".npz", ".json")).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
