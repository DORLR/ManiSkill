"""Aggregate closed-loop results into the pre-registered table.

python scripts/analyze.py --eval results/eval --runs runs --out results/summary.json

Per cell (fold role x condition x method):
  push_success     fraction of episodes in which the fixed straight push
                   inserted from the end state (operational ID membership)
  in_id            fraction of end states within r_ID of the ID set, r_ID =
                   95th percentile of nearest-neighbour distances inside M
  err_t_mm, err_r_deg  median end-state distance to the nearest ID state
  mmd2, mmd_p      unbiased MMD^2 between end states and the ID set, with a
                   permutation p-value (Gretton et al. 2012); start states
                   are reported the same way for contrast
95% CIs: Wilson for rates; paired difference field - none by bootstrap over
grasps.
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


PIN_LENGTH_MM = {"A": 16.0, "B": 18.0, "C": 20.0, "D": 14.0}  # full pin length per port (docs)


def inserted(e, port):
    """Insertion depth reached, relative to H_pre: start depth (t_x before the
    push, mm) + hand advance >= pin length + 3 mm - 1 mm. The raw push flag only
    checked the advance, which is also met when the push starts so far back that
    the pins never reach the board (bug found 2026-11-05)."""
    return e["u_final_true"][0] + e["advanced_mm"] >= PIN_LENGTH_MM[port] + 3.0 - 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mmd2_unbiased(X, Y, bw):
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * bw * bw))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    return ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
            - 2 * kxy.mean())


def mmd_test(X, Y, rng, n_perm=500, max_y=400):
    if len(X) < 3:
        return float("nan"), float("nan")
    Y = Y[rng.choice(len(Y), min(len(Y), max_y), replace=False)]
    Z = np.concatenate([X, Y])
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    bw = float(np.median(d[d > 0]))
    stat = mmd2_unbiased(X, Y, bw)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        cnt += mmd2_unbiased(Z[p[:len(X)]], Z[p[len(X):]], bw) >= stat
    return float(stat), float((cnt + 1) / (n_perm + 1))


def nn_dist(U, M):
    return np.sqrt(((U[:, None, :] - M[None]) ** 2).sum(-1)).min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/eval")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    cells = defaultdict(list)
    grasp_pairs = defaultdict(list)
    for f in glob.glob(f"{args.eval}/fold_*/*.json"):
        r = json.load(open(f))
        if not r.get("expert_ok"):
            continue
        by_m = defaultdict(list)
        for e in r["episodes"]:
            if "u_final_true" not in e:
                continue
            e["inserted"] = inserted(e, r["port"])
            cells[(r["fold"], r["role"], r["condition"], e["method"])].append(e)
            by_m[e["method"]].append(inserted(e, r["port"]))
        if by_m.get("field") and by_m.get("none"):
            grasp_pairs[(r["role"], r["condition"])].append(np.mean(by_m["field"]) - np.mean(by_m["none"]))
    M_by_fold = {}
    for fold in "ABCD":
        try:
            M = np.load(f"{args.runs}/fold_{fold}/id_set.npy")
        except FileNotFoundError:
            continue
        nn = np.sort(np.sqrt(((M[:, None] - M[None]) ** 2).sum(-1)), 1)[:, 1]
        M_by_fold[fold] = (M, float(np.percentile(nn, 95)))
    # pool folds for the table: role x condition x method
    table = defaultdict(lambda: defaultdict(list))
    for (fold, role, cond, method), eps in cells.items():
        table[(role, cond, method)]["eps"] += [(fold, e) for e in eps]
    out = {}
    for (role, cond, method), d in sorted(table.items()):
        eps = d["eps"]
        k = sum(e["inserted"] for _, e in eps)
        p, lo, hi = wilson(k, len(eps))
        U = np.array([e["u_final_true"] for _, e in eps])
        U0 = np.array([e["u0_true"] for _, e in eps])
        in_id, dist_t, dist_r, mmd, mmd0 = [], [], [], [], []
        for fold in sorted({f for f, _ in eps}):
            if fold not in M_by_fold:
                continue
            M, r_id = M_by_fold[fold]
            Uf = np.array([e["u_final_true"] for f, e in eps if f == fold])
            U0f = np.array([e["u0_true"] for f, e in eps if f == fold])
            dd = nn_dist(Uf, M)
            in_id += (dd <= r_id).tolist()
            j = np.sqrt(((Uf[:, None] - M[None]) ** 2).sum(-1)).argmin(1)
            dist_t += np.linalg.norm(Uf[:, :3] - M[j, :3], axis=1).tolist()
            dist_r += np.linalg.norm(Uf[:, 3:] - M[j, 3:], axis=1).tolist()
            mmd.append(mmd_test(Uf, M, rng))
            mmd0.append(mmd_test(U0f, M, rng))
        out[f"{role}|{cond}|{method}"] = {
            "n": len(eps), "push_success": p, "ci": [lo, hi],
            "in_id": float(np.mean(in_id)) if in_id else None,
            "end_dist_t_med_mm": float(np.median(dist_t)) if dist_t else None,
            "end_dist_r_med_deg": float(np.median(dist_r)) if dist_r else None,
            "start_norm_med_sigma": float(np.median(np.linalg.norm(U0, axis=1))),
            "end_norm_med_sigma": float(np.median(np.linalg.norm(U, axis=1))),
            "mmd2_end_per_fold": [m[0] for m in mmd], "mmd_p_end_per_fold": [m[1] for m in mmd],
            "mmd2_start_per_fold": [m[0] for m in mmd0], "mmd_p_start_per_fold": [m[1] for m in mmd0],
        }
    for key, diffs in grasp_pairs.items():
        diffs = np.array(diffs)
        boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)]
        out[f"{key[0]}|{key[1]}|field_minus_none"] = {
            "n_grasps": len(diffs), "mean": float(diffs.mean()),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"{'role':7}{'condition':14}{'method':10}{'n':>5}{'push':>7}{'95% CI':>15}{'inID':>7}{'end mm':>8}{'end deg':>8}")
    for key, v in out.items():
        role, cond, method = key.split("|")
        if method == "field_minus_none":
            continue
        print(f"{role:7}{cond:14}{method:10}{v['n']:5d}{v['push_success']:7.2f}"
              f"   [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]{(v['in_id'] or 0):7.2f}"
              f"{(v['end_dist_t_med_mm'] or 0):8.2f}{(v['end_dist_r_med_deg'] or 0):8.2f}")


if __name__ == "__main__":
    main()
