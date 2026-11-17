"""The two "goal" checks on the features saved by check_goal_reuse.py (2026-11-17).

Check 1  can a goal taken from another grasp serve as the target? Score "is this frame seated" by the distance of
         its feature to a goal = the end frame of ANOTHER FK-seated attempt: of the same grasp (own), of the nearest
         training grasp (nearest), of a random training grasp (random). Positives: end frames of FK-seated attempts;
         negatives: the same attempts' frames with 1.25-4 mm (near) and 4-8 mm (half) of FK depth left, and end frames
         of blocked attempts (more than 12 mm left). Three features: the picture's feature (no joints), the latent
         (3 pictures + joints), the 40x40 depth picture.
Check 1b a learned "done" readout with no goal at all (fitted on training-split frames).
Check 2  does a goal input add information to a value-like head? Heads on the latent at the board face and 0.8 s
         before, predicting "this attempt reaches depth" (training label from FK; scored against the simulator's
         relaxed success), with: no goal / own goal / random goal / nearest-training-grasp goal, and the own-goal head
         fed a foreign goal.
Groups (the user's new held-out definition): interior = pitch in (-42,-38) or position in (-21,-19) mm and not in an
outer strip; shape = in-band grasp on an unseen shape; extrap = an outer strip; train, dev = the training split.

    python world_model/scripts/analysis/analyse_goal_reuse.py --data <wm_v1b> --out <dir with points_*.npz>
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wm.tdmpc import report

PRE, FACE, HALF, NEAR, END = 0, 1, 2, 3, 4
W_POS, W_PITCH = 6.0, 1.2            # picture shift per mm of grasp position and per degree of pitch (geometry, 320 px)
SEEDS, RANDOM_DRAWS = 5, 5


def auc(score, label):
    label = np.asarray(label, bool)
    if len(label) < 4 or label.all() or not label.any():
        return None
    return float(report.auc(np.asarray(score, np.float64), label))


def load(out, root):
    meta = {}
    for f in glob.glob(f"{root}/*/*/grasp.json"):
        g = json.loads(Path(f).read_text())
        if g.get("error"):
            continue
        for s in g["starts"]:
            for b in s.get("branches", []):
                ok = None not in (b.get("depth_left_mm"), b.get("lateral_mm"), b.get("angle_deg"))
                relaxed = bool(ok and b["depth_left_mm"] < 1.25 and b["angle_deg"] < 4.0 and b["lateral_mm"] < 2.0)
                meta[f"{g['port']}/{g['grasp']}/s{s['idx']:02d}_b{b['idx']}"] = (g["grasp"], g["pitch_deg"], g["grasp_pos_m"] * 1000, s["idx"], bool(b["seated_fk"]), relaxed)
    parts, split = defaultdict(list), []
    for sp in ("train", "dev", "ood_val"):
        for f in sorted(glob.glob(f"{out}/points_{sp}_*.npz")):
            r = np.load(f)
            for k in r.files:
                parts[k].append(r[k])
            split += [sp] * len(r["rel"])
    A = {k: np.concatenate(v) for k, v in parts.items()}
    m = [meta[r] for r in A["rel"]]
    A["split"] = np.array(split)
    A["grasp"] = np.array([x[0] for x in m])
    A["pitch"], A["pos"] = np.array([x[1] for x in m]), np.array([x[2] for x in m])
    A["start"], A["seated"], A["relaxed"] = np.array([x[3] for x in m]), np.array([x[4] for x in m]), np.array([x[5] for x in m])
    ext = (A["pitch"] < -55) | (A["pitch"] > -25) | (A["pos"] < -28) | (A["pos"] > -12)
    inter = ((A["pitch"] > -42) & (A["pitch"] < -38)) | ((A["pos"] > -21) & (A["pos"] < -19))
    A["group"] = np.where(A["split"] != "ood_val", A["split"], np.where(ext, "extrap", np.where(inter, "interior", "shape")))
    return A


def goals(A, rng):
    """Row indices of the goal frames for every row: own (-1 if none), nearest training grasp, random training grasps."""
    N = len(A["rel"])
    good = A["seated"] & A["ok"][:, END]
    by = defaultdict(list)
    for r in np.nonzero(good)[0]:
        by[A["grasp"][r]].append(r)
    bank = sorted({A["grasp"][r] for r in np.nonzero(good & (A["split"] == "train"))[0]})
    first = np.array([by[g][0] for g in bank])
    bp, bx = A["pitch"][first], A["pos"][first]
    own, near, rand = np.full(N, -1), np.zeros(N, int), np.zeros((N, RANDOM_DRAWS), int)
    index = {g: i for i, g in enumerate(bank)}
    for r in range(N):
        g = A["grasp"][r]
        c = [x for x in by.get(g, []) if x != r]
        if c:
            other = [x for x in c if A["start"][x] != A["start"][r]]
            own[r] = (other or c)[0]
        d = W_POS * np.abs(bx - A["pos"][r]) + W_PITCH * np.abs(bp - A["pitch"][r])
        if g in index:
            d[index[g]] = np.inf
        near[r] = first[int(np.argmin(d))]
        p = np.isfinite(d).astype(float)
        rand[r] = first[rng.choice(len(bank), RANDOM_DRAWS, p=p / p.sum())]
    count = {g: len(v) for g, v in by.items()}
    two = np.array([count.get(g, 0) >= 2 for g in A["grasp"]])
    return own, near, rand, two, {"bank_grasps": len(bank), "grasps_with_two_seated": int(sum(v >= 2 for v in count.values()))}


def dist(F, kind, rows, slot, goal_rows):
    a, b = F[rows, slot], F[goal_rows, END]
    return np.abs(a - b).mean(1) if kind == "depth40" else np.linalg.norm(a - b, axis=1)


def check1(A, F, own, near, rand):
    out = {}
    for grp in ("train", "dev", "interior", "shape", "extrap"):
        g = A["group"] == grp
        S = np.nonzero(g & A["seated"] & A["ok"][:, END] & (own >= 0))[0]
        B = np.nonzero(g & A["ok"][:, END] & (A["left_fk"][:, END] > 12) & (own >= 0))[0]
        if grp == "train":
            S, B = S[::4], B[::4]
        e = {"seated_attempts": int(len(S)), "blocked_attempts": int(len(B))}
        for kind, X in F.items():
            for name, pick in (("own", lambda r, j: own[r]), ("nearest", lambda r, j: near[r]), ("random", lambda r, j: rand[r, j])):
                res = defaultdict(list)
                for j in range(RANDOM_DRAWS if name == "random" else 1):
                    d_end = dist(X, kind, S, END, pick(S, j))
                    for neg, tag in ((NEAR, "near_1.25-4mm"), (HALF, "half_4-8mm")):
                        m = A["ok"][S, neg]
                        d_neg = dist(X, kind, S[m], neg, pick(S[m], j))
                        res[f"auc_vs_{tag}"].append(auc(-np.concatenate([d_end[m], d_neg]), np.r_[np.ones(m.sum()), np.zeros(m.sum())]))
                        res[f"closer_than_own_{tag}"].append(float(np.mean(d_end[m] < d_neg)) if m.any() else None)
                    if len(B):
                        d_b = dist(X, kind, B, END, pick(B, j))
                        res["auc_vs_blocked_end"].append(auc(-np.concatenate([d_end, d_b]), np.r_[np.ones(len(S)), np.zeros(len(B))]))
                    res["distance_seated_median"].append(float(np.median(d_end)) if len(S) else None)
                e[f"{kind}/{name}"] = {k: (None if any(x is None for x in v) else round(float(np.mean(v)), 4)) for k, v in res.items()}
        out[grp] = e
    return out


def fit(Xa, ya, Xb, seed, epochs=30, resample=None, dev="cuda"):
    """Small MLP, logistic loss. `resample(epoch)` may return a fresh training matrix (random goals). -> scores on Xb."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xa.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    y = torch.from_numpy(ya.astype(np.float32)).to(dev)
    w = torch.tensor(float((1 - ya.mean()) / max(ya.mean(), 1e-3)), device=dev)
    X = torch.from_numpy(Xa).to(dev)
    for ep in range(epochs):
        if resample is not None and ep:
            X = torch.from_numpy(resample(ep)).to(dev)
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 512):
            k = perm[i:i + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[k])[:, 0], y[k], pos_weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return [net(torch.from_numpy(x).to(dev))[:, 0].float().cpu().numpy() for x in Xb]


def check1b(A, F):
    """A "done" readout with no goal: end frames of attempts that reached depth by FK against near / half / blocked / face."""
    reached = A["ok"][:, END] & (A["left_fk"][:, END] < 1.25)
    blocked = A["ok"][:, END] & (A["left_fk"][:, END] > 12)
    out = {}
    for kind in ("picture", "latent"):
        X = F[kind]
        tr = A["split"] == "train"
        sets = [(tr & reached, END, 1), (tr & A["ok"][:, NEAR], NEAR, 0), (tr & A["ok"][:, HALF], HALF, 0), (tr & blocked, END, 0), (tr & A["ok"][:, FACE], FACE, 0)]
        Xa = np.concatenate([X[m, s] for m, s, _ in sets]); ya = np.concatenate([np.full(m.sum(), y) for m, _, y in sets])
        e = {}
        for grp in ("dev", "interior", "shape", "extrap"):
            g = A["group"] == grp
            parts = {"reached_end": X[g & reached, END], "near": X[g & A["ok"][:, NEAR], NEAR], "half": X[g & A["ok"][:, HALF], HALF],
                     "blocked_end": X[g & blocked, END], "all_end": X[g & A["ok"][:, END], END]}
            sc = defaultdict(list)
            for seed in range(3):
                for k, s in zip(parts, fit(Xa, ya, list(parts.values()), seed)):
                    sc[k].append(s)
            sc = {k: np.mean(v, 0) for k, v in sc.items()}
            pos = sc["reached_end"]
            thr = np.percentile(pos, 5) if len(pos) else np.nan
            rec = {"reached": int(len(pos))}
            for k in ("near", "half", "blocked_end"):
                rec[f"auc_vs_{k}"] = auc(np.r_[pos, sc[k]], np.r_[np.ones(len(pos)), np.zeros(len(sc[k]))])
                rec[f"false_done_share_{k}_at_95pct_recall"] = float(np.mean(sc[k] > thr)) if len(sc[k]) else None
            rec["auc_all_ends_vs_truth_relaxed"] = auc(sc["all_end"], A["relaxed"][g & A["ok"][:, END]])
            e[grp] = rec
        out[kind] = e
    return out


def check2(A, F, own, near, rand, two, rng):
    Z, P = F["latent"], F["picture"]
    bank_rows = np.unique(rand)
    out = {}
    for point, tag in ((FACE, "at_face"), (PRE, "0.8s_before_face")):
        ok = A["ok"][:, point] & two
        tr = np.nonzero(ok & (A["split"] == "train"))[0]
        ya = (A["left_fk"][tr, END] < 1.25)
        z = lambda rows: Z[rows, point]
        gp = lambda rows: P[rows, END]
        cat = lambda rows, grow: np.concatenate([z(rows), gp(grow)], 1)
        ev = {grp: np.nonzero(ok & (A["group"] == grp))[0] for grp in ("dev", "interior", "shape", "extrap")}
        ev = {k: v for k, v in ev.items() if len(v) >= 20}
        runs = defaultdict(lambda: defaultdict(list))
        for seed in range(SEEDS):
            rows = list(ev.values())
            for name, s in zip(ev, fit(z(tr), ya, [z(r) for r in rows], seed)):
                runs["no_goal"][name].append(s)
            tests = [cat(r, own[r]) for r in rows] + [cat(r, near[r]) for r in rows] + [cat(r, rand[r, 0]) for r in rows]
            s = fit(cat(tr, own[tr]), ya, tests, seed)
            for i, name in enumerate(ev):
                runs["own_goal"][name].append(s[i]); runs["own_goal_fed_nearest"][name].append(s[len(ev) + i]); runs["own_goal_fed_random"][name].append(s[2 * len(ev) + i])
            for name, s in zip(ev, fit(cat(tr, near[tr]), ya, [cat(r, near[r]) for r in rows], seed)):
                runs["nearest_goal"][name].append(s)
            draw = lambda ep: cat(tr, bank_rows[rng.integers(0, len(bank_rows), len(tr))])
            for name, s in zip(ev, fit(draw(0), ya, [cat(r, rand[r, 0]) for r in rows], seed, resample=draw)):
                runs["random_goal"][name].append(s)
            for name, s in zip(ev, fit(np.concatenate([z(tr), Z[own[tr], END]], 1), ya, [np.concatenate([z(r), Z[own[r], END]], 1) for r in rows], seed)):
                runs["own_goal_latent_with_joints"][name].append(s)
        e = {"train_rows": int(len(tr)), "train_reached_share": float(ya.mean())}
        for name, rows in ev.items():
            y = A["relaxed"][rows]
            rec = {"rows": int(len(rows)), "grasps": int(len(set(A["grasp"][rows]))), "success_share": float(y.mean()),
                   "auc_of_the_model_V": auc(A["v"][rows, point], y)}
            for head, d in runs.items():
                a = [auc(s, y) for s in d[name]]
                rec[head] = None if None in a else [round(float(np.mean(a)), 4), round(float(np.std(a)), 4)]
            base, mine = np.mean(runs["no_goal"][name], 0), np.mean(runs["own_goal"][name], 0)       # paired, resampling grasps
            gs = A["grasp"][rows]
            ug = np.unique(gs)
            where = {g: np.nonzero(gs == g)[0] for g in ug}
            diffs = []
            for _ in range(300):
                k = np.concatenate([where[g] for g in rng.choice(ug, len(ug))])
                a, b = auc(mine[k], y[k]), auc(base[k], y[k])
                if a is not None and b is not None:
                    diffs.append(a - b)
            rec["own_minus_none_ci95"] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)] if diffs else None
            e[name] = rec
        out[tag] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        from wm.training.common import limit_gpu_memory
        limit_gpu_memory()
    except Exception as e:
        print("no GPU limit set:", e)
    rng = np.random.default_rng(0)
    A = load(args.out, args.data)
    own, near, rand, two, info = goals(A, rng)
    P = A["p"].astype(np.float32)
    seated = A["seated"] & A["ok"][:, END] & (A["split"] == "train")
    mu, sd = P[seated, END].mean(0), P[:, END].std(0) + 1e-4
    Z = A["z"].astype(np.float32)
    F = {"picture": (P - mu) / sd, "latent": (Z - Z[:, END].mean(0)) / (Z[:, END].std(0) + 1e-4),
         "depth40": A["thumb"].astype(np.float32).reshape(len(P), 5, -1)}
    rep = {"rows": int(len(P)), "by_group": {g: int((A["group"] == g).sum()) for g in sorted(set(A["group"]))}, "goals": info,
           "points_available_share": {n: float(A["ok"][:, i].mean()) for i, n in enumerate(("face-2", "face+0", "half", "near", "end"))},
           "zero_depth_share_at_seated_end": float((A["thumb"][A["seated"], END] < 1.0).mean())}
    rep["check1_distance_to_goal"] = check1(A, F, own, near, rand)
    (Path(args.out) / "report.json").write_text(json.dumps(rep, indent=1))
    rep["check1b_done_readout_no_goal"] = check1b(A, F)
    (Path(args.out) / "report.json").write_text(json.dumps(rep, indent=1))
    rep["check2_value_with_goal"] = check2(A, {k: F[k] for k in ("picture", "latent")}, own, near, rand, two, rng)
    (Path(args.out) / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
