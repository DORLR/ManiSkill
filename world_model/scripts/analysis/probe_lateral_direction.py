"""Does the model's score know which way the hole is? Offline, on recorded branches (2026-11-16, after the review).

The four attempts of one start arrive at the board at slightly different lateral positions; the simulator's truth
(eval_rel: plug pose in the socket frame, used here for judging only) says where each one is. Three questions:
  2 size       how far off laterally (truth) are the attempts at the board face, by how they end
  3 keeping    does the policy carry a lateral difference to the face, or pull every attempt back to one place?
               (difference between two attempts of a start at an earlier moment against the same difference at the face)
  1 direction  between two attempts of a start: does V prefer the one truly closer to the hole axis? And can the
               signed lateral error be read from the latent z (ridge and a small MLP fitted on training-split latents,
               tested on the held-out ones)?

    python world_model/scripts/analysis/probe_lateral_direction.py --ckpt <run>/last.pt --data <wm_v1b> --out <dir>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from wm.tdmpc import data, report
from wm.tdmpc.model import LatentWM, make_cfg
from wm.training.common import limit_gpu_memory

SCORED = ("face-5", "face-2", "face+0")                 # moments with pictures -> V and z
TRUTH = (("start", None), ("face-10", -10), ("face-5", -5), ("face-2", -2), ("face+0", 0), ("face+3", 3), ("last", None))


class Points(torch.utils.data.Dataset):
    def __init__(self, branches):
        self.b = branches

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        with np.load(self.b[i]["path"]) as z:
            a = data.decision_arrays(z)
            if a is None or a["face_k"] < 0:
                return None
            rel, dec = z["eval_rel"], z["dec_step"]
            n, f = a["n"], a["face_k"]
            truth = np.full((len(TRUTH), 3), np.nan, np.float32)          # depth left, y, z [mm]
            for j, (name, off) in enumerate(TRUTH):
                k = 0 if name == "start" else n - 1 if name == "last" else f + off
                if 0 <= k < n:
                    r = rel[dec[k]]
                    truth[j] = (-r[0] * 1000, r[1] * 1000, r[2] * 1000)
            ks = [f + o for o in (-5, -2, 0)]
            ok = [0 <= k < n for k in ks]
            pics = data._Pictures(z)
            rgb, depth, q = [], [], []
            for k, good in zip(ks, ok):
                idx = np.maximum(np.arange(2 * max(k, 0) - 2, 2 * max(k, 0) + 1), 0)
                r_, d_ = pics.get(idx)
                rgb.append(r_); depth.append(d_); q.append(a["q_img"][idx])
        return {"rgb": torch.from_numpy(np.stack(rgb)), "depth": torch.from_numpy(np.stack(depth)), "q": torch.from_numpy(np.stack(q)),
                "ok": torch.tensor(ok), "truth": torch.from_numpy(truth), "branch": i}


@torch.no_grad()
def run(model, branches, workers, device="cuda"):
    dl = torch.utils.data.DataLoader(Points(branches), batch_size=1, num_workers=workers, collate_fn=lambda x: x[0])
    out = {"branch": [], "ok": [], "truth": [], "v": [], "z": []}
    for b in dl:
        if b is None:
            continue
        z = model.encode(b["rgb"].to(device).permute(0, 1, 4, 2, 3), b["depth"].to(device), b["q"].to(device), chunk=24)
        out["v"].append(model.scalar(model.V(z))[..., 0].mean(0).cpu().numpy())
        out["z"].append(z.cpu().numpy().astype(np.float16))
        out["branch"].append(int(b["branch"]))                # copies: views would keep the shared memory of the workers open
        out["ok"].append(b["ok"].numpy().copy()); out["truth"].append(b["truth"].numpy().copy())
    return {k: np.stack(v) if k != "branch" else np.asarray(v) for k, v in out.items()}


def klass(b):
    if b["inserted"]:
        return "inserted"
    if b["stopped"] == "left" or b["outcome"] == "beside_board":
        return "left_or_beside"
    if b["outcome"] == "tilted":
        return "tilted"
    return "partial" if b["depth_left_mm"] < 12.0 else "blocked"


def pct(x, q=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 2) for v in np.percentile(x, q)] if len(x) else None


def analyse(res, branches):
    T = {name: j for j, (name, _) in enumerate(TRUTH)}
    S = {name: j for j, name in enumerate(SCORED)}
    cls = np.array([klass(branches[i]) for i in res["branch"]])
    ins = np.array([branches[i]["inserted"] for i in res["branch"]])
    lat = res["truth"][:, :, 1:]                                   # (N, moments, 2)
    mag = np.linalg.norm(lat, axis=2)
    rep = {"branches": int(len(cls)), "classes": {c: int((cls == c).sum()) for c in sorted(set(cls))}}
    # 2 ---- size
    rep["2_size_lateral_mm_pct_10_25_50_75_90"] = {name: {c: pct(mag[cls == c, j]) for c in ("inserted", "blocked", "tilted", "partial")}
                                                    for name, j in T.items()}
    rep["2_blocked_share_within_mm_at_face"] = {str(r): float(np.nanmean(mag[cls == "blocked", T["face+0"]] < r)) for r in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)}
    # groups of siblings
    groups = defaultdict(list)
    for row, i in enumerate(res["branch"]):
        groups[(branches[i]["grasp"], branches[i]["start"])].append(row)
    pairs = [(a, b) for g in groups.values() for x, a in enumerate(g) for b in g[x + 1:]]
    pa, pb = np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int)
    rep["sibling_pairs"] = int(len(pairs))
    # 3 ---- keeping: a lateral difference between two attempts of a start, earlier against at the face
    keep = {}
    for name in ("start", "face-10", "face-5", "face-2"):
        d0, d1 = lat[pa, T[name]] - lat[pb, T[name]], lat[pa, T["face+0"]] - lat[pb, T["face+0"]]
        ok = np.isfinite(d0).all(1) & np.isfinite(d1).all(1)
        x, y = d0[ok].ravel(), d1[ok].ravel()
        keep[name] = {"pairs": int(ok.sum()), "difference_then_mm_median": float(np.median(np.linalg.norm(d0[ok], axis=1))),
                      "difference_at_face_mm_median": float(np.median(np.linalg.norm(d1[ok], axis=1))),
                      "slope_face_on_then": float((x @ y) / (x @ x)), "corr": float(np.corrcoef(x, y)[0, 1])}
    rep["3_keeping"] = keep
    cent = {}
    for name in ("start", "face-10", "face-5", "face-2", "face+0"):
        within, vec = [], []
        for g in groups.values():
            v = lat[g, T[name]]
            v = v[np.isfinite(v).all(1)]
            if len(v) >= 2:
                within.append(np.linalg.norm(v - v.mean(0), axis=1).mean()); vec.append(v.mean(0))
        vec = np.asarray(vec)
        cent[name] = {"spread_within_start_mm_median": float(np.median(within)), "start_means_spread_mm_median": float(np.median(np.linalg.norm(vec - vec.mean(0), axis=1)))}
    rep["3_spread_of_lateral_position"] = cent
    x0, x1 = lat[:, T["start"]], lat[:, T["face+0"]]
    ok = np.isfinite(x0).all(1) & np.isfinite(x1).all(1)
    rep["3_corr_start_offset_vs_face_offset"] = float(np.corrcoef(x0[ok].ravel(), x1[ok].ravel())[0, 1])
    # 1 ---- direction: V between siblings, against the truth
    d1 = {}
    for name in SCORED:
        j, s = T[name], S[name]
        v, m = res["v"][:, s], mag[:, j]
        good = res["ok"][:, s] & np.isfinite(m)
        ok = good[pa] & good[pb]
        a, b = pa[ok], pb[ok]
        dm, dv = m[a] - m[b], v[a] - v[b]
        e = {"pairs": int(ok.sum()), "closer_gets_higher_V_by_difference_mm": {}}
        for lo, hi in ((0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 99.0)):
            sel = (np.abs(dm) >= lo) & (np.abs(dm) < hi)
            e["closer_gets_higher_V_by_difference_mm"][f"{lo}-{hi}"] = [int(sel.sum()), float(np.mean(np.sign(dv[sel]) == -np.sign(dm[sel]))) if sel.any() else None]
        sel = np.abs(dm) >= 0.3
        e["closer_gets_higher_V_all_over_0.3mm"] = float(np.mean(np.sign(dv[sel]) == -np.sign(dm[sel])))
        mixed = ins[a] != ins[b]
        e["mixed_pairs"] = int(mixed.sum())
        e["truth_closer_is_the_inserted_one"] = float(np.mean((dm[mixed] < 0) == ins[a][mixed]))
        e["V_higher_is_the_inserted_one"] = float(np.mean((dv[mixed] > 0) == ins[a][mixed]))
        same = (~ins[a]) & (~ins[b]) & (np.abs(dm) >= 0.3)
        e["both_fail_pairs_closer_gets_higher_V"] = [int(same.sum()), float(np.mean(np.sign(dv[same]) == -np.sign(dm[same]))) if same.any() else None]
        e["auc_V_inserted"] = report.auc(v[good], ins[good])
        e["auc_minus_lateral_inserted"] = report.auc(-m[good], ins[good])
        e["corr_V_vs_lateral_mm"] = float(np.corrcoef(v[good], m[good])[0, 1])
        d1[name] = e
    rep["1_direction_V"] = d1
    return rep


def probe(train, test, branches_test):
    """Signed lateral error (y, z in the socket frame, mm) from the latent: ridge and a small MLP."""
    T = {name: j for j, (name, _) in enumerate(TRUTH)}
    out = {}
    for s, name in enumerate(SCORED):
        j = T[name]
        def xy(res):
            ok = res["ok"][:, s] & np.isfinite(res["truth"][:, j]).all(1)
            return res["z"][ok, s].astype(np.float32), res["truth"][ok, j, 1:].astype(np.float32)
        Xa, Ya = xy(train)
        Xb, Yb = xy(test)
        keep = np.linalg.norm(Ya, axis=1) < 15                      # wandered-off attempts would dominate the fit
        Xa, Ya = Xa[keep], Ya[keep]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        A, B = (Xa - mu) / sd, (Xb - mu) / sd
        e = {"train_points": int(len(Xa)), "test_points": int(len(Xb)), "test_lateral_mm_median": float(np.median(np.linalg.norm(Yb, axis=1)))}

        def score(P, tag):
            err = np.linalg.norm(P - Yb, axis=1)
            base = np.linalg.norm(Ya.mean(0) - Yb, axis=1)
            big = np.linalg.norm(Yb, axis=1) > 0.5
            cos = (P[big] * Yb[big]).sum(1) / (np.linalg.norm(P[big], axis=1) * np.linalg.norm(Yb[big], axis=1) + 1e-9)
            e[tag] = {"error_mm_median": float(np.median(err)), "error_if_blind_mm_median": float(np.median(base)),
                      "direction_cosine_mean": float(cos.mean()), "direction_right_half": float((cos > 0).mean()),
                      "direction_within_45deg": float((cos > 0.707).mean())}
        best = None
        for lam in (1.0, 10.0, 100.0, 1000.0):
            W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean(0)))
            P = B @ W + Ya.mean(0)
            m = float(np.median(np.linalg.norm(P - Yb, axis=1)))
            if best is None or m < best[0]:
                best = (m, P, lam)
        score(best[1], "ridge")
        e["ridge"]["lambda_best_on_test"] = best[2]
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        At, Yt = torch.from_numpy(A), torch.from_numpy(Ya)
        for ep in range(60):
            perm = torch.randperm(len(At))
            for i in range(0, len(At), 256):
                k = perm[i:i + 256]
                loss = torch.nn.functional.smooth_l1_loss(net(At[k]), Yt[k])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            score(net(torch.from_numpy(B)).numpy(), "mlp")
            e["mlp_train_error_mm_median"] = float(np.median(np.linalg.norm(net(At).numpy() - Ya, axis=1)))
        out[name] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--max-branches", type=int)
    args = ap.parse_args()
    limit_gpu_memory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = LatentWM(make_cfg(**ck["cfg"]), ck["stats"]).cuda().eval()
    model.load_state_dict(ck["model"])
    res = {}
    for split, every in (("ood_val", 1), ("train", args.train_every)):
        bs = data.list_branches(args.data, split)
        starts = sorted({(b["grasp"], b["start"]) for b in bs})[::every]       # whole starts: the analysis pairs their attempts
        bs = [b for b in bs if (b["grasp"], b["start"]) in set(starts)]
        if args.max_branches:
            bs = bs[:args.max_branches]
        f = out / f"points_{split}.npz"
        if f.exists():
            res[split] = (dict(np.load(f)), bs)
        else:
            r = run(model, bs, args.workers)
            np.savez(f, **r)
            res[split] = (r, bs)
        print(split, len(res[split][0]["branch"]), "branches scored", flush=True)
    rep = {"ckpt": args.ckpt, "ood_val": analyse(*res["ood_val"]), "train_sample": analyse(*res["train"])}
    rep["probe_signed_lateral_from_latent"] = probe(res["train"][0], res["ood_val"][0], res["ood_val"][1])
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
