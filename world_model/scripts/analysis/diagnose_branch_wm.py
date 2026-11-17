"""Four offline checks of a trained branch model (wm/tdmpc/model.py) on recorded branches, before it is put in the loop.

    python world_model/scripts/analysis/diagnose_branch_wm.py --ckpt <run>/last.pt \
        --data /mnt/storage3/qingxin/data/wm_v1b --splits dev,ood_val --out <run>/diagnosis.json

A branch's end is put in one class from the simulator's truth (used here for judging only):
  inserted   fully in                      tilted    deep enough (<= 1 mm left) but crooked
  partial    went in, 1-12 mm left         blocked   12 mm or more left (never entered)      left / beside   wandered off
Scored moments: start, face-5 ... face+3 (decisions relative to the pins reaching the board face), last (the last decision).
  1 by_class      does the model score a not-fully-inserted end like an inserted one? Per class the median score, the
                  share of the class above the median of `inserted`, and the AUC of inserted against the class.
  2 label_truth   the FK label `seated` disagrees with the truth on about 5 % of the branches: are those scored like
                  their label or like their truth?
  3 imagination   the reward the model imagines for the next 4 decisions against the real one (mm of depth, + bonus),
                  per class; `through_board` = share of blocked branches where it imagines more than 1 mm of progress
                  while the real one is under 0.2 mm; `latent_ratio` = distance between the imagined latent and the
                  encoding of the real later picture, over the distance if nothing had changed (below 1: imagining helps).
  4 action        one state, different actions. At `start` the pictures of a start's branches are identical, only the
                  actions differ: `pairs` = share of (inserted, failed) pairs ranked right. At face-2: each state is
                  scored with the actions of the other branches of its start; spread over actions vs spread over states.
  calibration     the 5th percentile of v0 among inserted branches at face+3, and the share of failed branches below it.
  abort_rule      (--calibrate-abort <split>) the closed-loop test's early stop replayed on recorded branches: v0 at
                  every decision from face+3 on; an attempt is called off when v0 is under the threshold twice in a
                  row. Per threshold: share of inserted branches that would have been called off, share of failed ones,
                  decisions saved on those. `threshold` = the highest one that calls off at most 5 % of the inserted.
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

H = 4
TAGS1 = ("face+0", "face+3", "last")


def klass(b: dict) -> str:
    if b["inserted"]:
        return "inserted"
    if b["stopped"] == "left" or b["outcome"] == "beside_board":
        return "left_or_beside"
    if b["outcome"] == "tilted":
        return "tilted"
    return "partial" if b["depth_left_mm"] < 12.0 else "blocked"


class Points(torch.utils.data.Dataset):
    def __init__(self, branches, stats, gamma):
        self.b, self.stats, self.gamma = branches, stats, gamma

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        with np.load(self.b[i]["path"]) as z:
            a = data.decision_arrays(z)
            if a is None:
                return None
            pics = data._Pictures(z)
        n = a["n"]
        pts = data.eval_points(n, a["face_k"])
        if n - 1 not in [k for _, k in pts]:
            pts.append(("last", n - 1))
        act, ret = data.scale_action(a["delta"], self.stats), data.returns(a["reward"], self.gamma)
        out = defaultdict(list)
        for tag, k in pts:
            h = min(H, n - k)
            now, fut = np.maximum(np.arange(2 * k - 2, 2 * k + 1), 0), np.maximum(np.arange(2 * (k + h) - 2, 2 * (k + h) + 1), 0)
            rgb, depth = pics.get(np.r_[now, fut])
            action = np.zeros((H, data.ACTION_DIM), np.float32)
            action[:h] = act[k:k + h]
            out["rgb"].append(rgb); out["depth"].append(depth); out["q"].append(a["q_img"][np.r_[now, fut]])
            out["action"].append(action); out["valid"].append(h); out["ret"].append(ret[k]); out["k"].append(k)
            out["real"].append(a["reward"][k:k + h].sum()); out["branch"].append(i); out["tags"].append(tag)
        item = {k: torch.from_numpy(np.stack(v)) if k in ("rgb", "depth", "q", "action") else v for k, v in out.items()}
        return item


def collate(items):
    items = [it for it in items if it is not None]
    return {k: (torch.cat([it[k] for it in items]) if torch.is_tensor(items[0][k]) else [x for it in items for x in it[k]])
            for k in items[0]}


@torch.no_grad()
def run(model, branches, stats, workers, device="cuda"):
    model.eval()
    dl = torch.utils.data.DataLoader(Points(branches, stats, model.cfg.gamma), batch_size=3, num_workers=workers, collate_fn=collate)
    rows, keep = [], {}
    for b in dl:
        rgb, depth, q = b["rgb"].to(device).permute(0, 1, 4, 2, 3), b["depth"].to(device), b["q"].to(device)
        action, valid = b["action"].to(device), torch.tensor(b["valid"], device=device)
        z = model.encode(rgb[:, :3], depth[:, :3], q[:, :3], chunk=24)
        z_fut = model.encode(rgb[:, 3:], depth[:, 3:], q[:, 3:], chunk=24)
        sc = {k: v.mean(0).cpu().numpy() for k, v in model.scores(z, action, valid).items()}
        G, zt = torch.zeros(len(z), device=device), z
        for t in range(H):
            on = (t < valid).float()
            G = G + on * model.scalar(model.reward(zt, action[:, t]))[:, 0]
            zt = torch.where(on[:, None] > 0, model.next(zt, action[:, t]), zt)
        ratio = ((zt - z_fut).norm(dim=1) / (z - z_fut).norm(dim=1).clamp(min=1e-6)).cpu().numpy()
        for i, tag in enumerate(b["tags"]):
            rows.append({"branch": b["branch"][i], "tag": tag, "k": b["k"][i], "ret": float(b["ret"][i]), "real": float(b["real"][i]),
                         "imagined": float(G[i]), "latent_ratio": float(ratio[i]), **{k: float(v[i]) for k, v in sc.items()}})
            if tag in ("start", "face-2"):
                keep[(b["branch"][i], tag)] = (z[i].cpu(), action[i].cpu(), int(valid[i]))
    return rows, keep


class Dense(torch.utils.data.Dataset):
    """All decisions of a branch from face+3 on (at most 60): the pictures they need, as one consecutive run."""

    def __init__(self, branches):
        self.b = branches

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        with np.load(self.b[i]["path"]) as z:
            a = data.decision_arrays(z)
            if a is None or a["face_k"] < 0 or a["n"] - (a["face_k"] + 3) < 2:
                return None
            k0, k1 = a["face_k"] + 3, min(a["n"], a["face_k"] + 63)
            idx = np.arange(2 * k0 - 2, 2 * (k1 - 1) + 1)
            rgb, depth = data._Pictures(z).get(idx)
        return {"rgb": torch.from_numpy(rgb), "depth": torch.from_numpy(depth), "q": torch.from_numpy(a["q_img"][idx]),
                "m": k1 - k0, "branch": i, "left": a["n"] - k0}


@torch.no_grad()
def abort_rule(model, branches, workers, device="cuda"):
    model.eval()
    dl = torch.utils.data.DataLoader(Dense(branches), batch_size=1, num_workers=workers, collate_fn=lambda x: x[0])
    vs = []
    for b in dl:
        if b is None:
            continue
        feat = model.frames(b["rgb"].to(device).permute(0, 3, 1, 2), b["depth"].to(device), chunk=24)
        q = b["q"].to(device)
        j = torch.arange(b["m"], device=device)[:, None] * 2 + torch.arange(3, device=device)[None]
        v = model.scalar(model.V(model.fuse(feat[j], q[j])))[..., 0].mean(0).cpu().numpy()
        vs.append((v, branches[b["branch"]]["inserted"], b["left"]))
    if not vs:
        return None
    grid = np.percentile(np.concatenate([v for v, _, _ in vs]), np.arange(2, 60, 2))
    table = []
    for thr in grid:
        cut = {True: [], False: []}
        saved = []
        for v, ins, left in vs:
            low = (v[1:] < thr) & (v[:-1] < thr)
            at = int(np.argmax(low)) + 1 if low.any() else None
            cut[bool(ins)].append(at is not None)
            if at is not None and not ins:
                saved.append(left - at - 1)
        table.append({"threshold": float(thr), "inserted_called_off": float(np.mean(cut[True])) if cut[True] else None,
                      "failed_called_off": float(np.mean(cut[False])) if cut[False] else None,
                      "decisions_saved_mean": float(np.mean(saved)) if saved else None})
    okay = [t for t in table if t["inserted_called_off"] is not None and t["inserted_called_off"] <= 0.05]
    best = max(okay, key=lambda t: t["threshold"]) if okay else table[0]
    return {"branches": len(vs), "inserted": int(sum(i for _, i, _ in vs)), **best, "table": table}


def med(x):
    return float(np.median(x)) if len(x) else None


def analyse(model, rows, keep, branches, device="cuda"):
    rep = {"points": len(rows), "branches": len({r["branch"] for r in rows}),
           "classes": {c: sum(1 for i in {r["branch"] for r in rows} if klass(branches[i]) == c)
                       for c in ("inserted", "tilted", "partial", "blocked", "left_or_beside")}}
    cls = [klass(b) for b in branches]
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)
    # 1 ---- per class
    rep["1_by_class"] = {}
    for tag in TAGS1:
        rs = by_tag.get(tag, [])
        t = {}
        for s in ("v0", "q0", "im2"):
            ins = np.array([r[s] for r in rs if cls[r["branch"]] == "inserted"])
            t[s] = {}
            for c in ("inserted", "tilted", "partial", "blocked", "left_or_beside"):
                x = np.array([r[s] for r in rs if cls[r["branch"]] == c])
                t[s][c] = {"n": len(x), "median": med(x),
                           "above_median_of_inserted": float((x > np.median(ins)).mean()) if len(x) and len(ins) else None,
                           "auc_inserted_vs_class": report.auc(np.r_[ins, x], np.r_[np.ones(len(ins)), np.zeros(len(x))])
                           if c != "inserted" and len(x) and len(ins) else None}
        rep["1_by_class"][tag] = t
    # 2 ---- label against truth
    rep["2_label_truth"] = {}
    for tag in ("face+3", "last"):
        rs = by_tag.get(tag, [])
        g = defaultdict(list)
        for r in rs:
            b = branches[r["branch"]]
            g[f"label={b['seated_fk']},truth={b['inserted']}"].append(r["v0"])
        s = np.array([r["v0"] for r in rs])
        rep["2_label_truth"][tag] = {"median_v0": {k: {"n": len(v), "median": med(v)} for k, v in g.items()},
                                     "auc_vs_truth": report.auc(s, np.array([branches[r["branch"]]["inserted"] for r in rs])) if len(rs) else None,
                                     "auc_vs_label": report.auc(s, np.array([branches[r["branch"]]["seated_fk"] for r in rs])) if len(rs) else None}
    # 3 ---- imagination against reality
    rep["3_imagination"] = {}
    for tag in ("face-2", "face+0", "face+3"):
        rs = by_tag.get(tag, [])
        t = {}
        for c in ("inserted", "tilted", "partial", "blocked"):
            x = [r for r in rs if cls[r["branch"]] == c]
            real, im = np.array([r["real"] for r in x]), np.array([r["imagined"] for r in x])
            t[c] = {"n": len(x), "real_mean": float(real.mean()) if len(x) else None, "imagined_mean": float(im.mean()) if len(x) else None,
                    "abs_err_mean": float(np.abs(real - im).mean()) if len(x) else None, "latent_ratio_median": med([r["latent_ratio"] for r in x])}
            if c == "blocked" and len(x):
                still = real < 0.2
                t[c]["through_board"] = float((im[still] > 1.0).mean()) if still.any() else None
        real, im = np.array([r["real"] for r in rs]), np.array([r["imagined"] for r in rs])
        t["corr"] = float(np.corrcoef(real, im)[0, 1]) if len(rs) > 2 and real.std() > 0 and im.std() > 0 else None
        rep["3_imagination"][tag] = t
    # 4 ---- one state, different actions
    groups = defaultdict(list)
    for bi, b in enumerate(branches):
        groups[(b["grasp"], b["start"])].append(bi)
    a4 = {}
    rs = by_tag.get("start", [])
    row_of = {r["branch"]: r for r in rs}
    for s in ("v0", "q0", "im2", "im4"):
        ok = n = 0
        within = []
        for g in groups.values():
            g = [bi for bi in g if bi in row_of]
            if len(g) < 2:
                continue
            x, y = np.array([row_of[bi][s] for bi in g]), np.array([branches[bi]["inserted"] for bi in g])
            within.append(x.std())
            if 0 < y.sum() < len(y):
                d = x[y][:, None] - x[~y][None, :]
                ok += float((d > 0).sum() + 0.5 * (d == 0).sum()); n += d.size
        allx = np.array([r[s] for r in rs])
        a4[s] = {"pairs": ok / n if n else None, "n_pairs": n, "spread_within_start": float(np.mean(within)) if within else None,
                 "spread_across_starts": float(allx.std()) if len(allx) else None}
    rep["4_action"] = {"start": a4}
    sa, ss, n_groups = [], [], 0
    for g in groups.values():
        g = [bi for bi in g if (bi, "face-2") in keep and keep[(bi, "face-2")][2] >= 2]
        if len(g) < 2:
            continue
        z = torch.stack([keep[(bi, "face-2")][0] for bi in g]).to(device)
        act = torch.stack([keep[(bi, "face-2")][1] for bi in g]).to(device)
        m = len(g)
        zz, aa = z[:, None].expand(m, m, -1).reshape(m * m, -1), act[None].expand(m, m, -1, -1).reshape(m * m, H, -1)
        with torch.no_grad():
            S = model.scores(zz, aa, torch.full((m * m,), 2, device=device), hs=(1, 2))["im2"].mean(0).reshape(m, m).cpu().numpy()
        sa.append(S.std(1).mean()); ss.append(S.std(0).mean()); n_groups += 1       # rows: states, columns: actions
    rep["4_action"]["face-2_cross"] = {"starts": n_groups, "im2_spread_over_actions": float(np.mean(sa)) if sa else None,
                                       "im2_spread_over_states": float(np.mean(ss)) if ss else None}
    # calibration for the early-stop rule
    late = [r for r in rows if r["tag"] in ("face+3",)]
    ins = np.array([r["v0"] for r in late if branches[r["branch"]]["inserted"]])
    fail = np.array([r["v0"] for r in late if not branches[r["branch"]]["inserted"]])
    if len(ins) and len(fail):
        thr = float(np.percentile(ins, 5))
        rep["calibration"] = {"v0_p05_inserted_face+3": thr, "failed_below": float((fail < thr).mean()), "inserted_below": float((ins < thr).mean())}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="dev,ood_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-branches", type=int)
    ap.add_argument("--calibrate-abort", help="split on which the early-stop rule is replayed")
    ap.add_argument("--calibrate-branches", type=int, default=600)
    args = ap.parse_args()
    limit_gpu_memory()
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model = LatentWM(make_cfg(**ck["cfg"]), ck["stats"]).cuda()
    model.load_state_dict(ck["model"])
    out = {"ckpt": args.ckpt, "step": ck["step"]}
    for split in args.splits.split(","):
        bs = data.list_branches(args.data, split)
        if args.max_branches:
            bs = bs[:args.max_branches]
        if not bs:
            continue
        rows, keep = run(model, bs, ck["stats"], args.workers)
        out[split] = analyse(model, rows, keep, bs)
        if split == "ood_val":                   # the same checks separately for held-out shapes and for seen shapes with unseen grasps
            from wm.data.splits import OOD_VAL_SHAPES
            for name, want in (("ood_val_unseen_shape", True), ("ood_val_seen_shape_unseen_grasp", False)):
                sub = {i for i, b in enumerate(bs) if (b["port"] in OOD_VAL_SHAPES) == want}
                out[name] = analyse(model, [r for r in rows if r["branch"] in sub], {k: v for k, v in keep.items() if k[0] in sub}, bs)
                out[name]["auc_v0_by_tag"] = {t: report.auc(np.array([r["v0"] for r in rows if r["branch"] in sub and r["tag"] == t]),
                                                            np.array([bs[r["branch"]]["inserted"] for r in rows if r["branch"] in sub and r["tag"] == t]))
                                              for t in ("start", "face-5", "face-2", "face+0", "face+3")}
        print(f"==== {split}\n{json.dumps(out[split], indent=1)}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    if args.calibrate_abort:
        bs = data.list_branches(args.data, args.calibrate_abort)
        bs = bs[::max(len(bs) // args.calibrate_branches, 1)][:args.calibrate_branches]
        out["abort_rule"] = {"split": args.calibrate_abort, **(abort_rule(model, bs, args.workers) or {})}
        print("==== abort rule\n" + json.dumps({k: v for k, v in out["abort_rule"].items() if k != "table"}), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
