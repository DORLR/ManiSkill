"""Offline checks shared by the agreed model and the unchanged TD-MPC2.

Every scored decision of every branch has a tag (`start`, or `face-5` = five decisions before the pins reach the
board face, ...) and several scores (v0, q0, im1, ...). Per tag and score:
  auc_inserted   how well the score separates branches that end inserted (simulator's truth, used here only) from
                 the others: 0.5 = chance, 1 = perfect
  auc_seated     the same against the FK label the model was trained with
  same_start     only starts that have both an inserted and a failed branch: `pairs` = share of (inserted, failed)
                 pairs of one start in which the inserted branch has the higher score; `pick` = share of starts whose
                 highest-scored branch is inserted, next to `random` (a blind pick) -- `oracle` is 1 by construction
  all_starts     pick / random / oracle over all starts: what choosing by this score would give
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def auc(score: np.ndarray, label: np.ndarray) -> float | None:
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    rank = np.empty(len(s))
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rank[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((rank[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build(records: list[dict], branches: list[dict]) -> dict:
    """records: {"branch": index into branches, "tag", "k", "ret", "scores": {name: float}}."""
    rep = {"points": len(records), "branches": len({r["branch"] for r in records})}
    by_tag = defaultdict(list)
    for r in records:
        by_tag[r["tag"]].append(r)
    names = sorted(records[0]["scores"]) if records else []
    for tag, rs in by_tag.items():
        ins = np.array([branches[r["branch"]]["inserted"] for r in rs])
        seat = np.array([branches[r["branch"]]["seated_fk"] for r in rs])
        ret = np.array([r["ret"] for r in rs])
        groups = defaultdict(list)
        for i, r in enumerate(rs):
            b = branches[r["branch"]]
            groups[(b["grasp"], b["start"])].append(i)
        t = {"n": len(rs), "inserted": int(ins.sum()), "starts": len(groups),
             "starts_mixed": sum(1 for g in groups.values() if 0 < ins[g].sum() < len(g))}
        for name in names:
            s = np.array([r["scores"][name] for r in rs])
            pairs_ok = pairs = 0
            pick_mixed, rand_mixed, pick_all, rand_all, oracle_all = [], [], [], [], []
            for g in groups.values():
                gi, gs = ins[g], s[g]
                if len(g) < 2:
                    continue
                pick_all.append(bool(gi[int(np.argmax(gs))]))
                rand_all.append(float(gi.mean()))
                oracle_all.append(bool(gi.any()))
                if 0 < gi.sum() < len(g):
                    d = gs[gi][:, None] - gs[~gi][None, :]
                    pairs_ok += float((d > 0).sum() + 0.5 * (d == 0).sum())
                    pairs += d.size
                    pick_mixed.append(bool(gi[int(np.argmax(gs))]))
                    rand_mixed.append(float(gi.mean()))
            t[name] = {"auc_inserted": auc(s, ins), "auc_seated": auc(s, seat),
                       "corr_return": float(np.corrcoef(s, ret)[0, 1]) if len(s) > 2 and s.std() > 0 and ret.std() > 0 else None,
                       "abs_err_return": float(np.abs(s - ret).mean()),
                       "same_start": {"pairs": pairs_ok / pairs if pairs else None, "n_pairs": pairs,
                                      "pick": float(np.mean(pick_mixed)) if pick_mixed else None,
                                      "random": float(np.mean(rand_mixed)) if rand_mixed else None},
                       "all_starts": {"pick": float(np.mean(pick_all)) if pick_all else None,
                                      "random": float(np.mean(rand_all)) if rand_all else None,
                                      "oracle": float(np.mean(oracle_all)) if oracle_all else None}}
        rep[tag] = t
    return rep


def table(rep: dict) -> str:
    """A compact text view: per tag one line per score."""
    lines = []
    for tag, t in rep.items():
        if not isinstance(t, dict):
            continue
        lines.append(f"{tag}: {t['n']} points, {t['inserted']} inserted, {t['starts']} starts ({t['starts_mixed']} mixed)")
        for name, m in t.items():
            if not isinstance(m, dict):
                continue
            f = lambda v: "  -  " if v is None else f"{v:.3f}"
            lines.append(f"   {name:8s} auc {f(m['auc_inserted'])}  same-start pairs {f(m['same_start']['pairs'])} "
                         f"pick {f(m['same_start']['pick'])} vs random {f(m['same_start']['random'])}  | all starts: pick "
                         f"{f(m['all_starts']['pick'])} random {f(m['all_starts']['random'])} oracle {f(m['all_starts']['oracle'])}")
    return "\n".join(lines)
