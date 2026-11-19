"""Analysis of the in-hole agent's ablations (2026-11-19, the user's request): does imagination help, or any move?

Arms are result directories of run_wm320_attempts.py on the same split, grasps, starts and seeds (per-grasp files
<dir>/<port>/<grasp>.json). Two commands:

  pbins <wm dir> [--json <file>]
      the "wm" agent's correction rate per bin b of the estimated pin-tip gap g (common320.RANDOM_EDGES_MM):
          p_b = #(decisions with candidates offered, g in bin b, the model corrected) / #(decisions with candidates
                offered, g in bin b)
      printed as "p0,p1,p2,p3" (the --random-p of the ablation "random"); counts go to --json.

  cells DIR [--arm NAME=DIR ...] --out <json>      (make_combo_grasps.py's 8 paired cells, 2026-11-19)
      per arm and cell: judged starts, unusable grasps, success with a 95 % cluster-bootstrap interval (clusters: the
      shape for cells with "shape", the grasp otherwise; 2,000 seeded draws); factor effects paired by the cells' shared
      index i and the start: for every two cells that differ in one factor, mean(success_with - success_without) over
      starts judged in both with the same drawn start, with a cluster sign-flip p (clusters: the held-out shape if a
      cell has "shape", else i).

  compare --ref NAME=DIR --arm NAME=DIR [--arm ...] --out <json>
      per arm: success over all judged starts; paired against the reference on starts whose start pose is identical
      in both (placement can differ between runs): n, both rates, b = arm succeeds where the reference fails,
      c = the reverse, exact two-sided McNemar p = binomial test of min(b, c) in b + c at 1/2 (scipy); because starts
      of one grasp (and grasps of one unseen shape) are not independent, also a cluster sign-flip test: clusters as
      eval_policy320's standard errors (the port for unseen shapes, the grasp otherwise), d_c = (b - c) within cluster
      c, T = sum_c d_c, p = share of random sign flips s_c in {-1, 1} with |sum_c s_c d_c| >= |T| (exact up to 16
      clusters with d_c != 0, else 200,000 seeded draws); the same per held-out group; moves per attempt
      (non-proposal agent decisions; `interventions` for files without per_decision); each arm's settings from its
      summary and the keys that differ from the reference. Result files of crashed grasps (with "error") are skipped,
      as summarise does. A text table goes to stdout.

Held-out groups (demo320.split320's numbers): the combination of unseen factors, shape (OOD_VAL_SHAPES), angle (grasp
pitch in (-42, -38) deg) and position (grasp position in (-21, -19) mm), e.g. "shape", "angle+position",
"shape+angle+position" (wm_v1 has no "angle+position" grasp; scripts/eval/make_combo_grasps.py makes such test grasps).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.data.splits import OOD_VAL_SHAPES  # noqa: E402
from wm.online.common320 import RANDOM_EDGES_MM, gap_bin  # noqa: E402

N_BINS = len(RANDOM_EDGES_MM) + 1


def group_of(port: str, pitch_deg: float, pos_m: float) -> str:
    """Which of the three factors are unseen: shape (OOD_VAL_SHAPES), angle (pitch in the held-out strip), position
    (position in the held-out strip); "+"-joined in that order (7 combinations), "none" for none."""
    unseen = (("shape", port in OOD_VAL_SHAPES), ("angle", -42.0 < pitch_deg < -38.0), ("position", -21.0 < pos_m * 1000.0 < -19.0))
    return "+".join(n for n, v in unseen if v) or "none"


def load_arm(d: str) -> dict:
    """(grasp, start index) -> {success, start (canonical json), group, moves, per_decision}; judged attempts only."""
    out = {}
    files = sorted(glob.glob(f"{d}/*/*.json"))
    if not files:
        print(f"warning: no result files in {d}: arm left out", file=sys.stderr)
        return {}
    for f in files:
        j = json.loads(Path(f).read_text())
        if j.get("error") or "starts" not in j or "grasp" not in j or "pitch_deg" not in j:
            continue                                           # crashed grasps count nowhere (as eval_policy320.summarise)
        grp = group_of(j["port"], j["pitch_deg"], j["grasp_pos_m"])
        cluster = j["port"] if j["port"] in OOD_VAL_SHAPES else j["grasp"]
        for s in j["starts"]:
            a = s.get("attempt")
            if not a or a.get("success") is None:
                continue
            pdl = a.get("per_decision")
            moves = sum(1 for x in pdl if x["pick"] != 0) if pdl is not None else int(a.get("interventions", 0))
            out[(j["grasp"], int(s["idx"]))] = {"success": bool(a["success"]), "start": json.dumps(s["start"], sort_keys=True),
                                               "group": grp, "cluster": cluster, "moves": moves, "per_decision": pdl,
                                               "hand": s.get("start_hand_mm")}
    return out


def pbins(d: str) -> dict:
    offered, corrected = np.zeros(N_BINS, np.int64), np.zeros(N_BINS, np.int64)
    n_att, missing = 0, 0
    for v in load_arm(d).values():
        if v["per_decision"] is None:
            missing += 1
            continue
        n_att += 1
        for x in v["per_decision"]:
            if not x["offered"] or x["g"] is None:
                continue
            b = gap_bin(float(x["g"]))
            offered[b] += 1
            corrected[b] += int(x["pick"] > 0)
    if missing:
        raise SystemExit(f"{missing} attempts in {d} have no per_decision record: not a run of the logging agent")
    if n_att == 0:
        raise SystemExit(f"no attempts in {d}")
    p = [float(c) / float(o) if o else 0.0 for c, o in zip(corrected, offered)]
    return {"dir": d, "attempts": n_att, "edges_mm": list(RANDOM_EDGES_MM), "offered": offered.tolist(),
            "corrected": corrected.tolist(), "p": p}


def mcnemar_p(b: int, c: int) -> float:
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def cluster_signflip_p(d: list[float], draws: int = 200_000, seed: int = 0) -> float:
    """Two-sided sign-flip test of sum(d) = 0 over clusters (exact for <= 16 non-zero clusters)."""
    d = np.asarray([x for x in d if x != 0], np.float64)
    if len(d) == 0:
        return 1.0
    T = abs(d.sum())
    if len(d) <= 16:
        signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(d)))
    return float(np.mean(np.abs(signs @ d) >= T - 1e-9))


def same_start(x: dict, y: dict) -> bool:
    """The same drawn start, and (where both files have it) the placed hand within 0.5 mm."""
    if x["start"] != y["start"]:
        return False
    return x["hand"] is None or y["hand"] is None or float(np.linalg.norm(np.subtract(x["hand"], y["hand"]))) < 0.5


def paired(ref: dict, arm: dict, keys=None) -> dict:
    ks = [k for k in (keys if keys is not None else arm) if k in ref and k in arm and same_start(ref[k], arm[k])]
    b = sum(1 for k in ks if arm[k]["success"] and not ref[k]["success"])
    c = sum(1 for k in ks if ref[k]["success"] and not arm[k]["success"])
    n = len(ks)
    per = {}
    for k in ks:
        per.setdefault(arm[k]["cluster"], 0.0)
        per[arm[k]["cluster"]] += float(arm[k]["success"]) - float(ref[k]["success"])
    return {"n": n, "ref": float(np.mean([ref[k]["success"] for k in ks])) if n else None,
            "arm": float(np.mean([arm[k]["success"] for k in ks])) if n else None,
            "arm_wins": b, "ref_wins": c, "mcnemar_p": mcnemar_p(b, c), "clusters": len(per),
            "cluster_signflip_p": cluster_signflip_p(list(per.values()))}


SETTINGS = ("ckpt", "split", "mode", "starts_per_grasp", "horizon", "margin", "inside_mm", "max_retries", "candidates", "seed", "grasps_n")


def settings_of(d: str) -> dict:
    for name in ("summary.json", "summary_incomplete.json"):
        f = Path(d) / name
        if f.exists():
            s = json.loads(f.read_text())
            return {k: s.get(k) for k in SETTINGS} | {"complete": name == "summary.json"}
    return {"complete": None}


def compare(ref_spec: str, arm_specs: list[str]) -> dict:
    rname, rdir = ref_spec.split("=", 1)
    ref = load_arm(rdir)
    if not ref:
        raise SystemExit(f"the reference {rdir} has no results")
    arms, dirs, missing = {rname: ref}, {rname: rdir}, []
    for s in arm_specs:
        name, d = s.split("=", 1)
        if name in arms:
            raise SystemExit(f"arm name {name} given twice")
        a = load_arm(d)
        if a:
            arms[name], dirs[name] = a, d
        else:
            missing.append(name)
    groups = sorted({v["group"] for a in arms.values() for v in a.values()})
    res = {"reference": rname, "arms": {}, "missing_arms": missing}
    ref_set = settings_of(rdir)
    for name, a in arms.items():
        st = settings_of(dirs[name])
        r = {"dir": dirs[name], "settings": st, "differs_from_ref": [k for k in SETTINGS if st.get(k) != ref_set.get(k)],
             "starts": len(a), "success": float(np.mean([v["success"] for v in a.values()])) if a else None,
             "moves_per_attempt": float(np.mean([v["moves"] for v in a.values()])) if a else None,
             "by_group": {g: {"starts": sum(1 for v in a.values() if v["group"] == g),
                              "success": float(np.mean([v["success"] for v in a.values() if v["group"] == g]))
                              if any(v["group"] == g for v in a.values()) else None} for g in groups}}
        if name != rname:
            r["paired_vs_ref"] = paired(ref, a)
            # start 0 only: the same protocol in every run (runs before 2026-11-19 did not reset the state before start 1)
            r["paired_vs_ref_start0"] = paired(ref, a, [k for k in a if k[1] == 0])
            r["paired_vs_ref_by_group"] = {g: paired(ref, a, [k for k, v in a.items() if v["group"] == g]) for g in groups}
        res["arms"][name] = r
    return res


def show(res: dict) -> str:
    ref = res["reference"]
    lines = [f"paired against {ref} on identical starts; arm+ = arm succeeds where {ref} fails, ref+ = the reverse; "
             f"p = exact McNemar, p_cl = cluster sign-flip",
             f"{'arm':16s} {'starts':>6s} {'success':>8s} {'moves':>6s} | {'n':>4s} {'ref':>8s} {'arm':>6s} {'arm+':>5s} {'ref+':>5s} {'p':>9s} {'p_cl':>9s}"]
    for name, r in res["arms"].items():
        if not r["starts"]:
            lines.append(f"{name:16s} no judged starts")
            continue
        s = f"{name:16s} {r['starts']:6d} {100 * r['success']:7.1f}% {r['moves_per_attempt']:6.2f}"
        if "paired_vs_ref" in r:
            p = r["paired_vs_ref"]
            s += (f" | {p['n']:4d} {100 * (p['ref'] or 0):7.1f}% {100 * (p['arm'] or 0):5.1f}% "
                  f"{p['arm_wins']:5d} {p['ref_wins']:5d} {p['mcnemar_p']:9.2e} {p['cluster_signflip_p']:9.2e}")
        if r["differs_from_ref"]:
            s += f"   (settings differing from {ref}: {', '.join(r['differs_from_ref'])})"
        lines.append(s)
        if "paired_vs_ref_start0" in r:
            p0 = r["paired_vs_ref_start0"]
            lines.append(f"{'':16s} {'start 0 only':>24s} | {p0['n']:4d} {100 * (p0['ref'] or 0):7.1f}% {100 * (p0['arm'] or 0):5.1f}% "
                         f"{p0['arm_wins']:5d} {p0['ref_wins']:5d} {p0['mcnemar_p']:9.2e} {p0['cluster_signflip_p']:9.2e}")
    if res.get("missing_arms"):
        lines.append(f"missing arms (no result files): {', '.join(res['missing_arms'])}")
    lines.append("")
    for name, r in res["arms"].items():
        if "paired_vs_ref_by_group" not in r:
            continue
        for g, p in r["paired_vs_ref_by_group"].items():
            if p["n"]:
                lines.append(f"  {name:16s} {g:40s} n {p['n']:4d}  {ref} {100 * p['ref']:5.1f}%  arm {100 * p['arm']:5.1f}%  "
                             f"arm+ {p['arm_wins']:3d} ref+ {p['ref_wins']:3d}  p {p['mcnemar_p']:.2e}  p_cl {p['cluster_signflip_p']:.2e} "
                             f"({p['clusters']} clusters)")
    return "\n".join(lines)


def boot_ci(succ: list[float], clusters: list, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the success rate, resampling whole clusters."""
    ids = sorted(set(clusters))
    idx = {c: i for i, c in enumerate(ids)}
    k = np.array([idx[c] for c in clusters])
    s_c = np.bincount(k, weights=np.asarray(succ, float), minlength=len(ids))
    n_c = np.bincount(k, minlength=len(ids)).astype(float)
    d = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    r = s_c[d].sum(1) / n_c[d].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def cell_of(grasp: str) -> tuple[int, int] | None:
    """(cell index, pair index i) from a make_combo_grasps name <port>_c<cell><i:03d>."""
    tail = grasp.rsplit("_", 1)[-1]
    if len(tail) != 5 or tail[0] != "c" or not tail[1:].isdigit():
        return None
    return int(tail[1]), int(tail[2:])


def load_cells(d: str) -> tuple[dict, dict]:
    """-> ({(cell, i, start idx): {success, start, port, seed}}, {cell: counts of unusable / crashed grasps and unplaced starts})."""
    out, counts = {}, {}
    for f in sorted(glob.glob(f"{d}/*/*.json")):
        j = json.loads(Path(f).read_text())
        ci = cell_of(j.get("grasp", ""))
        if ci is None:
            continue
        c = counts.setdefault(ci[0], {"unusable_grasps": 0, "crashed_grasps": 0, "unplaced_starts": 0})
        if j.get("error"):
            c["unusable_grasps" if j.get("error_kind") == "setup" else "crashed_grasps"] += 1
            continue
        for st in j.get("starts", []):
            if st.get("placement_failed"):
                c["unplaced_starts"] += 1
            a = st.get("attempt")
            if a and a.get("success") is not None:
                out[(ci[0], ci[1], int(st["idx"]))] = {"success": bool(a["success"]), "port": j["port"],
                                                       "start": json.dumps(st["start"], sort_keys=True),
                                                       "seed": j.get("episode_seed_used", j.get("episode_seed"))}
    return out, counts


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values (family = the given list)."""
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def t_ci(succ: list[float], clusters: list) -> tuple[float, float]:
    """Cluster-robust 95 % interval with t on (clusters - 1) degrees of freedom (as eval_policy320's cluster_rate)."""
    from scipy.stats import t as student
    ids = sorted(set(clusters))
    y = np.array([sum(s for s, c in zip(succ, clusters) if c == k) for k in ids], float)
    m = np.array([sum(1 for c in clusters if c == k) for k in ids], float)
    G, r = len(ids), y.sum() / m.sum()
    if G < 2:
        return float(r), float(r)
    se = np.sqrt(G / (G - 1) * np.sum((y - r * m) ** 2)) / m.sum()
    q = student.ppf(0.975, G - 1)
    return float(max(0.0, r - q * se)), float(min(1.0, r + q * se))


def cells(arms: dict[str, str]) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from make_combo_grasps import CELLS
    res = {"cells": list(CELLS), "arms": {}}
    for name, d in arms.items():
        rows, counts = load_cells(d)
        per = {}
        for ci, cell in enumerate(CELLS):
            ks = [k for k in rows if k[0] == ci]
            base = counts.get(ci, {"unusable_grasps": 0, "crashed_grasps": 0, "unplaced_starts": 0})
            if not ks:
                per[cell] = {"starts": 0, **base}
                continue
            succ = [rows[k]["success"] for k in ks]
            cl = [rows[k]["port"] if "shape" in cell else k[1] for k in ks]
            lo, hi = boot_ci(succ, cl)
            tlo, thi = t_ci(succ, cl)
            per[cell] = {"starts": len(ks), **base, "clusters": len(set(cl)), "successes": int(sum(succ)),
                         "success": float(np.mean(succ)), "ci95": [lo, hi], "ci95_t": [tlo, thi]}
        eff = {}
        for a_i, a in enumerate(CELLS):
            fa = set(a.split("+")) if a != "none" else set()
            for b_i, b in enumerate(CELLS):
                fb = set(b.split("+")) if b != "none" else set()
                if not (fa < fb and len(fb - fa) == 1):
                    continue
                both = [(k, (b_i, k[1], k[2])) for k in rows if k[0] == a_i and (b_i, k[1], k[2]) in rows]
                pairs = [(ka, kb) for ka, kb in both if rows[ka]["start"] == rows[kb]["start"]]
                key = f"{a} -> {b} (+{next(iter(fb - fa))})"
                if not pairs:
                    eff[key] = {"judged_in_both": len(both), "pairs": 0, "error": "no pairs with the same start"}
                    continue
                diff = [float(rows[kb]["success"]) - float(rows[ka]["success"]) for ka, kb in pairs]
                shape_cell = "shape" in fa or "shape" in fb
                cl = [rows[kb]["port"] if shape_cell else ka[1] for ka, kb in pairs]
                per_c = {}
                for c, x in zip(cl, diff):
                    per_c[c] = per_c.get(c, 0.0) + x
                eff[key] = {"judged_in_both": len(both), "pairs": len(pairs),
                            "same_seed_share": float(np.mean([rows[ka]["seed"] == rows[kb]["seed"] for ka, kb in pairs])),
                            "rate_without": float(np.mean([rows[ka]["success"] for ka, _ in pairs])),
                            "rate_with": float(np.mean([rows[kb]["success"] for _, kb in pairs])),
                            "mean_diff": float(np.mean(diff)), "clusters": len(per_c),
                            "signflip_p": cluster_signflip_p(list(per_c.values()))}
        ok = [k for k, e in eff.items() if e.get("pairs")]
        for k, ph in zip(ok, holm([eff[k]["signflip_p"] for k in ok])):
            eff[k]["holm_p"] = ph
        res["arms"][name] = {"dir": d, "cells": per, "factor_effects": eff}
    return res


def show_cells(res: dict) -> str:
    lines = []
    for name, r in res["arms"].items():
        lines.append(f"== {name}")
        lines.append(f"   {'cell':22s} {'k/N':>9s} {'unusable':>8s} {'crashed':>7s} {'unplaced':>8s} {'clusters':>8s} {'success':>8s}  "
                     f"95 % bootstrap   95 % t")
        for cell, x in r["cells"].items():
            if x["starts"]:
                lines.append(f"   {cell:22s} {str(x['successes']) + '/' + str(x['starts']):>9s} {x['unusable_grasps']:8d} {x['crashed_grasps']:7d} "
                             f"{x['unplaced_starts']:8d} {x['clusters']:8d} {100 * x['success']:7.1f}%  [{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"
                             f"   [{100 * x['ci95_t'][0]:.1f}, {100 * x['ci95_t'][1]:.1f}]")
            else:
                lines.append(f"   {cell:22s} {'0/0':>9s} {x['unusable_grasps']:8d} {x['crashed_grasps']:7d} {x['unplaced_starts']:8d}")
        lines.append("   factor effects (paired by pair index and start; success with - without the unseen factor; Holm over the contrasts):")
        for k, e in r["factor_effects"].items():
            if not e.get("pairs"):
                lines.append(f"     {k:44s} NO PAIRS ({e['judged_in_both']} judged in both)")
                continue
            lines.append(f"     {k:44s} pairs {e['pairs']:4d}/{e['judged_in_both']:4d}  {100 * e['rate_without']:5.1f}% -> {100 * e['rate_with']:5.1f}%  "
                         f"{100 * e['mean_diff']:+6.1f} points  p {e['signflip_p']:.2e} (Holm {e['holm_p']:.2e}, {e['clusters']} clusters, "
                         f"same seed {100 * e['same_seed_share']:.0f}%)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pbins")
    a.add_argument("dir")
    a.add_argument("--json")
    cc = sub.add_parser("cells")
    cc.add_argument("dir")
    cc.add_argument("--name", default="wm")
    cc.add_argument("--arm", action="append", default=[])
    cc.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--arm", action="append", default=[])
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "cells":
        arms = {args.name: args.dir, **dict(x.split("=", 1) for x in args.arm)}
        r = cells(arms)
        Path(args.out).write_text(json.dumps(r, indent=1))
        print(show_cells(r))
        return
    if args.cmd == "pbins":
        r = pbins(args.dir)
        if args.json:
            Path(args.json).write_text(json.dumps(r, indent=1))
        print(",".join(f"{x:.4f}" for x in r["p"]))
    else:
        r = compare(args.ref, args.arm)
        Path(args.out).write_text(json.dumps(r, indent=1))
        print(show(r))


if __name__ == "__main__":
    main()
