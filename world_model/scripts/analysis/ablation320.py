"""Analysis of the in-hole agent's ablations (2026-11-19, the user's request): does imagination help, or any move?

Arms are result directories of run_wm320_attempts.py on the same split, grasps, starts and seeds (per-grasp files
<dir>/<port>/<grasp>.json). Two commands:

  pbins <wm dir> [--json <file>]
      the "wm" agent's correction rate per bin b of the estimated pin-tip gap g (common320.RANDOM_EDGES_MM):
          p_b = #(decisions with candidates offered, g in bin b, the model corrected) / #(decisions with candidates
                offered, g in bin b)
      printed as "p0,p1,p2,p3" (the --random-p of the ablation "random"); counts go to --json.

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

Held-out groups (demo320.split320's numbers): unseen shape (OOD_VAL_SHAPES) with grasp pitch and position outside the
held-out strips / inside a strip; seen shape with pitch in (-42, -38) deg only ("unseen angle"); with position in
(-21, -19) mm only ("unseen position"); both (no such grasp in wm_v1).
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
    strip_a, strip_x = -42.0 < pitch_deg < -38.0, -21.0 < pos_m * 1000.0 < -19.0
    if port in OOD_VAL_SHAPES:
        return "unseen shape, angle+position in strip" if (strip_a or strip_x) else "unseen shape, angle+position seen"
    if strip_a and strip_x:
        return "unseen angle AND position"
    if strip_a:
        return "unseen angle only"
    if strip_x:
        return "unseen position only"
    return "not held out"


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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pbins")
    a.add_argument("dir")
    a.add_argument("--json")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--arm", action="append", default=[])
    c.add_argument("--out", required=True)
    args = ap.parse_args()
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
