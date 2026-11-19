"""Insertion speed (2026-11-19, the user's request; revised after two reviews) from run_wm320_attempts.py results that
log, per executed decision, whether the simulator-truth success criterion held (`truth_ok`; attempt320, reporting only).

Per attempt (DT = 0.4 s per decision, CAP = 60 s = the 150-decision limit):
    T      = DT * (start of the final unbroken run of truth_ok == True), for attempts that END in success (the lasting
             insertion; an attempt seated, lost and re-seated counts from the re-seating); inf otherwise. A success
             reached only during the final 4-step hold gets T = DT * decisions (counted).
    T_first = DT * first_success_decision (first touch; secondary).
Per arm: F(t) = P(T <= t) on the 0.4 s grid (BUDGETS_S); RMST = E[min(T, CAP)] (failures count as CAP; so with
different success rates dRMST mixes success and speed -- read it with F(t)); median T over all attempts (the smallest
t with F(t) >= 0.5; "not reached" otherwise); mean T and mean first touch over successes; realised corrections per
attempt; agent latency per decision (median, p95; the first decision of every attempt is warm-up and left out; arms run
with the counterfactual probe give none). T is simulated robot time (the simulation waits while the agent computes);
T_wall = T + the agent's computation time up to the insertion (mean over successes) shows what a synchronous
controller would take.
Paired with the reference (same grasp, start index and start pose):
    dRMST = mean(min(T_arm, CAP) - min(T_ref, CAP)) with a cluster-bootstrap 95 % interval and a cluster sign-flip p;
    net benefit (who inserted first) = P(T_ref < T_arm) - P(T_ref > T_arm) (both inf = tie), same interval and test;
    where both succeed: shares ref faster / tied / arm faster (descriptive, no p: the subset depends on the agents).
Clusters: primary the port (grasps of one port share the shape), secondary the grasp. Monte-Carlo p = (1 + b) / (1 + B).
Every arm must hold result files with `truth_ok` (written after 2026-11-19 14:00); older files raise an error.

    python world_model/scripts/analysis/speed320.py --ref wm=DIR --arm act=DIR --arm random=DIR --out speed.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DT, CAP_S = 0.4, 60.0
BUDGETS_S = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0, 60.0)       # multiples of DT
BOOT, DRAWS = 2000, 200_000


def final_run_start(ok: list[bool]) -> int | None:
    """Number of decisions after which the final unbroken run of True began (1-based), None if the list ends False."""
    if not ok or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i + 1


def load(d: str) -> tuple[dict, dict]:
    out = {}
    chk = {"attempts": 0, "successes": 0, "crashed_or_unusable_grasps": 0, "unplaced_starts": 0, "success_only_in_hold": 0,
           "reseated_after_loss": 0, "touched_then_failed": 0, "modes": [], "model_steps": []}
    files = sorted(glob.glob(f"{d}/*/*.json"))
    if not files:
        raise SystemExit(f"no result files in {d}")
    for f in files:
        j = json.loads(Path(f).read_text())
        if j.get("error") or "starts" not in j:
            chk["crashed_or_unusable_grasps"] += 1
            continue
        for s in j["starts"]:
            a = s.get("attempt")
            if s.get("placement_failed"):
                chk["unplaced_starts"] += 1
            if not a or a.get("success") is None:
                continue
            if "truth_ok" not in a:
                raise SystemExit(f"{f}: no per-decision truth (result written before the speed logging): rerun this arm")
            if len(a["truth_ok"]) != a["decisions"]:
                raise SystemExit(f"{f}: {len(a['truth_ok'])} truth checks for {a['decisions']} decisions")
            key = (j["grasp"], int(s["idx"]))
            if key in out:
                raise SystemExit(f"{f}: attempt {key} appears twice in {d}")
            for name, v in (("modes", a.get("mode")), ("model_steps", a.get("model_step"))):
                if v not in chk[name]:
                    chk[name].append(v)
            chk["attempts"] += 1
            ok, fs = a["truth_ok"], a.get("first_success_decision")
            if a["success"]:
                chk["successes"] += 1
                st = final_run_start(ok)
                if st is None:
                    chk["success_only_in_hold"] += 1
                    st = a["decisions"]
                if fs is not None and st != fs:
                    chk["reseated_after_loss"] += 1
                T, Tf = DT * st, DT * (fs if fs is not None else st)
            else:
                if fs is not None:
                    chk["touched_then_failed"] += 1
                T = Tf = np.inf
            moves = sum(1 for x in a.get("per_decision") or [] if x.get("pick", 0) != 0)
            lat = [] if a.get("probes") else [ms for kk, ms in (a.get("latency_ms") or [])[1:]]     # first: warm-up
            comp = (sum(ms for kk, ms in (a.get("latency_ms") or []) if kk < T / DT - 1e-9) / 1000.0) if np.isfinite(T) else np.inf
            out[key] = {"T": T, "T_first": Tf, "T_wall": T + comp, "start": json.dumps(s["start"], sort_keys=True), "port": j["port"],
                        "grasp": j["grasp"], "moves": moves, "lat": lat}
    if len(chk["modes"]) > 1 or len(chk["model_steps"]) > 1:
        raise SystemExit(f"{d} mixes modes {chk['modes']} or model steps {chk['model_steps']}")
    return out, chk


def signflip(per: list[float], seed: int = 0) -> float:
    d = np.asarray([x for x in per if x != 0], float)
    if len(d) == 0:
        return 1.0
    T = abs(d.sum())
    if len(d) <= 16:
        s = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
        return float(np.mean(np.abs(s @ d) >= T - 1e-9))
    rng, hits = np.random.default_rng(seed), 0
    for start in range(0, DRAWS, 20_000):                 # in chunks: small memory
        s = rng.integers(0, 2, size=(min(20_000, DRAWS - start), len(d)), dtype=np.int8) * 2 - 1
        hits += int(np.sum(np.abs(s @ d) >= T - 1e-9))
    return float((1 + hits) / (1 + DRAWS))


def boot_ci(x: np.ndarray, cl: list, seed: int = 0) -> list[float]:
    ids = sorted(set(cl))
    if len(ids) < 10:                                      # too few clusters for a percentile bootstrap
        return [float("nan"), float("nan")]
    k = np.array([ids.index(c) for c in cl])
    sums, cnt = np.bincount(k, weights=x, minlength=len(ids)), np.bincount(k, minlength=len(ids)).astype(float)
    d = np.random.default_rng(seed).integers(len(ids), size=(BOOT, len(ids)))
    m = sums[d].sum(1) / cnt[d].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def arm_stats(a: dict) -> dict:
    if not a:
        return {"attempts": 0}
    T = np.array([v["T"] for v in a.values()])
    ok = np.isfinite(T)
    lat = np.concatenate([np.asarray(v["lat"], float) for v in a.values()])
    srt = np.sort(T)
    m = srt[int(np.ceil(len(srt) / 2)) - 1]                 # the smallest t with F(t) >= 0.5
    med = float(m) if np.isfinite(m) else None
    return {"attempts": int(len(T)), "success": float(ok.mean()),
            "mean_wall_s_successes": float(np.mean([v["T_wall"] for v in a.values() if np.isfinite(v["T"])])) if ok.any() else None,
            "within": {f"{t:g}": float(np.mean(T <= t + 1e-9)) for t in BUDGETS_S},
            "rmst_s": float(np.mean(np.minimum(T, CAP_S))) if len(T) else None,
            "median_s_all": med, "mean_s_successes": float(T[ok].mean()) if ok.any() else None,
            "mean_first_touch_s_successes": float(np.mean([v["T_first"] for v in a.values() if np.isfinite(v["T"])])) if ok.any() else None,
            "moves_per_attempt": float(np.mean([v["moves"] for v in a.values()])) if a else None,
            "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "n": int(len(lat))} if len(lat) else None}


def paired(ref: dict, arm: dict) -> dict:
    common = [k for k in ref if k in arm]
    ks = [k for k in common if ref[k]["start"] == arm[k]["start"]]
    if not ks:
        return {"pairs": 0, "start_mismatch": len(common), "unpaired_ref": len(ref), "unpaired_arm": len(arm)}
    tr = np.array([ref[k]["T"] for k in ks]); ta = np.array([arm[k]["T"] for k in ks])
    d = np.minimum(ta, CAP_S) - np.minimum(tr, CAP_S)
    win = np.where(tr < ta, 1.0, np.where(tr > ta, -1.0, 0.0))            # +1: the reference inserted first
    out = {"pairs": len(ks), "start_mismatch": len(common) - len(ks), "unpaired_ref": len(ref) - len(ks), "unpaired_arm": len(arm) - len(ks),
           "d_rmst_s": float(d.mean()), "net_benefit_ref": float(win.mean())}
    for name, key in (("port", "port"), ("grasp", "grasp")):
        cl = [ref[k][key] for k in ks]
        per_d, per_w = {}, {}
        for c, x, w in zip(cl, d, win):
            per_d[c] = per_d.get(c, 0.0) + x
            per_w[c] = per_w.get(c, 0.0) + w
        out[f"by_{name}"] = {"clusters": len(per_d), "d_rmst_ci95": boot_ci(d, cl), "d_rmst_p": signflip(list(per_d.values())),
                             "net_benefit_ci95": boot_ci(win, cl), "net_benefit_p": signflip(list(per_w.values()))}
    both = np.isfinite(tr) & np.isfinite(ta)
    if both.any():
        out["both_succeed"] = int(both.sum())
        out["ref_faster"], out["tied"], out["arm_faster"] = (float(np.mean(tr[both] < ta[both])), float(np.mean(tr[both] == ta[both])),
                                                              float(np.mean(tr[both] > ta[both])))
        out["mean_dT_s_both"] = float(np.mean(ta[both] - tr[both]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rname, rdir = args.ref.split("=", 1)
    arms = {rname: rdir}
    for x in args.arm:
        n, d = x.split("=", 1)
        if n in arms:
            raise SystemExit(f"arm name {n} given twice")
        arms[n] = d
    data, res = {}, {"dt_s": DT, "cap_s": CAP_S, "budgets_s": list(BUDGETS_S), "reference": rname, "arms": {}}
    for name, d in arms.items():
        data[name], chk = load(d)
        res["arms"][name] = {"dir": d, "checks": chk, **arm_stats(data[name])}
        if name != rname:
            res["arms"][name]["paired_vs_ref"] = paired(data[rname], data[name])
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"lasting insertion within a time budget (simulator truth; {DT} s per decision; F(60 s) = success rate)")
    print(f"{'arm':10s}" + "".join(f"{t:>7g}s" for t in BUDGETS_S) + "   RMST60  median(all)  mean(succ)  +compute  moves  latency ms (median / p95)")
    nan = float("nan")
    for name, r in res["arms"].items():
        if not r["attempts"]:
            print(f"{name:10s} NO ATTEMPTS ({r['checks']})")
            continue
        lat = r["latency_ms"] or {"median": nan, "p95": nan}
        med = f"{r['median_s_all']:.1f}s" if r["median_s_all"] is not None else "not reached"
        print(f"{name:10s}" + "".join(f"{100 * r['within'][f'{t:g}']:7.1f}%" for t in BUDGETS_S)
              + f"   {r['rmst_s']:5.1f}s  {med:>11s}  {r['mean_s_successes'] if r['mean_s_successes'] is not None else nan:8.1f}s  "
                f"{r['mean_wall_s_successes'] if r['mean_wall_s_successes'] is not None else nan:7.1f}s  {r['moves_per_attempt']:5.1f}  "
                f"{lat['median']:.0f} / {lat['p95']:.0f}")
    print(f"paired with {rname}: dRMST = arm - {rname} [s]; net = P({rname} inserts first) - P(arm first); clusters by port (by grasp)")
    for name, r in res["arms"].items():
        p = r.get("paired_vs_ref")
        if not p:
            continue
        if not p["pairs"]:
            print(f"  {name:10s} WARNING: no pairs with {rname} ({p['start_mismatch']} with a different start, {p['unpaired_arm']} unmatched)")
            continue
        bp, bg = p["by_port"], p["by_grasp"]
        line = (f"  {name:10s} pairs {p['pairs']:4d} ({bp['clusters']} ports, {bg['clusters']} grasps; start mismatch {p['start_mismatch']})  dRMST {p['d_rmst_s']:+.2f}s [{bp['d_rmst_ci95'][0]:+.2f}, {bp['d_rmst_ci95'][1]:+.2f}] "
                f"p {bp['d_rmst_p']:.1e} ({bg['d_rmst_p']:.1e})  net {p['net_benefit_ref']:+.3f} [{bp['net_benefit_ci95'][0]:+.3f}, "
                f"{bp['net_benefit_ci95'][1]:+.3f}] p {bp['net_benefit_p']:.1e} ({bg['net_benefit_p']:.1e})")
        if "both_succeed" in p:
            line += (f"  | both succeed {p['both_succeed']}: {rname} faster {100 * p['ref_faster']:.0f}%, tied {100 * p['tied']:.0f}%, "
                     f"arm faster {100 * p['arm_faster']:.0f}%, mean dT {p['mean_dT_s_both']:+.2f}s")
        print(line)
    for name, r in res["arms"].items():
        print(f"  checks {name}: {r['checks']}")


if __name__ == "__main__":
    main()
