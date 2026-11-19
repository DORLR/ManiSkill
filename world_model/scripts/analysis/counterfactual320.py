"""B2 (2026-11-19, the user's request): does imagination rank the candidate actions as the simulator does?

Reads the `probes` of a run_wm320_attempts.py --mode wm --probe-max N run (attempt320's counterfactual probe): per
probed decision, for the n = n_cand candidates (the proposal 0 and the corrected copies 1..n-1; n = 13 without
excluded directions) and, where offered, the servo's move (index n): the model's mean score S_i, its imagined reward
sum G_i, and the true return R_i of really executing it for one decision and the ACT head for the rest of the
look-ahead (FK/H* labels, the training reward). R0_repeat = the proposal executed once more (simulator noise).

Per probe, over the n candidates:
    rho_S   = Spearman(S, R),  rho_G = Spearman(G, R)                       (skipped when R, S or G is constant)
    top1    = 1[R_{argmax S} >= max R - eps];  its chance = #{i: R_i >= max R - eps} / n (per probe)
    regret  = max_i R_i - R_pick                                            (pick = the model's executed choice)
Where the model corrected (pick > 0), with eps = EPS:
    better_wm     = 1[R_pick > R_0 + eps],     worse_wm     = 1[R_pick < R_0 - eps]
    better_random = mean_{i=1..n-1} 1[R_i > R_0 + eps],  worse_random likewise  (a random corrected copy, same states)
    better_servo  = 1[R_n > R_0 + eps]  (probes with the servo's move)
Where it did not (pick = 0): room = 1[max_{i>=1} R_i > R_0 + eps] (a better corrected copy existed).
Checks: |R0_repeat - R_0| (noise floor for eps), restore deviation of the hand (mm), and the deviation of the real
decision from its branch (first_step_dev_mm; exact only for pick = 0, where branch 0 ran from the untouched state).
All of it in total and per bin of the estimated gap g (common320.gap_bin). Non-finite values are left out of means.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.online.common320 import RANDOM_EDGES_MM, gap_bin  # noqa: E402

EPS = 0.05


def load_probes(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{d}/*/*.json")):
        j = json.loads(Path(f).read_text())
        for s in j.get("starts", []):
            a = s.get("attempt") or {}
            for p in a.get("probes") or []:
                out.append({**p, "grasp": j["grasp"], "start": s["idx"], "success": a.get("success")})
    return out


def per_probe(p: dict, eps: float) -> dict:
    from scipy.stats import spearmanr
    n = int(p.get("n_cand", len(p["S"])))
    S, G, R = np.asarray(p["S"][:n], float), np.asarray(p["G_img"][:n], float), np.asarray(p["R"][:n], float)
    pick = int(p["pick"])
    r = {"bin": gap_bin(float(p["g"])), "n_cand": n, "corrected": pick > 0, "regret": float(R.max() - R[pick]),
         "top1": bool(R[int(np.argmax(S))] >= R.max() - eps), "top1_chance": float(np.mean(R >= R.max() - eps))}
    if np.ptp(R) > 1e-9 and np.ptp(S) > 1e-12:
        r["rho_S"] = float(spearmanr(S, R)[0])
    if np.ptp(R) > 1e-9 and np.ptp(G) > 1e-12:
        r["rho_G"] = float(spearmanr(G, R)[0])
    if "R0_repeat" in p:
        r["noise_R0"] = float(abs(p["R0_repeat"] - R[0]))
    if pick > 0:
        r["better_wm"], r["worse_wm"] = bool(R[pick] > R[0] + eps), bool(R[pick] < R[0] - eps)
        r["better_random"], r["worse_random"] = float(np.mean(R[1:] > R[0] + eps)), float(np.mean(R[1:] < R[0] - eps))
        r["gain_wm"], r["gain_random"] = float(R[pick] - R[0]), float(np.mean(R[1:] - R[0]))
    else:
        r["room"] = bool(R[1:].max() > R[0] + eps)
    if p.get("servo") and len(p["R"]) > n:
        r["better_servo"], r["worse_servo"] = bool(p["R"][n] > R[0] + eps), bool(p["R"][n] < R[0] - eps)
        r["gain_servo"] = float(p["R"][n] - R[0])
    r["restore_dev_mm"] = float(p.get("restore_dev_mm", np.nan))
    r["first_step_dev_mm"] = float(p.get("first_step_dev_mm", np.nan))
    r["repeat_first_dev_mm"] = float(p.get("repeat_first_dev_mm", np.nan))
    return r


def agg(rows: list[dict]) -> dict:
    def m(key, sel=None):
        v = [float(r[key]) for r in rows if key in r and (sel is None or sel(r)) and np.isfinite(float(r[key]))]
        return (float(np.mean(v)), len(v)) if v else (None, 0)
    out = {"probes": len(rows)}
    for key in ("rho_S", "rho_G", "top1", "top1_chance", "regret", "better_wm", "worse_wm", "better_random", "worse_random", "gain_wm",
                "gain_random", "room", "better_servo", "worse_servo", "gain_servo"):
        out[key], out[key + "_n"] = m(key)
    out["corrected_share"] = float(np.mean([r["corrected"] for r in rows])) if rows else None
    def dist(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "n": int(len(v))} if len(v) else None
    out["noise_R0"] = dist([r["noise_R0"] for r in rows if "noise_R0" in r])
    out["restore_dev_mm"] = dist([r["restore_dev_mm"] for r in rows])
    out["first_step_dev_mm_pick0"] = dist([r["first_step_dev_mm"] for r in rows if not r["corrected"]])
    out["first_step_dev_mm_corrected"] = dist([r["first_step_dev_mm"] for r in rows if r["corrected"]])
    out["repeat_first_dev_mm"] = dist([r["repeat_first_dev_mm"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=EPS)
    args = ap.parse_args()
    probes = load_probes(args.dir)
    if not probes:
        raise SystemExit(f"no probes in {args.dir}")
    rows = [per_probe(p, args.eps) for p in probes]
    res = {"dir": args.dir, "eps": args.eps, "all": agg(rows),
           "by_gap_bin": {f"bin{b} (edges {list(RANDOM_EDGES_MM)})": agg([r for r in rows if r["bin"] == b]) for b in range(len(RANDOM_EDGES_MM) + 1)}}
    Path(args.out).write_text(json.dumps(res, indent=1))
    a = res["all"]
    print(f"{a['probes']} probes, model corrected at {a['corrected_share']}")
    print(f"ranking: Spearman(S, R) {a['rho_S']} (n {a['rho_S_n']}), Spearman(G, R) {a['rho_G']}; top-1 {a['top1']} (chance {a['top1_chance']}); "
          f"regret {a['regret']}")
    print(f"when the model corrected: better than the proposal {a['better_wm']} / worse {a['worse_wm']} (n {a['better_wm_n']}); "
          f"a random corrected copy at the same states: better {a['better_random']} / worse {a['worse_random']}; mean gain WM {a['gain_wm']} "
          f"vs random {a['gain_random']}")
    print(f"servo's move: better {a['better_servo']} / worse {a['worse_servo']} (n {a['better_servo_n']}), mean gain {a['gain_servo']}")
    print(f"when the model kept the proposal: a better corrected copy existed at {a['room']} (n {a['room_n']})")
    print(f"checks: |R0_repeat - R0| {a['noise_R0']} (eps {args.eps}); restore deviation {a['restore_dev_mm']}; real decision vs its branch: "
          f"pick 0 {a['first_step_dev_mm_pick0']}, corrected {a['first_step_dev_mm_corrected']}; repeated proposal {a['repeat_first_dev_mm']}")
    for b, x in res["by_gap_bin"].items():
        print(f"  {b}: probes {x['probes']}, corrected {x['corrected_share']}, rho_S {x['rho_S']}, better WM {x['better_wm']} "
              f"vs random {x['better_random']}, servo {x['better_servo']}")


if __name__ == "__main__":
    main()
