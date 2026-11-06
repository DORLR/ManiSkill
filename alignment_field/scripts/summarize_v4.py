"""Table of v4 closed-loop results: insertion rate per role x condition x method.

python scripts/summarize_v4.py --eval results/eval_v4 --out results/summary_v4.json
"""
import argparse
import glob
import json
import math
from collections import defaultdict

METHODS = ("fm_id", "fm", "field+fm", "oracle+fm", "point+fm", "pointoracle+fm", "latent+fm", "fieldfuse+fm", "fieldplug+fm",
           "bc_id", "bc", "field+bc", "oracle+bc", "point+bc", "pointoracle+bc", "latent+bc", "fieldfuse+bc", "fieldplug+bc")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", default=["results/eval_v4"], help="one or more result directories")
    ap.add_argument("--out", default="results/summary_v4.json")
    args = ap.parse_args()
    cell, port_cell, skipped = defaultdict(list), defaultdict(list), defaultdict(int)
    for f in [x for e in args.eval for x in glob.glob(f"{e}/*.json")]:
        r = json.load(open(f))
        if not r.get("expert_ok"):
            skipped[(r["role"], r["condition"])] += 1
            continue
        for e in r["episodes"]:
            if "inserted" in e:
                cell[(r["role"], r["condition"], e["method"])].append(e["inserted"])
                port_cell[(r["role"], r["port"] if r["role"] == "unseen_shape" else "gen", e["method"])].append(e["inserted"])
    out = {}
    print(f"{'role':14}{'condition':14}" + "".join(f"{m:>22}" for m in METHODS) + "  skipped")
    for role in ("seen_shape", "unseen_shape"):
        for cond in ("train", "pitch_extrap", "pos_extrap", "combined"):
            row = []
            for m in METHODS:
                v = cell.get((role, cond, m), [])
                p, lo, hi = wilson(sum(v), len(v))
                out[f"{role}|{cond}|{m}"] = {"n": len(v), "rate": p, "ci": [lo, hi]}
                row.append(f"{p:6.2f} [{lo:.2f},{hi:.2f}] n{len(v):<3d}" if v else f"{'-':>22}")
            if any(cell.get((role, cond, m)) for m in METHODS):
                print(f"{role:14}{cond:14}" + "".join(f"{x:>22}" for x in row) + f"  {skipped.get((role, cond), 0)}")
    print()
    for (role, port, m), v in sorted(port_cell.items()):
        out[f"port|{role}|{port}|{m}"] = {"n": len(v), "rate": sum(v) / len(v)}
    for role, port in sorted({(r, p) for r, p, _ in port_cell}):
        print(f"{role:14}{port:6}" + "".join(
            f"{m}: {sum(port_cell[(role, port, m)])}/{len(port_cell[(role, port, m)])}   " for m in METHODS))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
