"""Mid-run health check of the closed-loop evaluation (partial results).

python scripts/quick_stats.py --eval results/eval
Prints per method: episodes, push success, crashes, IK failures, mean loop
iterations, median end distance |u| (sigma units), split by seen/unseen shape.
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/eval")
    args = ap.parse_args()
    files = glob.glob(f"{args.eval}/fold_*/*.json")
    st = defaultdict(lambda: defaultdict(list))
    n_grasp, n_expert_fail = 0, 0
    for f in files:
        r = json.load(open(f))
        n_grasp += 1
        if not r.get("expert_ok"):
            n_expert_fail += 1
            continue
        for e in r["episodes"]:
            k = (r["role"], e["method"])
            st[k]["crash"].append("failure" in e)
            if "push_success" in e:
                st[k]["push"].append(e["push_success"])
                st[k]["end"].append(np.linalg.norm(e["u_final_true"]))
                st[k]["start"].append(np.linalg.norm(e["u0_true"]))
            tr = e.get("traj", [])
            if tr:
                st[k]["iters"].append(len(tr))
                st[k]["ik"].append(any(t.get("ik_fail") for t in tr))
                st[k]["est_err"].append(np.linalg.norm(np.array(tr[-1]["u_est"]) - np.array(tr[-1]["u_true"])))
    print(f"grasps done {n_grasp}, expert could not insert {n_expert_fail}")
    print(f"{'role':7}{'method':10}{'n':>5}{'push':>7}{'crash':>7}{'ikfail':>7}{'iters':>7}"
          f"{'|u0|':>7}{'|u_end|':>8}{'est_err':>8}")
    for (role, m), d in sorted(st.items()):
        f = lambda key, fn=np.mean: (fn(d[key]) if d[key] else float("nan"))
        print(f"{role:7}{m:10}{len(d['push']):5d}{f('push'):7.2f}{f('crash'):7.2f}{f('ik'):7.2f}{f('iters'):7.1f}"
              f"{f('start', np.median):7.1f}{f('end', np.median):8.2f}{f('est_err', np.median):8.2f}")


if __name__ == "__main__":
    main()
