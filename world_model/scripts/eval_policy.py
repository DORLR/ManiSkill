"""Closed-loop evaluation of the latent FM policy; then a summary by shape group and grasp condition.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/eval_policy.py --vae $Q/runs/vae_v1/best.pt \
        --policy $Q/runs/policy_v1/best.pt --split ood_val --out $Q/runs/eval_policy_v1_ood --workers 8

--split test (A-D) is run once, at the end.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wm.eval import rollout


def summarise(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cells = defaultdict(lambda: [0, 0])
    skipped = sum(not r.get("expert_ok") for r in rows)
    for r in rows:
        for e in r["episodes"]:
            for key in (r["condition"], "ALL"):
                cells[key][0] += e["outcome"] == "inserted"
                cells[key][1] += 1
    print(f"grasps {len(rows)} (expert could not insert: {skipped}; errors: {sum('error' in r for r in rows)})")
    for k, (s, n) in cells.items():
        print(f"  {k:13s} inserted {s}/{n} = {s / max(n, 1):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", choices=["ood_val", "test"], default="ood_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grasps-per-cell", type=int, default=2)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--wm", help="world-model checkpoint: select among FM candidates by predicted progress")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--max-decisions", type=int, default=rollout.MAX_DECISIONS, help="0.4 s each")
    args = ap.parse_args()
    out = Path(args.out)
    rollout.run(args.vae, args.policy, args.split, out, args.workers, args.grasps_per_cell, args.starts, args.only, args.max_decisions,
                args.wm, args.candidates)
    summarise(out / "results.jsonl")


if __name__ == "__main__":
    main()
