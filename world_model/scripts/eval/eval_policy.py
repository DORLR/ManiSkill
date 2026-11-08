"""Closed-loop evaluation of a policy (latent FM or latent-free baseline; the checkpoint says which),
then a summary by grasp condition. Protocol v3: 60 s per episode, success = within 1 mm of full
depth, seeded sampling, 4 grasps per shape and condition, 2 starts per grasp.

    python world_model/scripts/eval/eval_policy.py --vae $Q/runs/vae_v3/best.pt \
        --policy $Q/runs/policy_v3/best.pt --split ood_val --out $Q/runs/eval_policy_v3_ood --workers 6

--split test (ports A-D) is run once, at the very end.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wm.eval import rollout


def summarise(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cells = defaultdict(lambda: [0, 0])
    skipped = sum(not r.get("expert_ok") for r in rows)
    for r in rows:
        for e in r["episodes"]:
            for key in (r["condition"], "ALL"):
                cells[key][0] += e["outcome"] == "inserted"
                cells[key][1] += 1
    out = {"grasps": len(rows), "expert_could_not_insert": skipped, "errors": sum("error" in r for r in rows),
           "inserted": {k: [s, n] for k, (s, n) in cells.items()}}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", help="autoencoder (latent policies only)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--split", choices=["ood_val", "test"], default="ood_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grasps-per-cell", type=int, default=4)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--only", nargs="+", help="smoke tests only: these shapes")
    ap.add_argument("--max-decisions", type=int, default=rollout.MAX_DECISIONS, help="0.4 s each")
    args = ap.parse_args()
    out = Path(args.out)
    rollout.run(args.vae, args.policy, args.split, out, args.workers, args.grasps_per_cell, args.starts, args.only,
                args.max_decisions)
    (out / "summary.json").write_text(json.dumps(summarise(out / "results.jsonl"), indent=1))


if __name__ == "__main__":
    main()
