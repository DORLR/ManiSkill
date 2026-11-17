"""The offline report (wm/tdmpc/report.py) of a trained model on whole splits of the branch set.

    python world_model/scripts/analysis/eval_branch_value.py --kind wm --ckpt <run>/last.pt \
        --data /mnt/storage3/qingxin/data/wm_v1b --splits dev,ood_val --out <run>/report_final.json
    ... --kind tdmpc2 --cache /mnt/storage3/qingxin/data/wm_v1b_c64
"""

import argparse
import json
from pathlib import Path

import torch

from wm.tdmpc import data, evaluate, original, report
from wm.tdmpc.model import LatentWM, make_cfg
from wm.training.common import limit_gpu_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["wm", "tdmpc2"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--splits", default="dev,ood_val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-branches", type=int)
    args = ap.parse_args()
    limit_gpu_memory()
    ck = torch.load(args.ckpt, map_location="cuda:0", weights_only=False)
    stats = ck["stats"]
    if args.kind == "wm":
        model = LatentWM(make_cfg(**ck["cfg"]), stats).cuda()
        model.load_state_dict(ck["model"])
    else:
        agent = original.make_agent(original.make_cfg(**{**ck["cfg"], "compile": False}))
        agent.model.load_state_dict(ck["model"])
    out = {"ckpt": args.ckpt, "step": ck["step"]}
    for split in args.splits.split(","):
        bs = data.list_branches(args.data, split)
        if args.max_branches:
            bs = bs[:args.max_branches]
        if not bs:
            continue
        recs = evaluate.score_wm(model, bs, stats, args.workers) if args.kind == "wm" else \
            evaluate.score_original(agent, original.load_cache(args.cache, bs), stats)
        out[split] = report.build(recs, bs)
        print(f"==== {split}\n{report.table(out[split])}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
