"""Joint adapter on held-out contact exploration: error of the predicted joints (mrad) for the next 0.4 s, next
to "the joints reach the commands", for one or several adapter checkpoints.

    python world_model/scripts/analysis/check_adapter_on_exploration.py --data $Q/data/wm_v1x \
        --adapters $Q/runs/adapter_v3/best.pt $Q/runs/adapter_v4e/best.pt --out $Q/runs/checks_v4/explore/adapter.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from wm.data import index
from wm.data.datasets import AdapterDataset
from wm.training.adapter import evaluate, load_adapter

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--adapters", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = torch.device("cuda")
ds = AdapterDataset(index.load(args.data), "ood_val")
rep = {"windows": len(ds)}
for a in args.adapters:
    torch.manual_seed(0)
    loader = DataLoader(ds, batch_size=1024, shuffle=True, drop_last=True, num_workers=4)
    rep[a] = evaluate(load_adapter(a, dev), loader, dev, 60)
    print(a, {k: round(v, 4) if isinstance(v, float) else v for k, v in rep[a].items()})
json.dump(rep, open(args.out, "w"), indent=1)
