"""Joint adapter on the contact trials of held-out shapes (wm.analysis.wm_checks.adapter_on_contact):
does it predict that the joints stop when a push is blocked? -> <out>/adapter.json

    python world_model/scripts/analysis/check_adapter.py --adapter $Q/runs/adapter_v3/best.pt --out $Q/runs/checks_v3
"""

import argparse
import json
from pathlib import Path

import torch

from wm.analysis.wm_checks import adapter_on_contact
from wm.training.adapter import load_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    rep = adapter_on_contact(load_adapter(args.adapter, dev), args.data, dev=dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
