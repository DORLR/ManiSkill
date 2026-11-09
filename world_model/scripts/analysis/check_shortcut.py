"""Does the world model read the plug-hole misalignment from the expert's commands instead of the images?
(wm.analysis.wm_checks.shortcut) -> <out>/shortcut_<tag>.json

    python world_model/scripts/analysis/check_shortcut.py --latents $Q/data/wm_v3g_latents \
        --vae $Q/runs/vae_v3g/best.pt --wm $Q/runs/wm_v3g_s/best.pt --out $Q/runs/checks_v3g --tag s
"""

import argparse
import json
from pathlib import Path

import torch

from wm.analysis import probes, wm_checks
from wm.models.vae import load_vae
from wm.training.common import limit_gpu_memory
from wm.training.wm import load_world_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    limit_gpu_memory()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    rd = wm_checks.Reader(load_vae(args.vae, dev), load_world_model(args.wm, dev), scale, dev)
    rep = {"wm": args.wm, "shortcut": wm_checks.shortcut(rd, probes.contact_trials(args.data, args.latents, "ood_val"))}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"shortcut_{args.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
