"""What does the latent keep? Jam-vs-insert classifiers and offset regressions on the contact trials
(wm.analysis.probes), written to <out>/latents.json, with the angle gate used by the pipeline.

    python world_model/scripts/analysis/check_latents.py --latents $Q/data/wm_v3_latents \
        --vae $Q/runs/vae_v3/best.pt --out $Q/runs/checks_v3
"""

import argparse
import json
from pathlib import Path

import torch

from wm.analysis import probes
from wm.models.vae import load_vae
from wm.training.common import limit_gpu_memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    limit_gpu_memory()
    torch.manual_seed(0)
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    tr = probes.contact_trials(args.data, args.latents, "train")
    te = probes.contact_trials(args.data, args.latents, "ood_val")
    dev = torch.device("cuda")
    vae = load_vae(args.vae, dev)
    rep = {"trials": {"train": [len(tr), sum(t["y"] for t in tr)], "ood_val": [len(te), sum(t["y"] for t in te)]},
           "jam_probes": probes.jam_probes(tr, te, scale)}
    rep["offset_probes"] = probes.offset_probes(tr, te, scale, vae, dev)
    rep["angle_gate"] = probes.angle_gate(rep["offset_probes"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latents.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["angle_gate"]))


if __name__ == "__main__":
    main()
