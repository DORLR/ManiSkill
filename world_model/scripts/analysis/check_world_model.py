"""What does the world model predict? (wm.analysis.wm_checks) -> <out>/world_model_<tag>.json

    python world_model/scripts/analysis/check_world_model.py --latents $Q/data/wm_v3_latents \
        --vae $Q/runs/vae_v3/best.pt --wm $Q/runs/wm_v3_s/best.pt --adapter $Q/runs/adapter_v3/best.pt \
        --out $Q/runs/checks_v3 --tag s
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wm.analysis import probes, wm_checks
from wm.data import index
from wm.data.datasets import port_axes_ok
from wm.data.kinematics import build_fk
from wm.models.vae import load_vae
from wm.training.adapter import load_adapter
from wm.training.common import limit_gpu_memory
from wm.training.wm import load_world_model


def long_trajectories(data_root, latent_root, n_min_images: int, limit: int) -> list[dict]:
    """ood_val expert trajectories long enough for a 16 s imagined rollout."""
    out = []
    for e in index.load(data_root):
        if e["split"] != "ood_val" or e["kind"] not in ("insert", "perturbed") or not port_axes_ok(e):
            continue
        if e["n_images"] < n_min_images:
            continue
        p = Path(latent_root) / Path(e["path"]).relative_to(data_root)
        if p.exists():
            with np.load(p) as f:
                out.append({"lat": {k: f[k] for k in f.files}, "R_P": np.asarray(e["R_P"], np.float64)})
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/storage2/qingxin/data/wm_v1")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rollouts", type=int, default=256)
    args = ap.parse_args()
    limit_gpu_memory()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    rd = wm_checks.Reader(load_vae(args.vae, dev), load_world_model(args.wm, dev), scale, dev)
    entries = index.load(args.data)
    rep = {"wm": args.wm, "horizon_accuracy": wm_checks.horizon_accuracy(rd, args.latents, entries, args.data)}
    print(json.dumps(rep["horizon_accuracy"], indent=1), flush=True)
    rep["jam_separation"] = wm_checks.jam_separation(rd, probes.contact_trials(args.data, args.latents, "ood_val"))
    print(json.dumps(rep["jam_separation"], indent=1), flush=True)
    fk, robot = build_fk()
    trajs = long_trajectories(args.data, args.latents, 6 + 2 * 40 + 1, args.rollouts)
    rep["rollout_stability"] = wm_checks.rollout_stability(rd, load_adapter(args.adapter, dev), fk, trajs)
    robot.env.close()
    print(json.dumps(rep["rollout_stability"], indent=1), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"world_model_{args.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
