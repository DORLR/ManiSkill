"""End-to-end smoke test of v3 on a small slice of the real dataset (minutes, not hours): every stage
runs a few steps with the real configs, and reports seconds per step and peak GPU memory, so the
long runs start only after everything has worked once.

    python world_model/scripts/smoke_v3.py --out $Q/runs/smoke_v3
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from wm.data import index
from wm.data.kinematics import compute_commanded_hands
from wm.training import adapter, common, encode, policy, vae, wm

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "v3"
Q = "/mnt/storage2/qingxin"


def subset(entries, per_split: int = 12):
    """Long expert trajectories and contact trials from train and ood_val (so every dataset is non-empty)."""
    out = []
    for split in ("train", "ood_val"):
        for kind in ("insert", "perturbed", "contact", "reference"):
            es = [e for e in entries if e["split"] == split and e["kind"] == kind and e["n_images"] > 60 and e["R_P"]]
            out += es[:per_split // 4 + 1]
    return out


def stage(name, fn, report):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = fn()
    report[name] = {"seconds": round(time.time() - t0, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(name, json.dumps(report[name]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    data = f"{Q}/data/wm_v1"
    entries = subset(index.load(data))
    rep = {"trajectories": len(entries)}
    cfg = lambda name, **kw: {**common.load_config(CONFIGS / name), **kw}
    n = args.steps

    res = stage("compute_ee", lambda: compute_commanded_hands(entries, Path(data), out / "ee", workers=2), rep)
    rep["fk_check"] = res
    best = stage("vae", lambda: vae.train(cfg("vae.json", workers=4, log_every=5, eval_every=n, eval_batches=2),
                                          out / "vae", entries, max_steps=n), rep)
    rep["vae_log"] = (out / "vae" / "log.jsonl").read_text().splitlines()[-1]
    from wm.models.vae import load_vae
    scale = stage("encode", lambda: encode.run(load_vae(str(best)), entries, Path(data), out / "ee", out / "lat",
                                               torch.device("cuda"), workers=4), rep)
    rep["latent_scale"] = scale
    stage("adapter", lambda: adapter.train(cfg("adapter.json", workers=2, log_every=10, eval_every=n, eval_batches=2),
                                           out / "adapter", entries, max_steps=n), rep)
    for size in ("s", "b"):
        stage(f"wm_{size}", lambda: wm.train(cfg(f"wm_{size}.json", latents=str(out / "lat"), vae_checkpoint=str(best),
                                                 workers=2, log_every=5, eval_every=n, eval_batches=1),
                                             out / f"wm_{size}", entries, max_steps=n), rep)
        rep[f"wm_{size}_last"] = (out / f"wm_{size}" / "log.jsonl").read_text().splitlines()[-1][:600]
    stage("policy", lambda: policy.train(cfg("policy.json", latents=str(out / "lat"), workers=2, log_every=5,
                                                eval_every=n, eval_batches=2), out / "policy", entries, max_steps=n), rep)
    stage("baseline", lambda: policy.train(cfg("baseline.json", workers=4, log_every=1, eval_every=5, eval_batches=1),
                                           out / "baseline", entries, max_steps=5), rep)
    (out / "smoke.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
