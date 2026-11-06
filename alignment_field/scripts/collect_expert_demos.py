"""Record IL demonstrations from Chongyu's expert (resumable).

python scripts/collect_expert_demos.py --out data/v4_demos --per-port 6 --val-per-port 4 --n-demo 3
Training shapes (G00-G44 that can be built) x train-band grasps; validation
shapes G46-G48 for model selection. A-D and extrapolated grasps are never used.
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

_R = {}


def _work(spec):
    from paf.env_adapter import Robot
    from paf.expert_demo import record_expert_demo
    try:
        if _R.get("port") != spec["port"]:
            if "robot" in _R:
                _R["robot"].env.close()
            _R["robot"], _R["port"] = Robot(spec["port"]), spec["port"]
        robot = _R["robot"]
        out, metas = {"rgb": [], "qpos": [], "act": [], "demo": []}, []
        for k in range(spec["n_demo"]):
            d, meta = record_expert_demo(robot, spec["seed"] + 104729 * k, spec["pitch"], spec["grasp_pos"])
            metas.append(meta)
            if d is None:
                continue
            for key in ("rgb", "qpos", "act"):
                out[key].append(d[key])
            out["demo"].append(np.full(len(d["act"]), k))
        meta = {**spec, "demos": metas}
        if out["act"]:
            np.savez_compressed(spec["path"], meta=json.dumps(meta), **{k: np.concatenate(v) for k, v in out.items()})
        else:
            np.savez(spec["path"], meta=json.dumps(meta))
        return {"path": spec["path"], "ok": sum(m["completed"] for m in metas), "frames": sum(len(a) for a in out["act"])}
    except Exception as exc:
        _R.pop("port", None)
        return {"path": spec["path"], "ok": 0, "error": f"{type(exc).__name__}: {exc}"}


def main():
    from paf import config as C
    from paf.ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v4_demos")
    ap.add_argument("--per-port", type=int, default=6)
    ap.add_argument("--val-per-port", type=int, default=4)
    ap.add_argument("--n-demo", type=int, default=3)
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bad, val = {"G19", "G42", "G45", "G49"}, ["G46", "G47", "G48"]
    train = [p for p in generated_specs() if p not in bad and p not in val]
    specs = []
    for pi, port in enumerate(train + val):
        n = args.val_per_port if port in val else args.per_port
        rng = np.random.default_rng(600000 + pi)
        for i in range(n):
            pitch, pos = C.sample_train_grasp(rng)
            path = out / f"{port}_{i:02d}.npz"
            if not path.exists():
                specs.append(dict(port=port, idx=i, seed=600000 + 1000 * pi + i, pitch=pitch, grasp_pos=pos,
                                  n_demo=args.n_demo, path=str(path)))
    specs.sort(key=lambda s: s["port"])
    print(f"{len(specs)} grasps, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, specs)):
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s] {r}", flush=True)


if __name__ == "__main__":
    main()
