"""Collect training grasps in parallel (resumable: existing shards are skipped).

python scripts/collect.py --out data/train --per-port 60 --workers 14
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

_ROBOT = {}


def _work(spec):
    from paf.collect import safe_collect, save
    from paf.env_adapter import Robot
    port = spec["port"]
    if port not in _ROBOT:
        for r in _ROBOT.values():
            if r is not None:
                r.env.close()
        _ROBOT.clear()
        try:
            _ROBOT[port] = Robot(port)
        except Exception as exc:  # e.g. a generated shape the env cannot grasp at reset
            _ROBOT[port] = None
            _ROBOT["error"] = f"{type(exc).__name__}: {exc}"
    if _ROBOT[port] is None:
        data = {"meta": {**spec, "expert_ok": False, "failure": "env_construction: " + _ROBOT.get("error", "")}}
    else:
        data = safe_collect(_ROBOT[port], spec)
    meta = save(spec["path"], data)
    return {k: meta.get(k) for k in ("path", "expert_ok", "n", "seconds", "failure", "expert_failure",
                                       "port_axis_error_deg")}


def main():
    from paf import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--per-port", type=int, default=60)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--ports", nargs="+", default=list(C.PORTS))
    ap.add_argument("--seed0", type=int, default=100000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for pi, port in enumerate(args.ports):
        rng = np.random.default_rng(args.seed0 + pi)
        for i in range(args.per_port):
            pitch, pos = C.sample_train_grasp(rng)
            path = out / f"{port}_{i:03d}.npz"
            if path.exists():
                continue
            specs.append(dict(port=port, idx=i, seed=args.seed0 + 1000 * pi + i, pitch=pitch, grasp_pos=pos,
                              n_sweep=C.N_SWEEP, n_push=C.N_PUSH, path=str(path)))
    print(f"{len(specs)} grasps to collect with {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool, open(out / "log.jsonl", "a") as log:
        for k, row in enumerate(pool.imap(_work, specs, chunksize=1)):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s] {row}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
