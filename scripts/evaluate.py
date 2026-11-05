"""Closed-loop evaluation of all folds in parallel (resumable).

python scripts/evaluate.py --runs runs --out results/eval --workers 15
Test grasps use seeds disjoint from training (900000+). Cells per fold:
  held-out shape : 6 conditions x 8 grasps x 4 OOD starts
  each seen shape: 6 conditions x 4 grasps x 4 OOD starts
conditions = train-band grasp, pitch interp/extrap, position interp/extrap,
and combined (pitch extrap + position extrap).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

CONDITIONS = {"train": ("train", "train"), "pitch_interp": ("interp", "train"),
              "pitch_extrap": ("extrap", "train"), "pos_interp": ("train", "interp"),
              "pos_extrap": ("train", "extrap"), "combined": ("extrap", "extrap")}
_CACHE = {}


def _work(spec):
    import torch
    from paf.env_adapter import Robot
    from paf.evaluate import eval_grasp
    from paf.policy import FieldPolicy
    torch.set_num_threads(1)
    if _CACHE.get("port") != spec["port"]:
        if "robot" in _CACHE:
            _CACHE["robot"].env.close()
        _CACHE["robot"] = Robot(spec["port"])
        _CACHE["port"] = spec["port"]
    key = ("pol", spec["fold"])
    if key not in _CACHE:
        run = spec["run_dir"]
        base = FieldPolicy(run, estimate="hat")
        _CACHE[key] = {"field": base, "field_pc": FieldPolicy(run, estimate="pc", share_from=base),
                       "oracle": FieldPolicy(run, estimate="hat", share_from=base), "none": None}
    try:
        res = eval_grasp(_CACHE["robot"], {m: p for m, p in _CACHE[key].items() if m in spec["methods"]}, spec)
    except Exception as exc:  # one bad grasp must not stop the whole evaluation
        return {"path": spec["path"], "error": f"{type(exc).__name__}: {exc}"}
    Path(spec["path"]).write_text(json.dumps(res))
    ok = [e.get("push_success") for e in res["episodes"]]
    return {"path": spec["path"], "expert_ok": res["expert_ok"], "seconds": res.get("seconds"),
            "n_ep": len(ok), "fail": sum(1 for e in res["episodes"] if "failure" in e)}


def main():
    from paf import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--folds", nargs="+", default=list(C.PORTS))
    ap.add_argument("--n-unseen", type=int, default=8)
    ap.add_argument("--n-seen", type=int, default=4)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--roles", nargs="+", default=["seen", "unseen"])
    ap.add_argument("--single-run", default=None,
                    help="v3: one model (trained on generated shapes) evaluated on A-D, all unseen")
    ap.add_argument("--methods", nargs="+", default=["field", "field_pc", "oracle", "none"])
    args = ap.parse_args()
    specs = []
    folds = ["gen"] if args.single_run else args.folds
    for fi, fold in enumerate(folds):
        for pi, port in enumerate(C.PORTS):
            role = "unseen" if (port == fold or args.single_run) else "seen"
            if role not in args.roles:
                continue
            n = args.n_unseen if role == "unseen" else args.n_seen
            for ci, (cond, (pk, xk)) in enumerate(CONDITIONS.items()):
                rng = np.random.default_rng(900000 + 10000 * fi + 1000 * pi + ci)
                for i in range(n):
                    pitch, pos = C.sample_grasp(rng, pk, xk)
                    out = Path(args.out) / f"fold_{fold}"
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / f"{port}_{cond}_{i}.json"
                    if path.exists():
                        continue
                    specs.append(dict(fold=fold, port=port, role=role, condition=cond, idx=i,
                                      seed=900000 + 100000 * fi + 10000 * pi + 100 * ci + i,
                                      pitch=pitch, grasp_pos=pos, n_starts=args.n_starts,
                                      run_dir=args.single_run or f"{args.runs}/fold_{fold}",
                                      methods=args.methods, path=str(path)))
    specs.sort(key=lambda s: (s["port"], s["fold"]))
    print(f"{len(specs)} test grasps, {args.workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for k, row in enumerate(pool.imap_unordered(_work, specs, chunksize=1)):
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s] {row}", flush=True)


if __name__ == "__main__":
    main()
