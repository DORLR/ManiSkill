"""Online stage (v4): actors collect episodes with the world model choosing among FM proposals; the
learners keep training the world model and the success classifier on them (wm/online/).

    python world_model/scripts/online/run_online.py actors  --config world_model/configs/v4/online.json --workers 8
    python world_model/scripts/online/run_online.py wm      --config world_model/configs/v4/online.json
    python world_model/scripts/online/run_online.py success --config world_model/configs/v4/online.json
`--episodes N` (actors): stop after N episodes per worker (smoke tests); otherwise forever.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from wm.training import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["actors", "wm", "success"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--steps", type=int, help="learners: smoke tests only (absolute step to stop at)")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    out = Path(cfg["dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.role == "actors":
        from wm.data import index
        from wm.online.actor import worker
        ports = sorted({e["port"] for e in index.load(cfg["data"]) if e["split"] == "train"})
        acfg = {**cfg["actor"], "ports": ports, "latents": cfg["latents"], "seed": cfg["seed"]}
        (out / "actor_config.json").write_text(json.dumps(acfg, indent=1))
        with mp.get_context("spawn").Pool(args.workers) as pool:
            print(pool.map(worker, [(w, acfg, str(out), args.episodes) for w in range(args.workers)]))
    elif args.role == "wm":
        from wm.online.learner import run_wm
        wcfg = common.load_config(cfg["wm_config"])
        o = dict(cfg["learner"], dir=str(out), wm_resume=cfg["actor"]["wm_init_last"])
        if args.steps:
            o["wm_steps"] = args.steps
        run = Path(cfg["actor"]["wm_live"]).parent
        last = run / "last.pt"
        if last.exists():                          # continue this learner's own run
            o["wm_resume"] = str(last)
        elif not Path(o["wm_resume"]).exists():    # no offline checkpoint: train from scratch on the mixture
            o["wm_resume"] = None
        run_wm(wcfg, o, run)
    else:
        from wm.online.learner import run_success
        scfg = common.load_config(cfg["success_config"])
        o = dict(cfg["learner"], dir=str(out), success_resume=cfg["actor"]["success_init_last"],
                 wm_live=cfg["actor"]["wm_live"])
        if args.steps:
            o["success_steps"] = args.steps
        run = Path(cfg["actor"]["success_live"]).parent
        last = run / "last.pt"
        if last.exists():
            o["success_resume"] = str(last)
        elif not Path(o["success_resume"]).exists():
            o["success_resume"] = None
        run_success(scfg, o, run)


if __name__ == "__main__":
    main()
