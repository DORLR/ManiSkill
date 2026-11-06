"""Interaction data for the latent field: random hand poses near the port.

python scripts/collect_interaction.py --out data/v4_interact --per-port 4 --views 64
Per grasp: the robot visits random hand poses in front of the port and stores
what the IL policy would see (RGB crop, 7 joint angles) and the hand pose (FK).
The latent field learns from pairs of views of the same grasp how a hand motion
(hand frame, FK only) changes the IL's features; it gets no port coordinates.
One insertion (FK) gives H_pre, R_P: used ONLY to place the poses and to log u
for diagnostics, never as model input.
Views: 35 % near the demonstrated states, 65 % spread over the field domain.
Training shapes and training-band grasps (new seeds); A-D grasps are collected
with a separate flag for diagnostics only (never trained on).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

_R = {}


def _sample_u(rng, near):
    if near:
        t = [rng.uniform(-0.009, 0.0), *np.clip(rng.normal(0, 0.0008, 2), -0.0025, 0.0025)]
        r = np.clip(rng.normal(0, np.radians(0.4), 3), -np.radians(1.0), np.radians(1.0))
    else:
        t = [rng.uniform(-0.018, 0.0), *rng.uniform(-0.011, 0.011, 2)]
        r = rng.uniform(-1, 1, 3) * np.radians(7.0)
    return np.r_[t, r]


def _work(spec):
    from paf import config as C
    from paf.collect import establish_reference
    from paf.env_adapter import Robot
    from paf.geometry import pose_from_state, relative_state
    from paf.perception import rgb_crop
    try:
        if _R.get("port") != spec["port"]:
            if "robot" in _R:
                _R["robot"].env.close()
            _R["robot"], _R["port"] = Robot(spec["port"]), spec["port"]
        robot = _R["robot"]
        rng = np.random.default_rng(spec["seed"])
        t0 = time.time()
        for attempt in range(3):
            robot.reset(spec["seed"] + 7919 * attempt, spec["pitch"], spec["grasp_pos"])
            H_pre, R_P, _ = establish_reference(robot)
            if H_pre is not None:
                break
        if H_pre is None:
            np.savez(spec["path"], meta=json.dumps({**spec, "ok": False}))
            return {"path": spec["path"], "ok": False}
        rec = {k: [] for k in ("rgb", "qpos", "H", "u", "near")}
        for _ in range(spec["views"]):
            near = rng.random() < 0.35
            if not robot.move_to(pose_from_state(_sample_u(rng, near), H_pre, R_P)):
                continue
            obs = robot.observe()
            rec["rgb"].append(rgb_crop(obs))
            rec["qpos"].append(obs.qpos[:7].astype(np.float32))
            rec["H"].append(obs.H.astype(np.float32))
            rec["u"].append((relative_state(obs.H, H_pre, R_P) / C.SIGMA).astype(np.float32))
            rec["near"].append(near)
        meta = {**spec, "ok": True, "n": len(rec["u"]), "seconds": time.time() - t0}
        np.savez(spec["path"], meta=json.dumps(meta), H_pre=H_pre, R_P=R_P,
                 **{k: np.asarray(v) for k, v in rec.items()})
        return {"path": spec["path"], "ok": True, "n": len(rec["u"]), "s": round(time.time() - t0)}
    except Exception as exc:
        _R.pop("port", None)
        return {"path": spec["path"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    from paf import config as C
    from paf.ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v4_interact")
    ap.add_argument("--per-port", type=int, default=4)
    ap.add_argument("--abcd-per-port", type=int, default=3)
    ap.add_argument("--views", type=int, default=64)
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bad, val = {"G19", "G42", "G45", "G49"}, {"G46", "G47", "G48"}
    train = [p for p in generated_specs() if p not in bad and p not in val]
    specs = []
    for pi, port in enumerate(train + ["A", "B", "C", "D"]):
        diag = port in ("A", "B", "C", "D")
        rng = np.random.default_rng(900000 + pi)
        for i in range(args.abcd_per_port if diag else args.per_port):
            pitch, pos = C.sample_train_grasp(rng)
            path = out / f"{'diag_' if diag else ''}{port}_{i:02d}.npz"
            if not path.exists():
                specs.append(dict(port=port, idx=i, seed=900000 + 1000 * pi + i, pitch=pitch, grasp_pos=pos,
                                  views=args.views, diagnostic_only=diag, path=str(path)))
    specs.sort(key=lambda s: s["port"])
    print(f"{len(specs)} grasps, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, specs)):
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s] {r}", flush=True)


if __name__ == "__main__":
    main()
