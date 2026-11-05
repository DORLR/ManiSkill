"""Measure where the learned insertion policy succeeds (its in-distribution set)
and fit the alignment field to it.

python scripts/measure_id.py --policy runs_v4/policy --out runs_v4/field --grasps 60 --starts 12
Training shapes and training-band grasps only, new seeds. Starts are drawn in a
box a little wider than the demonstrations (t_x -6..0 mm, lateral +-2.5 mm,
rotations +-2.5 deg); a start is ID if the policy inserts from it. The true
start state comes from FK relative to H_pre (one insertion per grasp).
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

_C = {}


def _work(spec):
    import torch
    from paf import config as C
    from paf.collect import establish_reference
    from paf.demo import rollout_policy
    from paf.env_adapter import Robot
    from paf.fm_policy import InsertionPolicy
    from paf.geometry import pose_from_state, relative_state
    torch.set_num_threads(1)
    try:
        if _C.get("port") != spec["port"]:
            if "robot" in _C:
                _C["robot"].env.close()
            _C["robot"], _C["port"] = Robot(spec["port"]), spec["port"]
        if "pol" not in _C:
            _C["pol"] = InsertionPolicy(spec["policy"])
        robot, pol = _C["robot"], _C["pol"]
        rng = np.random.default_rng(spec["seed"])
        for attempt in range(3):
            robot.reset(spec["seed"] + 7919 * attempt, spec["pitch"], spec["grasp_pos"])
            H_pre, R_P, _ = establish_reference(robot)
            if H_pre is not None:
                break
        if H_pre is None:
            return {**spec, "ok": False}
        pol.set_axes_from(robot.observe(), rng)
        rows = []
        for _ in range(spec["starts"]):
            q0 = np.r_[rng.uniform(-0.006, 0.0), rng.uniform(-0.0025, 0.0025, 2),
                       rng.uniform(-1, 1, 3) * np.radians(2.5)]
            if not robot.move_to(pose_from_state(q0, H_pre, R_P)):
                continue
            u0 = relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA
            res = rollout_policy(robot, pol, H_pre, R_P, rng)
            rows.append({"u0": u0.tolist(), **res})
            u = relative_state(robot.hand_pose(), H_pre, R_P)
            u[0] = min(u[0], 0.0) - 0.003
            robot.move_to(pose_from_state(u, H_pre, R_P))
        return {**spec, "ok": True, "rows": rows}
    except Exception as exc:
        _C.pop("port", None)
        return {**spec, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    from paf import config as C
    from paf.ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grasps", type=int, default=60)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ports = [p for p in generated_specs() if p not in args.val_ports and p not in ("G19", "G42", "G45", "G49")]
    rng = np.random.default_rng(500000)
    specs = []
    for i in range(args.grasps):
        pitch, pos = C.sample_train_grasp(rng)
        specs.append(dict(port=ports[i % len(ports)], seed=500000 + i, pitch=pitch, grasp_pos=pos,
                          starts=args.starts, policy=args.policy))
    specs.sort(key=lambda s: s["port"])
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, specs)):
            results.append(r)
            n = sum(x["inserted"] for x in r.get("rows", []))
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s] {r['port']} ok={r['ok']} "
                  f"inserted {n}/{len(r.get('rows', []))} {r.get('error', '')}", flush=True)
    (out / "trials.json").write_text(json.dumps(results))
    rows = [x for r in results for x in r.get("rows", [])]
    U = np.array([x["u0"] for x in rows])
    ok = np.array([x["inserted"] for x in rows])
    M = U[ok].astype(np.float32)
    np.save(out / "id_set.npy", M)
    print(f"policy trials {len(rows)}, inserted {ok.mean():.3f}; ID set {len(M)}; mean {M.mean(0).round(2)} "
          f"std {M.std(0).round(2)}", flush=True)
    if len(M) < 10:
        raise SystemExit(f"only {len(M)} successful starts: the policy is too weak to define an ID set")
    import torch
    from paf.field import fit_field
    field, fit = fit_field(M, 0.5, np.array([15.0, 10.0, 10.0, 8.0, 8.0, 8.0]), device="cuda")
    torch.save(field.state_dict(), out / "field.pt")
    (out / "summary.json").write_text(json.dumps({"trials": int(len(rows)), "insert_rate": float(ok.mean()),
                                                  "n_id": int(len(M)), "field_fit": fit}, indent=1))


if __name__ == "__main__":
    main()
