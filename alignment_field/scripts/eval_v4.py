"""v4 closed loop: does the alignment field extend a learned insertion policy?

python scripts/eval_v4.py --policies fm=runs_v4/policy bc=runs_v4/policy_bc \
    --fieldrels fm=runs_v4/fieldrel_fm bc=runs_v4/fieldrel_bc --out results/eval_v4
Per test grasp (one insertion gives H_pre, R_P by FK: used ONLY to place starts
and to score), from the same OOD starts:
for each IL policy P in {fm, bc} (RGB + joint angles, learned from demonstrations):
  P               the policy alone
  field+P         relational-net estimate -> P's field until it stops -> P
  oracle+P        true state -> same field -> P (upper bound for the field)
and, as the policy's own reference on this shape/grasp:
  P_id            P from a start inside its training distribution
Cells: A-D (never trained on) x {train, pitch_extrap, pos_extrap, combined};
training shapes x {pitch_extrap, pos_extrap, combined}.
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

COND = {"train": ("train", "train"), "pitch_extrap": ("extrap", "train"),
        "pos_extrap": ("train", "extrap"), "combined": ("extrap", "extrap")}
_C = {}


def _retreat(robot, H_pre, R_P):
    from paf.geometry import pose_from_state, relative_state
    u = relative_state(robot.hand_pose(), H_pre, R_P)
    u[0] = min(u[0], 0.0) - 0.003
    robot.move_to(pose_from_state(u, H_pre, R_P))


def _work(spec):
    import torch
    from paf import config as C
    from paf.collect import establish_reference
    from paf.demo import rollout_policy
    from paf.env_adapter import Robot
    from paf.evaluate import run_loop, sample_ood_start
    from paf.fm_policy import InsertionPolicy
    from paf.geometry import pose_from_state, relative_state
    from paf.policy import FieldPolicy
    torch.set_num_threads(1)
    try:
        if _C.get("port") != spec["port"]:
            if "robot" in _C:
                _C["robot"].env.close()
            _C["robot"], _C["port"] = Robot(spec["port"]), spec["port"]
        if "pols" not in _C:
            _C["pols"] = {n: (InsertionPolicy(spec["policies"][n]), FieldPolicy(spec["fieldrels"][n], estimate="rel"))
                          for n in spec["policies"]}
        robot = _C["robot"]
        rng = np.random.default_rng(spec["seed"])
        for attempt in range(3):
            robot.reset(spec["seed"] + 7919 * attempt, spec["pitch"], spec["grasp_pos"])
            H_pre, R_P, _ = establish_reference(robot)
            if H_pre is not None:
                break
        res = {**spec, "expert_ok": H_pre is not None, "episodes": []}
        if H_pre is None:
            return res
        obs0 = robot.observe()
        for pol, fp in _C["pols"].values():
            fp.begin_episode(obs0, rng)
        for s in range(spec["n_starts"]):
            q0 = sample_ood_start(rng)
            qid = np.r_[-0.002, rng.uniform(-0.0005, 0.0005, 2), rng.uniform(-1, 1, 3) * np.radians(0.5)]
            for name, (pol, fp), kind in [(n, _C["pols"][n], k) for n in _C["pols"]
                                          for k in spec["kinds"]]:
                method = {"alone": name, "field": f"field+{name}", "oracle": f"oracle+{name}", "id": f"{name}_id"}[kind]
                ep = {"start": s, "method": method, "policy": name, "kind": kind}
                start = qid if kind == "id" else q0
                if not robot.move_to(pose_from_state(start, H_pre, R_P)):
                    ep["failure"] = "start_ik"
                    res["episodes"].append(ep)
                    continue
                ep["u0"] = (relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA).tolist()
                if kind in ("field", "oracle"):
                    tr = run_loop(robot, fp, H_pre, R_P, kind, rng)
                    ep["field_iters"] = len(tr)
                    ep["u_after_field"] = (relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA).tolist()
                    if tr:
                        ep["field_est_err"] = float(np.linalg.norm(np.array(tr[-1]["u_est"]) - np.array(tr[-1]["u_true"])))
                ep.update(rollout_policy(robot, pol, H_pre, R_P, rng))
                res["episodes"].append(ep)
                _retreat(robot, H_pre, R_P)
        return res
    except Exception as exc:
        _C.pop("port", None)
        return {**spec, "expert_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    from paf import config as C
    from paf.ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True, help="name=run_dir, e.g. fm=runs_v4/policy")
    ap.add_argument("--fieldrels", nargs="+", required=True, help="name=dir with RelNet model.pt + that policy's field.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grasps", type=int, default=5)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N grasps")
    ap.add_argument("--kinds", nargs="+", default=["alone", "field", "oracle", "id"],
                    help="which of: alone (OOD start), field, oracle, id (start inside the policy's distribution)")
    ap.add_argument("--conditions", nargs="+", default=list(COND))
    ap.add_argument("--seen-grasps", type=int, default=None, help="grasps per seen-shape cell (default n-grasps)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_shapes = [p for p in generated_specs() if p not in ("G19", "G42", "G45", "G46", "G47", "G48", "G49")]
    specs = []
    for ci, (cond, (pk, xk)) in enumerate(COND.items()):
        if cond not in args.conditions:
            continue
        for pi, port in enumerate(["A", "B", "C", "D"]):
            rng = np.random.default_rng(700000 + 100 * ci + pi)
            for i in range(args.n_grasps):
                pitch, pos = C.sample_grasp(rng, pk, xk)
                specs.append(dict(role="unseen_shape", port=port, condition=cond, idx=i, pitch=pitch, grasp_pos=pos,
                                  seed=710000 + 1000 * ci + 100 * pi + i))
        n_seen = args.seen_grasps if args.seen_grasps is not None else args.n_grasps
        if cond != "train" or args.seen_grasps:
            rng = np.random.default_rng(800000 + ci)
            for i in range(n_seen):
                pitch, pos = C.sample_grasp(rng, pk, xk)
                port = train_shapes[int(rng.integers(len(train_shapes)))]
                specs.append(dict(role="seen_shape", port=port, condition=cond, idx=i, pitch=pitch, grasp_pos=pos,
                                  seed=810000 + 1000 * ci + i))
    todo = []
    for s in specs:
        s.update(n_starts=args.n_starts, kinds=args.kinds, policies=dict(x.split("=") for x in args.policies),
                 fieldrels=dict(x.split("=") for x in args.fieldrels),
                 path=str(out / f"{s['role']}_{s['port']}_{s['condition']}_{s['idx']}.json"))
        if not Path(s["path"]).exists():
            todo.append(s)
    todo.sort(key=lambda s: s["port"])
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} test grasps, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, todo)):
            Path(r["path"]).write_text(json.dumps(r))
            ins = {}
            for e in r.get("episodes", []):
                ins[e["method"]] = ins.get(e["method"], 0) + int(e.get("inserted", False))
            print(f"[{k + 1}/{len(todo)} {time.time() - t0:.0f}s] {r['role']} {r['port']} {r['condition']} "
                  f"{ins} {r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
