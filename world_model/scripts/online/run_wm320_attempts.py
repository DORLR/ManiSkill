"""Attempts of the world model's agent (wm/online/agent320.py, attempt320.py) on many grasps, in worker processes.

Two uses:
  online round (--mode collect --split train --pool <pool> --round R): every training grasp x --starts; every
      attempt is written as a trajectory in the fast copy's format (<pool>/r<R>/traj/<port>/<grasp>/s<i>.npz) and
      listed in <pool>/r<R>/manifest.json (paths relative to <pool>, with every decision's kind), then
      <pool>/r<R>/DONE is written when the round is complete or at most --max-crashed-frac of its grasps crashed
      (otherwise exit 2 and no DONE); the learner (train_wm320.py --online <pool>) reads rounds with DONE only.
  test (--mode act | wm, --split select | heldout): the same grasps, starts and seeds as eval_policy320.py (its
      grasp list, even selection, per-grasp start generator, placement retries, snapshot/restore), its result files
      and summary (summarise / show: clustered standard errors, the one success criterion). Nothing is written but
      the result files.
Per grasp: the reference insertion (H*, axis: labels of the collect mode only, and the start placement), then for
every start the placement, snapshot/restore, one attempt. Per-grasp result file <out>/<port>/<grasp>.json as
eval_policy320 (with error_kind setup / crash); crashed grasps are retried; a dying worker rebuilds the pool.
Workers are spread over the GPUs given (--gpus), one GPU per worker (renderer and model).

    python world_model/scripts/online/run_wm320_attempts.py --grasps $Q/data/wm_v1 --ckpt <model.pt> --mode collect \
        --split train --pool $Q/data/online320 --round 1 --starts 3 --workers 6 --gpus 3,4,5,6,7,2
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import eval_policy320 as E  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

MAX_POOL_REBUILDS = E.MAX_POOL_REBUILDS
_S = {"init_error": None, "robot": None, "port": None}


def _init(counter, gpus: list[int], ckpt: str, torch_mib: int, horizon: int):
    try:
        with counter.get_lock():
            i = counter.value
            counter.value += 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        from wm.env import robot as _robot
        if _robot.CAM_SCALE != 1:
            raise RuntimeError(f"WM_CAM_SCALE={_robot.CAM_SCALE}: the model reads the native 320 px render, unset it")
        import torch
        torch.set_num_threads(1)
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        torch.cuda.set_per_process_memory_fraction(min(1.0, torch_mib / total), 0)
        from wm.online.agent320 import WMAgent
        _S["agent"] = WMAgent(ckpt, "cuda", horizon=horizon)
    except Exception as exc:
        _S["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def write_fast(path: Path, out: dict, meta: dict) -> dict:
    """One attempt -> a fast-copy trajectory file (demo320's format, byte for byte the same codecs); its manifest entry."""
    jp = out["rgb_jpeg_list"]
    if not len(jp) == len(out["rgb"]) == len(out["depth"]) == len(out["img_step"]):
        raise ValueError("pictures, depth pictures and image steps differ in number")
    rgb_blob = np.frombuffer(b"".join(jp), np.uint8)
    rgb_off = np.cumsum([0] + [len(b) for b in jp]).astype(np.int64)
    dblobs = [D.encode_depth(d) for d in out["depth"]]
    dblob, doff = np.concatenate(dblobs), np.cumsum([0] + [len(b) for b in dblobs]).astype(np.int64)
    for i, d in enumerate(out["depth"]):
        if not np.array_equal(D.decode_depth(dblob, doff, i), d):
            raise ValueError(f"depth picture {i} does not survive the PNG round trip")
    kinds = [int(x) for x in out["dec_kind"]]
    if len(kinds) != len(out["action"]) // 8:
        raise ValueError(f"{len(kinds)} decision kinds for {len(out['action']) // 8} decisions")
    meta = {**meta, "steps": int(len(out["action"])), "format": D.FORMAT, "dec_kind": kinds}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.asarray(json.dumps(meta)), rgb_jpeg=rgb_blob, rgb_offsets=rgb_off, depth_png=dblob, depth_offsets=doff,
             img_step=out["img_step"], qpos=out["qpos"], hand=out["hand"], action=out["action"], dec_kind=np.asarray(kinds, np.int8))
    os.replace(tmp, path)
    return {**D.entry_of(meta, meta["source"], len(jp)), "dec_kind": kinds}


def work(job):
    g, out_file, starts, seed, a = job
    s = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    s.update(stop=a["mode"], starts=[], entries=[], error="interrupted", error_kind="crash")
    t0 = time.time()
    try:
        if _S["init_error"]:
            raise RuntimeError("init: " + _S["init_error"])
        import torch
        from wm.collect import branch as B
        from wm.env.expert import Expert
        from wm.env.robot import Robot
        from wm.online.attempt320 import OnlineTester
        if _S["port"] != g["port"]:
            if _S["robot"] is not None:
                old, _S["robot"], _S["port"] = _S["robot"], None, None
                old.env.close()
            _S["robot"], _S["port"] = Robot(g["port"]), g["port"]
        robot, agent = _S["robot"], _S["agent"]
        if a["pool"]:                                          # a retried grasp: no file of an earlier try stays behind
            shutil.rmtree(Path(a["pool"]) / f"r{a['round']}" / "traj" / g["port"] / g["grasp"], ignore_errors=True)
        rng = np.random.default_rng(seed)
        robot.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        expert = Expert(robot)
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *x: None, seed_offset=0).get("completed")):
            s.update(error="reference insertion failed", error_kind="setup")
            return s
        ex = OnlineTester(robot, robot.hand_pose(), np.asarray(g["axis"]))
        if not ex.remember_standoff():
            s.update(error="standoff not reachable", error_kind="setup")
            return s
        for i in range(starts):
            fails = []
            for _ in range(B.PLACEMENT_ATTEMPTS):
                start = B.sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            entry = {"idx": i, "start": start, "placement_failures": fails}
            s["starts"].append(entry)
            if stage != "ok":
                entry["placement_failed"] = True
                continue
            ex.snapshot()
            ex.restore()
            summ, out = ex.attempt(agent, a["mode"], int(seed + 1000 * i), margin=a["margin"], max_retries=a["max_retries"],
                                   p_shift=a["p_shift"], eps=a["eps"], v_low=a["v_low"])
            entry["attempt"] = summ
            if a["pool"]:
                rel = f"r{a['round']}/traj/{g['port']}/{g['grasp']}/s{i}.npz"
                meta = {**{k: g[k] for k in ("port", "grasp", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")},
                        "kind": "online", "outcome": "seated_fk" if summ["seated_fk"] else "not_seated", "source": rel,
                        "pin_length_m": float(ex.L), "H_star": ex.H_star.tolist(), "axis": ex.F[:, 0].tolist(),
                        "round": a["round"], "start": i, "start_pose": start, "attempt_seed": int(seed + 1000 * i),
                        "mode": a["mode"], "model_step": summ["model_step"],
                        "stopped": summ["stopped"], "interventions": summ["interventions"], "retries": summ["retries"]}
                s["entries"].append(write_fast(Path(a["pool"]) / rel, out, meta))
            del out
            torch.cuda.empty_cache()
        del s["error"], s["error_kind"]
    except Exception as exc:
        s.update(error=f"{type(exc).__name__}: {exc}"[:500], error_kind="crash", trace=traceback.format_exc()[-2000:])
    finally:
        s["seconds"] = round(time.time() - t0, 1)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(s))
    return s


def run_pool(jobs: list, args, gpus: list[int]) -> None:
    """eval_policy320.run_pool with this script's worker setup (that file must not change while other tests import it)."""
    left, rebuilds, counter = list(jobs), 0, mp.get_context("spawn").Value("i", 0)
    t0 = time.time()
    while left and rebuilds <= MAX_POOL_REBUILDS:
        ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(counter, gpus, args.ckpt, args.torch_mib, args.horizon))
        futs = {ex.submit(work, j): j for j in left[:10 * args.workers]}
        broken = False
        for f in as_completed(futs):
            try:
                s = f.result()
            except BrokenProcessPool:
                broken = True
                break
            except Exception as exc:
                print(f"{futs[f][0]['grasp']} failed: {type(exc).__name__}: {exc}"[:300], flush=True)
                continue
            o = " | ".join(f"{st['attempt']['stopped'][:5]}/{'ok' if st['attempt']['success'] else 'no'}/i{st['attempt']['interventions']}"
                           if "attempt" in st else "unplaced" for st in s["starts"])
            print(f"[{time.time() - t0:.0f}s] {s['grasp']} {o} {str(s.get('error', ''))[:150]} {s.get('seconds')}s", flush=True)
        ex.shutdown(wait=not broken, cancel_futures=True)
        if broken:
            rebuilds += 1
            print(f"a worker died; pool restart {rebuilds} of {MAX_POOL_REBUILDS}", flush=True)
        ran = {id(j) for f, j in futs.items() if f.done() and not f.cancelled() and not isinstance(f.exception(), BrokenProcessPool)}
        left = [j for j in left if id(j) not in ran]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", required=True, choices=["act", "wm", "collect"])
    ap.add_argument("--split", required=True, choices=["train", "select", "heldout", "test"])
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--out", help="result files (default: <pool>/r<round>/results for collect)")
    ap.add_argument("--pool", help="collect: the online pool")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--grasps-n", type=int)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpus", required=True, help="comma list; workers are spread over them")
    ap.add_argument("--torch-mib", type=int, default=1500, help="torch memory cap per worker")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--p-shift", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--v-low", type=float, help="back off when blocked only below this mean of the value heads (required with the world model)")
    ap.add_argument("--max-crashed-frac", type=float, default=0.02, help="collect: a round with more crashed grasps gets no DONE")
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        raise SystemExit("ports A-D are the final test and stay untouched: --split test needs --final-test")
    collect = args.mode == "collect"
    if collect != (args.split == "train"):
        raise SystemExit("collect runs on the training grasps only, and the training grasps are only collected on")
    if collect and not (args.pool and args.round > 0):
        raise SystemExit("collect needs --pool and --round >= 1")
    rdir = Path(args.pool) / f"r{args.round}" if collect else None
    if args.mode != "act" and args.v_low is None:
        raise SystemExit("--v-low is required with the world model (it depends on the value's scale)")
    if collect and (rdir / "DONE").exists():
        print(f"{rdir} is complete already", flush=True)
        sys.exit(0)
    out = Path(args.out) if args.out else (rdir / "results" if collect else None)
    if out is None:
        raise SystemExit("--out is needed for a test")
    gpus = [int(x) for x in args.gpus.split(",")]
    seed = args.seed + 1_000_003 * args.round if collect else args.seed
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    every = [g for g in every if g.get("reference_ok") and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == args.split]
    chosen = E.pick(len(every), args.grasps_n)                     # k = index in the whole list: a grasp keeps its seed
    gs = [every[k] for k in chosen]
    path = {g["grasp"]: out / g["port"] / f"{g['grasp']}.json" for g in gs}
    a = {k: getattr(args, k) for k in ("mode", "margin", "max_retries", "p_shift", "eps", "v_low", "round")}
    a["pool"] = str(args.pool) if collect else None
    print(f"{len(gs)} of {len(every)} grasps of split {args.split} x {args.starts} starts, mode {args.mode}, {args.workers} workers on GPUs {gpus}", flush=True)
    for _ in range(1 + args.retries):
        todo = [(k, g) for k, g in zip(chosen, gs) if E.grasp_state(path[g["grasp"]]) in ("missing", "crash")]
        if not todo:
            break
        for _, g in todo:
            path[g["grasp"]].unlink(missing_ok=True)
        todo.sort(key=lambda kg: kg[1]["port"])
        run_pool([(g, str(path[g["grasp"]]), args.starts, seed + 100_003 * k, a) for k, g in todo], args, gpus)
    out.mkdir(parents=True, exist_ok=True)
    summary, complete = E.summarise(out, gs, args.starts, {"policy": f"wm320 agent ({args.mode})", "ckpt": args.ckpt, "split": args.split,
                                                          "mode": args.mode, "round": args.round, "starts_per_grasp": args.starts,
                                                          "margin": args.margin, "horizon": args.horizon, "v_low": args.v_low})
    rows = [json.loads(path[g["grasp"]].read_text()) for g in gs if E.grasp_state(path[g["grasp"]]) == "done"]
    att = [st["attempt"] for r in rows for st in r.get("starts", []) if "attempt" in st]
    summary["attempts"] = len(att)
    summary["seated_fk"] = float(np.mean([x["seated_fk"] for x in att])) if att else None
    summary["interventions_per_attempt"] = float(np.mean([x["interventions"] for x in att])) if att else None
    summary["retries_per_attempt"] = float(np.mean([x["retries"] for x in att])) if att else None
    (out / ("summary.json" if complete else "summary_incomplete.json")).write_text(json.dumps(summary, indent=1))
    print(E.show(summary), flush=True)
    if collect:
        entries = [e for r in rows for e in r.get("entries", [])]
        (rdir / "manifest.json").write_text(json.dumps({"format": D.FORMAT, "round": args.round, "ckpt": args.ckpt, "entries": entries}))
        crashed = summary["crashed_grasps"] / max(1, summary["planned_grasps"])
        if entries and (complete or crashed <= args.max_crashed_frac):
            (rdir / "DONE").write_text(json.dumps({"entries": len(entries), "complete": complete, "crashed_grasps": summary["crashed_grasps"]}))
            print(f"round {args.round}: {len(entries)} trajectories in the pool", flush=True)
            sys.exit(0)
        print(f"round {args.round}: {len(entries)} trajectories, {summary['crashed_grasps']} crashed grasps: not marked complete", flush=True)
        sys.exit(2)
    sys.exit(0 if complete else 2)


if __name__ == "__main__":
    main()
