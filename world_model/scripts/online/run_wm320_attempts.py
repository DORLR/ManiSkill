"""Attempts of the world model's agent (wm/online/agent320.py, attempt320.py) on many grasps, in worker processes.

Two uses:
  online round (--mode collect --split train --pool <pool> --round R): every training grasp x --starts; every
      attempt is written as a trajectory in the fast copy's format (<pool>/r<R>/traj/<port>/<grasp>/s<i>.npz) and
      listed in <pool>/r<R>/manifest.json (paths relative to <pool>, with every decision's kind), then
      <pool>/r<R>/DONE is written when the round is complete or at most --max-crashed-frac of its grasps crashed
      (otherwise exit 2 and no DONE). Since 2026-11-18 the learner (train_wm320.py --online <pool>) also reads the
      finished grasps of a round without DONE (their result files, written whole); grasps run in random order so
      that those cover all ports from the start, and a finished grasp is never run again.
  test (--mode act | wm | servo | random, --split select | heldout): the same grasps, starts and seeds as
      eval_policy320.py (its grasp list, even selection, per-grasp start generator, placement retries,
      snapshot/restore), its result files and summary (summarise / show: clustered standard errors, the one success
      criterion). servo / random are the ablations of attempt320 (2026-11-19). Nothing is written but the result
      files, unless --record <dir>: then every attempt is also written as a fast-copy trajectory
      (<dir>/traj/<port>/<grasp>/s<i>.npz, kind "test") and listed in <dir>/manifest.json (for the fidelity checks).
Per grasp: the reference insertion (H*, axis: labels of the collect mode only, and the start placement), then for
every start the placement, snapshot/restore, one attempt. A grasp file without "axis" (test grasps made by
scripts/eval/make_combo_grasps.py, 2026-11-19: no demonstrations needed) gets it from this reference insertion, exactly
as wm/collect/episode._axis_from does at collection: the direction of the FK hand motion over the expert's
"axial_push" phase (no simulator truth). Per-grasp result file <out>/<port>/<grasp>.json as
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
REFERENCE_TRIES = 3                                         # as wm/collect/episode.REFERENCE_ATTEMPTS
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


def side_camera(robot, H_star: np.ndarray, axis: np.ndarray, w: int = 480, h: int = 360):
    """A static camera beside the socket for a visual check of insertion (2026-11-19; never seen by an agent): 12 cm to
    the side of the insertion axis and 3 cm back, looking at the seated hand position plus 2 cm along the axis. -> a
    function returning the current RGB picture (h, w, 3) uint8."""
    import sapien
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    side = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    target = np.asarray(H_star, float)[:3, 3] + 0.02 * a
    pos = target + 0.12 * side - 0.03 * a + 0.03 * np.array([0.0, 0.0, 1.0])
    f = (target - pos) / np.linalg.norm(target - pos)          # sapien camera: x forward, y left, z up
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = f, left, up, pos
    cam = sapien.render.RenderCameraComponent(w, h)
    cam.set_fovy(np.radians(30.0))
    cam.near, cam.far = 0.01, 3.0
    ent = sapien.Entity()
    ent.add_component(cam)
    ent.set_pose(sapien.Pose(T))
    robot.base.scene.sub_scenes[0].add_entity(ent)

    def grab() -> np.ndarray:
        robot.base.scene.update_render()
        cam.take_picture()
        rgb = np.asarray(cam.get_picture("Color"))[..., :3]
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    return grab


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
    if a["pool"] and E.grasp_state(Path(out_file)) == "done":        # finished, though its return was lost to a dead worker:
        return json.loads(Path(out_file).read_text())                # never re-run (the learner may be reading its files)
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
        root = Path(a["pool"]) if a["pool"] else (Path(a["record"]) if a["record"] else None)
        sub = f"r{a['round']}/traj" if a["pool"] else "traj"
        if root is not None:                                   # a retried grasp: no file of an earlier try stays behind
            shutil.rmtree(root / sub / g["port"] / g["grasp"], ignore_errors=True)
        rng = np.random.default_rng(seed)
        push: list[np.ndarray] = []                            # FK hand after every step of the axial push

        def on_step(phase, action, info):
            if phase == "axial_push":
                push.append(robot.hand_pose()[:3, 3].copy())
        # a grasp of wm_v1 has the episode seed whose reference insertion worked (one try, as before); a test grasp
        # without "axis" gets collection's retries (wm/collect/episode.py: seed + 7919 a, REFERENCE_TRIES tries)
        tries = 1 if g.get("axis") is not None else REFERENCE_TRIES
        done_ref, why = False, "reference insertion failed"
        for t in range(tries):
            ep_seed = int(g["episode_seed"]) + 7919 * t
            for rebuilt in (False, True):                      # an exception: once more with a fresh scene, same seed
                push.clear()
                try:
                    robot.reset(ep_seed, g["pitch_deg"], g["grasp_pos_m"])
                    expert = Expert(robot)
                    done_ref = bool(expert.to_safe_from_spawn() and expert.run("insert", on_step, seed_offset=0).get("completed"))
                    why = "reference insertion failed"
                    break
                except Exception as exc:                       # still failing on a fresh scene: unusable, not a crash
                    done_ref, why = False, f"reference setup failed: {type(exc).__name__}: {exc}"[:300]
                    if rebuilt:
                        break
                    old, _S["robot"], _S["port"] = _S["robot"], None, None
                    try:
                        old.env.close()
                    except Exception:
                        pass
                    _S["robot"], _S["port"] = Robot(g["port"]), g["port"]
                    robot = _S["robot"]
            if done_ref:
                break
        s["episode_seed_used"], s["reference_tries"] = ep_seed, t + 1
        if not done_ref:
            s.update(error=why, error_kind="setup")
            return s
        axis = g.get("axis")
        if axis is None:                                       # as wm/collect/episode._axis_from
            if len(push) < 2 or np.linalg.norm(push[-1] - push[0]) < 1e-4:
                s.update(error="no axial push in the reference insertion", error_kind="setup")
                return s
            axis = (push[-1] - push[0]) / np.linalg.norm(push[-1] - push[0])
            s["axis_measured"] = [float(v) for v in axis]
        ex = OnlineTester(robot, robot.hand_pose(), np.asarray(axis))
        if not ex.remember_standoff():
            s.update(error="standoff not reachable", error_kind="setup")
            return s
        # tests (2026-11-19): every start begins from the state after the reference insertion and the standoff, not
        # from where the previous attempt ended (a jam or slip in the fingers would otherwise carry over, differently
        # per agent); start 0 is as before. collect keeps its old behaviour.
        base = None if a["pool"] else robot.base.get_state_dict()
        for i in range(starts):
            if base is not None and i > 0:
                robot.base.set_state_dict(base)
            if "pair_index" in g:                              # paired cells: start i drawn from its own generator, so a
                rng = np.random.default_rng([seed, i])         # placement redraw at start 0 cannot change start 1
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
            entry["start_hand_mm"] = [round(float(x) * 1000.0, 3) for x in robot.hand_pose()[:3, 3]]
            side = []
            if a.get("side_video"):                            # visual check only: a static camera beside the socket
                grab = side_camera(robot, ex.H_star, ex.F[:, 0])
                side.append(grab())
                ex.side_capture = lambda k: side.append(grab())
            summ, out = ex.attempt(agent, a["mode"], int(seed + 1000 * i), margin=a["margin"], max_retries=a["max_retries"],
                                   p_shift=a["p_shift"], eps=a["eps"], n_candidates=a["candidates"], radius_mm=a["radius_mm"],
                                   random_reposition=a["random_reposition"], inside_mm=a["inside_mm"],
                                   servo_axis=a["servo_axis"], random_p=a["random_p"], probe_max=a["probe_max"],
                                   probe_gap_mm=a["probe_gap_mm"])
            entry["attempt"] = summ
            if side:
                from PIL import Image, ImageDraw
                ims = [Image.fromarray(x) for x in side]
                txt = "%s start %d: %s (depth left %.2f mm, lateral %.2f mm, tilt %.1f deg)" % (
                    g["grasp"], i, "SUCCESS" if summ["success"] else "FAILURE", summ["depth_left_mm"], summ["lateral_mm"], summ["angle_deg"])
                for im in ims[-1:]:
                    ImageDraw.Draw(im).text((6, 6), txt, fill=(255, 255, 0))
                ims += [ims[-1]] * 10
                vd = Path(a["side_video"])
                vd.mkdir(parents=True, exist_ok=True)
                ims[0].save(vd / f"{g['grasp']}_s{i}_{'success' if summ['success'] else 'failure'}.gif", save_all=True,
                            append_images=ims[1:], duration=120, loop=0)
                ex.side_capture = None
            if root is not None:
                rel = f"{sub}/{g['port']}/{g['grasp']}/s{i}.npz"
                meta = {**{k: g[k] for k in ("port", "grasp", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")},
                        "kind": "online" if a["pool"] else "test", "outcome": "seated_fk" if summ["seated_fk"] else "not_seated", "source": rel,
                        "pin_length_m": float(ex.L), "H_star": ex.H_star.tolist(), "axis": ex.F[:, 0].tolist(),
                        "round": a["round"], "start": i, "start_pose": start, "attempt_seed": int(seed + 1000 * i),
                        "mode": a["mode"], "model_step": summ["model_step"], "episode_seed_used": s["episode_seed_used"],
                        "stopped": summ["stopped"], "interventions": summ["interventions"], "retries": summ["retries"]}
                s["entries"].append(write_fast(root / rel, out, meta))
            del out
            torch.cuda.empty_cache()
        del s["error"], s["error_kind"]
    except Exception as exc:
        s.update(error=f"{type(exc).__name__}: {exc}"[:500], error_kind="crash", trace=traceback.format_exc()[-2000:])
    finally:
        s["seconds"] = round(time.time() - t0, 1)
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(out_file).with_name(Path(out_file).name + ".tmp")   # whole or absent: the learner reads finished grasps
        tmp.write_text(json.dumps(s))                                   # of a round in progress (train_wm320.online_entries)
        os.replace(tmp, out_file)
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
    ap.add_argument("--mode", required=True, choices=["act", "wm", "collect", "servo", "random"])
    ap.add_argument("--split", required=True, choices=["train", "select", "heldout", "test", "any"],
                    help="any: every grasp of --grasps but the test ports (tests only; for make_combo_grasps.py's cells)")
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
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--inside-mm", type=float, default=None,
                    help="wm/collect: no correction once the estimated pin-tip depth is below -this [mm] (needs the offset head)")
    ap.add_argument("--candidates", type=int, default=16, help="re-position candidates after a back-off")
    ap.add_argument("--radius-mm", type=float, default=3.0, help="re-position candidates lie within this radius")
    ap.add_argument("--random-reposition", type=float, default=0.5, help="collect: share of re-positions drawn at random")
    ap.add_argument("--p-shift", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--servo-axis", choices=["true", "motion"], default="true", help="servo: the frame of the read-out's lateral offset")
    ap.add_argument("--random-p", help="random: 4 comma-separated correction probabilities, one per bin of the estimated gap "
                                       "(< 0, 0-2, 2-6, >= 6 mm)")
    ap.add_argument("--record", help="test modes: also write every attempt as a fast-copy trajectory under this directory")
    ap.add_argument("--probe-max", type=int, default=0, help="wm: counterfactual probes per attempt and gap bin (attempt320; 0 = off)")
    ap.add_argument("--probe-gap-mm", type=float, default=6.0, help="wm: probe only where the estimated gap is below this [mm]")
    ap.add_argument("--side-video", help="tests: also write a GIF per attempt from a static camera beside the socket (visual check)")
    ap.add_argument("--max-crashed-frac", type=float, default=0.02, help="collect: a round with more crashed grasps gets no DONE")
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        raise SystemExit("ports A-D are the final test and stay untouched: --split test needs --final-test")
    collect = args.mode == "collect"
    if args.split == "any" and (collect or args.final_test or args.grasps_n):
        raise SystemExit("--split any is for tests of generated grasps (make_combo_grasps.py), all of them")
    if collect != (args.split == "train"):
        raise SystemExit("collect runs on the training grasps only, and the training grasps are only collected on")
    if collect and not (args.pool and args.round > 0):
        raise SystemExit("collect needs --pool and --round >= 1")
    if collect and args.record:
        raise SystemExit("--record is for tests; collect writes its trajectories to the pool")
    if args.mode in ("servo", "random") and args.inside_mm is None:
        raise SystemExit(f"--mode {args.mode} is an ablation of the in-hole agent: it needs --inside-mm")
    random_p = None
    if args.mode == "random":
        try:
            random_p = [float(x) for x in (args.random_p or "").split(",")]
        except ValueError:
            raise SystemExit(f"--random-p must be 4 numbers, got {args.random_p!r}")
        if len(random_p) != 4 or not all(0.0 <= x <= 1.0 for x in random_p):
            raise SystemExit(f"--random-p needs 4 probabilities in [0, 1], got {args.random_p!r}")
    elif args.random_p:
        raise SystemExit("--random-p is for --mode random only")
    if args.mode in ("servo", "random") and args.max_retries:
        raise SystemExit(f"--mode {args.mode} runs without back-off: give --max-retries 0")
    if args.probe_max < 0 or args.probe_gap_mm <= 0:
        raise SystemExit("--probe-max must be >= 0 and --probe-gap-mm > 0")
    if args.probe_max and (args.mode != "wm" or args.inside_mm is None):
        raise SystemExit("--probe-max needs --mode wm with --inside-mm (the in-hole agent)")
    rdir = Path(args.pool) / f"r{args.round}" if collect else None
    if collect and (rdir / "DONE").exists():
        print(f"{rdir} is complete already", flush=True)
        sys.exit(0)
    out = Path(args.out) if args.out else (rdir / "results" if collect else None)
    if out is None:
        raise SystemExit("--out is needed for a test")
    gpus = [int(x) for x in args.gpus.split(",")]
    if args.inside_mm is not None:                             # the rule reads the offset head; without it, it would be silently off
        import torch
        if args.inside_mm <= 0:
            raise SystemExit("--inside-mm must be > 0 (a depth inside the hole)")
        if not torch.load(args.ckpt, map_location="cpu", weights_only=False).get("offset_weight", 0.0) > 0:
            raise SystemExit(f"--inside-mm needs a checkpoint with the offset read-out head; {args.ckpt} has none")
    seed = args.seed + 1_000_003 * args.round if collect else args.seed
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{args.grasps}/*/*/grasp.json"))]
    every = [g for g in every if g.get("reference_ok") and (
        D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == args.split
        or (args.split == "any" and g["port"] not in D.TEST_PORTS))]
    if args.split == "any" and (not every or not all("pair_index" in g for g in every)):
        raise SystemExit("--split any needs make_combo_grasps.py's grasps (with pair_index) only")
    chosen = E.pick(len(every), args.grasps_n)                     # k = index in the whole list: a grasp keeps its seed
    gs = [every[k] for k in chosen]
    path = {g["grasp"]: out / g["port"] / f"{g['grasp']}.json" for g in gs}
    a = {k: getattr(args, k) for k in ("mode", "margin", "max_retries", "p_shift", "eps", "candidates", "radius_mm", "random_reposition", "round",
                                         "inside_mm", "servo_axis", "probe_max", "probe_gap_mm")}
    a["pool"] = str(args.pool) if collect else None
    a["random_p"] = random_p
    a["record"] = str(args.record) if args.record else None
    a["side_video"] = str(args.side_video) if args.side_video else None
    if collect and args.side_video:
        raise SystemExit("--side-video is for tests")
    if a["record"] and any(E.grasp_state(path[g["grasp"]]) != "missing" for g in gs):
        raise SystemExit(f"--record needs a fresh --out ({out} has result files: their attempts would not be recorded)")
    print(f"{len(gs)} of {len(every)} grasps of split {args.split} x {args.starts} starts, mode {args.mode}, {args.workers} workers on GPUs {gpus}", flush=True)
    for _ in range(1 + args.retries):
        todo = [(k, g) for k, g in zip(chosen, gs) if E.grasp_state(path[g["grasp"]]) in ("missing", "crash")]
        if not todo:
            break
        for _, g in todo:
            path[g["grasp"]].unlink(missing_ok=True)
        if collect:                                            # random order: the learner uses a round while it is collected
            todo = [todo[i] for i in np.random.default_rng(seed).permutation(len(todo))]     # (all ports from the start)
        else:
            todo.sort(key=lambda kg: kg[1]["port"])
        # paired cells (make_combo_grasps): the job seed follows the pair index, so every cell (and every arm with the same
        # --seed) draws the same starts and attempt seeds for grasp i
        run_pool([(g, str(path[g["grasp"]]), args.starts, seed + 100_003 * (g["pair_index"] if "pair_index" in g else k), a)
                  for k, g in todo], args, gpus)
    out.mkdir(parents=True, exist_ok=True)
    summary, complete = E.summarise(out, gs, args.starts, {"policy": f"wm320 agent ({args.mode})", "ckpt": args.ckpt, "split": args.split,
                                                          "mode": args.mode, "round": args.round, "starts_per_grasp": args.starts,
                                                          "margin": args.margin, "horizon": args.horizon, "candidates": args.candidates,
                                                          "radius_mm": args.radius_mm, "max_retries": args.max_retries,
                                                          "inside_mm": args.inside_mm, "p_shift": args.p_shift, "eps": args.eps,
                                                          "random_reposition": args.random_reposition,
                                                          "servo_axis": args.servo_axis if args.mode == "servo" else None,
                                                          "random_p": random_p, "record": a["record"],
                                                          "probe_max": args.probe_max, "probe_gap_mm": args.probe_gap_mm,
                                                          "seed": args.seed, "grasps_n": args.grasps_n})
    rows = [json.loads(path[g["grasp"]].read_text()) for g in gs if E.grasp_state(path[g["grasp"]]) == "done"]
    att = [st["attempt"] for r in rows for st in r.get("starts", []) if "attempt" in st]
    summary["attempts"] = len(att)
    summary["seated_fk"] = float(np.mean([x["seated_fk"] for x in att])) if att else None
    summary["interventions_per_attempt"] = float(np.mean([x["interventions"] for x in att])) if att else None
    summary["retries_per_attempt"] = float(np.mean([x["retries"] for x in att])) if att else None
    (out / ("summary.json" if complete else "summary_incomplete.json")).write_text(json.dumps(summary, indent=1))
    print(E.show(summary), flush=True)
    if a["record"]:
        entries = [e for r in rows for e in r.get("entries", [])]
        Path(a["record"]).mkdir(parents=True, exist_ok=True)
        (Path(a["record"]) / "manifest.json").write_text(json.dumps({"format": D.FORMAT, "ckpt": args.ckpt, "split": args.split,
                                                                     "mode": args.mode, "entries": entries}))
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
