"""AutoMate (Isaac Lab) branch of run_wm320_attempts.py: online rounds (--mode collect) and tests (--mode act | act_done |
wm | servo | random) of the world model's agent on AutoMate grasps (port plan docs/2026-11-23_automate_online_port_plan.md;
the user's decisions of 2026-11-25). Started as `run_wm320_attempts.py automate <stage> ...`, from a frozen code copy,
by scripts/automate/online_automate.sh (rounds beside the learner) or scripts/automate/online_job.sh (one job):

  prepare  (CPU, no Isaac) checks everything that can be checked before an Isaac start and writes <job>/job.json:
           the job dir is <pool>/r<R> (collect) or --out (tests). Refused (exit 3): an AutoMate checkpoint without
           offset_mode "gap" (common320.check_offset_mode) or whose provenance split differs from the grasps
           collection's demo320.split_identity (name, file, sha256, plan_sha; a checkpoint without it too); switches
           other than plan.json's (socket draw "per_reset", start_gap_from) or not approved (automate320.check_switches);
           an attempt reset seed equal to a grasp seed of the collection, --seed equal to the collection's entropy
           (automate320.check_plan_seeds); --max-retries other than 0 (decision 5); --probe-max > 0 (needs T8);
           --split any / extrap, pair_index grasps, ManiSkill grasps; test without --final-test; collect on anything but
           the training grasps (--smoke-grasp: named grasps, a comma list, for the smoke T7 only; any mode, but
           training grasps only for collect); switches other than plan.json's as a whole or a physics device other than plan.json
           physics_device; an asset folder (wm.sim_automate.assets.ASSET_DIR, WM_AUTOMATE_ASSETS) other than the
           collection's frozen <collection>/inputs/assets = plan.json input_roots.assets, or asset_files() other than
           plan.json asset_files; a collection config.json whose transfer is not automate320.TRANSFER (the Expert's);
           a checkpoint whose step is below --min-step. Recorded in job.json: the asset folder, the code identity
           (the frozen tree's code.sha256 and collect.physics_code_identity()). A job.json that exists must be the
           same job (a restart resumes it).
  worker   (Isaac; one capped systemd scope per start, scripts/automate/online_worker.sh, GPU 1 only) the AutoMate
           collector's worker infrastructure (wm.sim_automate.collect): heartbeat <job>/heartbeat (qx_watch.py), STOP
           file <job>/STOP, SIGTERM / SIGINT handlers, a flock claim per grasp <job>/claims/<port>/<grasp>/.claim,
           deaths <job>/deaths, scene-build failures <job>/build_failures (the collector's per-assembly skip limits),
           app.PRE_EXIT_HOOKS (the collector's: claim released, STOP on exit 7; this file's: the held grasp's result
           written as a crash before exit 8). The Isaac app starts BEFORE torch and the agent are imported (app.launch:
           Kit's GPU = config.KIT_GPU, the GPU-0 rule, exits 7 / 8). CUDA_VISIBLE_DEVICES must be 1; QX_GPU_GB (the
           scope's declaration) must exceed --torch-mib + 1 GB (the CUDA context); torch's allocator is capped at
           --torch-mib. The asset folder and the running code (code_identity) must be job.json's. One scene per assembly (closed by robot.sc.close() when
           the assembly changes); claim order: collect takes the grasps strictly in job.json's order (the round's
           seeded permutation, as ManiSkill's collect: early finished grasps cover many assemblies), every worker from
           its start; a test rotates the order by the worker index and prefers grasps of the assembly it has built.
           The scene build's seconds are in the grasp result (scene_build_s; None: the scene of the grasp before was
           used). Free disk: collect.DISK_FREE_MIN_CLAIM before a claim, DISK_FREE_MIN_SAVE before every trajectory
           file. Per grasp: for every start i, automate320.run_attempt (reset with the
           attempt's own seed and socket pose, reference, the collector's placement, the attempt); collect / --record:
           every attempt written as a fast-copy trajectory (write_fast_automate). A grasp result file is written whole
           (<results>/<port>/<grasp>.json, run_wm320_attempts' schema plus the AutoMate fields); a crashed grasp is run
           again until it has --retries + 1 tries (a process that died with the claim counts as a try; two deaths in
           one grasp end it as a crash, as the collector). A start whose placed state already satisfies AutoMate's rule
           (start_success; the entry is recorded with outcome "start_success") writes <job>/STOP and ends the worker
           with exit 3 (the run stops, plan item 5).
           Exit codes: 0 + "WORKER DONE" (no grasp left), 1 crash, 3 refused, 5 stopped, 6 scene build failure,
           7 / 8 the GPU check (app.py).
  finish   (CPU) the summary (summarise_automate: AutoMate's success, groups select / heldout_shape (cluster =
           assembly) / heldout_grasp_<variable> (cluster = grasp, band variable from plan.json grasp_kind) / test
           (cluster = assembly) / train (cluster = grasp); reference, placement, reset and contact-report failures,
           start_success counts, successes on assemblies whose success band reaches above the socket top) to
           summary.json (complete) or summary_incomplete.json; collect: <pool>/r<R>/manifest.json (entries, the split
           identity) and DONE when complete or at most --max-crashed-frac of the grasps crashed (exit 0; else exit 2,
           no DONE; exit 4 = EXIT_START_SUCCESS after a start_success stop, test or collect); --record:
           <record>/manifest.json. Every file is written atomically (written aside, then renamed). The summary also
           shows the online physics_code_sha256 beside the collection's (plan.json m1b_code; report only).

Fast-copy trajectories (plan item 7, option (b): demo320 unchanged): the meta holds only ONLINE_META_KEYS (all in
demo320.FAST_META_AUTOMATE, so demo320.entry_of accepts it; no reset seed, socket pose or other truth); the decision
kinds are the file's dec_kind array and the manifest entry's dec_kind; split_sha256 (the grasps collection's split
file sha256) is set on the entry after entry_of, as train_wm320.online_entries requires. The online-only facts (round,
start index, seeds, mode, model step, stop reason, interventions) are in the grasp result file.

    python run_wm320_attempts.py automate prepare --grasps <collection> --ckpt <pool>/r1/model.pt --mode collect \
        --split train --pool <pool> --round 1 --starts 3
    python -u run_wm320_attempts.py automate worker --job <pool>/r1 --index 0 --count 1
    python run_wm320_attempts.py automate finish --job <pool>/r1
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import eval_policy320 as E  # noqa: E402
import run_wm320_attempts as RA  # noqa: E402
from wm.online import automate320 as A  # noqa: E402
from wm.online.common320 import check_offset_mode  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]                   # world_model/
JOB_FORMAT = "wm320-automate-job-1"
EXIT_INCOMPLETE = 2
EXIT_START_SUCCESS = 4                                      # finish: a start_success stop (online_automate.sh: $P/STOP)
SPLITS = ("train", "select", "heldout", "test")
# the meta of an online / test trajectory (plan item 7): a subset of demo320.FAST_META_AUTOMATE
ONLINE_META_KEYS = ("grasp", "port", "condition", "kind", "outcome", "steps", "H_star", "axis", "pin_length_m",
                    "pitch_deg", "grasp_pos_m", "spin_deg", "u", "simulator", "symmetry_n", "switches", "assembly_id",
                    "config_sha", "plan_sha", "format", "source", "start")
GRASP_KEYS = ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "spin_deg", "u", "simulator", "assembly_id")
ATTEMPT_ARGS = ("margin", "p_shift", "eps", "candidates", "radius_mm", "random_reposition", "inside_mm", "servo_axis",
                "probe_gap_mm", "planner")
CRITERION = ("AutoMate check_plug_inserted_in_socket (0 < h < d_dis and mean keypoint distance < 15 mm) after any "
             "control step of the attempt; the placed start (row 0) is never counted")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- simulator-free parts
def write_fast_automate(path: Path, out: dict, meta: dict, split_sha256: str) -> dict:
    """One AutoMate attempt -> a fast-copy trajectory (run_wm320_attempts.fast_arrays / save_fast: the same codecs and
    arrays as ManiSkill's write_fast); the meta keeps ONLINE_META_KEYS only (anything else is refused, as entry_of
    refuses keys outside FAST_META_AUTOMATE). -> the manifest entry with dec_kind and split_sha256."""
    extra = sorted(set(meta) - set(ONLINE_META_KEYS))
    if extra:
        raise ValueError(f"{path}: online AutoMate meta keys outside ONLINE_META_KEYS: {extra}")
    if meta.get("simulator") != "automate":
        raise ValueError(f"{path}: an AutoMate trajectory needs meta simulator 'automate'")
    if not split_sha256:
        raise ValueError(f"{path}: no split sha256 (the grasps collection's split identity)")
    arrays = RA.fast_arrays(out)
    meta = {**meta, "steps": int(len(out["action"])), "format": D.FORMAT}
    RA.save_fast(path, out, meta, arrays)
    return {**D.entry_of(meta, meta["source"], len(out["rgb_jpeg_list"])), "dec_kind": arrays[4],
            "split_sha256": split_sha256}


def online_meta(g: dict, entry: dict, summ: dict, kind: str, rel: str, plan_info: dict, switches: dict,
                symmetry_n) -> dict:
    """The fast-copy meta of one attempt (ONLINE_META_KEYS; steps / format are added by write_fast_automate)."""
    return {"grasp": g["grasp"], "port": g["port"], "condition": g["condition"], "kind": kind,
            "outcome": "seated_fk" if summ["seated_fk"] else "not_seated", "H_star": entry["H_star"],
            "axis": entry["axis"], "pin_length_m": float(entry["pin_length_m"]), "pitch_deg": g["pitch_deg"],
            "grasp_pos_m": g["grasp_pos_m"], "spin_deg": g["spin_deg"], "u": g.get("u"), "simulator": "automate",
            "symmetry_n": symmetry_n, "switches": switches, "assembly_id": g["port"],
            "config_sha": plan_info.get("config_sha"), "plan_sha": plan_info.get("plan_sha"), "source": rel,
            "start": entry["start"]}


def code_identity() -> dict:
    """The code this process runs: the frozen tree's hash (<code>/../code.sha256 of collect_lib.sh cl_freeze_code;
    None outside a frozen copy) and collect.physics_code_identity() (the physics / robot / expert / truth files)."""
    from wm.sim_automate import collect as CO
    f = ROOT.parent / "code.sha256"
    tree = f.read_text().strip() if f.is_file() and ROOT.name == "code" else None
    return {"root": str(ROOT), "tree_sha256": tree, "physics_code_sha256": CO.physics_code_identity()}


def split_identity_matches(ck_split: dict | None, sid: dict | None) -> list[str]:
    """[] when the checkpoint's provenance split equals the collection's split identity in name, file, sha256 and
    plan_sha; else the differing keys (a missing identity on either side: ["missing"])."""
    if not isinstance(ck_split, dict) or not isinstance(sid, dict):
        return ["missing"]
    return [k for k in ("name", "file", "sha256", "plan_sha") if ck_split.get(k) != sid.get(k) or sid.get(k) is None]


def gi_of(key: str) -> int:
    """Grasp index of the key <assembly id>_gNN."""
    aid, _, n = key.rpartition("_g")
    if not aid or not n.isdigit():
        raise ValueError(f"grasp key {key!r} is not <assembly id>_gNN")
    return int(n)


def group_of(g: dict) -> tuple[str, str]:
    """(group, cluster) of an AutoMate grasp (plan item 10): select / train -> cluster grasp; heldout_shape -> cluster
    assembly; heldout_grasp -> group heldout_grasp_<band variable> (plan.json grasp_kind "band_<variable>"), cluster
    grasp; test -> cluster assembly."""
    c = D.automate_condition(g)
    if c in ("heldout_shape", "test"):
        return c, g["port"]
    if c == "heldout_grasp":
        kind = str(g.get("grasp_kind") or "")
        return ("heldout_grasp_" + kind[len("band_"):] if kind.startswith("band_") else "heldout_grasp"), g["grasp"]
    if c in ("train", "select"):
        return c, g["grasp"]
    raise ValueError(f"{g.get('grasp')}: no reporting group for condition {c!r}")


def result_tries(f: Path) -> int:
    try:
        return int(json.loads(f.read_text()).get("tries", 1))
    except (OSError, ValueError):
        return 1


def grasp_todo(f: Path, retries: int) -> bool:
    """A grasp is to be run when its result is missing, or a crash with fewer than retries + 1 tries."""
    st = E.grasp_state(f)
    return st == "missing" or (st == "crash" and result_tries(f) < 1 + retries)


def summarise_automate(out: Path, gs: list[dict], starts: int, extra: dict) -> tuple[dict, bool]:
    """The summary of an AutoMate job (module docstring, finish); the schema of eval_policy320.summarise plus the
    AutoMate counts. Success = the attempt's "success" (= automate_success_any)."""
    rows, count, outc, ref_fail = [], Counter(), Counter(), {}
    for g in gs:
        f = out / g["port"] / f"{g['grasp']}.json"
        state = E.grasp_state(f)
        count[state] += 1
        if state != "done":
            continue
        grp, cl = group_of(g)
        for st in json.loads(f.read_text())["starts"]:
            if "attempt" in st:
                a = st["attempt"]
                rows.append((grp, cl, bool(a["success"]), a["stopped"], g["grasp"], g["port"],
                             float(a.get("band_above_top_mm", 0.0)) > 0.0, bool(a.get("start_success")),
                             a.get("strict_success")))
            else:
                outc[st.get("outcome", "unknown")] += 1
                if st.get("outcome") in ("reference_failed", "reset_failed", "contact_report_error"):
                    why = (st.get("reference") or {}).get("failure") or st.get("error") or st.get("outcome")
                    ref_fail.setdefault(g["port"], []).append(str(why)[:200])
    groups = {}
    for name in ["all"] + sorted({r[0] for r in rows}):
        sel = [r for r in rows if name == "all" or r[0] == name]
        miss = sum(r[8] is None for r in sel)               # results of the code before strict_success: no strict figure
        st = (E.cluster_rate([bool(r[8]) for r in sel], [r[1] for r in sel]) if not miss
              else {"success": None, "se": None, "ci95": None})
        groups[name] = {**E.cluster_rate([r[2] for r in sel], [r[1] for r in sel]),
                        "strict_success": st["success"], "strict_se": st["se"], "strict_ci95": st["ci95"], "strict_missing": miss,
                        "grasps": len({r[4] for r in sel}),
                        "assemblies": len({r[5] for r in sel}),
                        "ended": dict(Counter(f"{r[3]}/{'success' if r[2] else 'failure'}" for r in sel)),
                        "successes_on_band_above_top": sum(r[2] and r[6] for r in sel),
                        "attempts_on_band_above_top": sum(r[6] for r in sel)}
    complete = count["crash"] == 0 and count["missing"] == 0
    summary = {"criterion": CRITERION, "simulator": "automate", "complete": complete, "planned_grasps": len(gs),
               "planned_starts": len(gs) * starts, "judged_starts": len(rows),
               "placement_failed_starts": outc["placement_failed"], "reference_failed_starts": outc["reference_failed"],
               "reset_failed_starts": outc["reset_failed"], "contact_report_error_starts": outc["contact_report_error"],
               "start_success_attempts": sum(r[7] for r in rows), "unusable_grasps": count["setup"],
               "reference_failures_per_assembly": ref_fail,
               "crashed_grasps": count["crash"] + count["missing"],
               "band_note": ("AutoMate's success region reaches above the socket top on assemblies with "
                             "min(d_dis, 15 mm) > L_top: a plug resting on the socket top there counts as success "
                             "(successes_on_band_above_top per group)"), "groups": groups, **extra}
    return summary, complete


# ---------------------------------------------------------------------------------------------- prepare
def prepare_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_wm320_attempts.py automate prepare")
    ap.add_argument("--grasps", required=True, help="the AutoMate collection (plan.json, split file, <id>/<id>_gNN/grasp.json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", required=True, choices=["act", "act_done", "wm", "collect", "servo", "random"])
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--out", help="tests: the job dir (result files <out>/<port>/<grasp>.json)")
    ap.add_argument("--pool", help="collect: the online pool (job dir <pool>/r<round>)")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--grasps-n", type=int)
    ap.add_argument("--smoke-grasp", help="smoke (T7) only: these grasps (comma list), any mode (also a test mode on "
                    "training grasps); collect: training grasps only")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--torch-mib", type=int, default=1500, help="torch memory cap per worker")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--max-retries", type=int, default=0, help="must be 0 on AutoMate (the user's decision 5 of 2026-11-25)")
    ap.add_argument("--inside-mm", type=float, default=None)
    ap.add_argument("--planner", choices=["fixed", "mppi"], default="fixed",
                    help="wm / collect: the fixed candidates or TD-MPC2's MPPI (agent320.MPPI; the user's decision of 2026-11-26)")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--radius-mm", type=float, default=3.0)
    ap.add_argument("--random-reposition", type=float, default=0.5)
    ap.add_argument("--p-shift", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--servo-axis", choices=["true", "motion"], default="true")
    ap.add_argument("--random-p")
    ap.add_argument("--record", help="test modes: also write every attempt as a fast-copy trajectory under this directory")
    ap.add_argument("--probe-max", type=int, default=0, help="must be 0 on AutoMate until the restore check T8")
    ap.add_argument("--probe-gap-mm", type=float, default=6.0)
    ap.add_argument("--max-crashed-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=9_100_000, help="the evaluation entropy of the attempt streams")
    ap.add_argument("--retries", type=int, default=1, help="a crashed grasp is run again until it has retries + 1 tries")
    ap.add_argument("--min-step", type=int, default=None, help="refuse a checkpoint whose step is below this")
    return ap


def build_job(args) -> tuple[Path, dict]:
    """The checks of prepare (module docstring) -> (job dir, job). SystemExit on any refusal."""
    collect = args.mode == "collect"
    if args.split == "test" and not args.final_test:
        raise SystemExit("the test assemblies are the final test and stay untouched: --split test needs --final-test")
    if args.smoke_grasp is None and collect != (args.split == "train"):
        raise SystemExit("collect runs on the training grasps only, and the training grasps are only collected on")
    if collect and not (args.pool and args.round > 0):
        raise SystemExit("collect needs --pool and --round >= 1")
    if collect and (args.record or args.out):
        raise SystemExit("collect writes its trajectories and results to the pool (<pool>/r<round>): no --record / --out")
    if not collect and (not args.out or args.pool):
        raise SystemExit("a test needs --out (and no --pool)")
    if args.max_retries != 0:
        raise SystemExit("--max-retries must be 0 on AutoMate (no back-off: the user's decision 5 of 2026-11-25)")
    if args.probe_max:
        raise SystemExit("--probe-max must be 0 on AutoMate until the simulator-state restore check (plan T8) has been run")
    if args.mode == "act_done" and args.inside_mm is not None:
        raise SystemExit("--mode act_done has no candidate corrections and does not accept --inside-mm")
    if args.planner == "mppi" and (args.mode not in ("wm", "collect") or args.inside_mm is None):
        raise SystemExit("--planner mppi plans in the modes wm and collect only, and needs --inside-mm (the in-hole rule)")
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
    if args.starts < 1 or args.torch_mib < 1 or args.retries < 0:
        raise SystemExit("--starts and --torch-mib must be >= 1, --retries >= 0")
    W = Path(args.pool) / f"r{args.round}" if collect else Path(args.out)
    grasps_dir = Path(args.grasps).resolve()
    # the checkpoint (CPU): offset mode, offset head, split identity
    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_offset_mode = str(ck.get("offset_mode", "along"))
    try:
        check_offset_mode(ck_offset_mode, "automate")
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.inside_mm is not None:
        if args.inside_mm <= 0:
            raise SystemExit("--inside-mm must be > 0 (a depth inside the hole)")
        if not ck.get("offset_weight", 0.0) > 0:
            raise SystemExit(f"--inside-mm needs a checkpoint with the offset read-out head; {args.ckpt} has none")
    sid = D.split_identity(grasps_dir)
    if sid is None or not sid.get("sha256"):
        raise SystemExit(f"{grasps_dir} has no plan.json with a split file: not an AutoMate collection with a decided split")
    bad = split_identity_matches((ck.get("provenance") or {}).get("split"), sid)
    if bad:
        raise SystemExit(f"{args.ckpt} was trained on the split {(ck.get('provenance') or {}).get('split')}, the grasps "
                         f"collection has {sid} (differs in {bad}): a model is evaluated on its own collection's split only")
    model_step = int(ck.get("step", -1))
    del ck
    if args.min_step is not None and model_step < args.min_step:
        raise SystemExit(f"{args.ckpt} has step {model_step} < --min-step {args.min_step}")
    plan = json.loads((grasps_dir / "plan.json").read_text())
    switches = A.check_switches(plan)
    assets = A.check_inputs(grasps_dir, plan)
    every = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{grasps_dir}/*/*/grasp.json"))]
    if not every or not all(D.is_automate(g) for g in every):
        raise SystemExit(f"{grasps_dir}: AutoMate grasps only (found {sum(not D.is_automate(g) for g in every)} others)")
    if any("pair_index" in g for g in every):
        raise SystemExit("pair_index grasps (make_combo_grasps.py, ManiSkill's test ports) are not AutoMate grasps")
    D.check_automate_conditions(every)
    by_key = {p["grasp"]: p for p in plan["grasps"]}
    every = [g for g in every if g.get("reference_ok")]
    if args.smoke_grasp is not None:
        want = [k for k in args.smoke_grasp.split(",") if k]
        every = [g for g in every if g["grasp"] in want]
        if not want or len(set(want)) != len(want) or len(every) != len(want):
            raise SystemExit(f"--smoke-grasp {args.smoke_grasp}: not a list of distinct grasps with reference_ok "
                             f"(found {sorted(g['grasp'] for g in every)})")
        bad = [g["grasp"] for g in every if D.split_of_entry(g) == "test" or (collect and D.split_of_entry(g) != "train")]
        if bad:
            raise SystemExit(f"--smoke-grasp: {bad} " + ("not training grasps (--mode collect)" if collect else
                                                          "in the test split"))
    else:
        every = [g for g in every if D.split_of_entry(g) == args.split]
    chosen = E.pick(len(every), args.grasps_n)
    gs = []
    for g in (every[k] for k in chosen):
        p = by_key.get(g["grasp"])
        if p is None or int(p["gi"]) != gi_of(g["grasp"]) or p["port"] != g["port"]:
            raise SystemExit(f"{g['grasp']}: not in plan.json (or another grasp index / assembly)")
        gs.append({**{k: g.get(k) for k in GRASP_KEYS}, "gi": int(p["gi"]), "grasp_kind": p.get("grasp_kind")})
    if not gs:
        raise SystemExit(f"no grasp of split {args.split} with reference_ok in {grasps_dir}")
    rnd = args.round if collect else None
    resets = [A.attempt_streams(args.seed, g["port"], g["gi"], i, rnd)["reset_seed"] for g in gs for i in range(args.starts)]
    A.check_plan_seeds(plan, args.seed, resets)
    order = [g["grasp"] for g in gs]
    if collect:                                                  # random order: the learner reads a round while it runs
        order = [order[i] for i in np.random.default_rng(args.seed + 1_000_003 * args.round).permutation(len(order))]
    else:
        order = [g["grasp"] for g in sorted(gs, key=lambda g: g["port"])]
    results = W / "results" if collect else W
    traj_root = Path(args.pool).resolve() if collect else (Path(args.record).resolve() if args.record else None)
    job = {"format": JOB_FORMAT, "kind": "collect" if collect else "test", "mode": args.mode, "split": args.split,
           "smoke_grasp": args.smoke_grasp, "grasps_dir": str(grasps_dir), "ckpt": str(Path(args.ckpt).resolve()),
           "ckpt_sha256": sha256_file(args.ckpt), "model_step": model_step, "offset_mode": ck_offset_mode,
           "round": args.round, "pool": str(Path(args.pool).resolve()) if collect else None,
           "record": str(Path(args.record).resolve()) if args.record else None, "results": str(results.resolve()),
           "traj_root": None if traj_root is None else str(traj_root), "traj_sub": f"r{args.round}/traj" if collect else "traj",
           "starts": args.starts, "seed": args.seed, "retries": args.retries, "grasps_n": args.grasps_n,
           "torch_mib": args.torch_mib, "horizon": args.horizon, "max_crashed_frac": args.max_crashed_frac,
           "max_retries": 0, "probe_max": 0, "random_p": random_p,
           "attempt_args": {k: getattr(args, k) for k in ATTEMPT_ARGS}, "split_identity": sid,
           "plan": {"config_sha": plan.get("config_sha"), "plan_sha": plan.get("plan_sha"),
                    "rng_entropy": (plan.get("rng") or {}).get("entropy"), "m1b_code": plan.get("m1b_code")},
           "switches": switches, "physics_device": plan.get("physics_device"), "assets": assets,
           "code": code_identity(), "min_step": args.min_step, "grasps": gs, "order": order}
    return W, job


def prepare(argv: list[str]) -> int:
    args = prepare_parser().parse_args(argv)
    collect = args.mode == "collect"
    if collect and args.pool and args.round > 0 and (Path(args.pool) / f"r{args.round}" / "DONE").exists():
        log(f"{Path(args.pool) / f'r{args.round}'} is complete already")
        return 0
    try:
        W, job = build_job(args)
    except SystemExit as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 3
    f = W / "job.json"
    if f.exists():
        old = json.loads(f.read_text())
        if old != job:
            diff = sorted(k for k in set(old) | set(job) if old.get(k) != job.get(k))
            print(f"REFUSED: {f} holds another job (differs in {diff}); use a new job dir", flush=True)
            return 3
    elif job["record"] and Path(job["record"]).exists() and any(Path(job["record"]).iterdir()):
        print(f"REFUSED: --record {job['record']} is not empty", flush=True)
        return 3
    elif any(E.grasp_state(Path(job["results"]) / g["port"] / f"{g['grasp']}.json") != "missing" for g in job["grasps"]):
        print(f"REFUSED: {job['results']} has result files of another job", flush=True)
        return 3
    W.mkdir(parents=True, exist_ok=True)
    _write_atomic(f, json.dumps(job, indent=1))
    log(f"PREPARED {W}: {len(job['grasps'])} grasps of split {job['split']} x {job['starts']} starts, mode {job['mode']}, "
        f"model step {job['model_step']}, split {job['split_identity']['sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------------------------- worker
_CUR: dict = {"path": None, "s": None, "claim_attempt": None}   # the grasp result being made (pre-exit hook, exit 8)


def _pre_exit_result(code: int):
    """app.PRE_EXIT_HOOKS (runs before the collector's hook releases the claim): before exit 8 the held grasp's result
    is written as a crash (one try), so that the next worker runs it again within --retries (plan item 9)."""
    from wm.sim_automate import app as _app
    if code != _app.EXIT_GPU_UNREADABLE or _CUR["s"] is None or _CUR["path"] is None:
        return
    s = dict(_CUR["s"])
    s.update(error=f"exit {code}: nvidia-smi unreadable at the GPU check", error_kind="crash",
             claim_attempt=_CUR["claim_attempt"], claim_kept=True)      # the collector's hook restores the claim text
    _write_atomic(Path(_CUR["path"]), json.dumps(s))


def foreign_wm_modules() -> list[str]:
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT.resolve()):
            bad.append(f"{name} ({f})")
    return bad


def worker_checks(job: dict) -> None:
    """Before the Isaac start (exit 3 on failure)."""
    from wm.sim_automate import config as C
    if job.get("format") != JOB_FORMAT:
        raise SystemExit(f"job.json format {job.get('format')!r}, this code reads {JOB_FORMAT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly 1 (GPU 1 only), found {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    bad = foreign_wm_modules()
    if bad:
        raise SystemExit(f"wm modules outside the code tree {ROOT}: {bad}")
    gb = os.environ.get("QX_GPU_GB", "")
    if not gb.isdigit() or int(gb) * 1e9 < job["torch_mib"] * 2 ** 20 + 1e9:
        raise SystemExit(f"QX_GPU_GB={gb!r}: the scope's GPU declaration must be at least --torch-mib {job['torch_mib']} "
                         "MiB + 1 GB (CUDA context); this check does not include Kit's own GPU memory (size QX_GPU_GB "
                         "from Kit's measured peak in the smoke T7)")
    if C.switches() != job["switches"]:
        raise SystemExit(f"switches {C.switches()} differ from the job's {job['switches']}")
    if C.PHYSICS_DEVICE != job["physics_device"]:
        raise SystemExit(f"physics device {C.PHYSICS_DEVICE!r} differs from the job's {job['physics_device']!r}")
    from wm.sim_automate import assets as AS
    if Path(AS.ASSET_DIR).resolve() != Path(job["assets"]).resolve():
        raise SystemExit(f"asset folder {AS.ASSET_DIR} (WM_AUTOMATE_ASSETS) differs from the job's {job['assets']}")
    if code_identity() != job["code"]:
        raise SystemExit(f"the running code {code_identity()} differs from the job's {job['code']} (prepare ran other code)")
    if sha256_file(job["ckpt"]) != job["ckpt_sha256"]:
        raise SystemExit(f"{job['ckpt']} changed since the job was prepared")


def run_grasp(robot, fk, agent, g: dict, job: dict, W: Path, beat, tries: int,
              scene_build_s: float | None = None) -> tuple[dict, bool]:
    """All starts of one grasp -> (result, scene_ok). scene_ok False: an exception inside the grasp (the scene is closed
    and built again for the next grasp). automate320.StartSuccess: its entry (outcome "start_success") is recorded and
    the exception raised again (the run stops)."""
    import torch
    from wm.sim_automate import collect as CO
    s = {**{k: g[k] for k in GRASP_KEYS}, "gi": g["gi"], "grasp_kind": g.get("grasp_kind"), "stop": job["mode"],
         "mode": job["mode"], "round": job["round"], "model_step": agent.step, "switches": job["switches"],
         "split_sha256": job["split_identity"]["sha256"], "max_decisions": int(robot.max_decisions),
         "band_above_top_mm": A.band_above_top_mm(robot.port_spec), "code": job["code"], "assets": job["assets"],
         "scene_build_s": None if scene_build_s is None else round(scene_build_s, 1), "starts": [], "entries": [],
         "error": "interrupted", "error_kind": "crash", "tries": tries}
    _CUR["s"] = s
    t0, scene_ok = time.time(), True
    s["started_unix"] = round(t0, 3)                       # the smoke lists a worker's grasps in this order
    root, sub = (Path(job["traj_root"]) if job["traj_root"] else None), job["traj_sub"]
    akw = {"margin": job["attempt_args"]["margin"], "max_retries": 0, "p_shift": job["attempt_args"]["p_shift"],
           "eps": job["attempt_args"]["eps"], "n_candidates": job["attempt_args"]["candidates"],
           "radius_mm": job["attempt_args"]["radius_mm"], "random_reposition": job["attempt_args"]["random_reposition"],
           "inside_mm": job["attempt_args"]["inside_mm"], "servo_axis": job["attempt_args"]["servo_axis"],
           "random_p": job["random_p"], "probe_max": 0, "probe_gap_mm": job["attempt_args"]["probe_gap_mm"],
           "planner": job["attempt_args"].get("planner", "fixed")}
    try:
        if root is not None:                                   # a retried grasp: no file of an earlier try stays behind
            shutil.rmtree(root / sub / g["port"] / g["grasp"], ignore_errors=True)
        rnd = job["round"] if job["kind"] == "collect" else None
        for i in range(job["starts"]):
            CO.check_stop(W)
            streams = A.attempt_streams(job["seed"], g["port"], g["gi"], i, rnd)
            entry, out, _ = A.run_attempt(robot, agent, fk, g, i, streams, job["mode"], akw, beat=beat)
            s["starts"].append(entry)
            if out is not None and root is not None:
                rel = f"{sub}/{g['port']}/{g['grasp']}/s{i}.npz"
                meta = online_meta(g, entry, entry["attempt"], "online" if job["kind"] == "collect" else "test", rel,
                                   job["plan"], job["switches"], robot.port_spec.symmetry_n)
                CO._disk_check(W, CO.DISK_FREE_MIN_SAVE, "before a trajectory file")
                s["entries"].append(write_fast_automate(root / rel, out, meta, job["split_identity"]["sha256"]))
            del out
            torch.cuda.empty_cache()
        del s["error"], s["error_kind"]
    except A.StartSuccess as exc:
        if exc.entry is not None and not any(e is exc.entry for e in s["starts"]):
            s["starts"].append(exc.entry)
        s.update(error=f"start_success: {exc}", error_kind="crash", start_success_stop=True, tries=1 + job["retries"],
                 trace=traceback.format_exc()[-2000:])
        raise
    except (CO.Stopped, SystemExit):
        raise
    except Exception as exc:
        s.update(error=f"{type(exc).__name__}: {exc}"[:500], error_kind="crash", trace=traceback.format_exc()[-2000:])
        scene_ok = False
    finally:
        s["seconds"] = round(time.time() - t0, 1)
        s["memory"] = CO._memory()
    return s, scene_ok


def claim_candidates(job: dict, index: int, count: int, port: str | None, todo, blocked) -> list[str]:
    """The grasps a worker tries to claim, in claim order (module docstring, worker): collect: job["order"] (the
    round's permutation) strictly, the same for every worker; tests: job["order"] rotated by the worker index, grasps
    of the built assembly `port` first. todo(key) -> bool; blocked: assemblies skipped after build failures."""
    gs = {g["grasp"]: g for g in job["grasps"]}
    order = list(job["order"])
    if job["kind"] != "collect":
        k0 = (index * len(order)) // max(count, 1)
        order = order[k0:] + order[:k0]
    cands = [k for k in order if gs[k]["port"] not in blocked and todo(k)]
    if job["kind"] != "collect":
        cands.sort(key=lambda k: gs[k]["port"] != port)            # stable: same assembly first
    return cands


def tries_before(f: Path, claim_attempt: int) -> tuple[int, int]:
    """(tries in the crash result f, process deaths after it) for a grasp claimed with claim attempt number
    claim_attempt. A process that died holding the claim counts as a try (module docstring). A finished grasp result
    deletes the claim file (attempt numbers start at 1 again): deaths = claim_attempt - 1. The exit-8 result
    (_pre_exit_result, claim_kept) counts the deaths before it in its tries and the collector's hook puts the claim's
    previous text back: deaths = claim_attempt - its claim_attempt. A stop or a build failure releases without a
    death and without a result."""
    if E.grasp_state(f) != "crash":
        return 0, max(0, int(claim_attempt) - 1)
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        r = {}
    base = int(r.get("claim_attempt") or 1) if r.get("claim_kept") else 1
    return int(r.get("tries", 1)), max(0, int(claim_attempt) - base)


def run_worker(job: dict, W: Path, index: int, count: int, hb) -> None:
    """The claim loop (module docstring, worker). Returns when no grasp is left."""
    import torch
    from wm.data.kinematics import HandFK
    from wm.online.agent320 import WMAgent
    from wm.sim_automate import collect as CO
    from wm.sim_automate.robot import Robot
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(1.0, job["torch_mib"] / total), 0)
    hb("agent")
    agent = WMAgent(job["ckpt"], "cuda", horizon=job["horizon"])
    gs = {g["grasp"]: g for g in job["grasps"]}
    results = Path(job["results"])
    path = {k: results / gs[k]["port"] / f"{k}.json" for k in job["order"]}
    robot, port, fk = None, None, None
    while True:
        CO.check_stop(W)
        CO._disk_check(W, CO.DISK_FREE_MIN_CLAIM, "before a claim")
        cands = claim_candidates(job, index, count, port, lambda k: grasp_todo(path[k], job["retries"]),
                                 CO.blocked_assemblies(W))
        key, c = None, None
        for cand in cands:
            c = CO.claim(W / "claims" / gs[cand]["port"] / cand)
            if c is None:
                continue
            if not grasp_todo(path[cand], job["retries"]):            # finished by another worker meanwhile
                CO.release(c, done=True)
                continue
            key = cand
            break
        if key is None:
            break
        g, f = gs[key], path[key]
        hb("claimed", grasp=key, claim=str(c.path))
        done_tries, deaths = tries_before(f, c.attempt)
        tries = done_tries + deaths + 1
        if c.old is not None:
            n = CO._record_death(W, g["port"], key, c.old)
            log(f"{key}: attempt {c.old['attempt']} (pid {c.old['pid']}) died; {n} deaths in {g['port']}")
        if c.attempt > 2 or tries > 1 + job["retries"]:
            why = ("the worker process died twice in this grasp" if c.attempt > 2 else
                   f"{done_tries} crashed tries and {deaths} process deaths: no try left (--retries {job['retries']})")
            _write_atomic(f, json.dumps({**{k: g[k] for k in GRASP_KEYS}, "starts": [], "entries": [],
                                         "tries": max(tries - 1, 1 + job["retries"]), "deaths": deaths,
                                         "error": why, "error_kind": "crash"}))
            CO.release(c, done=True)
            continue
        _CUR["path"], _CUR["claim_attempt"] = str(f), c.attempt
        build_s = None
        try:
            if port != g["port"] or robot is None:
                if robot is not None:
                    hb("close")
                    CO._close(robot)
                robot, port, fk = None, None, None
                hb(f"build {g['port']}")
                t_build = time.time()
                try:
                    robot, port = Robot(g["port"]), g["port"]
                except Exception as exc:
                    CO.release(c, done=False)                          # not a try of this grasp
                    msg = f"{type(exc).__name__}: {exc}"
                    counted = not CO.NETWORK_OR_GPU_MEMORY.search(msg)
                    n = CO._record_build_failure(W, g["port"], exc, counted)
                    log(f"{key}: building the scene failed ({'counted' if counted else 'not counted'}; {n} of "
                        f"{CO.BUILD_FAILURES_MAX} counted): {msg}")
                    raise CO.BuildFailed(f"{g['port']}: {msg}") from exc
                fk = HandFK(robot)
                build_s = time.time() - t_build
            s, scene_ok = run_grasp(robot, fk, agent, g, job, W, hb, tries, build_s)
        except A.StartSuccess as exc:
            s = _CUR["s"]
            _write_atomic(f, json.dumps(s))
            msg = f"{key}: {exc}"
            with open(W / CO.STOP_FILE, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start_success: {msg}\n")
            CO.alert(W, f"start_success: {msg}; STOP written (the run stops)")
            CO.release(c, done=True)
            raise CO.Refused(f"start_success: {msg}")
        except (CO.Stopped, CO.BuildFailed, SystemExit):
            if not c.released:
                CO.release(c, done=False)
            raise
        s["process"] = dict(CO.PROCESS_INFO)
        s["claim_attempt"] = c.attempt
        _write_atomic(f, json.dumps(s))
        _CUR["s"], _CUR["path"] = None, None
        CO.release(c, done=True)
        hb("done", grasp=None, claim=None)
        o = " | ".join(f"{st['attempt']['stopped'][:5]}/{'ok' if st['attempt']['success'] else 'no'}/i{st['attempt']['interventions']}"
                       if "attempt" in st else str(st.get("outcome")) for st in s["starts"])
        log(f"{key} {o} {str(s.get('error', ''))[:150]} {s.get('seconds')}s (scene build {s.get('scene_build_s')}s) "
            f"mem {s.get('memory')}")
        if not scene_ok:
            hb("close")
            CO._close(robot)
            robot, port, fk = None, None, None


def worker(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_wm320_attempts.py automate worker")
    ap.add_argument("--job", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--count", type=int, default=1)
    try:
        args = ap.parse_args(argv)
        W = Path(args.job)
        job = json.loads((W / "job.json").read_text())
        from wm.sim_automate import collect as CO
        worker_checks(job)
    except SystemExit as exc:
        print(f"REFUSED: {exc}", flush=True)
        os._exit(3)
    except Exception:
        traceback.print_exc()
        print("REFUSED: error before the Isaac start (job.json or the checks)", flush=True)
        os._exit(3)
    hb = CO.Heartbeat(W, args.index)
    CO.PROCESS_INFO.update(pid=os.getpid(), start=hb.info["start"], unit=hb.info["unit"])
    print(f"WORKER PID {os.getpid()} START {hb.info['start']} UNIT {hb.info['unit']}", flush=True)
    hb("launch")
    t_launch = time.time()
    from wm.sim_automate import app as _app
    _app.PRE_EXIT_HOOKS.insert(0, _pre_exit_result)       # before the collector's hook releases the claim
    _app.launch(headless=True, cameras=True)              # Isaac before torch (plan item 9)
    CO.PROCESS_INFO["launch_s"] = round(time.time() - t_launch, 1)
    log(f"Isaac started in {CO.PROCESS_INFO['launch_s']} s; memory {CO._memory()}")
    CO.install_stop_handlers(log)
    code = CO.EXIT_CRASH
    try:
        bad = foreign_wm_modules()
        if bad:
            raise CO.Refused(f"wm modules outside the code tree {ROOT}: {bad}")
        run_worker(job, W, args.index, args.count, hb)
        print("WORKER DONE", flush=True)
        code = CO.EXIT_DONE
    except CO.Stopped as exc:
        print(f"WORKER STOPPED: {exc}", flush=True)
        code = CO.EXIT_STOPPED
    except CO.BuildFailed as exc:
        print(f"WORKER BUILD FAILED: {exc}", flush=True)
        code = CO.EXIT_BUILD
    except SystemExit as exc:
        print(f"REFUSED: {exc}", flush=True)
        CO.alert(W, f"worker {args.index} refused after the Isaac start: {exc}")
        code = CO.EXIT_REFUSED
    except BaseException:
        traceback.print_exc()
        code = CO.EXIT_CRASH
    finally:
        hb.close()
    _app.shutdown(code)
    return code


# ---------------------------------------------------------------------------------------------- finish
def finish(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_wm320_attempts.py automate finish")
    ap.add_argument("--job", required=True)
    args = ap.parse_args(argv)
    W = Path(args.job)
    job = json.loads((W / "job.json").read_text())
    out, gs, collect = Path(job["results"]), job["grasps"], job["kind"] == "collect"
    extra = {"policy": f"wm320 agent ({job['mode']})", "ckpt": job["ckpt"], "model_step": job["model_step"],
             "split": job["split"], "offset_mode": job["offset_mode"], "mode": job["mode"], "round": job["round"],
             "starts_per_grasp": job["starts"], "horizon": job["horizon"], "max_retries": 0, "probe_max": 0,
             **job["attempt_args"], "random_p": job["random_p"], "record": job["record"], "seed": job["seed"],
             "grasps_n": job["grasps_n"], "smoke_grasp": job["smoke_grasp"], "split_identity": job["split_identity"],
             "switches": job["switches"], "max_decisions": "AutoMate rule per assembly (150 + ceil(max(0, L - 20 mm) / 0.5 mm))",
             "left_m": "0.06 + L", "physics_device": job["physics_device"], "assets": job["assets"], "code": job["code"],
             "min_step": job["min_step"],
             # report only: the online code's physics identity beside the collection's (plan.json m1b_code: the
             # milestone-1b results' physics_code_sha256, or their code_files_sha256 when field says so)
             "physics_code_sha256": {"online": job["code"]["physics_code_sha256"],
                                     "collection_m1b": (job["plan"].get("m1b_code") or {}).get("sha256"),
                                     "collection_m1b_field": (job["plan"].get("m1b_code") or {}).get("field")}}
    summary, complete = summarise_automate(out, gs, job["starts"], extra)
    rows = [json.loads((out / g["port"] / f"{g['grasp']}.json").read_text()) for g in gs
            if E.grasp_state(out / g["port"] / f"{g['grasp']}.json") == "done"]
    att = [st["attempt"] for r in rows for st in r.get("starts", []) if "attempt" in st]
    summary["attempts"] = len(att)
    summary["seated_fk"] = float(np.mean([x["seated_fk"] for x in att])) if att else None
    summary["interventions_per_attempt"] = float(np.mean([x["interventions"] for x in att])) if att else None
    summary["retries_per_attempt"] = float(np.mean([x["retries"] for x in att])) if att else None
    stopped = []
    for g in gs:
        f = out / g["port"] / f"{g['grasp']}.json"
        if f.exists():
            try:
                if json.loads(f.read_text()).get("start_success_stop"):
                    stopped.append(g["grasp"])
            except ValueError:
                pass
    summary["start_success_stop"] = stopped
    _write_atomic(out / ("summary.json" if complete and not stopped else "summary_incomplete.json"), json.dumps(summary, indent=1))
    print(E.show(summary), flush=True)
    entries = [e for r in rows for e in r.get("entries", [])]
    if job["record"]:
        _write_atomic(Path(job["record"]) / "manifest.json", json.dumps({
            "format": D.FORMAT, "simulator": "automate", "ckpt": job["ckpt"], "split": job["split_identity"],
            "split_name": job["split"], "offset_mode": job["offset_mode"], "mode": job["mode"], "entries": entries}))
    if stopped:
        print(f"start_success in {stopped}: the run stopped (plan item 5); no DONE", flush=True)
        return EXIT_START_SUCCESS
    if collect:
        _write_atomic(W / "manifest.json", json.dumps({"format": D.FORMAT, "simulator": "automate", "round": job["round"],
                                                       "ckpt": job["ckpt"], "offset_mode": job["offset_mode"],
                                                       "split": job["split_identity"], "code": job["code"],
                                                       "entries": entries}))
        crashed = summary["crashed_grasps"] / max(1, summary["planned_grasps"])
        if entries and (complete or crashed <= job["max_crashed_frac"]):
            _write_atomic(W / "DONE", json.dumps({"entries": len(entries), "complete": complete,
                                                  "crashed_grasps": summary["crashed_grasps"]}))
            print(f"round {job['round']}: {len(entries)} trajectories in the pool", flush=True)
            return 0
        print(f"round {job['round']}: {len(entries)} trajectories, {summary['crashed_grasps']} crashed grasps: not marked "
              "complete", flush=True)
        return EXIT_INCOMPLETE
    return 0 if complete else EXIT_INCOMPLETE


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("prepare", "worker", "finish"):
        print("usage: run_wm320_attempts.py automate prepare|worker|finish ... (module docstring)", flush=True)
        return 3
    return {"prepare": prepare, "worker": worker, "finish": finish}[argv[0]](argv[1:])


__all__ = ["write_fast_automate", "online_meta", "summarise_automate", "split_identity_matches", "group_of", "gi_of",
           "grasp_todo", "claim_candidates", "tries_before", "code_identity", "build_job", "prepare",
           "worker", "finish", "main", "ONLINE_META_KEYS", "EXIT_START_SUCCESS"]
