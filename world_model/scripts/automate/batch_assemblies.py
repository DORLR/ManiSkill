"""Milestone 1b: every assembly in simulation (table mount, nominal grasp unless stated), one app, one scene at a time.
These checks validate the ENVIRONMENT (physics, grasp boxes), not the task: (ii) and (iii) keep the agreed physical
meaning "reaches seated" (|depth_left| < 0.5 mm, as check (i)); AutoMate's success rule (the task criterion) is
recorded next to them (lead's resolution 2026-11-23 of r3-decisions-n1).

Per assembly (results in <out>/<id>.json; every reset uses socket pose 0 of seed 0, the CPU PhysX default):
  (i)   H* reached: from the reset (plug aligned, lowest point 12 mm above the highest socket point) move_to(H*),
        hold 5 steps. Pass: move_ok AND |depth_left| < 0.5 mm AND max penetration < 0.05 mm (checks.penetration).
        Recorded with checks.truth (height_mm, success, ...) and h_end_le_0.01mm, as (ii) / (iii) (information;
        r6-expert-02, r7-expert-01: move_to (2 mm waypoints at 20 mm/s, then settling) can end up to ~15 um below h = 0;
        the expert's 3 mm/s push lags its command and settles onto h = 0 from above, measured h_end 26-104 um over
        the 186 expert_smoke ends on 00015 / 00083 / 01102, never below).
  (ii)  aligned straight insertion from g = 10 mm in front of the entrance: move_to(H* - (L + 10 mm) a), then the
        commanded TCP moves along a at 10 mm/s to H* (stop "blocked" when it leads the FK hand by > 1.5 mm, or "ik"),
        hold 10 steps.
            pass_ii = placed AND completed AND |depth_left_end| < 0.5 mm AND S_end
        placed = the start was reached (move_to); completed = the straight push reached H* (push_end "done": not
        blocked, no IK failure); depth_left at the plug tip centre (checks.truth); S_end = AutoMate's
        check_plug_inserted_in_socket after the last step. Recorded: success_any (after any control step of the push
        and hold), success_end, height_mm, keypoint_dist_mm, lateral, tilt, h_end_le_0.01mm (information: AutoMate's
        h > 0 is strict, a seated plug rests tens of micrometres above h = 0).
  (iii) the grasp-range box point with the smallest mesh margin (grasp.py, if that margin < 3 mm): reset with it,
        move_to(H*), hold 5 steps.
            pass_iii = move_ok AND completed AND |depth_left| < 0.5 mm AND F_hf <= grasp.III_MAX_HAND_FINGER_N (1 N)
        move_ok = completed = move_to reached H* within 0.1 mm / 0.05 deg and settled (the only motion of (iii));
        F_hf = max over the hold of the contact sensor's largest hand / finger - socket force and the contact report's
        robot-socket pair sum (robot_N). 1 N is an information-level threshold of this margin check only, not a task
        criterion. AutoMate's success at the end is recorded alongside (success_end).
(i) and (ii) use the nominal grasp (0, 0, 0), also for assemblies without a box (reset allow_outside=True, recorded).
--parts selects which of i, ii, iii run. Every result records, at the top level: time (ISO) and unix_time, the code
identity (collect.code_identity files_sha256 = code_files_sha256; collect.physics_code_identity = physics_code_sha256,
the files that decide physics, robot, expert and truth, collect.PHYSICS_CODE_FILES), ranges_file and ranges_sha256 of
the assembly, grasp_py_sha256 (the grasp formula), spec_file and the asset hashes (collect.asset_identity: spec_sha256
of the geometry file, plug_obj_sha256, socket_obj_sha256, and the same as asset_sha256 {file name: sha256}; the geometry
decides L, d_dis, tip_centre, the symmetry and g_top, so (ii) / (iii) of another geometry are not current;
r5-integration-03 / r5-expert-03), shared_asset_sha256 (collect.shared_asset_identity: plug_grasps.json and the finger
collision meshes franka_mimic_collision/*.obj, {path: sha256}; r9-integration-01), switches (config.switches), physics_device, and usd_identity (sha256 of every USD
layer the scene loaded; scene.AutomateScene.usd_identity); each part also carries its own unix_time. The inclusion gate
(collect.py stale_fields) accepts a result only if ranges_sha256, grasp_py_sha256, physics_device, switches, the
asset hashes and shared_asset_sha256 equal the current ones (lead's resolution 2026-11-23; asset hashes since round 6, 2026-11-23);
physics_code_sha256 lets it ignore collector-only changes.
Every result also records thread_env (app.thread_env: PXR_WORK_THREAD_LIMIT, OPENBLAS_NUM_THREADS, ...; app.launch
limits SimulationApp's pools to config.PHYSX_CPU_THREADS) and usd_compared: the sha256 of the layers the collector
compares with this result (collect.USD_COMPARED: plug.usd, socket.usd, plug_sensor.usd), with usd_compared_ok True only
when each of them was loaded exactly once and read.
Switches: before the app starts, config.refuse_unapproved refuses any switch value outside config.APPROVED (exit 3,
nothing written), unless WM_AUTOMATE_DIAGNOSTIC=1. numpy is imported before the app, so OpenBLAS keeps the
launcher's thread count.
A process crash inside an assembly is recorded (marker file <id>.started, then an "error" result) and the batch
continues in a new process (run_batch.sh). An exit of app.assert_off_forbidden_gpu (7: GPU 0 rule broken, 8:
nvidia-smi unreadable) first removes the marker (app.PRE_EXIT_HOOKS): it is not a crash of the assembly. There is
deliberately no SIGTERM handler: a SIGTERM from `timeout` means an assembly hung, and its marker must stay so that the
next process books that assembly as an error instead of retrying it forever; a stop by the watcher (shed) is told
apart and its markers released by run_batch.sh, which knows the shed marker (r4-ops-01 / r4-ops-N1). Kit does not exit
cleanly on SIGTERM: its own SIGTERM path hits a carb assertion (carb.tasking Mutex.cpp 'Recursion not allowed') and
aborts with a core dump (SIGABRT; rs2 m1/logs/batch_m1b_1.log, batch2s_1.log), so a stopped process exits with 134
through `timeout` (124 only when `timeout` itself fired), never 143; run_batch.sh therefore tells a shed by the shed
marker, not by the exit code (r6-ops-chk-01). It releases only the markers this process wrote (not older than its
start); a marker left by an earlier process after a hang stays, so that this process books it after launch() (the
booking loop at the end of this file runs only after the app has started; r6-ops-02).
Heartbeat: <out>/heartbeat/<pid>_<start>.json in qx_watch.py's format (pid, process start time, unit, timeout_s,
claim None, stage "<id> <part>"), rewritten at every stage and touched at most every BEAT_EVERY_S during the checks.
When run_batch.sh registers <out> with the watcher (cl_watch_ensure), a process silent for HANG_TIMEOUT_S is killed and
its marker stays (booked as an error by the next process). Removed at the normal end and before an exit 7 / 8.
    python -u batch_assemblies.py [--out /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu] [--ids a,b] [--parts i,ii,iii]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402  (before the app: OpenBLAS keeps the launcher's thread count, module docstring)

EXIT_REFUSED = 3                            # run_batch.sh treats 3 as terminal (no restart)
try:
    from wm.sim_automate import config as C  # noqa: E402  (plain numpy module; refuses a mount / camera / GPU switch)
except RuntimeError as _exc:
    print(f"refused: {_exc}", flush=True)
    sys.exit(EXIT_REFUSED)
from wm.sim_automate.app import PRE_EXIT_HOOKS, launch, shutdown, thread_env  # noqa: E402

CURRENT = {"mark": None, "heartbeat": None}  # the <id>.started marker of the assembly being run; the heartbeat file


def release_marker(code: int):
    """app.PRE_EXIT_HOOKS: an exit of the GPU check (7 / 8) is not a crash of the assembly: remove its marker (and the
    heartbeat file, so the watcher does not count the exit as a hang)."""
    m = CURRENT["mark"]
    if m is not None and m.exists():
        m.unlink()
        print(f"exit {code}: removed {m} (not a crash of this assembly)", flush=True)
    hb = CURRENT["heartbeat"]
    if hb is not None:
        hb.close()


PRE_EXIT_HOOKS.append(release_marker)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1b/batch_r3_cpu")
ap.add_argument("--ids", default="")
ap.add_argument("--parts", default="i,ii,iii", help="subset of i,ii,iii")
ap.add_argument("--device", default=None, help="PhysX device (default config.PHYSICS_DEVICE)")
args = ap.parse_args()
try:
    C.refuse_unapproved("batch_assemblies.py")
except RuntimeError as exc:
    print(f"refused: {exc}", flush=True)
    sys.exit(EXIT_REFUSED)
app = launch(headless=True, cameras=False)

from wm.sim.geometry import pose_distance  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate.collect import (BEAT_EVERY_S, HANG_TIMEOUT_S, USD_COMPARED, _start_time,  # noqa: E402
                                     asset_identity, code_identity, own_unit, physics_code_identity, sha256_file,
                                     shared_asset_identity)
from wm.sim_automate import grasp as G  # noqa: E402
from wm.sim_automate.robot import Robot  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
CODE = code_identity()
PHYS = physics_code_identity()
DEVICE = args.device or C.PHYSICS_DEVICE
DEPTH_SEATED_MM = 0.5                       # |depth_left| of a seated plug in (i), (ii), (iii) (module docstring)


class Heartbeat:
    """<out>/heartbeat/<pid>_<start>.json in the format qx_watch.py check_heartbeats reads (module docstring)."""

    def __init__(self, out: Path, timeout_s: int = HANG_TIMEOUT_S):
        d = out / "heartbeat"
        d.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        start = _start_time(pid)
        self.path = d / f"{pid}_{start}.json"
        self.info = {"pid": pid, "start": start, "index": "batch", "unit": own_unit(), "timeout_s": int(timeout_s),
                     "claim": None, "grasp": None, "stage": "start", "time": None}
        self._last = 0.0
        self._write()

    def _write(self):
        self.info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_atomic(self.path, json.dumps(self.info))
        self._last = time.monotonic()

    def __call__(self, stage: str | None = None):
        """With a stage: rewrite the file; without: touch it at most every BEAT_EVERY_S."""
        if stage is not None:
            self.info["stage"] = stage
            self._write()
        elif time.monotonic() - self._last >= BEAT_EVERY_S:
            try:
                os.utime(self.path)
            except OSError:
                self._write()
            self._last = time.monotonic()

    def close(self):
        self.path.unlink(missing_ok=True)


def provenance(aid: str) -> dict:
    rf = G.ranges_file(aid)
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "unix_time": time.time(),
            "code_files_sha256": CODE["files_sha256"], "physics_code_sha256": PHYS, "ranges_file": rf.name,
            "ranges_sha256": sha256_file(rf) if rf.exists() else None, "grasp_py_sha256": sha256_file(G.__file__),
            "spec_file": A.SPEC_FILE, **asset_identity(aid), "shared_asset_sha256": shared_asset_identity(),
            "switches": C.switches(), "physics_device": DEVICE,
            "thread_env": thread_env()}


def usd_compared(identity: dict) -> dict:
    """The layers of usd_identity whose identifier ends in one of collect.USD_COMPARED -> {"layers": {suffix:
    [(identifier, sha256 or 'unreadable: ...'), ...]}, "ok": each suffix loaded exactly once and read}."""
    layers = identity.get("layers") if isinstance(identity, dict) else None
    found = {sfx: [] for sfx in USD_COMPARED}
    for ident, h in (layers or {}).items():
        for sfx in USD_COMPARED:
            if str(ident).endswith("/" + sfx) or str(ident) == sfx:
                found[sfx].append([ident, h])
    ok = layers is not None and all(len(v) == 1 and not str(v[0][1]).startswith("unreadable") for v in found.values())
    return {"layers": found, "ok": bool(ok)}


def write_atomic(f: Path, text: str):
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(text)
    tmp.replace(f)


def beat(stage: str | None = None):
    hb = CURRENT["heartbeat"]
    if hb is not None:
        hb(stage)


def contact_max(r, n):
    f, g, rb = 0.0, 0.0, 0.0
    for _ in range(n):
        beat()
        r._command(r._q_cmd, 1)
        f = max(f, r.sc.finger_socket_force())
        g = max(g, float(np.linalg.norm(r.sc.contact_force())))
        rb = max(rb, r.sc.socket_contacts(r.env_id)["robot_N"])
    return f, g, rb


def nominal_reset(r, aid):
    no_box = r.check_grasp_in_box(0.0, 0.0, 0.0) is not None
    r.reset(0, clamp_deg=0.0, dp_m=0.0, spin_deg=0.0, allow_outside=no_box, socket_index=0)
    return no_box


def run(aid: str) -> dict:
    res = {"assembly_id": aid, **provenance(aid)}
    t0 = time.time()
    beat(f"{aid} build")
    try:
        r = Robot(aid, camera=False, device=DEVICE)
    except Exception:
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        try:                                                     # leave a clean stage for the next assembly
            ctx = sim_utils.SimulationContext.instance()
            if ctx is not None:
                ctx._disable_app_control_on_stop_handle = True
                ctx.stop()
                ctx.clear_all_callbacks()
                ctx.clear_instance()
        finally:
            stage_utils.create_new_stage()
        raise
    res["build_s"] = time.time() - t0
    try:
        res["usd_identity"] = r.sc.usd_identity()
    except Exception as exc:                                     # recorded, never fatal for the checks
        res["usd_identity"] = {"error": f"{type(exc).__name__}: {exc}"}
    cmp_ = usd_compared(res["usd_identity"])
    res["usd_compared"], res["usd_compared_ok"] = cmp_["layers"], cmp_["ok"]
    s, L = r.port_spec, r.port_spec.pin_length_m
    res["L_mm"], res["max_decisions"] = L * 1e3, r.max_decisions
    try:
        parts = args.parts.split(",")
        if "i" in parts:
            beat(f"{aid} i")
            no_box = nominal_reset(r, aid)
            H = r.H_star()
            ok = r.move_to(H)
            fmax, pmax, rbmax = contact_max(r, 5)
            t = K.truth(r)
            dt, dr = pose_distance(r.hand_pose(), H)
            ok = bool(ok)
            i = {"move_ok": ok, **t, **K.penetration(r), "h_end_le_0.01mm": bool(t["height_mm"] <= 0.01),
                 "hand_to_H*_mm": dt * 1e3, "nominal_outside_box": no_box,
                 "hand_to_H*_deg": float(np.degrees(dr)), "hand_finger_socket_N": fmax, "plug_socket_N": pmax,
                 "robot_socket_pairs_N": rbmax, "unix_time": time.time()}
            i["pass"] = bool(ok and abs(t["depth_left_mm"]) < DEPTH_SEATED_MM and i["max_penetration_mm"] < 0.05)
            i["pass_rule"] = "move_ok AND |depth_left| < 0.5 mm AND max_penetration < 0.05 mm"
            res["i"] = i
        if "ii" in parts:
            beat(f"{aid} ii")
            nominal_reset(r, aid)
            H = r.H_star()
            placed = bool(r.move_to(K.along(r, H, L + 0.010)))
            t_start = K.truth(r)
            peak = [0.0, 0.0]
            any_success = [False]

            def rec():
                beat()
                peak[0] = max(peak[0], float(np.linalg.norm(r.sc.contact_force())))
                peak[1] = max(peak[1], r.sc.finger_socket_force())
                any_success[0] = any_success[0] or K.truth(r)["success"]
            p = K.drive_line(r, r.hand_pose(), H, 0.010, rec, lag_stop=0.0015)
            for _ in range(10):
                r._command(p["q"], 1)
                rec()
            t = K.truth(r)
            completed = p["end"] == "done"
            res["ii"] = {"placed": placed, "start": t_start, "push_end": p["end"], "push_steps": p["steps"], **t,
                         "completed": completed, "success_end": bool(t["success"]), "success_any": bool(any_success[0]),
                         "h_end_le_0.01mm": bool(t["height_mm"] <= 0.01), "max_plug_socket_N": peak[0],
                         "max_hand_finger_socket_N": peak[1], "unix_time": time.time(),
                         "pass": bool(placed and completed and abs(t["depth_left_mm"]) < DEPTH_SEATED_MM and t["success"]),
                         "pass_rule": "placed AND completed (push_end done) AND |depth_left_end| < 0.5 mm AND "
                                      "AutoMate success after the last step"}
        if "iii" in parts:
            rg = G.load_ranges(aid)
            wp, wm_ = rg.get("worst_point"), rg.get("worst_margin_m")
            if wp is not None and wm_ is not None and wm_ < 0.003:
                beat(f"{aid} iii")
                r.reset(0, clamp_deg=wp[0], dp_m=wp[1], spin_deg=wp[2], socket_index=0)
                H = r.H_star()
                ok = bool(r.move_to(H))
                fmax, pmax, rbmax = contact_max(r, 5)
                t = K.truth(r)
                f_hf = max(fmax, rbmax)
                res["iii"] = {"grasp": wp, "mesh_margin_mm": wm_ * 1e3, "move_ok": ok, "completed": ok, **t,
                              "success_end": bool(t["success"]), "h_end_le_0.01mm": bool(t["height_mm"] <= 0.01),
                              "hand_finger_socket_N": fmax, "plug_socket_N": pmax, "robot_socket_pairs_N": rbmax,
                              "hand_finger_max_N": f_hf, "unix_time": time.time(),
                              "pass": bool(ok and abs(t["depth_left_mm"]) < DEPTH_SEATED_MM
                                           and f_hf <= G.III_MAX_HAND_FINGER_N),
                              "pass_rule": "move_ok AND completed AND |depth_left| < 0.5 mm AND max(hand_finger_socket_N, "
                                           f"robot_socket_pairs_N) <= {G.III_MAX_HAND_FINGER_N} N"}
            else:
                res["iii"] = {"skipped": "no box" if wp is None else f"mesh margin {wm_ * 1e3:.1f} mm >= 3 mm",
                              "unix_time": time.time()}
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-2000:]
    res["seconds"] = time.time() - t0
    r.sc.close()
    return res


ids = args.ids.split(",") if args.ids else A.assembly_ids()
CURRENT["heartbeat"] = Heartbeat(OUT)
for aid in ids:
    f, mark = OUT / f"{aid}.json", OUT / f"{aid}.started"
    if f.exists():
        continue
    if mark.exists():                                            # an earlier process died inside this assembly
        write_atomic(f, json.dumps({"assembly_id": aid, **provenance(aid), "error": "the simulator process died during "
                                    "this assembly", "started": mark.read_text().strip()}))
        mark.unlink()                                            # the marker is resolved by the result file
        continue
    mark.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    CURRENT["mark"] = mark
    try:
        res = run(aid)
    except Exception as exc:
        res = {"assembly_id": aid, **provenance(aid), "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc()[-2000:]}
    write_atomic(f, json.dumps(res, indent=1, default=float))
    mark.unlink()
    CURRENT["mark"] = None
    print(aid, json.dumps({k: res.get(k, {}).get("pass") if isinstance(res.get(k), dict) else res.get(k)
                           for k in ("i", "ii", "iii", "error", "seconds")}), flush=True)
CURRENT["heartbeat"].close()
print("BATCH DONE", flush=True)
shutdown(0)
