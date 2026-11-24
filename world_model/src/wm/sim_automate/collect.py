"""Expert demonstrations on AutoMate in the wm_v1 format (the role of wm.collect.{plan, episode, runner}).

Per assembly and grasp (one directory <out>/<assembly id>/<assembly id>_gNN/, as <out>/<port>/<port>_gNN/ in wm_v1):
    t00_reference.npz       the reference insertion from the reset spawn (aligned, 12 mm above the socket top);
                            H_star = FK hand pose at its end, axis = direction of the FK hand during its axial push
                            (episode._axis_from), in grasp.json and, with socket draw "per_grasp", every
                            trajectory's meta. With "per_reset" (AutoMate's original, config.SOCKET_DRAW) every
                            planned trajectory first resets with its own socket pose (socket_index = file index) and
                            runs an unrecorded reference insertion there (Expert.reference_label): H_star / axis of
                            its meta come from that one (meta label_reference; a failed one gives outcome
                            "label_failed" and no file). grasp.json timing counts resets, their settling steps and
                            these label steps apart from the recorded and transfer steps.
    t01-t04_insert.npz      start (unrecorded transfer) -> approach -> align -> insert
    t05-t06_perturbed.npz   start -> stochastic window -> recovery align -> insert with insertion noise
    t07-t11_contact.npz     start -> approach / align -> offset -> straight push, stop on jam
    t12_random.npz          start -> three random waypoints in front of the socket
    grasp.json              summary, written last and atomically (its presence marks the grasp done)
Recorded with wm.data.recorder.Recorder (unchanged, via record.py) and written with wm.data.store.save (unchanged):
keys, dtypes, units, frames, the 5 Hz picture schedule and the truth field eval_rel are those of wm_v1. Added truth
array (evaluation only; never copied to the fast copy): eval_tip (T + 1, len(EVAL_TIP_COLUMNS)) float32 =
checks.truth_from_rel_spec of eval_rel, columns EVAL_TIP_COLUMNS = checks.TRUTH_COLUMNS (AutoMate's success as 0 / 1,
height_mm, keypoint_dist_mm, then the ManiSkill-style information numbers). Row 0 is the start state, row t the state
after control step t. grasp.json lists every trajectory; "file" names its .npz, and only trajectories with "file" have
one (label_failed, placement_failed, reset_failed, contact_report_error and error rows have none).

Meta / grasp.json fields that mean something different on AutoMate, or are new (all other fields as wm_v1):
    port          the assembly id ("00015"); grasp = "<id>_gNN"
    pitch_deg     the clamp angle offset d_theta [deg] (rotation about the hand y axis, wm.sim_automate.grasp)
    grasp_pos_m   the offset along the plug axis d_p [m] (> 0 = fingers closer to the tip)
    spin_deg      (new) the spin d_psi about the plug axis [deg]
    u             (new) the normalised grasp variables u = (v - v_min) / (v_max - v_min) in this assembly's box;
                  None for a variable whose range has zero width (listed in plan.json, the held-out band cannot apply)
    condition     the split of the grasp: split "automate_v1" (rule at SPLIT_VERSION below; production) gives
                  train / select / heldout_grasp / heldout_shape / test; "unassigned" only outside production
    grasp_kind    (new, plan.json) automate_v1: "train_range" or "band_<variable>" (the held-out band grasp)
    outcome       insert / perturbed: "inserted" iff S_any (AutoMate's episode success, the reported success number;
                  decision 1 of 2026-11-22), else "expert_failed". With h_t, k_t the plug height and mean keypoint
                  distance after control step t = 1..T (checks.automate_success, Isaac Lab's
                  check_plug_inserted_in_socket) and d_dis the assembly's disassembly distance:
                      s_t   = [0 < h_t < d_dis  AND  k_t < 15 mm]
                      S_any = OR_{t = 1..T} s_t      (the start state t = 0 is never evaluated, as in AutoMate, which
                                                      checks only after each step)
                  Recorded as automate_success_any (= S_any), success (with inserted_end = s_T, first_success_step =
                  the smallest t with s_t, success_at_start = s_0 for information) and expert_completed (the expert
                  passed Chongyu's final_verify). The imitation set (demo320 bc) needs outcome "inserted" AND
                  expert_completed, as in ManiSkill (lead resolution r3, option A). contact: "reached_depth" when the
                  straight push reached the wm_v1 depth (expert.contact_push "inserted"), else its own stop reason
                  ("jammed", ...), never "inserted" (that name is kept for AutoMate's rule). random: "done".
                  "reached_depth" keeps wm_v1's contact_push rule unchanged (wm/env/expert.py:115-134): the push ends
                  "inserted" when the command reaches the seat (x = 0) while the plug lags by less than JAM_LAG_M
                  (not jammed), so a row can read "reached_depth" with depth_left_mm > 1 mm; depth_left_mm is recorded
                  in the same row (r4-expert-08).
                  The ManiSkill-style truth (checks.truth: depth / lateral / tilt) and finger_contact (peak hand /
                  finger - socket force > 1 N) are information only.
    reference     (t00, and the unrecorded per-reset label references) accepted iff the expert completed AND s_T
                  (AutoMate's rule holds at its end): reference_ok = completed AND inserted_end (lead resolution r3,
                  as wm_v1's episode.py:62 plus AutoMate's rule). Both parts are recorded
                  (reference_expert_completed, reference_success); a refused per-reset label gives "label_failed".
    H_star, axis  as wm_v1: the FK hand pose at the end of the reference insertion; axis from the FK hand during the
                  axial push. A reference without an axial push (< 2 steps) has no FK axis: the grasp is closed with
                  reference_ok False ("no axial push"); the true axis is never used in its place. H_star is the taught pose
                  a real robot can record. The task definition writes H* = T_socket T_hp^-1 (simulator truth); that value is kept as
                  grasp.json H_star_model, with H_star_fk_vs_model_mm_deg (0.04-0.12 mm, < 0.01 deg in the smoke runs;
                  per_grasp: also in every trajectory meta; per_reset: in meta label_reference).
    start         Offset.as_dict of the start: tip gap g0 [mm], lateral [mm] along the goal frame's y, z
                  (frames.frame(a) columns 1, 2 = -e2, e1), static XYZ angles [deg] in that frame about the TCP. g0 is
                  measured from where config.START_GAP_FROM says (switches.start_gap_from): "socket_top" (the user's
                  decision of 2026-11-23, option B of r4-expert-01) = from the higher of {socket top, entrance}, i.e.
                  the TCP of the start lies L + g_top + g0 back from H* along a (frames.start_pose)
    g_top_m       (new) g_top = max(L_top, L) - L of the assembly (checks.g_top_of), the height of the higher of
                  {socket top, entrance} above the entrance
    offset / waypoints   Offset.as_dict in the expert's pivot convention (expert.py): top gap [mm], lateral of the
                  plug tip centre [mm], static XYZ angles [deg] about it
    peak          the expert's peak contact forces [N] as expert.py reports them (plug_socket_N from the contact
                  report's per-shape-pair impulse sums, finger_socket_N from the hand / finger contact sensors, ...)
    pin_length_m, assembly_id, simulator (new): L, the assembly, "automate"
    symmetry_n, tip_centre_m (new): the plug's rotational symmetry and tip centre (truth conversions, labels)
    eval_socket_pose (new, truth): the episode's socket pose (AutoMate's per-episode randomisation, config.socket_pose)
    config_sha, plan_sha, physics_device (new): the frozen collection (plan.json) and the PhysX device
    switches      (new) config.switches() of the process (socket draw, grasp spin axis, success rule; environment
                  variables WM_AUTOMATE_*), frozen in plan.json and fixed for a collection by <out>/inputs/switches.env
    qpos[7:9]     (wm_v1 key, unchanged) the finger opening: on AutoMate an exact function of the true grasp
                  (grasp.FingerSocketCheck.closing of (d_theta, d_p, d_psi)), so it reveals the grasp. It is never a
                  model input: every model reads qpos[:7] (demo320, wmdata320, agent320, policy320).
    rng (new): the random streams of this trajectory (below)

Grasp ranges: always the verified box of grasp.load_ranges (grasp.RANGES_FILE). An assembly is collected when
    box exists and verified  AND  current(ii) AND pass_ii  AND  ((iii) skipped OR (current(iii) AND pass_iii))
(milestone-1b checks of batch_assemblies.py; they validate the environment, not the task; lead resolution r3):
    pass_ii  = placed AND completed AND |depth_left_mm| < 0.5 AND s_end      (batch_assemblies "pass", re-checked here
                                                                              from the recorded numbers)
    pass_iii = pass AND move_ok AND completed AND |depth_left_mm| < 0.5 AND F_hf <= III_MAX_FINGER_N (1 N, an
               information-level threshold for the margin check only, not a task criterion), with
               F_hf = hand_finger_max_N = max(contact-sensor hand / finger - socket force, contact-report robot-socket
               pair sum) as batch_assemblies.py records it (grasp.py module docstring, pass_iii)
    "pass", "placed" / "move_ok" and "completed" must be recorded and true; a missing key or a non-number force fails.
For (ii) and (iii) separately the result is the newest one (by unix_time) without "error" among the files <id>.json
of cfg.m1b_dirs (default M1B_DIRS: only the re-run under the current code with CPU PhysX). current(x): the result's
provenance equals this process's: ranges_sha256 = sha256(ranges file), grasp_py_sha256 = sha256(grasp.py),
physics_device = cfg.device, switches = config.switches(), the asset hashes spec_sha256 / plug_obj_sha256 /
socket_obj_sha256 / asset_sha256 = asset_identity (geometry file assets.SPEC_FILE and plug / socket meshes of the
collector's asset folder: they decide L, d_dis, tip_centre and L_top, hence (ii)'s pass and the expert's pivot and push
depth; r5-integration-03), shared_asset_sha256 = shared_asset_identity (plug_grasps.json and the finger collision meshes
franka_mimic_collision/*.obj: AutoMate's grasp and the finger opening / start clearance; r9-integration-01) and
physics_code_sha256 = physics_code_identity() (a result without any of them is stale);
otherwise the assembly is excluded with "re-run: stale <field>" (assemblies_force cannot override
this). All included results must share one physics_code_sha256 (or, without it, one code_files_sha256); plan.json
"m1b_code" records it. USD identity: per built scene, the sha256 of the layers plug.usd, socket.usd and
plug_sensor.usd (scene.usd_identity) must equal those recorded by the assembly's (ii) result (grasp.json usd_vs_m1b;
otherwise Refused and an ALERT line: the Isaac nucleus asset changed after the gate; a forced assembly without a (ii)
result is recorded, not refused). "(iii) skipped" means only that no simulator check was needed inside that box (its
smallest mesh margin is >= 3 mm); a (iii) result counts only if its grasp point lies in the current box and its mesh
margin equals the box's smallest mesh margin within 0.01 mm (otherwise "re-run (iii)").
Every assembly gets a status row (included / excluded / forced, with the reason, the depth_left_mm of (ii) / (iii) and
the start-box property below) in plan.json; the table is printed.
Start box and AutoMate's rule (information, r3-decisions-06, r4-expert-01): with b = frames.start_back(L, g0, g_top,
config.START_GAP_FROM) (b = L + g_top + g0 for "socket_top", the user's decision of 2026-11-23; L + g0 for "entrance",
diagnostics only),
an aligned start (no lateral offset, no rotation) has h = b and k = b, so it already satisfies s_t iff
b < min(d_dis, 15 mm). Row "start_box" (None if the assembly's spec or AutoMate grasp cannot be read, or it has no box):
    h_min_mm  = b at g0 = g0_min (aligned start),  possible = [h_min < min(d_dis, 15 mm)]
    fraction  = share of START_SUCCESS_DRAWS draws (grasp (d_theta, d_p, d_psi) uniform in the assembly's box;
                start = sample_start, with tip gap, lateral and rotation; socket yaw of AutoMate's draw) for which
                checks.start_box_success (the start's plug pose under AutoMate's rule, gap_from =
                config.START_GAP_FROM) says success
    plus gap_from, g_top_mm, d_dis_mm, limit_mm, draws. Not modelled: the start clearance check (a start that the
    collector would redraw is still counted) and the 0.04-0.12 mm between the FK H* and the model H*.

Random streams. Every stream is numpy default_rng(SeedSequence(entropy=cfg.seed, spawn_key=(int(assembly id), grasp
index, trajectory index, purpose))), with purpose STREAMS[...]: grasp values (trajectory index 0), trajectory plan
(index 0), episode seed (index 0; the reset seed, which also draws the socket pose), placement redraws, random-waypoint
redraws (decision 7 of 2026-11-23) and expert noise (trajectory index = file index, 0 = reference). Different
assemblies, grasps, trajectories and purposes therefore never share random numbers (wm_v1 used default_rng(seed + id)
per shape and one seed_offset for both redraws and noise). seed_offset is still drawn per trajectory (same order of draws as wm.collect.plan) and recorded,
but no longer seeds anything.

Frozen plan. config.json and plan.json are written once per output folder (atomically, under a lock) with config_sha,
plan_sha, the sha256 of every input file (ranges, geometry, milestone-1b results, plug / socket meshes, plug_grasps.json,
the Franka collision meshes), the names of the versioned geometry and ranges files ("asset_files": assets.SPEC_FILE,
grasp.RANGES_FILE, read from the code, never named in this module) and the code identity (sha256 of the collector's source files, plus the git commit if the
folder is a git checkout). Every worker rebuilds the plan and refuses to run if any of it differs. A first plan.json
is refused when the folder already holds grasp data (an older plan's grasps would count as done). The launchers
(scripts/automate/collect_launch.sh) run everything from a read-only copy of the code (<out>/code) and of the inputs
(<out>/inputs: assets via WM_AUTOMATE_ASSETS, milestone-1b results via WM_AUTOMATE_M1B, the config), so that later
edits of the live tree or of the inputs do not reach a running collection.

Production config. CollectConfig.load(path, production=True) (used by collect_expert.py) refuses a config file that
does not name every key of PRODUCTION_KEYS explicitly (physics device, grasps per assembly, split, expected number of
grasps, milestone-1b folders): these are the user's decisions and inputs and have no silent default. The plan must
have exactly expected_grasps grasps. The physics device must equal config.PHYSICS_DEVICE (WM_AUTOMATE_DEVICE of the
frozen switches.env). collect_subset (smoke runs only, empty in production): the split and the gate are computed over
every assembly as in production, but only these assemblies' grasps are planned.
Pinned values of a production config (validate; a different value is refused, naming the agreement):
    split automate_v1 and, without collect_subset, grasps_per_assembly = 15 (decision 7 of 2026-11-22; a smoke with
    collect_subset keeps >= 4); trajectories (kinds, numbers and order), random_waypoints, images, image_every,
    rgb_codec, jpeg_quality = wm_v1's (wm.collect.plan.CollectConfig defaults: 13 trajectories per grasp as in
    wm_v1); random_speed_m_s = this module's default ((2.7, 2.7) mm/s: wm_v1's constant random-motion speed,
    0.9 x close_speed 3 mm/s, wm/env/expert.py:95; restored by the user's decision of 2026-11-23, deviation table row
    27; before it a speed was drawn per waypoint in 3-15 mm/s. The field and its per-waypoint draw are kept, so the
    trajectory-plan stream's order of draws is unchanged; uniform(a, a) returns a, and expert.execute refuses any
    speed other than expert.RANDOM_SPEED_M_S. The expert moves every random waypoint and the contact offset with
    wm_v1's _move_outside at that speed and CLOSE_ANG_SPEED_DEG_S = 2 deg/s, no ramp, no hold (decision 4 of
    2026-11-23, deviation rows 27 and 38); grasp.json stage rows record speed_m_s / ang_speed_deg_s); the switches
    config.APPROVED (config.unapproved_switches() empty: socket draw, spin axis, weld, PhysX threads, Kit GPU, mount,
    camera scale, forbidden GPU, start gap from the socket top (decision of 2026-11-23)).
Selection grasps (r4-integration-02 / r4-data-01): split automate_v1 counts, over every included assembly that is
neither test nor held out and every train-range grasp index (whatever collect_subset is), n_select (condition
"select"), n_heldout_grasp and n_heldout_shape (plan.json split "counts", printed). train_wm320.py refuses fewer than
demo320.MIN_SELECT_GRASPS (m = 20) selection grasps or no held-out trajectory. Grasps whose t00 reference is refused
later add no entries, so a production plan (no collect_subset, expected_grasps set) needs the margin
    n_select >= m_prod = ceil(SELECT_MARGIN_FACTOR x m) = ceil(1.25 x 20) = 25    (r5-integration-02)
and n_heldout_grasp > 0 and n_heldout_shape > 0; otherwise it is Refused before plan.json is written. A smoke subset
(collect_subset or no expected_grasps) is never refused: n_select < m_prod gives a WARNING line, n_select < m names
train_wm320's own limit. Both numbers and the margin n_select - m are printed for the user.

Workers and crashes. A worker claims a grasp by holding an exclusive flock on <grasp>/.claim for the whole grasp (the
kernel releases it when the process dies; the file records attempt, pid and process start time). A claim found
unlocked with an unfinished attempt is a process death ("crash"; "hang" when the watcher qx_watch.py killed a worker
whose heartbeat stopped and marked the claim): each one is also recorded once in <out>/deaths/<id>.json, and an
assembly with DEATHS_MAX deaths over its grasps is skipped. After two deaths in one grasp, the grasp gets grasp.json
with that error. Inside a process, each trajectory runs in its own try: an exception that is not an expert failure
marks that trajectory "error", the Robot is closed and rebuilt, and the grasp is retried once (stale t*.npz removed
first; the first error is kept in grasp.json "retries"); a second such exception writes the error into grasp.json. Any
other exception inside a grasp is written to grasp.json once (no process death). A failure to build the Robot never
writes grasp.json: the claim is released, the failure recorded in <out>/build_failures/<id>.json (with "counted":
false for a network or GPU-memory error: the USDs still come from the Isaac nucleus) and the worker exits with
EXIT_BUILD (the restart loop backs off outside any scope); after BUILD_FAILURES_MAX counted failures of one assembly,
workers skip it. Both skip lists are reversible: move the file away (scripts/automate/collect_unblock.sh). Every
worker writes <out>/logs/summary.json when it finds no grasp left (skipped assemblies with reasons, grasps without
grasp.json, trajectory files without grasp.json) and appends to <out>/logs/ALERT if anything was skipped or left.

Stopping. <out>/STOP (scripts/automate/collect_stop.sh) makes every worker stop before its next claim or trajectory;
SIGTERM / SIGINT (systemctl --user stop of the worker's scope) stops at once. Both release the claim with its previous
text (not a death) and exit with EXIT_STOPPED. Heartbeat: <out>/heartbeat/<pid>_<start>.json (pid, process start time,
unit, claim, stage, timeout_s), touched at most every BEAT_EVERY_S during recorded steps and at every stage.

Exits by the GPU check (app.assert_off_forbidden_gpu, os._exit, which skips finally blocks): the module registers
_pre_exit in app.PRE_EXIT_HOOKS. It releases the held claim with its previous text (not a death), drops the heartbeat
and alerts; for EXIT_FORBIDDEN_GPU it also writes <out>/STOP, so that every worker of the collection stops and no
loop restarts (a real GPU-0 violation). Disk: a worker stops (Stopped, <out>/STOP with the reason) when <out> has
less than DISK_FREE_MIN_CLAIM bytes free before a claim, or less than DISK_FREE_MIN_SAVE before a trajectory file is
written (the claim is released, the grasp is not marked failed).

Exit codes of collect_expert.py: EXIT_DONE 0 (no grasp left), EXIT_CRASH 1, EXIT_REFUSED 3 (config, plan, inputs or code
differ or are invalid; never retried), EXIT_STOPPED 5 (STOP file or signal), EXIT_BUILD 6 (scene build failure),
EXIT_FORBIDDEN_GPU 7 (this process holds more than a small graphics context on GPU 0: the run stops, no restart),
EXIT_GPU_UNREADABLE 8 (nvidia-smi could not be read for the GPU check: a counted crash, rate-capped).
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data import schema, store
from ..env.sampling import Offset, StartBox, sample_contact_offset, sample_start
from . import app as _app
from . import assets as A
from . import checks as K
from . import config as C
from . import grasp as G

REFERENCE_ATTEMPTS = 1          # one attempt: a same-seed retry repeats the first; wm_v1's new-seed retry (seed + 7919 k)
                                # is not ported because it would redraw the socket pose (deviation row 22)
PLACEMENT_ATTEMPTS = 3          # as wm.collect.episode
# grasp.json stage rows: the expert's phase, its steps and failure, and the speeds the expert recorded for the phase
# (expert._move_outside: RANDOM_SPEED_M_S, CLOSE_ANG_SPEED_DEG_S for contact_offset / random_motion_i)
STAGE_KEYS = ("phase", "steps", "failure", "speed_m_s", "ang_speed_deg_s")
SELECT_MARGIN_FACTOR = 1.25     # a production plan needs n_select >= ceil(1.25 x MIN_SELECT_GRASPS) (r5-integration-02)
# milestone-1b results: only the re-run under the current code with CPU PhysX (lead resolution r3); every result is
# also checked for its provenance (assembly_status), so an older folder named here would only give "re-run: stale"
M1B_DIRS = ["/mnt/storage3/qingxin/automate/m1b/batch_r3_cpu"]
III_MAX_FINGER_N = G.III_MAX_HAND_FINGER_N   # pass_iii: hand / finger - socket force <= 1 N (grasp.py's one constant;
                                             # information-level threshold of the margin check, lead resolution r3)
SEAT_DEPTH_MM = 0.5             # pass_ii / pass_iii: |depth_left_mm| < 0.5 (as check (i); lead resolution r3)
FINGER_MAX_N = 1.0              # finger_contact (information) = peak hand / finger - socket force > 1 N
III_SKIP_MARGIN_M = 0.003       # batch_assemblies.py runs (iii) only when the box's smallest mesh margin is < 3 mm
III_MARGIN_TOL_MM = 0.01        # a (iii) result counts for a box whose smallest mesh margin is within 0.01 mm of its own
STREAMS = {"grasp_values": 0, "trajectory_plan": 1, "placement": 2, "expert": 3, "episode": 4, "reference_label": 5,
           "waypoint": 6}
# random waypoints whose commanded hand / finger is closer than expert.HAND_FINGER_COMMAND_CLEARANCE_M to the socket are
# redrawn (the user's decision 7 of 2026-11-23, deviation row 39; redraw_waypoints): at most this many draws per waypoint
WAYPOINT_DRAWS = 20
WAYPOINT_BOX = StartBox(tip_gap_m=(0.004, 0.018))     # the random-waypoint box of _trajectories (wm.collect.plan)
BUILD_FAILURES_MAX = 3         # counted scene-build failures of one assembly before workers skip it
DEATHS_MAX = 4                 # process deaths (crash or hang) over one assembly's grasps before workers skip it
MEMORY_WARN_FRACTION = 0.85
EVAL_TIP_COLUMNS = tuple(K.TRUTH_COLUMNS)           # the env API: AutoMate's columns first, then information numbers
CODE_FILES = ("src/wm/sim_automate/*.py", "src/wm/sim_automate/*.urdf", "src/wm/data/*.py", "src/wm/env/sampling.py",
              "src/wm/sim/*.py", "scripts/automate/collect_expert.py")
# the files that decide the physics, the robot, the expert and the truth of a milestone-1b result and of a collection
# (not the collector's bookkeeping): physics_code_identity
# (r4-expert-07: also the batch's (ii) / (iii) procedure and pass rules, interpolate_T / pose_distance used by the
# expert's segments, transfers and move_to, and Offset / StartBox used by the expert and the starts)
PHYSICS_CODE_FILES = tuple(f"src/wm/sim_automate/{m}.py" for m in ("scene", "robot", "config", "expert", "checks", "grasp",
                                                                   "assets", "kinematics", "fit", "frames")) + \
    ("src/wm/sim_automate/*.urdf", "scripts/automate/batch_assemblies.py", "src/wm/sim/geometry.py",
     "src/wm/env/sampling.py")
ROOT = Path(__file__).resolve().parents[3]          # world_model/
EXIT_DONE, EXIT_CRASH, EXIT_REFUSED, EXIT_STOPPED, EXIT_BUILD = 0, 1, 3, 5, 6
EXIT_FORBIDDEN_GPU = getattr(_app, "EXIT_FORBIDDEN_GPU", 7)       # app.assert_off_forbidden_gpu: a real GPU-0 violation
EXIT_GPU_UNREADABLE = getattr(_app, "EXIT_GPU_UNREADABLE", 8)     # app.assert_off_forbidden_gpu: nvidia-smi unreadable
STOP_FILE = "STOP"
DISK_FREE_MIN_CLAIM = 30e9      # bytes free in <out> before a claim (one grasp ~ 22 MB; /mnt/storage3 is shared)
DISK_FREE_MIN_SAVE = 2e9        # bytes free in <out> before a trajectory file is written
START_SUCCESS_DRAWS = 2000      # draws per assembly for the start-box fraction (plan.json, information): each draw
                                # is one checks.start_box_success call (~200 us per draw on rs2 under load: 0.42 s
                                # per assembly, 2026-11-23); prepare() builds the plan twice per worker start
                                # (collect_expert.py before Isaac, then worker()) and once per preflight: ~80 s per
                                # start; standard error of the share <= 1.1 points
RESET_OR_REPORT_ALERT_SHARE = 0.2   # ALERT when more of a grasp's trajectories end reset_failed / contact_report_error /
                                    # label_failed (a refused t00 reference counts as all of the grasp's trajectories)
SOFT_OUTCOMES = ("reset_failed", "contact_report_error", "label_failed")
PRODUCTION_GRASPS_PER_ASSEMBLY = 15     # decision 7 of 2026-11-22 (split automate_v1)
USD_COMPARED = ("plug.usd", "socket.usd", "plug_sensor.usd")   # layers compared with the milestone-1b (ii) result
BEAT_EVERY_S = 5.0
# The watcher kills a worker whose heartbeat is older than this: 3 x the slowest measured Isaac start (5.5 min, CPU
# PhysX, rs2 2026-11-22); recorded steps beat every BEAT_EVERY_S, every other stage (start, build, reset, transfer,
# close) beats when it begins.
HANG_TIMEOUT_S = 1020
# Per-worker resources by PhysX device, from the measured peaks (rs2 2026-11-22, memory.peak of the worker's scope):
# CPU PhysX 3.5-11.8 GB (throughput_cpu_n2), GPU PhysX 13-15 GB. MemoryMax = default, refused below the minimum
# (11.8 GB = 84 % of 14 GB; 14.95 GB = 83 % of 18 GB). GPU GB = the declaration for caps.sh (not yet measured with
# graphics contexts: smoke S2); the production config sets worker_gpu_gb (integer 1..48) to the smoke's
# worker_gpu_gb_recommended = ceil(1.3 x G1), also when that is below 7 (caps.sh charges every worker its declaration
# against GPU 1's free memory F, so a smaller measured need lets more workers start), so production needs no code
# change (r7-ops-05, r11-ops-01). Workers that fit: n = min(6, floor((F - 12 GB) / worker_gpu_gb)) (caps.sh rule
# F - P - G_new >= 12 GB); report the time estimate for that n and re-check F before each collect_add_worker.sh.
WORKER_RESOURCES = {"cpu": {"memory_max": "16G", "memory_min_gb": 14},
                    "cuda:0": {"memory_max": "18G", "memory_min_gb": 18}}
WORKER_CPU_QUOTA = "500%"
WORKER_GPU_GB = 7
PRODUCTION_KEYS = ("device", "grasps_per_assembly", "split", "expected_grasps", "m1b_dirs")
NETWORK_OR_GPU_MEMORY = re.compile(r"out of memory|OutOfMemory|ERROR_OUT_OF_DEVICE_MEMORY|cudaErrorMemoryAllocation|"
                                   r"URLError|ConnectionError|timed out|Temporary failure in name resolution|"
                                   r"Network is unreachable|HTTP Error 5\d\d", re.I)


class Refused(SystemExit):
    """A config, plan, input or code error: no restart can help (exit EXIT_REFUSED)."""


class Stopped(BaseException):
    """A requested stop (STOP file or signal); not an Exception, so no per-trajectory try catches it."""


class BuildFailed(Exception):
    """Building the scene of an assembly failed (exit EXIT_BUILD)."""


# ---------------------------------------------------------------------------------------------- split (pluggable)
def split_unassigned(assembly_id: str, values: dict, u: dict) -> str:
    """Placeholder until the user decides the train / selection / held-out split."""
    return "unassigned"


def grasp_rule_heldout(u: dict) -> bool:
    """The agreed grasp rule (task definition 11): held out if 0.55 <= u <= 0.65 for any grasp variable (a zero-width
    variable, u None, never counts). Recorded for information only; the split itself is SPLIT_RULES[cfg.split]."""
    return any(x is not None and G.HELDOUT_U[0] <= x <= G.HELDOUT_U[1] for x in u.values())


# ---- the split "automate_v1" (the user's decision of 2026-11-22: ManiSkill's proportions)
# Assemblies. I = the plan's included (and forced) assemblies, N = |I|; per assembly a: category group g_a
# (CATEGORY_GROUP of fit.py's category), x_a = (d_a, L_a, c_a) = (fit equiv_diameter_m, pin_length_m, clearance_m)
# from the frozen geometry file (assets.SPEC_FILE).
#   near-twins       a ~ b  iff  g_a = g_b AND |d_a - d_b| <= 0.5 mm AND |L_a - L_b| <= 3 mm; a cluster = a connected
#                    component of ~; a cluster is always on one side (all its members chosen, or none)
#   numbers          n_test = 5 (final test, "test": never trained on, never used for selection or held-out reports),
#                    n_heldout = floor(0.10 N + 0.5) (held-out shapes, "heldout_shape")
#   quotas           per role r and group g: q_rg = largest-remainder apportionment of n_r over the groups in
#                    proportion to N_g (ties: group name order)
#   order            clusters K by h_r(K) = crc32("automate_v1/<r>/" + smallest assembly id of K)
#   interpolation    with T = I minus every chosen assembly (both roles): for every chosen a and every j in (d, L, c),
#                    min_{b in T} x_bj < x_aj < max_{b in T} x_bj
#   choice           role test first, then heldout: pass 1, groups in name order, clusters of the group in h_r order:
#                    take K if |K| <= the group's remaining quota and the interpolation rule holds for all chosen
#                    assemblies with K added; pass 2 (only if a quota could not be filled): all remaining clusters in
#                    h_r order, |K| <= the role's remaining number (recorded as "outside the group quota")
# Grasps (G = grasps_per_assembly; the last three are the band grasps of dtheta, dp, dpsi). For grasp i of assembly a
# three uniform numbers r_1..3 are drawn from the grasp_values stream (always all three, in variable order) and per
# variable k with box [lo_k, hi_k]:  u_k = 0.55 + 0.10 r_k  for the band variable of a band grasp, otherwise
# u_k = x if x < 0.55 else 1 - (x - 0.55), x = 0.9 r_k (uniform on [0, 0.55) U (0.65, 1]);  v_k = lo_k + u_k (hi_k - lo_k);
# a zero-width variable has v_k = lo_k, u_k = None, and a band grasp of a zero-width variable becomes a train-range
# grasp (recorded as band_skipped).
#   condition        test assemblies: "test"; held-out assemblies: "heldout_shape"; other assemblies: band grasp
#                    "heldout_grasp", train-range grasp "select" if crc32(grasp name) % 20 == 0 else "train"
#                    (demo320.SELECT_ONE_IN, ManiSkill's rule)
# The lists, the reasons, the quotas and the final training ranges are plan.json "split" (in plan_sha) and
# <out>/split_automate_v1.json.
SPLIT_VERSION = "automate_v1"
# The rule above as formulas, written into plan.json and split_automate_v1.json (r3-decisions-05: the near-twin
# clusters, the category quotas, the crc32 order, the second pass and the interior rule for both roles were explained
# to the user and accepted; lead resolution r3).
SPLIT_RULE_TEXT = [
    "I = included assemblies, N = |I|; g_a = CATEGORY_GROUP[category_a]; x_a = (d_a, L_a, c_a) = (fit equiv_diameter_m, "
    "pin_length_m, clearance_m)",
    "near twins: a ~ b iff g_a = g_b AND |d_a - d_b| <= 0.5 mm AND |L_a - L_b| <= 3 mm; cluster = connected component "
    "of ~; a cluster is chosen whole or not at all",
    "numbers: n_test = 5, n_heldout = floor(0.10 N + 0.5)",
    "quotas: q_rg = largest-remainder apportionment of n_r over the groups in proportion to N_g (ties: group name)",
    "order: clusters K by h_r(K) = crc32('automate_v1/<r>/' + min id of K)",
    "interior: T = I minus all chosen; for every chosen a and j in (d, L, c): min_{b in T} x_bj < x_aj < max_{b in T} x_bj",
    "choice: role test, then heldout; pass 1: groups in name order, clusters in h_r order, take K if |K| <= remaining "
    "group quota and interior holds; pass 2 (quota unfilled): remaining clusters in h_r order, |K| <= remaining n_r "
    "and interior holds ('outside the group quota')",
    "grasps: G per assembly, the last three band grasps of (dtheta, dp, dpsi); r_1..3 ~ U[0,1) from the grasp_values "
    "stream; band variable u = 0.55 + 0.10 r; other u = x if x < 0.55 else 1 - (x - 0.55), x = 0.9 r; "
    "v = lo + u (hi - lo); zero width: v = lo, u = None (band grasp -> train_range, band_skipped)",
    "condition: test -> 'test'; heldout -> 'heldout_shape'; band grasp -> 'heldout_grasp'; else 'select' if "
    "crc32(grasp name) % 20 == 0 else 'train'",
]
CATEGORY_GROUP = {"round": "round", "polygonal": "non-round keyed", "keyed": "non-round keyed", "flat": "flat",
                  "multi-pin": "multi-pin", "sleeve": "sleeve"}
N_TEST = 5
HELDOUT_FRACTION = 0.10
TWIN_D_M, TWIN_L_M = 0.0005, 0.003
SELECT_ONE_IN = 20
GRASP_VARS = ("dtheta_deg", "dp_m", "dpsi_deg")
SPLIT_RULES = {"unassigned": split_unassigned, SPLIT_VERSION: None}     # automate_v1 is applied in build()


def _crc(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode())


def _shape(aid: str) -> dict:
    g = json.loads((A.ASSET_DIR / aid / A.SPEC_FILE).read_text())
    cat = g.get("category")
    if cat not in CATEGORY_GROUP:
        raise Refused(f"{aid}: category {cat!r} has no group in the split rule {SPLIT_VERSION}")
    return {"group": CATEGORY_GROUP[cat], "category": cat, "d_m": float(g["fit"]["equiv_diameter_m"]),
            "L_m": float(g["pin_length_m"]), "c_m": float(g["clearance_m"])}


def _apportion(n: int, sizes: dict) -> dict:
    """Largest-remainder apportionment of n over the keys of sizes (ties: key order)."""
    tot = sum(sizes.values())
    exact = {k: n * v / tot for k, v in sizes.items()}
    q = {k: int(np.floor(e)) for k, e in exact.items()}
    rest = sorted(sizes, key=lambda k: (-(exact[k] - q[k]), k))
    for k in rest[: n - sum(q.values())]:
        q[k] += 1
    return q


def split_document(ids: list[str]) -> dict:
    """The assembly lists of the split automate_v1 (rule above) for the included assemblies `ids`."""
    ids = sorted(ids)
    sh = {a: _shape(a) for a in ids}
    N = len(ids)
    # near-twin clusters (union-find)
    parent = {a: a for a in ids}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if sh[a]["group"] == sh[b]["group"] and abs(sh[a]["d_m"] - sh[b]["d_m"]) <= TWIN_D_M + 1e-12 \
                    and abs(sh[a]["L_m"] - sh[b]["L_m"]) <= TWIN_L_M + 1e-12:
                parent[root(b)] = root(a)
    clusters = {}
    for a in ids:
        clusters.setdefault(root(a), []).append(a)
    clusters = sorted((sorted(m) for m in clusters.values()), key=lambda m: m[0])
    sizes = {}
    for a in ids:
        sizes[sh[a]["group"]] = sizes.get(sh[a]["group"], 0) + 1
    n_role = {"test": N_TEST, "heldout": int(np.floor(HELDOUT_FRACTION * N + 0.5))}
    if sum(n_role.values()) >= N:
        raise Refused(f"split {SPLIT_VERSION}: {N} included assemblies are too few for {n_role}")
    chosen, why = {}, {}

    def interior(extra: list) -> bool:
        sel = set(chosen) | set(extra)
        T = [b for b in ids if b not in sel]
        if not T:
            return False
        for j in ("d_m", "L_m", "c_m"):
            lo, hi = min(sh[b][j] for b in T), max(sh[b][j] for b in T)
            if not all(lo < sh[a][j] < hi for a in sel):
                return False
        return True

    quotas = {}
    for role in ("test", "heldout"):
        q = _apportion(n_role[role], sizes)
        quotas[role] = q
        order = sorted((m for m in clusters), key=lambda m: _crc(f"{SPLIT_VERSION}/{role}/{m[0]}"))
        left = dict(q)
        for g in sorted(sizes):
            for m in order:
                if sh[m[0]]["group"] != g or m[0] in chosen or len(m) > left[g]:
                    continue
                if interior(m):
                    for a in m:
                        chosen[a] = role
                        why[a] = (f"{role}: group {g} (quota {q[g]}), cluster {m}, crc order "
                                  f"{_crc(f'{SPLIT_VERSION}/{role}/{m[0]}')}, inside the training ranges")
                    left[g] -= len(m)
        need = n_role[role] - sum(r == role for r in chosen.values())
        for m in order:
            if need <= 0:
                break
            if m[0] in chosen or len(m) > need or not interior(m):
                continue
            for a in m:
                chosen[a] = role
                why[a] = f"{role}: outside the group quota (group {sh[a]['group']} could not fill it), cluster {m}"
            need -= len(m)
    T = [b for b in ids if b not in chosen]
    ranges = {j: [min(sh[b][j] for b in T), max(sh[b][j] for b in T)] for j in ("d_m", "L_m", "c_m")}
    assert interior([]), "split rule: a chosen assembly lies outside the training ranges"
    return {"rule": SPLIT_VERSION, "version": 1, "n_included": N, "rule_text": SPLIT_RULE_TEXT,
            "numbers": {r: {"wanted": n, "chosen": sum(v == r for v in chosen.values())} for r, n in n_role.items()},
            "quotas": quotas, "group_sizes": sizes, "test": sorted(a for a, r in chosen.items() if r == "test"),
            "heldout_shape": sorted(a for a, r in chosen.items() if r == "heldout"), "reasons": why,
            "training_ranges": ranges, "clusters": [m for m in clusters if len(m) > 1], "shapes": sh,
            "grasps": {"band_u": list(G.HELDOUT_U), "band_grasps": "the last three: " + ", ".join(GRASP_VARS),
                       "select_one_in": SELECT_ONE_IN}}


def grasp_values_v1(cfg, aid: str, gi: int, box: dict) -> tuple[dict, dict, str]:
    """Values, u and kind of grasp gi of the split automate_v1 (rule above)."""
    r = np.random.default_rng(stream(cfg, aid, gi, 0, "grasp_values")).random(3)
    band_i = gi - (cfg.grasps_per_assembly - 3)
    band = GRASP_VARS[band_i] if band_i >= 0 else None
    kind = f"band_{band}" if band else "train_range"
    v, u = {}, {}
    for k, rk in zip(GRASP_VARS, r):
        lo, hi = box[k]
        if not hi > lo:
            v[k], u[k] = float(lo), None
            if band == k:
                kind = f"train_range (band_skipped: {k} has zero width)"
            continue
        if band == k:
            uk = G.HELDOUT_U[0] + (G.HELDOUT_U[1] - G.HELDOUT_U[0]) * float(rk)
        else:
            x = 0.9 * float(rk)                       # [0, 0.9) -> [0, 0.55) U (0.65, 1], uniform
            uk = x if x < G.HELDOUT_U[0] else 1.0 - (x - G.HELDOUT_U[0])
        u[k], v[k] = uk, float(lo + uk * (hi - lo))
    return v, u, kind


def condition_v1(role: str | None, key: str, kind: str) -> str:
    if role == "test":
        return "test"
    if role == "heldout":
        return "heldout_shape"
    if kind.startswith("band_"):
        return "heldout_grasp"
    return "select" if _crc(key) % SELECT_ONE_IN == 0 else "train"


# ---------------------------------------------------------------------------------------------- plan
@dataclass
class TrajectorySpec:
    idx: int
    kind: str
    seed_offset: int
    start: Offset
    offset: Offset | None = None
    waypoints: list[Offset] | None = None
    speeds: list[float] | None = None


@dataclass
class GraspSpec:
    key: str
    port: str
    condition: str
    pitch: float                # d_theta [deg]
    grasp_pos: float            # d_p [m]
    spin: float                 # d_psi [deg]
    u: dict
    seed: int                   # episode (reset) seed
    gi: int = 0
    trajectories: list[TrajectorySpec] = field(default_factory=list)
    grasp_kind: str = "uniform"  # automate_v1: "train_range" or "band_<variable>" (the split rule)


@dataclass
class CollectConfig:
    name: str = "automate_v1"
    seed: int = 5_000_000
    assemblies: list = field(default_factory=list)       # empty = every assembly with a usable grasp range
    assemblies_force: list = field(default_factory=list)  # collected although the gate fails (needs a verified box)
    grasps_per_assembly: int = 2
    trajectories: dict = field(default_factory=lambda: {"insert": 4, "perturbed": 2, "contact": 5, "random": 1})
    random_waypoints: int = 3
    # random-motion speed per waypoint, drawn uniformly in this range: (0.0027, 0.0027) = wm_v1's constant 2.7 mm/s
    # (0.9 x close_speed 3 mm/s, wm/env/expert.py:95), restored by the user's decision of 2026-11-23 (deviation table
    # row 27; before: 3-15 mm/s). The draw is kept, so the trajectory-plan stream's order of draws is unchanged.
    # expert.RANDOM_SPEED_M_S is the same constant; the expert refuses any other speed.
    random_speed_m_s: tuple = (0.0027, 0.0027)
    images: bool = True
    image_every: int = 4
    rgb_codec: str = "jpeg"
    jpeg_quality: int = 95
    split: str = "unassigned"
    transfer: str = "move"
    # PhysX device of the scene: the user's decision 2026-11-22 is CPU PhysX for collection (GPU PhysX an option;
    # rendering stays on GPU 1)
    device: str = field(default_factory=lambda: C.PHYSICS_DEVICE)
    m1b_dirs: list = field(default_factory=lambda: list(M1B_DIRS))
    collect_subset: list = field(default_factory=list)   # smoke runs only: plan only these assemblies' grasps
    expected_grasps: int | None = None      # the plan must have exactly this many grasps (production configs)
    worker_memory_max: str | None = None    # MemoryMax per worker ("16G"); None = WORKER_RESOURCES[device]
    worker_cpu_quota: str | None = None     # CPUQuota per worker ("500%"); None = WORKER_CPU_QUOTA
    worker_gpu_gb: int | None = None        # GPU-1 GB declared per worker (integer 1..48); None = WORKER_GPU_GB

    @classmethod
    def load(cls, path, production: bool = False) -> "CollectConfig":
        """Read a config json (a config.json written by freeze, with its config_sha, also loads; the sha is checked).
        Unknown keys refuse. production: every key of PRODUCTION_KEYS must be in the file and not null, and the values
        must be valid (module docstring). Raises Refused."""
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise Refused(f"config {path}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw, dict):
            raise Refused(f"config {path}: not a json object")
        sha = raw.pop("config_sha", None)
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise Refused(f"config {path}: unknown keys {unknown}")
        cfg = cls(**raw)
        if sha is not None and sha != sha256_text(_canon(cfg.dump())):
            raise Refused(f"config {path}: config_sha does not match its content")
        if production:
            missing = [k for k in PRODUCTION_KEYS if raw.get(k) is None]
            if missing:
                raise Refused(f"config {path}: the production keys {missing} are not set; they are the user's decision "
                              f"and have no default ({', '.join(PRODUCTION_KEYS)})")
            cfg.validate()
        return cfg

    def validate(self):
        """The values of a production config (Refused if invalid)."""
        bad = []
        if self.device not in WORKER_RESOURCES:
            bad.append(f"device {self.device!r} (only {sorted(WORKER_RESOURCES)}; 'cuda:0' = GPU 1 under "
                       "CUDA_VISIBLE_DEVICES=1)")
        if not (isinstance(self.grasps_per_assembly, int) and self.grasps_per_assembly > 0):
            bad.append(f"grasps_per_assembly {self.grasps_per_assembly!r}")
        if self.split != SPLIT_VERSION:
            bad.append(f"split {self.split!r} (production: {SPLIT_VERSION!r}, decided 2026-11-22)")
        elif not self.collect_subset and self.grasps_per_assembly != PRODUCTION_GRASPS_PER_ASSEMBLY:
            bad.append(f"grasps_per_assembly {self.grasps_per_assembly!r} (decision 7 of 2026-11-22: "
                       f"{PRODUCTION_GRASPS_PER_ASSEMBLY} grasps per assembly)")
        elif not (isinstance(self.grasps_per_assembly, int) and self.grasps_per_assembly >= 4):
            bad.append(f"grasps_per_assembly {self.grasps_per_assembly!r} (split {SPLIT_VERSION} needs >= 4)")
        bad += pinned_differences(self)
        sw = C.unapproved_switches()
        if sw:
            bad.append("switches not approved (config.APPROVED; the launching shell exported them?): " +
                       ", ".join(f"{k} = {v[0]!r}, approved {v[1]!r}" for k, v in sorted(sw.items())))
        if not (isinstance(self.expected_grasps, int) and self.expected_grasps > 0):
            bad.append(f"expected_grasps {self.expected_grasps!r}")
        if not (isinstance(self.m1b_dirs, list) and self.m1b_dirs and all(isinstance(d, str) for d in self.m1b_dirs)):
            bad.append(f"m1b_dirs {self.m1b_dirs!r} (a non-empty list of folders)")
        if not (isinstance(self.collect_subset, list) and all(isinstance(a, str) for a in self.collect_subset)):
            bad.append(f"collect_subset {self.collect_subset!r}")
        if bad:
            raise Refused("config: invalid " + "; ".join(bad))
        worker_resources(self)

    def dump(self) -> dict:
        return asdict(self)


def worker_resources(cfg: CollectConfig) -> dict:
    """MemoryMax / CPUQuota / GPU GB of one worker (WORKER_RESOURCES, WORKER_CPU_QUOTA and WORKER_GPU_GB unless the
    config names them); Refused if MemoryMax is not <integer>G or lies below the device's measured minimum, the
    CPUQuota is not <integer>%, or the GPU GB is not an integer 1..48 (caps.sh: 48 GB of GPU 1 for all our jobs)."""
    dev = WORKER_RESOURCES.get(cfg.device)
    if dev is None:
        raise Refused(f"device {cfg.device!r}: no worker resources known")
    mem = cfg.worker_memory_max or dev["memory_max"]
    cpu = cfg.worker_cpu_quota or WORKER_CPU_QUOTA
    if not re.fullmatch(r"[0-9]+G", str(mem)) or int(mem[:-1]) < dev["memory_min_gb"]:
        raise Refused(f"worker_memory_max {mem!r}: device {cfg.device} needs <integer>G >= {dev['memory_min_gb']}G "
                      "(measured peaks with headroom)")
    if not re.fullmatch(r"[0-9]+%", str(cpu)) or int(cpu[:-1]) <= 0:
        raise Refused(f"worker_cpu_quota {cpu!r}: needs <integer>%")
    gpu = WORKER_GPU_GB if cfg.worker_gpu_gb is None else cfg.worker_gpu_gb
    if not isinstance(gpu, int) or isinstance(gpu, bool) or not 1 <= gpu <= 48:
        raise Refused(f"worker_gpu_gb {gpu!r}: needs an integer 1..48")
    return {"memory_max": mem, "cpu_quota": cpu, "gpu_gb": gpu, "hang_timeout_s": HANG_TIMEOUT_S}


def pinned_values() -> dict:
    """The trajectory mix and picture settings a production config must keep (module docstring): wm_v1's values from
    wm.collect.plan.CollectConfig (not copied here), and this module's random_speed_m_s default (wm_v1's constant
    2.7 mm/s, decision of 2026-11-23)."""
    from ..collect.plan import CollectConfig as V1
    v1, ours = V1(), CollectConfig.__dataclass_fields__
    out = {k: getattr(v1, k) for k in ("trajectories", "random_waypoints", "images", "image_every", "rgb_codec",
                                       "jpeg_quality")}
    out["random_speed_m_s"] = ours["random_speed_m_s"].default
    return out


def pinned_differences(cfg: CollectConfig) -> list[str]:
    """Refusal texts for every pinned value cfg differs from (the trajectory kinds also in their order: the order sets
    the file indices)."""
    bad = []
    for k, want in pinned_values().items():
        got = getattr(cfg, k)
        if k == "trajectories":
            same = isinstance(got, dict) and list(got.items()) == list(want.items())
        elif k == "random_speed_m_s":
            try:
                same = [float(x) for x in got] == [float(x) for x in want]
            except (TypeError, ValueError):
                same = False
        else:
            same = type(got) is type(want) and got == want
        if not same:
            src = ("this module's default: wm_v1's constant 2.7 mm/s, wm/env/expert.py:95, decision of 2026-11-23"
                   if k == "random_speed_m_s" else "wm_v1 (wm.collect.plan.CollectConfig)")
            bad.append(f"{k} {got!r} (agreed: {want!r}, as {src}; 13 trajectories per grasp as in wm_v1)")
    return bad


def m1b_dirs(cfg: CollectConfig) -> list[str]:
    """The milestone-1b result folders actually read: cfg.m1b_dirs, or their frozen copies <WM_AUTOMATE_M1B>/<k> (k =
    position in cfg.m1b_dirs) when the launcher set WM_AUTOMATE_M1B."""
    root = os.environ.get("WM_AUTOMATE_M1B")
    if not root:
        return list(cfg.m1b_dirs)
    out = [str(Path(root) / str(k)) for k in range(len(cfg.m1b_dirs))]
    missing = [d for d in out if not Path(d).is_dir()]
    if missing:
        raise Refused(f"WM_AUTOMATE_M1B={root}: frozen copies {missing} missing")
    return out


def stream(cfg: CollectConfig, aid: str, gi: int, idx: int, purpose: str) -> np.random.SeedSequence:
    """The SeedSequence of one random stream (module docstring)."""
    return np.random.SeedSequence(entropy=int(cfg.seed), spawn_key=(int(aid), int(gi), int(idx), STREAMS[purpose]))


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else float(o))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(root: Path = ROOT) -> dict:
    """sha256 over the collector's source files (CODE_FILES, sorted paths and contents) and, if root is in a git
    checkout, the commit and whether those files differ from it. Workers compare files_sha256."""
    h = hashlib.sha256()
    files = sorted({p for pat in CODE_FILES for p in root.glob(pat)})
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    out = {"files_sha256": h.hexdigest(), "n_files": len(files), "git_head": None, "git_dirty": None}
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            st = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *[str(p) for p in files]],
                                capture_output=True, text=True, timeout=10)
            out.update(git_head=head.stdout.strip(), git_dirty=bool(st.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return out


@functools.lru_cache(maxsize=4)
def physics_code_identity(root: Path = ROOT) -> str:
    """sha256 over PHYSICS_CODE_FILES (sorted relative paths and contents): batch_assemblies.py records it as
    physics_code_sha256 and the gate compares it; a collector-only fix does not alter it (r3-check-N3)."""
    h = hashlib.sha256()
    for p in sorted({p for pat in PHYSICS_CODE_FILES for p in root.glob(pat)}):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _unix_time(res: dict, f: Path) -> float:
    t = res.get("unix_time")
    return float(t) if isinstance(t, (int, float)) else -1.0        # results without a time lose against any timed one


def m1b_results(assembly_id: str, dirs: list) -> dict:
    """For each check x in ("ii", "iii"): the newest result (by unix_time) among <dir>/<id>.json of `dirs` that holds x
    and no "error" -> {x: (result, path) or (None, None)}. A process-crash file ({"error": ...}) never counts, for
    either check."""
    found = {"ii": [], "iii": []}
    for d in dirs:
        f = Path(d) / f"{assembly_id}.json"
        if not f.exists():
            continue
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if not isinstance(r, dict) or "error" in r:
            continue
        for x in found:
            if isinstance(r.get(x), dict):
                found[x].append((_unix_time(r, f), str(f), r))
    out = {}
    for x, rows in found.items():
        if rows:
            t, path, r = max(rows, key=lambda row: (row[0], row[1]))
            out[x] = (r, path)
        else:
            out[x] = (None, None)
    return out


def m1b_result(assembly_id: str, dirs: list) -> tuple[dict | None, str | None]:
    """The newest error-free milestone-1b result holding (ii) (m1b_results) -> (result, path); (None, None) if none."""
    return m1b_results(assembly_id, dirs)["ii"]


# the per-assembly asset files whose sha256 batch_assemblies.py records (asset_hashes) and the gate compares
# (r5-integration-03 / r5-expert-03): None = the geometry file assets.SPEC_FILE
ASSET_HASH_FILES = {"spec_sha256": None, "plug_obj_sha256": "plug.obj", "socket_obj_sha256": "socket.obj"}


def asset_identity(assembly_id: str) -> dict:
    """sha256 of the assembly's geometry file (assets.SPEC_FILE: L, d_dis, tip_centre, symmetry_n, L_top) and plug /
    socket meshes in this process's A.ASSET_DIR (under the launchers the frozen <out>/inputs/assets), None for a
    missing file: {"spec_sha256", "plug_obj_sha256", "socket_obj_sha256", "asset_sha256": {file name: sha256}}, the
    fields batch_assemblies.asset_hashes records."""
    out, by_name = {}, {}
    for key, name in ASSET_HASH_FILES.items():
        f = A.ASSET_DIR / assembly_id / (name or A.SPEC_FILE)
        out[key] = by_name[f.name] = sha256_file(f) if f.exists() else None
    out["asset_sha256"] = by_name
    return out


# the shared asset files (not per assembly) that decide a milestone-1b result (r9-integration-01): plug_grasps.json
# (AutoMate's grasp T_hp0) and the finger collision meshes franka_mimic_collision/*.obj (grasp.FingerSocketCheck: the
# finger opening at every reset and the start clearance). disassembly_dist.json is not needed: the d_dis the results
# use is cached in the geometry file, which spec_sha256 covers.
SHARED_ASSET_FILES = ("plug_grasps.json", "franka_mimic_collision/panda_hand.obj",
                      "franka_mimic_collision/panda_leftfinger.obj", "franka_mimic_collision/panda_rightfinger.obj")


def shared_asset_identity() -> dict:
    """{path relative to A.ASSET_DIR: sha256, or None for a missing file} of SHARED_ASSET_FILES and of every other
    franka_mimic_collision/*.obj in this process's A.ASSET_DIR. batch_assemblies.py records it as shared_asset_sha256
    and stale_fields compares it (both sides must be equal)."""
    names = set(SHARED_ASSET_FILES) | {f"franka_mimic_collision/{f.name}"
                                       for f in (A.ASSET_DIR / "franka_mimic_collision").glob("*.obj")}
    return {n: (sha256_file(A.ASSET_DIR / n) if (A.ASSET_DIR / n).is_file() else None) for n in sorted(names)}


def stale_fields(res: dict, assembly_id: str, cfg: "CollectConfig") -> list[str]:
    """Provenance fields of a milestone-1b result that differ from this process's (module docstring, current(x)); a
    missing or None value is stale."""
    rg_f = G.ranges_file(assembly_id)
    want = {"ranges_sha256": sha256_file(rg_f) if rg_f.exists() else None,
            "grasp_py_sha256": sha256_file(G.__file__), "physics_device": cfg.device, "switches": C.switches(),
            **asset_identity(assembly_id), "shared_asset_sha256": shared_asset_identity()}
    bad = [k for k, v in want.items() if res.get(k) != v or v is None]
    if res.get("physics_code_sha256") != physics_code_identity():          # a result without it is stale too
        bad.append("physics_code_sha256")
    return bad


def _seat_ok(x: dict, first: str) -> bool:
    """|depth_left_mm| < SEAT_DEPTH_MM and completed and `first` (placed / move_ok) of a (ii) / (iii) result, each
    recorded and True (lead resolution r3; r4-decisions-01: a missing key fails)."""
    try:
        d = float(x.get("depth_left_mm"))
    except (TypeError, ValueError):
        return False
    return x.get(first) is True and x.get("completed") is True and abs(d) < SEAT_DEPTH_MM


def pass_ii(ii: dict) -> bool:
    """pass_ii = placed AND completed AND |depth_left_mm| < 0.5 AND s_end (batch_assemblies "pass" AND the recorded
    numbers)."""
    return ii.get("pass") is True and _seat_ok(ii, "placed") and bool(ii.get("success"))


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def pass_iii(iii: dict) -> bool:
    """pass_iii = pass AND move_ok AND completed AND |depth_left_mm| < 0.5 AND F_hf <= III_MAX_FINGER_N, F_hf =
    hand_finger_max_N (max of the sensor and the robot-socket pair force, as batch_assemblies.py); a missing key or a
    non-number force fails (r4-integration-07 / r4-decisions-01)."""
    f = _float(iii.get("hand_finger_max_N"))
    if f is None:
        return False
    return iii.get("pass") is True and _seat_ok(iii, "move_ok") and f <= III_MAX_FINGER_N


def ii_reason(ii: dict) -> str:
    """The exclusion reason of a failed (ii), with the end height and keypoint distance of AutoMate's rule and, when
    the rule fails only by the strict h > 0 (|h_end| < 0.05 mm, the plug seated a few micrometres below h = 0), that
    case named (r4-expert-02 a)."""
    h, k = _float(ii.get("height_mm")), _float(ii.get("keypoint_dist_mm"))
    txt = (f"(ii) aligned insertion did not pass (pass={ii.get('pass')}, placed={ii.get('placed')}, "
           f"completed={ii.get('completed')}, success={ii.get('success')}, depth_left_mm={ii.get('depth_left_mm')}, "
           f"height_mm={h}, keypoint_dist_mm={k})")
    others_ok = ii.get("pass") is not True and _seat_ok(ii, "placed")
    if h is not None and abs(h) < 0.05 and h <= 0 and others_ok and (k is None or k < K.CLOSE_ERROR_THRESH_M * 1e3):
        txt += f"; fails only by h > 0: h_end = {h * 1e3:.1f} um"
    return txt


def start_box_success(assembly_id: str, box: dict | None) -> dict | None:
    """The start-box property of AutoMate's rule (module docstring; information): None without a box or if the spec
    or AutoMate's grasp of the assembly cannot be read. Each draw: a grasp uniform in the box, a start of
    sample_start (tip gap, lateral, rotation), a socket yaw of AutoMate's draw (config.SOCKET_YAW_DEG +
    SOCKET_YAW_RANGE_DEG U[0, 1]: it turns the robot base x axis in the socket frame, which sets the start frame's
    lateral axes), then checks.start_box_success with gap_from = config.START_GAP_FROM (as
    scripts/automate/diag/start_success.py)."""
    from .frames import start_back
    if box is None:
        return None
    try:
        spec = A.load_spec(assembly_id)
        L, d, g_top = float(spec.pin_length_m), float(spec.disassembly_dist_m), K.g_top_of(spec)
        T0, axis_xy = G.T_hp0(assembly_id), G.spin_axis_xy(spec)
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None
    gap_from, sb = C.START_GAP_FROM, StartBox()
    lim = min(d, K.CLOSE_ERROR_THRESH_M)
    h_min = start_back(L, sb.tip_gap_m[0], g_top, gap_from)
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(START_SUCCESS_DRAWS):
        v = [float(rng.uniform(*box[k])) for k in GRASP_VARS]
        T_tcp_plug = G.T_hp(T0, *v, axis_xy=axis_xy)
        off = sample_start(rng, sb)
        yaw = np.radians(C.SOCKET_YAW_DEG + C.SOCKET_YAW_RANGE_DEG * rng.uniform())
        base_x = (np.cos(yaw), -np.sin(yaw), 0.0)                       # R_z(yaw)^T e_x
        n_ok += K.start_box_success(spec, T_tcp_plug, off, base_x, gap_from=gap_from)["success"]
    return {"gap_from": gap_from, "g_top_mm": round(g_top * 1e3, 3), "h_min_mm": round(h_min * 1e3, 3),
            "d_dis_mm": round(d * 1e3, 3), "limit_mm": round(lim * 1e3, 3), "possible": bool(h_min < lim),
            "draws": START_SUCCESS_DRAWS, "fraction": round(n_ok / START_SUCCESS_DRAWS, 4)}


def assembly_status(assembly_id: str, cfg: CollectConfig) -> dict:
    """Status row of one assembly (module docstring): status, reason, box, inputs (path -> sha256), zero_width, the
    milestone-1b provenance and depth_left_mm of (ii) / (iii), start_box."""
    row = {"assembly_id": assembly_id, "status": "excluded", "reason": None, "box": None, "inputs": {},
           "zero_width": [], "ii_pass": None, "iii": None, "m1b": {}, "start_box": None}
    spec_f, rg_f = A.ASSET_DIR / assembly_id / A.SPEC_FILE, G.ranges_file(assembly_id)
    for f in (spec_f, rg_f, A.ASSET_DIR / assembly_id / "plug.obj", A.ASSET_DIR / assembly_id / "socket.obj"):
        if f.exists():
            row["inputs"][str(f)] = sha256_file(f)
    if not rg_f.exists():
        row["reason"] = f"no {G.RANGES_FILE}"
        return row
    rg = G.load_ranges(assembly_id)
    if rg.get("box") is None:
        row["reason"] = "no box: " + str(rg.get("reason", ""))[:120]
        return row
    box = {k: [float(v[0]), float(v[1])] for k, v in rg["box"].items()}
    s = G.spin_limit_deg(rg.get("n"))
    base = {"dtheta_deg": G.BASE["dtheta_deg"], "dp_m": G.BASE["dp_m"], "dpsi_deg": (-s, s)}
    for k, (lo, hi) in box.items():
        assert base[k][0] - 1e-9 <= lo <= hi <= base[k][1] + 1e-9, (assembly_id, k, box[k], base[k])
    row["box"] = box
    row["zero_width"] = [k for k, (lo, hi) in box.items() if not hi > lo]
    row["start_box"] = start_box_success(assembly_id, box)
    if rg.get("verified") is not True:
        row["reason"] = "box not verified on the 9 x 9 x 9 grid"
        return row
    found = m1b_results(assembly_id, m1b_dirs(cfg))
    (res_ii, path_ii), (res_iii, path_iii) = found["ii"], found["iii"]
    if res_ii is None:
        row["reason"] = "no milestone-1b (ii) result without an error (re-run the batch)"
        return _forced(row, cfg)
    for x, (res, path) in found.items():
        if res is None:
            continue
        row["inputs"][path] = sha256_file(path)
        row["m1b"][x] = {"file": path, "unix_time": res.get("unix_time"), "ranges_sha256": res.get("ranges_sha256"),
                         "grasp_py_sha256": res.get("grasp_py_sha256"), "physics_device": res.get("physics_device"),
                         "code_files_sha256": res.get("code_files_sha256"),
                         "physics_code_sha256": res.get("physics_code_sha256"),
                         "depth_left_mm": (res.get(x) or {}).get("depth_left_mm")}
        stale = stale_fields(res, assembly_id, cfg)
        if stale:                                     # never forced: the evidence is from other code / inputs
            row["reason"] = f"re-run: stale {', '.join(stale)} ({x} result {path})"
            return row
    ii, iii = res_ii["ii"], (res_iii or {}).get("iii")
    row["ii_pass"] = pass_ii(ii)
    row["iii"] = iii
    wp, wm_ = rg.get("worst_point"), rg.get("worst_margin_m")
    if not row["ii_pass"]:
        row["reason"] = ii_reason(ii)
    elif iii is None:
        row["reason"] = "no (iii) result"
    elif "skipped" in iii:
        if wp is not None and wm_ is not None and wm_ < III_SKIP_MARGIN_M:
            row["reason"] = (f"(iii) was skipped for an earlier box, the current box's smallest mesh margin is "
                             f"{wm_ * 1e3:.2f} mm < 3 mm: re-run (iii) in simulation")
    elif not _iii_current(iii, box, wm_):
        row["reason"] = (f"(iii) was run at {iii.get('grasp')} (mesh margin {iii.get('mesh_margin_mm')} mm), the current box's "
                         f"smallest mesh margin is {None if wm_ is None else round(wm_ * 1e3, 4)} mm at {wp}: re-run (iii)")
    elif not pass_iii(iii):
        row["reason"] = (f"(iii) failed: pass={iii.get('pass')}, move_ok={iii.get('move_ok')}, "
                         f"completed={iii.get('completed')}, depth_left_mm={iii.get('depth_left_mm')}, "
                         f"F_hf = hand_finger_max_N {iii.get('hand_finger_max_N')} N (sensor hand_finger_socket_N "
                         f"{iii.get('hand_finger_socket_N')} N, robot_socket_pairs_N {iii.get('robot_socket_pairs_N')} N; "
                         f"limit {III_MAX_FINGER_N} N)")
    if row["reason"] is None:
        row["status"] = "included"
        return row
    return _forced(row, cfg)


def m1b_code(table: list[dict]) -> dict:
    """The code identity of the milestone-1b results the plan uses: all included / forced rows must share one
    physics_code_sha256 (or, for results without it, one code_files_sha256); Refused otherwise."""
    keys = set()
    for r in table:
        if r["status"] == "excluded":
            continue
        for x, m in (r.get("m1b") or {}).items():
            keys.add(("physics_code_sha256", m["physics_code_sha256"]) if m.get("physics_code_sha256")
                     else ("code_files_sha256", m.get("code_files_sha256")))
    if len(keys) > 1:
        raise Refused(f"the milestone-1b results of the included assemblies come from different code {sorted(map(str, keys))}: "
                      "re-run the batch from one frozen code tree")
    k = next(iter(keys), (None, None))
    return {"field": k[0], "sha256": k[1]}


def _iii_current(iii: dict, box: dict, worst_margin_m) -> bool:
    """Does a (iii) result test the current box? Its grasp point lies in the box and its mesh margin equals the box's
    smallest mesh margin within III_MARGIN_TOL_MM (several grid points can share the smallest margin; the point
    itself may then differ)."""
    g = iii.get("grasp")
    if g is None or worst_margin_m is None or iii.get("mesh_margin_mm") is None:
        return False
    inside = all(box[k][0] - 1e-9 <= float(v) <= box[k][1] + 1e-9 for k, v in zip(("dtheta_deg", "dp_m", "dpsi_deg"), g))
    return inside and abs(float(iii["mesh_margin_mm"]) - worst_margin_m * 1e3) <= III_MARGIN_TOL_MM


def _forced(row: dict, cfg: CollectConfig) -> dict:
    if row["assembly_id"] in cfg.assemblies_force:
        row["status"] = "forced"
    return row


def redraw_waypoints(waypoints: list[Offset], clearance, rng: np.random.Generator, limit: float,
                     draws: int = WAYPOINT_DRAWS) -> tuple[list[Offset], list[dict]]:
    """Decision 7 of 2026-11-23 (deviation row 39): every waypoint whose clearance(offset) [m] (expert.Expert.
    waypoint_clearance: hand / finger - socket mesh distance of the commanded pose) is below `limit` is replaced by a new
    draw from WAYPOINT_BOX (rng: the trajectory's "waypoint" stream), at most `draws` draws per waypoint (the planned one
    included). A waypoint still below the limit after that is kept: the expert then fails the trajectory at it
    ("hand_finger_command_clearance", expert_failed). -> (waypoints, one record per rejected draw)."""
    out, rejected = [], []
    for i, wp in enumerate(waypoints):
        for k in range(draws):
            d = float(clearance(wp))
            if d >= limit:
                break
            rejected.append({"waypoint": i, "draw": k, "clearance_mm": round(d * 1e3, 3), "offset": wp.as_dict()})
            if k + 1 < draws:
                wp = sample_start(rng, WAYPOINT_BOX)
        out.append(wp)
    return out, rejected


def _trajectories(rng: np.random.Generator, cfg: CollectConfig) -> list[TrajectorySpec]:
    """As wm.collect.plan._trajectories (same samplers, same order of draws), plus random-motion speeds."""
    out = []
    for kind, n in cfg.trajectories.items():
        for _ in range(n):
            spec = TrajectorySpec(idx=len(out), kind=kind, seed_offset=int(rng.integers(1, 2**31)),
                                  start=sample_start(rng))
            if kind == "contact":
                spec.offset = sample_contact_offset(rng)
            if kind == "random":
                spec.waypoints = [sample_start(rng, WAYPOINT_BOX) for _ in range(cfg.random_waypoints)]
                spec.speeds = [float(v) for v in rng.uniform(*cfg.random_speed_m_s, cfg.random_waypoints)]
            out.append(spec)
    return out


def build(cfg: CollectConfig) -> tuple[list[GraspSpec], list[dict]]:
    """Every grasp of the plan (deterministic) and the status table of the assemblies (with the split automate_v1,
    each row also has its split role; split_document gives the lists)."""
    ids = list(cfg.assemblies) or A.assembly_ids()
    specs, table = [], []
    for aid in ids:
        table.append(assembly_status(aid, cfg))
    for row in table:
        if row["status"] != "excluded" and row["box"] is None:
            raise ValueError(f"{row['assembly_id']} is in assemblies_force but has no verified box: forcing needs a "
                             "box rule, which is the user's decision")
    m1b_code(table)                                     # one code identity behind every included result (Refused)
    subset = set(cfg.collect_subset)
    if subset:
        bad = sorted(subset - {r["assembly_id"] for r in table if r["status"] != "excluded"})
        if bad:
            raise Refused(f"collect_subset {bad}: not included by the gate (status table above)")
    role = {}
    if cfg.split == SPLIT_VERSION:
        if cfg.grasps_per_assembly < 4:
            raise Refused(f"split {SPLIT_VERSION} needs grasps_per_assembly >= 4 (3 band grasps), got {cfg.grasps_per_assembly}")
        sd = split_document([r["assembly_id"] for r in table if r["status"] != "excluded"])
        role = {**{a: "test" for a in sd["test"]}, **{a: "heldout" for a in sd["heldout_shape"]}}
    for row in table:
        aid = row["assembly_id"]
        if row["status"] == "excluded":
            continue
        box = row["box"]
        row["split_role"] = role.get(aid, "train") if cfg.split == SPLIT_VERSION else None
        if subset and aid not in subset:                # smoke runs: the split as in production, fewer grasps
            continue
        for gi in range(cfg.grasps_per_assembly):
            key = f"{aid}_g{gi:02d}"
            if cfg.split == SPLIT_VERSION:
                v, u, kind = grasp_values_v1(cfg, aid, gi, box)
                cond = condition_v1(role.get(aid), key, kind)
            else:
                rv = np.random.default_rng(stream(cfg, aid, gi, 0, "grasp_values"))
                v = {k: float(rv.uniform(*box[k])) for k in GRASP_VARS}
                u = {k: G.normalise(v[k], *box[k]) for k in v}
                kind, cond = "uniform", SPLIT_RULES[cfg.split](aid, v, u)
            seed = int(stream(cfg, aid, gi, 0, "episode").generate_state(1)[0])
            rt = np.random.default_rng(stream(cfg, aid, gi, 0, "trajectory_plan"))
            specs.append(GraspSpec(key=key, port=aid, condition=cond, pitch=v["dtheta_deg"], grasp_pos=v["dp_m"],
                                   spin=v["dpsi_deg"], u=u, seed=seed, gi=gi, trajectories=_trajectories(rt, cfg),
                                   grasp_kind=kind))
    return specs, table


def min_select_grasps() -> int:
    """demo320.MIN_SELECT_GRASPS: the fewest selection grasps train_wm320.py accepts for AutoMate data."""
    from ..tdmpc.demo320 import MIN_SELECT_GRASPS
    return int(MIN_SELECT_GRASPS)


def split_counts(cfg: CollectConfig, table: list[dict]) -> dict | None:
    """Split automate_v1 (module docstring, selection grasps): grasp counts per condition over EVERY included assembly
    and every grasp index of cfg.grasps_per_assembly, whatever collect_subset is; None for another split."""
    if cfg.split != SPLIT_VERSION:
        return None
    rows = {r["assembly_id"]: r for r in table if r["status"] != "excluded"}
    sd = split_document(list(rows))
    role = {**{a: "test" for a in sd["test"]}, **{a: "heldout" for a in sd["heldout_shape"]}}
    n = {"select": 0, "train": 0, "heldout_grasp": 0, "heldout_shape": 0, "test": 0}
    for aid, row in sorted(rows.items()):
        for gi in range(cfg.grasps_per_assembly):
            key = f"{aid}_g{gi:02d}"
            kind = grasp_values_v1(cfg, aid, gi, row["box"])[2]
            n[condition_v1(role.get(aid), key, kind)] += 1
    m = min_select_grasps()
    return {"grasps": n, "min_select_grasps": m, "select_margin": n["select"] - m,
            "production_min_select_grasps": production_select_min(m),
            "rule": "over every included assembly and grasp index (collect_subset ignored); train_wm320 needs "
                    f"n_select >= {m} (demo320.MIN_SELECT_GRASPS) and a held-out trajectory; a production plan "
                    f"needs n_select >= ceil({SELECT_MARGIN_FACTOR} x {m}) = {production_select_min(m)}"}


def production_select_min(m: int) -> int:
    """m_prod = ceil(SELECT_MARGIN_FACTOR x m): the selection grasps a production plan needs (module docstring)."""
    return math.ceil(SELECT_MARGIN_FACTOR * m)


def counts_problem(counts: dict | None, production: bool = False) -> str | None:
    """Why the split cannot be trained, None if it can: n_select < m_prod (production) or < m (train_wm320's own
    refusal, both), or no held-out grasp / shape."""
    if counts is None:
        return None
    n, m = counts["grasps"], counts["min_select_grasps"]
    why = []
    mp = production_select_min(m)
    if n["select"] < m:
        why.append(f"{n['select']} selection grasps < demo320.MIN_SELECT_GRASPS {m}")
    elif production and n["select"] < mp:
        why.append(f"{n['select']} selection grasps < ceil({SELECT_MARGIN_FACTOR} x MIN_SELECT_GRASPS {m}) = {mp} "
                   "(refused t00 references add no entry, so a production plan needs this margin)")
    elif not production and n["select"] < mp:
        why.append(f"{n['select']} selection grasps < {mp} = ceil({SELECT_MARGIN_FACTOR} x {m}), the production "
                   "minimum (refused t00 references add no entry)")
    for k in ("heldout_grasp", "heldout_shape"):
        if n[k] == 0:
            why.append(f"no {k} grasp")
    return "; ".join(why) or None


def counts_text(counts: dict | None) -> str:
    if counts is None:
        return "split counts: none (split is not automate_v1)"
    n = counts["grasps"]
    return (f"split counts (every included assembly, {', '.join(f'{k} {v}' for k, v in n.items())}): selection "
            f"grasps {n['select']} vs MIN_SELECT_GRASPS {counts['min_select_grasps']} (train_wm320's limit; margin "
            f"{counts['select_margin']}) and vs {production_select_min(counts['min_select_grasps'])} = "
            f"ceil({SELECT_MARGIN_FACTOR} x {counts['min_select_grasps']}) (a production plan's minimum: a grasp "
            "whose t00 reference is refused adds no entry)")


def status_text(table: list[dict]) -> str:
    lines = [f"{'assembly':9s} {'status':9s} reason / zero-width variables"]
    for r in table:
        zw = f" zero-width: {','.join(r['zero_width'])}" if r["zero_width"] else ""
        dl = {x: m.get("depth_left_mm") for x, m in (r.get("m1b") or {}).items() if m.get("depth_left_mm") is not None}
        dl = (" depth_left_mm " + ", ".join(f"({x}) {float(v):.3f}" for x, v in dl.items())) if dl else ""
        sb = r.get("start_box") or {}
        sb = (f" START ALREADY SUCCESSFUL possible (gap from {sb['gap_from']}: aligned h_min {sb['h_min_mm']} mm vs "
              f"{sb['limit_mm']} mm; {100 * sb['fraction']:.1f} % of {sb['draws']} draws with rotation)"
              ) if sb.get("possible") or sb.get("fraction") else ""
        lines.append(f"{r['assembly_id']:9s} {r['status']:9s} {r['reason'] or ''}{zw}{dl}{sb}")
    n = {s: sum(r["status"] == s for r in table) for s in ("included", "forced", "excluded")}
    lines.append(f"{n['included']} included, {n['forced']} forced, {n['excluded']} excluded of {len(table)}")
    return "\n".join(lines)


def asset_files() -> dict:
    """The versioned per-assembly asset files the plan reads (plan.json "asset_files"): the geometry file
    assets.SPEC_FILE and the grasp-ranges file grasp.RANGES_FILE, as the code's config names them (never named here)."""
    return {"spec_file": A.SPEC_FILE, "ranges_file": G.RANGES_FILE, "grasp_spin_axis": C.GRASP_SPIN_AXIS}


def plan_document(cfg: CollectConfig, specs: list[GraspSpec], table: list[dict], code: dict | None = None) -> dict:
    """config.json / plan.json content with their hashes (module docstring)."""
    conf = cfg.dump()
    shared = [A.ASSET_DIR / "plug_grasps.json", A.ASSET_DIR / "disassembly_dist.json",
              *sorted((A.ASSET_DIR / "franka_mimic_collision").glob("*"))]
    doc = {"config_sha": sha256_text(_canon(conf)), "physics_device": cfg.device, "switches": C.switches(),
           "code": code if code is not None else code_identity(),
           "inputs_shared": {str(f): sha256_file(f) for f in shared if f.is_file()},
           "input_roots": {"assets": str(A.ASSET_DIR), "m1b_dirs": m1b_dirs(cfg)}, "m1b_code": m1b_code(table),
           "asset_files": asset_files(),
           "assemblies": table,
           "zero_width_assemblies": {r["assembly_id"]: r["zero_width"] for r in table
                                     if r["zero_width"] and r["status"] != "excluded"},
           "rng": {"entropy": cfg.seed, "streams": STREAMS,
                   "spawn_key": "(int(assembly id), grasp index, trajectory index, purpose)"},
           "split": {"name": cfg.split, "file": f"split_{cfg.split}.json" if cfg.split == SPLIT_VERSION else None,
                     "document": split_document([r["assembly_id"] for r in table if r["status"] != "excluded"])
                     if cfg.split == SPLIT_VERSION else None, "counts": split_counts(cfg, table)},
           "grasps": [{"grasp": s.key, "port": s.port, "condition": s.condition, "grasp_kind": s.grasp_kind,
                       "dtheta_deg": s.pitch,
                       "dp_m": s.grasp_pos, "dpsi_deg": s.spin, "u": s.u, "seed": s.seed, "gi": s.gi,
                       "trajectories": [asdict(t) for t in s.trajectories]} for s in specs]}
    doc = json.loads(_canon(doc))                                   # what a reader of plan.json sees
    doc["plan_sha"] = sha256_text(_canon({k: v for k, v in doc.items() if k != "code"} |
                                         {"code_files_sha256": doc["code"]["files_sha256"]}))
    return doc


def _write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def grasp_data(out: Path) -> list[str]:
    """Grasp data already in an output folder (grasp.json, t*.npz or .claim in <out>/<id>/<id>_g*/), first 5."""
    found = []
    for pat in ("*/*_g*/grasp.json", "*/*_g*/t*.npz", "*/*_g*/.claim"):
        found += [str(p.relative_to(out)) for p in out.glob(pat)]
        if len(found) >= 5:
            break
    return found[:5]


def freeze(out: Path, cfg: CollectConfig, doc: dict) -> dict:
    """Write config.json and plan.json once; afterwards require that they equal this worker's (Refused, a SystemExit,
    if not). A first plan.json is refused when the folder already holds grasp data."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".freeze.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        plan_f, conf_f = out / "plan.json", out / "config.json"
        if not plan_f.exists():
            old_data = grasp_data(out)
            if old_data:
                raise Refused(f"{out}: no plan.json but grasp data from an earlier plan ({old_data}); use a new output "
                              "folder (and never delete plan.json of a running or finished collection)")
            _write_atomic(conf_f, json.dumps({**cfg.dump(), "config_sha": doc["config_sha"]}, indent=1))
            sp = doc.get("split") or {}
            if sp.get("file"):                           # the versioned split file next to plan.json (also inside it)
                _write_atomic(out / sp["file"], json.dumps({**sp["document"], "plan_sha": doc["plan_sha"]}, indent=1))
            _write_atomic(plan_f, json.dumps(doc, indent=1))
            return doc
        old = json.loads(plan_f.read_text())
        conf = json.loads(conf_f.read_text()) if conf_f.exists() else {}
    bad = [k for k in ("config_sha", "physics_device", "switches", "assemblies", "grasps", "rng", "inputs_shared",
                        "input_roots", "split", "m1b_code", "asset_files")
           if old.get(k) != doc.get(k)]
    if old.get("code", {}).get("files_sha256") != doc["code"]["files_sha256"]:
        bad.append("code files_sha256")
    if old.get("plan_sha") != doc["plan_sha"]:
        bad.append("plan_sha")
    if conf.get("config_sha") != doc["config_sha"]:
        bad.append("config.json config_sha")
    if bad:
        raise Refused(f"{out}: this worker's plan differs from the frozen plan.json in {bad}; refusing to run "
                      "(start a new output folder for a changed plan, config, input or code; do not delete plan.json)")
    return old


# ---------------------------------------------------------------------------------------------- one grasp
class SimulatorError(RuntimeError):
    """An exception that is not an expert failure, inside a grasp; carries the grasp's partial summary."""

    def __init__(self, msg: str, summary: dict):
        super().__init__(msg)
        self.summary = summary


def _axis_from(arrays: dict) -> np.ndarray | None:
    """Insertion direction from the FK hand positions during the axial push (wm.collect.episode._axis_from)."""
    push = arrays["phase"] == schema.PHASES.index("axial_push")
    if push.sum() < 2:
        return None
    pos = arrays["hand"][1:][push][:, :3, 3]
    d = pos[-1] - pos[0]
    return d / np.linalg.norm(d)


PROCESS_INFO: dict = {}        # set by collect_expert.py: pid, start, launch_s (Isaac start time), unit


def gpu_memory_mib(pid: int | None = None) -> dict | None:
    """GPU memory of one process on GPU 0 and GPU 1 [MiB], from the full nvidia-smi process table (compute and
    graphics contexts, nvidia-smi -q -d PIDS); None if nvidia-smi fails."""
    pid = os.getpid() if pid is None else pid
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    out = {}
    try:
        for i in (0, 1):
            txt = subprocess.run(["nvidia-smi", "-i", str(i), "-q", "-d", "PIDS"], capture_output=True, text=True,
                                 timeout=20, env=env, check=True).stdout
            out[f"gpu{i}_mib"] = parse_pid_table(txt).get(pid, 0)
    except (OSError, subprocess.SubprocessError):
        return None
    return out


def parse_pid_table(txt: str) -> dict:
    """nvidia-smi -q -d PIDS text -> {pid: MiB} (the C and G entries of one pid summed)."""
    out, pid = {}, None
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        k = k.strip()
        if k == "Process ID":
            pid = int(v.strip())
        elif k == "Used GPU Memory" and pid is not None:
            m = re.match(r"\s*(\d+)", v)
            out[pid] = out.get(pid, 0) + (int(m.group(1)) if m else 0)
    return out


def _memory(gpu: bool = True) -> dict:
    """Process peak RSS, this scope's memory.peak / memory.current / memory.max and memory.stat anon / file / shmem
    (cgroup v2), with a warning when memory.peak passes MEMORY_WARN_FRACTION of memory.max; this process's GPU memory."""
    out = {"maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
    try:
        cg = Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
        for k in ("memory.peak", "memory.current", "memory.max"):
            p = Path("/sys/fs/cgroup") / cg.lstrip("/") / k
            if p.exists():
                v = p.read_text().strip()
                out[k.replace(".", "_") + "_gb"] = None if v == "max" else round(int(v) / 1e9, 2)
        st = Path("/sys/fs/cgroup") / cg.lstrip("/") / "memory.stat"
        if st.exists():
            rows = dict(line.split() for line in st.read_text().splitlines() if len(line.split()) == 2)
            out["stat_gb"] = {k: round(int(rows[k]) / 1e9, 2) for k in ("anon", "file", "shmem") if k in rows}
        peak, mx = out.get("memory_peak_gb"), out.get("memory_max_gb")
        if peak is not None and mx:
            out["peak_fraction"] = round(peak / mx, 3)
            out["warning"] = peak > MEMORY_WARN_FRACTION * mx
    except Exception:
        pass
    if gpu:
        out["gpu"] = gpu_memory_mib()
    return out


def truth_summary(robot) -> dict:
    """ManiSkill-style truth (checks.truth: depth / lateral / tilt at the tip centre): information only."""
    return K.truth(robot)


def automate_success(robot, arrays: dict | None = None) -> dict:
    """AutoMate's own success rule (checks.py: Isaac Lab's check_plug_inserted_in_socket; the user's decision of
    2026-11-22; module docstring, outcome):
        inserted_end        s now (checks.truth of the robot at the end; for a recorded trajectory = s_T)
        inserted_any_step   S_any = OR_{t = 1..T} s_t from eval_rel (row 0, the start state, is never evaluated, as
                            in AutoMate, which checks only after each control step)
        first_success_step  the smallest t >= 1 with s_t (None if none); success_at_start = s_0 (information)
        inserted            inserted_any_step for a recorded trajectory (arrays given), else inserted_end
    plus height_mm / keypoint_dist_mm at the end."""
    end = K.truth(robot)
    out = {"inserted_end": bool(end["success"]), "height_mm": float(end["height_mm"]),
           "keypoint_dist_mm": float(end["keypoint_dist_mm"])}
    if arrays is None:
        out["inserted"] = out["inserted_end"]
        return out
    st = K.success_steps(arrays["eval_rel"], robot.port_spec)          # the start state (row 0) is never counted
    out["inserted_any_step"] = bool(st["success_any"])
    out["first_success_step"] = st["first_success_step"]
    out["success_at_start"] = bool(st["start_success"])
    out["inserted"] = out["inserted_any_step"]
    return out


def require_success_rule():
    """Refused unless checks.py has AutoMate's success rule (checked before any Isaac start)."""
    if not all(callable(getattr(K, f, None)) for f in ("automate_success", "truth_from_rel_spec", "truth", "success_steps")):
        raise Refused("checks.py lacks AutoMate's success rule (automate_success / truth_from_rel_spec)")


def eval_tip(arrays: dict, robot) -> np.ndarray:
    """(T + 1, len(EVAL_TIP_COLUMNS)) float32 truth per step from eval_rel (EVAL_TIP_COLUMNS = checks.TRUTH_COLUMNS,
    checks.truth_from_rel_spec; success as 0 / 1; row 0 = the start state)."""
    t = K.truth_from_rel_spec(np.asarray(arrays["eval_rel"], np.float64), robot.port_spec)
    return np.stack([np.asarray(t[k], np.float64) for k in EVAL_TIP_COLUMNS], -1).astype(np.float32)


def _disk_check(out: Path, need: float, where: str):
    """Stopped (with <out>/STOP and an ALERT line) when <out> has fewer than `need` bytes free (module docstring)."""
    import shutil
    try:
        free = shutil.disk_usage(out).free
    except OSError:
        return
    if free < need:
        msg = f"disk free {free / 1e9:.1f} GB < {need / 1e9:.0f} GB in {out} ({where}): every worker stops"
        alert(out, msg)
        try:
            with open(out / STOP_FILE, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except OSError:
            pass
        raise Stopped(msg)


def usd_identity(robot) -> dict | None:
    """scene.usd_identity() of the robot's scene, computed once per built scene (cached on the robot); None without
    one (CPU tests); {"error": ...} when it fails (information, never fatal)."""
    cached = getattr(robot, "_usd_identity", None)
    if cached is not None:
        return cached
    f = getattr(getattr(robot, "sc", None), "usd_identity", None)
    if not callable(f):
        return None
    try:
        v = f()
    except Exception as exc:
        v = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    try:
        robot._usd_identity = v
    except Exception:
        pass
    return v


def _usd_layers(ident) -> dict | None:
    """The compared layers (USD_COMPARED) of a usd_identity dict; None if it has no layer table."""
    if not isinstance(ident, dict) or not isinstance(ident.get("layers"), dict):
        return None
    return {str(k): v for k, v in ident["layers"].items() if str(k).endswith(USD_COMPARED)}


def usd_vs_m1b(robot, aid: str, cfg: CollectConfig) -> dict:
    """Compare the plug / socket / plug_sensor USD layers this scene loaded with those of the assembly's milestone-1b
    (ii) result (module docstring; r4-integration-05). -> {"status", "refuse", ...}; status "equal", "different",
    "missing" (no layer table or no compared layer on either side), "unreadable", "no (ii) result" or "no scene
    identity" (no scene: CPU tests). Refused statuses: all but "equal" / "no scene identity", and "no (ii) result" of a
    forced assembly (forcing means there is no gate evidence). Computed once per built scene."""
    cached = getattr(robot, "_usd_vs_m1b", None)
    if cached is not None:
        return cached
    mine = usd_identity(robot)
    out = {"status": "no scene identity", "refuse": False}
    if mine is not None:
        res, path = m1b_result(aid, m1b_dirs(cfg))
        out = {"m1b_file": path}
        if res is None:
            out["status"] = "no (ii) result"
            out["refuse"] = aid not in cfg.assemblies_force
        else:
            a, b = _usd_layers(mine), _usd_layers(res.get("usd_identity"))
            out.update(scene=a, m1b=b)
            if not a or not b:
                out["status"] = "missing"
            elif any(str(v).startswith("unreadable") for v in list(a.values()) + list(b.values())):
                out["status"] = "unreadable"
            elif a != b:
                out["status"] = "different"
            else:
                out = {"status": "equal", "m1b_file": path}
            out["refuse"] = out["status"] != "equal"
    try:
        robot._usd_vs_m1b = out
    except Exception:
        pass
    return out


def _clean(gdir: Path):
    """Remove the trajectory files of an earlier, unfinished attempt (readers index every t*.npz)."""
    for f in list(gdir.glob("t*.npz")) + list(gdir.glob("*.tmp.npz")):
        f.unlink()


def write_summary(gdir: Path, summary: dict):
    _write_atomic(gdir / "grasp.json", json.dumps(summary, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else float(o)))


def collect_grasp(robot, spec: GraspSpec, out: Path, cfg: CollectConfig, frozen: dict | None = None,
                  beat=None) -> dict:
    """wm.collect.episode.collect_grasp on AutoMate. robot: wm.sim_automate.robot.Robot of spec.port. Returns the
    grasp summary (the caller writes grasp.json); raises SimulatorError (with the partial summary) on any exception
    that is not an expert failure, and Stopped when <out>/STOP appears (checked before every trajectory).
    beat(stage=None): the worker's heartbeat (called at every stage and, throttled, at every recorded step).
    Socket draws (config.SOCKET_DRAW, frozen per collection): "per_reset" (AutoMate's original) resets before every
    planned trajectory with socket_index = its file index, runs an unrecorded reference insertion there
    (Expert.reference_label) for that trajectory's FK H* / axis label, then places the start from that label;
    "per_grasp" keeps the label of t00 and chains the trajectories without a reset (re-homing only after a failed
    placement).
    Per trajectory, a checks.ResetError of the reset (no IK solution for the spawn, socket pose not taken) gives
    outcome "reset_failed" and a checks.ContactReportError (contact report empty while the sensor reads force) of
    the label reference, the unrecorded transfer (stage "placement") or the recorded run gives "contact_report_error":
    recorded with the message, no file, and the grasp continues. At t00 the same two exceptions close the grasp softly
    (reference_ok False, reference_soft_outcome "reset_failed" / "contact_report_error", no scene rebuild: the reset
    is deterministic). An ALERT line when more than RESET_OR_REPORT_ALERT_SHARE of the planned trajectories end
    reset_failed / contact_report_error / label_failed, and whenever the t00 reference is refused (all trajectories of
    the grasp are lost). Any other exception is an error (SimulatorError). per_grasp: a failed re-home after a
    contact-report error ends the grasp's remaining trajectories (grasp.json rehome_failed; not an error).
    grasp.json "usd_identity": the USD layers this scene loaded with their sha256 (scene.usd_identity, once per built
    scene; the USDs still come from the Isaac nucleus, so this records what PhysX actually used); "usd_vs_m1b" their
    comparison with the milestone-1b (ii) result (usd_vs_m1b; Refused, with an ALERT line, unless equal).
    timing "placement" rows (entry placement_legs / placement_failures) carry every transfer leg's numbers.
    Placement (at most PLACEMENT_ATTEMPTS starts per trajectory, redraws from its "placement" stream): a start whose
    expert.start_clearance minimum (plug, hand and both fingers at the start, the plug along the descent, to the
    socket mesh) is below expert.START_CLEARANCE_M (1 mm; the expert's constant, not a copy) is redrawn without a
    transfer (failure stage "clearance"). This filters the start distribution (deviation table row 35).
    Random waypoints (decision 7 of 2026-11-23, row 39): before the recorded run, redraw_waypoints replaces every
    waypoint whose commanded hand / finger is closer than expert.HAND_FINGER_COMMAND_CLEARANCE_M (1 mm) to the socket
    mesh (Expert.waypoint_clearance) by a draw from the trajectory's "waypoint" stream; the row records
    waypoint_redraws (one record per rejected draw) and, if any, planned_waypoints; the meta's "waypoints" are the
    ones run."""
    from .expert import HAND_FINGER_COMMAND_CLEARANCE_M, START_CLEARANCE_M, Expert
    from .frames import frame
    from .record import Recorder
    frozen = frozen or {}
    beat = beat or (lambda stage=None: None)
    per_reset = C.SOCKET_DRAW == "per_reset"
    t0 = time.time()
    gdir = out / spec.port / spec.key
    ps = robot.port_spec
    base_meta = {"grasp": spec.key, "port": spec.port, "condition": spec.condition,
                 "pitch_deg": spec.pitch, "grasp_pos_m": spec.grasp_pos}
    extra = {"spin_deg": spec.spin, "u": spec.u, "assembly_id": robot.assembly_id, "simulator": "automate",
             "pin_length_m": float(ps.pin_length_m), "symmetry_n": ps.symmetry_n,
             "tip_centre_m": [float(v) for v in ps.tip_centre_m], "config_sha": frozen.get("config_sha"),
             "plan_sha": frozen.get("plan_sha"), "physics_device": cfg.device, "switches": C.switches(),
             "disassembly_dist_m": float(getattr(ps, "disassembly_dist_m", float("nan"))),
             "g_top_m": K.g_top_of(ps)}
    summary = {**base_meta, **extra, "seed": spec.seed, "trajectories": [], "bytes": 0,
               "grasp_rule_heldout": grasp_rule_heldout(spec.u), "max_decisions": robot.max_decisions,
               "outcome_rule": ("insert / perturbed: inserted iff S_any = OR_{t>=1} AutoMate success (Isaac Lab "
                                "check_plug_inserted_in_socket; start state excluded); reference / label: expert "
                                "completed AND AutoMate success at the end; truth = information")}
    reset_err, contact_err = K.ResetError, K.ContactReportError
    summary["usd_identity"] = usd_identity(robot)
    cmp_ = usd_vs_m1b(robot, spec.port, cfg)
    summary["usd_vs_m1b"] = cmp_["status"]
    if cmp_["status"] != "equal":
        summary["usd_vs_m1b_detail"] = cmp_
    if cmp_["refuse"]:
        msg = (f"{spec.key}: the USD layers {USD_COMPARED} of this scene vs the milestone-1b (ii) result "
               f"{cmp_.get('m1b_file')}: {cmp_['status']} (the Isaac nucleus asset changed after the gate?); refused")
        alert(out, msg)
        raise Refused(msg)
    timing = {"reset_s": 0.0, "resets": 0, "reset_settle_steps": 0, "recorded_s": 0.0, "recorded_steps": 0,
              "transfer_s": 0.0, "transfer_steps": 0, "label_s": 0.0, "label_steps": 0}
    grasp_kw = {"clamp_deg": spec.pitch, "dp_m": spec.grasp_pos, "spin_deg": spec.spin}

    def rng_info(idx: int, purposes) -> dict:
        return {"entropy": cfg.seed, "spawn_keys": {p: [int(spec.port), spec.gi, idx, STREAMS[p]] for p in purposes}}

    def fail(stage: str, exc: Exception, entry: dict | None = None):
        msg = f"{stage}: {type(exc).__name__}: {exc}"
        if entry is not None:
            entry.update(outcome="error", error=msg, trace=traceback.format_exc()[-2000:])
            summary["trajectories"].append(entry)
        summary.update(error=msg, trace=traceback.format_exc()[-2000:], timing=timing, memory=_memory(),
                       seconds=round(time.time() - t0, 1), pid=os.getpid())
        raise SimulatorError(msg, summary) from exc

    def on_step(*a, **k):
        beat()
        return rec.on_step(*a, **k)

    def reset(index: int):
        ts0 = time.time()
        robot.reset(spec.seed, **grasp_kw, socket_index=index)
        timing["reset_s"] += time.time() - ts0
        timing["resets"] += 1
        timing["reset_settle_steps"] += int(getattr(C, "SETTLE_STEPS", 0))   # robot.reset settles this many steps

    def finger_info(res: dict) -> dict:
        f = float((res.get("peak") or {}).get("finger_socket_N", 0.0))
        return {"finger_socket_N": f, "finger_contact": f > FINGER_MAX_N}

    def t00_refused(why: str):
        """A refused t00 reference loses every trajectory of the grasp: always an ALERT line (r4-expert-02 c)."""
        alert(out, f"{spec.key}: t00 reference refused ({why}); the grasp's {len(spec.trajectories)} planned "
                   "trajectories are not collected; grasp.json")

    def rehome(entry: dict) -> bool:
        """per_grasp: back to the reset state before the next start; False (recorded) if the reset fails."""
        try:
            reset(0)
            return True
        except reset_err as exc:
            entry["rehome_error"] = f"{type(exc).__name__}: {exc}"
            return False

    rec, began, stage = None, False, "reference reset"
    try:
        beat("prepare")
        gdir.mkdir(parents=True, exist_ok=True)
        _clean(gdir)
        rec = Recorder(robot, images=cfg.images, image_every=cfg.image_every)
        beat("reset")
        reset(0)
        extra.update(eval_socket_pose=robot.episode["socket_pose"], socket_index=0)
        summary.update(eval_socket_pose=robot.episode["socket_pose"], socket_draw=robot.episode["socket_draw"])
        expert = Expert(robot, transfer=cfg.transfer)
        # 1. reference insertion from the reset spawn (aligned, 12 mm above the socket top), recorded
        beat("reference")
        stage = "reference run"
        rec.begin()
        began = True
        res = expert.run("reference", on_step, stream(cfg, spec.port, spec.gi, 0, "expert"))
        arrays = rec.end()
        began = False
        arrays["eval_tip"] = eval_tip(arrays, robot)
        tr = truth_summary(robot)
        succ = automate_success(robot, arrays)
    except (reset_err, contact_err) as exc:          # deterministic for this seed: a rebuild would repeat it
        if began:
            try:
                rec.end()
            except Exception:
                pass
        soft = "reset_failed" if isinstance(exc, reset_err) else "contact_report_error"
        why = f"{stage}: {type(exc).__name__}: {exc}"
        summary.update(episode_seed=spec.seed, reference_attempts=1, reference_ok=False, reference_failure=why,
                       reference_soft_outcome=soft)
        t00_refused(why)
        return _finish(summary, timing, t0)
    except Exception as exc:
        fail("reference", exc)
    try:
        timing["recorded_s"] += res["wall_s"]
        timing["recorded_steps"] += res["steps"]
        completed = bool(res.get("completed"))
        ok = completed and succ["inserted_end"]       # lead resolution r3: completed AND AutoMate's rule at the end
        failure = None
        if not completed:
            failure = "expert did not complete: " + str(res.get("failure_reason") or "no failure reason")
        elif not succ["inserted_end"]:
            failure = (f"expert completed but AutoMate's rule does not hold at the end (h {succ['height_mm']:.3f} mm, "
                       f"keypoint distance {succ['keypoint_dist_mm']:.3f} mm)")
        summary.update(episode_seed=spec.seed, reference_attempts=1, reference_ok=ok, reference_failure=failure,
                       reference_expert_completed=completed, reference_success=succ,
                       reference_attempt_failures=[], reference_truth=tr, reference_peak=res.get("peak"),
                       reference_stages=res.get("stages"), **{"reference_" + k: v for k, v in finger_info(res).items()})
        if not ok:
            t00_refused(failure)
            return _finish(summary, timing, t0)
        H_star, axis = robot.hand_pose(), _axis_from(arrays)
        if axis is None:                                # no FK axis: never fall back to the true axis (simulator truth)
            summary.update(reference_ok=False, reference_failure="no axial push (FK axis undefined)")
            t00_refused(summary["reference_failure"])
            return _finish(summary, timing, t0)
        label = {"H_star": H_star.tolist(), "axis": axis.tolist()}
        dt = float(np.linalg.norm(H_star[:3, 3] - robot.H_star()[:3, 3]))
        dr = float(np.degrees(np.arccos(np.clip((np.trace(H_star[:3, :3].T @ robot.H_star()[:3, :3]) - 1) / 2, -1, 1))))
        summary.update(label, H_star_model=robot.H_star().tolist(), H_star_fk_vs_model_mm_deg=[dt * 1e3, dr],
                       axis_vs_true_deg=float(np.degrees(np.arccos(np.clip(axis @ robot.axis(), -1, 1)))))
        if not per_reset:                               # per_grasp: every trajectory's label is t00's
            extra["H_star_fk_vs_model_mm_deg"] = [dt * 1e3, dr]
        meta = {**base_meta, **label, **extra, "episode_seed": spec.seed, "kind": "reference", "idx": 0,
                "outcome": "inserted", "expert_completed": completed, "success": succ,
                "automate_success_any": bool(succ["inserted_any_step"]), "file": "t00_reference.npz",
                "steps": int(res["steps"]), "wall_s": res["wall_s"], "peak": res.get("peak"), **finger_info(res),
                "eval_tip_columns": list(EVAL_TIP_COLUMNS), "rng": rng_info(0, ("expert",))}
        F = frame(np.asarray(axis), robot.kin.base_T[:3, 0])
        L = float(ps.pin_length_m)
    except Exception as exc:
        fail("reference label", exc)
    _disk_check(out, DISK_FREE_MIN_SAVE, f"{spec.key} t00")
    try:
        summary["bytes"] += store.save(gdir / "t00_reference.npz", arrays, meta, cfg.rgb_codec, cfg.jpeg_quality)
    except Exception as exc:
        fail("reference save", exc)
    summary["trajectories"].append({"idx": 0, "kind": "reference", "outcome": "inserted", "steps": int(res["steps"]),
                                    "wall_s": res["wall_s"], "success": succ, "truth": tr, "peak": res.get("peak"),
                                    "expert_completed": completed, "automate_success_any": bool(succ["inserted_any_step"]),
                                    "socket_index": 0, "file": "t00_reference.npz"})

    # 2. planned trajectories, each from its own start (transfer not recorded), each in its own try
    for ts in spec.trajectories:
        idx = ts.idx + 1
        entry = {"idx": idx, "kind": ts.kind}
        check_stop(out)                                     # between trajectories: a STOP file stops here
        beat(f"trajectory {idx} ({ts.kind})")
        try:
            sidx = idx if per_reset else 0                  # the socket pose number of this trajectory
            lab_info = {}
            if per_reset:
                beat(f"reset {idx}")
                try:
                    reset(sidx)
                except reset_err as exc:                    # no IK solution, socket pose not taken: this trajectory only
                    entry.update(outcome="reset_failed", socket_index=sidx, error=f"{type(exc).__name__}: {exc}")
                    summary["trajectories"].append(entry)
                    continue
                beat(f"label {idx}")
                tl0 = time.time()
                try:
                    lab = expert.reference_label(stream(cfg, spec.port, spec.gi, idx, "reference_label"),
                                                 on_step=lambda *a: beat())
                except contact_err as exc:
                    timing["label_s"] += time.time() - tl0
                    entry.update(outcome="contact_report_error", socket_index=sidx, stage="label",
                                 error=f"{type(exc).__name__}: {exc}")
                    summary["trajectories"].append(entry)
                    continue
                timing["label_s"] += time.time() - tl0
                timing["label_steps"] += int(lab["steps"])
                lsucc = automate_success(robot)
                lab_completed = lab.get("completed")
                # lead resolution r3: the label counts only if the expert completed AND AutoMate's rule holds at the end
                lab_ok = (bool(lab["ok"]) and (lab_completed is None or bool(lab_completed)) and lsucc["inserted_end"]
                          and lab["axis"] is not None)
                lab_info = {"label_reference": {
                    "accepted": lab_ok, "ok": bool(lab["ok"]), "completed": lab_completed,
                    "inserted_end": lsucc["inserted_end"], "success": lsucc, "success_any": lab.get("success_any"),
                    "steps": int(lab["steps"]), "failure": lab.get("failure"), "truth": lab.get("truth"),
                    "peak": lab.get("peak"), "H_star_fk_vs_model_mm_deg": lab.get("H_star_fk_vs_model_mm_deg"),
                    "axis_defined": lab["axis"] is not None}}
                entry.update(socket_index=sidx, eval_socket_pose=robot.episode["socket_pose"], **lab_info)
                if not lab_ok:
                    entry["outcome"] = "label_failed"           # no FK label for this socket pose: not collected
                    summary["trajectories"].append(entry)
                    continue
                t_label = {"H_star": np.asarray(lab["H_star"]).tolist(), "axis": list(lab["axis"])}
                T_H = np.asarray(lab["H_star"])
                F_t = frame(np.asarray(lab["axis"]), robot.kin.base_T[:3, 0])
            else:
                entry["socket_index"] = 0
                t_label, T_H, F_t = label, H_star, F
            placement_rng = np.random.default_rng(stream(cfg, spec.port, spec.gi, idx, "placement"))
            start, placed, fails, k, mv, contact_fail = ts.start, False, [], 0, None, None
            tr0 = time.time()
            for k in range(PLACEMENT_ATTEMPTS):
                beat(f"placement {idx} attempt {k + 1}")
                if k:
                    start = sample_start(placement_rng)
                # g0 measured from config.START_GAP_FROM ("socket_top", decision of 2026-11-23: b = L + g_top + g0)
                T_start = Expert.start_pose(T_H, F_t, L, start, extra["g_top_m"], C.START_GAP_FROM)
                clr = expert.start_clearance(T_start)
                if clr["min"] < START_CLEARANCE_M:          # the expert's own start-clearance limit (1 mm)
                    fails.append({"stage": "clearance", "clearance_mm": {a: round(b * 1e3, 2) for a, b in clr.items()}})
                    continue
                try:
                    mv = expert.move_to_start(T_start)
                except contact_err as exc:                  # the transfer's contact report (r4-expert-03)
                    contact_fail = f"{type(exc).__name__}: {exc}"
                    break
                timing["transfer_steps"] += mv["steps"]
                if mv["ok"]:
                    placed = True
                    break
                fails.append({"stage": mv.get("stage") or "start_error", "start_error_mm": mv["start_error_mm"],
                              "legs": mv.get("legs")})
                try:
                    reset(sidx)                             # re-home (reset state, same grasp and socket pose)
                except reset_err as exc:
                    fails.append({"stage": "re-home reset", "error": f"{type(exc).__name__}: {exc}"})
                    break
            timing["transfer_s"] += time.time() - tr0
            entry.update(start=start.as_dict(), placement_attempts=k + 1, placement_failures=fails)
            if contact_fail is not None:
                entry.update(outcome="contact_report_error", stage="placement", error=contact_fail)
                summary["trajectories"].append(entry)
                if not per_reset and not rehome(entry):     # per_grasp chains the starts: re-home before the next
                    summary["rehome_failed"] = {"after": idx, "error": entry["rehome_error"]}
                    break
                continue
            if not placed:
                entry["outcome"] = "placement_failed"
                summary["trajectories"].append(entry)
                continue
            entry["start_error_mm"] = mv["start_error_mm"]
            entry["placement_legs"] = mv.get("legs")
            waypoints = ts.waypoints
            if waypoints is not None:                   # decision 7 of 2026-11-23 (row 39): redraw_waypoints
                waypoints, wrej = redraw_waypoints(
                    list(waypoints), expert.waypoint_clearance,
                    np.random.default_rng(stream(cfg, spec.port, spec.gi, idx, "waypoint")),
                    HAND_FINGER_COMMAND_CLEARANCE_M)
                entry["waypoint_redraws"] = wrej
                if wrej:
                    entry["planned_waypoints"] = [w.as_dict() for w in ts.waypoints]
            rec.begin()
            try:
                # the expert's interface (decision 4 of 2026-11-23): random waypoints and the contact offset move at
                # its constant RANDOM_SPEED_M_S / CLOSE_ANG_SPEED_DEG_S (wm_v1's _move_outside); speeds = the plan's
                # pinned cfg.random_speed_m_s draws, which the expert accepts only when equal to RANDOM_SPEED_M_S
                res = expert.run(ts.kind, on_step, stream(cfg, spec.port, spec.gi, idx, "expert"), offset=ts.offset,
                                 waypoints=waypoints, speeds=ts.speeds)
            except contact_err as exc:
                rec.end()
                entry.update(outcome="contact_report_error", stage="run", error=f"{type(exc).__name__}: {exc}")
                summary["trajectories"].append(entry)
                if not per_reset and not rehome(entry):     # per_grasp chains the starts: re-home before the next
                    summary["rehome_failed"] = {"after": idx, "error": entry["rehome_error"]}
                    break
                continue
            arrays = rec.end()
            arrays["eval_tip"] = eval_tip(arrays, robot)
            timing["recorded_s"] += res["wall_s"]
            timing["recorded_steps"] += res["steps"]
            tr = truth_summary(robot)
            succ = automate_success(robot, arrays)
            if ts.kind in ("insert", "perturbed"):
                outcome = "inserted" if succ["inserted"] else "expert_failed"      # S_any (module docstring)
            elif ts.kind == "contact":                      # never "inserted": that name is AutoMate's rule
                o = res.get("outcome", "expert_failed") if res.get("completed") else "expert_failed"
                outcome = "reached_depth" if o == "inserted" else o
            else:
                outcome = "done" if res.get("completed") else "expert_failed"
            entry.update(outcome=outcome, expert_completed=bool(res.get("completed")), success=succ,
                         automate_success_any=bool(succ["inserted"]),
                         steps=int(res.get("steps", 0)), failure=res.get("failure_reason"),
                         depth_left_mm=res.get("depth_left_mm"), wall_s=res["wall_s"], truth=tr,
                         peak=res.get("peak"), **finger_info(res),
                         stages=[{k_: v for k_, v in s.items() if k_ in STAGE_KEYS} for s in res.get("stages", [])])
            if ts.offset is not None:
                entry["offset"] = ts.offset.as_dict()
            meta = {**base_meta, **t_label, **extra, "episode_seed": spec.seed, **entry, "seed_offset": ts.seed_offset,
                    "waypoints": None if waypoints is None else [w.as_dict() for w in waypoints],
                    "speeds_m_s": ts.speeds, "eval_tip_columns": list(EVAL_TIP_COLUMNS),
                    "rng": rng_info(idx, ("placement", "expert") + (("reference_label",) if per_reset else ())
                                    + (("waypoint",) if waypoints is not None else ()))}
            name = f"t{idx:02d}_{ts.kind}.npz"
            meta["file"] = entry["file"] = name
            _disk_check(out, DISK_FREE_MIN_SAVE, f"{spec.key} t{idx:02d}")
            summary["bytes"] += store.save(gdir / name, arrays, meta, cfg.rgb_codec, cfg.jpeg_quality)
            summary["trajectories"].append(entry)
        except Exception as exc:
            entry.pop("file", None)
            fail(f"trajectory {idx} ({ts.kind})", exc, entry)
    soft = [t for t in summary["trajectories"] if t.get("outcome") in SOFT_OUTCOMES]
    if spec.trajectories and len(soft) > RESET_OR_REPORT_ALERT_SHARE * len(spec.trajectories):
        alert(out, f"{spec.key}: {len(soft)} of {len(spec.trajectories)} trajectories ended "
                   f"{' / '.join(SOFT_OUTCOMES)} ({sorted({t['outcome'] for t in soft})}); grasp.json")
    if "rehome_failed" in summary:
        alert(out, f"{spec.key}: re-home failed after trajectory {summary['rehome_failed']['after']} (per_grasp): "
                   "the remaining trajectories were not run; grasp.json rehome_failed")
    return _finish(summary, timing, t0)


def _finish(summary: dict, timing: dict, t0: float) -> dict:
    for k in ("transfer_s", "recorded_s", "reset_s", "label_s"):
        timing[k] = round(timing[k], 1)
    summary.update(seconds=round(time.time() - t0, 1), timing=timing, memory=_memory(), pid=os.getpid())
    return summary


# ---------------------------------------------------------------------------------------------- worker (one Isaac app)
_STATE: dict = {"claim": None, "heartbeat": None, "signal": None, "out": None}


def _start_time(pid: int) -> str | None:
    """Process start time (field 22 of /proc/<pid>/stat, clock ticks since boot), None if there is no such process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()[19]


def _alive(pid: int, start: str | None) -> bool:
    """Is the process that wrote a claim still running? Decided by pid AND process start time (a reused pid has
    another start time), never by a name."""
    return start is not None and pid > 0 and _start_time(pid) == start


def own_unit() -> str | None:
    """The systemd unit (scope) of this process, from /proc/self/cgroup (None outside a scope)."""
    try:
        last = Path("/proc/self/cgroup").read_text().strip().split("::")[-1].rstrip("/").split("/")[-1]
    except OSError:
        return None
    return last if last.endswith(".scope") else None


class _SignalsBlocked:
    """SIGTERM / SIGINT held back while a claim file is written, so that a stop never leaves half a claim."""

    def __enter__(self):
        self.mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})

    def __exit__(self, *exc):
        signal.pthread_sigmask(signal.SIG_SETMASK, self.mask)


@dataclass
class Claim:
    fd: int
    attempt: int
    path: Path
    previous: str
    old: dict | None = None         # the unfinished attempt found in the claim file (a process death), if any
    released: bool = False


def _parse_claim(text: str) -> dict:
    parts = text.split()
    kv = {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}
    return {"attempt": int(kv.get("attempt", 0)), "pid": int(kv.get("pid", -1)), "start": kv.get("start"),
            "time": kv.get("time"), "hang": kv.get("hang")}


def claim(gdir: Path) -> Claim | None:
    """Claim a grasp for this process (module docstring): an exclusive, non-blocking flock on <grasp>/.claim, held
    until release(). None if another live worker holds it or the grasp is done. Claim.old is the dead attempt found
    in the file (None if there was none)."""
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / ".claim"
    with _SignalsBlocked():
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        prev = os.pread(fd, 4096, 0).decode(errors="ignore").strip()
        old = _parse_claim(prev) if prev else None
        if (gdir / "grasp.json").exists() or (old is not None and old["pid"] != os.getpid() and _alive(old["pid"], old["start"])):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            return None
        attempt = (old["attempt"] if old else 0) + 1
        legacy = gdir / ".started"                                  # first smoke version: an unfinished attempt
        if legacy.exists():
            attempt += 1
            legacy.unlink()
        text = f"attempt {attempt} pid {os.getpid()} start {_start_time(os.getpid())} time {time.strftime('%Y-%m-%d_%H:%M:%S')}"
        os.ftruncate(fd, 0)
        os.pwrite(fd, text.encode(), 0)
        os.fsync(fd)
        c = Claim(fd, attempt, path, prev, old)
        _STATE["claim"] = c
    return c


def release(c: Claim, done: bool):
    """done: the grasp has its grasp.json, the claim file goes. Not done (a build failure, a stop): the previous claim
    text is put back, so that this process does not count as an attempt. Idempotent."""
    with _SignalsBlocked():
        if c.released:
            return
        if done:
            c.path.unlink(missing_ok=True)
        else:
            os.ftruncate(c.fd, 0)
            os.pwrite(c.fd, c.previous.encode(), 0)
            os.fsync(c.fd)
        fcntl.flock(c.fd, fcntl.LOCK_UN)
        os.close(c.fd)
        c.released = True
        if _STATE["claim"] is c:
            _STATE["claim"] = None


# ---- stop, heartbeat, alerts
def check_stop(out: Path):
    """Raise Stopped when a stop was requested (STOP file in <out>, or a signal)."""
    if _STATE["signal"] is not None:
        raise Stopped(f"signal {_STATE['signal']}")
    if (out / STOP_FILE).exists():
        raise Stopped(f"{out / STOP_FILE}")


def install_stop_handlers(log=print):
    """SIGTERM / SIGINT: release the held claim (previous text back: not a death), drop the heartbeat, exit
    EXIT_STOPPED at once (systemctl stop would follow with SIGKILL after 90 s)."""
    def handler(signum, frame):
        _STATE["signal"] = signum
        c = _STATE["claim"]
        if c is not None:
            release(c, done=False)
        hb = _STATE["heartbeat"]
        if hb is not None:
            hb.close()
        try:
            log(f"WORKER STOPPED (signal {signum}); claim {'released' if c is not None else 'none held'}")
        finally:
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(EXIT_STOPPED)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


class Heartbeat:
    """<out>/heartbeat/<pid>_<start>.json: pid, process start time, unit, worker index, timeout_s, the claim and grasp
    held, the stage. Calling it with a stage (or fields) rewrites the file; without, it touches the file at most every
    BEAT_EVERY_S. scripts/automate/qx_watch.py kills (by pid + start time) a worker whose file is older than
    timeout_s and marks its claim "hang"."""

    def __init__(self, out: Path, index: int, timeout_s: int = HANG_TIMEOUT_S):
        self.dir = out / "heartbeat"
        self.dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        start = _start_time(pid)
        self.path = self.dir / f"{pid}_{start}.json"
        self.info = {"pid": pid, "start": start, "index": index, "unit": own_unit(), "timeout_s": int(timeout_s),
                     "claim": None, "grasp": None, "stage": "start", "time": None}
        self._last = 0.0
        self.gaps: dict = {}           # stage word -> largest time [s] between two file updates while in that stage
        self._write()
        _STATE["heartbeat"] = self
        _STATE["out"] = out

    def _gap(self):
        """Book the time since the last file update to the stage that was current during it (r4-ops-05: the watcher
        sees exactly these gaps as the file's age)."""
        if self._last:
            k = str(self.info.get("stage") or "").split(" ")[0] or "?"
            self.gaps[k] = round(max(self.gaps.get(k, 0.0), time.monotonic() - self._last), 1)

    def pop_gaps(self) -> dict:
        """The largest gaps per stage word since the last call (grasp.json timing.heartbeat_max_gap_s)."""
        g, self.gaps = self.gaps, {}
        return g

    def _write(self):
        self._gap()
        self.info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_atomic(self.path, json.dumps(self.info))
        self._last = time.monotonic()

    def __call__(self, stage: str | None = None, **fields):
        if stage is not None or fields:
            if stage is not None:
                self._gap()                     # the gap before this call belongs to the stage that ends here
                self._last = self._last and time.monotonic()
                self.info["stage"] = stage
            self.info.update(fields)
            self._write()
        elif time.monotonic() - self._last >= BEAT_EVERY_S:
            self._gap()
            try:
                os.utime(self.path)
            except OSError:
                self._write()
            self._last = time.monotonic()

    def close(self):
        self.path.unlink(missing_ok=True)


def alert(out: Path, msg: str):
    """One line in <out>/logs/ALERT (read by the person running the collection)."""
    d = out / "logs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "ALERT", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}: {msg}\n")


def _pre_exit(code: int):
    """app.PRE_EXIT_HOOKS entry (module docstring): run by app.assert_off_forbidden_gpu just before os._exit(code).
    Releases the held claim with its previous text (the exit is not a death of the grasp), drops the heartbeat, alerts,
    and for EXIT_FORBIDDEN_GPU writes <out>/STOP so that the whole collection stops (no restart)."""
    c, hb, out = _STATE["claim"], _STATE["heartbeat"], _STATE["out"]
    if c is not None:
        release(c, done=False)
    if hb is not None:
        hb.close()
    if out is None:
        return
    held = f"claim {c.path} released (not a death)" if c is not None else "no claim held"
    if code == EXIT_FORBIDDEN_GPU:
        msg = f"exit {code}: this worker holds a forbidden context on GPU 0; {held}; STOP written: the collection stops"
        with open(out / STOP_FILE, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    else:
        msg = f"exit {code}: the GPU check could not read nvidia-smi; {held}; the restart loop counts it as a crash"
    alert(out, msg)


def register_pre_exit_hook():
    """Add _pre_exit to app.PRE_EXIT_HOOKS (list of callables taking the exit code; created if app has none yet)."""
    hooks = _app.__dict__.setdefault("PRE_EXIT_HOOKS", [])
    if not any(getattr(h, "__qualname__", None) == "_pre_exit" and getattr(h, "__module__", None) == __name__
               for h in hooks):
        hooks.append(_pre_exit)


register_pre_exit_hook()


# ---- per-assembly accounting: build failures and process deaths (both reversible: move the file away)
def _rows(f: Path) -> list:
    try:
        return json.loads(f.read_text()) if f.exists() else []
    except ValueError:
        return []


def _append_row(d: Path, aid: str, row: dict, key=None) -> list:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        rows = _rows(d / f"{aid}.json")
        if key is None or all(key(r) != key(row) for r in rows):
            rows.append(row)
            _write_atomic(d / f"{aid}.json", json.dumps(rows, indent=1))
    return rows


def _build_failures(out: Path, aid: str) -> list:
    return _rows(out / "build_failures" / f"{aid}.json")


def _record_build_failure(out: Path, aid: str, exc: BaseException, counted: bool) -> int:
    """Append the failure; -> the number of counted failures of the assembly."""
    rows = _append_row(out / "build_failures", aid, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
                                                      "error": f"{type(exc).__name__}: {exc}"[:2000], "counted": counted})
    return sum(bool(r.get("counted", True)) for r in rows)


def _record_death(out: Path, aid: str, grasp: str, old: dict) -> int:
    """A dead attempt found in a claim (once per pid + start + grasp); -> the number of deaths of the assembly."""
    row = {"grasp": grasp, "pid": old["pid"], "start": old["start"], "attempt": old["attempt"], "claimed": old.get("time"),
           "kind": "hang" if old.get("hang") else "crash", "found": time.strftime("%Y-%m-%d %H:%M:%S")}
    return len(_append_row(out / "deaths", aid, row, key=lambda r: (r.get("grasp"), r.get("pid"), r.get("start"))))


def blocked_assemblies(out: Path) -> dict:
    """{assembly id: reason} of the assemblies workers skip (BUILD_FAILURES_MAX counted build failures or DEATHS_MAX
    process deaths); move <out>/build_failures/<id>.json or <out>/deaths/<id>.json away to retry."""
    out_ = {}
    for f in sorted((out / "build_failures").glob("*.json")):
        rows = _rows(f)
        n = sum(bool(r.get("counted", True)) for r in rows)
        if n >= BUILD_FAILURES_MAX:
            out_[f.stem] = f"{n} scene-build failures (last: {rows[-1].get('error', '')[:200]}); file {f}"
    for f in sorted((out / "deaths").glob("*.json")):
        rows = _rows(f)
        if len(rows) >= DEATHS_MAX:
            kinds = [r.get("kind") for r in rows]
            reason = (f"{len(rows)} process deaths over its grasps ({kinds.count('crash')} crash, "
                      f"{kinds.count('hang')} hang); file {f}")
            out_[f.stem] = out_[f.stem] + "; " + reason if f.stem in out_ else reason
    return out_


def run_summary(out: Path, specs: list[GraspSpec]) -> dict:
    """<out>/logs/summary.json: skipped assemblies with reasons, grasps without grasp.json (with whether a live
    worker holds them), trajectory files without grasp.json."""
    blocked = blocked_assemblies(out)
    undone = []
    for s in specs:
        g = out / s.port / s.key
        if (g / "grasp.json").exists():
            continue
        held = False
        if (g / ".claim").exists():
            txt = (g / ".claim").read_text(errors="ignore").strip()
            old = _parse_claim(txt) if txt else None
            held = old is not None and _alive(old["pid"], old["start"])
        undone.append({"grasp": s.key, "skipped": s.port in blocked, "held_by_live_worker": held})
    orphans = sorted(str(p.relative_to(out)) for p in out.glob("*/*_g*/t*.npz") if not (p.parent / "grasp.json").exists())
    doc = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), "grasps": len(specs),
           "done": len(specs) - len(undone), "skipped_assemblies": blocked, "undone_grasps": undone,
           "orphan_trajectory_files": orphans}
    (out / "logs").mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "logs" / "summary.json", json.dumps(doc, indent=1))
    return doc


def prepare(cfg: CollectConfig, out: Path, log=print) -> tuple[list[GraspSpec], dict]:
    """Build the plan, print the assembly status table, freeze or check plan.json / config.json; Refused when the
    config names expected_grasps and the plan has another number of grasps."""
    require_success_rule()
    if cfg.device != C.PHYSICS_DEVICE:
        raise Refused(f"config device {cfg.device!r} differs from WM_AUTOMATE_DEVICE / config.PHYSICS_DEVICE "
                      f"{C.PHYSICS_DEVICE!r} (the frozen switches.env): one physics device per collection")
    specs, table = build(cfg)
    doc = plan_document(cfg, specs, table)
    log(status_text(table))
    counts = doc["split"]["counts"]
    log(counts_text(counts))
    production = not cfg.collect_subset and cfg.expected_grasps is not None
    why = counts_problem(counts, production=production)
    if why:
        if production:                  # train_wm320 would refuse it, or could after t00 refusals
            raise Refused(f"split {SPLIT_VERSION}: {why}; train_wm320.py would or could refuse this collection after "
                          "it ran")
        log(f"WARNING split {SPLIT_VERSION}: {why} (not refused: collect_subset / no expected_grasps)")
    if cfg.expected_grasps is not None and len(specs) != cfg.expected_grasps:
        raise Refused(f"the plan has {len(specs)} grasps, the config expects {cfg.expected_grasps}")
    frozen = freeze(out, cfg, doc)
    return specs, frozen


def _grasp_fields(s: GraspSpec) -> dict:
    """grasp.json fields of a grasp summary written by the worker itself (no trajectory ran): the same identifying
    keys as collect_grasp's, so that readers recognise an AutoMate grasp (simulator, u, assembly_id)."""
    return {"grasp": s.key, "port": s.port, "condition": s.condition, "pitch_deg": s.pitch, "grasp_pos_m": s.grasp_pos,
            "spin_deg": s.spin, "u": s.u, "assembly_id": s.port, "simulator": "automate", "episode_seed": s.seed,
            "seed": s.seed, "switches": C.switches(),
            "trajectories": [], "bytes": 0, "reference_ok": False}


def _close(robot):
    try:
        robot.sc.close()
    except Exception:
        pass


def worker(cfg: CollectConfig, out: Path, index: int = 0, count: int = 1, log=print, beat=None) -> dict:
    """Take grasps of the plan until none is left (claim-based): any number of workers, started or added at any time.
    `index` / `count` only choose where in the plan a worker starts, so that workers begin on different assemblies;
    a worker prefers further grasps of the assembly it has built. Resumable: grasps with grasp.json are skipped.
    Raises Stopped (STOP file / signal), BuildFailed (scene build), Refused (plan). Returns run_summary()."""
    from .robot import Robot
    beat = beat or (lambda stage=None, **kw: None)
    _STATE["out"] = out
    beat("plan")
    specs, frozen = prepare(cfg, out, log)
    stamp = {"config_sha": frozen["config_sha"], "plan_sha": frozen["plan_sha"], "physics_device": cfg.device}
    k0 = (index * len(specs)) // max(count, 1)
    order = specs[k0:] + specs[:k0]
    log(f"worker {index}/{count}: plan {len(specs)} grasps, {sum((out / s.port / s.key / 'grasp.json').exists() for s in specs)} done")
    robot, port = None, None
    while True:
        check_stop(out)
        _disk_check(out, DISK_FREE_MIN_CLAIM, "before a claim")
        blocked = blocked_assemblies(out)
        cands = [s for s in order if not (out / s.port / s.key / "grasp.json").exists() and s.port not in blocked]
        cands.sort(key=lambda s: s.port != port)                    # stable: same assembly first, then plan order
        s, c = None, None
        for cand in cands:
            c = claim(out / cand.port / cand.key)
            if c is not None:
                s = cand
                break
        if s is None:
            break
        gdir = out / s.port / s.key
        beat("claimed", grasp=s.key, claim=str(c.path))
        if c.old is not None:
            n = _record_death(out, s.port, s.key, c.old)
            log(f"{s.key}: attempt {c.old['attempt']} (pid {c.old['pid']}) died ({'hang' if c.old.get('hang') else 'crash'}); "
                f"{n} deaths in {s.port}")
        if c.attempt > 2:
            deaths = [r for r in _rows(out / "deaths" / f"{s.port}.json") if r.get("grasp") == s.key]
            _clean(gdir)
            write_summary(gdir, {**_grasp_fields(s), **stamp, "deaths": deaths,
                                 "error": "the simulator process died twice in this grasp "
                                          f"({', '.join(str(r.get('kind')) for r in deaths) or 'unrecorded'})"})
            release(c, done=True)
            continue
        summ, retries = None, []
        try:
            for tries in (1, 2):
                if port != s.port or robot is None:
                    if robot is not None:
                        beat("close")
                        _close(robot)
                    robot, port = None, None
                    beat(f"build {s.port}")
                    try:
                        robot, port = Robot(s.port, device=cfg.device), s.port
                    except Exception as exc:
                        release(c, done=False)                      # not an attempt of this grasp
                        msg = f"{type(exc).__name__}: {exc}"
                        # the scene still loads USDs from the Isaac nucleus (no local mirror yet): a network or GPU
                        # memory error is never the assembly's fault, whether or not this process built one before
                        counted = not NETWORK_OR_GPU_MEMORY.search(msg)
                        n = _record_build_failure(out, s.port, exc, counted)
                        how = "counted" if counted else "not counted: network or GPU memory"
                        log(f"{s.key}: building the scene failed ({how}; {n} of {BUILD_FAILURES_MAX} counted for "
                            f"{s.port}): {msg}")
                        raise BuildFailed(f"{s.port}: {msg}") from exc
                try:
                    summ = collect_grasp(robot, s, out, cfg, stamp, beat=beat)
                    break
                except SimulatorError as exc:
                    summ = exc.summary
                    retries.append({"try": tries, "error": str(exc), "trace": summ.get("trace")})
                    log(f"{s.key}: {exc} (try {tries} of 2); the scene is closed and rebuilt")
                    beat("close")
                    _close(robot)
                    robot, port = None, None
                except Exception as exc:                            # outside the grasp's own handling: record once
                    summ = {**_grasp_fields(s), "error": f"unhandled in the grasp: {type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc()[-2000:], "memory": _memory(), "pid": os.getpid()}
                    log(f"{s.key}: {summ['error']}; written to grasp.json, the scene is closed and rebuilt")
                    beat("close")
                    _close(robot)
                    robot, port = None, None
                    break
        except (Stopped, Refused):                                  # neither is an attempt of this grasp
            release(c, done=False)
            raise
        if retries:
            summ["retries"] = retries
        pop = getattr(beat, "pop_gaps", None)
        if callable(pop):
            summ.setdefault("timing", {})["heartbeat_max_gap_s"] = pop()
        summ.update(stamp, claim_attempt=c.attempt, process=dict(PROCESS_INFO))
        write_summary(gdir, summ)
        release(c, done=True)
        beat("done", grasp=None, claim=None)
        kinds = " ".join(f"{t['kind'][:3]}:{str(t.get('outcome', '?'))[:8]}" for t in summ["trajectories"])
        mem = summ.get("memory") or {}
        log(f"{s.key} ref={summ.get('reference_ok')} {kinds} {summ.get('error', '')[:100]} {summ.get('seconds')}s "
            f"mem {mem}" + (" WARNING memory.peak above 85 % of MemoryMax" if mem.get("warning") else ""))
    doc = run_summary(out, specs)
    left = [u for u in doc["undone_grasps"] if not u["held_by_live_worker"]]
    if doc["skipped_assemblies"] or left or doc["orphan_trajectory_files"]:
        alert(out, f"worker {index} found no grasp left: {len(doc['skipped_assemblies'])} assemblies skipped, "
                   f"{len(left)} grasps without grasp.json and not held, {len(doc['orphan_trajectory_files'])} trajectory "
                   "files without grasp.json (logs/summary.json)")
    return doc
