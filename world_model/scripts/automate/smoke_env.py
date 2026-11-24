"""Milestone-1 smoke test of the AutoMate scene + adapter (wm.sim_automate). Run on rs2 (GPU 1), only inside our caps,
exactly as the header of scripts/automate/smoke/collect_smoke.sh shows (watcher registration, then in tmux a loop that
tries again every 60 s while the caps refuse the start, one log smoke_env_r3_cpu_<i> per try; r10-ops-01):

    run_on_rs2.sh smoke_env_r3_cpu_<i> scripts/automate/smoke_env.py --assembly 00015 --checks fk,b,g,s,e \
        --out <M>/m1b/smoke_env_r3_cpu

(with QX_SWITCHES=<batch>/switches.env MEMMAX=40G CPUQUOTA=400% GPU_GB=7 TIMEOUT=7200, from the batch's frozen code).
Never start it directly with python (no scope, no MemoryMax, no GPU-1 declaration). --out is required (r9-expert-02:
the old default /mnt/storage3/qingxin/automate/m1 holds the results of 2026-11-21), and a start is refused when --out
already holds a result smoke_<assembly>_*.json, so no earlier result or fatal file is overwritten.

Checks (every threshold is stated in the result next to the measured value). Notation: T_X(t) simulator pose of body
X at time t (env frame), T_hp(t) = T_hand(t)^-1 T_plug(t), angle(R) = |log R|, a = insertion direction (unit),
truth = wm.sim_automate.checks.truth: success = AutoMate's check_plug_inserted_in_socket; depth_left, lateral at the
plug tip centre, axis tilt, spin, full and symmetric angles recorded as information. Grasps outside an assembly's verified box (the offset grasps of b)
are reset with allow_outside=True (recorded in the episode).
  fk  Pinocchio FK vs simulator body poses at 30 joint configurations: max |p_FK - p_sim| and angle(R_FK^T R_sim).
  a   weld rigidity over 20 s (400 control steps) of motion, pushing against the socket face and fast free motion:
      max_t |p(T_hp(t)) - p(T_hp(0))| < 0.05 mm and max_t angle(T_hp(0)^-1 T_hp(t)) < 0.1 deg.
  b   seated pose: the hand driven to H* (from the socket pose and the weld) along a: |depth_left| < 0.5 mm, and
      no interpenetration: checks.penetration (meshes at the simulator poses of env r.env_id, plug points in the
      socket and socket points in the plug; inside by ray parity, or by the winding number for a mesh that is not
      watertight; depth = distance to the surface; chunked) < 0.05 mm. Centre grasp and the offset grasps.
  c   joint PD tracking: max_j |q_j(t0 + 0.4 s) - q*_j| after a step of the targets (joint step and 2 mm hand step).
  d   wall time per 20 Hz control step (free space, in contact), and with a camera frame.
  e   camera: PNGs at the start pose and near the socket; reprojection of the socket opening (rim vertices of
      socket.obj through K and the FK + mount extrinsics) against the opening seen in the depth image (centroid
      distance [px] and IoU); FK camera pose vs Isaac's camera pose; freshness of a frame rendered right after motion.
  f   save / restore: errors right after a restore and max differences after replaying the same 8 commands
      (hand FK [mm], plug [mm], joints [rad]), in free space and in contact.
  g   short insertion: from 10 mm in front of H* along a, push along a at 3 mm/s to H*; final depth_left, lateral,
      tilt (truth) and jam (commanded hand leads the FK hand by > 1.5 mm). Extra: from 2 mm outside the face with
      lateral start offsets 0 / 0.3 / 1.0 mm.
  s   socket randomisation, state restore, inertias and the contact report (round 2):
      s1  reset(seed 0, centre grasp), one step: socket pose read from PhysX = config.socket_pose(0, k = 0) within
          1e-5 m / 1e-5 rad, H_star() = T_socket T_hp^-1;
      s2  reset(seed 1, a second grasp inside the box): forces an "articulation" weld restart; same comparison, and the
          inertia read-back after the restart (scene.inertia_checks: USD inertias restored, + 0.01 exactly once);
      s3  get_state at the seed-0 episode, reset seed 1 (other grasp, other socket), set_state, step: socket_T and
          H_star() equal the seed-0 values;
      s4  per-reset draws: reset(seed 0, centre, socket_index k) for k = 1, 2: each pose = config.socket_pose(0, k)
          and differs from k = 0; Expert.reference_label after k = 1: ok (= completed AND AutoMate success after
          the last step), H* FK vs model;
      s5  contact report vs the contact sensor: plug pressed on the socket face 3 mm off the axis, and seated at H*
          pushed 1 mm further, 10 control steps each: socket_contacts (per shape pair: plug_N, robot_N, penetration,
          shapes) next to the sensor's net plug force F_s. Pass (smoke thresholds), for every row:
              plug_points > 0 whenever F_s > 1 N             (the report delivers plug-socket data; guard_contacts)
              |plug_net_N - F_s| <= max(0.5 N, 0.02 F_s)      (same units and sign: the report's net = the sensor's)
              plug_N >= plug_net_N - 1e-6 N                   (per-pair sum >= net, triangle inequality)
              0 <= penetration_mm < 0.5                       (no guard-level penetration while pressing)
          and the socket shapes seen in contact are listed (one SDF shape: plug_N = plug_net_N). Information, not in
          the pass rule (r8-expert-01): robot_points per row, and robot_pairs_info (rows with robot-socket points, the
          largest hand / finger sensor sum, rows where that sensor is > 1 N with no robot point: guard_contacts
          raises there).
      s6  the guards trip (expert.Execution / Expert._retreat on the real contact report), 3 mm off the axis above
          the socket face: (a) a command pushed 0.5 mm per control step into the face through Execution._step
          (outside rule off) must stop with socket_contact_force_guard (F > 80 N) or socket_penetration_guard
          (> 0.5 mm), with the measured value above its threshold at the trip; (b) the guarded descent leg
          (_retreat with the 1 N outside rule) into the face must stop with plug_socket_contact; (c) the same leg
          without the outside rule must stop with lag (command 1.5 mm ahead of the hand), socket_force or
          socket_penetration.
  e   (round 2 addition) camera registration with the plug hidden (UsdGeom invisible): the opening reprojection of e
      without the plug in front of it. Pass (smoke thresholds): centroid error < 3 px and IoU > 0.7 on the
      640 x 480 stream (reprojection_near_plug_hidden).
GPU 0 rule (app.py): checked by app.launch after the start, after the scene build and after the first rendered frame
(exit 7 / 8 on a violation); the smoke records the reading at the end (gpu_check_end) and the Kit arguments. The
USD layers loaded and their sha256 are recorded (usd_identity). Code identity (r4-expert-06): code_files_sha256
(collect.code_identity) and physics_code_sha256 (collect.physics_code_identity, the identity the milestone-1b batch
results carry), computed at the start and again at the end of the run (physics_code_identity is cached, so its cache is
cleared before the end value is read from the files); code_changed_during_run is True when they differ (then the
results are not evidence for either version). thread_env records the thread-count variables in force.
A smoke_env result counts for the code of a batch only if its physics_code_sha256 equals that batch's (r4-expert-06):
after the milestone-1b batch, run --assembly 00015 --checks fk,b,g,s,e from the batch's frozen code on GPU 1 and accept
it only if s1-s6 and e pass, physics_code_sha256 equals the one in the batch results and code_changed_during_run is
False. The runs of 2026-11-23 00:18 / 00:22 (m1/smoke_r3_env, smoke_r3_env_kit1) predate the current expert and carry
no physics_code_sha256.
  h   compatibility (extra): wm.collect.explore.Explorer.remember_standoff / place, wm.data.kinematics.HandFK,
      wm.collect.branch.FullRecorder and wm.data.recorder._truth_rel run unchanged on the adapter (ManiSkill-only
      imports of those modules stubbed in sys.modules, nothing edited).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.sim_automate.app import launch, shutdown, thread_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--assembly", default="00015")
ap.add_argument("--mount", default=None)
ap.add_argument("--out", required=True, help="a folder without smoke_<assembly>_*.json (no default; r9-expert-02)")
ap.add_argument("--checks", default="fk,a,b,c,d,e,f,g,h,s")
ap.add_argument("--weld", default=None, help="articulation | maximal (default: config.WELD_MODE)")
args = ap.parse_args()
_old = sorted(Path(args.out).glob(f"smoke_{args.assembly}_*.json"))
if _old:                                                     # never overwrite an earlier result (or its fatal file)
    print(f"REFUSED: {args.out} already holds {[f.name for f in _old]}; use a new --out folder", flush=True)
    sys.exit(2)
app = launch(headless=True, cameras=True)

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from wm.sim.geometry import interpolate_T, pose_distance  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402
from wm.sim_automate.robot import Robot, _trans, grasp_kwargs  # noqa: E402

OUT = Path(args.out)
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)
CENTRE = (0.0, 0.0, 0.0)                                     # (clamp deg, dp m, spin deg): AutoMate's grasp
OFFSETS = {"clamp+10deg_dp-3mm": (10.0, -0.003, 0.0),
           "clamp-10deg_dp-5mm_spin+10deg": (-10.0, -0.005, 10.0),
           "clamp+10deg_dp+3mm (fingers expected near the socket face)": (10.0, 0.003, 0.0)}
from wm.sim_automate.collect import code_identity, physics_code_identity  # noqa: E402

R: dict = {"assembly": args.assembly, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "switches": C.switches(),
           "code_files_sha256": code_identity()["files_sha256"], "physics_code_sha256": physics_code_identity(),
           "thread_env": thread_env()}


def reset_g(r: Robot, seed: int, g, **kw) -> dict:
    """Robot.reset with grasp g = (clamp deg, dp m, spin deg) and socket pose kw socket_index (default 0); smoke grasps
    may lie outside the verified box."""
    return r.reset(seed, **grasp_kwargs(g), allow_outside=True, **{"socket_index": 0, **kw})


def ang_deg(Ra, Rb) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra.T @ Rb).as_rotvec())))


def truth(r: Robot) -> dict:
    return K.truth(r)


def T_hp(r: Robot) -> np.ndarray:
    return np.linalg.inv(r.sc.body_T("panda_hand")) @ r.sc.plug_T()


def save():
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / f"smoke_{args.assembly}_{r.mount}_{r.sc.weld_mode}.json").write_text(json.dumps(R, indent=1, default=float))


def back(r: Robot, H: np.ndarray, dist: float, lateral=(0.0, 0.0)) -> np.ndarray:
    """H moved by -dist along a and by `lateral` (m) along the socket's x and y axes."""
    S = r.sc.socket_T()
    T = H.copy()
    T[:3, 3] = H[:3, 3] - dist * r.axis() + lateral[0] * S[:3, 0] + lateral[1] * S[:3, 1]
    return T


def drive_line(r: Robot, T0: np.ndarray, T1: np.ndarray, speed: float, rec=None, lag_stop: float | None = None,
               max_steps: int = 10_000) -> dict:
    """Commanded TCP pose moves on a straight line from T0 to T1 at `speed` [m/s] (rotation at most 2 deg/step),
    one IK target per control step. Stops when the command leads the FK hand by more than lag_stop [m]."""
    dt, dr = pose_distance(T0, T1)
    n = max(1, int(np.ceil(max(dt / (speed * C.CONTROL_DT), dr / np.radians(2.0)))))
    seed, steps, why, cmd = r.qpos(), 0, "done", T0
    for k in range(1, min(n, max_steps) + 1):
        cmd = interpolate_T(T0, T1, k / n)
        q = r._ik(cmd, seed)
        if q is None:
            why = "ik"
            break
        r._command(q, 1)
        seed, steps = q, steps + 1
        if rec is not None:
            rec()
        if lag_stop is not None and np.linalg.norm(cmd[:3, 3] - r.hand_pose()[:3, 3]) > lag_stop:
            why = "blocked"
            break
    return {"steps": steps, "end": why, "q": seed, "cmd": cmd}


def hold(r: Robot, q, n: int, rec=None):
    for _ in range(n):
        r._command(q, 1)
        if rec is not None:
            rec()


# ----------------------------------------------------------------------------------------------- checks
def check_fk(r: Robot) -> dict:
    stage = r.sc.sim.stage
    j = stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_hand/panda_fingertip_centered_joint")
    usd = {"fingertip_joint_localPos0": list(j.GetAttribute("physics:localPos0").Get()) if j.IsValid() else None,
           "fingertip_joint_localRot0": str(j.GetAttribute("physics:localRot0").Get()) if j.IsValid() else None,
           "fingertip_joint_body0": [str(t) for t in j.GetRelationship("physics:body0").GetTargets()] if j.IsValid() else None}
    rng = np.random.default_rng(1)
    err = {b: [0.0, 0.0] for b in ("panda_link7", "panda_hand", "panda_fingertip_centered", "panda_leftfinger")}
    T_hp0 = r.sc.T_hand_plug[0]
    for _ in range(30):
        q = np.r_[np.clip(C.Q_NOMINAL + rng.uniform(-0.4, 0.4, 7), r.kin.lower[:7] + 0.05, r.kin.upper[:7] - 0.05), 0.04, 0.04]
        r.sc.write_state(0, q, r.kin.link("panda_hand", q) @ T_hp0, np.r_[q[:7], 0.0, 0.0])
        r.sc.step(1)
        qm = r.qpos()
        for b in err:
            Tf, Ts = r.kin.link(b, qm), r.sc.body_T(b)
            err[b][0] = max(err[b][0], float(np.linalg.norm(Tf[:3, 3] - Ts[:3, 3]) * 1000))
            err[b][1] = max(err[b][1], ang_deg(Tf[:3, :3], Ts[:3, :3]))
    Tm = np.linalg.inv(r.sc.body_T("panda_hand")) @ r.sc.body_T("panda_fingertip_centered")
    return {"usd": usd, "fingertip_in_hand_sim_m": Tm[:3, 3].tolist(), "fingertip_in_hand_urdf_m": (np.linalg.inv(r.kin.link("panda_hand", r.qpos())) @ r.kin.link("panda_fingertip_centered", r.qpos()))[:3, 3].tolist(),
            "max_err_mm_deg_per_body": err, "configs": 30,
            "rule": "max |p_FK - p_sim| [mm], max angle(R_FK^T R_sim) [deg] over 30 configs (q_nominal +- 0.4 rad)"}


def check_a(r: Robot) -> dict:
    reset_g(r, 0, CENTRE)
    T0 = T_hp(r)
    set_err = (float(np.linalg.norm(T0[:3, 3] - r.sc.T_hand_plug[0][:3, 3]) * 1000), ang_deg(T0[:3, :3], r.sc.T_hand_plug[0][:3, :3]))
    log = {"dp": [], "da": [], "f": [], "phase": []}
    phase = ["start"]

    def rec():
        T = T_hp(r)
        log["dp"].append(float(np.linalg.norm(T[:3, 3] - T0[:3, 3]) * 1000))
        log["da"].append(ang_deg(T0[:3, :3], T[:3, :3]))
        log["f"].append(float(np.linalg.norm(r.sc.contact_force())))
        log["phase"].append(phase[0])
        if len(log["dp"]) % 50 == 0:
            print(f"a: step {len(log['dp'])} {phase[0]} wall {time.time() - ta:.1f} s", flush=True)
    ta = time.time()
    H = r.hand_pose()
    phase[0] = "lateral"                                         # 3 mm across the axis: the plug misses the hole
    o = drive_line(r, H, back(r, H, 0.0, (0.003, 0.0)), 0.01, rec)
    phase[0] = "push_face"                                       # push along a until blocked (lead 1.5 mm)
    Hs = r.hand_pose()
    p = drive_line(r, Hs, back(r, Hs, -0.020), 0.005, rec, lag_stop=0.0015)
    phase[0] = "press_3mm"                                       # press harder: command 3 mm past the blocked hand
    q = r._ik(back(r, r.hand_pose(), -0.003), r.qpos())
    hold(r, q, 60, rec)
    phase[0] = "free_motion"                                     # retreat, then fast random moves with rotations
    print("a: pressed, steps", len(log["dp"]), "push", p["end"], p["steps"], flush=True)
    Hb = r.hand_pose()
    drive_line(r, Hb, back(r, Hb, 0.05), 0.05, rec)
    rng = np.random.default_rng(3)
    Hf, tries, ik_fail = r.hand_pose(), 0, 0
    print("a: free motion from step", len(log["dp"]), flush=True)
    while len(log["dp"]) < 400 and tries < 200:
        tries += 1
        Hc = r.hand_pose()
        Tn = Hf.copy()                                            # random poses within 3 cm / 10 deg of the start
        Tn[:3, 3] += rng.uniform(-0.03, 0.03, 3)
        Tn[:3, :3] = Rotation.from_rotvec(np.radians(rng.uniform(-10, 10, 3))).as_matrix() @ Hf[:3, :3]
        d = drive_line(r, Hc, Tn, 0.1, rec, max_steps=400 - len(log["dp"]))
        ik_fail += d["end"] == "ik"
    ph = np.array(log["phase"])
    by = {k: {"max_dp_mm": float(np.max(np.array(log["dp"])[ph == k])), "max_da_deg": float(np.max(np.array(log["da"])[ph == k])),
              "max_force_N": float(np.max(np.array(log["f"])[ph == k])), "steps": int((ph == k).sum())} for k in dict.fromkeys(ph)}
    mdp, mda = float(np.max(log["dp"])), float(np.max(log["da"]))
    return {"steps": len(log["dp"]), "seconds_sim": len(log["dp"]) * C.CONTROL_DT, "max_dp_mm": mdp, "max_da_deg": mda,
            "pass": bool(mdp < 0.05 and mda < 0.1), "rule": "max_t |dp| < 0.05 mm and max_t angle < 0.1 deg",
            "by_phase": by, "push_end": p["end"], "T_hp0_vs_set_weld_mm_deg": set_err,
            "free_motion_segments": tries, "free_motion_ik_failures": int(ik_fail)}


def _penetration(r: Robot) -> dict:
    """checks.penetration: the one penetration measure (round 1 env-06), env r.env_id."""
    return K.penetration(r)


def check_b(r: Robot) -> dict:
    out = {}
    for name, g in (("centre", CENTRE), *OFFSETS.items()):
        t0 = time.time()
        info = reset_g(r, 0, g)
        t_reset = time.time() - t0
        H = r.H_star()
        L = r.port_spec.pin_length_m
        ok1 = r.move_to(back(r, H, L + 0.003))
        ok2 = r.move_to(H)
        hold(r, r.qpos()[:7] if r._q_cmd is None else r._q_cmd, 20)
        t = truth(r)
        dt, dr = pose_distance(r.hand_pose(), H)
        res = {"reset_s": t_reset, "sim_restarted": info["automate"]["sim_restarted"], "move_ok": [ok1, ok2], **t, "fk_hand_to_H*_mm": dt * 1000, "fk_hand_to_H*_deg": float(np.degrees(dr)),
               "contact_force_N": float(np.linalg.norm(r.sc.contact_force())),
               "finger_socket_force_N": r.sc.finger_socket_force(), **_penetration(r),
               "fingertip_above_socket_face_mm": (r.port_spec.fingertip_z_m - r.port_spec.socket_top_z_m) * 1000}
        res["pass"] = bool(abs(t["depth_left_mm"]) < 0.5 and res["max_penetration_mm"] < 0.05)
        out[name] = res
    out["rule"] = "|depth_left| < 0.5 mm and checks.penetration max_penetration_mm < 0.05 mm"
    return out


def check_c(r: Robot) -> dict:
    reset_g(r, 0, CENTRE)
    r.move_to(back(r, r.hand_pose(), 0.10))                     # 10 cm back from the spawn: free space
    hold(r, r._q_cmd, 10)
    out = {}
    q0 = r.qpos()
    steps = {"joint_step_0.02rad_alternating": q0[:7] + 0.02 * np.array([1, -1, 1, -1, 1, -1, 1]),
             "hand_step_2mm_along_a": None}
    for name in steps:
        q_start = r.qpos()
        tgt = steps[name] if steps[name] is not None else r._ik(back(r, r.hand_pose(), -0.002), q_start)[:7]
        curve, fmax = [], 0.0
        for k in range(8):
            r._command(tgt, 1)
            fmax = max(fmax, float(np.linalg.norm(r.sc.contact_force())), r.sc.finger_socket_force())
            e = np.abs(r.qpos()[:7] - tgt)
            curve.append({"t_s": (k + 1) * C.CONTROL_DT, "max_err_rad": float(e.max()),
                          "hand_err_mm": float(np.linalg.norm(r.kin.tcp(np.r_[tgt, r.qpos()[7:]])[:3, 3] - r.hand_pose()[:3, 3]) * 1000)})
        e = np.abs(r.qpos()[:7] - tgt)
        out[name] = {"step_size_rad": float(np.abs(tgt - q_start[:7]).max()), "err_at_0.4s_rad": e.tolist(),
                     "max_socket_contact_N": fmax,
                     "max_err_at_0.4s_deg": float(np.degrees(e.max())), "hand_err_at_0.4s_mm": curve[-1]["hand_err_mm"],
                     "curve": curve}
        hold(r, tgt, 20)
        out[name]["max_err_at_1.4s_deg"] = float(np.degrees(np.abs(r.qpos()[:7] - tgt).max()))
    return out


def check_d(r: Robot) -> dict:
    import torch
    reset_g(r, 0, CENTRE)
    out = {}

    def timed(fn, n):
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts = np.array(ts[2:])
        return {"median_ms": float(np.median(ts) * 1000), "mean_ms": float(ts.mean() * 1000), "n": len(ts)}
    q = r.qpos()[:7]
    out["control_step_free"] = timed(lambda: (r._command(q, 1), r.qpos(), r.hand_pose()), 60)
    out["control_step_plus_frame_free"] = timed(lambda: (r._command(q, 1), r.frame()), 30)
    out["frame_only"] = timed(lambda: r.frame(), 30)
    out["observe_only_full_640x480"] = timed(lambda: r.observe(), 10)
    H = r.hand_pose()                                              # into contact: pressed on the face, 3 mm off axis
    drive_line(r, H, back(r, H, 0.0, (0.003, 0.0)), 0.01)
    Hs = r.hand_pose()
    drive_line(r, Hs, back(r, Hs, -0.020), 0.005, lag_stop=0.0015)
    qc = r._ik(back(r, r.hand_pose(), -0.001), r.qpos())[:7]
    out["control_step_contact"] = timed(lambda: (r._command(qc, 1), r.qpos(), r.hand_pose()), 60)
    out["contact_force_N"] = float(np.linalg.norm(r.sc.contact_force()))
    out["note"] = "wall time incl. cuda sync; GPU 1 shared with other users' jobs"
    return out


def _save_frame(r: Robot, tag: str) -> dict:
    import cv2
    f = r.frame()
    full = r._last
    cv2.imwrite(str(IMG / f"{tag}_rgb320.png"), f.rgb[..., ::-1])
    cv2.imwrite(str(IMG / f"{tag}_depth320_0.1mm.png"), f.depth)
    cv2.imwrite(str(IMG / f"{tag}_rgb640x480.png"), full["rgb"][..., ::-1])
    d = full["depth"]
    v = d[d > 0]
    lo, hi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0, 1)
    vis = np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    vis[d <= 0] = 0
    cv2.imwrite(str(IMG / f"{tag}_depth640x480_vis.png"), vis)
    cv2.imwrite(str(IMG / f"{tag}_seg320.png"), (f.seg * 60).astype(np.uint8))
    return {"files": sorted(str(p) for p in IMG.glob(f"{tag}_*.png")), "depth_range_m": [float(lo), float(hi)],
            "seg_class_pixels": {C.SEG_CLASSES[k]: int((f.seg == k).sum()) for k in range(len(C.SEG_CLASSES))}}


def _reprojection(r: Robot, tag: str) -> dict:
    """Socket opening: rim vertices projected with K and FK extrinsics vs the opening in the rendered depth."""
    import cv2
    import trimesh
    sock = trimesh.load(A.fetch(f"{r.assembly_id}/socket.obj"), force="mesh")
    plug = trimesh.load(A.fetch(f"{r.assembly_id}/plug.obj"), force="mesh")
    s = r.port_spec
    band = plug.vertices[(plug.vertices[:, 2] > s.plug_bottom_z_m + 5e-4) & (plug.vertices[:, 2] < s.socket_top_z_m - 5e-4)]
    c = band[:, :2].mean(0)
    r_plug = np.max(np.linalg.norm(band[:, :2] - c, axis=1))
    v = sock.vertices                                             # opening = inner boundary of the socket face:
    top = v[(v[:, 2] > s.socket_top_z_m - 1e-4) & (np.linalg.norm(v[:, :2] - c, axis=1) < r_plug + 5e-3)]
    rad = np.linalg.norm(top[:, :2] - c, axis=1)                  # per 10-deg sector the face vertex nearest the axis
    ang = np.floor((np.arctan2(top[:, 1] - c[1], top[:, 0] - c[0]) + np.pi) / np.radians(10)).astype(int)
    rim = np.array([top[ang == k][np.argmin(rad[ang == k])] for k in np.unique(ang)])
    f = r._render()
    K, E = C.intrinsics(), r.cam_T_world()
    Pw = (r.sc.socket_T() @ np.c_[rim, np.ones(len(rim))].T).T[:, :3]
    Pc = (E[:3, :3] @ Pw.T).T + E[:3, 3]
    uv = (K @ (Pc / Pc[:, 2:3]).T).T[:, :2]
    def px(Pw):
        Pc = E[:3, :3] @ Pw + E[:3, 3]
        return (K @ (Pc / Pc[2]))[:2]
    r0, r1, c0, c1 = C.CROP
    tip = px((r.sc.plug_T() @ np.r_[c, s.plug_bottom_z_m, 1.0])[:3])
    hole = px((r.sc.socket_T() @ np.r_[c, s.socket_top_z_m, 1.0])[:3])
    inside = lambda uv: bool(c0 <= uv[0] < c1 and r0 <= uv[1] < r1)
    crop = {"plug_tip_px": tip.tolist(), "hole_centre_px": hole.tolist(), "crop_rows_cols": [r0, r1, c0, c1],
            "plug_tip_in_crop": inside(tip), "hole_centre_in_crop": inside(hole)}
    pred = np.zeros(f["depth"].shape, np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.fillConvexPoly(pred, np.round(hull).astype(np.int32), 1)
    x0, y0, w, h = cv2.boundingRect(np.round(hull).astype(np.int32))
    pad = max(w, h)
    win = np.zeros_like(pred)
    win[max(0, y0 - pad):y0 + h + pad, max(0, x0 - pad):x0 + w + pad] = 1
    ring = cv2.dilate(pred, np.ones((9, 9), np.uint8), iterations=3) - cv2.dilate(pred, np.ones((5, 5), np.uint8))
    d = f["depth"]
    dface = float(np.median(d[(ring > 0) & (d > 0)]))
    seg = r._seg_classes(f["seg_ids"], f["seg_labels"]) if "seg_ids" in f else np.zeros(d.shape, np.uint8)
    opening = ((d > dface + 1e-3) | (d == 0)) & (seg != 1) & (seg != 2) & (win > 0)
    n, lab = cv2.connectedComponents(opening.astype(np.uint8))
    cy, cx = np.argwhere(pred > 0).mean(0)
    best = max(range(1, n), key=lambda k: ((lab == k) & (pred > 0)).sum(), default=None)
    if best is None:
        return {"error": "no opening found in the depth image", "rim_points": int(len(rim)), **crop}
    meas = lab == best
    my, mx = np.argwhere(meas).mean(0)
    vis = cv2.cvtColor(f["rgb"], cv2.COLOR_RGB2BGR).copy()
    cv2.polylines(vis, [np.round(hull).astype(np.int32)], True, (0, 255, 0), 1)
    vis[meas] = (0.5 * vis[meas] + [0, 0, 127]).astype(np.uint8)
    cv2.imwrite(str(IMG / f"{tag}_reprojection_opening.png"), vis)
    inter, union = float((meas & (pred > 0)).sum()), float((meas | (pred > 0)).sum())
    Tsim = f["cam_T_cv_sim"]
    Tfk = np.linalg.inv(E)
    cam_cmp = {} if Tsim is None else {"fk_vs_isaac_camera_mm": float(np.linalg.norm(Tsim[:3, 3] - Tfk[:3, 3]) * 1000),
                                       "fk_vs_isaac_camera_deg": ang_deg(Tsim[:3, :3], Tfk[:3, :3])}
    return {"centroid_error_px": float(np.hypot(mx - cx, my - cy)), "pred_centroid_px": [float(cx), float(cy)], **crop,
            "meas_centroid_px": [float(mx), float(my)], "iou": inter / max(union, 1.0), "pred_area_px": int(pred.sum()),
            "meas_area_px": int(meas.sum()), "rim_points": int(len(rim)), "face_depth_m": dface,
            **cam_cmp, "K_ours": K.tolist(), "K_isaac": f["K_sim"].tolist(), "image": str(IMG / f"{tag}_reprojection_opening.png")}


def check_e(r: Robot) -> dict:
    reset_g(r, 0, CENTRE)
    out = {"start": _save_frame(r, "start"), "reprojection_start": _reprojection(r, "start")}
    H = r.H_star()
    r.move_to(back(r, H, r.port_spec.pin_length_m + 0.003))
    hold(r, r._q_cmd, 10)
    out["near"] = _save_frame(r, "near_gap3mm")
    out["reprojection_near"] = _reprojection(r, "near_gap3mm")
    from pxr import UsdGeom                                       # registration without the plug in front of the hole
    plug_prim = UsdGeom.Imageable(r.sc.sim.stage.GetPrimAtPath(f"/World/envs/env_{r.env_id}/HeldAsset"))
    plug_prim.MakeInvisible()
    try:
        out["near_plug_hidden"] = _save_frame(r, "near_gap3mm_plug_hidden")
        out["reprojection_near_plug_hidden"] = _reprojection(r, "near_gap3mm_plug_hidden")
    finally:
        plug_prim.MakeVisible()
    rp = out["reprojection_near_plug_hidden"]
    out["pass"] = bool("error" not in rp and rp["centroid_error_px"] < 3.0 and rp["iou"] > 0.7)
    out["rule"] = "plug hidden, 3 mm before the entrance: opening centroid error < 3 px and IoU > 0.7 (640 x 480)"
    d0 = r._render()["depth"].copy()
    d1 = r._render()["depth"].copy()
    Hn = r.hand_pose()
    r.move_to(back(r, Hn, 0.005))
    hold(r, r._q_cmd, 4)
    m1 = r._render()["depth"].copy()
    m2 = r._render()["depth"].copy()
    out["freshness"] = {"same_pose_two_renders_max_abs_mm": float(np.abs(d1 - d0).max() * 1000),
                        "after_5mm_move_first_vs_second_render_max_abs_mm": float(np.abs(m2 - m1).max() * 1000),
                        "after_5mm_move_first_render_vs_before_median_abs_mm": float(np.median(np.abs(m1 - d1)) * 1000)}
    return out


def _replay(r: Robot, cmds: list) -> dict:
    hands, plugs, qs = [], [], []
    for q in cmds:
        r._command(q, 1)
        hands.append(r.hand_pose()[:3, 3].copy())
        plugs.append(r.sc.plug_T()[:3, 3].copy())
        qs.append(r.qpos()[:7].copy())
    return {"hand": np.array(hands), "plug": np.array(plugs), "q": np.array(qs)}


def _restore_case(r: Robot, name: str) -> dict:
    H = r.hand_pose()
    seed = r.qpos()
    cmds = []
    for k in range(1, 9):                                         # 8 commands: 0.25 mm along a + 0.1 mm across each
        T = back(r, H, -0.00025 * k, (0.0001 * k, 0.0))
        seed = r._ik(T, seed)
        cmds.append(seed[:7].copy())
    snap = r.base.get_state_dict()
    q_s, hand_s, plug_s = r.qpos().copy(), r.hand_pose().copy(), r.sc.plug_T().copy()
    first = _replay(r, cmds)
    res = {"case": name, "force_at_snapshot_N": None}
    worst = {"restore_q_rad": 0.0, "restore_hand_mm": 0.0, "restore_plug_mm": 0.0,
             "replay_hand_mm": 0.0, "replay_plug_mm": 0.0, "replay_q_rad": 0.0}
    for _ in range(2):
        r.base.set_state_dict(snap)
        worst["restore_q_rad"] = max(worst["restore_q_rad"], float(np.abs(r.qpos() - q_s).max()))
        worst["restore_hand_mm"] = max(worst["restore_hand_mm"], float(np.linalg.norm(r.hand_pose()[:3, 3] - hand_s[:3, 3]) * 1000))
        worst["restore_plug_mm"] = max(worst["restore_plug_mm"], float(np.linalg.norm(r.sc.plug_T()[:3, 3] - plug_s[:3, 3]) * 1000))
        again = _replay(r, cmds)
        worst["replay_hand_mm"] = max(worst["replay_hand_mm"], float(np.abs(again["hand"] - first["hand"]).max() * 1000))
        worst["replay_plug_mm"] = max(worst["replay_plug_mm"], float(np.abs(again["plug"] - first["plug"]).max() * 1000))
        worst["replay_q_rad"] = max(worst["replay_q_rad"], float(np.abs(again["q"] - first["q"]).max()))
    res.update(worst)
    res["hand_moved_mm"] = float(np.linalg.norm(first["hand"][-1] - hand_s[:3, 3]) * 1000)
    return res


def check_f(r: Robot) -> dict:
    reset_g(r, 0, CENTRE)
    free = _restore_case(r, "free space (spawn, 12 mm gap)")
    H = r.hand_pose()
    drive_line(r, H, back(r, H, 0.0, (0.003, 0.0)), 0.01)
    Hs = r.hand_pose()
    drive_line(r, Hs, back(r, Hs, -0.020), 0.005, lag_stop=0.0015)
    f0 = float(np.linalg.norm(r.sc.contact_force()))
    contact = _restore_case(r, "contact (pressed on the socket face 3 mm off axis, command 1.5 mm ahead)")
    contact["force_at_snapshot_N"] = f0
    reset_g(r, 0, CENTRE)
    Hst = r.H_star()
    r.move_to(back(r, Hst, 0.006))
    inserted = _restore_case(r, "inside the hole (6 mm before H*, pushing deeper)")
    return {"free": free, "contact": contact, "inserted": inserted,
            "rule": "restore_*: error right after set_state_dict vs the saved state; replay_*: max over 8 steps and "
                    "2 restores of |x_replay - x_first|"}


def _insertion(r: Robot, start_back: float, lateral=(0.0, 0.0)) -> dict:
    reset_g(r, 0, CENTRE)
    H = r.H_star()
    ok = r.move_to(back(r, H, start_back, lateral))
    hold(r, r._q_cmd, 10)
    t0 = truth(r)
    Hs = r.hand_pose()
    fmax = [0.0, 0.0]

    def rec():
        fmax[0] = max(fmax[0], float(np.linalg.norm(r.sc.contact_force())))
        fmax[1] = max(fmax[1], r.sc.finger_socket_force())
    target = back(r, H, 0.0, lateral)
    p = drive_line(r, Hs, target, 0.003, rec, lag_stop=0.0015)
    hold(r, p["q"], 20, rec)
    t = truth(r)
    dt, dr = pose_distance(r.hand_pose(), H)
    return {"start_back_mm": start_back * 1000, "start_lateral_mm": [v * 1000 for v in lateral], "placed": ok,
            "truth_at_start": t0, "final": t, "jam": p["end"] == "blocked", "push_end": p["end"], "push_steps": p["steps"],
            "fk_hand_to_H*_mm": dt * 1000, "fk_hand_to_H*_deg": float(np.degrees(dr)), "max_contact_force_N": fmax[0],
            "max_finger_socket_force_N": fmax[1]}


def check_g(r: Robot) -> dict:
    out = {"from_10mm_before_H*": _insertion(r, 0.010)}
    L = r.port_spec.pin_length_m
    for lat in (0.0, 0.0003, 0.001):
        out[f"extra_tip_2mm_outside_lateral_{lat * 1000:.1f}mm"] = _insertion(r, L + 0.002, (lat, 0.0))
    return out


def _socket_cmp(r: Robot, seed: int, k: int = 0) -> dict:
    T = C.socket_pose(seed, r.mount, k)[0]
    got = r.sc.socket_T()
    Hm = T @ np.linalg.inv(r.T_tcp_plug)
    dH, rH = pose_distance(r.H_star(), Hm)
    return {"socket_err_mm": float(np.linalg.norm(got[:3, 3] - T[:3, 3]) * 1e3), "socket_err_deg": ang_deg(got[:3, :3], T[:3, :3]),
            "H_star_vs_formula_mm": dH * 1e3, "H_star_vs_formula_deg": float(np.degrees(rH)),
            "socket_offset_mm": [round(v * 1e3, 3) for v in (T[:3, 3] - np.asarray(C.SOCKET_POS[r.mount]))]}


def _second_grasp(r: Robot) -> tuple:
    """A grasp inside the verified box, different from the centre (half-way to the upper / lower faces)."""
    b = G.load_ranges(r.assembly_id).get("box")
    if b is None:
        return (5.0, -0.002, 5.0)
    mid = lambda lo, hi, f: lo + f * (hi - lo)
    return (mid(*b["dtheta_deg"], 0.75), mid(*b["dp_m"], 0.25), mid(*b["dpsi_deg"], 0.75))


def _contact_row(r: Robot) -> dict:
    c = r.sc.socket_contacts()
    return {**{k: c[k] for k in ("plug_N", "robot_N", "total_N", "plug_points_N", "robot_points_N", "plug_net_N",
                                 "penetration_mm", "n_pairs", "n_points", "plug_points", "robot_points",
                                 "socket_shapes")},
            "sensor_plug_net_N": r.sc.plug_socket_force_n(), "sensor_finger_sum_N": r.sc.finger_socket_force_sum()}


def _s5_row_ok(x: dict) -> dict:
    """The s5 pass rule for one row (module docstring)."""
    fs = x["sensor_plug_net_N"]
    return {"report_has_plug_points": bool(x["plug_points"] > 0 or fs <= 1.0),
            "net_matches_sensor": bool(abs(x["plug_net_N"] - fs) <= max(0.5, 0.02 * fs)),
            "pair_sum_ge_net": bool(x["plug_N"] >= x["plug_net_N"] - 1e-6),
            "penetration_ok": bool(0.0 <= x["penetration_mm"] < 0.5)}


def _s6(r: Robot) -> dict:
    """Guard trips (module docstring s6)."""
    from wm.sim_automate import expert as EX
    out = {}

    def above_face():
        reset_g(r, 0, CENTRE)
        H = r.hand_pose()
        drive_line(r, H, back(r, H, 0.0, (0.003, 0.0)), 0.01)
        return r.hand_pose()
    Hs = above_face()                                            # (a) Execution guards, 0.5 mm per step into the face
    ex = EX.Execution(r, EX.PlugGeometry(r), None, np.random.default_rng(0), max_steps=400)
    ex.outside = False
    seed, reason, steps, peak_f, peak_p = r.qpos(), None, 0, 0.0, 0.0
    for k in range(1, 301):
        q = r._ik(back(r, Hs, -0.0005 * k), seed)
        if q is None:
            reason = "ik"
            break
        seed = q
        try:
            ex._step(q[:7])
        except EX.ExpertFailure as exc:
            reason = str(exc)
        steps = k
        peak_f, peak_p = max(peak_f, ex.m["total_force"]), max(peak_p, ex.m["penetration_mm"])
        if reason:
            break
    trip = {"reason": reason, "steps": steps, "total_force_at_trip_N": ex.m["total_force"],
            "penetration_at_trip_mm": ex.m["penetration_mm"], "plug_N_at_trip": ex.m["force"],
            "sensor_plug_N_at_trip": ex.m["sensor_plug_N"], "peak_total_N": peak_f, "peak_penetration_mm": peak_p}
    trip["pass"] = bool((reason == "socket_contact_force_guard" and ex.m["total_force"] > EX.MAX_FORCE_N)
                        or (reason == "socket_penetration_guard" and ex.m["penetration_mm"] > EX.MAX_PENETRATION_MM))
    out["a_execution_push_into_face"] = trip
    E = EX.Expert(r)
    Hs = above_face()                                            # (b) guarded descent with the 1 N outside rule
    leg = E._retreat(back(r, Hs, -0.020), plug_contact_max_n=EX.OUTSIDE_CONTACT_N)
    out["b_descent_outside_rule"] = {**leg, "pass": leg["reason"] == "plug_socket_contact"}
    Hs = above_face()                                            # (c) the same leg without the outside rule
    leg = E._retreat(back(r, Hs, -0.020))
    out["c_descent_no_outside_rule"] = {**leg, "pass": leg["reason"] in ("lag", "socket_force", "socket_penetration")}
    out["pass"] = bool(all(v["pass"] for v in out.values()))
    return out


def check_s(r: Robot) -> dict:
    tol = {"socket_mm": 1e-2, "socket_deg": np.degrees(1e-5), "H_mm": 1e-3, "H_deg": 1e-4}
    out = {"rule": "socket pose read from PhysX after a step = config.socket_pose within 1e-5 m / 1e-5 rad; H_star() = "
                   "T_socket T_hp^-1; restore gives the saved socket and H*; inertias USD + 0.01 once after restarts"}
    gB = _second_grasp(r)
    # s1
    reset_g(r, 0, CENTRE)
    r._command(r._q_cmd, 1)
    s1 = _socket_cmp(r, 0)
    s1["pass"] = bool(s1["socket_err_mm"] < tol["socket_mm"] and s1["socket_err_deg"] < tol["socket_deg"]
                      and s1["H_star_vs_formula_mm"] < tol["H_mm"])
    out["s1"] = s1
    st0, T0, H0 = r.get_state(), r.sc.socket_T().copy(), r.H_star().copy()
    # s2
    n_checks = len(r.sc.inertia_checks)
    info = reset_g(r, 1, gB)
    r._command(r._q_cmd, 1)
    s2 = {**_socket_cmp(r, 1), "grasp": list(gB), "sim_restarted": info["automate"]["sim_restarted"],
          "inertia_checks_since_s1": r.sc.inertia_checks[n_checks:], "regrasps": r.sc.regrasps}
    s2["pass"] = bool(s2["socket_err_mm"] < tol["socket_mm"] and s2["socket_err_deg"] < tol["socket_deg"]
                      and s2["H_star_vs_formula_mm"] < tol["H_mm"] and s2["sim_restarted"]
                      and all(c["max_err_kg_m2"] < 1e-6 for c in s2["inertia_checks_since_s1"]))
    out["s2"] = s2
    # s3
    r.set_state(st0)
    r._command(r._q_cmd, 1)
    T3 = r.sc.socket_T()
    s3 = {"socket_vs_saved_mm": float(np.linalg.norm(T3[:3, 3] - T0[:3, 3]) * 1e3), "socket_vs_saved_deg": ang_deg(T3[:3, :3], T0[:3, :3]),
          "H_star_vs_saved_mm": float(np.linalg.norm(r.H_star()[:3, 3] - H0[:3, 3]) * 1e3),
          "H_star_vs_saved_deg": ang_deg(r.H_star()[:3, :3], H0[:3, :3]), "seed1_socket_offset_mm": s2["socket_offset_mm"],
          "seed0_socket_offset_mm": s1["socket_offset_mm"], "inertia_checks": r.sc.inertia_checks}
    s3["pass"] = bool(s3["socket_vs_saved_mm"] < tol["socket_mm"] and s3["socket_vs_saved_deg"] < tol["socket_deg"]
                      and s3["H_star_vs_saved_mm"] < tol["socket_mm"])
    out["s3"] = s3
    # s4
    from wm.sim_automate.expert import Expert
    s4 = {"draw": C.SOCKET_DRAW, "poses": {}}
    for k in (1, 2):
        reset_g(r, 0, CENTRE, socket_index=k)
        r._command(r._q_cmd, 1)
        s4["poses"][k] = _socket_cmp(r, 0, k)
        if k == 1:
            lab = Expert(r).reference_label(np.random.SeedSequence(0))
            s4["reference_label_k1"] = {key: lab[key] for key in ("ok", "completed", "success_end", "success_any",
                                                                   "first_success_step", "steps",
                                                                   "H_star_fk_vs_model_mm_deg", "failure", "peak")}
            s4["reference_label_k1"]["truth"] = {key: lab["truth"][key] for key in ("success", "height_mm", "keypoint_dist_mm",
                                                                                   "depth_left_mm", "lateral_mm", "tilt_deg")}
    offs = [tuple(v["socket_offset_mm"]) for v in s4["poses"].values()] + [tuple(s1["socket_offset_mm"])]
    s4["pass"] = bool(all(v["socket_err_mm"] < tol["socket_mm"] and v["H_star_vs_formula_mm"] < tol["H_mm"]
                          for v in s4["poses"].values()) and len(set(offs)) == (3 if C.SOCKET_DRAW == "per_reset" else 1)
                      and s4["reference_label_k1"]["ok"])
    out["s4"] = s4
    # s5
    reset_g(r, 0, CENTRE)
    s5 = {"free_at_spawn": _contact_row(r)}
    H = r.hand_pose()
    drive_line(r, H, back(r, H, 0.0, (0.003, 0.0)), 0.01)
    Hs = r.hand_pose()
    drive_line(r, Hs, back(r, Hs, -0.020), 0.005, lag_stop=0.0015)
    q = r._ik(back(r, r.hand_pose(), -0.002), r.qpos())
    rows = []
    for _ in range(10):
        r._command(q[:7], 1)
        rows.append(_contact_row(r))
    s5["pressed_on_face_3mm_off_axis"] = rows[-1]
    s5["pressed_plug_N_vs_sensor_max_abs_diff"] = float(max(abs(x["plug_N"] - x["sensor_plug_net_N"]) for x in rows))
    s5["_pressed_rows"] = rows
    reset_g(r, 0, CENTRE)
    Hst = r.H_star()
    r.move_to(back(r, Hst, r.port_spec.pin_length_m + 0.003))
    drive_line(r, r.hand_pose(), Hst, 0.003, lag_stop=0.0015)
    q = r._ik(back(r, Hst, -0.001), r.qpos())
    rows = []
    for _ in range(10):
        r._command(q[:7], 1)
        rows.append(_contact_row(r))
    s5["seated_pushed_1mm"] = rows[-1]
    s5["seated_plug_N_vs_sensor_max_abs_diff"] = float(max(abs(x["plug_N"] - x["sensor_plug_net_N"]) for x in rows))
    all_rows = s5.pop("_pressed_rows") + rows
    checks = [_s5_row_ok(x) for x in all_rows]
    s5["row_checks_failed"] = {k: sum(1 for c in checks if not c[k]) for k in checks[0]}
    s5["max_net_vs_sensor_N"] = float(max(abs(x["plug_net_N"] - x["sensor_plug_net_N"]) for x in all_rows))
    s5["socket_shapes_seen"] = sorted({sh for x in all_rows for sh in x["socket_shapes"]})
    # information (r8-expert-01; not in the pass rule): robot-socket pairs in the report next to the hand / finger
    # sensor; guard_contacts raises ContactReportError when robot_points = 0 while sensor_finger_sum_N > 1 N
    s5["robot_pairs_info"] = {
        "rows_with_robot_points": sum(1 for x in all_rows if x["robot_points"] > 0), "rows": len(all_rows),
        "max_robot_points": max(x["robot_points"] for x in all_rows),
        "max_sensor_finger_sum_N": float(max(x["sensor_finger_sum_N"] for x in all_rows)),
        "rows_finger_sensor_gt_1N_without_robot_points": sum(1 for x in all_rows if x["robot_points"] == 0
                                                             and x["sensor_finger_sum_N"] > 1.0),
        "max_abs_robot_N_minus_finger_sensor_N": float(max(abs(x["robot_N"] - x["sensor_finger_sum_N"]) for x in all_rows)),
        "note": ("earlier evidence (smoke_r3_env, 00015, seated_pushed_1mm): the report had 3 pairs and 62 points, 50 "
                 "of them plug points, so 2 robot-socket pairs (12 points) at 0 N; a non-zero robot force against the "
                 "finger sensor is not measured here")}
    s5["pass"] = bool(all(all(c.values()) for c in checks))
    s5["rule"] = ("every row: plug_points > 0 if sensor > 1 N; |plug_net_N - sensor| <= max(0.5 N, 0.02 sensor); "
                  "plug_N >= plug_net_N; 0 <= penetration < 0.5 mm")
    s5["note"] = ("plug_N = per shape pair (Chongyu's rule); with one socket shape plug_N = plug_net_N = the sensor's net "
                  "force; plug_points_N = sum of point magnitudes (upper bound)")
    out["s5"] = s5
    try:
        out["s6"] = _s6(r)
    except Exception as exc:
        out["s6"] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:], "pass": False}
    return out


def check_h(r: Robot) -> dict:
    """Run unchanged wm.collect / wm.data code on the adapter; ManiSkill-only imports stubbed."""
    from wm.sim_automate import robot as AR
    pose_mod = types.ModuleType("stub_scripted")
    pose_mod._copy_pose = lambda p: p                              # our Pose objects already are copies
    for name in ("mani_skill", "mani_skill.examples", "mani_skill.examples.motionplanning",
                 "mani_skill.examples.motionplanning.panda", "mani_skill.examples.motionplanning.panda.solutions",
                 "mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted"] = pose_mod
    envrobot = types.ModuleType("wm.env.robot")
    envrobot.Robot, envrobot.Frame = AR.Robot, AR.Frame
    envrobot.CAM_SCALE, envrobot.CROP, envrobot.DEPTH_MIN_M, envrobot.DEPTH_UNIT_M = C.CAM_SCALE, C.CROP, C.DEPTH_MIN_M, C.DEPTH_UNIT_M
    sys.modules["wm.env.robot"] = envrobot
    expert = types.ModuleType("wm.env.expert")
    expert.Expert = lambda robot: types.SimpleNamespace(robot=robot, _planned_move=lambda *a, **k: False)
    sys.modules["wm.env.expert"] = expert
    ea = types.ModuleType("wm.sim.env_adapter")
    ea._np = lambda x: np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)
    sys.modules["wm.sim.env_adapter"] = ea
    from wm.collect import branch as B
    from wm.collect import explore as X
    from wm.data.kinematics import HandFK
    from wm.data.recorder import _truth_rel
    out = {}
    reset_g(r, 0, CENTRE)
    H = r.H_star()
    r.move_to(back(r, H, r.port_spec.pin_length_m + 0.003))
    r.move_to(H)
    out["truth_rel_at_H*"] = _truth_rel(r).tolist()
    fk = HandFK(r)
    out["HandFK_vs_hand_pose_mm"] = float(np.abs(fk(r.qpos()[None])[0] - r.hand_pose()).max() * 1000)
    with np.errstate(all="ignore"):
        F = X.frame(r.axis())
    out["explore.frame(axis)"] = {"axis": r.axis().tolist(), "finite": bool(np.isfinite(F).all())}
    if not np.isfinite(F).all():
        out["note"] = "explore.frame() is undefined for a vertical insertion axis: Explorer/Brancher not run"
        rec = B.FullRecorder(r)
        rec.begin()
        q = r.qpos()[:7]
        for _ in range(8):
            r._command(q, 1)
            rec.step(q, q)
        out["FullRecorder"] = {k: list(np.shape(v)) for k, v in rec.end().items()}
        out["FullRecorder_frame_rgb_shape"] = list(rec.frame.rgb.shape)
        return out
    ex = B.Brancher(r, r.hand_pose(), r.axis())
    out["remember_standoff"] = bool(ex.remember_standoff())
    rng = np.random.default_rng(0)
    stages = []
    for _ in range(3):
        st = X.sample_start(rng, near=True)
        stages.append({"start": st, "stage": ex.place(st), "start_error_mm": getattr(ex, "start_error_mm", None)})
    out["Explorer.place"] = stages
    rec = B.FullRecorder(r)
    rec.begin()
    q = r.qpos()[:7]
    for _ in range(8):
        r._command(q, 1)
        rec.step(q, q)
    arr = rec.end()
    out["FullRecorder"] = {k: list(np.shape(v)) for k, v in arr.items()}
    out["FullRecorder_frame_rgb_shape"] = list(rec.frame.rgb.shape)
    ex.snapshot()
    out["Brancher.restore_dev_mm_deg"] = ex.restore()
    out["fk_state"] = ex.fk_state()
    return out


r = None
try:
    t0 = time.time()
    r = Robot(args.assembly, mount=args.mount, weld=args.weld)
    R["mount"], R["weld"] = r.mount, r.sc.weld_mode
    R["build_s"] = time.time() - t0
    R["port_spec"] = r.port_spec.as_dict()
    t0 = time.time()
    R["reset_info"] = reset_g(r, 0, CENTRE)
    R["reset_s"] = time.time() - t0
    R["joint_names"] = list(r.sc.robot.joint_names)
    R["drive"] = {"stiffness": r.sc.robot.data.joint_stiffness[0].tolist(), "damping": r.sc.robot.data.joint_damping[0].tolist(),
                  "effort_limit": r.sc.robot.data.joint_effort_limits[0].tolist()}
    from wm.sim_automate import app as APP
    R["kit_args"] = APP.kit_args()
    try:
        R["usd_identity"] = r.sc.usd_identity()
    except Exception as exc:
        R["usd_identity"] = {"error": f"{type(exc).__name__}: {exc}"}
    save()
    for name in args.checks.split(","):
        t0 = time.time()
        try:
            R[name] = globals()[f"check_{name}"](r)
        except Exception as exc:
            R[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
        R[name + "_wall_s"] = time.time() - t0
        print(name, "done", round(time.time() - t0, 1), "s", flush=True)
        save()
    from wm.sim_automate import app as APP
    R["gpu_check_end"] = APP.assert_off_forbidden_gpu("smoke end")
except Exception as exc:
    R["fatal"] = f"{type(exc).__name__}: {exc}"
    R["trace"] = traceback.format_exc()[-4000:]
    print(R["trace"], flush=True)
finally:
    try:
        physics_code_identity.cache_clear()                      # lru_cache: re-read the files for the end value
        end_code, end_phys = code_identity()["files_sha256"], physics_code_identity()
        R.update(code_files_sha256_end=end_code, physics_code_sha256_end=end_phys,
                 code_changed_during_run=bool(end_code != R["code_files_sha256"] or end_phys != R["physics_code_sha256"]))
    except Exception as exc:                                     # recorded; never hides the results
        R["code_identity_end_error"] = f"{type(exc).__name__}: {exc}"
    if r is not None:
        save()
    else:
        (OUT / f"smoke_{args.assembly}_fatal.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: v for k, v in R.items() if k not in ("reset_info",)}, default=float)[:20000], flush=True)
    shutdown(0)
