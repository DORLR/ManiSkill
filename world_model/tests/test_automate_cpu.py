"""AutoMate collection code on a CPU, without Isaac (the scene module is replaced by a stand-in where a module needs
it; no simulator runs):
  1 grasp box: always the verified ranges box (never the base box), box inside the base box asserted; the gate
    pass_ii AND ((iii) skipped OR pass_iii) on results whose provenance (ranges / grasp.py sha256, PhysX device,
    switches) equals the current one; a crash file in one folder falls back to another; a result without provenance
    is excluded as stale
  2 penetration: ray-test (watertight) or winding-number inside + unsigned distance, chunked, both ways; a mesh with
    inconsistent winding is repaired before the winding number
  5 reset API: keyword-only grasp without defaults, positional calls raise, socket_index required under per_reset,
    reset_from_meta; state save / restore carries the grasp
  7 contact force: Chongyu's rule per shape pair (points of a pair add as vectors, pairs as magnitudes)
 11 frozen plan: config.json / plan.json written once; a changed input file or config refuses
 13 random streams: SeedSequence spawn keys, separate placement and expert streams, no cross-assembly overlap
 15 zero-width ranges: u = None, flagged in plan.json, never held out
 22 online guard: AutoMate grasps with condition "unassigned" / missing refuse; ManiSkill grasps and manifest entries
    unchanged
 25 socket pose: AutoMate's randomisation (+-5 cm, yaw 0-10 deg), a new pose per reset index, deterministic
 26 truth: AutoMate's success rule (height, keypoints, after any control step, never the start state;
    first_success_step); information numbers (tilt of the axis, spin modulo symmetry, full / symmetric angle, depth /
    lateral at the tip centre); labels: the full angle for both simulators (symmetric_angle refused), ManiSkill labels
    bit-identical; step budget
 27 GPU 0 rule (app.py): nvidia-smi PIDS parsing, graphics <= 50 MiB accepted, compute or more graphics -> 7, no
    memory value -> 8; PRE_EXIT_HOOKS run with the exit code, a failing hook does not stop the others
 28 start pose and the start-box helper: frames.start_pose = Expert.start_pose; the START_GAP_FROM switch (the user's
    decision 2026-11-23, option B): default and approved "socket_top", h = L + g_top + g0 against AutoMate's rule, a
    missing g_top raises; "entrance" (h = L + g0, the old formula bit-identical) only under diagnostics, refused at
    import (subprocess) and by frames.start_back otherwise; Expert.reference_label: ok = completed AND success after
    the last step
 29 approved switches: config.APPROVED / unapproved_switches / refuse_unapproved (each differing key named, a missing
    key differs, diagnostics pass), camera far clip = ManiSkill's 100 m; report_m1b's "fails only by h > 0" count and
    its list of parts ending at h <= 0.01 mm (from height_mm, (i) included)
 30 physical symmetry (bore rule option 2, fit.physical_symmetry_sections): a square peg in a square hole n = 4, a
    round peg in a round hole round, a square peg loose in a round hole round, a keyed peg 1
 31 decisions of 2026-11-23 in the spin and the expert: the named list of 16 round-spin assemblies, geometry_v7 /
    ranges_v5, load_spec refuses a listed assembly whose file is not round, spin limit 15 deg; the perturbation window
    moves at most 3 mm/s and 2 deg/s per control step (Chongyu's close_speed); random motion and the contact offset
    by wm_v1's _move_outside (rows 27 / 38): every step <= 0.135 mm (pivot and TCP) and <= 0.1 deg, no hold, ceil(d /
    0.135 mm) commands per pure translation, the 600-command limit, other speeds refused, the contact push starts at
    the offset pose; random motion that reaches the step budget ends as "ExpertFailure: step_budget_<N>"; row 24:
    the true plug's top gap over the outside steps is recorded (peak min_actual_top_gap_mm_outside, information only)
    and the 1 N plug-socket contact guard is unchanged
 32 decision 7 of 2026-11-23 (row 39): the hand / finger - socket mesh distance of every nominal outside command
    (Execution._send, HAND_FINGER_COMMAND_CLEARANCE_M = 1 mm, the commanded TCP pose R_G rel h^-1 in the socket frame,
    fingers at robot.q_finger) fails the trajectory with "<phase>_nominal_command_guard:
    hand_finger_command_clearance" before the step, exactly 1 mm passes, the smallest value is recorded
    (min_command_hand_finger_clearance_mm), no check once inside (insert / contact push); noisy and compensated
    candidates are not checked (plug rules only); full_clearance = plug + hand_finger_clearance; Expert.waypoint_
    clearance uses the same frame as the guard (h = T_tcp_plug R_G); collect.redraw_waypoints redraws from its stream,
    at most WAYPOINT_DRAWS draws, keeps the last one when none passes
 33 move_to stillness (2026-11-24): reported joint speed < 2e-3 rad/s OR the FK hand pose moved < 1 um and
    < 0.001 deg in one control step (Isaac's stale joint_vel); old rule kept, fingers ignored, tolerance binding
Needs numpy, scipy, trimesh, PIL (the Isaac Lab environment has them).   python world_model/tests/test_automate_cpu.py
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# Isaac-only modules: a stand-in scene so that robot.py / expert.py import on a CPU (nothing here builds a scene)
_scene = types.ModuleType("wm.sim_automate.scene")
_scene.AutomateScene = object
sys.modules["wm.sim_automate.scene"] = _scene

from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import collect as CO  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402


def setup_assets(tmp: Path) -> CO.CollectConfig:
    """Fake asset / milestone-1b folders (results as batch_assemblies.py writes them, with the current provenance):
    00001 included ((iii) skipped), 00002 included through the re-run folder (batch holds a crash), 00003 (ii) failed,
    00004 (iii) for another grasp point, 00005 a zero-width variable, 00006 a result without provenance (stale).
    The provenance carries the asset hashes as batch_assemblies writes them (collect.asset_identity: geometry file,
    plug / socket meshes)."""
    A.ASSET_DIR = tmp / "assets"
    b2, rr = tmp / "m1b" / "batch_r3_cpu", tmp / "m1b" / "rerun"
    b2.mkdir(parents=True)
    rr.mkdir(parents=True)
    box = {"dtheta_deg": [-13.3, 6.7], "dp_m": [-0.01, 0.0], "dpsi_deg": [-7.5, 7.5]}
    ok_ii = {"pass": True, "placed": True, "completed": True, "depth_left_mm": 0.02, "success": True}
    iii_ok = lambda g, m: {"grasp": g, "pass": True, "move_ok": True, "completed": True, "depth_left_mm": 0.01,
                           "success": True, "hand_finger_socket_N": 0.0, "robot_socket_pairs_N": 0.0,
                           "hand_finger_max_N": 0.0, "mesh_margin_mm": m}
    rows = {"00001": (box, 0.004, None, {"ii": ok_ii, "iii": {"skipped": "mesh margin 4.0 mm >= 3 mm"}}),
            "00002": (box, 0.002, [-13.3, 0.0, 7.5], {"ii": ok_ii, "iii": iii_ok([-13.3, 0.0, 7.5], 2.0)}),
            "00003": (box, 0.004, None, {"ii": {**ok_ii, "pass": False, "completed": False, "depth_left_mm": 6.0},
                                         "iii": {"skipped": "x"}}),
            "00004": (box, 0.002, [-13.3, -0.01, 7.5], {"ii": ok_ii, "iii": iii_ok([0.0, 0.0, 0.0], 2.5)}),
            "00005": ({**box, "dp_m": [0.0, 0.0]}, 0.004, None, {"ii": ok_ii, "iii": {"skipped": "x"}}),
            "00006": (box, 0.004, None, {"ii": ok_ii, "iii": {"skipped": "x"}})}
    cfg = CO.CollectConfig(assemblies=sorted(rows), m1b_dirs=[str(b2), str(rr)], grasps_per_assembly=3)
    for aid, (bx, margin, wp, res) in rows.items():
        d = A.ASSET_DIR / aid
        d.mkdir(parents=True)
        (d / G.RANGES_FILE).write_text(json.dumps({"n": "round", "box": bx, "verified": True, "worst_margin_m": margin,
                                                   "worst_point": wp}))
        (d / A.SPEC_FILE).write_text(json.dumps({"assembly_id": aid}))
        for mesh in ("plug.obj", "socket.obj"):
            (d / mesh).write_text(f"o {aid}_{mesh}\n")
        hashes = {"spec_sha256": CO.sha256_file(d / A.SPEC_FILE), "plug_obj_sha256": CO.sha256_file(d / "plug.obj"),
                  "socket_obj_sha256": CO.sha256_file(d / "socket.obj")}
        assert CO.asset_identity(aid) == {**hashes, "asset_sha256": {A.SPEC_FILE: hashes["spec_sha256"], "plug.obj":
                                          hashes["plug_obj_sha256"], "socket.obj": hashes["socket_obj_sha256"]}}
        prov = {} if aid == "00006" else {
            "unix_time": 1.0, "ranges_sha256": CO.sha256_file(G.ranges_file(aid)),
            "grasp_py_sha256": CO.sha256_file(G.__file__), "physics_device": cfg.device, "switches": C.switches(),
            "physics_code_sha256": CO.physics_code_identity(), **hashes, "shared_asset_sha256": CO.shared_asset_identity(),
            "asset_sha256": {A.SPEC_FILE: hashes["spec_sha256"], "plug.obj": hashes["plug_obj_sha256"],
                             "socket.obj": hashes["socket_obj_sha256"]}}
        if aid == "00002":
            (b2 / f"{aid}.json").write_text(json.dumps({"assembly_id": aid, **prov, "error": "the simulator process died"}))
            (rr / f"{aid}.json").write_text(json.dumps({"assembly_id": aid, **prov, **res}))
        else:
            (b2 / f"{aid}.json").write_text(json.dumps({"assembly_id": aid, **prov, **res}))
    return cfg


# ------------------------------------------------------------------------------------------------------------- 1
def test_grasp_box(tmp: Path):
    cfg = setup_assets(tmp)
    specs, table = CO.build(cfg)
    st = {r["assembly_id"]: r for r in table}
    assert [st[a]["status"] for a in sorted(st)] == ["included", "included", "excluded", "excluded", "included",
                                                     "excluded"], {a: (st[a]["status"], st[a]["reason"]) for a in st}
    assert st["00001"]["box"]["dp_m"] == [-0.01, 0.0], "the verified box, not the base box (+-10 mm)"
    assert "re-run" in st["00004"]["reason"] and "(ii)" in st["00003"]["reason"]
    assert "stale" in st["00006"]["reason"], st["00006"]["reason"]
    assert any("rerun" in p for p in st["00002"]["inputs"]), "01125-like: result from the re-run folder"
    for s in specs:
        b = st[s.port]["box"]
        assert b["dtheta_deg"][0] <= s.pitch <= b["dtheta_deg"][1] and b["dp_m"][0] <= s.grasp_pos <= b["dp_m"][1]
    rg = json.loads((A.ASSET_DIR / "00001" / G.RANGES_FILE).read_text())
    rg["box"]["dtheta_deg"] = [-25.0, 5.0]                                    # outside the base box (+-20 deg)
    (A.ASSET_DIR / "00001" / G.RANGES_FILE).write_text(json.dumps(rg))
    try:
        CO.assembly_status("00001", cfg)
        raise AssertionError("a box outside the base box was accepted")
    except AssertionError as e:
        assert "00001" in str(e)
    try:
        G.load_ranges("09999")
        raise AssertionError("load_ranges computed a missing file")
    except FileNotFoundError:
        pass
    return CO.status_text(table).splitlines()[-1] + "; samples inside the verified boxes; box outside base refused; " \
        "load_ranges read-only"


# ------------------------------------------------------------------------------------------------------------- 2
def test_penetration():
    import trimesh
    sock = trimesh.creation.box((0.02, 0.02, 0.02))
    pts = np.array([[0.0, 0.0, 0.008], [0.0, 0.0, 0.0105], [0.0, 0.0, 0.012], [0.0099, 0.0099, 0.0]])
    d = K.mesh_penetration(sock, pts, chunk=2)
    assert abs(d[0] - 0.002) < 1e-9 and d[1] == 0.0 and d[2] == 0.0 and abs(d[3] - 0.0001) < 1e-9, d
    plug = trimesh.creation.box((0.01, 0.01, 0.01))
    M = np.eye(4)
    M[:3, 3] = [0.0, 0.0, 0.0145]                                             # plug 0.5 mm into the socket's top face
    r = K.overlap(plug, sock, M, n=2000, chunk=500)
    assert 0.4 < r["max_penetration_mm"] < 0.51, r
    M[:3, 3] = [0.0, 0.0, 0.0155]                                             # 0.5 mm above: no overlap
    assert K.overlap(plug, sock, M, n=2000)["max_penetration_mm"] == 0.0
    # a mesh that is not watertight (one of the 12 triangles of the box removed): winding number, ray parity not used
    holed = trimesh.Trimesh(sock.vertices, sock.faces[1:], process=False)
    assert not holed.is_watertight
    w = K.winding_number(holed, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]))
    assert w[0] > 0.8 and abs(w[1]) < 0.2, w
    assert abs(K.winding_number(sock, np.array([[0.001, 0.002, 0.003]]))[0] - 1.0) < 1e-9
    d2 = K.mesh_penetration(holed, pts, chunk=2)
    assert abs(d2[0] - 0.002) < 1e-9 and d2[2] == 0.0, d2
    r2 = K.overlap(plug, holed, np.eye(4) @ M, n=2000)
    assert r2["socket_watertight"] is False and r2["inside_test"]["plug points in socket"] == "winding number"
    # inconsistent winding (two triangles flipped as well): repaired before the winding number, recorded
    faces = holed.faces.copy()
    faces[:2] = faces[:2, ::-1]
    flipped = trimesh.Trimesh(sock.vertices, faces, process=False)
    assert not flipped.is_watertight and not flipped.is_winding_consistent
    q = np.array([[0.0, 0.0, 0.0], [0.002, -0.003, 0.004], [0.0, 0.0, 0.03], [0.02, 0.0, 0.0]])
    assert K.inside(flipped, q).tolist() == [True, True, False, False], K.winding_number(flipped, q)
    assert K.overlap(plug, flipped, M, n=2000)["winding_fixed"]["socket"] is True
    return f"inside by ray test (watertight) or winding number (not watertight), depth = distance to the surface, " \
        f"outside points 0, both ways ({r['max_penetration_mm']:.3f} mm)"


# ------------------------------------------------------------------------------------------------------------- 5
def test_reset_api():
    from wm.sim_automate import robot as R
    r = R.Robot.__new__(R.Robot)
    for call in (lambda: r.reset(1, 2.0, 0.001, 3.0), lambda: r.reset(1, 2.0, 0.001), lambda: r.reset(1),
                 lambda: r.reset(1, clamp_deg=2.0, dp_m=0.001), lambda: r.reset(1, clamp_deg=2.0, dp_m=0.001, socket_index=0)):
        try:
            call()
            raise AssertionError("accepted")
        except TypeError:
            pass
    got = {}
    r.reset = lambda seed, **kw: got.update(seed=seed, **kw)
    g7 = {"episode_seed": 7, "pitch_deg": 2.0, "grasp_pos_m": 0.001, "spin_deg": 3.0, "simulator": "automate"}
    R.Robot.reset_from_meta(r, {**g7, "socket_index": 5})
    assert got == {"seed": 7, "clamp_deg": 2.0, "dp_m": 0.001, "spin_deg": 3.0, "socket_index": 5}, got
    R.Robot.reset_from_meta(r, g7, socket_index=3)
    assert got["socket_index"] == 3
    assert C.SOCKET_DRAW == "per_reset"
    try:
        R.Robot.reset_from_meta(r, g7)                                         # per_reset: the pose must not be guessed
        raise AssertionError("a meta without socket_index was accepted under per_reset")
    except KeyError:
        pass
    try:
        R.Robot(port_type="00015", num_envs=2)
        raise AssertionError("num_envs = 2 accepted")
    except ValueError:
        pass
    try:
        R.Robot.reset_from_meta(r, {"episode_seed": 7, "pitch_deg": -40.0, "grasp_pos_m": -0.02})      # ManiSkill meta
        raise AssertionError("a meta without spin_deg was accepted")
    except KeyError:
        pass
    assert R.grasp_kwargs((1, 2, 3)) == {"clamp_deg": 1.0, "dp_m": 2.0, "spin_deg": 3.0}

    class Sc:                                                                 # state save / restore of the grasp
        def __init__(self):
            self.T_hand_plug = [np.eye(4)]

        def get_state(self):
            return {"T": [T.copy() for T in self.T_hand_plug]}

        def set_state(self, st):
            self.T_hand_plug = [T.copy() for T in st["T"]]
    r = R.Robot.__new__(R.Robot)
    r.sc, r.env_id, r._q_cmd, r.episode = Sc(), 0, np.zeros(7), {"a": 1}
    Ta = np.eye(4)
    Ta[:3, 3] = [0.0, 0.0, 0.01]
    r.T_tcp_plug, r.q_finger, r.finger_note = Ta, 0.012, "ok"
    r.sc.T_hand_plug = [C.T_HAND_TCP @ Ta]
    st = r.get_state()
    r.T_tcp_plug, r.q_finger, r.finger_note, r.sc.T_hand_plug = np.eye(4), 0.02, "other", [C.T_HAND_TCP]
    r.set_state(st)
    assert np.array_equal(r.T_tcp_plug, Ta) and r.q_finger == 0.012 and r.finger_note == "ok"
    st["T_tcp_plug"] = np.eye(4)
    try:
        r.set_state(st)
        raise AssertionError("a state whose weld does not hold its grasp was accepted")
    except RuntimeError:
        pass
    rb = R.Robot.__new__(R.Robot)                                             # grasp box check (reset raises outside)
    rb.assembly_id = "00001"
    load_ranges = R.G.load_ranges
    R.G.load_ranges = lambda aid: {"box": {"dtheta_deg": [-10.0, 10.0], "dp_m": [-0.01, 0.0], "dpsi_deg": [-5.0, 5.0]}}
    try:
        assert rb.check_grasp_in_box(10.0, -0.01, 5.0) is None
        assert rb.check_grasp_in_box(10.0 + 1e-3, 0.0, 0.0)["outside"] == {"dtheta_deg": 10.001}
        try:
            rb.reset(1, clamp_deg=0.0, dp_m=0.002, spin_deg=0.0, socket_index=0)
            raise AssertionError("a grasp outside the box was accepted")
        except ValueError as e:
            assert "not in the verified box" in str(e), e
        try:
            rb.reset(1, clamp_deg=0.0, dp_m=-0.002, spin_deg=0.0)              # inside the box, no socket_index
            raise AssertionError("reset without socket_index accepted under per_reset")
        except ValueError as e:
            assert "socket_index" in str(e), e
    finally:
        R.G.load_ranges = load_ranges
    return "positional / partial grasp arguments raise TypeError; reset_from_meta maps pitch_deg, grasp_pos_m, spin_deg, " \
        "socket_index (required under per_reset, also by reset itself); a ManiSkill meta raises; num_envs != 1 raises; " \
        "a grasp outside the box raises; get/set_state carry T_tcp_plug, q_finger, finger_note"


# ------------------------------------------------------------------------------------------------------------- 7
def test_force_sum():
    dt = C.PHYSICS_DT
    # Chongyu's rule: per contact pair (pair of collision shapes) the vector sum of its points' impulses, then the sum
    # of the pairs' magnitudes. A plug squeezed between two walls that are separate shapes: two pairs, forces add.
    walls = [np.array([[3.0, 0.0, 0.0]]) * dt, np.array([[-3.0, 0.0, 0.0]]) * dt, np.array([[0.0, 4.0, 0.0]]) * dt]
    assert abs(K.pair_force_sum(walls, dt) - 10.0) < 1e-9
    # one pair (one SDF shape pair) with opposite points: they cancel inside the pair, as in SAPIEN; the per-point sum
    # (information, upper bound) still sees 6 N
    one = [np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]) * dt]
    assert abs(K.pair_force_sum(one, dt)) < 1e-9 and abs(K.point_force_sum(one, dt) - 6.0) < 1e-9
    assert K.pair_force_sum([], dt) == 0.0
    # the contact sensor's force matrix (1 x 1 x 3 for plug-socket) is one net vector: the two walls cancel there
    net = np.array([[sum(J.sum(0) for J in walls[:2]) / dt]])
    assert K.force_magnitude_sum(net) == 0.0 and abs(K.pair_force_sum(walls[:2], dt) - 6.0) < 1e-9
    fm = np.array([[[3.0, 0.0, 0.0]], [[-3.0, 0.0, 0.0]], [[0.0, 4.0, 0.0]]])   # finger sensor: 3 bodies x 1 filter
    assert abs(K.force_magnitude_sum(fm) - 10.0) < 1e-12
    return "per shape pair: opposite walls (2 pairs) add to 10 N; points of one pair cancel (0 N, per-point bound 6 N)"


def test_contact_report_rule():
    """r8-expert-01 (2): guard_contacts' consistency rule (checks.check_contact_report) checks the plug pairs against
    the plug sensor and the robot pairs against the hand / finger sensor, each kind on its own."""
    def c(plug, robot):
        return {"plug_points": plug, "robot_points": robot, "n_points": plug + robot}
    out = K.check_contact_report(c(50, 12), 8.166, 0.0)               # 00015 seated row: both kinds present
    assert out["sensor_plug_net_N"] == 8.166 and out["sensor_finger_sum_N"] == 0.0, out
    K.check_contact_report(c(0, 0), 1.0, 1.0)                          # <= 1 N: no points needed
    for args, word in (((c(0, 12), 2.0, 0.0), "plug-socket"),          # robot points never hide a missing plug pair
                       ((c(50, 0), 0.0, 5.0), "robot-socket"),         # plug points never hide a missing robot pair
                       ((c(0, 0), 0.5, 1.5), "robot-socket")):
        try:
            K.check_contact_report(*args)
        except K.ContactReportError as exc:
            assert word in str(exc), exc
        else:
            raise AssertionError(f"no ContactReportError for {args}")
    assert issubclass(K.ContactReportError, RuntimeError)
    return "plug pair vs plug sensor and robot pair vs finger sensor, each > 1 N without points -> ContactReportError"


# ------------------------------------------------------------------------------------------------------------- 11
def test_frozen_plan(tmp: Path):
    cfg = setup_assets(tmp)
    code = {"files_sha256": "x", "n_files": 1, "git_head": None, "git_dirty": None}
    specs, table = CO.build(cfg)
    doc = CO.plan_document(cfg, specs, table, code)
    out = tmp / "out"
    CO.freeze(out, cfg, doc)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["plan_sha"] == doc["plan_sha"] and json.loads((out / "config.json").read_text())["config_sha"] == doc["config_sha"]
    assert all(r["inputs"] for r in plan["assemblies"]), "sha256 of every input file"
    again = CO.plan_document(cfg, *CO.build(cfg), code)
    assert again["plan_sha"] == doc["plan_sha"]
    CO.freeze(out, cfg, again)                                                # same plan: accepted
    refusals = []
    rf = A.ASSET_DIR / "00001" / G.RANGES_FILE                                # an input file changes (same box)
    rf.write_text(rf.read_text() + " ")
    for c2, cd in ((cfg, code), (CO.CollectConfig(**{**cfg.dump(), "device": "cpu"}), code),
                   (cfg, {**code, "files_sha256": "y"})):
        try:
            CO.freeze(out, c2, CO.plan_document(c2, *CO.build(c2), cd))
            raise AssertionError("accepted")
        except SystemExit as e:
            refusals.append(str(e).split("in ")[1][:60])
    return "written once; rebuilt plan identical; refused for a changed input file, device, code: " + " | ".join(refusals)


# ------------------------------------------------------------------------------------------------------------- 13
def test_streams(tmp: Path):
    cfg = setup_assets(tmp)
    s1, _ = CO.build(cfg)
    s2, _ = CO.build(cfg)
    assert [(s.pitch, s.seed) for s in s1] == [(s.pitch, s.seed) for s in s2], "deterministic"
    pl = np.random.default_rng(CO.stream(cfg, "00001", 0, 5, "placement")).uniform(size=6)
    ex = np.random.default_rng(CO.stream(cfg, "00001", 0, 5, "expert")).uniform(size=6)
    assert not np.allclose(pl, ex), "placement redraws and expert noise share no numbers"
    # wm_v1's seed + int(id): base seed + 1 for 00001 equals base seed for 00002; spawn keys do not collide
    c2 = CO.CollectConfig(**{**cfg.dump(), "seed": cfg.seed + 1})
    a = np.random.default_rng(CO.stream(c2, "00001", 0, 0, "grasp_values")).uniform(size=3)
    b = np.random.default_rng(CO.stream(cfg, "00002", 0, 0, "grasp_values")).uniform(size=3)
    assert not np.allclose(a, b)
    keys = {(int(s.port), s.gi, p) for s in s1 for p in CO.STREAMS.values()}
    assert len(keys) == len(s1) * len(CO.STREAMS)
    seeds = [s.seed for s in s1]
    assert len(set(seeds)) == len(seeds)
    return f"{len(s1)} grasps deterministic; placement vs expert streams differ; no overlap of shifted base seeds"


# ------------------------------------------------------------------------------------------------------------- 15
def test_zero_width(tmp: Path):
    cfg = setup_assets(tmp)
    specs, table = CO.build(cfg)
    doc = CO.plan_document(cfg, specs, table, {"files_sha256": "x"})
    assert doc["zero_width_assemblies"] == {"00005": ["dp_m"]}, doc["zero_width_assemblies"]
    for s in specs:
        if s.port == "00005":
            assert s.u["dp_m"] is None and s.grasp_pos == 0.0 and s.u["dtheta_deg"] is not None
    assert G.normalise(1.0, 0.0, 2.0) == 0.5 and G.normalise(0.0, 0.0, 0.0) is None
    assert not CO.grasp_rule_heldout({"a": None, "b": 0.1}) and CO.grasp_rule_heldout({"a": None, "b": 0.6})
    return "u = None for the zero-width dp of 00005, listed in plan.json; never counted as held out"


# ------------------------------------------------------------------------------------------------------------- 22
def test_online_guard():
    ms = [{"grasp": "A_g00", "port": "A", "condition": "train"}, {"grasp": "G3_g01", "port": "G3"}]
    D.check_automate_conditions(ms)                                           # ManiSkill: untouched
    D.check_automate_conditions([{"grasp": "00015_g00", "simulator": "automate", "condition": "train"}])
    for bad in ({"grasp": "00015_g00", "simulator": "automate", "condition": "unassigned"},
                {"grasp": "00015_g01", "simulator": "automate"}):
        try:
            D.check_automate_conditions(ms + [bad])
            raise AssertionError("accepted")
        except SystemExit:
            pass
    meta = {"steps": 10, "port": "A", "grasp": "A_g00", "condition": "train", "kind": "insert", "outcome": "inserted",
            "pitch_deg": -40.0, "grasp_pos_m": -0.02, "pin_length_m": 0.02}
    assert D.entry_of(meta, "x.npz", 3) == {"path": "x.npz", "n_images": 3, "steps": 10, **{k: meta[k] for k in meta if k != "steps"}}
    assert D.entry_of({**meta, "simulator": "automate"}, "x.npz", 3)["simulator"] == "automate"
    return "AutoMate grasps without a split refuse; ManiSkill grasps and manifest entries unchanged"


# ------------------------------------------------------------------------------------------------------------- 25
def test_socket_pose():
    poses = [C.socket_pose(s, "table") for s in range(2000)]
    off = np.array([d["offset_m"] for _, d in poses])
    yaw = np.array([d["yaw_deg"] for _, d in poses])
    assert np.all(np.abs(off) <= 0.05) and off.min() < -0.045 and off.max() > 0.045
    assert yaw.min() >= 0.0 and yaw.max() <= 10.0 and yaw.max() > 9.5
    T, d = poses[3]
    assert np.allclose(T[:3, 3], np.asarray(C.SOCKET_POS["table"]) + d["offset_m"])
    assert np.allclose(T[:3, 2], [0, 0, 1]), "roll = pitch = 0: the socket stays upright"
    assert abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])) - d["yaw_deg"]) < 1e-9
    assert np.array_equal(C.socket_pose(3, "table")[0], T), "deterministic per seed"
    # reset index k: per_reset draws a new pose per k (k = 0 = the pose of 2026-11-22), per_grasp ignores k
    P = [C.socket_pose(3, "table", k, "per_reset")[0] for k in range(13)]
    assert np.array_equal(P[0], T) and len({tuple(np.round(x[:3, 3], 9)) for x in P}) == 13
    assert all(np.array_equal(C.socket_pose(3, "table", k, "per_grasp")[0], T) for k in range(13))
    assert np.array_equal(C.socket_pose(3, "table", 5)[0], P[5]), "default draw is per_reset"
    return f"offsets within +-50 mm (seen {off.min() * 1e3:.1f} .. {off.max() * 1e3:.1f}), yaw {yaw.min():.2f} .. " \
        f"{yaw.max():.2f} deg; per_reset: 13 distinct poses for 13 resets, index 0 unchanged; per_grasp: one pose"


# ------------------------------------------------------------------------------------------------------------- 26
def test_truth_symmetry():
    from scipy.spatial.transform import Rotation
    from wm.sim_automate import expert as EX

    def rel7(R, t=(0, 0, 0)):
        x, y, z, w = R.as_quat()
        return np.array([*t, w, x, y, z])
    tip, d = (0.0, 0.0, 0.0), 0.02                                           # disassembly distance 20 mm
    up = lambda mm, lat=0.0: (-mm / 1000.0, lat / 1000.0, 0.0)                # plug mm above assembled (x = insertion)
    T = lambda R, t=(0, 0, 0), n=1: K.truth_from_rel(rel7(R, t), tip, n, d)
    I = Rotation.identity()
    # AutoMate's rule: 0 < h < d and mean keypoint distance < 15 mm
    t = T(I, up(1.0))
    assert bool(t["success"]) and abs(t["height_mm"] - 1.0) < 1e-9 and abs(t["keypoint_dist_mm"] - 1.0) < 1e-9 and t["seated"] == t["success"]
    assert not T(I)["success"], "h = 0 is not > 0 (AutoMate's strict test)"
    assert bool(T(I, up(14.0))["success"]) and not T(I, up(16.0))["success"], "axial offset counts in the keypoints"
    h6 = K.truth_from_rel(rel7(I, up(6.0)), tip, 1, 0.005)
    assert not h6["success"] and h6["keypoint_dist_mm"] < 15, "height above the disassembly distance"
    assert bool(T(I, up(1.0, 14.0))["success"]) and not T(I, up(1.0, 16.0))["success"], "keypoints 15 mm"
    # tilt about a lateral axis: mean_j |(R - I) o_j| = 2 sin(a / 2) mean_j |o_j| = 2 sin(a / 2) 50 mm
    for deg, ok in ((10.0, True), (20.0, False)):
        Ry = Rotation.from_euler("y", deg, degrees=True)
        assert abs(T(Ry)["keypoint_dist_mm"] - 100 * np.sin(np.radians(deg / 2))) < 1e-6
        assert bool(T(Ry, up(0.5))["success"]) == ok
    # spin about the insertion axis does not move the axial keypoints: success, whatever the symmetry
    assert bool(T(Rotation.from_euler("x", 90, degrees=True), up(1.0), 1)["success"])
    # information numbers: axis tilt, spin modulo symmetry, full angle, symmetric angle
    spin = Rotation.from_euler("x", 30, degrees=True)
    t = T(spin, n=4)
    assert abs(t["tilt_deg"]) < 1e-9 and abs(t["spin_deg"] - 30) < 1e-9 and abs(t["angle_deg"] - 30) < 1e-6
    t = T(Rotation.from_euler("x", 90, degrees=True), n=4)
    assert t["spin_deg"] < 1e-9 and t["tilt_symmetric_deg"] < 1e-6 and abs(t["angle_deg"] - 90) < 1e-6
    t = T(Rotation.from_euler("x", 40, degrees=True), n="round")
    assert t["spin_deg"] == 0 and t["tilt_symmetric_deg"] < 1e-9
    t = T(Rotation.from_euler("y", 3, degrees=True) * spin, n="round")
    assert abs(t["tilt_deg"] - 3) < 1e-9 and abs(t["tilt_symmetric_deg"] - 3) < 1e-6
    assert set(K.TRUTH_COLUMNS) <= set(t) and "tilt_definition" not in C.switches()
    seq = np.stack([rel7(I, up(30.0)), rel7(I, up(5.0)), rel7(I, up(30.0))])
    spec = types.SimpleNamespace(tip_centre_m=tip, symmetry_n=1, disassembly_dist_m=d)
    assert K.success_any(seq, spec) and not K.success_any(seq[[0, 2]], spec), "success after any control step"
    # the start state (row 0 of eval_rel) is never evaluated (AutoMate checks after each step only)
    st = K.success_steps(seq[[1, 0, 0]], spec)
    assert not st["success_any"] and st["start_success"] and st["first_success_step"] is None and not st["success_end"]
    st = K.success_steps(np.stack([rel7(I, up(30.0))] * 2 + [rel7(I, up(5.0))] * 2), spec)
    assert st["success_any"] and st["success_end"] and st["first_success_step"] == 2
    assert st["per_step"].tolist() == [False, True, True]
    st = K.success_steps(seq[:1], spec)
    assert not st["success_any"] and not st["success_end"] and len(st["per_step"]) == 0, "no control step, no success"
    # depth / lateral at the tip centre: plug tip 30 mm from the plug origin (plug frame z = -0.03, charger x = +0.03)
    t = K.truth_from_rel(rel7(Rotation.from_euler("y", 2, degrees=True)), (0.0, 0.0, -0.03), "round", d)
    assert abs(t["lateral_mm"] - 30 * np.sin(np.radians(2))) < 1e-6, t
    t0 = K.truth_from_rel(rel7(Rotation.identity(), (-0.001, 0.0005, 0.0)), (0.0, 0.0, -0.03), "round", d)
    assert abs(t0["depth_left_mm"] - 1.0) < 1e-9 and abs(t0["lateral_mm"] - 0.5) < 1e-9
    # labels: ManiSkill path bit-identical, AutoMate path symmetric
    g = np.random.default_rng(0)
    hand = np.tile(np.eye(4), (5, 1, 1))
    hand[:, :3, :3] = Rotation.from_rotvec(g.normal(size=(5, 3)) * 0.05).as_matrix()
    hand[:, :3, 3] = g.normal(size=(5, 3)) * 0.003
    meta = {"H_star": np.eye(4).tolist(), "pin_length_m": 0.02, "axis": [0.0, 0.0, -1.0]}
    lab = WD.decision_labels(meta, hand, np.arange(5))
    R = hand[:, :3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    assert np.array_equal(lab["probe"][:, 3], ang.astype(np.float32)), "ManiSkill: the full angle, unchanged"
    hs = hand.copy()
    hs[:, :3, :3] = Rotation.from_euler("z", 37, degrees=True).as_matrix()   # spin about the (vertical) axis only
    la = WD.decision_labels({**meta, "simulator": "automate", "symmetry_n": "round"}, hs, np.arange(5))
    assert np.allclose(la["probe"][:, 3], 37, atol=1e-4), "AutoMate labels = ManiSkill labels: the full angle"
    try:
        WD.decision_labels({**meta, "simulator": "automate", "symmetry_n": 4, "tilt_definition": "symmetric_angle"},
                           hs, np.arange(5))
        raise AssertionError("tilt_definition symmetric_angle accepted")
    except ValueError:
        pass
    assert EX.step_budget(0.020, 0.0) == 800 and EX.step_budget(0.0251, 0.0) == 800 + 8 * 11
    assert EX.step_budget(0.015, 0.008) == 800 + 8 * 6
    return "success = AutoMate's check_plug_inserted_in_socket (0 < h < d_dis, mean keypoint distance < 15 mm, after " \
        "any control step, start state excluded, first_success_step); depth / lateral / tilt / spin / angles " \
        "information only; labels: full angle for both simulators; step budget 800 + 8 per 0.5 mm beyond 20 mm"


# ------------------------------------------------------------------------------------------------------------- 27
def test_gpu_rule():
    from wm.sim_automate import app as AP
    text = """GPU 00000000:16:00.0
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 100
            Type                          : G
            Name                          : python
            Used GPU Memory               : 8 MiB
        Process ID                        : 200
            Type                          : C+G
            Name                          : python
            Used GPU Memory               : 6618 MiB
        Process ID                        : 300
            Type                          : G
            Name                          : python
            Used GPU Memory               : N/A
        Process ID                        : 400
            Type                          : G
            Name                          : x
            Used GPU Memory               : 30 MiB
        Process ID                        : 400
            Type                          : G
            Name                          : x
            Used GPU Memory               : 30 MiB
"""
    ctx = AP.parse_pids(text)
    assert ctx[100] == [{"type": "G", "mib": 8.0}] and ctx[200][0]["type"] == "C+G" and ctx[300][0]["mib"] is None
    lim = C.FORBIDDEN_GPU_GRAPHICS_MAX_MIB
    assert lim == 50 and C.FORBIDDEN_GPU == 0
    assert AP.gpu_rule([], lim)[0] == 0 and AP.gpu_rule(ctx[100], lim)[0] == 0, "8 MiB graphics context accepted"
    assert AP.gpu_rule(ctx[200], lim)[0] == AP.EXIT_FORBIDDEN_GPU == 7, "compute context"
    assert AP.gpu_rule([{"type": "C", "mib": 1.0}], lim)[0] == 7
    assert AP.gpu_rule(ctx[400], lim)[0] == 7, "60 MiB of graphics contexts in total > 50 MiB"
    assert AP.gpu_rule(ctx[300], lim)[0] == AP.EXIT_GPU_UNREADABLE == 8
    calls, old = [], (AP.shutdown, AP.contexts_on_gpu, list(AP.PRE_EXIT_HOOKS))

    def bad_hook(code):
        raise RuntimeError("hook fails")

    class Exited(Exception):
        pass

    def fake_shutdown(code):
        calls.append(("shutdown", code))
        raise Exited
    try:
        AP.shutdown = fake_shutdown
        AP.PRE_EXIT_HOOKS[:] = [bad_hook, lambda code: calls.append(("hook", code))]
        for ctx_now, want in (({os.getpid(): [{"type": "C", "mib": 500.0}]}, 7), (None, 8)):
            AP.contexts_on_gpu = lambda i, c=ctx_now: c
            calls.clear()
            try:
                AP.assert_off_forbidden_gpu("test")
                raise AssertionError("no exit")
            except Exited:
                pass
            assert calls == [("hook", want), ("shutdown", want)], calls
        AP.contexts_on_gpu = lambda i: {os.getpid(): [{"type": "G", "mib": 8.0}]}
        assert AP.assert_off_forbidden_gpu("test")["contexts"] == [{"type": "G", "mib": 8.0}]
    finally:
        AP.shutdown, AP.contexts_on_gpu = old[0], old[1]
        AP.PRE_EXIT_HOOKS[:] = old[2]
    assert "--/renderer/activeGpu=1" in AP.kit_args() and C.switches()["kit_gpu"] == 1
    return "G <= 50 MiB accepted; C / C+G or > 50 MiB -> 7; no memory value or no nvidia-smi -> 8; hooks run with " \
        "the code before the exit, a failing hook does not stop the next"


# ------------------------------------------------------------------------------------------------------------- 28
def test_start_and_label():
    from scipy.spatial.transform import Rotation
    from wm.env.sampling import Offset
    from wm.sim_automate import expert as EX
    from wm.sim_automate import frames as FR
    g = np.random.default_rng(1)
    H = np.eye(4)
    H[:3, :3] = Rotation.from_rotvec(g.normal(size=3)).as_matrix()
    H[:3, 3] = g.normal(size=3)
    F = FR.frame(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.2, 0.0]))
    off = Offset(0.01, (0.003, -0.002), (1.0, -2.0, 3.0))
    assert np.array_equal(EX.Expert.start_pose(H, F, 0.02, off, 0.004), FR.start_pose(H, F, 0.02, off, 0.004))
    T_tcp_plug = np.eye(4)
    T_tcp_plug[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()   # plug along -z of the TCP
    T_tcp_plug[:3, 3] = [0.0, 0.0, 0.05]
    spec = types.SimpleNamespace(pin_length_m=0.005, disassembly_dist_m=0.02, depth_top_m=0.005)       # g_top 0
    r6 = K.start_box_success(spec, T_tcp_plug, Offset(0.006, (0.0, 0.0), (0.0, 0.0, 0.0)))
    assert r6["gap_from"] == "socket_top", r6
    assert r6["success"] and abs(r6["height_mm"] - 11.0) < 1e-9 and abs(r6["keypoint_dist_mm"] - 11.0) < 1e-9, r6
    assert not K.start_box_success(spec, T_tcp_plug, Offset(0.012, (0.0, 0.0), (0.0, 0.0, 0.0)))["success"], "h 17 mm"
    s10 = types.SimpleNamespace(pin_length_m=0.005, disassembly_dist_m=0.010, depth_top_m=0.005)
    assert not K.start_box_success(s10, T_tcp_plug, Offset(0.006, (0.0, 0.0), (0.0, 0.0, 0.0)))["success"], "h > d_dis"
    rot = K.start_box_success(spec, T_tcp_plug, Offset(0.006, (0.0, 0.0), (6.0, 0.0, 0.0)))
    assert rot["keypoint_dist_mm"] > r6["keypoint_dist_mm"], "a rotation about the TCP moves the keypoints"
    # reference_label: ok = completed AND AutoMate success after the last step; both recorded
    rob = types.SimpleNamespace(hand_pose=lambda: np.eye(4), H_star=lambda: np.eye(4), episode={"socket_index": 3})
    E = EX.Expert.__new__(EX.Expert)
    E.robot = rob
    old = EX.truth
    EX.truth = lambda r: {"success": True}
    try:
        got = {}
        for completed, s_end in ((True, True), (False, True), (True, False)):
            E.run = lambda kind, cb, ss, c=completed, e=s_end: {"completed": c, "success_end": e, "success": e,
                                                               "first_success_step": 4 if e else None, "steps": 9,
                                                               "failure_reason": None if c else "ExpertFailure: x"}
            lab = E.reference_label(np.random.SeedSequence(0))
            got[(completed, s_end)] = lab
            assert lab["completed"] is completed and lab["success_end"] is s_end and lab["ok"] is (completed and s_end)
        assert got[(False, True)]["failure"] == "ExpertFailure: x" and "AutoMate" in got[(True, False)]["failure"]
        assert all(k in got[(True, True)] for k in ("H_star", "axis", "H_star_model", "H_star_fk_vs_model_mm_deg",
                                                    "success_any", "first_success_step", "socket_index"))
    finally:
        EX.truth = old
    # START_GAP_FROM (the user's decision 2026-11-23): default and approved "socket_top" moves the start back by
    # L + g_top + g0; "entrance" (the old formula, bit-identical) only under diagnostics
    assert C.START_GAP_FROM == "socket_top" and C.APPROVED["start_gap_from"] == "socket_top"
    assert C.switches()["start_gap_from"] == "socket_top"
    old_formula = H.copy()
    old_formula[:3, 3] = H[:3, 3] + F @ np.array([-(0.02 + off.tip_gap_m), off.yz_m[0], off.yz_m[1]])
    old_formula[:3, :3] = F @ FR.euler_R(off.rpy_deg) @ F.T @ H[:3, :3]
    top = FR.start_pose(H, F, 0.02, off, 0.004)
    assert np.allclose(top[:3, 3] - old_formula[:3, 3], -0.004 * F[:, 0], atol=1e-15) and \
        np.array_equal(top[:3, :3], old_formula[:3, :3])
    assert np.array_equal(FR.start_pose(H, F, 0.02, off, 0.004, "socket_top"), top)
    assert np.array_equal(EX.Expert.start_pose(H, F, 0.02, off, 0.004), top)
    assert FR.start_back(0.02, 0.006, 0.004) == 0.02 + 0.004 + 0.006
    assert np.array_equal(FR.start_pose(H, F, 0.02, off, 0.0), old_formula), "g_top 0: the two rules agree"
    for bad in ((None,), (None, "socket_top"), (0.004, "rim"), (0.004, "entrance")):
        try:
            FR.start_pose(H, F, 0.02, off, *bad)
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass
    try:
        EX.Expert.start_pose(H, F, 0.02, off)
        raise AssertionError("a start without g_top accepted")
    except ValueError:
        pass
    deep = types.SimpleNamespace(pin_length_m=0.005, disassembly_dist_m=0.02, depth_top_m=0.009)   # g_top 4 mm
    t = K.start_box_success(deep, T_tcp_plug, Offset(0.006, (0.0, 0.0), (0.0, 0.0, 0.0)))
    assert t["gap_from"] == "socket_top" and abs(t["height_mm"] - 15.0) < 1e-9 and not t["success"], t
    try:
        K.start_box_success(deep, T_tcp_plug, Offset(0.006, (0.0, 0.0), (0.0, 0.0, 0.0)), gap_from="entrance")
        raise AssertionError("entrance accepted outside diagnostics")
    except ValueError:
        pass
    old_diag = C.DIAGNOSTIC
    try:
        C.DIAGNOSTIC = True
        assert np.array_equal(FR.start_pose(H, F, 0.02, off, 0.004, "entrance"), old_formula), "entrance path changed"
        assert np.array_equal(FR.start_pose(H, F, 0.02, off, None, "entrance"), old_formula), "g_top unused"
        e = K.start_box_success(deep, T_tcp_plug, Offset(0.006, (0.0, 0.0), (0.0, 0.0, 0.0)), gap_from="entrance")
        assert e["gap_from"] == "entrance" and abs(e["height_mm"] - 11.0) < 1e-9 and e["success"], e
    finally:
        C.DIAGNOSTIC = old_diag
    assert K.g_top_of(types.SimpleNamespace(pin_length_m=0.005, depth_top_m=0.003)) == 0.0
    # the environment switch at import: "entrance" refused unless WM_AUTOMATE_DIAGNOSTIC=1
    import subprocess
    code = "import wm.sim_automate.config as C; print(C.START_GAP_FROM)"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WM_AUTOMATE_", "WM_CAM", "WM_FORBIDDEN", "WM_KIT"))}
    env["PYTHONPATH"] = str(HERE.parent / "src")
    run = lambda extra: subprocess.run([sys.executable, "-c", code], env={**env, **extra}, capture_output=True, text=True)
    r_def, r_ent = run({}), run({"WM_AUTOMATE_START_GAP_FROM": "entrance"})
    r_diag = run({"WM_AUTOMATE_START_GAP_FROM": "entrance", "WM_AUTOMATE_DIAGNOSTIC": "1"})
    assert r_def.returncode == 0 and r_def.stdout.strip() == "socket_top", (r_def.stdout, r_def.stderr[-500:])
    assert r_ent.returncode != 0 and "only socket_top outside diagnostics" in r_ent.stderr, r_ent.stderr[-500:]
    assert r_diag.returncode == 0 and r_diag.stdout.strip() == "entrance", r_diag.stderr[-500:]
    return "Expert.start_pose = frames.start_pose; default socket_top: aligned start at g0 = 6 mm, L = 5 mm, g_top 0: " \
        "h = keypoint distance = 11 mm (success), 17 mm or h > d_dis not; g_top 4 mm: h = L + g_top + g0 = 15 mm (no " \
        "success); no g_top raises; entrance = old formula bit-identical only under diagnostics, refused otherwise " \
        "(argument and environment variable at import); reference_label ok only when completed AND success_end"


# ------------------------------------------------------------------------------------------------------------ 29
def test_approved_switches():
    import importlib.util
    assert C.CLIP_FAR == 100.0 and C.CLIP_NEAR == 0.01, (C.CLIP_NEAR, C.CLIP_FAR)
    now = C.switches()
    assert set(C.APPROVED) <= set(now), set(C.APPROVED) - set(now)
    assert C.unapproved_switches() == {}, C.unapproved_switches()
    C.refuse_unapproved("test")                                   # the defaults pass
    bad = {**now, "socket_draw": "per_grasp", "kit_gpu": 0}
    del bad["weld_mode"]
    got = C.unapproved_switches(bad)
    assert got == {"socket_draw": ("per_grasp", "per_reset"), "kit_gpu": (0, 1), "weld_mode": (None, "articulation")}, got
    old = C.DIAGNOSTIC
    try:
        C.DIAGNOSTIC = False
        try:
            C.refuse_unapproved("test", bad)
            raise AssertionError("unapproved switches accepted")
        except RuntimeError as exc:
            msg = str(exc)
            assert all(k in msg for k in ("socket_draw='per_grasp'", "kit_gpu=0", "weld_mode=None")), msg
        C.DIAGNOSTIC = True
        C.refuse_unapproved("test", bad)                          # diagnostics are not refused
    finally:
        C.DIAGNOSTIC = old
    spec = importlib.util.spec_from_file_location("report_m1b", HERE.parent / "scripts/automate/report_m1b.py")
    RM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RM)
    ii = {"pass": False, "placed": True, "completed": True, "depth_left_mm": 0.01, "height_mm": -0.007,
          "keypoint_dist_mm": 0.2}
    assert RM.fails_only_by_height(ii)
    assert not RM.fails_only_by_height({**ii, "height_mm": 0.007}) and not RM.fails_only_by_height({**ii, "pass": True})
    assert not RM.fails_only_by_height({**ii, "completed": False}) and not RM.fails_only_by_height({**ii, "height_mm": -0.2})
    # r6-expert-02: parts ending at h <= 0.01 mm from height_mm, (i) included (older (i) results have no
    # h_end_le_0.01mm key); skipped / missing parts left out; the gate parts' state next to them
    b = {"i": {"pass": False, "height_mm": -0.0027}, "ii": {**ii, "pass": True, "height_mm": 0.012},
         "iii": {"pass": True, "height_mm": 0.01}}
    assert RM.low_end_heights(b) == {"i": -0.0027, "iii": 0.01}, RM.low_end_heights(b)
    assert RM.low_end_heights({"i": {"error": "x"}, "iii": {"skipped": "no box"}}) == {}
    assert RM.low_end_heights({"ii": {"height_mm": float("nan")}}) == {}
    assert [RM.part_state(b, "ii"), RM.part_state({"iii": {"skipped": "no box"}}, "iii"), RM.part_state(b, "x"),
            RM.part_state({"ii": {"pass": False}}, "ii")] == ["pass", "skipped", "missing", "FAIL"]
    return "defaults approved; per_grasp / kit_gpu 0 / missing weld named and refused, not in diagnostics; far clip " \
        "100 m; h > 0-only failures counted; parts ending at h <= 0.01 mm listed from height_mm, (i) included"


# ------------------------------------------------------------------------------------------------------------ 30
def test_physical_symmetry():
    from shapely.geometry import Point, Polygon, box
    from wm.sim_automate import fit as FT
    sq = box(-5, -5, 5, 5)
    ring = lambda outer, hole: outer.difference(hole)
    socket_sq = ring(box(-20, -20, 20, 20), box(-5.01, -5.01, 5.01, 5.01))        # square hole, 0.01 clearance
    socket_round = ring(box(-20, -20, 20, 20), Point(0, 0).buffer(5.05, 256))
    socket_loose = ring(box(-20, -20, 20, 20), Point(0, 0).buffer(7.2, 256))     # round hole around the square peg
    peg_round = Point(0, 0).buffer(5.0, 256)
    keyed = Polygon([(-5, -5), (5, -5), (5, 5), (-3, 5), (-5, 2)])
    socket_keyed = ring(box(-20, -20, 20, 20), keyed.buffer(0.01, join_style=2))
    r4 = FT.physical_symmetry_sections([sq] * 3, [socket_sq] * 3, (0.0, 0.0), step_deg=1.0)
    rr = FT.physical_symmetry_sections([peg_round] * 3, [socket_round] * 3, (0.0, 0.0), step_deg=1.0)
    rl = FT.physical_symmetry_sections([sq] * 3, [socket_loose] * 3, (0.0, 0.0), step_deg=1.0)
    rk = FT.physical_symmetry_sections([keyed] * 3, [socket_keyed] * 3, (0.0, 0.0), step_deg=1.0)
    assert r4["n_phys"] == 4, r4
    assert rr["n_phys"] == "round" and rl["n_phys"] == "round", (rr, rl)
    assert rk["n_phys"] == 1, rk
    filled = FT.physical_symmetry_sections([sq] * 3, [box(-20, -20, 20, 20)] * 3, (0.0, 0.0), step_deg=10.0)
    assert filled["n_phys"] is None and not filled["reliable"], filled
    return "square in square 4, round in round and square loose in a round hole 'round', keyed 1, a socket section " \
        "filled across its hole unreliable (None)"


# ------------------------------------------------------------------------------------------------------------ 31
def test_decisions_2026_11_23(tmp: Path):
    from scipy.spatial.transform import Rotation
    from wm.env.sampling import Offset
    from wm.sim_automate import expert as EX
    # spin: exactly the 16 named assemblies, new file versions, a listed assembly whose file is not round is refused
    named = {"00320", "00863", "00030", "00103", "00110", "00117", "00143", "00187", "00514", "00537", "00559", "00686",
             "00700", "00783", "00831", "01041"}
    assert set(A.SPIN_ROUND_2026_11_23) == named and len(A.SPIN_ROUND_2026_11_23) == 16
    assert A.SPEC_FILE == "geometry_v7.json" and G.RANGES_FILE == "ranges_v5.json", (A.SPEC_FILE, G.RANGES_FILE)
    assert G.spin_limit_deg("round") == 15.0 and G.spin_limit_deg(12) == 7.5 and G.spin_limit_deg(8) == 11.25
    old_dir = A.ASSET_DIR
    try:
        A.ASSET_DIR = tmp / "assets"
        base = {"assembly_id": "00320", "pin_length_m": 0.01, "clearance_m": 1e-4, "disassembly_dist_m": 0.02,
                "entrance_z_m": 0.0, "depth_top_m": 0.01, "socket_top_z_m": 0.0, "plug_bottom_z_m": -0.01,
                "plug_top_z_m": 0.03, "seated_offcentre_m": 0.0, "min_gap_m": 1e-4, "fingertip_z_m": 0.02,
                "plug_xy_m": [0.01, 0.01], "symmetry_n": 12, "category": "polygonal", "fit": {"n": 12},
                "axis_xy_m": [0.0, 0.0], "tip_centre_m": [0.0, 0.0, -0.01]}
        for aid, n in (("00320", 12), ("00015", 12)):
            (A.ASSET_DIR / aid).mkdir(parents=True)
            (A.ASSET_DIR / aid / A.SPEC_FILE).write_text(json.dumps({**base, "assembly_id": aid, "symmetry_n": n}))
        try:
            A.load_spec("00320")
            raise AssertionError("a listed assembly with n = 12 accepted")
        except ValueError as exc:
            assert "2026-11-23" in str(exc), exc
        assert A.load_spec("00015").symmetry_n == 12, "an assembly outside the list keeps its n"
        (A.ASSET_DIR / "00320" / A.SPEC_FILE).write_text(json.dumps({**base, "symmetry_n": "round"}))
        sp = A.load_spec("00320")
        assert sp.symmetry_n == "round" and sp.category == "polygonal", "category unchanged"
    finally:
        A.ASSET_DIR = old_dir
    # expert speeds: a stand-in Execution with an identity goal and a 10 mm plug hull
    assert (EX.CLOSE_SPEED_M_S, EX.CLOSE_ANG_SPEED_DEG_S, EX.RANDOM_SPEED_M_S) == (0.003, 2.0, 0.0027)
    geo = EX.PlugGeometry.__new__(EX.PlugGeometry)
    geo.c, geo.x_ent, geo.g_top, geo.n = np.zeros(3), 0.0, 0.0, "round"
    geo.V = np.array([[x, y, z] for x in (-0.03, 0.0) for y in (-0.005, 0.005) for z in (-0.005, 0.005)])
    ex = EX.Execution.__new__(EX.Execution)
    ex.g, ex.G, ex.hi = geo, np.eye(4), np.eye(4)
    ex.h = np.eye(4)
    ex.h[0, 3] = -0.05                                        # TCP 5 cm behind the pivot
    ex.hi = EX._inv(ex.h)
    ex.cmd, ex.stages, ex.steps, ex.perturbation_seen = None, [], 0, False
    ex.m = {"top_gap": 0.0, "lateral": 0.0, "angle": 0.0, "axis_tilt": 0.0, "spin": 0.0}
    ex.rng = np.random.default_rng(0)
    src = geo.outside(0.015, (0.008, -0.006), np.eye(3))
    ex.actual = lambda: src
    sent, calls = [], []
    ex._send = lambda rel, **kw: (sent.append(rel.copy()), calls.append(kw))
    ex.perturbation()
    assert len(sent) == EX.STOCHASTIC_STEPS and all(c.get("approach_noise") and c.get("compensate") is False
                                                    for c in calls)
    prev = src
    for rel in sent:
        d_piv = float(np.linalg.norm(geo.t_of(rel) - geo.t_of(prev)))
        d_tcp = float(np.linalg.norm(ex.tcp(rel)[:3, 3] - ex.tcp(prev)[:3, 3]))
        d_rot = EX._angle_deg(prev[:3, :3].T @ rel[:3, :3])
        assert max(d_piv, d_tcp) <= 0.003 * EX.DT + 1e-9 and d_rot <= 2.0 * EX.DT + 1e-7, (d_piv, d_tcp, d_rot)
        prev = rel
    far = geo.rel([-0.001, 0.0, 0.0], Rotation.from_euler("x", 10, degrees=True).as_matrix())   # TCP on the x axis
    one = ex._advance(np.eye(4), far, EX.CLOSE_SPEED_M_S, EX.CLOSE_ANG_SPEED_DEG_S)
    assert abs(EX._angle_deg(one[:3, :3]) - 2.0 * EX.DT) < 1e-9, "rotation-limited step = 0.1 deg"
    # random motion and the contact offset: wm_v1's _move_outside (one helper, rows 27 / 38, restored 2026-11-23):
    # constant steps <= 2.7 mm/s and 2 deg/s, no ramp, no hold at a waypoint; other speeds refused before any motion
    v_step, w_step = EX.RANDOM_SPEED_M_S * EX.DT, EX.CLOSE_ANG_SPEED_DEG_S * EX.DT        # 0.135 mm, 0.1 deg

    def stand_in(start):
        e = EX.Execution.__new__(EX.Execution)
        e.g, e.G, e.h, e.hi = geo, np.eye(4), ex.h, ex.hi
        e.stages, e.steps, e.perturbation_seen, e.cmd = [], 0, False, start.copy()
        e.m = dict(ex.m, success=False)
        e.result, e.peak = {"completed": False, "failure_reason": None}, {"finger_socket_N": 0.0}
        e.success_any, e.first_success_step, e.sent, e.phases = False, None, [], []

        def send(rel, **kw):
            assert not kw, f"_move_outside sends plain commands (as wm_v1), got {kw}"
            e.sent.append(rel.copy())
            e.phases.append(e.phase)
            e.cmd = rel.copy()

        def no_segment(*a, **k):
            raise AssertionError("the trapezoidal _segment is not wm_v1's motion")
        e._send, e._segment = send, no_segment
        return e

    def check_steps(e, start, phase_of, targets):
        """every command step <= 0.135 mm (pivot and TCP) and <= 0.1 deg, none repeated; each phase ends on its
        target (wm_v1's stop: TCP mm + deg < 1e-3) -> number of commands per phase"""
        prev, counts = start, {}
        for rel, ph in zip(e.sent, e.phases):
            d_piv = float(np.linalg.norm(geo.t_of(rel) - geo.t_of(prev)))
            d_tcp = float(np.linalg.norm(e.tcp(rel)[:3, 3] - e.tcp(prev)[:3, 3]))
            d_rot = EX._angle_deg(prev[:3, :3].T @ rel[:3, :3])
            assert max(d_piv, d_tcp) <= v_step + 1e-9 and d_rot <= w_step + 1e-7, (ph, d_piv, d_tcp, d_rot)
            assert max(d_piv, d_tcp) > 1e-9 or d_rot > 1e-7, f"{ph}: a repeated (holding) command"
            assert ph.startswith(phase_of), ph
            counts[ph] = counts.get(ph, 0) + 1
            prev = rel
        last = {}
        for rel, ph in zip(e.sent, e.phases):
            last[ph] = rel
        for ph, tgt in targets.items():
            assert e._pose_error_mm(last[ph], tgt) < EX.MOVE_OUTSIDE_TOL, (ph, e._pose_error_mm(last[ph], tgt))
        return counts

    start = geo.outside(0.012, (0.0, 0.0), np.eye(3))
    # (a) translations only: the command count per waypoint is exactly ceil(distance / 0.135 mm), no extra commands
    wps = [Offset(0.004, (0.010, -0.010), (0.0, 0.0, 0.0)), Offset(0.018, (-0.010, 0.010), (0.0, 0.0, 0.0)),
           Offset(0.0101, (0.0003, 0.0), (0.0, 0.0, 0.0))]
    e = stand_in(start)
    res = e.execute("random", waypoints=wps)
    assert res["completed"], res
    tg = {f"random_motion_{i}": e.offset_rel(wp) for i, wp in enumerate(wps)}
    counts = check_steps(e, start, "random_motion_", tg)
    prev, want = start, {}
    for i, wp in enumerate(wps):
        d = float(np.linalg.norm(geo.t_of(tg[f"random_motion_{i}"]) - geo.t_of(prev)))
        want[f"random_motion_{i}"] = int(np.ceil(d / v_step))
        prev = tg[f"random_motion_{i}"]
    assert counts == want and len(e.sent) == sum(want.values()), (counts, want)
    assert [st["speed_m_s"] for st in e.stages] == [0.0027] * 3
    assert [st["ang_speed_deg_s"] for st in e.stages] == [2.0] * 3
    # (b) with rotations (the waypoint box corners, +-6 deg): the same step limits, every waypoint reached
    wps_r = [Offset(0.004, (0.010, -0.010), (6.0, -6.0, 6.0)), Offset(0.018, (-0.010, 0.010), (-6.0, 6.0, -6.0)),
             Offset(0.010, (0.0, 0.0), (0.0, 0.0, 0.0))]
    e = stand_in(start)
    res = e.execute("random", waypoints=wps_r)
    assert res["completed"], res
    check_steps(e, start, "random_motion_", {f"random_motion_{i}": e.offset_rel(wp) for i, wp in enumerate(wps_r)})
    # (c) wm_v1's 600-command limit: a waypoint 0.2 m away fails with "<phase>_did_not_reach_target"
    e = stand_in(start)
    res = e.execute("random", waypoints=[Offset(0.2, (0.0, 0.0), (0.0, 0.0, 0.0))])
    assert not res["completed"] and "random_motion_0_did_not_reach_target" in res["failure_reason"], res
    assert len(e.sent) == EX.MOVE_OUTSIDE_MAX_STEPS == 600
    # (d) speeds: None or the constant accepted, others refused before any motion
    e = stand_in(start)
    e.execute("random", waypoints=wps, speeds=[0.0027] * 3)
    for bad in ([0.003, 0.0027, 0.0027], [0.0027] * 2):
        e = stand_in(start)
        try:
            e.execute("random", waypoints=wps, speeds=bad)
            raise AssertionError(f"speeds {bad} accepted")
        except ValueError:
            assert not e.sent
    # (e) contact: approach / align to the stand-off, then the offset move by the same helper, the push from the offset
    # pose (wm_v1: _contact_push(target))
    e = stand_in(start)
    pushed = []
    e.stand_off = lambda: start
    e.approach_align = lambda target: None
    e.contact_push = lambda s0: (pushed.append(s0.copy()), {"outcome": "jammed"})[1]
    off = Offset(0.003, (0.003, 0.0), (3.0, -3.0, 3.0))
    res = e.execute("contact", offset=off)
    assert res["completed"] and res["outcome"] == "jammed", res
    tgt = e.offset_rel(off)
    n_c = check_steps(e, start, "contact_offset", {"contact_offset": tgt})["contact_offset"]
    assert len(pushed) == 1 and np.array_equal(pushed[0], tgt), "contact push starts at the offset pose"
    assert e.stages[0]["phase"] == "contact_offset" and e.stages[0]["speed_m_s"] == 0.0027
    d_c = max(float(np.linalg.norm(geo.t_of(tgt) - geo.t_of(start))),
              float(np.linalg.norm(e.tcp(tgt)[:3, 3] - e.tcp(start)[:3, 3])))
    assert n_c >= int(np.ceil(d_c / v_step)), (n_c, d_c)
    # (f) the step budget also caps random motion (r6-expert-01, information only): every command goes through the
    # real Execution._step, which raises at max_steps; the trajectory ends not completed with the failure text that
    # check_expert_data counts ("ExpertFailure: step_budget_<N>"), after exactly max_steps commands
    e = stand_in(start)
    e.max_steps, e.outside, e.on_step, e.q_sent = 50, True, None, None
    e.m.update(total_force=0.0, penetration_mm=0.0, force=0.0)
    e.r = types.SimpleNamespace(_command=lambda q, n: None)
    e._snapshot = lambda: None

    def send_step(rel, **kw):
        e._step(np.zeros(7))
        e.sent.append(rel.copy())
    e._send = send_step
    res = e.execute("random", waypoints=wps)
    assert not res["completed"] and res["failure_reason"] == "ExpertFailure: step_budget_50", res
    assert e.steps == 50 and len(e.sent) == 50, (e.steps, len(e.sent))
    # (g) row 24 (r7-decisions-01): the true plug's top gap is recorded over the outside steps only (information:
    # peak min_actual_top_gap_mm_outside, Chongyu's 1 mm gap guard is not ported), the 1 N contact guard is unchanged
    e = stand_in(start)
    e.peak = {"finger_socket_N": 0.0, "min_actual_top_gap_mm_outside": None}
    e.max_steps, e.outside, e.on_step, e.q_sent = 50, True, None, None
    e.r = types.SimpleNamespace(_command=lambda q, n: None)
    seq = iter([(0.004, 0.0), (0.0008, 0.0), (0.002, 0.0), (0.0001, 0.0), (0.003, 2.0)])

    def snap():
        gap, f = next(seq)
        e.m = dict(e.m, top_gap=gap, force=f, total_force=f, penetration_mm=0.0)
    e._snapshot = snap
    for _ in range(3):
        e._step(np.zeros(7))
    assert abs(e.peak["min_actual_top_gap_mm_outside"] - 0.8) < 1e-9, e.peak        # 0.8 mm < 1 mm: not refused
    e.outside = False
    e._step(np.zeros(7))
    assert abs(e.peak["min_actual_top_gap_mm_outside"] - 0.8) < 1e-9, "inside steps are not counted"
    e.outside = True
    try:
        e._step(np.zeros(7))
        raise AssertionError("plug-socket force 2 N before the insertion accepted")
    except EX.ExpertFailure as exc:
        assert str(exc) == "actual_plug_socket_contact_before_insertion", exc
    assert EX.OUTSIDE_CONTACT_N == 1.0
    return "16 named round-spin assemblies, geometry_v7 / ranges_v5, a listed n = 12 file refused, category kept; " \
        "perturbation window <= 3 mm/s (pivot and TCP) and <= 2 deg/s per step; random motion and contact offset by " \
        f"wm_v1's _move_outside: every step <= 0.135 mm and 0.1 deg, no hold, ceil(d / step) commands, 600-command " \
        f"limit, other speeds refused, push from the offset pose ({n_c} contact-offset commands); random motion over " \
        "the step budget ends as 'ExpertFailure: step_budget_<N>'; the outside true-plug top gap is recorded " \
        "(min_actual_top_gap_mm_outside, outside steps only), the 1 N contact guard unchanged"


# ------------------------------------------------------------------------------------------------------------- 9
def test_claims(tmp: Path):
    g = tmp / "claims" / "00001_g00"
    c = CO.claim(g)
    assert c is not None and c.attempt == 1
    assert CO.claim(g) is None, "a held claim is not taken again"
    CO.release(c, done=False)
    c = CO.claim(g)
    assert c.attempt == 1, "a released, unfinished claim does not count as an attempt"
    os.close(c.fd)                                                            # the holder dies (lock released, text kept)
    (g / ".claim").write_text("attempt 1 pid 999999999 start 1 time x")
    c = CO.claim(g)
    assert c.attempt == 2, c
    CO.release(c, done=True)
    assert not (g / ".claim").exists()
    assert CO._alive(os.getpid(), CO._start_time(os.getpid())) and not CO._alive(os.getpid(), "1")
    return "flock claims: held claims skipped, a dead holder counts as an attempt, liveness by pid + start time"


# ------------------------------------------------------------------------------------------------------------- 32
def test_hand_finger_command_clearance():
    from scipy.spatial.transform import Rotation
    from wm.env.sampling import Offset
    from wm.sim_automate import expert as EX
    assert EX.HAND_FINGER_COMMAND_CLEARANCE_M == EX.COMMAND_CLEARANCE_M == 0.001
    g = np.random.default_rng(7)

    def rnd_T():
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(g.normal(size=3) * 0.3).as_matrix()
        T[:3, 3] = g.normal(size=3) * 0.05
        return T
    # PlugGeometry: full_clearance = plug + hand_finger_clearance, fingers at +-q along the hand's y
    seen = []

    class FakeFcl:
        plug, hand, left, right, sock = "plug", "hand", "left", "right", "sock"

        def _collide(self, geom, T, other):
            return False

        def _dist(self, geom, T, other):
            seen.append((geom, T.copy()))
            return {"plug": 0.004, "hand": 0.003, "left": 0.002, "right": 0.0015}[geom]
    geo = EX.PlugGeometry.__new__(EX.PlugGeometry)
    geo.fcl = FakeFcl()
    geo.c, geo.x_ent, geo.g_top, geo.n = np.zeros(3), 0.0, 0.0, "round"
    geo.V = np.array([[x, y, z] for x in (-0.03, 0.0) for y in (-0.005, 0.005) for z in (-0.005, 0.005)])
    Ts, Ttp, q = rnd_T(), rnd_T(), 0.011
    fc = geo.full_clearance(Ts, Ttp, q)
    assert fc == {"plug": 0.004, "hand": 0.003, "left": 0.002, "right": 0.0015}, fc
    T = dict((k, v) for k, v in seen)
    hand = Ts @ np.linalg.inv(G.Tz(G.TCP_Z))
    assert np.allclose(T["plug"], Ts @ Ttp) and np.allclose(T["hand"], hand)
    for name, sgn in (("left", 1.0), ("right", -1.0)):
        assert np.allclose(T[name], hand @ G._T(p=[0.0, sgn * q, G.FINGER_JOINT_Z])), name
    # Execution._send: the guard on the nominal command, outside only
    socket_T = rnd_T()
    ex = EX.Execution.__new__(EX.Execution)
    ex.g = geo
    ex.G = socket_T @ C.R_G
    ex.h = Ttp @ C.R_G                                        # TCP^-1 CH with CH = plug R_G, plug = TCP T_tcp_plug
    ex.hi = EX._inv(ex.h)
    ex.r = types.SimpleNamespace(q_finger=q)
    ex.phase, ex.outside, ex.steps = "continuous_approach", True, 0
    ex.peak = {"min_command_clearance_mm": None, "min_command_hand_finger_clearance_mm": None}
    steps, frames = [], []
    ex._ik = lambda rel: np.zeros(9)
    ex._validate = lambda q9, rel, clr: None
    ex._step = lambda q7: steps.append(1)
    ex._compensate = lambda q9: q9
    val = {"d": 0.002}

    def hf(Tst, qf):
        frames.append((Tst.copy(), qf))
        return {"hand": 0.05, "left": val["d"], "right": 0.05}
    geo.hand_finger_clearance = hf
    rel = geo.outside(0.006, (0.002, -0.001), EX.euler_R([2.0, -1.0, 3.0]))
    ex._send(rel)
    assert len(steps) == 1 and abs(ex.peak["min_command_hand_finger_clearance_mm"] - 2.0) < 1e-12
    Tst, qf = frames[-1]
    assert qf == q and np.allclose(Tst, np.linalg.inv(socket_T) @ ex.tcp(rel)), "commanded TCP in the socket frame"
    val["d"] = 0.001                                          # exactly the limit passes
    ex._send(rel)
    assert len(steps) == 2 and abs(ex.peak["min_command_hand_finger_clearance_mm"] - 1.0) < 1e-12
    for d in (0.000999, -1.0):
        val["d"] = d
        try:
            ex._send(rel)
            raise AssertionError(f"hand / finger clearance {d} accepted")
        except EX.ExpertFailure as exc:
            assert str(exc) == "continuous_approach_nominal_command_guard: hand_finger_command_clearance", exc
        assert len(steps) == 2, "no step after a refused command"
    assert ex.peak["min_command_hand_finger_clearance_mm"] == -1000.0
    ex.outside = False                                        # insert / contact push: no check
    n = len(frames)
    ex._send(rel, compensate=False)
    assert len(frames) == n and len(steps) == 3
    # noisy candidates: the plug rule only (decision 7 is about the nominal command)
    ex.outside, val["d"] = True, 0.05
    ex.residual, ex.rng = np.zeros(7), np.random.default_rng(0)
    ex.rel_of_q = lambda q9: rel
    n = len(frames)
    ex._send(rel, approach_noise=True, compensate=False)
    assert len(frames) == n + 1, "one hand / finger check (the nominal command), none for the noisy candidates"
    # the execute() wrapper: the failure ends the trajectory as not completed with the distinct text
    e2 = EX.Execution.__new__(EX.Execution)
    e2.__dict__.update({k: v for k, v in ex.__dict__.items()})
    e2.stages, e2.perturbation_seen, e2.cmd = [], False, rel.copy()
    e2.m = {"top_gap": 0.0, "lateral": 0.0, "angle": 0.0, "axis_tilt": 0.0, "spin": 0.0, "success": False}
    e2.result, e2.success_any, e2.first_success_step = {"completed": False, "failure_reason": None}, False, None
    e2.peak = {"finger_socket_N": 0.0, "min_command_clearance_mm": None, "min_command_hand_finger_clearance_mm": None}
    val["d"] = 0.0005
    res = e2.execute("random", waypoints=[Offset(0.008, (0.004, 0.0), (0.0, 0.0, 0.0))])
    assert not res["completed"] and res["failure_reason"] == ("ExpertFailure: random_motion_0_nominal_command_guard: "
                                                              "hand_finger_command_clearance"), res
    assert abs(res["peak"]["min_command_hand_finger_clearance_mm"] - 0.5) < 1e-12 and e2.stages[-1]["failure"].endswith(
        "hand_finger_command_clearance")
    # Expert.waypoint_clearance: the same commanded pose as the guard when h = T_tcp_plug R_G (the redraw and the
    # guard agree)
    xp = EX.Expert.__new__(EX.Expert)
    xp.geom, xp.robot = geo, types.SimpleNamespace(T_tcp_plug=Ttp, q_finger=q)
    off = Offset(0.009, (0.003, -0.002), (1.0, -2.0, 4.0))
    val["d"] = 0.0004
    assert xp.waypoint_clearance(off) == 0.0004
    Tw = frames[-1][0]
    assert np.allclose(Tw, np.linalg.inv(socket_T) @ ex.tcp(ex.offset_rel(off))), "waypoint frame = guard frame"
    # collect.redraw_waypoints
    wps = [Offset(0.004 + 0.001 * i, (0.0, 0.0), (0.0, 0.0, 0.0)) for i in range(3)]
    bad = {0.005}
    clr = lambda o: 0.0005 if round(o.tip_gap_m, 6) in bad else 0.002  # noqa: E731
    out, rej = CO.redraw_waypoints(wps, clr, np.random.default_rng(3), 0.001)
    r2 = np.random.default_rng(3)
    assert out[0] is wps[0] and out[2] is wps[2] and out[1].as_dict() == CO.sample_start(r2, CO.WAYPOINT_BOX).as_dict()
    assert [r["waypoint"] for r in rej] == [1] and rej[0]["clearance_mm"] == 0.5 and rej[0]["offset"] == wps[1].as_dict()
    out, rej = CO.redraw_waypoints(wps, lambda o: -1.0, np.random.default_rng(3), 0.001, draws=4)
    assert len(rej) == 12 and [r["draw"] for r in rej[:4]] == [0, 1, 2, 3], rej
    r2 = np.random.default_rng(3)
    exp = [CO.sample_start(r2, CO.WAYPOINT_BOX) for _ in range(9)]
    assert [o.as_dict() for o in out] == [exp[2].as_dict(), exp[5].as_dict(), exp[8].as_dict()], \
        "the last of 4 draws (3 redraws) is kept"
    out, rej = CO.redraw_waypoints(wps, lambda o: 0.001, np.random.default_rng(3), 0.001)
    assert out == wps and rej == []
    assert CO.WAYPOINT_DRAWS == 20 and CO.STREAMS["waypoint"] == 6 and len(set(CO.STREAMS.values())) == len(CO.STREAMS)
    return "nominal outside command: hand / finger < 1 mm fails before the step with the distinct text, = 1 mm " \
           "passes, minimum recorded, no check inside, noisy candidates unchecked; full_clearance unchanged; " \
           "waypoint_clearance frame = guard frame; redraw_waypoints from its stream, last draw kept"


# ------------------------------------------------------------------------------------------------------------ 33
def test_move_to_still():
    """move_to stillness (2026-11-24): still = max|qvel[:7]| < 2e-3 rad/s OR the FK hand pose moved < 1 um and
    < 0.001 deg over one control step. The new rule only adds passes: the single new passing case is "qvel >= 2e-3
    and pose frozen" (Isaac's stale joint_vel); a moving pose that reports qvel 0 passed before and still passes; the
    position tolerance is never excused by stillness."""
    from wm.sim_automate import robot as R

    def T(dx=0.0, rot_deg=0.0):
        M = np.eye(4)
        c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
        M[:2, :2] = [[c, -s], [s, c]]
        M[0, 3] = dx
        return M

    def bare(qv):
        r = R.Robot.__new__(R.Robot)
        r.get_qvel = lambda: np.asarray(qv, np.float64)
        return r
    stale, zero = [0, 0, 0, 0, 0, 2.4e-3, 0, 0, 0], [0.0] * 9
    assert R.Robot._still(bare(stale), T(), T()) is True                               # frozen pose, stale qvel
    assert R.Robot._still(bare(zero), T(5e-6), T()) is True                            # the old rule alone
    assert R.Robot._still(bare(stale), T(2e-6), T()) is False                          # translation limit
    assert R.Robot._still(bare(stale), T(0.0, 0.002), T()) is False                    # rotation limit
    assert R.Robot._still(bare([0, 0, 0, 0, 0, 2e-3, 0, 0, 0]), T(1e-6), T()) is False  # both comparisons strict
    assert R.Robot._still(bare([0] * 7 + [0.5, 0.5]), T(2e-6), T()) is True            # only qvel[:7] counts

    def scripted(poses, qv):
        """hand_pose() = poses[i], i = control steps commanded so far (the last entry repeats); start pose poses[0]
        = goal gives one waypoint (2 control steps), so the pose before settling is poses[2]."""
        r = bare(qv)
        st = {"i": 0, "calls": 0}
        r.qpos = lambda: np.zeros(9)
        r._ik = lambda T_, seed: np.zeros(9)
        r.hand_pose = lambda: poses[min(st["i"], len(poses) - 1)]

        def cmd(q, n=1):
            st["i"] += n
            st["calls"] += 1
        r._command = cmd
        return r, st
    goal = T()
    # arrives within tolerance at settle step 2, unchanged at step 3 (stale qvel): True after 3 settle steps
    r, st = scripted([goal, T(5e-4), T(5e-4), T(2e-4), T(5e-5), T(5e-5)], stale)
    assert R.Robot.move_to(r, goal) is True and st["calls"] == 1 + 3, st
    # within tolerance but drifting 2 um per step (<= 84 um after 40 steps): never still
    r, st = scripted([goal] + [T(2e-6 * k) for k in range(1, 100)], stale)
    assert R.Robot.move_to(r, goal) is False and st["calls"] == 1 + 40, st
    # frozen 0.2 mm from the goal with qvel 0: stillness never excuses the tolerance
    r, st = scripted([goal, T(2e-4)], zero)
    assert R.Robot.move_to(r, goal) is False
    # frozen at the goal from the start: still on the first settle step
    r, st = scripted([goal], stale)
    assert R.Robot.move_to(r, goal) is True and st["calls"] == 2, st
    return "stale qvel with a frozen pose settles; 2 um / 0.002 deg per step and qvel == 2e-3 do not; old rule kept; " \
           "fingers ignored; tolerance binding"


if __name__ == "__main__":
    out = []
    for name, fn, needs_tmp in (("1 test_grasp_box", test_grasp_box, True), ("2 test_penetration", test_penetration, False),
                                ("5 test_reset_api", test_reset_api, False), ("7 test_force_sum", test_force_sum, False),
                                ("7b test_contact_report_rule", test_contact_report_rule, False),
                                ("9 test_claims", test_claims, True), ("11 test_frozen_plan", test_frozen_plan, True),
                                ("13 test_streams", test_streams, True), ("15 test_zero_width", test_zero_width, True),
                                ("22 test_online_guard", test_online_guard, False),
                                ("25 test_socket_pose", test_socket_pose, False),
                                ("26 test_truth_symmetry", test_truth_symmetry, False),
                                ("27 test_gpu_rule", test_gpu_rule, False),
                                ("28 test_start_and_label", test_start_and_label, False),
                                ("29 test_approved_switches", test_approved_switches, False),
                                ("30 test_physical_symmetry", test_physical_symmetry, False),
                                ("31 test_decisions_2026_11_23", test_decisions_2026_11_23, True),
                                ("32 test_hand_finger_command_clearance", test_hand_finger_command_clearance, False),
                                ("33 test_move_to_still", test_move_to_still, False)):
        with tempfile.TemporaryDirectory() as tmp:
            print(name, "OK:", fn(Path(tmp)) if needs_tmp else fn(), flush=True)
