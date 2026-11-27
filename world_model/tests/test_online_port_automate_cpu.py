"""Port plan T3-T6 (CPU, no Isaac app): the AutoMate side of the online attempts on stand-ins.

wm.sim_automate.robot (which imports Isaac Lab) is replaced by a stand-in module whose Robot class lives under
wm.sim_automate (so common320.simulator_of_robot calls it AutoMate) with the facade's interface (qpos, get_qvel,
hand_pose, frame(), _command, base.goal_pose / charger.pose, port_spec, max_decisions, planner shim for HandFK).
wm.sim_automate.record (the collection's Recorder: wm.data.recorder with its stand-ins), checks, config and frames are
the real ones.
  T3  the AutoMate recorder (shared320.FullRecorder with automate320's reads) stores, step for step, what the
      collection's Recorder (record.Recorder) stores on the same robot and commands: qpos, hand, eval_rel, rgb, depth,
      img_step, cam_T_world, K, action (the pictures are Robot.frame()'s, byte for byte); write_fast_automate's file
      passes demo320.entry_of, its entry has simulator "automate", the collection's split_sha256 and the grasp's
      condition split (split_of_entry), its meta keys are ONLINE_META_KEYS (all in FAST_META_AUTOMATE); a meta with
      episode_seed is refused (write_fast_automate and entry_of); the trajectory's labels (wmdata320.trajectory) are
      those of the same motion; prepare (build_job) refuses a checkpoint whose provenance split differs from the
      collection's, one without it, and an offset_mode "along" checkpoint. Server part (needs train_wm320 / torch,
      skipped only with WM_LOCAL_LIGHT=1): train_wm320.online_entries reads the pool entry, refuses it under another
      split sha256 and refuses a ManiSkill pool entry under an AutoMate fast copy.
  T4  attempt seeds: the same for every mode (prepare of act / act_done / wm gives the same reset seeds), independent
      of --grasps-n, new per round; a reset seed planted as a grasp seed of plan.json and --seed = the collection's
      entropy are refused; socket draw "per_grasp" (plan.json or config) is refused.
  T5  success_steps / truth_final on synthetic eval_rel: a successful start row and no later success -> success
      (automate_success_any) False, start_success True; AutomateTester raises StartSuccess on it; prepare refuses a
      START_GAP_FROM that differs from plan.json's. (The start-box share over all assemblies, 0 of 1,820,000 on
      2026-11-23, is not re-run: the start rule, ranges and geometry file are unchanged; scripts/automate/diag/
      start_success.py.)
  T6  summarise_automate: groups select / heldout_shape (cluster = assembly) / heldout_grasp_<variable> (band from
      plan.json grasp_kind, cluster = grasp) / test (cluster = assembly), the failure counts, the band-above-top count;
      split_identity_matches. (ManiSkill's summary: tests/test_eval_stats.py, eval_policy320 unchanged.)
  The AutoMate loop (decisions 3, 4, 5, 6): AutomateTester runs the loop with max_decisions = robot.max_decisions,
  refuses max_retries != 0, another max_decisions and probe_max > 0; left_m = 0.06 + L; face_gap_mm = s - L [mm];
  the labels' meta says simulator "automate"; first_success_decision = ceil(first_success_step / 8); a ManiSkill robot
  is refused.
  Audit fixes of 2026-11-25 (F1-F8): F1 collect claims strictly in the round's seeded order (claim_candidates), a test
  keeps the worker rotation and built-assembly preference, scene_build_s is in the grasp result; F2/F3 prepare refuses
  another asset folder, input_roots.assets, asset_files, transfer, switches (as a whole) or physics device, records
  assets and code identity; worker_checks refuses another asset folder / switches; collect --smoke-grasp needs a
  training grasp; QX_GPU_GB message; tries_before (deaths count as tries); F5 online_automate.sh act_done_options and
  paired (bash); F6 check_start, run_grasp records the start_success entry, finish exits 4, online_automate.sh
  stop_on_start_success writes <pool>/STOP and exits 1 (bash); F7 finish writes summary / manifests / DONE through
  _write_atomic; F8 --min-step. Re-audit R2: online_automate.sh round_files_differ, online_smoke.sh grasp list / act_done /
  exit-4 stop, online_smoke_check.py's rebuild table in run order; F2 also: --smoke-grasp comma lists, worker_checks
  refuses other running code; F7 also: the summary's physics_code_sha256 (online vs collection).

    WM_LOCAL_LIGHT=1 python world_model/tests/test_online_port_automate_cpu.py     (laptop: skips the train_wm320 part)
    python world_model/tests/test_online_port_automate_cpu.py                      (server)
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WM_ROOT = HERE.parent
sys.path.insert(0, str(WM_ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WM_ROOT / "scripts" / "online"))
import _online_port_stubs as S  # noqa: E402

LIGHT = os.environ.get("WM_LOCAL_LIGHT") == "1"
L_M = 0.016


# ---------------------------------------------------------------------------------------------- AutoMate stand-in
@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    seg: np.ndarray
    K: np.ndarray
    cam_T_world: np.ndarray
    qpos: np.ndarray
    hand: np.ndarray


class Pose:
    """wm.sim_automate.robot.Pose: p, q (w, x, y, z), inv(), *."""

    def __init__(self, T):
        from scipy.spatial.transform import Rotation
        self.T = np.asarray(T, np.float64)
        x, y, z, w = Rotation.from_matrix(self.T[:3, :3]).as_quat()
        self.p, self.q = self.T[:3, 3].copy(), np.array([w, x, y, z])

    def inv(self):
        return Pose(np.linalg.inv(self.T))

    def __mul__(self, other):
        return Pose(self.T @ other.T)


def install_robot_stand_in():
    import wm.sim_automate
    from wm.sim_automate import config as C
    mod = types.ModuleType("wm.sim_automate.robot")

    class Robot:
        """The facade's interface on the toy arm (H_goal: the seated TCP pose; the plug is the TCP)."""

        def __init__(self, H_goal, n_max=8, spec=None):
            self.arm = S.ToyArm(L_M)
            self.assembly_id = "00015"
            self.port_spec = spec or types.SimpleNamespace(pin_length_m=L_M, tip_centre_m=(0.0, 0.0, 0.0), symmetry_n=4,
                                                           disassembly_dist_m=0.01, depth_top_m=L_M)
            self._n_max, self.H_goal = n_max, np.asarray(H_goal, np.float64)
            r = self
            # goal_pose / charger.pose are read on every access (wm.data.recorder._truth_rel), as the facade's _Base
            base_cls = type("Base", (), {"goal_pose": property(lambda b: r.goal_pose()),
                                         "charger": property(lambda b: types.SimpleNamespace(pose=r.charger_pose()))})
            self.base = base_cls()
            self.base.__dict__.update(
                _planner=lambda: S.PlannerShim(),
                agent=types.SimpleNamespace(robot=types.SimpleNamespace(pose=S.BasePose(S.BASE_T),
                                                                        get_qvel=lambda: r.get_qvel()[None])),
                get_state_dict=lambda: r.arm.get_state(), set_state_dict=lambda s: r.arm.set_state(s))

        @property
        def max_decisions(self):
            return self._n_max

        def qpos(self):
            return self.arm.qpos()

        def get_qvel(self):
            return self.arm.v.copy()

        def hand_pose(self):
            return self.arm.hand_pose()

        def _command(self, q7, n=1):
            self.arm._command(q7, n)

        def goal_pose(self):
            return Pose(self.H_goal @ C.R_G)

        def charger_pose(self):
            return Pose(self.hand_pose() @ C.R_G)

        def frame(self):
            rgb, depth = self.arm.picture(40, 40)
            r0, r1, c0, c1 = 4, 36, 4, 36
            depth = depth.copy()
            depth[depth < C.DEPTH_MIN_M] = 0.0
            d = np.clip(np.round(depth[r0:r1, c0:c1] / C.DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
            K = np.array([[30.0, 0, 20.0], [0, 30.0, 20.0], [0, 0, 1.0]])
            K[0, 2] -= c0
            K[1, 2] -= r0
            return Frame(rgb=np.ascontiguousarray(rgb[r0:r1, c0:c1]), depth=d, seg=np.zeros(d.shape, np.uint8), K=K,
                         cam_T_world=np.linalg.inv(self.hand_pose()), qpos=self.qpos(), hand=self.hand_pose())

    Robot.__module__ = "wm.sim_automate.robot"
    mod.Robot, mod.Frame, mod.Pose = Robot, Frame, Pose
    sys.modules["wm.sim_automate.robot"] = mod
    wm.sim_automate.robot = mod
    return Robot


def make_robot(Robot, H_goal, **kw):
    return Robot(H_goal, **kw)


def setup():
    Robot = install_robot_stand_in()
    arm = S.ToyArm(L_M)
    axis = np.array([0.2, -0.1, -0.97])
    axis /= np.linalg.norm(axis)
    H_goal = arm.hand_pose().copy()
    H_goal[:3, 3] += (L_M + 0.003) * axis
    return Robot, H_goal, axis


# ---------------------------------------------------------------------------------------------- fixture collection
SHARED_ASSETS: dict = {"dir": None}


def use_shared_assets(tmp: Path) -> Path:
    """One asset folder for every fixture collection (each collection's inputs/assets links to it), set as this
    process's wm.sim_automate.assets.ASSET_DIR (what WM_AUTOMATE_ASSETS sets on the server)."""
    from wm.sim_automate import assets as AS
    d = tmp / "shared_assets"
    d.mkdir(parents=True, exist_ok=True)
    SHARED_ASSETS["dir"] = d
    AS.ASSET_DIR = d
    return d


def collection(root: Path, plan_overrides: dict | None = None, grasp_seeds: dict | None = None,
               transfer: str = "move") -> Path:
    """A minimal AutoMate collection: plan.json (split automate_v1, switches, physics device, input_roots.assets =
    <root>/inputs/assets (a link to the shared asset folder), asset_files, grasps with gi / seed / grasp_kind, rng),
    config.json (transfer), inputs/switches.env, split_automate_v1.json, <aid>/<aid>_gNN/grasp.json."""
    from wm.sim_automate import collect as CO
    from wm.sim_automate import config as C
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    if not (root / "inputs" / "assets").exists():
        (root / "inputs" / "assets").symlink_to(SHARED_ASSETS["dir"], target_is_directory=True)
    (root / "inputs" / "switches.env").write_text("WM_AUTOMATE_SOCKET_DRAW=per_reset\n")
    (root / "config.json").write_text(json.dumps({"transfer": transfer}))
    grasps = [("00015_g00", "train", "train_range"), ("00015_g01", "train", "train_range"), ("00015_g02", "select", "train_range"),
              ("00117_g00", "heldout_grasp", "band_dtheta_deg"), ("00117_g01", "heldout_grasp", "band_dp_m"),
              ("00133_g00", "heldout_shape", "train_range"), ("00133_g01", "heldout_shape", "train_range"),
              ("00444_g00", "test", "train_range"), ("00470_g00", "select", "train_range"),
              ("00117_g02", "train", "train_range"), ("00470_g01", "train", "train_range"),
              ("00863_g00", "train", "train_range"), ("00863_g01", "train", "train_range")]
    doc = {"version": "automate_v1", "test": ["00444"], "heldout_shape": ["00133"]}
    plan = {"plan_sha": "p" * 64, "config_sha": "c" * 64, "switches": C.switches(), "physics_device": C.PHYSICS_DEVICE,
            "input_roots": {"assets": str(root / "inputs" / "assets"), "m1b_dirs": []}, "asset_files": CO.asset_files(),
            "m1b_code": {"field": "physics_code_sha256", "sha256": "m" * 64},
            "rng": {"entropy": 20261122, "streams": {"grasp_values": 0, "trajectory_plan": 1, "placement": 2, "expert": 3,
                                                       "episode": 4, "reference_label": 5, "waypoint": 6}},
            "split": {"name": "automate_v1", "file": "split_automate_v1.json", "document": doc, "counts": {}},
            "grasps": [{"grasp": k, "port": k[:5], "condition": c, "grasp_kind": kind, "gi": int(k[-2:]),
                        "seed": (grasp_seeds or {}).get(k, 1000 + i)} for i, (k, c, kind) in enumerate(grasps)]}
    plan.update(plan_overrides or {})
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "split_automate_v1.json").write_text(json.dumps({**doc, "plan_sha": plan["plan_sha"]}))
    for k, c, kind in grasps:
        d = root / k[:5] / k
        d.mkdir(parents=True, exist_ok=True)
        (d / "grasp.json").write_text(json.dumps({"grasp": k, "port": k[:5], "condition": c, "pitch_deg": 1.0,
                                                  "grasp_pos_m": 0.001, "spin_deg": 2.0, "u": {"dtheta_deg": 0.5},
                                                  "simulator": "automate", "assembly_id": k[:5], "reference_ok": True,
                                                  "episode_seed": 1}))
    return root


def ckpt(path: Path, split: dict | None, offset_mode: str = "gap") -> Path:
    import torch
    torch.save({"offset_mode": offset_mode, "offset_weight": 1.0, "step": 77,
                "provenance": {"split": split} if split is not None else {}}, path)
    return path


def job_args(grasps: Path, ck: Path, **kw) -> Namespace:
    import run_wm320_automate as RAA
    a = RAA.prepare_parser().parse_args(["--grasps", str(grasps), "--ckpt", str(ck), "--mode", "wm", "--split", "select",
                                         "--out", str(grasps.parent / "out")])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def refused(fn, text: str) -> str:
    try:
        fn()
    except SystemExit as exc:
        assert text in str(exc), (text, str(exc))
        return str(exc)
    raise AssertionError(f"not refused ({text})")


# ---------------------------------------------------------------------------------------------- tests
def test_t3(tmp: Path) -> str:
    Robot, H_goal, axis = setup()
    from wm.online import automate320 as A
    from wm.online.shared320 import FullRecorder
    from wm.sim_automate import record
    from wm.tdmpc import demo320 as D
    from wm.tdmpc import wmdata320 as WD
    import run_wm320_automate as RAA
    r1, r2 = make_robot(Robot, H_goal), make_robot(Robot, H_goal)
    fr, rr = FullRecorder(r1, A.automate_io()), record.Recorder(r2)
    fr.begin()
    rr.begin()
    rng = np.random.default_rng(3)
    q = r1.qpos()[:7].copy()
    for t in range(37):
        q = q + rng.normal(size=7) * 0.003
        qq = q.astype(np.float32).astype(float)
        r1._command(qq, 1)
        r2._command(qq, 1)
        fr.step(qq, qq)
        rr.on_step("insert", qq, {})
    a, b = fr.end(), rr.end()
    for k in ("qpos", "hand", "eval_rel", "rgb", "depth", "img_step", "cam_T_world", "K", "action"):
        assert np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype, k
    # write_fast_automate
    from wm.online.common320 import jpeg
    out = {**a, "rgb_jpeg_list": [jpeg(x) for x in a["rgb"]], "dec_kind": np.zeros(len(a["action"]) // 8, np.int8)}
    grasps = collection(tmp / "coll")
    sid = D.split_identity(grasps)
    g = {"grasp": "00015_g00", "port": "00015", "condition": "train", "pitch_deg": 1.0, "grasp_pos_m": 0.001, "spin_deg": 2.0,
         "u": {"dtheta_deg": 0.5}, "simulator": "automate", "assembly_id": "00015"}
    entry = {"H_star": H_goal.tolist(), "axis": axis.tolist(), "pin_length_m": L_M,
             "start": {"tip_gap_mm": 9.0, "yz_mm": [1.0, -2.0], "rpy_deg": [1.0, 0.0, -2.0]}}
    rel = "r1/traj/00015/00015_g00/s0.npz"
    meta = RAA.online_meta(g, entry, {"seated_fk": False}, "online", rel, {"config_sha": "c", "plan_sha": "p"}, {"x": 1}, 4)
    assert set(meta) | {"steps", "format"} == set(RAA.ONLINE_META_KEYS) and set(RAA.ONLINE_META_KEYS) <= set(D.FAST_META_AUTOMATE)
    pool = tmp / "pool"
    e = RAA.write_fast_automate(pool / rel, out, meta, sid["sha256"])
    assert e["simulator"] == "automate" and e["split_sha256"] == sid["sha256"] and D.split_of_entry(e) == "train"
    assert e["dec_kind"] == [0] * (len(a["action"]) // 8) and e["kind"] == "online" and e["outcome"] == "not_seated"
    with np.load(pool / rel) as z:
        m = json.loads(str(z["meta"]))
        assert set(m) == set(RAA.ONLINE_META_KEYS) and "dec_kind" not in m and list(z["dec_kind"]) == e["dec_kind"]
        tr = WD.trajectory(z, m, 0.3, 0.967, offset_mode="gap")
    lab = WD.decision_labels({**m, "simulator": "automate"}, a["hand"].astype(np.float64), 8 * np.arange(len(a["action"]) // 8 + 1),
                             offset_mode="gap")
    assert np.array_equal(tr["probe"], lab["probe"]) and np.array_equal(tr["done"], lab["seated"].astype(np.float32))
    for bad in ({"episode_seed": 3}, {"socket_pose": [1]}):
        try:
            RAA.write_fast_automate(pool / "x.npz", out, {**meta, **bad}, sid["sha256"])
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass
        try:
            D.entry_of({**meta, **bad, "steps": 8, "format": D.FORMAT}, rel, 3)
            raise AssertionError(f"entry_of accepted {bad}")
        except ValueError:
            pass
    # prepare refuses a checkpoint of another split, one without a split, an "along" checkpoint
    ok = ckpt(tmp / "ok.pt", sid)
    W, job = RAA.build_job(job_args(grasps, ok))
    assert [x["grasp"] for x in job["grasps"]] == ["00015_g02", "00470_g00"] and job["split_identity"] == sid
    refused(lambda: RAA.build_job(job_args(grasps, ckpt(tmp / "other.pt", {**sid, "sha256": "0" * 64}))), "trained on the split")
    refused(lambda: RAA.build_job(job_args(grasps, ckpt(tmp / "none.pt", None))), "trained on the split")
    refused(lambda: RAA.build_job(job_args(grasps, ckpt(tmp / "along.pt", sid, "along"))), "offset_mode 'gap'")
    refused(lambda: RAA.build_job(job_args(grasps, ok, max_retries=1)), "--max-retries must be 0")
    refused(lambda: RAA.build_job(job_args(grasps, ok, probe_max=2)), "--probe-max must be 0")
    msg = "train_wm320 part skipped (WM_LOCAL_LIGHT=1; run it on the server)"
    if not LIGHT:
        sys.path.insert(0, str(WM_ROOT / "scripts" / "train"))
        import train_wm320 as T
        (pool / "r1" / "manifest.json").write_text(json.dumps({"entries": [e]}))
        (pool / "r1" / "DONE").write_text("{}")
        got, rounds = T.online_entries(pool, sid["sha256"])
        assert rounds == ["r1"] and len(got) == 1 and got[0]["bc"] is False and got[0]["split"] == "train"
        refused(lambda: T.online_entries(pool, "f" * 64), "carries the split")
        ms = {k: v for k, v in e.items() if k not in ("simulator", "split_sha256")}
        (pool / "r1" / "manifest.json").write_text(json.dumps({"entries": [ms]}))
        refused(lambda: T.online_entries(pool, sid["sha256"]), "fast copy is AutoMate")
        msg = "train_wm320.online_entries reads the entry, refuses another split and a ManiSkill entry"
    return f"recorder = collection Recorder on {len(a['img_step'])} pictures; fast copy entry ok; split checks ok; {msg}"


def test_t4(tmp: Path) -> str:
    from wm.online import automate320 as A
    from wm.sim_automate import config as C
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    grasps = collection(tmp / "c4")
    sid = D.split_identity(grasps)
    ok = ckpt(tmp / "ok4.pt", sid)
    resets = {}
    for mode, extra in (("act", {}), ("act_done", {}), ("wm", {"inside_mm": 1.0})):
        a = job_args(grasps, ok, mode=mode, starts=3, out=str(tmp / f"o_{mode}"), **extra)
        _, job = RAA.build_job(a)
        resets[mode] = [A.attempt_streams(job["seed"], g["port"], g["gi"], i)["reset_seed"] for g in job["grasps"] for i in range(3)]
    assert resets["act"] == resets["act_done"] == resets["wm"]
    s1 = A.attempt_streams(9_100_000, "00015", 2, 1)
    _, job_n = RAA.build_job(job_args(grasps, ok, grasps_n=1, starts=3, out=str(tmp / "o_n")))
    g1 = job_n["grasps"][0]
    assert A.attempt_streams(job_n["seed"], g1["port"], g1["gi"], 1)["reset_seed"] == \
        A.attempt_streams(9_100_000, g1["port"], g1["gi"], 1)["reset_seed"]
    r1, r2 = A.attempt_streams(9_100_000, "00015", 2, 1, 1), A.attempt_streams(9_100_000, "00015", 2, 1, 2)
    assert len({s1["reset_seed"], r1["reset_seed"], r2["reset_seed"]}) == 3 and s1["attempt_seed"] != r1["attempt_seed"]
    assert all(np.random.default_rng(s1[k]).random() != np.random.default_rng(r1[k]).random() for k in ("start", "place", "reference"))
    assert not set(A.PURPOSES.values()) & {0, 1, 2, 3, 4, 5, 6}
    # a planted collision: grasp 00015_g01's collection seed = the reset seed of 00015_g02's attempt 1
    planted = collection(tmp / "c4b", grasp_seeds={"00015_g01": s1["reset_seed"]})
    sid_b = D.split_identity(planted)
    refused(lambda: RAA.build_job(job_args(planted, ckpt(tmp / "okb.pt", sid_b), starts=3, out=str(tmp / "ob"))), "equal a grasp seed")
    refused(lambda: RAA.build_job(job_args(grasps, ok, seed=20261122, out=str(tmp / "oc"))), "collection's entropy")
    pg = collection(tmp / "c4c", plan_overrides={"switches": {**C.switches(), "socket_draw": "per_grasp"}})
    refused(lambda: RAA.build_job(job_args(pg, ckpt(tmp / "okc.pt", D.split_identity(pg)), out=str(tmp / "od"))), "per_reset")
    old = C.SOCKET_DRAW
    try:
        C.SOCKET_DRAW = "per_grasp"
        refused(lambda: A.check_switches(json.loads((grasps / "plan.json").read_text())), "per_reset")
    finally:
        C.SOCKET_DRAW = old
    return "seeds equal across modes and --grasps-n, new per round; collision, entropy and per_grasp refused"


def test_t5(tmp: Path) -> str:
    Robot, H_goal, axis = setup()
    from wm.online import automate320 as A
    from wm.sim_automate import config as C
    from wm.sim_automate import frames
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    spec = types.SimpleNamespace(pin_length_m=L_M, tip_centre_m=(0.0, 0.0, 0.0), symmetry_n=4, disassembly_dist_m=0.01,
                                 depth_top_m=L_M)
    rel = np.zeros((10, 7))
    rel[:, 3] = 1.0
    rel[0, 0] = -0.005                                   # start: 5 mm out on the axis: AutoMate's rule holds (h 5 mm)
    rel[1:, 0] = -0.03                                   # then 30 mm out: it does not
    t = A.truth_final(rel, spec)
    assert t["success"] is False and t["automate_success_any"] is False and t["start_success"] is True
    assert t["first_success_step"] is None and t["success_end"] is False
    rel2 = rel.copy()
    rel2[0, 0], rel2[7, 0] = -0.03, -0.004
    t2 = A.truth_final(rel2, spec)
    assert t2["success"] and t2["first_success_step"] == 7 and not t2["start_success"] and not t2["success_end"]
    # AutomateTester stops the run on a successful start
    r = make_robot(Robot, H_goal, n_max=3)
    r.H_goal = r.hand_pose().copy()
    r.H_goal[:3, 3] += 0.005 * axis                       # the hand starts 5 mm from the seated pose: s_0 holds
    F = frames.frame(axis, S.BASE_T[:3, 0])
    from wm.data.kinematics import HandFK
    tester = A.AutomateTester(r, H_goal, axis, HandFK(r), F)
    try:
        tester.attempt(S.StubAgent("stall"), "act", 5)
        raise AssertionError("a successful start was accepted")
    except A.StartSuccess:
        pass
    # prepare refuses a start gap rule other than plan.json's
    grasps = collection(tmp / "c5", plan_overrides={"switches": {**C.switches(), "start_gap_from": "entrance"}})
    refused(lambda: RAA.build_job(job_args(grasps, ckpt(tmp / "ok5.pt", D.split_identity(grasps)))), "start gap from")
    return "start row counted apart (success False, start_success True); StartSuccess raised; START_GAP_FROM checked"


def test_t6(tmp: Path) -> str:
    import run_wm320_automate as RAA
    out = tmp / "res"
    gs = [{"grasp": "00015_g02", "port": "00015", "condition": "select", "grasp_kind": "train_range"},
          {"grasp": "00470_g00", "port": "00470", "condition": "select", "grasp_kind": "train_range"},
          {"grasp": "00117_g00", "port": "00117", "condition": "heldout_grasp", "grasp_kind": "band_dtheta_deg"},
          {"grasp": "00117_g01", "port": "00117", "condition": "heldout_grasp", "grasp_kind": "band_dp_m"},
          {"grasp": "00133_g00", "port": "00133", "condition": "heldout_shape", "grasp_kind": "train_range"},
          {"grasp": "00133_g01", "port": "00133", "condition": "heldout_shape", "grasp_kind": "train_range"},
          {"grasp": "00444_g00", "port": "00444", "condition": "test", "grasp_kind": "train_range"},
          {"grasp": "00863_g00", "port": "00863", "condition": "select", "grasp_kind": "train_range"}]
    succ = {"00015_g02": [True, False], "00470_g00": [True, True], "00117_g00": [False, True], "00117_g01": [False, False],
            "00133_g00": [True, False], "00133_g01": [True, True], "00444_g00": [False, False]}
    for g in gs[:-1]:
        f = out / g["port"] / f"{g['grasp']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        starts = [{"idx": i, "attempt": {"success": ok, "stopped": "seated" if ok else "time", "band_above_top_mm": 1.0 if g["port"] == "00133" else -2.0,
                                         "start_success": False, "seated_fk": ok, "interventions": 0, "retries": 0}}
                  for i, ok in enumerate(succ[g["grasp"]])]
        starts.append({"idx": 2, "outcome": "placement_failed"} if g["port"] != "00117" else {"idx": 2, "outcome": "reference_failed"})
        f.write_text(json.dumps({"grasp": g["grasp"], "starts": starts}))
    s, complete = RAA.summarise_automate(out, gs, 3, {})
    G = s["groups"]
    assert not complete and s["crashed_grasps"] == 1 and s["judged_starts"] == 14
    assert s["placement_failed_starts"] == 5 and s["reference_failed_starts"] == 2
    assert set(G) == {"all", "select", "heldout_shape", "heldout_grasp_dtheta_deg", "heldout_grasp_dp_m", "test"}, set(G)
    assert G["heldout_shape"]["clusters"] == 1 and G["heldout_shape"]["starts"] == 4 and G["heldout_shape"]["success"] == 0.75
    assert G["select"]["clusters"] == 2 and G["test"]["clusters"] == 1 and G["heldout_grasp_dp_m"]["success"] == 0.0
    assert G["heldout_shape"]["successes_on_band_above_top"] == 3 and G["select"]["successes_on_band_above_top"] == 0
    assert RAA.group_of({"grasp": "x_g00", "port": "x", "condition": "train"}) == ("train", "x_g00")
    sid = {"name": "automate_v1", "file": "split_automate_v1.json", "sha256": "a" * 64, "plan_sha": "p"}
    assert RAA.split_identity_matches(sid, dict(sid)) == []
    assert RAA.split_identity_matches({**sid, "plan_sha": "q"}, sid) == ["plan_sha"]
    assert RAA.split_identity_matches(None, sid) == ["missing"]
    return f"groups {sorted(G)}; heldout_shape clustered by assembly; counts and band successes"


def test_loop(tmp: Path) -> str:
    Robot, H_goal, axis = setup()
    from wm.data.kinematics import HandFK
    from wm.online import automate320 as A
    from wm.online.common320 import simulator_of_robot
    from wm.sim_automate import frames
    from wm.tdmpc import wmdata320 as WD
    r = make_robot(Robot, H_goal, n_max=7)
    assert simulator_of_robot(r) == "automate"
    F = frames.frame(axis, S.BASE_T[:3, 0])
    t = A.AutomateTester(r, H_goal, axis, HandFK(r), F)
    assert abs(t.left_m - (0.06 + L_M)) < 1e-15 and t.label_meta == {"simulator": "automate"}
    s_mm = -float((r.hand_pose()[:3, 3] - H_goal[:3, 3]) @ axis) * 1000.0
    assert abs(t.face_gap_mm() - (s_mm - L_M * 1000.0)) < 1e-9
    for kw, text in (({"max_retries": 1}, "decision 5"), ({"max_decisions": 150}, "decision 6"), ({"probe_max": 1}, "T8")):
        try:
            t.attempt(S.StubAgent(), "wm", 1, **kw)
            raise AssertionError(kw)
        except ValueError as exc:
            assert text in str(exc), (kw, str(exc))
    res = {}
    for mode, kw in (("act", {}), ("act_done", {}), ("wm", {"inside_mm": 1.0, "margin": 0.001}),
                     ("collect", {"margin": 0.001, "p_shift": 1.0, "eps": 0.5}),
                     ("wm", {"inside_mm": 1.0, "planner": "mppi", "max_retries": 0}),
                     ("collect", {"inside_mm": 1.0, "planner": "mppi", "max_retries": 0, "p_shift": 1.0, "eps": 0.5})):
        rr = make_robot(Robot, H_goal, n_max=7)
        tt = A.AutomateTester(rr, H_goal, axis, HandFK(rr), F)
        summ, out = tt.attempt(S.StubAgent(), mode, 11, **kw)
        assert summ["decisions"] <= 7 and summ["max_decisions"] == 7 and summ["stopped"] in ("time", "stalled", "seated", "left")
        assert summ["success"] == summ["automate_success_any"] and summ["start_success"] is False
        fs = summ["first_success_step"]
        assert summ["first_success_decision"] == (None if fs is None else math.ceil(fs / 8))
        assert out["eval_tip"].shape == (len(out["eval_rel"]), 9) and len(summ["truth_ok"]) == summ["decisions"]
        lab = WD.decision_labels({"H_star": H_goal, "axis": F[:, 0], "pin_length_m": L_M, "simulator": "automate"},
                                 out["hand"].astype(np.float64), np.array([8 * summ["decisions"]]))
        assert summ["seated_fk"] == bool(lab["seated"][0])
        if kw.get("planner") == "mppi":                 # MPPI: every agent decision planned outside the hole (wm),
            pdec = summ["per_decision"]                  # always (collect); noisy plans stored as exploration
            ia = summ["inside_at"]
            if mode == "wm":                             # the in-hole rule switches MPPI off in the test mode only
                assert not any(d["offered"] for d in pdec if ia is not None and d["k"] >= ia), (ia, pdec)
            assert any(d["offered"] for d in pdec), pdec
            if mode == "wm":
                assert all(not d.get("noisy") for d in pdec) and summ["random_candidates"] == 0
            assert summ["interventions"] == sum(1 for d in pdec if d.get("pick") == 1 and not d.get("noisy")), (summ["interventions"], pdec)
            assert "strict_success" in summ
        res[mode + ("_mppi" if kw.get("planner") else "")] = f"{summ['stopped']}/{summ['decisions']}"
    ms = type("Robot", (), {"__module__": "wm.env.robot"})()
    try:
        A.AutomateTester(ms, H_goal, axis, None, F)
        raise AssertionError("a ManiSkill robot was accepted")
    except ValueError:
        pass
    return f"limits refused; left_m = 0.06 + L; face gap = s - L; runs {res}"


# ---------------------------------------------------------------------------------------------- audit fixes (2026-11-25)
AUTOMATE_SH = WM_ROOT / "scripts" / "automate" / "online_automate.sh"


def shell_function(script: Path, name: str) -> str:
    """The text of the bash function `name` of `script` (from "name() {" to the first line "}")."""
    m = re.search(rf"^{name}\(\) \{{.*?^\}}\n", script.read_text(), re.S | re.M)
    assert m, f"{name} not in {script}"
    return m.group(0)


def bash(script: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", "set -u\n" + script], capture_output=True, text=True,
                          env={**os.environ, "CL_PY": sys.executable, **env})


def collect_args(grasps: Path, ck: Path, pool: Path, rnd: int = 1, **kw) -> Namespace:
    return job_args(grasps, ck, mode="collect", split="train", pool=str(pool), round=rnd, out=None, **kw)


def test_f1_claim_order(tmp: Path) -> str:
    """Fix 1: collect claims strictly in the round's seeded permutation (every worker, any built assembly); a test
    rotates by the worker index and prefers the built assembly; scene_build_s in the grasp result (run_grasp)."""
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    grasps = collection(tmp / "f1")
    ok = ckpt(tmp / "f1.pt", D.split_identity(grasps))
    jobs = {r: RAA.build_job(collect_args(grasps, ok, tmp / "pool1", r))[1] for r in (1, 2)}
    train = sorted(g["grasp"] for g in jobs[1]["grasps"])
    assert len(train) == 6 and len({g["port"] for g in jobs[1]["grasps"]}) == 4
    for r, job in jobs.items():
        keys = [g["grasp"] for g in job["grasps"]]
        perm = np.random.default_rng(job["seed"] + 1_000_003 * r).permutation(len(keys))
        assert job["order"] == [keys[i] for i in perm], (r, job["order"])
    assert jobs[1]["order"] != jobs[2]["order"]
    job = jobs[1]
    every = lambda k: True
    for index, count, port in ((0, 1, None), (0, 3, "00015"), (2, 3, "00863"), (1, 2, job["grasps"][-1]["port"])):
        assert RAA.claim_candidates(job, index, count, port, every, {}) == job["order"], (index, port)
    first = job["order"][0]
    assert RAA.claim_candidates(job, 1, 2, "00863", lambda k: k != first, {}) == job["order"][1:]
    blocked = {job["grasps"][0]["port"]: "x"}
    assert RAA.claim_candidates(job, 0, 1, None, every, blocked) == [
        k for k in job["order"] if {g["grasp"]: g["port"] for g in job["grasps"]}[k] not in blocked]
    # a test keeps the worker rotation and the built-assembly preference
    _, tj = RAA.build_job(job_args(grasps, ok, out=str(tmp / "f1t")))
    port_of = {g["grasp"]: g["port"] for g in tj["grasps"]}
    k0 = (1 * len(tj["order"])) // 2
    rot = tj["order"][k0:] + tj["order"][:k0]
    built = port_of[tj["order"][0]]
    assert RAA.claim_candidates(tj, 1, 2, built, every, {}) == sorted(rot, key=lambda k: port_of[k] != built)
    # scene_build_s in the grasp result
    from wm.online import automate320 as A
    robot = types.SimpleNamespace(max_decisions=7, port_spec=types.SimpleNamespace(
        disassembly_dist_m=0.01, depth_top_m=L_M, symmetry_n=4))
    old = A.run_attempt
    try:
        A.run_attempt = lambda *a, **k: ({"idx": a[4], "outcome": "placement_failed"}, None, None)
        s, scene_ok = RAA.run_grasp(robot, None, types.SimpleNamespace(step=77), job["grasps"][0], job, tmp / "f1w",
                                    lambda *a, **k: None, 1, 12.34)
        assert scene_ok and "error" not in s and s["scene_build_s"] == 12.3 and len(s["starts"]) == job["starts"]
        s2, _ = RAA.run_grasp(robot, None, types.SimpleNamespace(step=77), job["grasps"][0], job, tmp / "f1w",
                              lambda *a, **k: None, 1, None)
        assert s2["scene_build_s"] is None and s2["code"] == job["code"] and s2["assets"] == job["assets"]
    finally:
        A.run_attempt = old
    return f"collect order = the round's permutation for every worker ({job['order'][:3]}...); test rotation kept; scene_build_s recorded"


def test_f2_f3_inputs(tmp: Path) -> str:
    """Fixes 2, 3, 4, 10: frozen assets (prepare, worker_checks), switches as a whole and the physics device, transfer,
    code identity, collect --smoke-grasp on a training grasp only, the QX_GPU_GB message, tries_before."""
    from wm.sim_automate import assets as AS
    from wm.sim_automate import collect as CO
    from wm.sim_automate import config as C
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    grasps = collection(tmp / "f2")
    ok = ckpt(tmp / "f2.pt", D.split_identity(grasps))
    _, job = RAA.build_job(job_args(grasps, ok, out=str(tmp / "f2o")))
    assert Path(job["assets"]).resolve() == SHARED_ASSETS["dir"].resolve()
    assert job["physics_device"] == C.PHYSICS_DEVICE and job["switches"] == C.switches()
    assert job["code"] == {"root": str(WM_ROOT), "tree_sha256": None, "physics_code_sha256": CO.physics_code_identity()}
    # assets
    other = tmp / "other_assets"
    other.mkdir()
    try:
        AS.ASSET_DIR = other
        refused(lambda: RAA.build_job(job_args(grasps, ok, out=str(tmp / "f2o"))), "WM_AUTOMATE_ASSETS")
    finally:
        AS.ASSET_DIR = SHARED_ASSETS["dir"]
    g2 = collection(tmp / "f2b", plan_overrides={"input_roots": {"assets": str(other)}})
    refused(lambda: RAA.build_job(job_args(g2, ckpt(tmp / "f2b.pt", D.split_identity(g2)), out=str(tmp / "o"))),
            "input_roots.assets")
    g3 = collection(tmp / "f2c", plan_overrides={"asset_files": {**CO.asset_files(), "spec_file": "geometry_v0.json"}})
    refused(lambda: RAA.build_job(job_args(g3, ckpt(tmp / "f2c.pt", D.split_identity(g3)), out=str(tmp / "o"))),
            "asset_files")
    g4 = collection(tmp / "f2d", transfer="teleport")
    refused(lambda: RAA.build_job(job_args(g4, ckpt(tmp / "f2d.pt", D.split_identity(g4)), out=str(tmp / "o"))),
            "transfer")
    # switches as a whole, physics device
    g5 = collection(tmp / "f2e", plan_overrides={"switches": {**C.switches(), "cam_scale": 2}})
    refused(lambda: RAA.build_job(job_args(g5, ckpt(tmp / "f2e.pt", D.split_identity(g5)), out=str(tmp / "o"))),
            "switches differ from plan.json")
    g6 = collection(tmp / "f2f", plan_overrides={"physics_device": "cuda" if C.PHYSICS_DEVICE == "cpu" else "cpu"})
    refused(lambda: RAA.build_job(job_args(g6, ckpt(tmp / "f2f.pt", D.split_identity(g6)), out=str(tmp / "o"))),
            "physics device")
    # worker_checks: the job's assets, switches, device; the QX_GPU_GB message
    env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "QX_GPU_GB")}
    old_dev = C.PHYSICS_DEVICE
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES="1", QX_GPU_GB="8")
        RAA.worker_checks(job)
        AS.ASSET_DIR = other
        refused(lambda: RAA.worker_checks(job), "differs from the job's")
        AS.ASSET_DIR = SHARED_ASSETS["dir"]
        C.PHYSICS_DEVICE = "cuda" if old_dev == "cpu" else "cpu"
        refused(lambda: RAA.worker_checks(job), "differ from the job's")
        C.PHYSICS_DEVICE = old_dev
        refused(lambda: RAA.worker_checks({**job, "code": {**job["code"], "physics_code_sha256": "0" * 64}}),
                "running code")
        os.environ["QX_GPU_GB"] = "2"
        refused(lambda: RAA.worker_checks(job), "does not include Kit's own GPU memory")
    finally:
        AS.ASSET_DIR, C.PHYSICS_DEVICE = SHARED_ASSETS["dir"], old_dev
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # --smoke-grasp: a comma list; collect: training grasps only; tests: anything outside the test split
    refused(lambda: RAA.build_job(collect_args(grasps, ok, tmp / "pool2", smoke_grasp="00015_g02")), "not training grasps")
    refused(lambda: RAA.build_job(collect_args(grasps, ok, tmp / "pool2", smoke_grasp="00015_g00,00015_g02")),
            "not training grasps")
    refused(lambda: RAA.build_job(collect_args(grasps, ok, tmp / "pool2", smoke_grasp="00015_g00,00015_g00")), "distinct")
    refused(lambda: RAA.build_job(collect_args(grasps, ok, tmp / "pool2", smoke_grasp="00015_g00,00999_g00")), "distinct")
    refused(lambda: RAA.build_job(job_args(grasps, ok, smoke_grasp="00444_g00", out=str(tmp / "o"))), "test split")
    keys = "00015_g00,00117_g02,00863_g00,00470_g01,00015_g01"
    _, sj = RAA.build_job(collect_args(grasps, ok, tmp / "pool2", smoke_grasp=keys))
    assert sorted(g["grasp"] for g in sj["grasps"]) == sorted(keys.split(",")) and sorted(sj["order"]) == sorted(keys.split(","))
    _, tj = RAA.build_job(job_args(grasps, ok, smoke_grasp="00015_g00,00015_g02", out=str(tmp / "o")))
    assert [g["grasp"] for g in tj["grasps"]] == ["00015_g00", "00015_g02"]
    # tries_before: deaths count as tries; the exit-8 result's deaths are not counted twice
    f = tmp / "tb.json"
    assert RAA.tries_before(f, 1) == (0, 0) and RAA.tries_before(f, 2) == (0, 1)
    f.write_text(json.dumps({"error": "x", "error_kind": "crash", "tries": 1}))
    assert RAA.tries_before(f, 1) == (1, 0) and RAA.tries_before(f, 2) == (1, 1)
    f.write_text(json.dumps({"error": "x", "error_kind": "crash", "tries": 2, "claim_kept": True, "claim_attempt": 2}))
    assert RAA.tries_before(f, 2) == (2, 0) and RAA.tries_before(f, 3) == (2, 1)
    return "assets, asset_files, transfer, switches, device refused; worker_checks refuses; code identity recorded"


def test_f5_pairing(tmp: Path) -> str:
    """Fix 5: act_done's options = ATT_EXTRA without --inside-mm <v> plus ACT_EXTRA (online_automate.sh
    act_done_options); prepare accepts them; paired() accepts the two test jobs and refuses another seed."""
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    fn = shell_function(AUTOMATE_SH, "act_done_options")
    r = bash(fn + 'att_extra=(${ATT_EXTRA:-}); act_extra=(${ACT_EXTRA:-}); mapfile -t extra < <(act_done_options); '
             'printf "[%s]" "${extra[@]}"', ATT_EXTRA="--inside-mm 3 --candidates 8 --margin=0.4 --inside-mm=2",
             ACT_EXTRA="--eps 0.2")
    assert r.returncode == 0 and r.stdout == "[--candidates][8][--margin=0.4][--eps][0.2]", (r.stdout, r.stderr)
    r = bash(fn + 'att_extra=(); act_extra=(); mapfile -t extra < <(act_done_options); echo "${#extra[@]}"')
    assert r.returncode == 0 and r.stdout.strip() == "0", (r.stdout, r.stderr)
    grasps = collection(tmp / "f5")
    ok = ckpt(tmp / "f5.pt", D.split_identity(grasps))
    _, wm = RAA.build_job(job_args(grasps, ok, mode="wm", starts=4, inside_mm=3.0, candidates=8, out=str(tmp / "w")))
    _, ad = RAA.build_job(job_args(grasps, ok, mode="act_done", starts=4, candidates=8, out=str(tmp / "a")))
    _, other = RAA.build_job(job_args(grasps, ok, mode="act_done", starts=4, seed=9_100_001, out=str(tmp / "b")))
    for name, j in (("w", wm), ("a", ad), ("b", other)):
        (tmp / f"{name}.json").write_text(json.dumps(j))
    paired = shell_function(AUTOMATE_SH, "paired")
    r = bash(paired + f'paired {tmp / "a.json"} {tmp / "w.json"}')
    assert r.returncode == 0, r.stderr
    r = bash(paired + f'paired {tmp / "b.json"} {tmp / "w.json"}')
    assert r.returncode == 1 and "not paired" in r.stderr, (r.returncode, r.stderr)
    return "act_done options derived from ATT_EXTRA; paired tests accepted, another seed refused"


def test_f6_start_success(tmp: Path) -> str:
    """Fix 6: check_start right after the placement (entry outcome start_success, StartSuccess carries it); run_grasp
    records the entry; finish exits EXIT_START_SUCCESS (no DONE); online_automate.sh writes <pool>/STOP and exits 1."""
    Robot, H_goal, axis = setup()
    from wm.online import automate320 as A
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    far = make_robot(Robot, H_goal)
    entry = {"idx": 0}
    A.check_start(far, entry)
    assert "outcome" not in entry
    near = make_robot(Robot, H_goal)
    near.H_goal = near.hand_pose().copy()
    near.H_goal[:3, 3] += 0.005 * axis
    entry = {"idx": 1}
    try:
        A.check_start(near, entry)
        raise AssertionError("a successful start was accepted")
    except A.StartSuccess as exc:
        assert exc.entry is entry and entry["outcome"] == "start_success" and 0 < entry["start_truth"]["height_mm"] < 10
    # run_grasp records the entry
    grasps = collection(tmp / "f6")
    ok = ckpt(tmp / "f6.pt", D.split_identity(grasps))
    _, job = RAA.build_job(collect_args(grasps, ok, tmp / "pool6"))
    robot = types.SimpleNamespace(max_decisions=7, port_spec=types.SimpleNamespace(
        disassembly_dist_m=0.01, depth_top_m=L_M, symmetry_n=4))

    def fake(robot, agent, fk, g, i, streams, mode, akw, beat=None):
        raise A.StartSuccess("placed start satisfies the rule", {"idx": i, "outcome": "start_success"})
    old = A.run_attempt
    try:
        A.run_attempt = fake
        RAA.run_grasp(robot, None, types.SimpleNamespace(step=77), job["grasps"][0], job, tmp / "pool6" / "r1",
                      lambda *a, **k: None, 1)
        raise AssertionError("StartSuccess swallowed")
    except A.StartSuccess:
        s = RAA._CUR["s"]
        assert s["starts"] == [{"idx": 0, "outcome": "start_success"}] and s["start_success_stop"] is True
    finally:
        A.run_attempt = old
    # finish: exit 4, no DONE (collect) / no summary.json (test)
    W = tmp / "pool6" / "r1"
    W.mkdir(parents=True, exist_ok=True)
    (W / "job.json").write_text(json.dumps(job))
    g0 = job["grasps"][0]
    f = Path(job["results"]) / g0["port"] / f"{g0['grasp']}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(s))
    assert RAA.finish(["--job", str(W)]) == RAA.EXIT_START_SUCCESS == 4
    assert not (W / "DONE").exists() and (Path(job["results"]) / "summary_incomplete.json").exists()
    _, tj = RAA.build_job(job_args(grasps, ok, out=str(tmp / "f6t")))
    (tmp / "f6t").mkdir()
    (tmp / "f6t" / "job.json").write_text(json.dumps(tj))
    tg = tj["grasps"][0]
    (tmp / "f6t" / tg["port"]).mkdir()
    (tmp / "f6t" / tg["port"] / f"{tg['grasp']}.json").write_text(json.dumps({**s, "grasp": tg["grasp"]}))
    assert RAA.finish(["--job", str(tmp / "f6t")]) == 4 and not (tmp / "f6t" / "summary.json").exists()
    # online_automate.sh: exit 4 -> <pool>/STOP, the rounds end with exit 1; any other code passes
    fn = shell_function(AUTOMATE_SH, "stop_on_start_success")
    P = tmp / "pool_sh"
    P.mkdir()
    r = bash(fn + f'P={P}; olog() {{ echo "$*"; }}; stop_on_start_success 0 x; stop_on_start_success 2 x; echo passed; '
             'stop_on_start_success 4 "round 1 collection"; echo not-reached')
    assert r.returncode == 1 and "passed" in r.stdout and "not-reached" not in r.stdout, (r.returncode, r.stdout)
    assert "start_success" in (P / "STOP").read_text()
    return "start checked after placement and recorded; finish exit 4; rounds write STOP and exit 1"


def test_f7_atomic(tmp: Path) -> str:
    """Fix 7: finish writes summary.json, the pool's manifest.json and DONE, and a --record manifest.json through
    _write_atomic (no partial file can be read by the learner)."""
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    grasps = collection(tmp / "f7")
    ok = ckpt(tmp / "f7.pt", D.split_identity(grasps))
    att = {"success": True, "stopped": "seated", "band_above_top_mm": -1.0, "start_success": False, "seated_fk": True,
           "interventions": 0, "retries": 0}

    def fill(job):
        for g in job["grasps"]:
            f = Path(job["results"]) / g["port"] / f"{g['grasp']}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"grasp": g["grasp"], "starts": [{"idx": i, "attempt": att} for i in range(job["starts"])],
                                     "entries": [{"path": f"{g['grasp']}/s{i}.npz"} for i in range(job["starts"])]}))
    written, orig = [], RAA._write_atomic

    def spy(path, text):
        written.append(Path(path))
        return orig(path, text)
    try:
        RAA._write_atomic = spy
        W, job = RAA.build_job(collect_args(grasps, ok, tmp / "pool7"))
        W.mkdir(parents=True)
        (W / "job.json").write_text(json.dumps(job))
        fill(job)
        assert RAA.finish(["--job", str(W)]) == 0
        want = {Path(job["results"]) / "summary.json", W / "manifest.json", W / "DONE"}
        assert want <= set(written), (want, written)
        pc = json.loads((Path(job["results"]) / "summary.json").read_text())["physics_code_sha256"]
        assert pc == {"online": job["code"]["physics_code_sha256"], "collection_m1b": "m" * 64,
                      "collection_m1b_field": "physics_code_sha256"}, pc
        assert json.loads((W / "DONE").read_text())["entries"] == len(job["grasps"]) * job["starts"]
        T, tj = RAA.build_job(job_args(grasps, ok, out=str(tmp / "f7t"), record=str(tmp / "rec")))
        T.mkdir(parents=True)
        (T / "job.json").write_text(json.dumps(tj))
        fill(tj)
        written.clear()
        assert RAA.finish(["--job", str(T)]) == 0
        assert {T / "summary.json", tmp / "rec" / "manifest.json"} <= set(written), written
    finally:
        RAA._write_atomic = orig
    left = [p for p in tmp.rglob("*.tmp*")]
    assert not left, left
    return "summary, manifest, DONE and the record manifest written atomically"


def test_f8_min_step(tmp: Path) -> str:
    """Fix 8: prepare refuses a model whose step is below --min-step (the round-1 guard)."""
    from wm.tdmpc import demo320 as D
    import run_wm320_automate as RAA
    grasps = collection(tmp / "f8")
    ok = ckpt(tmp / "f8.pt", D.split_identity(grasps))                # step 77
    refused(lambda: RAA.build_job(collect_args(grasps, ok, tmp / "pool8", min_step=78)), "--min-step 78")
    _, job = RAA.build_job(collect_args(grasps, ok, tmp / "pool8", min_step=77))
    assert job["min_step"] == 77 and job["model_step"] == 77
    return "step 77 refused under --min-step 78, accepted under 77"



SMOKE_SH = WM_ROOT / "scripts" / "automate" / "smoke" / "online_smoke.sh"


def test_r2_code_and_smoke(tmp: Path) -> str:
    """Re-audit: online_automate.sh round_files_differ (the learner's code is used only when every round file equals
    the live tree's); online_smoke.sh refuses a grasp list that is not 4-6 grasps of >= 3 assemblies, runs act_done
    first, and its run_job ends the smoke (exit 1) on a job exit 4; online_smoke_check.py lists the collect job's grasps
    in the order the worker ran them with pid, scene_build_s and memory."""
    txt = AUTOMATE_SH.read_text()
    files = re.search(r'^ROUND_FILES="(.*?)"', txt, re.S | re.M).group(1).split()
    assert {"scripts/online/run_wm320_automate.py", "scripts/online/run_wm320_attempts.py",
            "scripts/eval/eval_policy320.py", "scripts/automate/online_job.sh",
            "scripts/automate/online_worker.sh"} <= set(files)
    a, b = tmp / "code_a", tmp / "code_b"
    for d in (a, b):
        for f in files:
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text(f"same {f}\n")
    fn = shell_function(AUTOMATE_SH, "round_files_differ")
    rf = 'ROUND_FILES="' + " ".join(files) + '"\n'
    r = bash(rf + fn + f'round_files_differ {a} {b}')
    assert r.returncode == 0 and r.stdout == "", (r.stdout, r.stderr)
    (b / "scripts/eval/eval_policy320.py").write_text("changed\n")
    (a / "scripts/automate/online_worker.sh").unlink()
    r = bash(rf + fn + f'round_files_differ {a} {b}')
    assert r.stdout.split() == ["scripts/eval/eval_policy320.py", "scripts/automate/online_worker.sh(missing)"], r.stdout
    # online_smoke.sh: the grasp list check happens before anything is sourced or frozen
    for keys in ("00015_g00,00117_g00,00863_g00", "00015_g00,00015_g01,00117_g00,00117_g01",
                 "a_g00,b_g00,c_g00,d_g00,e_g00,f_g00,g_g00"):
        r = subprocess.run(["bash", str(SMOKE_SH), "ck", "gr", str(tmp / "smoke_out"), keys, "16G", "500%", "12", "1500"],
                           capture_output=True, text=True)
        assert r.returncode == 1 and "4-6 grasps from at least 3 assemblies" in r.stdout, (keys, r.stdout, r.stderr)
    assert not (tmp / "smoke_out").exists()
    smoke = SMOKE_SH.read_text()
    assert "--mode act_done" in smoke and "--mode act " not in smoke and "16G 500% 12" in smoke
    run_job = shell_function(SMOKE_SH, "run_job")
    code = tmp / "fake_code"
    (code / "scripts" / "automate").mkdir(parents=True)
    out = tmp / "smoke_rj"
    out.mkdir()
    for rc, want in ((0, 0), (2, 0), (4, 1)):
        (code / "scripts" / "automate" / "online_job.sh").write_text(f"exit {rc}\n")
        r = bash(run_job + f'CODE={code}; SW=x; MEM=1G; CPU=100%; GPU=1; L={out}/smoke.log; OUT={out}; true & spid=$!; '
                 f'run_job {out}/j --mode act_done; echo continued')
        assert r.returncode == want and (("continued" in r.stdout) == (want == 0)), (rc, r.returncode, r.stdout, r.stderr)
    assert "start_success stop" in (out / "smoke.log").read_text()
    # online_smoke_check.py: the collect grasps in run order, one process
    from wm.tdmpc import demo320 as D  # noqa: F401
    grasps = collection(tmp / "r2c")
    so = tmp / "smoke_chk"
    res = so / "pool" / "r1" / "results"
    (so / "pool" / "r1").mkdir(parents=True)
    (so / "pool" / "r1" / "job.json").write_text(json.dumps({"results": str(res)}))
    for k, t, b_s in (("00863_g00", 30.0, 41.0), ("00015_g00", 10.0, 40.0), ("00015_g01", 20.0, None)):
        f = res / k[:5] / f"{k}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"grasp": k, "port": k[:5], "started_unix": t, "process": {"pid": 7}, "scene_build_s": b_s,
                                 "seconds": 5.0, "memory": {"maxrss_gb": t / 10, "memory_peak_gb": t / 5}, "starts": []}))
    r = subprocess.run([sys.executable, str(WM_ROOT / "scripts" / "automate" / "smoke" / "online_smoke_check.py"),
                        str(so), str(grasps)], capture_output=True, text=True)
    rep = json.loads((so / "t7.json").read_text())["jobs"]["collect"]
    assert [x["grasp"] for x in rep["rebuilds"]] == ["00015_g00", "00015_g01", "00863_g00"], (rep, r.stderr)
    assert rep["one_worker_process"] is True and [x["scene_build_s"] for x in rep["rebuilds"]] == [40.0, None, 41.0]
    assert [x["memory_peak_gb"] for x in rep["rebuilds"]] == [2.0, 4.0, 6.0]
    return "round files compared; smoke grasp list, act_done, exit-4 stop; smoke check lists rebuilds in run order"

if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="online_port_am_"))
    use_shared_assets(tmp)
    try:
        print("T3 OK:", test_t3(tmp), flush=True)
        print("T4 OK:", test_t4(tmp), flush=True)
        print("T5 OK:", test_t5(tmp), flush=True)
        print("T6 OK:", test_t6(tmp), flush=True)
        print("loop OK:", test_loop(tmp), flush=True)
        print("F1 OK:", test_f1_claim_order(tmp), flush=True)
        print("F2/F3 OK:", test_f2_f3_inputs(tmp), flush=True)
        print("F5 OK:", test_f5_pairing(tmp), flush=True)
        print("F6 OK:", test_f6_start_success(tmp), flush=True)
        print("F7 OK:", test_f7_atomic(tmp), flush=True)
        print("F8 OK:", test_f8_min_step(tmp), flush=True)
        print("R2 OK:", test_r2_code_and_smoke(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
