"""Port plan T1 (CPU, no simulator): the online attempt on ManiSkill is unchanged by the AutoMate port.

The code of git c7b37d7 (the branch's base: attempt320.py and run_wm320_attempts.py before the port) and this tree's
(wm.online.maniskill320.ManiSkillTester = attempt320.OnlineTester + shared320 + the ManiSkill adapter) run on the same
deterministic stand-in robot and agent (tests/_online_port_stubs.py) with the same seeds, for every mode and the
branches of the loop (act, act_done, wm with the in-hole rule, the done stop, the back-off and re-position, collect's
exploration, servo true / motion, random, the counterfactual probe with its save / restore). Both use the REAL
wm.collect.branch / wm.collect.explore (placement, the old recorder) and wm.data / wm.tdmpc code; only the simulator
modules they import are replaced by stand-ins (wm.env.robot, wm.env.expert, wm.data.recorder, wm.sim.env_adapter:
ManiSkill, SAPIEN and MPlib are not needed). Checked equal:
  - the summary of every attempt, key order included (json.dumps), except wall-clock numbers (latency_ms values, the
    probe's seconds), whose decision indices are compared
  - every array of the recording (rgb, depth, qpos, qvel, hand, eval_rel, action, action_policy, img_step,
    cam_T_world, K), the stored JPEG bytes and dec_kind
  - the fast-copy file of run_wm320_attempts.write_fast (every member of the npz, byte for byte) and its manifest entry
  - the constants the loop reads (EXECUTE, MAX_DECISIONS, STALL_*, IMAGE_EVERY) equal wm.collect.branch's
  - shared320.frame / offset_pose equal explore.frame / offset_pose on random axes, bit for bit
c7b37d7's files come from `git show` of this repository, or from WM_OLD_SRC = that commit's world_model folder (e.g.
`git archive c7b37d7 world_model | tar -x -C <dir>`, then WM_OLD_SRC=<dir>/world_model).

    python world_model/tests/test_online_port_maniskill_cpu.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WM_ROOT = HERE.parent
sys.path.insert(0, str(WM_ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WM_ROOT / "scripts" / "online"))
import _online_port_stubs as S  # noqa: E402

OLD = "c7b37d7"
H_GOAL = None
AXIS = None


# ---------------------------------------------------------------------------------------------- simulator stand-ins
@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    seg: np.ndarray
    K: np.ndarray
    cam_T_world: np.ndarray
    qpos: np.ndarray
    hand: np.ndarray


class _Cam:
    def __init__(self, r):
        self.r = r

    def capture(self):
        self.r._pic = self.r.arm.picture()

    def get_picture(self, name):
        rgb = self.r._pic[0]
        return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1)[None]


class _Comp:
    def __init__(self, r):
        self.r = r

    def get_intrinsic_matrix(self):
        return np.array([[30.0, 0.0, 12.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]])

    def get_extrinsic_matrix(self):
        return np.linalg.inv(self.r.hand_pose())[:3, :]


class _Twin:
    def __init__(self, r):
        self.r = r

    def take_picture(self):
        pass

    def get_picture(self, name):
        d = self.r._pic[1]
        return np.stack([d, d, -d, np.ones_like(d)], -1).astype(np.float32)


class StubManiSkillRobot:
    """The parts of wm.env.robot.Robot the attempt, Brancher and HandFK read."""

    def __init__(self):
        self.arm = S.ToyArm()
        self.port_spec = types.SimpleNamespace(pin_length_m=self.arm.L)
        r = self
        self.base = types.SimpleNamespace(
            _planner=lambda: S.PlannerShim(),
            agent=types.SimpleNamespace(robot=types.SimpleNamespace(pose=S.BasePose(S.BASE_T),
                                                                    get_qvel=lambda: r.arm.v[None].copy())),
            scene=types.SimpleNamespace(update_render=lambda: None),
            get_state_dict=lambda: r.arm.get_state(), set_state_dict=lambda s: r.arm.set_state(s))
        self._cam, self._comp, self._twin = _Cam(self), _Comp(self), _Twin(self)
        self._pic = None

    def qpos(self):
        return self.arm.qpos()

    def hand_pose(self):
        return self.arm.hand_pose()

    def _command(self, q7, n=1):
        self.arm._command(q7, n)

    def _camera(self):
        return self._cam, self._comp


def install_stand_ins():
    """wm.env.robot, wm.env.expert, wm.data.recorder, wm.sim.env_adapter -> stand-ins (before wm.collect is imported)."""
    import wm.data
    import wm.env
    import wm.sim
    envrobot = types.ModuleType("wm.env.robot")
    envrobot.Frame, envrobot.Robot = Frame, StubManiSkillRobot
    envrobot.CAM_SCALE, envrobot.CROP = 1, (4, 20, 4, 20)
    envrobot.DEPTH_MIN_M, envrobot.DEPTH_UNIT_M = 0.07, 1e-4
    expert = types.ModuleType("wm.env.expert")
    expert.Expert = lambda robot: types.SimpleNamespace(robot=robot)
    rec = types.ModuleType("wm.data.recorder")
    rec._truth_rel = lambda robot: S.truth7(robot.arm, H_GOAL, AXIS)
    rec.Recorder = object
    ea = types.ModuleType("wm.sim.env_adapter")
    ea._np = lambda x: np.asarray(x)
    for name, mod, pkg in (("wm.env.robot", envrobot, wm.env), ("wm.env.expert", expert, wm.env),
                           ("wm.data.recorder", rec, wm.data), ("wm.sim.env_adapter", ea, wm.sim)):
        sys.modules[name] = mod
        setattr(pkg, name.rsplit(".", 1)[1], mod)


def old_source(rel: str) -> str:
    base = os.environ.get("WM_OLD_SRC")
    if base:
        return (Path(base) / rel).read_text()
    r = subprocess.run(["git", "-C", str(WM_ROOT), "show", f"{OLD}:world_model/{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"needs git {OLD}'s world_model/{rel} (git show failed: {r.stderr.strip()[:200]}); set "
                           "WM_OLD_SRC to that commit's world_model folder (not skipped)")
    return r.stdout


def load_old(rel: str, name: str, package: str | None, file: str):
    mod = types.ModuleType(name)
    mod.__file__, mod.__package__ = file, package
    sys.modules[name] = mod
    exec(compile(old_source(rel), f"{OLD}:{rel}", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------------------------- the comparison
CASES = [  # (name, agent kind, mode, kwargs)
    ("act", "stall", "act", {}),
    ("act_done_seat", "seat", "act_done", {}),
    ("wm_inside", "stall", "wm", {"inside_mm": 1.0, "margin": 0.001, "max_retries": 2}),
    ("wm_seat", "seat", "wm", {"inside_mm": 1.0, "margin": 0.001}),
    ("wm_backoff", "stall", "wm", {"margin": 0.001, "max_retries": 3, "n_candidates": 6}),    # ends "left"
    ("collect", "stall", "collect", {"margin": 0.001, "max_retries": 2, "p_shift": 1.0, "eps": 0.5, "n_candidates": 5,
                                     "random_reposition": 0.5}),
    ("servo_true", "stall", "servo", {"inside_mm": 1.0, "max_retries": 0, "servo_axis": "true"}),
    ("servo_motion", "stall", "servo", {"inside_mm": 1.0, "max_retries": 0, "servo_axis": "motion"}),
    ("random", "stall", "random", {"inside_mm": 1.0, "max_retries": 0, "random_p": [0.3, 0.6, 0.9, 0.5]}),
    ("probe", "stall", "wm", {"inside_mm": 1.0, "margin": 0.001, "max_retries": 0, "probe_max": 2, "probe_gap_mm": 8.0}),
]


def strip_clock(s: dict) -> dict:
    s = json.loads(json.dumps(s, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    s["latency_ms"] = [k for k, _ in s["latency_ms"]]
    for p in s.get("probes", []):
        p.pop("seconds", None)
    return s


def run(tester_cls, kind, mode, kw, max_decisions=40):
    robot = StubManiSkillRobot()
    tester = tester_cls(robot, H_GOAL, AXIS)
    kw = dict(kw)
    return tester.attempt(S.StubAgent(kind), mode, 1234, max_decisions=max_decisions, **kw)


def test_t1() -> str:
    global H_GOAL, AXIS
    install_stand_ins()
    arm = S.ToyArm()
    AXIS = np.array([0.97, 0.12, -0.2])
    AXIS /= np.linalg.norm(AXIS)
    H_GOAL = arm.hand_pose().copy()
    H_GOAL[:3, 3] += (arm.L + 0.003) * AXIS                             # the pin tips start 3 mm in front of the face
    from wm.collect import branch as BR
    from wm.collect import explore as EX
    from wm.online import shared320 as SH
    from wm.online.maniskill320 import ManiSkillTester
    old_at = load_old("src/wm/online/attempt320.py", "wm.online._old_attempt320", "wm.online",
                      str(WM_ROOT / "src" / "wm" / "online" / "attempt320.py"))
    old_ra = load_old("scripts/online/run_wm320_attempts.py", "_old_run_wm320_attempts", None,
                      str(WM_ROOT / "scripts" / "online" / "run_wm320_attempts.py"))
    import run_wm320_attempts as RA
    for k in ("EXECUTE", "MAX_DECISIONS", "STALL_DECISIONS", "STALL_M", "IMAGE_EVERY"):
        assert getattr(SH, k) == getattr(BR, k), k
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.normal(size=3)
        a[2] *= 0.5
        assert np.array_equal(SH.frame(a), EX.frame(a))
        H = np.eye(4)
        H[:3, 3] = rng.normal(size=3)
        F = EX.frame(a)
        args = (rng.normal() * 0.01, rng.normal(size=2) * 0.01, rng.normal(size=3) * 3)
        assert np.array_equal(SH.offset_pose(H, F, *args), EX.offset_pose(H, F, *args))
    tmp = Path(tempfile.mkdtemp(prefix="t1_"))
    report, seen = [], {}
    for name, kind, mode, kw in CASES:
        s_old, o_old = run(old_at.OnlineTester, kind, mode, kw)
        s_new, o_new = run(ManiSkillTester, kind, mode, kw)
        a, b = strip_clock(s_old), strip_clock(s_new)
        assert json.dumps(a) == json.dumps(b), (name, [k for k in a if a.get(k) != b.get(k)], list(a) == list(b))
        assert set(o_old) == set(o_new), name
        for k in o_old:
            if k == "rgb_jpeg_list":
                assert o_old[k] == o_new[k], name
            else:
                assert np.array_equal(o_old[k], o_new[k]) and np.asarray(o_old[k]).dtype == np.asarray(o_new[k]).dtype, (name, k)
        meta = {"port": "G01", "grasp": "G01_g00", "condition": "train", "pitch_deg": -30.0, "grasp_pos_m": -0.015,
                "episode_seed": 7, "kind": "online", "outcome": "seated_fk" if s_new["seated_fk"] else "not_seated",
                "source": f"r1/traj/G01/G01_g00/s0.npz", "pin_length_m": 0.016, "H_star": H_GOAL.tolist(),
                "axis": AXIS.tolist(), "round": 1, "start": 0}
        e_old = old_ra.write_fast(tmp / name / "old.npz", o_old, meta)
        e_new = RA.write_fast(tmp / name / "new.npz", o_new, meta)
        assert e_old == e_new, name
        with zipfile.ZipFile(tmp / name / "old.npz") as zo, zipfile.ZipFile(tmp / name / "new.npz") as zn:
            assert zo.namelist() == zn.namelist(), name
            for m in zo.namelist():
                assert zo.read(m) == zn.read(m), (name, m)
        seen[name] = s_new
        report.append(f"{name}: {s_new['stopped']} after {s_new['decisions']} decisions, kinds {s_new['decision_kinds']}, "
                      f"interventions {s_new['interventions']}, probes {len(s_new['probes'])}")
    # the cases reach the branches they are named for (otherwise the comparison would not cover them)
    assert seen["act_done_seat"]["stopped"] == "seated" and seen["wm_seat"]["stopped"] == "seated"
    assert seen["wm_inside"]["inside_at"] is not None
    assert seen["wm_inside"]["retries"] > 0 and seen["wm_inside"]["decision_kinds"]["reposition"] > 0
    assert seen["wm_backoff"]["stopped"] == "left"
    assert seen["collect"]["shift"] is not None and seen["collect"]["random_candidates"] > 0
    assert seen["servo_true"]["servo_moves"] > 0
    assert seen["random"]["random_candidates"] > 0
    assert len(seen["probe"]["probes"]) > 0
    assert any(s["interventions"] > 0 for s in seen.values()), "no model correction in any case"
    return "; ".join(report)


if __name__ == "__main__":
    print("T1 OK:", test_t1(), flush=True)
