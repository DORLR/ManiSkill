"""The offset mode "gap" on AutoMate (the user's decision 6 of 2026-11-23, written with wmdata320 OFFSET_MODES since
2026-11-25) on a CPU, no simulator, no GPU:
  1 labels: for AutoMate trajectories (meta simulator "automate") with offset_mode "gap" the world model's offset
    target (wmdata320 "probe" of trajectory() and window()) has as first component the pin-tip gap g = s - L [mm],
    with s = -(hand - H*) . axis computed here independently in float64 (g > 0 in front of the entrance, g < 0 past
    it, g = -L seated), equal to "gap_mm"; its other three components equal those of offset_mode "along"; the
    "along" probe equals decision_labels' default; an unknown offset_mode is refused
  2 no pin length at test time: attempt320._read_agent_offset_and_gap in mode "gap" calls agent.gap_mm(z) without a
    pin length and refuses one handed to it (a poisoned number that raises on any use) untouched; in mode "along" it
    passes L; the WMAgent of an AutoMate checkpoint (offset_mode "gap") gives gap_mm(z) = offset_mm(z)[0] and refuses
    a poisoned pin length untouched; check_offset_mode refuses an AutoMate attempt with an "along" checkpoint and takes
    either mode on ManiSkill (the old refusal of a gap checkpoint on ManiSkill is gone: offset_mode "gap" is the
    ManiSkill default since d1-gap-fix); WMAgent refuses an AutoMate checkpoint (provenance split) without
    offset_mode "gap" (also one with the pre-merge key offset_label "gap_mm" only); simulator_of_robot reads the class
    only (a robot whose every attribute raises); in attempt320.OnlineTester.attempt self.L is read only in meta_l
    (labels for reporting), in the "along" branch of the pin length given to _read_agent_offset_and_gap and in the
    final labels; face_gap_mm only under mode "collect"; port_spec never; check_offset_mode is called
  3 ManiSkill unchanged: trajectory() / window() of a ManiSkill trajectory with offset_mode "along" equal git
    6b37757's for every key of that commit bit for bit (new keys: gap_mm, offset_mode); the gap of an "along"
    checkpoint equals 6b37757's formula bit for bit; a checkpoint without offset_mode gives a WMAgent with "along"
    whose gap_mm(z, L) equals 6b37757's WMAgent.gap_mm(z, L); WM320.loss of the same window (offset head on) equals
    6b37757's, same seed
Needs TD-MPC2's code (wm.tdmpc.use_tdmpc2) and git 6b37757's world_model/src/wm: `git archive` of this repository, or
WM_LEGACY_SRC = a folder holding that commit's world_model/src/wm (as tests/test_encoders_automate.py).

    python world_model/tests/test_offset_gap_automate.py
"""

import ast
import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from wm.online import common320 as C  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.wm320 import WM320  # noqa: E402

ATTEMPT320 = HERE.parent / "src" / "wm" / "online" / "attempt320.py"     # not imported: ManiSkill imports

LEGACY_COMMIT = "6b37757"
STATS = {"s0": np.full(WD.ACTION_DIM, 0.05, np.float32), "top": np.full(WD.ACTION_DIM, 0.5, np.float32),
         "a_mean": np.zeros(16 * 7, np.float32), "a_std": np.full(16 * 7, 0.1, np.float32)}
L_M = 0.0213                                          # a pin length that is not round


class Poison(float):
    """A pin length that raises on any use (arithmetic, comparison, conversion)."""

    def _no(self, *a, **k):
        raise AssertionError("the pin length was read")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _no
    __neg__ = __float__ = __lt__ = __gt__ = __le__ = __ge__ = __eq__ = __bool__ = _no
    __hash__ = float.__hash__


class GapAgent:
    """A stand-in agent for attempt320._read_agent_offset_and_gap: records the pin length given to gap_mm."""

    def __init__(self, offset_mode: str, o: np.ndarray):
        self.offset_mode, self.o, self.got = offset_mode, o, []

    def offset_mm(self, z):
        return self.o

    def gap_mm(self, z, pin_length_mm=None):
        self.got.append(pin_length_mm)
        return float(self.o[0]) if pin_length_mm is None else float(-self.o[0] - pin_length_mm)


def read_agent_offset_and_gap():
    """attempt320._read_agent_offset_and_gap, executed alone from the source (attempt320 imports ManiSkill)."""
    tree = ast.parse(ATTEMPT320.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_read_agent_offset_and_gap")
    ns = {"np": np, "__name__": "attempt320_extract"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(ATTEMPT320), "exec"), ns)
    return ns["_read_agent_offset_and_gap"]


def raises(fn, exc, text: str) -> str:
    try:
        fn()
    except exc as e:
        assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError(f"expected {exc.__name__} with {text!r}")


def legacy_package(tmp: Path):
    """git 6b37757's world_model/src/wm, importable as the package wm_legacy."""
    dst = tmp / "legacy"
    if not (dst / "wm_legacy").exists():
        src = os.environ.get("WM_LEGACY_SRC")
        if src:
            shutil.copytree(Path(src) / "world_model" / "src" / "wm", dst / "wm_legacy")
        else:
            arc = tmp / "legacy.tar"
            with open(arc, "wb") as f:
                r = subprocess.run(["git", "-C", str(HERE.parents[1]), "archive", LEGACY_COMMIT, "world_model/src/wm"],
                                   stdout=f, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"needs git {LEGACY_COMMIT}'s world_model/src/wm: `git archive` failed here "
                                   f"({r.stderr.strip()[:200]}); set WM_LEGACY_SRC (not skipped)")
            with tarfile.open(arc) as t:
                t.extractall(dst / "x", filter="data")
            shutil.move(str(dst / "x" / "world_model" / "src" / "wm"), str(dst / "wm_legacy"))
    if str(dst) not in sys.path:
        sys.path.insert(0, str(dst))
    return (importlib.import_module("wm_legacy.tdmpc.wmdata320"), importlib.import_module("wm_legacy.tdmpc.wm320"),
            importlib.import_module("wm_legacy.online.agent320"))


def synthetic(seed: int, n: int = 6):
    """One fast-copy trajectory of n decisions: the hand comes in along the axis from 12 mm in front of the entrance
    to seated, with a lateral offset and a rotation. -> (z, H*, axis)."""
    g = np.random.default_rng(seed)
    T = 8 * n
    axis = g.normal(size=3)
    axis /= np.linalg.norm(axis)
    Hs = np.eye(4)
    Hs[:3, :3] = Rotation.from_rotvec(g.normal(size=3) * 0.3).as_matrix()
    Hs[:3, 3] = g.normal(size=3) * 0.1
    s = np.linspace(L_M + 0.012, 0.0, T + 1)                         # distance back from seated along the axis [m]
    B = WD._basis(axis)
    hand = np.tile(np.eye(4), (T + 1, 1, 1))
    hand[:, :3, 3] = Hs[:3, 3] - s[:, None] * axis + (g.normal(size=(T + 1, 2)) * 0.0015) @ B[1:]
    hand[:, :3, :3] = Rotation.from_rotvec(g.normal(size=(T + 1, 3)) * 0.02).as_matrix() @ Hs[:3, :3]
    z = {"action": g.normal(size=(T, 7)) * 0.01, "img_step": np.arange(0, T + 1, 4), "qpos": g.normal(size=(T + 1, 9)),
         "hand": hand}
    return z, Hs, axis


def meta_of(Hs, axis, automate: bool) -> dict:
    m = {"H_star": Hs.tolist(), "axis": axis.tolist(), "pin_length_m": L_M, "kind": "contact", "outcome": "inserted",
         "grasp": "x"}
    return {**m, "simulator": "automate", "symmetry_n": 4} if automate else m


# ------------------------------------------------------------------------------------------------------------- 1
def test_labels():
    z, Hs, axis = synthetic(0)
    meta = meta_of(Hs, axis, True)
    tr = WD.trajectory(z, meta, 0.3, 0.967, offset_mode="gap")
    tr_a = WD.trajectory(z, meta, 0.3, 0.967, offset_mode="along")
    assert tr["offset_mode"] == "gap" and tr_a["offset_mode"] == "along"
    steps = 8 * np.arange(tr["n"] + 1)
    s = -((z["hand"][steps, :3, 3] - Hs[:3, 3]) @ axis)               # [m], float64, independent of wmdata320
    g = (s - L_M) * 1000.0
    assert np.allclose(tr["probe"][:, 0], g, atol=1e-4, rtol=0), np.abs(tr["probe"][:, 0] - g).max()
    assert np.allclose(tr["gap_mm"], g, atol=1e-4, rtol=0) and np.array_equal(tr["probe"][:, 0], tr["gap_mm"])
    assert g[0] > 11.9 and abs(g[-1] + L_M * 1000.0) < 1e-6, "in front at the start (g > 0), g = -L when seated"
    assert np.array_equal(tr["probe"][:, 1:], tr_a["probe"][:, 1:]), "lateral x, y and tilt as the along probe"
    ref = WD.decision_labels(meta, z["hand"], steps)["probe"]
    assert np.array_equal(tr_a["probe"], ref), "the along probe = decision_labels' default"
    raises(lambda: WD.trajectory(z, meta, 0.3, 0.967, offset_mode="gap_mm"), ValueError, "unknown offset_mode")
    for k in range(tr["n"]):
        w = WD.window(tr, k, STATS["s0"], STATS["top"])
        dec = np.minimum(k + np.arange(WD.H + 1), tr["n"])
        assert np.array_equal(w["probe"], tr["probe"][dec]) and np.array_equal(w["gap_mm"], tr["gap_mm"][dec])
    return f"AutoMate gap probe[0] = s - L within 1e-4 mm over {len(g)} decisions (g from {g[0]:.2f} to {g[-1]:.2f} mm); other components = along probe"


# ------------------------------------------------------------------------------------------------------------- 2
def ckpt(m: WM320, **extra) -> dict:
    return {"format": "wm320-1", "labels": WD.LABELS, "stem": "standard", "imagenet": False, "offset_weight": 1.0,
            "shaping": 0.3, "cfg": {"gamma": 0.967}, "stats": STATS, "state": m.state_dict(), "step": 1, **extra}


def observation(seed: int):
    g = np.random.default_rng(seed)
    return (g.integers(0, 256, (4, 320, 320, 3), dtype=np.uint8), g.integers(500, 20000, (4, 320, 320)).astype(np.uint16),
            g.normal(size=7).astype(np.float32))


class RaisingRobot:
    """A robot whose every attribute raises (port_spec, pin length, poses)."""

    def __getattribute__(self, name):
        raise AssertionError(f"robot.{name} was read")


def test_no_pin_length(tmp: Path):
    o = np.array([-3.25, 0.4, -0.2, 1.5])
    read = read_agent_offset_and_gap()
    ag_g = GapAgent("gap", o)
    o_r, g_r = read(ag_g, "z", pin_length_m=None)
    assert o_r is o and g_r == -3.25 and ag_g.got == [None]
    raises(lambda: read(ag_g, "z", pin_length_m=Poison(0.0213)), RuntimeError, "zero-privilege")     # refused before any use
    assert ag_g.got == [None]
    ag_a = GapAgent("along", o)
    assert read(ag_a, "z", pin_length_m=0.0213)[1] == float(-o[0] - 0.0213 * 1000.0) and ag_a.got == [0.0213 * 1000.0]
    raises(lambda: read(ag_a, "z", pin_length_m=None), ValueError, "requires pin_length_m")
    raises(lambda: read(GapAgent("gap_mm", o), "z", pin_length_m=None), ValueError, "unknown offset_mode")
    try:                                                                          # the poison works
        float(Poison(1.0))
        raise RuntimeError("Poison did not raise")
    except AssertionError:
        pass
    C.check_offset_mode("gap", "automate")
    C.check_offset_mode("along", "maniskill")
    C.check_offset_mode("gap", "maniskill")
    raises(lambda: C.check_offset_mode("along", "automate"), ValueError, "train a new one")
    am = type("Robot", (), {"__module__": "wm.sim_automate.robot", "__getattribute__": RaisingRobot.__getattribute__})()
    assert C.simulator_of_robot(am) == "automate" and C.simulator_of_robot(RaisingRobot()) == "maniskill"
    torch.manual_seed(0)
    split = {"name": "split_automate_v1", "sha256": "a" * 64}
    p = tmp / "am.pt"
    m = WM320("standard", STATS, 0.3, offset_weight=1.0, offset_mode="gap", gamma=0.967).eval()
    torch.save(ckpt(m, offset_mode="gap", provenance={"split": split}), p)
    ag = WMAgent(str(p), "cpu")
    assert ag.offset_mode == "gap"
    z = ag.encode(*observation(1))
    o = ag.offset_mm(z)
    assert ag.gap_mm(z) == float(o[0])
    raises(lambda: ag.gap_mm(z, Poison(21.3)), ValueError, "do not pass pin_length_mm")
    torch.save(ckpt(m, provenance={"split": split}), tmp / "am_old.pt")            # AutoMate, before decision 6
    raises(lambda: WMAgent(str(tmp / "am_old.pt"), "cpu"), ValueError, "train a new one")
    torch.save(ckpt(m, offset_label="gap_mm", provenance={"split": split}), tmp / "am_label.pt")   # pre-merge key only
    raises(lambda: WMAgent(str(tmp / "am_label.pt"), "cpu"), ValueError, "train a new one")
    torch.save(ckpt(m, offset_mode="along", provenance={"split": split}), tmp / "am_along.pt")
    raises(lambda: WMAgent(str(tmp / "am_along.pt"), "cpu"), ValueError, "train a new one")
    torch.save(ckpt(m, offset_mode="probe"), tmp / "bad.pt")
    raises(lambda: WMAgent(str(tmp / "bad.pt"), "cpu"), ValueError, "unknown offset_mode")
    # attempt320: where self.L, face_gap_mm and port_spec appear in OnlineTester.attempt (source, no simulator)
    tree = ast.parse(ATTEMPT320.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OnlineTester")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "attempt")
    parent = {c: n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}

    def up(n):
        while n in parent:
            n = parent[n]
            yield n

    where = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr == "L":
            ctx = None
            for a in up(n):
                if isinstance(a, ast.IfExp) and a.orelse is n and ast.unparse(a.test) == "offset_mode == 'gap'" \
                        and isinstance(parent[a], ast.keyword) and parent[a].arg == "pin_length_m" \
                        and getattr(parent[parent[a]].func, "id", "") == "_read_agent_offset_and_gap":
                    ctx = "_read_agent_offset_and_gap's pin length (offset_mode along only)"
                    break
                if isinstance(a, ast.Assign) and [t.id for t in a.targets if isinstance(t, ast.Name)] == ["meta_l"]:
                    ctx = "meta_l"
                    break
                if isinstance(a, ast.Assign) and [t.id for t in a.targets if isinstance(t, ast.Name)] == ["lab"] \
                        and not any(isinstance(b, ast.While) for b in up(a)):
                    ctx = "final labels (after the loop)"
                    break
            assert ctx is not None, f"self.L read at line {n.lineno} outside the allowed places"
            where.append((n.lineno, ctx))
        if isinstance(n, ast.Attribute) and n.attr == "port_spec":
            raise AssertionError(f"port_spec read at line {n.lineno}")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "face_gap_mm":
            assert any(isinstance(a, ast.If) and "collect" in ast.unparse(a.test) for a in up(n)), n.lineno
    assert sorted(c for _, c in where) == ["_read_agent_offset_and_gap's pin length (offset_mode along only)",
                                           "final labels (after the loop)", "meta_l"], where
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_read_agent_offset_and_gap"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == "agent"
    chk = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "check_offset_mode"]
    assert len(chk) == 1 and ast.unparse(chk[0]) == "check_offset_mode(offset_mode, simulator_of_robot(self.robot))"
    return (f"offset_mode gap: g = o[0], no pin length passed, a poisoned pin length refused untouched; an along "
            f"checkpoint refused on AutoMate; old AutoMate checkpoints refused; attempt reads self.L only at {where}")


# ------------------------------------------------------------------------------------------------------------- 3
def windows_W(tr: dict, k: int, rgb_seed: int) -> dict:
    w = WD.window(tr, k, STATS["s0"], STATS["top"])
    w.pop("slots")
    g = torch.Generator().manual_seed(rgb_seed)
    S = WD.H + 4
    W = {key: torch.from_numpy(np.asarray(v))[None] for key, v in w.items()}
    W["rgb"] = torch.randint(0, 256, (1, S, 320, 320, 3), dtype=torch.uint8, generator=g)
    W["depth"] = torch.randint(500, 20000, (1, S, 320, 320), dtype=torch.int16, generator=g)
    return W


def test_maniskill_unchanged(tmp: Path):
    OD, OW, OA = legacy_package(tmp)
    z, Hs, axis = synthetic(1)
    meta = meta_of(Hs, axis, False)
    tr, tr0 = WD.trajectory(z, meta, 0.3, 0.967, offset_mode="along"), OD.trajectory(z, meta, 0.3, 0.967)
    for k, v in tr0.items():
        assert type(v) is type(tr[k]) and (np.array_equal(v, tr[k]) and np.asarray(v).dtype == np.asarray(tr[k]).dtype
                                          if isinstance(v, np.ndarray) else v == tr[k]), k
    assert set(tr) - set(tr0) == {"gap_mm", "offset_mode"} and tr["offset_mode"] == "along"
    for k in range(tr["n"]):
        w, w0 = WD.window(tr, k, STATS["s0"], STATS["top"]), OD.window(tr0, k, STATS["s0"], STATS["top"])
        assert set(w) - set(w0) == {"gap_mm"}
        assert all(np.array_equal(w[key], w0[key]) and w[key].dtype == w0[key].dtype for key in w0), k
    rng = np.random.default_rng(5)
    for _ in range(1000):                              # the gap of an along checkpoint = 6b37757's formula, bit for bit
        o, L = rng.normal(size=4) * 30, float(rng.uniform(5, 40))
        stub = type("Stub", (), {"offset_mode": "along", "offset_mm": lambda self, z, o=o: o})()
        assert WMAgent.gap_mm(stub, None, L) == float(-o[0] - L)
    torch.manual_seed(2)
    m = WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).eval()
    p = tmp / "ms.pt"
    torch.save(ckpt(m), p)                                                        # a ManiSkill checkpoint: no key
    a_new, a_old = WMAgent(str(p), "cpu"), OA.WMAgent(str(p), "cpu")
    assert a_new.offset_mode == "along"
    obs = observation(3)
    zn, zo = a_new.encode(*obs), a_old.encode(*obs)
    assert torch.equal(zn, zo)
    assert a_new.gap_mm(zn, 21.3) == a_old.gap_mm(zo, 21.3)
    raises(lambda: a_new.gap_mm(zn), ValueError, "requires pin_length_mm")
    # the loss of a ManiSkill window, offset head on: this code and 6b37757 (same seed)
    torch.manual_seed(4)
    old = OW.WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).train()
    torch.manual_seed(4)
    new = WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).train()
    W = windows_W(tr, 1, 7)
    torch.manual_seed(5)
    l_old, i_old = old.loss({k: v for k, v in W.items() if k != "gap_mm"}, None, 0, 5, 0.0)
    torch.manual_seed(5)
    l_new, i_new = new.loss(W, None, 0, 5, 0.0)
    assert torch.equal(l_old, l_new), (float(l_old), float(l_new))
    assert list(i_old) == list(i_new) and all(np.array_equal(i_old[k], i_new[k], equal_nan=True) for k in i_old), (i_old, i_new)
    assert i_new["offset_loss"] > 0                       # the offset head was trained in both
    return (f"trajectory / window (along) = {LEGACY_COMMIT} for every key bit for bit; along gap = old formula "
            f"(1000 draws); old checkpoint: along, gap_mm = {LEGACY_COMMIT}'s; WM320.loss identical ({float(l_new):.6g})")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="offset_gap_"))
    try:
        print("1 test_labels OK:", test_labels(), flush=True)
        print("2 test_no_pin_length OK:", test_no_pin_length(tmp), flush=True)
        print("3 test_maniskill_unchanged OK:", test_maniskill_unchanged(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
