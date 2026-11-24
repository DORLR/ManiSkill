"""The offset label of AutoMate (the user's decision 6 of 2026-11-23) on a CPU, no simulator, no GPU:
  1 labels: for AutoMate trajectories (meta simulator "automate") the world model's offset target (wmdata320 "offset"
    of trajectory() and window()) has as first component the pin-tip gap g = s - L [mm], with s = -(hand - H*) . axis
    computed here independently in float64 (g > 0 in front of the entrance, g < 0 past it, g = -L seated); its
    other three components equal "probe"; offset_label "gap_mm"; "probe" itself unchanged
  2 no pin length at test time: common320.gap_estimate_mm with "gap_mm" returns o[0] and never calls the pin-length
    function (a poisoned one raises when called); the WMAgent of an AutoMate checkpoint (offset_label "gap_mm") gives
    gap_mm(z) = offset_mm(z)[0], and a pin length handed to it (a poisoned number that raises on any use) is refused
    untouched; check_offset_label refuses an AutoMate attempt with an "along_mm" checkpoint and a ManiSkill attempt with
    a "gap_mm" one; WMAgent refuses an AutoMate checkpoint (provenance split) without the gap label;
    simulator_of_robot reads the class only (a robot whose every attribute raises); in attempt320.OnlineTester.attempt
    self.L is read only in meta_l (labels for reporting), in the pin-length function given to gap_estimate_mm and in
    the final labels; face_gap_mm only under mode "collect"; port_spec never
  3 ManiSkill unchanged: trajectory() / window() of a ManiSkill trajectory equal git 6b37757's for every key of that
    commit bit for bit, and "offset" equals "probe" bit for bit; the gap of an "along_mm" checkpoint equals 6b37757's
    formula bit for bit; a checkpoint without offset_label gives a WMAgent with "along_mm" whose gap_mm(z, L) equals
    6b37757's WMAgent.gap_mm(z, L); WM320.loss of the same window (offset head on) equals 6b37757's, same seed
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


def poisoned_length():
    raise AssertionError("the pin-length function was called")


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
    assert WD.offset_label(meta) == WD.OFFSET_GAP and WD.offset_label(meta_of(Hs, axis, False)) == WD.OFFSET_ALONG
    tr = WD.trajectory(z, meta, 0.3, 0.967)
    steps = 8 * np.arange(tr["n"] + 1)
    s = -((z["hand"][steps, :3, 3] - Hs[:3, 3]) @ axis)               # [m], float64, independent of wmdata320
    g = (s - L_M) * 1000.0
    assert np.allclose(tr["offset"][:, 0], g, atol=1e-4, rtol=0), np.abs(tr["offset"][:, 0] - g).max()
    assert g[0] > 11.9 and abs(g[-1] + L_M * 1000.0) < 1e-6, "in front at the start (g > 0), g = -L when seated"
    assert np.array_equal(tr["offset"][:, 1:], tr["probe"][:, 1:]), "lateral x, y and tilt as probe"
    ref = WD.decision_labels(meta, z["hand"], steps)["probe"]
    assert np.array_equal(tr["probe"], ref), "probe itself unchanged"
    for k in range(tr["n"]):
        w = WD.window(tr, k, STATS["s0"], STATS["top"])
        dec = np.minimum(k + np.arange(WD.H + 1), tr["n"])
        assert np.array_equal(w["offset"], tr["offset"][dec]) and np.array_equal(w["probe"], tr["probe"][dec])
    return f"AutoMate offset[0] = s - L within 1e-4 mm over {len(g)} decisions (g from {g[0]:.2f} to {g[-1]:.2f} mm); other components = probe"


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
    assert C.gap_estimate_mm(o, WD.OFFSET_GAP, poisoned_length) == -3.25
    assert C.gap_estimate_mm(o, WD.OFFSET_ALONG, lambda: 21.3) == float(-o[0] - 21.3)
    raises(lambda: C.gap_estimate_mm(o, "probe", lambda: 21.3), ValueError, "unknown offset label")
    raises(lambda: C.gap_estimate_mm(o, WD.OFFSET_ALONG, poisoned_length), AssertionError, "was called")   # the poison works
    C.check_offset_label(WD.OFFSET_GAP, "automate")
    C.check_offset_label(WD.OFFSET_ALONG, "maniskill")
    raises(lambda: C.check_offset_label(WD.OFFSET_ALONG, "automate"), ValueError, "train a new one")
    raises(lambda: C.check_offset_label(WD.OFFSET_GAP, "maniskill"), ValueError, "'along_mm'")
    am = type("Robot", (), {"__module__": "wm.sim_automate.robot", "__getattribute__": RaisingRobot.__getattribute__})()
    assert C.simulator_of_robot(am) == "automate" and C.simulator_of_robot(RaisingRobot()) == "maniskill"
    torch.manual_seed(0)
    m = WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).eval()
    split = {"name": "split_automate_v1", "sha256": "a" * 64}
    p = tmp / "am.pt"
    torch.save(ckpt(m, offset_label=WD.OFFSET_GAP, provenance={"split": split}), p)
    ag = WMAgent(str(p), "cpu")
    assert ag.offset_label == WD.OFFSET_GAP
    z = ag.encode(*observation(1))
    o = ag.offset_mm(z)
    assert ag.gap_mm(z) == float(o[0])
    raises(lambda: ag.gap_mm(z, Poison(21.3)), ValueError, "no pin length")
    torch.save(ckpt(m, provenance={"split": split}), tmp / "am_old.pt")            # AutoMate, before decision 6
    raises(lambda: WMAgent(str(tmp / "am_old.pt"), "cpu"), ValueError, "train a new one")
    torch.save(ckpt(m, offset_label="probe"), tmp / "bad.pt")
    raises(lambda: WMAgent(str(tmp / "bad.pt"), "cpu"), ValueError, "this code knows")
    # attempt320: where self.L, face_gap_mm and port_spec appear in OnlineTester.attempt (source, no simulator)
    tree = ast.parse((HERE.parent / "src" / "wm" / "online" / "attempt320.py").read_text())   # not imported: ManiSkill imports
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
                if isinstance(a, ast.Lambda) and isinstance(parent[a], ast.Call) and getattr(parent[a].func, "id", "") == "gap_estimate_mm":
                    assert parent[a].args[2] is a, "the pin-length function of gap_estimate_mm"
                    ctx = "gap_estimate_mm's pin-length function"
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
    assert sorted(c for _, c in where) == ["final labels (after the loop)", "gap_estimate_mm's pin-length function", "meta_l"], where
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "gap_estimate_mm"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[1]) == "agent.offset_label"
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "check_offset_label" for n in ast.walk(fn))
    return (f"gap_mm label: g = o[0], the pin-length function never called, a poisoned pin length refused untouched; wrong "
            f"labels refused per simulator; old AutoMate checkpoint refused; attempt reads self.L only at {where}")


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
    tr, tr0 = WD.trajectory(z, meta, 0.3, 0.967), OD.trajectory(z, meta, 0.3, 0.967)
    for k, v in tr0.items():
        assert type(v) is type(tr[k]) and (np.array_equal(v, tr[k]) and np.asarray(v).dtype == np.asarray(tr[k]).dtype
                                          if isinstance(v, np.ndarray) else v == tr[k]), k
    assert set(tr) - set(tr0) == {"offset"} and tr["offset"] is tr["probe"]
    for k in range(tr["n"]):
        w, w0 = WD.window(tr, k, STATS["s0"], STATS["top"]), OD.window(tr0, k, STATS["s0"], STATS["top"])
        assert set(w) - set(w0) == {"offset"} and np.array_equal(w["offset"], w["probe"]) and w["offset"].dtype == w["probe"].dtype
        assert all(np.array_equal(w[key], w0[key]) and w[key].dtype == w0[key].dtype for key in w0), k
    rng = np.random.default_rng(5)
    for _ in range(1000):                              # the gap of an along_mm checkpoint = 6b37757's formula, bit for bit
        o, L = rng.normal(size=4) * 30, float(rng.uniform(5, 40))
        assert C.gap_estimate_mm(o, WD.OFFSET_ALONG, lambda: L) == float(-o[0] - L)
    torch.manual_seed(2)
    m = WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).eval()
    p = tmp / "ms.pt"
    torch.save(ckpt(m), p)                                                        # a ManiSkill checkpoint: no key
    a_new, a_old = WMAgent(str(p), "cpu"), OA.WMAgent(str(p), "cpu")
    assert a_new.offset_label == WD.OFFSET_ALONG
    obs = observation(3)
    zn, zo = a_new.encode(*obs), a_old.encode(*obs)
    assert torch.equal(zn, zo)
    assert a_new.gap_mm(zn, 21.3) == a_old.gap_mm(zo, 21.3)
    raises(lambda: a_new.gap_mm(zn), ValueError, "needs the pin length")
    # the loss of a ManiSkill window, offset head on: this code and 6b37757 (same seed)
    torch.manual_seed(4)
    old = OW.WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).train()
    torch.manual_seed(4)
    new = WM320("standard", STATS, 0.3, offset_weight=1.0, gamma=0.967).train()
    W = windows_W(tr, 1, 7)
    torch.manual_seed(5)
    l_old, i_old = old.loss({k: v for k, v in W.items() if k != "offset"}, None, 0, 5, 0.0)
    torch.manual_seed(5)
    l_new, i_new = new.loss(W, None, 0, 5, 0.0)
    assert torch.equal(l_old, l_new), (float(l_old), float(l_new))
    assert list(i_old) == list(i_new) and all(np.array_equal(i_old[k], i_new[k], equal_nan=True) for k in i_old), (i_old, i_new)
    assert i_new["offset_loss"] > 0                       # the offset head was trained in both
    return (f"trajectory / window = {LEGACY_COMMIT} for every key bit for bit, offset = probe; along_mm gap = old formula "
            f"(1000 draws); old checkpoint: along_mm, gap_mm = {LEGACY_COMMIT}'s; WM320.loss identical ({float(l_new):.6g})")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="offset_gap_"))
    try:
        print("1 test_labels OK:", test_labels(), flush=True)
        print("2 test_no_pin_length OK:", test_no_pin_length(tmp), flush=True)
        print("3 test_maniskill_unchanged OK:", test_maniskill_unchanged(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
