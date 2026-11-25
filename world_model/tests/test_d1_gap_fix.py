#!/usr/bin/env python3
"""Thorough CPU unit tests for the D1 zero-privilege gap-fix overlay, finetune_r8_gap_head.py, and apply_overlay.py.

Structured in three tiers so it runs both:
  - Locally with Python 3.9+ stdlib only (Tier 1: AST privilege audit, exact gap/fidelity math, split320 vs
    strict_extrap logic, act_done validation, OnlineTester.L runtime guard, WMAgent.gap_mm contract, real
    SIGTERM/SIGUSR1 GracefulStopper subprocess test, and apply_overlay.py end-to-end staged apply/check/rollback).
  - On `fir` (or any numpy + torch CPU venv), automatically running Tier 2 (real wmdata320 + fidelity_factual320
    numpy tests) and Tier 3 (real PyTorch WM320, WMAgent, checkpoint backwards-compat, and warm-start tests).

Usage:
  python3 test_d1_gap_fix_cpu.py
  python3 test_d1_gap_fix_cpu.py --require-torch   # fail if torch/numpy are missing (use on fir)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import py_compile
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types
from pathlib import Path

D1_DIR = Path(__file__).resolve().parent
OVERLAY = D1_DIR / "overlay"
WM_SRC = OVERLAY / "world_model" / "src"
WM_SCRIPTS = OVERLAY / "world_model" / "scripts"
LOCAL_REPO = D1_DIR.parents[2] / "task_release" / "ManiSkill-WMInsertion-v2"

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _exec_selected_defs(py_path: Path, names: set[str], extra_globals: dict | None = None) -> dict:
    """Parse `py_path` and exec only top-level Assign / FunctionDef / ClassDef whose names are in `names`.
    Allows testing exact functions from overlay modules even when optional deps (numpy/torch/scipy) are absent."""
    src = py_path.read_text()
    tree = ast.parse(src, filename=str(py_path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            targets = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
                elif isinstance(t, ast.Tuple):
                    targets.extend(e.id for e in t.elts if isinstance(e, ast.Name))
            if any(t in names for t in targets):
                selected.append(node)
    mod_ast = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod_ast)
    ns: dict = {"__name__": f"_extracted_{py_path.stem}"}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(mod_ast, str(py_path), "exec"), ns)
    return ns


# ==============================================================================
# TIER 1: Pure stdlib tests (always run locally AND on fir)
# ==============================================================================
def test_1_syntax_and_zero_privilege_ast() -> str:
    py_files = [D1_DIR / "apply_overlay.py", *sorted(OVERLAY.rglob("*.py"))]
    assert len(py_files) == 10, f"expected 10 python files, got {len(py_files)}"
    for p in py_files:
        py_compile.compile(str(p), doraise=True)

    # Audit agent320.py: never references pin_length_m or self.L; gap_mm enforces zero-privilege in gap mode
    agent_src = (WM_SRC / "wm" / "online" / "agent320.py").read_text()
    assert "pin_length_m\"" not in agent_src and "self.L" not in agent_src
    assert 'if self.offset_mode == "gap":' in agent_src
    assert "if pin_length_mm is not None:" in agent_src

    # Audit attempt320.py: no inline `-o[0] - self.L` leak; uses _read_agent_offset_and_gap and _ForbidPinLength
    att_src = (WM_SRC / "wm" / "online" / "attempt320.py").read_text()
    assert "-o[0] - self.L" not in att_src, "legacy inline self.L gap computation still present in attempt320.py!"
    assert "_read_agent_offset_and_gap(agent, z, pin_length_m=None if offset_mode == \"gap\" else self.L)" in att_src
    assert "with _ForbidPinLength(self, forbid_L):" in att_src
    assert 'bad = mode not in ("act", "act_done") and p < NOT_DONE_P' in att_src

    # Audit train_wm320.py defaults: --offset-mode defaults to "gap", --shift-px defaults to 0
    tr_src = (WM_SCRIPTS / "train" / "train_wm320.py").read_text()
    assert '--offset-mode", choices=list(WD.OFFSET_MODES), default="gap"' in tr_src
    assert '--shift-px", type=int, default=0' in tr_src
    assert "--max-steps-this-job" in tr_src and "--stop-after-steps" in tr_src
    return f"all {len(py_files)} overlay/script files compile cleanly; AST zero-privilege audit passed"


def test_2_gap_math_and_fidelity_equivalence_pure() -> str:
    # Verify exact gap_mm = -along_mm - L_mm math and fidelity along-progress cancellation -(g2 - g1) == along2 - along1
    # Test across all connector pin lengths 14..20 mm and full approach -> face -> inside -> seated trajectory
    for L_mm in (14.0, 15.5, 16.0, 17.3, 18.0, 20.0):
        # along_mm <= 0 before H*: start 10 mm in front of face -> face -> 3 mm inside -> seated (along=0)
        alongs = [-(L_mm + 10.0), -(L_mm + 2.0), -L_mm, -(L_mm - 0.5), -(L_mm - 3.0), 0.0]
        gaps = [-a - L_mm for a in alongs]
        assert abs(gaps[0] - 10.0) < 1e-12
        assert abs(gaps[2] - 0.0) < 1e-12, "at board face (along = -L), gap must be 0"
        assert abs(gaps[3] - (-0.5)) < 1e-12, "0.5 mm inside hole -> gap = -0.5"
        assert abs(gaps[4] - (-3.0)) < 1e-12, "3 mm inside hole -> gap = -3.0"
        assert abs(gaps[5] - (-L_mm)) < 1e-12, "at H* (along = 0), gap = -L"
        for i in range(len(alongs)):
            for j in range(i, len(alongs)):
                prog_from_along = alongs[j] - alongs[i]
                prog_from_gap = -(gaps[j] - gaps[i])
                assert abs(prog_from_gap - prog_from_along) < 1e-12, (L_mm, i, j, prog_from_gap, prog_from_along)
                # face_items equivalence: along >= -L - 0.5  <=>  gap <= 0.5
                assert (alongs[j] >= -L_mm - 0.5 - 1e-12) == (gaps[j] <= 0.5 + 1e-12)
    # Also execute true_gap_from_probe and along_progress_from_delta directly from overlay fidelity_factual320.py
    class _FakeArray(list):
        def __neg__(self):
            return _FakeArray([-x for x in self])
        def __sub__(self, other):
            return _FakeArray([a - b for a, b in zip(self, other)])

    fake_np = types.SimpleNamespace(
        ndarray=object,
        float64=float,
        asarray=lambda x, dtype=None: _FakeArray([float(v) for v in x]) if isinstance(x, (list, tuple)) else float(x),
    )
    ff_ns = _exec_selected_defs(
        WM_SCRIPTS / "analysis" / "fidelity_factual320.py",
        {"true_gap_from_probe", "along_progress_from_delta"},
        {"np": fake_np},
    )
    L_vec = [14.0, 16.5, 20.0]
    a0 = [-20.0, -16.5, -14.0]
    a1 = [-18.0, -16.4, -10.5]
    g0 = [-x - l for x, l in zip(a0, L_vec)]
    g1 = [-x - l for x, l in zip(a1, L_vec)]
    got_gap_g = ff_ns["true_gap_from_probe"](g0, L_vec, "gap")
    got_gap_a = ff_ns["true_gap_from_probe"](a0, L_vec, "along")
    assert all(abs(x - y) < 1e-12 for x, y in zip(got_gap_g, got_gap_a))
    got_prog_g = ff_ns["along_progress_from_delta"]([y - x for x, y in zip(g0, g1)], "gap")
    got_prog_a = ff_ns["along_progress_from_delta"]([y - x for x, y in zip(a0, a1)], "along")
    assert all(abs(x - y) < 1e-12 for x, y in zip(got_prog_g, got_prog_a))
    return "gap_mm = -along_mm - L_mm, face threshold, and overlay fidelity_factual320 -(g2-g1)==along2-along1 exact"


def test_3_split320_and_strict_extrap_pure() -> str:
    import zlib
    OOD_VAL_SHAPES = ("G46", "G47", "G48") + tuple(f"G{i}" for i in range(55, 150, 10))
    TEST_PORTS = ("A", "B", "C", "D")
    ns = _exec_selected_defs(
        WM_SRC / "wm" / "tdmpc" / "demo320.py",
        {"SELECT_ONE_IN", "SPLIT_MODES", "is_interior_strip", "is_extrap_band", "split320", "_call_split", "group320"},
        {"zlib": zlib, "OOD_VAL_SHAPES": OOD_VAL_SHAPES, "TEST_PORTS": TEST_PORTS},
    )
    split320, group320, _call_split = ns["split320"], ns["group320"], ns["_call_split"]

    # Test ports A-D always 'test'
    for p in TEST_PORTS:
        for m in ("split320", "strict_extrap"):
            assert split320(p, -45.0, -0.025, f"{p}_g00", split_mode=m) == "test"
            assert group320(p, -45.0, -0.025, f"{p}_g00", split_mode=m) == "test_port"

    # Unseen shapes always 'heldout' / 'unseen_shape'
    for m in ("split320", "strict_extrap"):
        assert split320("G46", -45.0, -0.025, "G46_g00", split_mode=m) == "heldout"
        assert group320("G46", -45.0, -0.025, "G46_g00", split_mode=m) == "unseen_shape"

    # Interior strips always 'heldout' / 'interior_strip'
    for m in ("split320", "strict_extrap"):
        assert split320("G01", -40.0, -0.025, "G01_g00", split_mode=m) == "heldout"
        assert group320("G01", -40.0, -0.025, "G01_g00", split_mode=m) == "interior_strip"
        assert split320("G01", -45.0, -0.020, "G01_g01", split_mode=m) == "heldout"
        assert group320("G01", -45.0, -0.020, "G01_g01", split_mode=m) == "interior_strip"

    # Extrap outer bands: in train/select under split320, heldout/extrap_outer under strict_extrap!
    extrap_cases = [
        ("G01", -58.0, -0.025, "G01_ep0"),
        ("G01", -55.0, -0.025, "G01_ep1"),
        ("G01", -25.0, -0.025, "G01_ep2"),
        ("G01", -22.0, -0.025, "G01_ep3"),
        ("G01", -45.0, -0.029, "G01_ex0"),
        ("G01", -45.0, -0.028, "G01_ex1"),
        ("G01", -45.0, -0.012, "G01_ex2"),
        ("G01", -45.0, -0.011, "G01_ex3"),
    ]
    for port, pitch, pos, grasp in extrap_cases:
        assert split320(port, pitch, pos, grasp, split_mode="split320") in ("train", "select")
        assert split320(port, pitch, pos, grasp, split_mode="strict_extrap") == "heldout"
        assert group320(port, pitch, pos, grasp, split_mode="strict_extrap") == "extrap_outer"

    # In-band training grasps identical in both modes
    for port, pitch, pos, grasp in [("G01", -45.0, -0.025, "G01_tr0"), ("G02", -33.0, -0.016, "G02_tr1")]:
        s1 = split320(port, pitch, pos, grasp, split_mode="split320")
        s2 = split320(port, pitch, pos, grasp, split_mode="strict_extrap")
        assert s1 == s2 and s1 in ("train", "select")
        if s1 == "train":
            assert group320(port, pitch, pos, grasp, split_mode="split320") == "train_band"
            assert group320(port, pitch, pos, grasp, split_mode="strict_extrap") == "train_band"

    # Backwards-compat 4-arg monkeypatch of split320 (as in test_wm_cpu.py)
    orig = ns["split320"]
    ns["split320"] = lambda port, pitch, pos, grasp: "select" if grasp == "G00_g02" else orig(port, pitch, pos, grasp)
    assert _call_split("G00", -45.0, -0.025, "G00_g02", "split320") == "select"
    ns["split320"] = orig

    # Unknown split_mode raises ValueError
    try:
        split320("G01", -45.0, -0.025, "G01_g00", split_mode="bogus")
        raise AssertionError("expected ValueError for unknown split_mode")
    except ValueError:
        pass

    # Verify run_wm320_attempts._matches_split directly from overlay
    fake_D = types.SimpleNamespace(TEST_PORTS=TEST_PORTS, group320=group320, _call_split=_call_split)
    run_ns = _exec_selected_defs(
        WM_SCRIPTS / "online" / "run_wm320_attempts.py",
        {"_matches_split"},
        {"D": fake_D},
    )
    _matches_split = run_ns["_matches_split"]
    g_extrap = {"port": "G01", "pitch_deg": -58.0, "grasp_pos_m": -0.025, "grasp": "G01_ep0"}
    g_inband = {"port": "G01", "pitch_deg": -45.0, "grasp_pos_m": -0.025, "grasp": "G01_tr0"}
    assert _matches_split(g_extrap, "heldout", "strict_extrap") is True
    assert _matches_split(g_extrap, "extrap", "strict_extrap") is True
    assert _matches_split(g_extrap, "train", "strict_extrap") is False
    assert _matches_split(g_inband, "extrap", "strict_extrap") is False
    assert _matches_split(g_extrap, "heldout", "split320") is False
    return "split320 vs strict_extrap boundaries, extrap_outer grouping, _matches_split, and 4-arg monkeypatch compat verified"


def test_4_agent_gap_mm_and_attempt_guards_pure() -> str:
    # 1) Test WMAgent.gap_mm directly from agent320.py using a stub torch.no_grad decorator
    fake_torch = types.SimpleNamespace(no_grad=lambda: (lambda fn: fn), Tensor=object)
    ag_ns = _exec_selected_defs(
        WM_SRC / "wm" / "online" / "agent320.py",
        {"WMAgent"},
        {"torch": fake_torch, "np": types.SimpleNamespace(ndarray=object)},
    )
    gap_mm_fn = ag_ns["WMAgent"].gap_mm

    class _MockAgent:
        def __init__(self, offset_mode, off_val):
            self.offset_mode = offset_mode
            self._off_val = off_val
            self.offset_calls = 0

        def offset_mm(self, z):
            self.offset_calls += 1
            return self._off_val

        def gap_mm(self, z, pin_length_mm=None):
            return gap_mm_fn(self, z, pin_length_mm=pin_length_mm)

    # Zero-privilege gap mode: reads float(o[0]) directly, NO pin_length_mm allowed
    ag_gap = _MockAgent("gap", [-1.25, 0.4, -0.3, 1.1])
    assert ag_gap.gap_mm("z_dummy") == -1.25 and ag_gap.offset_calls == 1
    try:
        ag_gap.gap_mm("z_dummy", pin_length_mm=16.0)
        raise AssertionError("expected ValueError when pin_length_mm passed in gap mode")
    except ValueError as exc:
        assert "zero-privilege" in str(exc)
    assert ag_gap.offset_calls == 1, "must fail BEFORE touching offset_mm"

    # Gap mode without offset head returns None
    ag_gap_none = _MockAgent("gap", None)
    assert ag_gap_none.gap_mm("z_dummy") is None

    # Legacy along mode: requires pin_length_mm and computes -o[0] - pin_length_mm
    ag_along = _MockAgent("along", [-14.75, 0.4, -0.3, 1.1])
    assert abs(ag_along.gap_mm("z_dummy", pin_length_mm=16.0) - (-1.25)) < 1e-12
    try:
        ag_along.gap_mm("z_dummy")
        raise AssertionError("expected ValueError when pin_length_mm omitted in along mode")
    except ValueError:
        pass

    # 2) Test attempt320.py validation, _read_agent_offset_and_gap, _ForbidPinLength, and OnlineTester.L property
    class _DummyBrancher:
        pass

    att_ns = _exec_selected_defs(
        WM_SRC / "wm" / "online" / "attempt320.py",
        {"MODES", "validate_attempt_args", "_read_agent_offset_and_gap", "_ForbidPinLength", "OnlineTester"},
        {"RANDOM_EDGES_MM": (0.0, 2.0, 6.0), "Brancher": _DummyBrancher,
         "B": types.SimpleNamespace(MAX_DECISIONS=150), "np": types.SimpleNamespace(ndarray=object)},
    )
    assert "act_done" in att_ns["MODES"]
    validate = att_ns["validate_attempt_args"]
    read_og = att_ns["_read_agent_offset_and_gap"]
    ForbidL = att_ns["_ForbidPinLength"]
    OnlineTester = att_ns["OnlineTester"]

    # act_done validation rules
    validate("act_done", inside_mm=None, max_retries=3)
    validate("act", inside_mm=None, max_retries=3)
    validate("wm", inside_mm=0.5, max_retries=3)
    for bad_kwargs in (
        {"mode": "act_done", "inside_mm": 0.5},
        {"mode": "act_done", "probe_max": 1},
        {"mode": "servo", "inside_mm": None, "max_retries": 0},
        {"mode": "servo", "inside_mm": 0.5, "max_retries": 1},
        {"mode": "random", "inside_mm": 0.5, "max_retries": 0, "random_p": [0.1, 0.2]},
        {"mode": "invalid_mode"},
    ):
        try:
            validate(**bad_kwargs)
            raise AssertionError(f"expected ValueError for {bad_kwargs}")
        except ValueError:
            pass

    # _read_agent_offset_and_gap enforces zero-privilege in gap mode and backwards-compat in along mode
    o_g, g_g = read_og(ag_gap, "z_dummy", pin_length_m=None)
    assert o_g == [-1.25, 0.4, -0.3, 1.1] and g_g == -1.25
    try:
        read_og(ag_gap, "z_dummy", pin_length_m=0.016)
        raise AssertionError("expected RuntimeError when pin_length_m passed in gap mode")
    except RuntimeError as exc:
        assert "zero-privilege" in str(exc)

    o_a, g_a = read_og(ag_along, "z_dummy", pin_length_m=0.016)
    assert o_a == [-14.75, 0.4, -0.3, 1.1] and abs(g_a - (-1.25)) < 1e-12
    try:
        read_og(ag_along, "z_dummy", pin_length_m=None)
        raise AssertionError("expected ValueError when pin_length_m=None in along mode")
    except ValueError:
        pass

    # OnlineTester.L runtime guard via _ForbidPinLength
    tester = OnlineTester()
    tester.L = 0.0165
    assert tester.L == 0.0165
    with ForbidL(tester, True):
        try:
            _ = tester.L
            raise AssertionError("expected RuntimeError when reading tester.L under _ForbidPinLength")
        except RuntimeError as exc:
            assert "zero-privilege" in str(exc)
    assert tester.L == 0.0165, "tester.L must be readable again after exiting _ForbidPinLength"
    with ForbidL(tester, False):
        assert tester.L == 0.0165
    return "WMAgent.gap_mm, act_done validation, _read_agent_offset_and_gap, and OnlineTester.L guard verified"


def test_5_graceful_stopper_sigterm_sigusr1() -> str:
    # 1) Direct in-process test of GracefulStopper extracted from train_wm320.py
    ns = _exec_selected_defs(WM_SCRIPTS / "train" / "train_wm320.py", {"GracefulStopper"}, {"signal": signal})
    GracefulStopper = ns["GracefulStopper"]

    st = GracefulStopper().install()
    try:
        assert not st.requested and st.reason is None
        os.kill(os.getpid(), signal.SIGUSR1)
        assert st.requested and st.signum == int(signal.SIGUSR1) and st.reason == "signal_SIGUSR1"
    finally:
        st.restore()

    # 2) Subprocess test: real SIGTERM delivery triggers clean atomic last.pt save, NO DONE.json, exit 0
    with tempfile.TemporaryDirectory(prefix="d1_sigterm_") as td:
        td_path = Path(td)
        ready_file = td_path / "ready"
        last_pt = td_path / "last.pt"
        done_json = td_path / "DONE.json"
        child_script = textwrap.dedent(f"""
            from __future__ import annotations
            import ast, json, os, signal, time
            from pathlib import Path

            src = Path({str(WM_SCRIPTS / 'train' / 'train_wm320.py')!r}).read_text()
            tree = ast.parse(src)
            cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GracefulStopper"]
            ns = {{"signal": signal}}
            exec(compile(ast.Module(body=cls, type_ignores=[]), "train_wm320.py", "exec"), ns)
            GracefulStopper = ns["GracefulStopper"]

            stopper = GracefulStopper().install()
            step = 0
            event = "done"
            try:
                Path({str(ready_file)!r}).write_text("ok")
                while step < 10000:
                    if stopper.requested:
                        tmp = Path({str(last_pt)!r} + ".tmp")
                        tmp.write_text(json.dumps({{"step": step, "reason": stopper.reason}}))
                        os.replace(tmp, {str(last_pt)!r})
                        event = stopper.reason
                        break
                    step += 1
                    time.sleep(0.01)
            finally:
                stopper.restore()
            if event == "done":
                Path({str(done_json)!r}).write_text(json.dumps({{"step": step}}))
        """)
        proc = subprocess.Popen([sys.executable, "-c", child_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            t0 = time.time()
            while not ready_file.exists():
                if proc.poll() is not None:
                    out_s, err_s = proc.communicate()
                    raise AssertionError(f"child exited early rc={proc.returncode}: {err_s}")
                assert time.time() - t0 < 5.0, "child did not become ready"
                time.sleep(0.01)
            time.sleep(0.05)  # let a few steps run
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=5.0)
            assert proc.returncode == 0, f"expected clean exit 0, got rc={proc.returncode}, stderr={stderr}"
            assert last_pt.exists(), "last.pt was not saved on SIGTERM"
            assert not done_json.exists(), "DONE.json must NOT be written on SIGTERM"
            saved = json.loads(last_pt.read_text())
            assert saved["step"] >= 1 and saved["reason"] == "signal_SIGTERM", saved
        finally:
            if proc.poll() is None:
                proc.kill()
    return "GracefulStopper caught real SIGUSR1 (in-proc) and SIGTERM (subproc -> atomic last.pt, no DONE.json, rc=0)"


def test_6_apply_overlay_end_to_end() -> str:
    if not (LOCAL_REPO / ".git").exists():
        return "SKIPPED (local ManiSkill git repo not found)"
    sys.path.insert(0, str(D1_DIR))
    import apply_overlay as AO  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="d1_stage_maniskill_") as td:
        target = Path(td) / "ManiSkill"
        # Populate clean 324aa3f base files via git show (read-only on LOCAL_REPO)
        for rel, base_sha in AO.EXPECTED_BASE_SHA256.items():
            if base_sha is None:
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = subprocess.check_output(["git", "-C", str(LOCAL_REPO), "show", f"origin/world-model:{rel}"])
            dst.write_bytes(data)
            assert AO.sha256_file(dst) == base_sha, f"base hash mismatch on {rel}"

        # 1) --dry-run modifies nothing
        res_dry = AO.apply(target, dry_run=True)
        assert res_dry["would_apply"] == 9
        assert not (target / "world_model/scripts/train/finetune_r8_gap_head.py").exists()

        # 2) Real apply succeeds and writes backup
        res = AO.apply(target)
        assert res["applied"] == 9 and res["backup"] is not None
        assert (target / "world_model/scripts/train/finetune_r8_gap_head.py").is_file()

        # 3) --check exits 0 via CLI
        rc = subprocess.run([sys.executable, str(D1_DIR / "apply_overlay.py"), "--target", str(target), "--check"], check=False).returncode
        assert rc == 0

        # 4) Re-apply is idempotent
        res2 = AO.apply(target)
        assert res2["applied"] == 0 and res2["skipped"] == 9

        # 5) Unexpected local edit triggers refusal without --force
        p = target / "world_model/src/wm/tdmpc/wm320.py"
        orig_bytes = p.read_bytes()
        p.write_bytes(orig_bytes + b"\n# unexpected edit\n")
        try:
            AO.apply(target, force=False)
            raise AssertionError("expected SystemExit on unexpected local modification")
        except SystemExit:
            pass
        p.write_bytes(orig_bytes)

        # 6) Rollback restores exact 324aa3f hashes and removes new file
        AO.rollback(target, Path(res["backup"]))
        assert not (target / "world_model/scripts/train/finetune_r8_gap_head.py").exists()
        for rel, base_sha in AO.EXPECTED_BASE_SHA256.items():
            if base_sha is not None:
                assert AO.sha256_file(target / rel) == base_sha, f"rollback hash mismatch on {rel}"
    return "apply_overlay dry-run -> apply -> --check -> idempotent -> conflict refusal -> exact rollback passed"


def _configure_target_paths(target: Path | None) -> tuple[Path, Path]:
    """Resolve world_model/src and world_model/scripts from an applied --target ManiSkill checkout (or PYTHONPATH)."""
    if target is not None:
        wm_root = target / "world_model"
        if not (wm_root / "src" / "wm" / "tdmpc" / "wm320.py").is_file():
            raise SystemExit(f"--target {target} does not contain world_model/src/wm/tdmpc/wm320.py")
        src, scripts = wm_root / "src", wm_root / "scripts"
        for p in (str(src), str(scripts / "train"), str(scripts / "analysis"), str(scripts / "online"), str(scripts / "eval")):
            if p not in sys.path:
                sys.path.insert(0, p)
        return src, scripts
    return WM_SRC, WM_SCRIPTS


# ==============================================================================
# TIER 2: Numpy tests against real applied modules (run automatically on fir)
# ==============================================================================
def test_7_wmdata320_and_fidelity_numpy(target: Path | None = None) -> str:
    import numpy as np  # noqa: F401
    _configure_target_paths(target)
    from wm.tdmpc import wmdata320 as WD  # noqa: E402
    from wm.tdmpc import demo320 as D  # noqa: E402

    rng = np.random.default_rng(42)
    for L_m in (0.014, 0.016, 0.0185, 0.020):
        Hs = np.eye(4)
        Hs[:3, 3] = [0.5, -0.1, 0.3]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        B = WD._basis(axis)
        T = 24
        hand = np.repeat(Hs[None], T + 1, axis=0)
        # Move hand smoothly from 12 mm before face to seated at H* with random lateral jitter
        s_grid = np.linspace(-(L_m + 0.012), 0.0, T + 1)
        for t in range(T + 1):
            hand[t, :3, 3] = Hs[:3, 3] + s_grid[t] * B[0] + rng.normal(scale=0.002) * B[1] + rng.normal(scale=0.002) * B[2]
        meta = {"H_star": Hs.tolist(), "axis": axis.tolist(), "pin_length_m": L_m, "kind": "insert", "outcome": "inserted"}
        steps = np.arange(0, T + 1, WD.EXECUTE)
        lab_a = WD.decision_labels(meta, hand, steps, offset_mode="along")
        lab_g = WD.decision_labels(meta, hand, steps, offset_mode="gap")
        assert np.allclose(lab_g["probe"][:, 0], -lab_a["probe"][:, 0] - L_m * 1000.0, atol=1e-4)
        assert np.allclose(lab_g["probe"][:, 1:], lab_a["probe"][:, 1:], atol=1e-6)
        assert np.array_equal(lab_g["seated"], lab_a["seated"])
        assert np.allclose(lab_g["depth"], lab_a["depth"]) and np.allclose(lab_g["phi"], lab_a["phi"])

        z = {
            "action": np.zeros((T, D.JOINTS), np.float32),
            "img_step": 4 * np.arange(2 * (T // WD.EXECUTE) + 1),
            "qpos": np.zeros((T + 1, 9), np.float32),
            "hand": hand.astype(np.float32),
        }
        tr_a = WD.trajectory(z, meta, 0.3, 0.967, offset_mode="along")
        tr_g = WD.trajectory(z, meta, 0.3, 0.967, offset_mode="gap")
        assert np.allclose(tr_a["reward"], tr_g["reward"]) and np.array_equal(tr_a["term"], tr_g["term"])
        assert np.allclose(tr_g["probe"][:, 0], tr_g["gap_mm"])
        w_g = WD.window(tr_g, 0, np.ones(WD.ACTION_DIM, np.float32), 2 * np.ones(WD.ACTION_DIM, np.float32))
        assert np.allclose(w_g["probe"][:, 0], w_g["gap_mm"])

    # Also test fidelity_factual320 helper equivalence directly if torch is present (fidelity imports torch)
    if HAS_TORCH:
        import fidelity_factual320 as FF  # noqa: E402
        L_mm = np.array([14.0, 16.5, 19.8])
        along0 = np.array([-20.0, -16.5, -15.0])
        along1 = np.array([-18.0, -16.4, -12.0])
        gap0 = -along0 - L_mm
        gap1 = -along1 - L_mm
        assert np.allclose(FF.true_gap_from_probe(gap0, L_mm, "gap"), FF.true_gap_from_probe(along0, L_mm, "along"))
        assert np.allclose(FF.along_progress_from_delta(gap1 - gap0, "gap"), FF.along_progress_from_delta(along1 - along0, "along"))
    return "wmdata320 decision_labels/trajectory/window (and fidelity_factual320 helpers) exact across L=14..20 mm"


# ==============================================================================
# TIER 3: PyTorch tests against real applied WM320 / WMAgent / finetune script
# ==============================================================================
def test_8_wm320_agent_and_warmstart_torch(target: Path | None = None) -> str:
    import numpy as np
    import torch

    _configure_target_paths(target)
    from wm.tdmpc import demo320 as D  # noqa: E402
    from wm.tdmpc import wmdata320 as WD  # noqa: E402
    from wm.tdmpc.wm320 import WM320, load  # noqa: E402
    from wm.online.agent320 import WMAgent  # noqa: E402
    from finetune_r8_gap_head import init_gap_head_from_along, verify_zero_privilege_checkpoint  # noqa: E402

    stats = {
        "s0": np.ones(WD.ACTION_DIM, np.float32),
        "top": 2.0 * np.ones(WD.ACTION_DIM, np.float32),
        "a_mean": np.zeros(D.CHUNK * D.JOINTS, np.float32),
        "a_std": np.ones(D.CHUNK * D.JOINTS, np.float32),
    }

    with tempfile.TemporaryDirectory(prefix="d1_torch_") as td:
        td_path = Path(td)
        # 1) Legacy checkpoint (no offset_mode key) defaults to "along"
        m_along = WM320("standard", stats, shaping=0.3, offset_weight=1.0, offset_mode="along", gamma=0.967).eval()
        ck_old = {
            "format": "wm320-1", "labels": WD.LABELS, "stem": "standard", "imagenet": False,
            "offset_weight": 1.0, "shaping": 0.3, "cfg": {"gamma": 0.967}, "stats": stats,
            "state": m_along.state_dict(), "step": 61000,
        }
        p_old = td_path / "legacy_r8.pt"
        torch.save(ck_old, p_old)
        m_loaded_old = load(str(p_old), "cpu")
        assert m_loaded_old.offset_mode == "along"
        ag_old = WMAgent(str(p_old), "cpu")
        assert ag_old.offset_mode == "along"
        z = torch.randn(1, m_along.cfg.latent_dim)
        o_old = ag_old.offset_mm(z)
        assert abs(ag_old.gap_mm(z, pin_length_mm=16.0) - float(-o_old[0] - 16.0)) < 1e-5
        try:
            ag_old.gap_mm(z)
            raise AssertionError("legacy along agent accepted gap_mm(z) without pin_length_mm")
        except ValueError:
            pass

        # 2) Warm-start dim 0 from along -> gap preserves dims 1..3 bit-for-bit and transforms dim 0 exactly
        mean_L_mm = 17.0
        m_gap = WM320("standard", stats, shaping=0.3, offset_weight=1.0, offset_mode="gap", gamma=0.967).eval()
        m_gap.load_state_dict(ck_old["state"])
        with torch.no_grad():
            before = (m_gap._offset(z) * m_gap.off_scale)[0].double().numpy()
        ok = init_gap_head_from_along(m_gap._offset, float(m_gap.off_scale[0]), mean_L_mm)
        assert ok, "expected Linear warm-start to succeed"
        with torch.no_grad():
            after = (m_gap._offset(z) * m_gap.off_scale)[0].double().numpy()
        assert abs(after[0] - (-before[0] - mean_L_mm)) < 1e-4, (before[0], after[0])
        assert np.allclose(after[1:], before[1:], atol=1e-6)

        # 3) Save gap checkpoint and verify zero-privilege WMAgent contract
        ck_gap = {**ck_old, "offset_mode": "gap", "state": m_gap.state_dict()}
        p_gap = td_path / "r8_gap.pt"
        torch.save(ck_gap, p_gap)
        zp = verify_zero_privilege_checkpoint(p_gap, device="cpu")
        assert zp["offset_mode"] == "gap" and zp["rejects_pin_length_mm"] is True
        ag_gap = WMAgent(str(p_gap), "cpu")
        assert abs(ag_gap.gap_mm(z) - float(after[0])) < 1e-5
    return "PyTorch WM320 legacy/gap load, warm-start math, and WMAgent zero-privilege contract verified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="path to applied ManiSkill checkout for Tier 2/3 imports on fir")
    ap.add_argument("--require-torch", action="store_true", help="fail if numpy or torch is unavailable (use on fir)")
    args = ap.parse_args()
    target = Path(args.target) if args.target else None

    print(f"Environment: Python {sys.version.split()[0]}, numpy={HAS_NUMPY}, torch={HAS_TORCH}", flush=True)
    if args.require_torch and not (HAS_NUMPY and HAS_TORCH):
        raise SystemExit("--require-torch set, but numpy and/or torch is not installed in this environment")

    tier1 = [
        test_1_syntax_and_zero_privilege_ast,
        test_2_gap_math_and_fidelity_equivalence_pure,
        test_3_split320_and_strict_extrap_pure,
        test_4_agent_gap_mm_and_attempt_guards_pure,
        test_5_graceful_stopper_sigterm_sigusr1,
        test_6_apply_overlay_end_to_end,
    ]
    passed, skipped = 0, []
    for fn in tier1:
        msg = fn()
        print(f"[PASS][Tier 1] {fn.__name__}: {msg}", flush=True)
        passed += 1

    if HAS_NUMPY:
        msg = test_7_wmdata320_and_fidelity_numpy(target)
        print(f"[PASS][Tier 2] test_7_wmdata320_and_fidelity_numpy: {msg}", flush=True)
        passed += 1
    else:
        print("[SKIP][Tier 2] test_7_wmdata320_and_fidelity_numpy (numpy not installed locally; run in fir venv)", flush=True)
        skipped.append("test_7_wmdata320_and_fidelity_numpy")

    if HAS_NUMPY and HAS_TORCH:
        msg = test_8_wm320_agent_and_warmstart_torch(target)
        print(f"[PASS][Tier 3] test_8_wm320_agent_and_warmstart_torch: {msg}", flush=True)
        passed += 1
    else:
        print("[SKIP][Tier 3] test_8_wm320_agent_and_warmstart_torch (torch not installed locally; run in fir venv)", flush=True)
        skipped.append("test_8_wm320_agent_and_warmstart_torch")

    print(f"\nSUMMARY: {passed} passed, {len(skipped)} skipped (pending fir torch venv).", flush=True)


if __name__ == "__main__":
    main()
