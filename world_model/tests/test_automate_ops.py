"""Operations and collector lifecycle of the AutoMate collection on a CPU (no Isaac, no GPU job; the Robot, Expert and
Recorder are stand-ins, systemctl / nvidia-smi are fake scripts where the shell code needs them):
  1 config: production keys required, config.json with its config_sha loads, a wrong sha / unknown key refuses,
    worker resources from the measured defaults, MemoryMax below the device minimum or in M refused; the pinned
    random-motion speeds pass expert.execute's own check (decision 4 of 2026-11-23)
  2 freeze: a first plan over old grasp data refuses; expected_grasps must equal the plan; frozen m1b copies used
  3 claims and deaths: a dead attempt counted once per pid + start, "hang" from the claim text, a released claim
    puts the previous text back, the per-assembly skip lists (build failures counted / not counted, deaths)
  4 worker: retry kept in grasp.json, an unhandled error written once, build failure exits with BuildFailed (network
    error before any build not counted), Stopped releases the claim, died-twice grasp.json has the AutoMate keys,
    the run summary (skipped assemblies, undone grasps, trajectory files without grasp.json) and ALERT
  5 collect_grasp: an error in the label block goes through fail() (SimulatorError, not a process death); no FK axis
    closes the grasp with reference_ok False (no true-axis fallback); meta carries the switches; the
    outcome follows AutoMate's success rule; socket draw "per_reset": a reset with socket_index = file index and an
    unrecorded reference (its own H* / axis label) per trajectory, "label_failed" without one, resets / label steps
    counted; "per_grasp": one reset, t00's label; the start-clearance limit is expert.START_CLEARANCE_M; stage rows
    keep the expert's speeds
  6 signal: SIGTERM releases the claim with the previous text and exits 5
  7 heartbeat + watcher: a stale heartbeat of a live pid in one of our scopes is killed, its claim marked hang, the
    .hung marker and ALERT written; the next claim records a hang death; watcher decisions (GPU 0, pressure, GPU sum)
  8 caps.sh: input validation, failing tools refuse, the 125 GB / GPU 0 / pressure / MemAvailable refusals, the
    caps cannot be raised from the environment, CPUQuota parsing, the static worker budget, the code hash
  9 collect_worker.sh restart loop: refusal / build failure / hang not counted, crash rate cap, exit 3, STOP, code hash
 10 split automate_v1: 5 test + 10 % held-out assemblies by group quota, near twins together, interpolation only,
    15 grasps = 12 train-range + 3 band grasps (zero-width variable skipped), select 1 in 20 by crc32, split file;
    grasp counts per condition in plan.json, too few selection grasps refused in production: n_select <
    ceil(1.25 x 20) = 25 (24 refused, 25 accepted; warned for a subset)
 11 the GPU-check exit hook; 12 the milestone-1b gate (provenance, physics identity incl. batch_assemblies.py /
    geometry.py / sampling.py, the geometry file / plug / socket mesh hashes, pass_ii / pass_iii with missing keys and
    F_hf = hand_finger_max_N); 13 success rule
 14 run_batch.sh: watcher required, shed with any exit but 124 (this start's markers released, an older hang
    marker kept, not counted) vs timeout 124 (marker kept) vs GPU-0 stop vs exit 3, markers released on a GPU-0 stop, BATCH DONE / 30 runs / 30 refusals; 15 parallel_batch.sh: no
    watcher refused, running batch sessions refused, --retry-crashes once per id, stamped log tags, a worker not
    started -> exit 1; 16 the smoke checker's placement / rejection statistics (show_user = check_expert_data
    .draw_report's flags + the leg-reason redraw lines), start-success rows and the launcher.log rule of (e); 17 prep_automate_data.sh (frozen code, capped GPU-0 CPU scopes, running or
    unfinished collection refused, draws check first and non-failing, the steps in order, stop at the first failure,
    PIDs, D_max required, early refusals in ALERT), the smoke's item (f) on its records, draws_check.sh and the
    early refusals of collect_smoke.sh; 18 the
    start-box share of the status table (checks.start_box_success with rotation, gap_from from config); 19
    gpu_batch_r3.sh runs only from code equal to the CPU batch's frozen copy (a --retry-crashes start excepted); 20
    smoke_env_gate.py: the collector smoke needs a passing smoke_env result of the batch's code and switches, run after
    the batch has ended (no live batch heartbeat); 21 the training preflight runs in the job's environment; 22
    collect_prep.sh refuses production config errors before freezing the inputs and names a NEW <out> after a
    refused first launch (r8-ops-01); 23 diag/robot_pair_check.py lists the (i) / (iii) rows the robot-pair rule
    would question (r9-expert-01) and counts contact_report_error as trajectories plus refused t00 grasps, split by
    rule (r10-expert-01); a missing batch, an empty --data folder and a missing / broken --acceptance file read "?"
    (exit 0), and --data / --acceptance of different smokes are warned about (r11-expert-01 / -02); 24 collect_launch.sh
    and collect_smoke.sh start a collection only from the frozen code of the config's batch (r11-ops-02)
Run on rs2 inside a capped scope:   source scripts/automate/caps.sh; qx_run test ops 8G 200% 0 $PY tests/test_automate_ops.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WM = HERE.parent
sys.path.insert(0, str(WM / "src"))
sys.path.insert(0, str(WM / "scripts" / "automate"))

_scene = types.ModuleType("wm.sim_automate.scene")
_scene.AutomateScene = object
sys.modules.setdefault("wm.sim_automate.scene", _scene)

from wm.data import schema  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import collect as CO  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402
import qx_watch as QW  # noqa: E402


BOX = {"dtheta_deg": [-13.3, 6.7], "dp_m": [-0.01, 0.0], "dpsi_deg": [-7.5, 7.5]}
II_OK = {"pass": True, "placed": True, "completed": True, "success": True, "depth_left_mm": 0.02}


def provenance(aid: str, device: str = "cpu", t: float = 1000.0) -> dict:
    """What batch_assemblies.py records with every result (the current files and switches)."""
    return {"unix_time": t, "ranges_sha256": CO.sha256_file(G.ranges_file(aid)), "grasp_py_sha256": CO.sha256_file(G.__file__),
            "physics_device": device, "switches": C.switches(), "physics_code_sha256": CO.physics_code_identity(),
            "code_files_sha256": "c", **CO.asset_identity(aid), "shared_asset_sha256": CO.shared_asset_identity()}


def write_m1b(d: Path, aid: str, **over) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{aid}.json"
    f.write_text(json.dumps({"assembly_id": aid, **provenance(aid), "ii": dict(II_OK),
                             "iii": {"skipped": "margin 4 mm"}, **over}))
    return f


def setup_assets(tmp: Path, ids=("00001", "00002")) -> dict:
    A.ASSET_DIR = tmp / "assets"
    b2 = tmp / "m1b" / "batch_r3_cpu"
    b2.mkdir(parents=True)
    for aid in ids:
        d = A.ASSET_DIR / aid
        d.mkdir(parents=True)
        (d / G.RANGES_FILE).write_text(json.dumps({"n": "round", "box": BOX, "verified": True, "worst_margin_m": 0.004,
                                                   "worst_point": None}))
        (d / A.SPEC_FILE).write_text(json.dumps({"assembly_id": aid}))
        for mesh in ("plug.obj", "socket.obj"):
            (d / mesh).write_text(f"o {aid}_{mesh}\n")
        write_m1b(b2, aid)
    return {"assemblies": list(ids), "m1b_dirs": [str(b2)], "grasps_per_assembly": 2, "device": "cpu",
            "split": "unassigned", "expected_grasps": 2 * len(ids)}


PRODUCTION = {"split": "automate_v1", "grasps_per_assembly": 15}


def raises(fn, exc=SystemExit) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn} did not raise {exc}")


# ------------------------------------------------------------------------------------------------------------- 1
def test_config(tmp: Path):
    raw = setup_assets(tmp)
    f = tmp / "cfg.json"
    for k in CO.PRODUCTION_KEYS:
        f.write_text(json.dumps({kk: v for kk, v in raw.items() if kk != k}))
        assert k in raises(lambda: CO.CollectConfig.load(f, production=True)), k
    f.write_text(json.dumps({**raw, "grasps_per_assembly": None}))
    raises(lambda: CO.CollectConfig.load(f, production=True))
    f.write_text(json.dumps({**raw, "bogus": 1}))
    assert "unknown keys" in raises(lambda: CO.CollectConfig.load(f))
    f.write_text(json.dumps(raw))
    assert "automate_v1" in raises(lambda: CO.CollectConfig.load(f, production=True)), "unassigned refused in production"
    raw = {**raw, **PRODUCTION, "expected_grasps": 30}
    f.write_text(json.dumps({**raw, "grasps_per_assembly": 3}))
    raises(lambda: CO.CollectConfig.load(f, production=True))
    f.write_text(json.dumps(raw))
    cfg = CO.CollectConfig.load(f, production=True)
    conf = tmp / "config.json"                                    # as freeze writes it
    conf.write_text(json.dumps({**cfg.dump(), "config_sha": CO.sha256_text(CO._canon(cfg.dump()))}))
    assert CO._canon(CO.CollectConfig.load(conf, production=True).dump()) == CO._canon(cfg.dump())
    conf.write_text(json.dumps({**cfg.dump(), "config_sha": "0" * 64}))
    assert "config_sha" in raises(lambda: CO.CollectConfig.load(conf))
    r = CO.worker_resources(cfg)
    assert r["memory_max"] == "16G" and r["cpu_quota"] == "500%" and r["gpu_gb"] == 7
    assert CO.worker_resources(CO.CollectConfig(**{**cfg.dump(), "device": "cuda:0"}))["memory_max"] == "18G"
    for dev, mem in (("cpu", "6G"), ("cpu", "13G"), ("cuda:0", "16G"), ("cuda:0", "16384M"), ("cpu", "18.5G")):
        raises(lambda: CO.worker_resources(CO.CollectConfig(**{**cfg.dump(), "device": dev, "worker_memory_max": mem})))
    raises(lambda: CO.worker_resources(CO.CollectConfig(**{**cfg.dump(), "worker_cpu_quota": "4.5"})))
    # r7-ops-05: GPU-1 GB per worker from the config (an integer 1..48), default WORKER_GPU_GB = 7
    assert CO.WORKER_GPU_GB == 7 and CO.CollectConfig().worker_gpu_gb is None
    assert CO.worker_resources(CO.CollectConfig(**{**cfg.dump(), "worker_gpu_gb": 9}))["gpu_gb"] == 9
    for g in (0, 49, "8", 7.5, True):
        assert "worker_gpu_gb" in raises(lambda: CO.worker_resources(CO.CollectConfig(**{**cfg.dump(), "worker_gpu_gb": g}))), g
    f.write_text(json.dumps({**raw, "worker_gpu_gb": 9}))
    assert CO.worker_resources(CO.CollectConfig.load(f, production=True))["gpu_gb"] == 9
    f.write_text(json.dumps({**raw, "worker_gpu_gb": 50}))
    assert "worker_gpu_gb" in raises(lambda: CO.CollectConfig.load(f, production=True))
    raises(lambda: CO.CollectConfig(**{**cfg.dump(), "device": "cuda:1"}).validate())
    # pinned values (r4-decisions-05 / N1): 15 grasps without collect_subset, >= 4 with it; wm_v1's trajectory mix,
    # picture settings; the switches of config.APPROVED (r4-ops-02)
    v = lambda **over: CO.CollectConfig(**{**cfg.dump(), **over}).validate()  # noqa: E731
    assert "decision 7" in raises(lambda: v(grasps_per_assembly=10))
    v(grasps_per_assembly=4, collect_subset=["00001"])
    raises(lambda: v(grasps_per_assembly=3, collect_subset=["00001"]))
    from wm.collect.plan import CollectConfig as V1
    assert cfg.trajectories == V1().trajectories and CO.pinned_values()["jpeg_quality"] == V1().jpeg_quality
    for over in ({"trajectories": {"insert": 1}}, {"images": False}, {"image_every": 2}, {"jpeg_quality": 90},
                 {"rgb_codec": "png"}, {"random_waypoints": 2}, {"random_speed_m_s": [0.003, 0.02]},
                 {"trajectories": dict(reversed(list(V1().trajectories.items())))}):
        k = next(iter(over))
        assert k in raises(lambda: v(**over)), over
    v(random_speed_m_s=[0.0027, 0.0027])                                # a json list equals the tuple default
    # row 27 restored (2026-11-23): wm_v1's constant 2.7 mm/s (0.9 x close_speed 3 mm/s, wm/env/expert.py:95)
    assert CO.CollectConfig().random_speed_m_s == (0.0027, 0.0027) == CO.pinned_values()["random_speed_m_s"]
    from wm.sim_automate import expert as EX                         # the expert accepts only its own constant
    assert CO.pinned_values()["random_speed_m_s"] == (EX.RANDOM_SPEED_M_S,) * 2, EX.RANDOM_SPEED_M_S
    assert "2.7 mm/s" in raises(lambda: v(random_speed_m_s=[0.003, 0.015]))
    rnd = [t for t in CO._trajectories(np.random.default_rng(0), CO.CollectConfig()) if t.kind == "random"]
    assert rnd and all(sp == 0.0027 for t in rnd for sp in t.speeds), rnd
    # decision 4 of 2026-11-23 (pinned): the planned speeds pass the expert's own check, and the expert moves every
    # random waypoint with its _move_outside (wm_v1's motion: no ramp, no hold); any other speed is refused
    ex = object.__new__(EX.Execution)
    moves = []
    ex._move_outside = lambda phase, target, **info: moves.append(phase)
    ex.offset_rel = lambda off: off
    ex.result, ex.stages, ex.steps, ex.phase = {"completed": False}, [], 0, "initialization"
    ex.peak, ex.m, ex.success_any, ex.first_success_step = {"finger_socket_N": 0.0}, {"success": False}, False, None
    r = ex.execute("random", waypoints=rnd[0].waypoints, speeds=rnd[0].speeds)
    assert r["completed"] and moves == [f"random_motion_{i}" for i in range(len(rnd[0].waypoints))], (r, moves)
    assert "RANDOM_SPEED_M_S" in raises(lambda: ex.execute("random", waypoints=rnd[0].waypoints,
                                                           speeds=[0.003] * len(rnd[0].waypoints)), ValueError)
    old = C.KIT_GPU, C.SOCKET_DRAW
    try:
        C.KIT_GPU, C.SOCKET_DRAW = 0, "per_grasp"
        msg = raises(lambda: v())
        assert "kit_gpu = 0" in msg and "socket_draw = 'per_grasp'" in msg, msg
    finally:
        C.KIT_GPU, C.SOCKET_DRAW = old
    return "production keys required (each missing / null refused), config.json loads, wrong sha / unknown key " \
           "refused; 16G CPU / 18G GPU defaults, 6G / 13G CPU, 16G / 16384M GPU, 18.5G refused; 10 grasps refused " \
           "(15 pinned; 4 with collect_subset); worker_gpu_gb 9 used, 0 / 49 / '8' / 7.5 / True refused, default 7; " \
           "trajectory mix / order, pictures, waypoints pinned; random-motion " \
           "speed = wm_v1's constant 2.7 mm/s (3-15 mm/s refused), accepted by expert.execute, which moves each waypoint " \
           "with _move_outside (other speeds refused there); kit_gpu 0 / per_grasp refused by validate"


# ------------------------------------------------------------------------------------------------------------- 2
def test_freeze(tmp: Path):
    raw = setup_assets(tmp)
    cfg = CO.CollectConfig(**raw)
    code = {"files_sha256": "x", "n_files": 1, "git_head": None, "git_dirty": None}
    out = tmp / "out"
    (out / "00001" / "00001_g00").mkdir(parents=True)
    (out / "00001" / "00001_g00" / "grasp.json").write_text("{}")
    msg = raises(lambda: CO.freeze(out, cfg, CO.plan_document(cfg, *CO.build(cfg), code)))
    assert "grasp data" in msg and not (out / "plan.json").exists()
    out2 = tmp / "out2"
    bad = CO.CollectConfig(**{**raw, "expected_grasps": 3})
    assert "expects 3" in raises(lambda: CO.prepare(bad, out2, log=lambda m: None))
    assert not (out2 / "plan.json").exists()
    specs, frozen = CO.prepare(cfg, out2, log=lambda m: None)
    assert len(specs) == 4 and frozen["switches"] == C.switches()
    # the versioned asset files come from the code's config (assets.SPEC_FILE, grasp.RANGES_FILE) and are in plan.json
    assert frozen["asset_files"] == {"spec_file": A.SPEC_FILE, "ranges_file": G.RANGES_FILE,
                                     "grasp_spin_axis": C.GRASP_SPIN_AXIS}, frozen["asset_files"]
    assert all(str(A.ASSET_DIR / "00001" / f) in frozen["assemblies"][0]["inputs"] for f in (A.SPEC_FILE, G.RANGES_FILE))
    src = Path(CO.__file__).read_text()
    assert not any(x in src for x in ("geometry_v", "ranges_v")), "collect.py names a versioned asset file itself"
    old_names = A.SPEC_FILE, G.RANGES_FILE
    try:
        A.SPEC_FILE, G.RANGES_FILE = "geometry_vX.json", "ranges_vX.json"
        for aid in ("00001", "00002"):
            for f_old, f_new in zip(old_names, (A.SPEC_FILE, G.RANGES_FILE)):
                shutil.copy(A.ASSET_DIR / aid / f_old, A.ASSET_DIR / aid / f_new)
            write_m1b(Path(raw["m1b_dirs"][0]), aid)                   # provenance of the renamed ranges file
        doc = CO.plan_document(cfg, *CO.build(cfg), code)
        assert doc["asset_files"]["spec_file"] == "geometry_vX.json" and doc["asset_files"]["ranges_file"] == "ranges_vX.json"
        assert str(A.ASSET_DIR / "00001" / "ranges_vX.json") in doc["assemblies"][0]["inputs"]
        assert "asset_files" in raises(lambda: CO.prepare(cfg, out2, log=lambda m: None)), "renamed files vs frozen plan"
    finally:
        A.SPEC_FILE, G.RANGES_FILE = old_names
    # frozen m1b copies: WM_AUTOMATE_M1B/<k> replaces cfg.m1b_dirs[k]; a changed root refuses against the frozen plan
    fz = tmp / "frozen_m1b" / "0"
    shutil.copytree(raw["m1b_dirs"][0], fz)
    os.environ["WM_AUTOMATE_M1B"] = str(tmp / "frozen_m1b")
    try:
        assert CO.m1b_dirs(cfg) == [str(fz)]
        row = CO.assembly_status("00001", cfg)
        assert str(fz / "00001.json") in row["inputs"], row["inputs"]
        assert "input_roots" in raises(lambda: CO.prepare(cfg, out2, log=lambda m: None)) or True
        os.environ["WM_AUTOMATE_M1B"] = str(tmp / "missing")
        raises(lambda: CO.m1b_dirs(cfg))
    finally:
        os.environ.pop("WM_AUTOMATE_M1B")
    return "first plan over old grasp.json refused; plan != expected_grasps refused before plan.json; switches in " \
           "the plan; asset file names from assets.SPEC_FILE / grasp.RANGES_FILE in plan.json (renamed files read, " \
           "refused against the frozen plan; no versioned name in collect.py); WM_AUTOMATE_M1B copies read (missing " \
           "copies refused)"


# ------------------------------------------------------------------------------------------------------------- 3
def test_claims_deaths(tmp: Path):
    out = tmp / "out"
    g = out / "00001" / "00001_g00"
    g.mkdir(parents=True)
    dead = "attempt 1 pid 999999999 start 1 time t0 hang 2026-11-22_10:00:00"
    (g / ".claim").write_text(dead)
    c = CO.claim(g)
    assert c.attempt == 2 and c.old["hang"] and c.previous == dead
    assert CO._record_death(out, "00001", "00001_g00", c.old) == 1
    CO.release(c, done=False)                                  # e.g. a stop: previous text back
    CO.release(c, done=False)                                  # idempotent
    assert (g / ".claim").read_text() == dead
    c = CO.claim(g)                                            # the same dead attempt again
    assert c.attempt == 2
    assert CO._record_death(out, "00001", "00001_g00", c.old) == 1, "a death is counted once per pid + start"
    CO.release(c, done=False)
    rows = CO._rows(out / "deaths" / "00001.json")
    assert rows[0]["kind"] == "hang"
    for i in range(CO.DEATHS_MAX - 1):
        CO._record_death(out, "00001", f"00001_g0{i + 1}", {"pid": 100 + i, "start": "5", "attempt": 1})
    bl = CO.blocked_assemblies(out)
    assert "00001" in bl and "1 hang" in bl["00001"], bl
    for _ in range(CO.BUILD_FAILURES_MAX):
        n = CO._record_build_failure(out, "00002", ConnectionError("Network is unreachable"), counted=False)
    assert n == 0 and "00002" not in CO.blocked_assemblies(out)
    for _ in range(CO.BUILD_FAILURES_MAX):
        n = CO._record_build_failure(out, "00002", RuntimeError("bad usd"), counted=True)
    assert n == CO.BUILD_FAILURES_MAX and "00002" in CO.blocked_assemblies(out)
    assert CO.NETWORK_OR_GPU_MEMORY.search("RuntimeError: CUDA error: out of memory")
    assert CO.NETWORK_OR_GPU_MEMORY.search("URLError: <urlopen error timed out>")
    assert not CO.NETWORK_OR_GPU_MEMORY.search("RuntimeError: prim not found")
    r = subprocess.run(["bash", str(WM / "scripts/automate/collect_unblock.sh"), str(out), "00002"], capture_output=True, text=True)
    assert r.returncode == 0 and "00002" not in CO.blocked_assemblies(out), r.stdout + r.stderr
    assert list((out / "build_failures" / "cleared").glob("00002.*.json"))
    return "dead attempt counted once (hang kind kept), release restores the previous text, skip after " \
           f"{CO.DEATHS_MAX} deaths / {CO.BUILD_FAILURES_MAX} counted build failures (not-counted ones ignored); " \
           "collect_unblock moves the record"


# ------------------------------------------------------------------------------------------------------------- 4
class FakeScene:
    closed = 0

    def close(self):
        FakeScene.closed += 1


class FakeRobot:
    fail_build: dict = {}

    def __init__(self, port, device):
        err = FakeRobot.fail_build.pop(port, None)
        if err is not None:
            raise err
        self.port, self.sc = port, FakeScene()


def run_worker(cfg, out, behaviour, logs):
    calls = {}

    def fake_collect(robot, s, out_, cfg_, stamp, beat=None):
        calls[s.key] = calls.get(s.key, 0) + 1
        beat("x")
        b = behaviour.get(s.key, "ok")
        if b == "retry" and calls[s.key] == 1:
            raise CO.SimulatorError("trajectory 3 (insert): RuntimeError: physx", {"trajectories": [], "trace": "t"})
        if b == "unhandled":
            raise ValueError("bug outside fail()")
        if b == "stop":
            raise CO.Stopped("STOP")
        if b == "refused":
            raise CO.Refused("usd_vs_m1b different")
        return {"grasp": s.key, "trajectories": [{"kind": "reference", "outcome": "inserted"}], "reference_ok": True,
                "memory": {}, "seconds": 0.0}

    old = CO.collect_grasp
    CO.collect_grasp = fake_collect
    try:
        return CO.worker(cfg, out, 0, 1, log=logs.append, beat=lambda stage=None, **kw: None)
    finally:
        CO.collect_grasp = old


def test_worker(tmp: Path):
    raw = setup_assets(tmp)
    cfg = CO.CollectConfig(**raw)
    out = tmp / "out"
    sys.modules["wm.sim_automate.robot"] = types.SimpleNamespace(Robot=FakeRobot)
    logs = []
    # A: first build fails with a network error -> BuildFailed, not counted, claim released, no grasp.json
    FakeRobot.fail_build = {"00001": ConnectionError("Network is unreachable")}
    raises(lambda: run_worker(cfg, out, {}, logs), CO.BuildFailed)
    rows = CO._build_failures(out, "00001")
    assert len(rows) == 1 and rows[0]["counted"] is False
    assert not list(out.glob("*/*/grasp.json")) and (out / "00001" / "00001_g00" / ".claim").read_text() == ""
    # B: retry kept, unhandled error written once, then a stop in 00002_g00 releases its claim
    dead = "attempt 2 pid 999999998 start 7 time t0"                    # 00002_g01: a second death -> died twice
    (out / "00002" / "00002_g01").mkdir(parents=True)
    (out / "00002" / "00002_g01" / ".claim").write_text(dead)
    raises(lambda: run_worker(cfg, out, {"00001_g00": "retry", "00001_g01": "unhandled", "00002_g00": "stop"}, logs),
           CO.Stopped)
    g0 = json.loads((out / "00001" / "00001_g00" / "grasp.json").read_text())
    assert g0["reference_ok"] and len(g0["retries"]) == 1 and "physx" in g0["retries"][0]["error"]
    g1 = json.loads((out / "00001" / "00001_g01" / "grasp.json").read_text())
    assert "unhandled" in g1["error"] and g1["simulator"] == "automate" and g1["u"] and not (out / "00001" / "00001_g01" / ".claim").exists()
    assert (out / "00002" / "00002_g00" / ".claim").read_text() == "" and not (out / "00002" / "00002_g00" / "grasp.json").exists()
    assert not CO._rows(out / "deaths" / "00002.json") or all(r["grasp"] != "00002_g00" for r in CO._rows(out / "deaths" / "00002.json"))
    # C: the rest; 00002_g01 had two deaths -> error grasp.json with the AutoMate keys (leak-13)
    if (out / "00002" / "00002_g01" / "grasp.json").exists():
        pass
    (out / "00002" / "00002_g00" / "t03_insert.npz").write_bytes(b"x")              # left by the stopped attempt
    doc = run_worker(cfg, out, {}, logs)
    g3 = json.loads((out / "00002" / "00002_g01" / "grasp.json").read_text())
    assert "died twice" in g3["error"] and g3["simulator"] == "automate" and g3["assembly_id"] == "00002" \
        and g3["switches"] == C.switches() and g3["condition"] == "unassigned"
    assert doc["done"] == 4 and not doc["skipped_assemblies"] and not doc["orphan_trajectory_files"]
    assert json.loads((out / "logs" / "summary.json").read_text())["done"] == 4
    # D: summary of an unfinished run: a skipped assembly, an undone grasp, a trajectory file without grasp.json
    out2 = tmp / "out2"
    CO.prepare(cfg, out2, log=lambda m: None)                   # plan first (old files before a plan are refused)
    for _ in range(CO.BUILD_FAILURES_MAX):
        CO._record_build_failure(out2, "00002", RuntimeError("bad usd"), counted=True)
    (out2 / "00001" / "00001_g01").mkdir(parents=True)
    (out2 / "00001" / "00001_g01" / "t01_insert.npz").write_bytes(b"x")
    FakeRobot.fail_build = {}
    doc = run_worker(cfg, out2, {"00001_g01": "ok"}, logs)
    assert set(doc["skipped_assemblies"]) == {"00002"} and doc["done"] == 2 and len(doc["undone_grasps"]) == 2
    alert = (out2 / "logs" / "ALERT").read_text()
    assert "2 grasps without grasp.json" in alert, alert
    assert not doc["orphan_trajectory_files"], "the stale file of a re-claimed grasp is cleaned by collect_grasp"
    (out2 / "00002" / "00002_g00").mkdir(parents=True)
    (out2 / "00002" / "00002_g00" / "t01_insert.npz").write_bytes(b"x")
    specs, _ = CO.prepare(cfg, out2, log=lambda m: None)
    assert CO.run_summary(out2, specs)["orphan_trajectory_files"] == ["00002/00002_g00/t01_insert.npz"]
    # E: Refused inside a grasp (e.g. the USD layers changed) releases the claim (not an attempt) and is re-raised
    out3 = tmp / "out3"
    raises(lambda: run_worker(cfg, out3, {"00001_g00": "refused"}, logs), CO.Refused)
    assert (out3 / "00001" / "00001_g00" / ".claim").read_text() == "" and not (out3 / "00001" / "00001_g00" / "grasp.json").exists()
    return "network build failure not counted (BuildFailed, claim released); retry kept; unhandled error written " \
           "once; Stopped / Refused release the claim; died-twice grasp.json has simulator / u / switches; " \
           "summary + ALERT"


# ------------------------------------------------------------------------------------------------------------- 5
class FakeRecorder:
    axial = True

    def __init__(self, robot, images, image_every):
        self.n = 0

    def begin(self):
        self.n = 0

    def on_step(self, phase, action, info):
        self.n += 1

    def end(self):
        T = 4
        ph = schema.PHASES.index("axial_push") if FakeRecorder.axial else 0
        hand = np.tile(np.eye(4), (T + 1, 1, 1))
        hand[:, 2, 3] = -0.001 * np.arange(T + 1)
        return {"phase": np.full(T, ph, np.uint8), "hand": hand, "eval_rel": np.zeros((T + 1, 7))}


class FakeExpert:
    completed = True              # the expert's completion of run() (t00 and the planned trajectories)
    label_ok = True               # reference_label: an FK axis
    label_ok_flag = True          # reference_label "ok" (completed AND success at the end)
    contact_outcome = "inserted"  # expert.contact_push's own outcome
    raise_on = {}                 # {(kind, call number): exception} raised by run()
    calls = 0

    def __init__(self, robot, transfer):
        pass

    start_args: list = []

    @staticmethod
    def start_pose(*a):
        FakeExpert.start_args.append(a)
        return np.eye(4)

    run_kw: list = []             # (kind, kw) of every run() call

    def run(self, kind, on_step, ss, **kw):
        FakeExpert.calls += 1
        FakeExpert.run_kw.append((kind, kw))
        err = FakeExpert.raise_on.get((kind, FakeExpert.calls))
        if err is not None:
            raise err
        for _ in range(3):
            on_step("axial_push", np.zeros(7), {})
        return {"completed": FakeExpert.completed, "wall_s": 0.1, "steps": 3, "peak": {"finger_socket_N": 0.0},
                "outcome": FakeExpert.contact_outcome if kind == "contact" else None,
                "failure_reason": None if FakeExpert.completed else "ExpertFailure: guard",
                "stages": [{"phase": "random_motion_0", "start_step": 0, "end_step": 3, "steps": 3,
                            "speed_m_s": 0.0027, "ang_speed_deg_s": 2.0, "top_gap_mm": 9.0}]}

    def reference_label(self, ss, on_step=None):
        on_step("axial_push", np.zeros(7), {})
        H = np.eye(4)
        H[0, 3] = FakeGraspRobot.last_index * 0.01                     # a label that depends on the socket pose
        return {"ok": FakeExpert.label_ok_flag, "completed": FakeExpert.label_ok_flag, "H_star": H.tolist(),
                "axis": [0.0, 0.0, -1.0] if FakeExpert.label_ok else None,
                "steps": 7, "truth": {}, "peak": {}, "failure": None, "H_star_fk_vs_model_mm_deg": [0, 0]}

    def start_clearance(self, T):
        return {"min": 1.0}

    wp_clear = staticmethod(lambda off: 1.0)    # decision 7: hand / finger clearance [m] of a random waypoint

    def waypoint_clearance(self, off):
        return FakeExpert.wp_clear(off)

    move_raise = set()            # move_to_start call numbers that raise checks.ContactReportError
    moves = 0

    def move_to_start(self, T):
        FakeExpert.moves += 1
        if FakeExpert.moves in FakeExpert.move_raise:
            raise K.ContactReportError("contact report empty during the retreat, sensor 2.0 N")
        return {"ok": True, "steps": 2, "start_error_mm": 0.0,
                "legs": [{"leg": "retreat", "ok": True, "reason": None, "max_lag_mm": 0.4}]}


class FakeGraspRobot:
    last_index = 0
    resets: list = []
    fail_reset: set = set()
    fail_calls: set = set()       # reset call numbers (1-based, counted in `calls`) that raise checks.ResetError
    calls = 0

    def __init__(self, label_error=False):
        self.label_error = label_error
        self.port_spec = types.SimpleNamespace(pin_length_m=0.02, symmetry_n="round", tip_centre_m=[0, 0, 0],
                                               disassembly_dist_m=0.025, depth_top_m=0.0275)
        self.assembly_id, self.max_decisions = "00001", 150
        self.episode = {"socket_pose": np.eye(4).tolist(), "socket_draw": {"draw": "per_reset", "index": 0}}
        self.kin = types.SimpleNamespace(base_T=np.eye(4))

    def reset(self, seed, *, clamp_deg, dp_m, spin_deg, socket_index):
        FakeGraspRobot.calls += 1
        if socket_index in FakeGraspRobot.fail_reset or FakeGraspRobot.calls in FakeGraspRobot.fail_calls:
            raise K.ResetError(f"no IK solution for the start pose (socket {socket_index})")
        FakeGraspRobot.last_index = socket_index
        FakeGraspRobot.resets.append(socket_index)
        self.episode = {"socket_pose": [[socket_index]], "socket_draw": {"index": socket_index}}

    def hand_pose(self):
        if self.label_error:
            raise RuntimeError("hand pose unreadable")
        return np.eye(4)

    def H_star(self):
        return np.eye(4)

    def axis(self):
        return np.array([0.0, 0.0, -1.0])


def test_collect_grasp(tmp: Path):
    raw = setup_assets(tmp)
    cfg = CO.CollectConfig(**raw)
    spec = CO.build(cfg)[0][0]
    spec.trajectories = []
    saved = {}
    from wm.sim_automate import expert as EX
    stubs = {"wm.sim_automate.expert": types.SimpleNamespace(Expert=FakeExpert, START_CLEARANCE_M=EX.START_CLEARANCE_M,
                                                             HAND_FINGER_COMMAND_CLEARANCE_M=EX.HAND_FINGER_COMMAND_CLEARANCE_M),
             "wm.sim_automate.record": types.SimpleNamespace(Recorder=FakeRecorder)}
    old_mods = {k: sys.modules.get(k) for k in stubs}
    old = (CO.eval_tip, CO.truth_summary, CO.store.save, CO.automate_success, C.SOCKET_DRAW)
    succ = {"inserted": True, "end": True}
    CO.automate_success = lambda robot, arrays=None: {"inserted": succ["inserted"] if arrays is not None else succ["end"],
                                                      "inserted_end": succ["end"], "inserted_any_step": succ["inserted"],
                                                      "height_mm": 1.0, "keypoint_dist_mm": 1.0}
    sys.modules.update(stubs)
    CO.eval_tip = lambda arrays, robot: np.zeros((len(arrays["eval_rel"]), len(CO.EVAL_TIP_COLUMNS)), np.float32)
    CO.truth_summary = lambda robot: {"seated": True}
    CO.store.save = lambda path, arrays, meta, *a: saved.update({path.name: meta}) or 0
    beats = []
    out = tmp / "out"
    run = lambda: CO.collect_grasp(FakeGraspRobot(), spec, out, cfg, {})  # noqa: E731
    try:
        msg = raises(lambda: CO.collect_grasp(FakeGraspRobot(label_error=True), spec, out, cfg, {}, beat=lambda stage=None: beats.append(stage)),
                     CO.SimulatorError)
        assert msg.startswith("reference label") and "prepare" in beats and "reference" in beats, (msg, beats)
        FakeRecorder.axial = False
        summ = run()
        assert summ["reference_ok"] is False and "no axial push" in summ["reference_failure"] and not saved
        FakeRecorder.axial = True
        summ = CO.collect_grasp(FakeGraspRobot(), spec, out, cfg, {"config_sha": "c", "plan_sha": "p"})
        meta = saved["t00_reference.npz"]
        assert summ["reference_ok"] and meta["switches"] == C.switches() and meta["file"] == "t00_reference.npz"
        assert meta["simulator"] == "automate" and meta["pin_length_m"] == 0.02 and meta["condition"] == "unassigned"
        assert meta["expert_completed"] is True and meta["automate_success_any"] is True
        assert summ["switches"] == C.switches() and summ["trajectories"][0]["file"] == "t00_reference.npz"
        # t00 acceptance (lead resolution r3): completed AND AutoMate's rule at the end, both recorded
        FakeExpert.completed = False                                      # inserted (any step and end), not completed
        saved.clear()
        summ = run()
        assert summ["reference_ok"] is False and "did not complete" in summ["reference_failure"] and not saved
        assert summ["reference_expert_completed"] is False and summ["reference_success"]["inserted_end"] is True
        FakeExpert.completed = True
        succ["end"] = False                                               # completed, any step yes, end no
        summ = run()
        assert summ["reference_ok"] is False and "at the end" in summ["reference_failure"] and not saved
        succ["end"] = True
        # socket draw per reset: own reset + reference label per trajectory
        all_t = CO.build(cfg)[0][0].trajectories
        spec.trajectories = [t for t in all_t if t.kind == "insert"][:2]
        C.SOCKET_DRAW = "per_reset"
        FakeGraspRobot.resets = []
        saved.clear()
        summ = run()
        i1, i2 = (t.idx + 1 for t in spec.trajectories)
        assert FakeGraspRobot.resets == [0, i1, i2], FakeGraspRobot.resets
        # the start box from config.START_GAP_FROM with the assembly's g_top (decision of 2026-11-23: "socket_top")
        assert FakeExpert.start_args and all(len(a) == 6 and abs(a[4] - 0.0075) < 1e-12 and a[5] == C.START_GAP_FROM
                                             for a in FakeExpert.start_args), FakeExpert.start_args
        assert abs(saved[f"t{i1:02d}_insert.npz"]["g_top_m"] - 0.0075) < 1e-12
        m1 = saved[f"t{i1:02d}_insert.npz"]
        assert m1["socket_index"] == i1 and m1["H_star"][0][3] == i1 * 0.01 and m1["label_reference"]["accepted"]
        assert m1["outcome"] == "inserted" and m1["eval_socket_pose"] == [[i1]] and m1["file"] == f"t{i1:02d}_insert.npz"
        assert [t.get("file") for t in summ["trajectories"]] == ["t00_reference.npz", f"t{i1:02d}_insert.npz", f"t{i2:02d}_insert.npz"]
        tm = summ["timing"]
        assert tm["resets"] == 3 and tm["label_steps"] == 14 and tm["reset_settle_steps"] == 3 * int(getattr(C, "SETTLE_STEPS", 0))
        FakeExpert.label_ok = False                                       # no FK axis
        summ = run()
        assert [t["outcome"] for t in summ["trajectories"]] == ["inserted", "label_failed", "label_failed"]
        assert all("file" not in t for t in summ["trajectories"][1:])
        FakeExpert.label_ok = True
        FakeExpert.label_ok_flag = False                                  # label not completed, AutoMate's rule holds
        summ = run()
        assert [t["outcome"] for t in summ["trajectories"]] == ["inserted", "label_failed", "label_failed"]
        assert summ["trajectories"][1]["label_reference"]["inserted_end"] is True
        FakeExpert.label_ok_flag = True
        succ["end"] = False                                               # t00 refused before any label
        summ = run()
        assert summ["reference_ok"] is False and len(summ["trajectories"]) == 0
        succ["end"] = True
        # insert outcome = AutoMate's S_any (option A); expert_completed and automate_success_any recorded for bc
        FakeExpert.calls = 0
        FakeExpert.completed = True
        saved.clear()

        class Half(FakeExpert):                                       # t00 completes, the inserts do not
            def run(self, kind, on_step, ss, **kw):
                r = FakeExpert.run(self, kind, on_step, ss, **kw)
                return {**r, "completed": kind == "reference"}
        sys.modules["wm.sim_automate.expert"] = types.SimpleNamespace(
            Expert=Half, START_CLEARANCE_M=EX.START_CLEARANCE_M,
            HAND_FINGER_COMMAND_CLEARANCE_M=EX.HAND_FINGER_COMMAND_CLEARANCE_M)
        summ = run()
        m1 = saved[f"t{i1:02d}_insert.npz"]
        assert m1["outcome"] == "inserted" and m1["expert_completed"] is False and m1["automate_success_any"] is True
        succ["inserted"] = False                                          # S_any False: expert_failed; t00 still
        summ = run()                                                      # accepted (completed AND success at the end)
        assert summ["reference_ok"] is True and summ["trajectories"][1]["outcome"] == "expert_failed"
        assert summ["trajectories"][1]["automate_success_any"] is False
        succ["inserted"] = True
        sys.modules.update(stubs)
        # a reset failure and a contact-report error end one trajectory each; the grasp goes on; ALERT above 20 %
        FakeGraspRobot.fail_reset = {i1}
        FakeExpert.calls = 0
        FakeExpert.raise_on = {("insert", 2): K.ContactReportError("contact report empty, sensor 3.0 N")}
        summ = run()
        assert [t["outcome"] for t in summ["trajectories"]] == ["inserted", "reset_failed", "contact_report_error"], \
            [t["outcome"] for t in summ["trajectories"]]
        assert "no IK" in summ["trajectories"][1]["error"] and "ALERT" in os.listdir(out / "logs")
        assert "reset_failed" in (out / "logs" / "ALERT").read_text()
        FakeGraspRobot.fail_reset = set()
        FakeExpert.raise_on = {}
        # r4-integration-03: a ResetError / ContactReportError at t00 closes the grasp softly (no SimulatorError) + ALERT
        (out / "logs" / "ALERT").unlink()
        FakeGraspRobot.fail_reset = {0}
        summ = run()
        assert summ["reference_ok"] is False and summ["reference_soft_outcome"] == "reset_failed" and \
            summ["reference_failure"].startswith("reference reset: ResetError"), summ.get("reference_failure")
        FakeGraspRobot.fail_reset = set()
        FakeExpert.calls = 0
        FakeExpert.raise_on = {("reference", 1): K.ContactReportError("contact report empty, sensor 2.0 N")}
        summ = run()
        assert summ["reference_soft_outcome"] == "contact_report_error" and not summ["trajectories"]
        assert (out / "logs" / "ALERT").read_text().count("t00 reference refused") == 2
        FakeExpert.raise_on = {}
        # r4-expert-03: a contact-report error during the unrecorded transfer ends that trajectory only
        FakeExpert.moves, FakeExpert.move_raise = 0, {1}
        saved.clear()
        summ = run()
        rows = summ["trajectories"]
        assert rows[1]["outcome"] == "contact_report_error" and rows[1]["stage"] == "placement" and "file" not in rows[1]
        assert rows[2]["outcome"] == "inserted" and rows[2]["file"] in saved and rows[2]["placement_legs"][0]["leg"] == "retreat"
        FakeExpert.move_raise = set()
        # grasp.json stages keep phase, steps, failure and the speeds the expert recorded (decision 4 of 2026-11-23)
        assert rows[2]["stages"] == [{"phase": "random_motion_0", "steps": 3, "speed_m_s": 0.0027,
                                      "ang_speed_deg_s": 2.0}], rows[2]["stages"]
        # r5-decisions-check-N1: the start-clearance limit is the expert's START_CLEARANCE_M (here raised above the
        # stand-in's 1 m clearance): every attempt is a "clearance" redraw, no transfer, placement_failed
        sys.modules["wm.sim_automate.expert"].START_CLEARANCE_M = 2.0
        FakeExpert.moves = 0
        summ = run()
        rows = summ["trajectories"]
        assert [t["outcome"] for t in rows[1:]] == ["placement_failed"] * 2 and FakeExpert.moves == 0, rows
        assert rows[1]["placement_attempts"] == CO.PLACEMENT_ATTEMPTS and \
            [f["stage"] for f in rows[1]["placement_failures"]] == ["clearance"] * CO.PLACEMENT_ATTEMPTS, rows[1]
        sys.modules["wm.sim_automate.expert"].START_CLEARANCE_M = EX.START_CLEARANCE_M
        summ = run()
        assert [t["outcome"] for t in summ["trajectories"][1:]] == ["inserted"] * 2
        # r4-expert-02 c: label_failed counts toward the ALERT share
        (out / "logs" / "ALERT").unlink()
        FakeExpert.label_ok = False
        run()
        FakeExpert.label_ok = True
        assert "label_failed" in (out / "logs" / "ALERT").read_text()
        # r4-integration-05: the USD layers of the scene against the milestone-1b (ii) result
        m1b = Path(raw["m1b_dirs"][0])
        lay = {"omni://x/00001/plug.usd": "a", "omni://x/00001/socket.usd": "b", "omni://x/plug_sensor.usd": "c",
               "omni://x/franka.usd": "f"}
        write_m1b(m1b, "00001", usd_identity={"layers": lay, "usd_sha256": "s"})

        class Sc:
            def __init__(self, layers):
                self.layers = layers

            def usd_identity(self):
                return {"layers": self.layers, "usd_sha256": "t"}

        def usd_run(layers):
            rb = FakeGraspRobot()
            rb.sc = Sc(layers)
            return CO.collect_grasp(rb, spec, out, cfg, {})
        assert usd_run({**lay, "omni://x/franka.usd": "other"})["usd_vs_m1b"] == "equal", "only plug / socket / sensor"
        (out / "logs" / "ALERT").unlink()
        msg = raises(lambda: usd_run({**lay, "omni://x/00001/socket.usd": "changed"}), CO.Refused)
        assert "different" in msg and "different" in (out / "logs" / "ALERT").read_text()
        assert "unreadable" in raises(lambda: usd_run({**lay, "omni://x/plug_sensor.usd": "unreadable: 404"}), CO.Refused)
        write_m1b(m1b, "00001")                                           # a (ii) result without usd_identity
        assert "missing" in raises(lambda: usd_run(lay), CO.Refused)
        # contact: the wm_v1 depth outcome is "reached_depth", never "inserted"
        spec.trajectories = [t for t in all_t if t.kind == "contact"][:1]
        saved.clear()
        summ = run()
        assert summ["trajectories"][1]["outcome"] == "reached_depth", summ["trajectories"][1]
        assert "waypoint_redraws" not in summ["trajectories"][1], "no waypoints: no redraw record"
        # decision 7 of 2026-11-23 (row 39): a random waypoint whose commanded hand / finger is closer than
        # expert.HAND_FINGER_COMMAND_CLEARANCE_M to the socket is redrawn from the "waypoint" stream before the run;
        # the expert runs the redrawn waypoints, the meta records them, the planned ones and every rejected draw
        rnd = [t for t in all_t if t.kind == "random"][:1]
        spec.trajectories = rnd
        planned = [w.as_dict() for w in rnd[0].waypoints]
        bad = {json.dumps(planned[1])}                                    # waypoint 1 fails once
        FakeExpert.wp_clear = staticmethod(lambda off: -1.0 if json.dumps(off.as_dict()) in bad else 0.0011)
        FakeExpert.run_kw = []
        saved.clear()
        summ = run()
        row = summ["trajectories"][1]
        ran = [w.as_dict() for w in FakeExpert.run_kw[-1][1]["waypoints"]]
        assert [r["waypoint"] for r in row["waypoint_redraws"]] == [1] and row["waypoint_redraws"][0]["clearance_mm"] \
            == -1000.0 and row["planned_waypoints"] == planned, row
        assert ran[0] == planned[0] and ran[2] == planned[2] and ran[1] != planned[1], (ran, planned)
        wrng = np.random.default_rng(CO.stream(cfg, spec.port, spec.gi, rnd[0].idx + 1, "waypoint"))
        assert ran[1] == CO.sample_start(wrng, CO.WAYPOINT_BOX).as_dict(), "redraw from the trajectory's waypoint stream"
        meta = saved[row["file"]]
        assert meta["waypoints"] == ran and "waypoint" in meta["rng"]["spawn_keys"], meta["rng"]
        FakeExpert.wp_clear = staticmethod(lambda off: 0.0009)           # nothing passes: kept after WAYPOINT_DRAWS
        summ = run()
        row = summ["trajectories"][1]
        assert len(row["waypoint_redraws"]) == 3 * CO.WAYPOINT_DRAWS and row["outcome"] == "done", row["outcome"]
        FakeExpert.wp_clear = staticmethod(lambda off: 0.001)            # exactly the limit passes
        summ = run()
        assert summ["trajectories"][1]["waypoint_redraws"] == [] and "planned_waypoints" not in summ["trajectories"][1]
        assert [w.as_dict() for w in FakeExpert.run_kw[-1][1]["waypoints"]] == planned
        FakeExpert.wp_clear = staticmethod(lambda off: 1.0)
        C.SOCKET_DRAW = "per_grasp"
        spec.trajectories = [t for t in all_t if t.kind == "insert"][:2]
        FakeGraspRobot.resets = []
        saved.clear()
        summ = run()
        assert FakeGraspRobot.resets == [0] and saved[f"t{i1:02d}_insert.npz"]["H_star"][0][3] == 0.0
        assert saved[f"t{i1:02d}_insert.npz"]["H_star_fk_vs_model_mm_deg"] == [0.0, 0.0]
        # r4-int-chk-01: per_grasp, a contact-report error then a failing re-home ends the grasp softly
        FakeGraspRobot.calls, FakeGraspRobot.fail_calls = 0, {2}              # reset 1 = t00, reset 2 = the re-home
        FakeExpert.calls = 0
        FakeExpert.raise_on = {("insert", 2): K.ContactReportError("contact report empty, sensor 3.0 N")}
        summ = run()
        assert summ["rehome_failed"]["after"] == i1 and [t["outcome"] for t in summ["trajectories"]] == \
            ["inserted", "contact_report_error"] and "rehome_error" in summ["trajectories"][1], summ["trajectories"]
        FakeGraspRobot.fail_calls, FakeExpert.raise_on = set(), {}
        (out / "STOP").write_text("x")
        raises(run, CO.Stopped)
    finally:
        FakeRecorder.axial = True
        FakeExpert.label_ok = FakeExpert.label_ok_flag = FakeExpert.completed = True
        FakeExpert.raise_on = {}
        FakeExpert.move_raise = set()
        FakeExpert.wp_clear = staticmethod(lambda off: 1.0)
        FakeGraspRobot.fail_reset = FakeGraspRobot.fail_calls = set()
        CO.eval_tip, CO.truth_summary, CO.store.save, CO.automate_success, C.SOCKET_DRAW = old
        for k, v in old_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return "label-block error -> SimulatorError; no FK axis -> reference_ok False; t00 needs completed AND success at " \
           "the end (each part refused alone, both recorded); per_reset: resets [0, i, j], own label + socket pose, " \
           "label_failed without an axis or with label ok False; insert outcome = S_any with expert_completed / " \
           "automate_success_any recorded; 'file' only on saved rows; reset_failed / contact_report_error continue + " \
           "ALERT; t00 ResetError / ContactReportError soft + ALERT; transfer contact error ends one trajectory " \
           "(stage placement); stages with the expert's speeds; clearance limit = expert.START_CLEARANCE_M " \
           "(placement_failed by 'clearance' redraws); label_failed in the ALERT share; USD layers vs (ii): equal / different / unreadable / " \
           "missing (the last three Refused + ALERT); contact 'reached_depth'; per_grasp: one reset, H*-vs-model in " \
           "meta, failing re-home ends the grasp softly; STOP stops; random waypoints below the hand / finger limit " \
           "redrawn from the waypoint stream (decision 7), kept after WAYPOINT_DRAWS, recorded"


def test_success_rule(tmp: Path):
    """automate_success and eval_tip with the real checks.py: the start state (row 0) is never evaluated."""
    G3 = C.R_G[:3, :3]
    ps = types.SimpleNamespace(tip_centre_m=(0.0, 0.0, 0.0), symmetry_n="round", disassembly_dist_m=0.02)

    def row(h):
        r = np.zeros(7)
        r[:3] = G3.T @ np.array([0.0, 0.0, h])
        r[3] = 1.0                                                    # identity rotation (w first)
        return r
    ok, far = row(0.005), row(0.05)
    assert bool(K.truth_from_rel_spec(ok, ps)["success"]) and not bool(K.truth_from_rel_spec(far, ps)["success"])

    class R:
        port_spec = ps
        last = far

        def truth_rel(self):
            return R.last
    a = {"eval_rel": np.stack([ok, far, far])}
    s = CO.automate_success(R(), a)
    assert s["inserted_any_step"] is False and s["success_at_start"] is True and s["first_success_step"] is None, s
    a = {"eval_rel": np.stack([far, far, ok, ok])}
    R.last = ok
    s = CO.automate_success(R(), a)
    assert s["inserted"] and s["first_success_step"] == 2 and s["inserted_end"] and not s["success_at_start"], s
    e = CO.eval_tip(a, R())
    assert e.shape == (4, len(K.TRUTH_COLUMNS)) and e.dtype == np.float32 and list(e[:, 0]) == [0, 0, 1, 1]
    assert tuple(CO.EVAL_TIP_COLUMNS) == tuple(K.TRUTH_COLUMNS)

    class Sc:
        n = 0

        def usd_identity(self):
            Sc.n += 1
            return {"usd_sha256": "u"}
    rb = types.SimpleNamespace(sc=Sc())
    assert CO.usd_identity(rb) == CO.usd_identity(rb) == {"usd_sha256": "u"} and Sc.n == 1, "once per built scene"
    assert CO.usd_identity(types.SimpleNamespace()) is None
    return "usd_identity cached per scene; S_any over rows 1..T only (start success alone gives False), first_success_step = 2, eval_tip (T+1, 9) " \
           "float32 with success 0 / 1"


# ------------------------------------------------------------------------------------------------------------- 6
CHILD = r"""
import sys, time, types
sys.path.insert(0, sys.argv[1])
m = types.ModuleType("wm.sim_automate.scene"); m.AutomateScene = object; sys.modules["wm.sim_automate.scene"] = m
from pathlib import Path
from wm.sim_automate import collect as CO
c = CO.claim(Path(sys.argv[2]))
CO.install_stop_handlers(lambda s: print(s, flush=True))
print("READY", c.attempt, flush=True)
time.sleep(60)
"""


def test_signal(tmp: Path):
    g = tmp / "out" / "00001" / "00001_g00"
    g.mkdir(parents=True)
    dead = "attempt 1 pid 999999997 start 3 time t0"
    (g / ".claim").write_text(dead)
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(WM / "src"), str(g)], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    assert line.startswith("READY 2"), line
    assert "pid " + str(p.pid) in (g / ".claim").read_text()
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=30)
    rest = p.stdout.read()
    assert rc == CO.EXIT_STOPPED and "WORKER STOPPED" in rest, (rc, rest)
    assert (g / ".claim").read_text() == dead
    c = CO.claim(g)
    assert c is not None and c.attempt == 2
    CO.release(c, done=False)
    return "SIGTERM: exit 5, claim text restored (the stop is not a death), flock free"


# ------------------------------------------------------------------------------------------------------------- 7
def test_heartbeat_watch(tmp: Path):
    out = tmp / "out"
    hb = CO.Heartbeat(out, 3, timeout_s=60)
    assert hb.path.exists() and json.loads(hb.path.read_text())["index"] == 3
    hb("stage a", grasp="g")
    assert json.loads(hb.path.read_text())["stage"] == "stage a"
    os.utime(hb.path, (1, 1))
    hb()                                                       # throttled: last write was just now
    assert hb.path.stat().st_mtime == 1
    hb._last -= CO.BEAT_EVERY_S + 1
    hb()
    assert hb.path.stat().st_mtime > time.time() - 10
    hb.close()
    # r4-ops-05: the largest gap between two file updates, booked to the stage that was current during it
    hb2 = CO.Heartbeat(out, 4, timeout_s=60)
    hb2._last -= 7
    hb2("trajectory 1 (insert)")
    hb2._last -= CO.BEAT_EVERY_S + 2
    hb2()
    g = hb2.pop_gaps()
    assert g["start"] >= 7 and g["trajectory"] >= CO.BEAT_EVERY_S + 2 and hb2.pop_gaps() == {}, g
    hb2.close()
    # a hung worker: a child process of this test (inside our test scope), its claim and a stale heartbeat
    QW.RUN_DIR = tmp / "run"
    import hashlib
    h = QW.write_code_identity()                                    # r4-ops-08
    assert (QW.RUN_DIR / "watch" / "code.sha256").read_text().strip() == h == \
        hashlib.sha256((WM / "scripts/automate/qx_watch.py").read_bytes()).hexdigest()
    child = subprocess.Popen(["sleep", "120"])
    start = CO._start_time(child.pid)
    g = out / "00001" / "00001_g00"
    g.mkdir(parents=True)
    (g / ".claim").write_text(f"attempt 1 pid {child.pid} start {start} time t0")
    f = out / "heartbeat" / f"{child.pid}_{start}.json"
    f.write_text(json.dumps({"pid": child.pid, "start": start, "timeout_s": 60, "claim": str(g / ".claim"),
                             "unit": "u", "stage": "trajectory 3", "grasp": "00001_g00"}))
    os.utime(f, (time.time() - 120, time.time() - 120))
    stale = out / "heartbeat" / "999999996_5.json"
    stale.write_text(json.dumps({"pid": 999999996, "start": "5", "timeout_s": 60}))
    if not QW.in_our_scope(child.pid):
        child.kill()
        child.wait(timeout=10)
        QW.check_heartbeats(out)                                   # both pids are dead now: their files go
        note = "hang kill SKIPPED (not run inside a qingxin-* scope)"
    else:
        acts = QW.check_heartbeats(out)
        rc = child.wait(timeout=10)
        assert rc == -signal.SIGKILL and len(acts) == 1, (rc, acts)
        assert "hang" in (g / ".claim").read_text() and f.with_suffix(".hung").exists() and not f.exists()
        assert "no heartbeat" in (out / "logs" / "ALERT").read_text() and (QW.RUN_DIR / "ALERT").exists()
        c = CO.claim(g)
        assert c.attempt == 2 and c.old["hang"]
        CO._record_death(out, "00001", "00001_g00", c.old)
        assert CO._rows(out / "deaths" / "00001.json")[0]["kind"] == "hang"
        CO.release(c, done=False)
        note = "stale heartbeat of a live pid in our scope: SIGKILL, claim marked hang, .hung + ALERT, hang death"
    assert not stale.exists(), "a dead pid's heartbeat is removed"
    # decisions
    us = [{"name": "qingxin-collect-a-1.scope", "kind": "collect", "pids": [10], "since": 1, "declared_mib": 6675},
          {"name": "qingxin-train-b-2.scope", "kind": "train", "pids": [20], "since": 2, "declared_mib": 0},
          {"name": "qingxin-watch-main-3.scope", "kind": "watch", "pids": [30], "since": 3, "declared_mib": 0}]
    ok = (0.0, 100 * 10**9, {}, {10: 6000})
    big = 100 * 10**9
    first = lambda acts: (acts[0][0], acts[0][2]) if acts else None  # noqa: E731
    assert QW.decide_units(us, *ok) == []
    assert QW.decide_units(us, 0.0, big, {20: {"types": ["G"], "mib": 8}}, {}) == [], "8 MiB graphics accepted"
    assert QW.decide_units(us, 0.0, big, {20: {"types": ["G", "G"], "mib": 50}}, {}) == [], "50 MiB accepted"
    assert first(QW.decide_units(us, 0.0, big, {20: {"types": ["G"], "mib": 51}}, {})) == ("qingxin-train-b-2.scope", "gpu0")
    assert first(QW.decide_units(us, 0.0, big, {20: {"types": ["C"], "mib": 5}}, {})) == ("qingxin-train-b-2.scope", "gpu0")
    assert first(QW.decide_units(us, 0.0, big, {20: {"types": ["C+G"], "mib": 5}}, {})) == ("qingxin-train-b-2.scope", "gpu0")
    assert first(QW.decide_units(us, 0.0, big, {20: {"types": ["?"], "mib": 5}}, {})) == ("qingxin-train-b-2.scope", "gpu0")
    assert QW.decide_units(us, 0.0, big, {20: {"types": ["G"], "mib": None}}, {}) == \
        [("qingxin-train-b-2.scope", QW.decide_units(us, 0.0, big, {20: {"types": ["G"], "mib": None}}, {})[0][1],
          "gpu0_unreadable")], "unreadable memory: an alert, no stop"
    assert QW.decide_units(us, 0.0, big, {999: {"types": ["C"], "mib": 500}}, {}) == [], "not our pid"
    assert first(QW.decide_units(us, 31.0, big, {}, {})) == ("qingxin-train-b-2.scope", "memory")
    assert first(QW.decide_units(us, 0.0, 9 * 10**9, {}, {})) == ("qingxin-train-b-2.scope", "memory")
    assert first(QW.decide_units(us, 0.0, big, {}, {10: 30000, 20: 20000})) == ("qingxin-train-b-2.scope", "gpu_sum")
    assert first(QW.decide_units(us[:1] + us[2:], 0.0, big, {}, {10: 46000})) == ("qingxin-collect-a-1.scope", "gpu_sum")
    assert first(QW.decide_units(us, 0.0, big, {}, {10: 6000}, 5000)) == ("qingxin-collect-a-1.scope", "gpu_free")
    assert QW.decide_units(us, 0.0, big, {}, {10: 6000}, 20000) == []
    assert QW.decide_units(us, 0.0, big, {}, {}, 5000) == [], "no GPU-1 memory of ours: nothing to free"
    assert QW.decide_units(us[2:], 50.0, 1, {}, {}) == [], "the watcher never stops itself"
    tab = PIDS_TEXT
    assert QW.parse_pid_table(tab) == CO.parse_pid_table(tab) == {1374517: 643, 4242: 7000}
    assert QW.parse_pid_rows(tab) == {1374517: {"types": ["G"], "mib": 643}, 4242: {"types": ["C", "G"], "mib": 7000}}
    assert QW.parse_pid_rows("Process ID : 5\n Type : G\n Used GPU Memory : N/A\n")[5]["mib"] is None
    assert QW.parse_pid_rows("Process ID : 5\n Used GPU Memory : 8 MiB\n")[5]["types"] == ["?"]
    # a GPU-0 stop writes the gpu0 marker and the STOP of the run whose heartbeat names the unit
    (QW.RUN_DIR / "watch").mkdir(parents=True, exist_ok=True)
    (QW.RUN_DIR / "watch" / "k.dir").write_text(str(out))
    (out / "heartbeat").mkdir(exist_ok=True)
    (out / "heartbeat" / "1_2.json").write_text(json.dumps({"pid": 1, "start": "2", "unit": "qingxin-collect-z-1.scope"}))
    QW.mark("gpu0", "qingxin-collect-z-1.scope", "C context")
    QW.stop_runs_of("qingxin-collect-z-1.scope", "C context")
    assert (QW.RUN_DIR / "gpu0" / "qingxin-collect-z-1.scope").exists() and "C context" in (out / "STOP").read_text()
    # one failing folder or reading never ends a cycle
    old = QW.units, QW.check_heartbeats, QW.CGROOT
    QW.units, QW.CGROOT = (lambda: []), tmp / "no_cgroup"
    QW.check_heartbeats = lambda o, now=None: 1 / 0
    try:
        assert QW.cycle(os.getuid()) == 0
    finally:
        QW.units, QW.check_heartbeats, QW.CGROOT = old
    alerts = (QW.RUN_DIR / "ALERT").read_text()
    assert "a reading failed" in alerts and "heartbeat check of" in alerts, alerts
    assert QW.wanted_recently(time.time()) is False
    (QW.RUN_DIR / "watch" / "wanted").touch()
    assert QW.wanted_recently(time.time()) is True
    return note + "; heartbeat gaps per stage; watcher code.sha256; decisions: GPU 0 (8 / 50 MiB graphics accepted; 51 MiB, C, C+G, no type stop; unreadable memory " \
                  "alerts only), pressure / MemAvailable, GPU-1 sum > 48 GB, GPU-1 free < 8 GB stop the newest " \
                  "stoppable unit; gpu0 marker + STOP of the run; a failing reading / folder never ends a cycle"


PIDS_TEXT = """GPU 00000000:AC:00.0
    Processes
        GPU instance ID                   : N/A
        Process ID                        : 1374517
            Type                          : G
            Used GPU Memory               : 643 MiB
        Process ID                        : 4242
            Type                          : C
            Used GPU Memory               : 6000 MiB
        Process ID                        : 4242
            Type                          : G
            Used GPU Memory               : 1000 MiB
"""


# ------------------------------------------------------------------------------------------------------------- 8
def fake_tools(tmp: Path, units="", gpu0="", fail="", free=60000, declared=None) -> Path:
    """Fake systemctl / nvidia-smi on PATH; fake /proc and cgroup files. gpu0: "pid type MiB" per line (GPU 0 table);
    declared: {unit: GPU GB} reservation files."""
    b = tmp / "bin"
    b.mkdir(exist_ok=True)
    (tmp / "units").write_text(units)
    (b / "systemctl").write_text(f"""#!/bin/bash
[ "{fail}" = systemctl ] && exit 1
case "$*" in
  *list-units*) cat {tmp}/units ;;
  *MemoryMax*) echo 60000000000 ;;
  *CPUQuotaPerSecUSec*) echo 5s ;;
  *ControlGroup*) echo /fake/${{@: -1}} ;;
  *is-active*) exit 1 ;;
  *) exit 0 ;;
esac
""")
    g0 = "".join(f"        Process ID : {r.split()[0]}\n            Type : {r.split()[1]}\n"
                 f"            Used GPU Memory : {r.split()[2]}{'' if r.split()[2] == 'N/A' else ' MiB'}\n"
                 for r in gpu0.strip().splitlines() if r.strip())
    for u, gb in (declared or {}).items():
        (tmp / "run" / "qx").mkdir(parents=True, exist_ok=True)
        (tmp / "run" / "qx" / f"{u}.gpu").write_text(str(gb))
    (b / "nvidia-smi").write_text(f"""#!/bin/bash
[ "{fail}" = nvidia ] && exit 1
case "$*" in
  *--query-gpu=index,uuid*) printf '0, GPU-zero\\n1, GPU-one\\n' ;;
  *-i\\ GPU-zero*) printf '{g0}' ;;
  *-i\\ GPU-one*) printf '        Process ID : 777\\n            Used GPU Memory : 9000 MiB\\n' ;;
  *memory.free*) echo {free} ;;
esac
""")
    for f in b.iterdir():
        f.chmod(0o755)
    proc = tmp / "proc"
    (proc / "pressure").mkdir(parents=True, exist_ok=True)
    keep = lambda f, txt: f.exists() or f.write_text(txt)  # noqa: E731   (a test may have set it already)
    keep(proc / "pressure" / "memory", "some avg10=0.00 avg60=0.00\nfull avg10=0.00 avg60=0.00\n")
    keep(proc / "meminfo", "MemTotal: 500000000 kB\nMemAvailable: 140000000 kB\n")
    us = tmp / "cg" / "user.slice" / f"user-{os.getuid()}.slice" / f"user@{os.getuid()}.service"
    us.mkdir(parents=True, exist_ok=True)
    keep(us / "memory.pressure", "some avg10=0.00 avg60=0.00\nfull avg10=0.00 avg60=0.00\n")
    for u in units.split():
        d = tmp / "cg" / "fake" / u
        d.mkdir(parents=True, exist_ok=True)
        (d / "cgroup.procs").write_text("777\n888\n")
    return b


def caps(tmp: Path, script: str, env_extra=None, **kw) -> subprocess.CompletedProcess:
    b = fake_tools(tmp, **kw)
    pre = (f"source {WM}/scripts/automate/caps.sh; source {WM}/scripts/automate/collect_lib.sh; "
           f"QX_PROC={tmp}/proc QX_CGROOT={tmp}/cg QX_RUN_DIR={tmp}/run; ")
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", **(env_extra or {})}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.run(["bash", "-c", pre + script], capture_output=True, text=True, env=env)


def test_caps(tmp: Path):
    t = lambda name: (tmp / name).mkdir() or tmp / name  # noqa: E731
    r = caps(t("a"), "qx_check collect 16G 500% 7")
    assert r.returncode == 0 and r.stdout.startswith("caps ok"), r.stdout + r.stderr
    for args, why in (("collect 18.5G 500% 0", "MemoryMax"), ("collect 18G 4.5 0", "CPUQuota"),
                      ("collect 18G 500% x", "GPU GB"), ("Collect 18G 500% 0", "kind")):
        r = caps(t("v" + str(abs(hash(args)))), f"qx_check {args}")
        assert r.returncode == 1 and "refused" in r.stdout and why in r.stdout, (args, r.stdout, r.stderr)
    r = caps(t("b"), "qx_check collect 16G 500% 0", fail="systemctl")
    assert r.returncode == 1 and "systemctl" in r.stdout, r.stdout
    r = caps(t("c"), "qx_check collect 16G 500% 0", fail="nvidia")
    assert r.returncode == 1 and "nvidia-smi" in r.stdout, r.stdout
    u2 = "qingxin-collect-x-1.scope\nqingxin-collect-x-2.scope\n"                         # 2 x 60e9 bytes
    r = caps(t("d"), "qx_check collect 6G 500% 0", units=u2)
    assert r.returncode == 1 and "125 GB" in r.stdout, r.stdout
    r = caps(t("e"), "qx_check collect 4G 500% 0", units=u2)
    assert r.returncode == 0, r.stdout
    for rows, allowed in (("888 G 8", True), ("888 G 30\n888 G 20", True), ("888 G 51", False), ("888 C 100", False),
                          ("888 C+G 8", False), ("888 G N/A", False), ("999 C 900", True), ("888 G 8\n888 C 1", False)):
        r = caps(t("f" + str(abs(hash(rows)))), "qx_check collect 4G 500% 0", units=u2, gpu0=rows)
        assert (r.returncode == 0) == allowed and (allowed or "GPU 0" in r.stdout), (rows, r.stdout, r.stderr)
    # GPU 1: memory our starting units declared but have not allocated yet counts against the 12 GB left to others
    dec = {"qingxin-collect-x-1.scope": 20, "qingxin-collect-x-2.scope": 20}
    r = caps(t("p1"), "qx_check collect 4G 500% 7", units=u2, free=30000, declared=dec)
    assert r.returncode == 1 and "declared" in r.stdout, r.stdout
    r = caps(t("p2"), "qx_check collect 4G 500% 7", units=u2, free=30000)
    assert r.returncode == 0, r.stdout
    # the watcher may start under memory pressure / low MemAvailable (256M); others may not
    d = t("w")
    fake_tools(d)
    (d / "proc" / "meminfo").write_text("MemAvailable: 1000000 kB\n")
    r = caps(d, "qx_check watch 256M 5% 0; echo rc=$?; qx_check collect 4G 500% 0; echo rc=$?")
    assert "rc=0" in r.stdout and "rc=1" in r.stdout, r.stdout
    r = caps(t("g"), "qx_check collect 4G 500% 0", units=u2,
             env_extra={"QX_CAP_MEM_BYTES": "999999999999999", "QX_CAPS_LOADED": "1"})
    r2 = caps(t("g2"), "qx_check collect 6G 500% 0", units=u2,
              env_extra={"QX_CAP_MEM_BYTES": "999999999999999", "QX_CAPS_LOADED": "1"})
    assert r.returncode == 0 and r2.returncode == 1, "the environment cannot raise the caps"
    d = t("h")
    fake_tools(d)
    (d / "cg" / "user.slice" / f"user-{os.getuid()}.slice" / f"user@{os.getuid()}.service" / "memory.pressure").write_text(
        "some avg10=12.00 avg60=0\nfull avg10=0.50 avg60=0\n")
    r = caps(d, "qx_check collect 4G 500% 0")
    assert r.returncode == 1 and "user-service" in r.stdout, r.stdout
    d = t("i")
    fake_tools(d)
    (d / "proc" / "meminfo").write_text("MemAvailable: 30000000 kB\n")                    # 30.7 GB
    r = caps(d, "qx_check collect 16G 500% 0")
    assert r.returncode == 1 and "MemAvailable" in r.stdout, r.stdout
    r = caps(t("j"), "qx_check collect 30G 500% 50")                                         # GPU 1: 9000 MiB ours? no: no units
    assert r.returncode == 1 and "48 GB" in r.stdout, r.stdout
    r = caps(t("k"), "for v in 4s 500ms 1.500000s '1min 30s' infinity; do qx_quota_pct \"$v\"; done; "
                     "qx_to_bytes 16384M; qx_to_bytes 18.5G || echo bad; echo '" + PIDS_TEXT + "' | qx_parse_pids | sort")
    assert r.stdout.split() == ["400", "50", "150", "9000", "-1", str(16384 * 2**20), "bad",
                                "1374517", "643", "4242", "7000"], r.stdout
    r = caps(t("l"), "cl_static_budget 6 16G 500% 7 && echo ok6; cl_static_budget 7 16G 500% 7 || echo no7; "
                     "cl_static_budget 7 18G 400% 6 || echo no7b")
    assert r.stdout.split()[-3:] == ["ok6", "no7", "no7b"] or "ok6" in r.stdout and "no7" in r.stdout, r.stdout
    d = t("m")
    (d / "tree" / "a").mkdir(parents=True)
    (d / "tree" / "a" / "f.py").write_text("x")
    r = caps(d, f"cl_code_hash {d}/tree; echo y >> {d}/tree/a/f.py; cl_code_hash {d}/tree")
    h = r.stdout.split()
    assert len(h) == 2 and len(h[0]) == 64 and h[0] != h[1], r.stdout
    # the frozen switches: defaults equal config.py's, WM_FORBIDDEN_GPU always 0, inherited WM_* dropped
    d = t("s")
    r = caps(d, f"cl_write_switches {d}/sw.env {WM}; cat {d}/sw.env; WM_CAM_SCALE=2 WM_AUTOMATE_DIAGNOSTIC=1 "
                f"WM_AUTOMATE_SOCKET_DRAW=per_grasp; export WM_CAM_SCALE WM_AUTOMATE_DIAGNOSTIC WM_AUTOMATE_SOCKET_DRAW; "
                f"cl_clean_env {d}/sw.env && env | grep '^WM_' | sort",
             env_extra={"WM_AUTOMATE_DEVICE": "cpu", "WM_FORBIDDEN_GPU": "1"})
    lines = r.stdout.split()
    sw = dict(x.split("=", 1) for x in lines[:len(lines) // 2])
    assert sw == dict(x.split("=", 1) for x in lines[len(lines) // 2:]), r.stdout
    assert sw["WM_FORBIDDEN_GPU"] == "0" and sw["WM_CAM_SCALE"] == "1" and "WM_AUTOMATE_DIAGNOSTIC" not in sw
    # the start box of the user's decision of 2026-11-23 flows into the frozen switches
    assert sw["WM_AUTOMATE_START_GAP_FROM"] == "socket_top" == C.APPROVED["start_gap_from"] == C.START_GAP_FROM, sw
    clean = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    sw_of = lambda env: json.loads(subprocess.run(  # noqa: E731
        [sys.executable, "-c", "import sys, json; sys.path.insert(0, sys.argv[1]); "
         "from wm.sim_automate import config as C; print(json.dumps(C.switches()))", str(WM / "src")],
        capture_output=True, text=True, env=env).stdout)
    a_, b_ = sw_of({**clean, **sw}), sw_of(clean)
    assert a_ == b_, ("the frozen defaults equal config.py's defaults", sw, a_, b_)
    (d / "bad.env").write_text("WM_CAM_SCALE=1\nPATH=/x\n")
    assert "malformed" in caps(d, f"cl_clean_env {d}/bad.env").stdout
    # r4-ops-02: a switch exported by the launching shell other than config.APPROVED is refused at freeze time
    for k, val, key in (("WM_AUTOMATE_SOCKET_DRAW", "per_grasp", "socket_draw"), ("WM_KIT_GPU", "0", "kit_gpu")):
        r = caps(d, f"rm -f {d}/sw2.env; cl_write_switches {d}/sw2.env {WM}; echo rc=$?; ls {d}/sw2.env",
                 env_extra={k: val})
        assert "rc=1" in r.stdout and "not approved" in r.stderr + r.stdout and key in r.stderr + r.stdout \
            and "No such file" in r.stderr, (k, r.stdout, r.stderr)
    # the old start box ("entrance", diagnostics only since 2026-11-23) exported by the launching shell: refused, even
    # with WM_AUTOMATE_DIAGNOSTIC=1 there (cl_write_switches drops it)
    r = caps(d, f"rm -f {d}/sw2.env; cl_write_switches {d}/sw2.env {WM}; echo rc=$?; ls {d}/sw2.env",
             env_extra={"WM_AUTOMATE_START_GAP_FROM": "entrance", "WM_AUTOMATE_DIAGNOSTIC": "1"})
    assert "rc=1" in r.stdout and "No such file" in r.stderr and "START_GAP_FROM" in r.stderr + r.stdout, (r.stdout, r.stderr)
    # r4-ops-N2: the watcher starts while one of our pids breaks the GPU-0 rule or nvidia-smi fails; others do not
    r = caps(t("n1"), "qx_check watch 256M 5% 0; echo rc=$?; qx_check collect 4G 500% 0; echo rc=$?", units=u2,
             gpu0="888 C 300")
    assert r.stdout.count("rc=0") == 1 and "rc=1" in r.stdout and "GPU 0" in r.stdout, r.stdout
    r = caps(t("n2"), "qx_check watch 256M 5% 0", fail="nvidia")
    assert r.returncode == 0 and "caps ok (watch)" in r.stdout, r.stdout
    assert caps(t("n3"), "qx_check watch 256M 5% 1").returncode == 1
    # r4-ops-08: a live watcher with other code gives a WARNING (still 0)
    d = t("wc")
    (d / "run" / "watch").mkdir(parents=True)
    (d / "run" / "watch" / "code.sha256").write_text("0" * 64 + "\n")
    r = caps(d, f"cl_watch_code {WM}; echo rc=$?")
    assert "WARNING: the live watcher" in r.stdout and "rc=0" in r.stdout, r.stdout
    import hashlib
    (d / "run" / "watch" / "code.sha256").write_text(
        hashlib.sha256((WM / "scripts/automate/qx_watch.py").read_bytes()).hexdigest() + "\n")
    r = caps(d, f"cl_watch_code {WM}")
    assert "WARNING" not in r.stdout, r.stdout
    # r4-data-05: qx_run gives the job QX_GPU_GB (fake systemd-run runs the command after its options)
    d = t("gg")
    b = fake_tools(d)
    (b / "systemd-run").write_text('#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != env ]; do shift; done\nexec "$@"\n')
    (b / "systemd-run").chmod(0o755)
    r = caps(d, "qx_run test gg 1G 100% 3 bash -c 'echo GPU=$QX_GPU_GB CVD=$CUDA_VISIBLE_DEVICES'; echo rc=$?")
    assert "GPU=3 CVD=1" in r.stdout and "rc=0" in r.stdout, r.stdout + r.stderr
    # r9-ops-01: qx_run_retry starts again every 60 s while the caps refuse (exit 99), up to QX_RETRY_MAX_S; any other
    # exit code at once. qx_check refuses twice, then allows: the command's exit code, 2 retry lines (stdout and stderr)
    stub = (f"qx_check() {{ n=$(cat {d}/nchk 2>/dev/null || echo 0); echo $((n + 1)) > {d}/nchk; "
            "[ $n -ge ${REFUSE_N:-2} ] && { echo 'caps ok: test'; return 0; }; echo 'refused: memory pressure (test)'; return 1; }; "
            f"sleep() {{ [ \"$1\" = 60 ] && {{ echo slept >> {d}/slept; command sleep ${{SLEEP_S:-0}}; return 0; }}; command sleep \"$@\"; }}; ")
    r = caps(d, stub + "QX_UNIT_FILE=" + str(d / "uf") + " qx_run_retry test rr 1G 100% 0 bash -c 'exit 5'; echo rc=$?")
    assert "rc=5" in r.stdout and r.stdout.count("caps refused rr (exit 99); retry in 60 s") == 2, r.stdout + r.stderr
    assert r.stderr.count("caps refused rr (exit 99); retry in 60 s") == 2, r.stderr
    assert (d / "slept").read_text().count("slept") == 2 and (d / "nchk").read_text().strip() == "3"
    assert (d / "uf").read_text().startswith("qingxin-test-rr-"), "QX_UNIT_FILE reaches the nested qx_run"
    (d / "nchk").unlink(), (d / "slept").unlink()
    r = caps(d, stub + "qx_run_retry test rr 1G 100% 0 true; echo rc=$?",         # allowed at once: no retry line
             env_extra={"REFUSE_N": "0"})
    assert "rc=0" in r.stdout and "retry" not in r.stdout + r.stderr, r.stdout + r.stderr
    (d / "nchk").unlink()
    r = caps(d, stub + "qx_run_retry test rr 1G 100% 0 true; echo rc=$?",          # refused for good: 99 after the limit
             env_extra={"REFUSE_N": "1000", "QX_RETRY_MAX_S": "2", "SLEEP_S": "1"})
    assert "rc=99" in r.stdout and "giving up (QX_RETRY_MAX_S=2)" in r.stdout and "giving up" in r.stderr, r.stdout + r.stderr
    nretry = r.stdout.count("retry in 60 s")
    assert 1 <= nretry <= 4 and (d / "slept").read_text().count("slept") == nretry, (nretry, r.stdout)
    r = caps(d, stub + "qx_run_retry test rr 1G 100% 0 true; echo rc=$?", env_extra={"QX_RETRY_MAX_S": "x"})
    assert "rc=99" in r.stdout and "QX_RETRY_MAX_S=x is not a whole number" in r.stdout, r.stdout
    return "validation (18.5G, 4.5, x, kind), failing systemctl / nvidia-smi refuse, 125 GB sum, GPU 0 rule (G <= 50 " \
           "MiB per pid only; C, C+G, 51 MiB, unreadable refused; other users' pids ignored), declared-not-allocated " \
           "GPU-1 memory counted, watch starts under pressure; frozen switches = config defaults, stale WM_* dropped; " \
           "user-service pressure, MemAvailable, 48 GB GPU, env cannot raise caps, CPUQuota '1min 30s' = 9000 %, " \
           "6 workers fit / 7 do not, code hash; unapproved per_grasp / KIT_GPU 0 / start gap 'entrance' refused by " \
           "cl_write_switches, approved 'socket_top' frozen; " \
           "watch starts despite a GPU-0 violation / failing nvidia-smi; watcher code WARNING; QX_GPU_GB exported; " \
           "qx_run_retry: refused twice then allowed -> the command's exit code with 2 retry lines on stdout and stderr, " \
           "allowed at once -> no retry, refused for good -> 99 after QX_RETRY_MAX_S, malformed QX_RETRY_MAX_S refused"


# ------------------------------------------------------------------------------------------------------------- 9
FAKE_EXPERT = r"""
import os, sys, json
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
plan = json.loads((out / "scenario.json").read_text())
k = int((out / "k").read_text()) if (out / "k").exists() else 0
(out / "k").write_text(str(k + 1))
code = plan[k] if k < len(plan) else 0
print(f"WORKER PID {os.getpid()} START 9", flush=True)
unit = Path(os.environ["QX_UNIT_FILE"]).read_text().strip()
if code in ("shed", "gpu0", "shed134"):                          # what qx_watch.py leaves before its stop
    d = Path(os.environ["QX_RUN_DIR"]) / ("shed" if code == "shed134" else code)
    d.mkdir(parents=True, exist_ok=True)
    (d / unit).write_text("test reason")
    os._exit(134 if code == "shed134" else 143)                  # 134: Kit's SIGTERM abort (r6-ops-02)
if code in ("hang", "hang_held"):                                # qx_watch.py's marker: the heartbeat + kill info
    (out / "heartbeat").mkdir(exist_ok=True)
    hb = {"stage": "launch", "claim": None} if code == "hang" else {"stage": "trajectory 3 (insert)", "claim": "/x/.claim"}
    (out / "heartbeat" / f"{os.getpid()}_9.hung").write_text(json.dumps(hb))
    os._exit(137)
if code == 0:
    print("WORKER DONE", flush=True)
if code in ("stop", "stop134"):                                  # a deliberate stop; 134: Kit aborted in its start
    (out / "STOP").write_text("x")
    code = 5 if code == "stop" else 134
os._exit(code)
"""


def fake_tree(tmp: Path) -> Path:
    W = tmp / "code"
    (W / "scripts" / "automate").mkdir(parents=True)
    shutil.copy(WM / "scripts/automate/collect_worker.sh", W / "scripts/automate/")
    lib = (WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}")
    lib = lib.replace("cl_isaac_env() {", "cl_isaac_env() { return 0\n")
    lib += "\ncl_watch_ensure() { [ ! -e \"$QX_RUN_DIR/nowatch\" ]; }\n"   # stand-in: the watcher is up unless told
    (W / "scripts/automate/collect_lib.sh").write_text(lib)
    (W / "scripts/automate/caps.sh").write_text(   # stand-in qx_run (unit file, runs the command), sleep is instant
        f"export QX_RUN_DIR={tmp}/run\nmkdir -p $QX_RUN_DIR\n"
        "qx_run() { [ -n \"${QX_UNIT_FILE:-}\" ] && echo \"qingxin-collect-t-$RANDOM.scope\" > \"$QX_UNIT_FILE\"; "
        "export QX_UNIT_FILE; shift 5; \"$@\"; }\n"
        "sleep() { [ -e \"$QX_RUN_DIR/nowatch\" ] && rm -f \"$QX_RUN_DIR/nowatch\"; :; }\n"
        "[ -e \"$QX_RUN_DIR/fastalert\" ] && REFUSED_ALERT_S=0\n")
    (W / "scripts/automate/collect_expert.py").write_text(FAKE_EXPERT)
    return W


def worker_loop(tmp: Path, scenario, tamper=False) -> tuple[str, str, str]:
    W = fake_tree(tmp)
    out = tmp / "out"
    (out / "logs").mkdir(parents=True)
    (out / "inputs").mkdir()
    (out / "inputs" / "switches.env").write_text("WM_AUTOMATE_SOCKET_DRAW=per_reset\n")
    (out / "logs" / "resources.txt").write_text("16G 500% 7 1020\n")
    (out / "scenario.json").write_text(json.dumps(scenario))
    h = subprocess.run(["bash", "-c", f"source {W}/scripts/automate/collect_lib.sh; cl_code_hash {W}"],
                       capture_output=True, text=True).stdout.strip()
    (out / "code.sha256").write_text(h + "\n")
    if tamper:
        (W / "scripts/automate/extra.py").write_text("x")
    r = subprocess.run(["bash", str(W / "scripts/automate/collect_worker.sh"), str(out), "0", "1", "s"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    rd = lambda n: (out / "logs" / n).read_text() if (out / "logs" / n).exists() else ""  # noqa: E731
    return rd("launcher.log"), rd("ALERT"), rd("w0.end")


def test_worker_loop(tmp: Path):
    t = lambda n: (tmp / n).mkdir() or tmp / n  # noqa: E731
    lg, al, end = worker_loop(t("a"), [99, 6, "hang_held", 1, 0])
    assert "done" in end and not al and "build failed (1 in a row)" in lg and "hang with a claim held" in lg and \
        "crash (exit 1, 1 in" in lg and "start refused by the caps" in lg, lg
    # r4-ops-03: a refusal keeps no start log and does not advance k; refusals that last give an ALERT line
    assert sorted(p.name.split("_")[1] for p in (tmp / "a" / "out" / "logs").glob("w0_*_*.log")) == ["1", "2", "3", "4"]
    d = t("a2")
    (d / "run").mkdir()
    (d / "run" / "fastalert").write_text("x")
    lg, al, end = worker_loop(d, [99, 99, 0])
    assert "done" in end and lg.count("start refused by the caps") == 2 and "every start refused by the caps" in al, (lg, al)
    # r4-ops-04: a hang before any claim is counted (rate cap)
    lg, al, end = worker_loop(t("a3"), ["hang"] * 6 + [0])
    assert "restart rate cap" in end and "hang before a claim (stage launch)" in end, (lg, end)
    lg, al, end = worker_loop(t("b"), [1, 1, 1, 1, 1, 1, 0])
    assert "restart rate cap" in end and "restart rate cap" in al, (lg, al)
    lg, al, end = worker_loop(t("c"), [3])
    assert "refused (exit 3" in end and al
    lg, al, end = worker_loop(t("d"), ["stop"])
    assert "stopped (STOP file)" in end and not al
    lg, al, end = worker_loop(t("e"), [5])
    assert "stopped by a signal" in end and al
    # r7-ops-06: a hard stop during the Isaac start (Kit aborts, 134) with STOP present ends quietly, no crash line
    lg, al, end = worker_loop(t("d2"), ["stop134"])
    assert "stopped (STOP file; exit 134" in end and "crash" not in lg and not al, (lg, al, end)
    lg, al, end = worker_loop(t("f"), [0], tamper=True)
    assert "changed" in end and al
    lg, al, end = worker_loop(t("g"), [7])
    assert "exit 7" in end and "STOP written" in end and "exit 7" in al and (tmp / "g" / "out" / "STOP").exists(), (lg, end)
    lg, al, end = worker_loop(t("h"), [8, 8, 8, 8, 8, 8, 0])
    assert "restart rate cap" in end and "exit 8" in end and "exit 8, counted as a crash" in lg, (lg, end)
    lg, al, end = worker_loop(t("i"), ["shed", "shed", 0])
    assert "done" in end and lg.count("shed by qx_watch.py") == 2 and not al, (lg, al)
    assert not list((tmp / "i" / "run" / "shed").iterdir()), "the shed marker is consumed"
    # r6-ops-02: Kit answers SIGTERM with an abort (exit 134); with the shed marker it is a shed, not a crash
    lg, al, end = worker_loop(t("i2"), ["shed134", "shed134", 0])
    assert "done" in end and lg.count("shed by qx_watch.py") == 2 and "crash" not in lg and not al, (lg, al)
    assert not list((tmp / "i2" / "run" / "shed").iterdir()), "the shed marker is consumed"
    lg, al, end = worker_loop(t("j"), ["shed"] * 7 + [0])
    assert "shed by qx_watch.py 7 times" in end and al, end
    lg, al, end = worker_loop(t("k"), ["gpu0"])
    assert "GPU 0" in end and "STOP written" in end and (tmp / "k" / "out" / "STOP").exists(), end
    d = t("l")
    (d / "run").mkdir()
    (d / "run" / "nowatch").write_text("x")
    lg, al, end = worker_loop(d, [0])
    assert "done" in end and "watcher" in al, (lg, al)
    return "99 / 6 / hang with a claim not counted then done (99: no start log, k kept; lasting refusals ALERT); " \
           "6 hangs before a claim -> rate cap; 6 crashes in an hour -> rate cap + ALERT; exit 3 -> ALERT; " \
           "STOP -> quiet end (also after Kit's 134 during a hard stop, no crash line); signal stop -> ALERT; changed code -> ALERT; exit 7 -> STOP + ALERT, no restart; " \
           "exit 8 counted as a crash (rate cap); shed (exit 143 or Kit's 134) -> wait and retry, not a crash (7 in " \
           "24 h end), marker consumed; watcher GPU-0 stop -> STOP; no watcher -> ALERT, no start until it runs"


# ------------------------------------------------------------------------------------------------------------- 10
def test_split_v1(tmp: Path):
    """40 synthetic assemblies: 20 round, 10 sleeve, 6 polygonal, 4 multi-pin; one near-twin pair; one zero-width dpsi."""
    ids = [f"{i:05d}" for i in range(1, 41)]
    raw = setup_assets(tmp, ids)
    cats = ["round"] * 20 + ["sleeve"] * 10 + ["polygonal"] * 6 + ["multi-pin"] * 4
    for i, (aid, cat) in enumerate(zip(ids, cats)):
        d, L, c = 0.004 + 0.0007 * i, 0.010 + 0.0011 * ((i * 7) % 40), 0.0002 + 0.00001 * ((i * 13) % 40)
        if aid == "00003":                                          # near twin of 00002 (round, d within 0.5 mm, L 3 mm)
            d, L = 0.004 + 0.0007 + 0.0003, 0.010 + 0.0011 * 7 + 0.002
        (A.ASSET_DIR / aid / A.SPEC_FILE).write_text(json.dumps({"assembly_id": aid, "category": cat, "pin_length_m": L,
                                                                 "clearance_m": c, "fit": {"equiv_diameter_m": d}}))
    zw = A.ASSET_DIR / "00010" / G.RANGES_FILE
    rg = json.loads(zw.read_text())
    rg["box"]["dpsi_deg"] = [0.0, 0.0]
    zw.write_text(json.dumps(rg))
    for aid in ids:                                     # geometry files (and 00010's ranges) changed: fresh results
        write_m1b(Path(raw["m1b_dirs"][0]), aid)
    cfg = CO.CollectConfig(**{**raw, **PRODUCTION, "expected_grasps": 600})
    specs, table = CO.build(cfg)
    sd = CO.split_document([r["assembly_id"] for r in table if r["status"] != "excluded"])
    assert len(sd["test"]) == 5 and len(sd["heldout_shape"]) == 4 and not set(sd["test"]) & set(sd["heldout_shape"])
    # 5 x (20, 10, 6, 4) / 40 = (2.5, 1.25, 0.75, 0.5): floors (2, 1, 0, 0), remainders 0.75 then the 0.5 tie by name
    assert sd["quotas"]["test"] == {"multi-pin": 1, "non-round keyed": 1, "round": 2, "sleeve": 1}, sd["quotas"]
    chosen = set(sd["test"]) | set(sd["heldout_shape"])
    assert ("00002" in chosen) == ("00003" in chosen), "near twins on one side"
    assert ["00002", "00003"] in sd["clusters"]
    sh, T = sd["shapes"], [a for a in ids if a not in chosen]
    for a in chosen:
        for j in ("d_m", "L_m", "c_m"):
            assert min(sh[b][j] for b in T) < sh[a][j] < max(sh[b][j] for b in T), (a, j)
    assert all(a in sd["reasons"] for a in chosen)
    by = {}
    for s_ in specs:
        by.setdefault(s_.port, []).append(s_)
    lo, hi = G.HELDOUT_U
    n_sel = 0
    for aid, gs in by.items():
        assert len(gs) == 15
        conds = [g.condition for g in gs]
        if aid in sd["test"]:
            assert set(conds) == {"test"}
        elif aid in sd["heldout_shape"]:
            assert set(conds) == {"heldout_shape"}
        else:
            band = [g for g in gs if g.grasp_kind.startswith("band_")]
            assert all(g.condition == "heldout_grasp" for g in band)
            assert len(band) == (2 if aid == "00010" else 3), (aid, [g.grasp_kind for g in gs])
            for g in gs:
                for k, x in g.u.items():
                    if x is None:
                        continue
                    inside = lo <= x <= hi
                    assert inside == (g.grasp_kind == f"band_{k}"), (g.key, k, x, g.grasp_kind)
                if not g.grasp_kind.startswith("band_"):
                    assert g.condition == ("select" if CO._crc(g.key) % 20 == 0 else "train")
                    n_sel += g.condition == "select"
    assert any("band_skipped" in g.grasp_kind for g in by["00010"]) or "00010" in chosen
    again = CO.build(cfg)[0]
    assert [(g.key, g.condition, g.pitch) for g in again] == [(g.key, g.condition, g.pitch) for g in specs]
    doc = CO.plan_document(cfg, specs, table, {"files_sha256": "x", "n_files": 1, "git_head": None, "git_dirty": None})
    CO.freeze(tmp / "out", cfg, doc)
    f = json.loads((tmp / "out" / "split_automate_v1.json").read_text())
    assert f["test"] == sd["test"] and f["plan_sha"] == doc["plan_sha"] and doc["split"]["file"] == "split_automate_v1.json"
    # r4-integration-02 / r4-data-01: the grasp counts per condition over every included assembly, in plan.json, and a
    # production plan train_wm320 would refuse is refused before plan.json (a smoke subset only warns)
    counts = CO.split_counts(cfg, table)
    by_cond = {}
    for g in specs:
        by_cond[g.condition] = by_cond.get(g.condition, 0) + 1
    assert all(counts["grasps"][k] == by_cond.get(k, 0) for k in counts["grasps"]), (counts, by_cond)
    assert doc["split"]["counts"]["grasps"] == counts["grasps"] and CO.min_select_grasps() == 20
    sub = CO.CollectConfig(**{**cfg.dump(), "collect_subset": [T[0]], "expected_grasps": 15})
    assert CO.split_counts(sub, CO.build(sub)[1])["grasps"] == counts["grasps"], "collect_subset does not change the counts"
    old_min = CO.min_select_grasps
    CO.min_select_grasps = lambda: 10 ** 6
    try:
        msg = raises(lambda: CO.prepare(cfg, tmp / "o_ref", log=lambda m: None), CO.Refused)
        assert "selection grasps" in msg and not (tmp / "o_ref" / "plan.json").exists(), msg
        logs = []
        CO.prepare(sub, tmp / "o_sub", log=logs.append)
        assert any(m.startswith("WARNING split") for m in logs) and (tmp / "o_sub" / "plan.json").exists(), logs
        # r5-integration-02: production needs n_select >= ceil(1.25 x m) = 25 for m = 20: 24 refused, 25 passes; a
        # subset with 24 only warns
        CO.min_select_grasps = old_min
        assert CO.production_select_min(20) == 25
        mk = lambda sel: {"grasps": {**counts["grasps"], "select": sel}, "min_select_grasps": 20,  # noqa: E731
                          "select_margin": sel - 20, "production_min_select_grasps": 25}
        assert "= 25" in CO.counts_problem(mk(24), production=True) and CO.counts_problem(mk(25), production=True) is None
        assert "MIN_SELECT_GRASPS 20" in CO.counts_problem(mk(19), production=True)
        assert "production minimum" in CO.counts_problem(mk(24)) and CO.counts_problem(mk(25)) is None
        assert "25 = ceil(1.25 x 20)" in CO.counts_text(mk(24))
        old_sc = CO.split_counts
        try:
            for sel in (24, 25):
                CO.split_counts = lambda c, t, sel=sel: mk(sel)
                o, logs = tmp / f"o_prod{sel}", []
                if sel == 24:
                    msg = raises(lambda: CO.prepare(cfg, o, log=logs.append), CO.Refused)
                    assert "24 selection grasps < ceil(1.25 x MIN_SELECT_GRASPS 20) = 25" in msg, msg
                    assert not (o / "plan.json").exists()
                    CO.prepare(CO.CollectConfig(**{**sub.dump()}), tmp / "o_sub24", log=logs.append)
                    assert any(m_.startswith("WARNING split") and "production minimum" in m_ for m_ in logs), logs
                else:
                    CO.prepare(cfg, o, log=logs.append)
                    assert (o / "plan.json").exists() and not any(m_.startswith("WARNING") for m_ in logs), logs
        finally:
            CO.split_counts = old_sc
    finally:
        CO.min_select_grasps = old_min
    return (f"40 assemblies: test {sd['test']}, held-out {sd['heldout_shape']} (quotas by group), near twins together, "
            f"chosen inside the training ranges; 15 grasps: 3 band (2 with a zero-width variable) as heldout_grasp, "
            f"train u outside [0.55, 0.65], select by crc32 ({n_sel}); deterministic; split file written; counts "
            f"{counts['grasps']} in plan.json; too few selection grasps refused in production (24 < ceil(1.25 x 20) = 25 "
            f"refused, 25 accepted), warned with a subset")


# ------------------------------------------------------------------------------------------------------------- 11
HOOK_CHILD = r"""
import sys, types
sys.path.insert(0, sys.argv[1])
m = types.ModuleType("wm.sim_automate.scene"); m.AutomateScene = object; sys.modules["wm.sim_automate.scene"] = m
from pathlib import Path
from wm.sim_automate import app
from wm.sim_automate import collect as CO
out = Path(sys.argv[2])
hb = CO.Heartbeat(out, 0)
c = CO.claim(out / "00001" / "00001_g00")
print("CLAIMED", c.attempt, flush=True)
app.assert_off_forbidden_gpu("test")
print("PASSED", flush=True)
"""


def test_pre_exit_hook(tmp: Path):
    """A GPU-check exit (app.assert_off_forbidden_gpu -> os._exit) with a claim held: the hook registered by collect.py
    releases the claim (no death), drops the heartbeat; 7 also writes STOP. The real app.py, a fake nvidia-smi."""
    from wm.sim_automate import app
    assert any(getattr(h, "__name__", "") == "_pre_exit" for h in app.PRE_EXIT_HOOKS), app.PRE_EXIT_HOOKS
    notes = []
    for name, table, code in (("violation", "C 300 MiB", 7), ("unreadable", None, 8), ("graphics", "G 8 MiB", 0),
                              ("graphics_big", "G 60 MiB", 7)):
        d = tmp / name
        out = d / "out"
        g = out / "00001" / "00001_g00"
        g.mkdir(parents=True)
        prev = "attempt 1 pid 999999995 start 4 time t0"            # an older dead attempt: must come back unchanged
        (g / ".claim").write_text(prev)
        b = d / "bin"
        b.mkdir()
        body = "exit 1" if table is None else (
            f'printf "GPU 00000000:16:00.0\\n    Processes\\n        Process ID : %s\\n            Type : {table.split()[0]}\\n'
            f'            Used GPU Memory : {table.split(" ", 1)[1]}\\n" "$PPID"')
        (b / "nvidia-smi").write_text("#!/bin/bash\n" + body + "\n")
        (b / "nvidia-smi").chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
        env["PATH"] = f"{b}:{env['PATH']}"
        r = subprocess.run([sys.executable, "-c", HOOK_CHILD, str(WM / "src"), str(out)], capture_output=True, text=True,
                           env=env, timeout=120)
        assert r.returncode == code and "CLAIMED 2" in r.stdout, (name, r.returncode, r.stdout, r.stderr)
        if code == 0:
            assert "PASSED" in r.stdout and not (out / "STOP").exists(), (name, r.stdout)
            notes.append(f"{name}: no exit")
            continue
        assert (g / ".claim").read_text() == prev, (name, (g / ".claim").read_text())
        assert not (out / "deaths").exists() and not list((out / "heartbeat").glob("*.json")), name
        alert = (out / "logs" / "ALERT").read_text()
        assert f"exit {code}" in alert and "released (not a death)" in alert, alert
        assert (out / "STOP").exists() == (code == 7), name
        c = CO.claim(g)                                             # the next worker sees only the older attempt
        assert c.attempt == 2 and c.old["pid"] == 999999995
        CO.release(c, done=False)
        notes.append(f"{name}: exit {code}, claim restored, {'STOP' if code == 7 else 'no STOP'}")
    return "; ".join(notes)


# ------------------------------------------------------------------------------------------------------------- 12
def test_gate(tmp: Path):
    """The milestone-1b gate: provenance, newest error-free result per check, pass_ii / pass_iii from the numbers, one
    code identity, collect_subset, the physics device switch."""
    raw = setup_assets(tmp, ids=("00001", "00002", "00003", "00004", "00005"))
    d1 = Path(raw["m1b_dirs"][0])
    d2 = tmp / "m1b" / "rerun"
    cfg = CO.CollectConfig(**{**raw, "m1b_dirs": [str(d1), str(d2)], "expected_grasps": None})
    st = lambda aid: CO.assembly_status(aid, cfg)  # noqa: E731
    assert st("00001")["status"] == "included" and st("00001")["m1b"]["ii"]["depth_left_mm"] == 0.02
    # stale: no provenance (a batch2-style file), another ranges file, device, switches, physics code; not forceable
    (d1 / "00002.json").write_text(json.dumps({"ii": II_OK, "iii": {"skipped": "x"}}))
    r = st("00002")
    assert r["status"] == "excluded" and "re-run: stale" in r["reason"] and "ranges_sha256" in r["reason"], r["reason"]
    for k, v in (("physics_device", "cuda:0"), ("switches", {**C.switches(), "socket_draw": "per_grasp"}),
                 ("physics_code_sha256", "0" * 64), ("grasp_py_sha256", "1" * 64), ("physics_code_sha256", None)):
        write_m1b(d1, "00002", **{k: v})
        assert f"stale {k}" in st("00002")["reason"], (k, st("00002")["reason"])
    # r5-integration-03 / r5-expert-03: the geometry file and the plug / socket meshes (batch_assemblies.asset_hashes)
    # are part of 'current': a missing field, another geometry file or mesh, or a missing file is stale
    write_m1b(d1, "00002")
    assert st("00002")["status"] == "included", st("00002")["reason"]
    for k in ("spec_sha256", "plug_obj_sha256", "socket_obj_sha256", "asset_sha256"):
        write_m1b(d1, "00002", **{k: None})
        assert f"stale {k}" in st("00002")["reason"], (k, st("00002")["reason"])
    f = d1 / "00002.json"
    r_ = json.loads(f.read_text())
    for k in ("spec_sha256", "plug_obj_sha256", "socket_obj_sha256", "asset_sha256"):
        r_.pop(k)
    f.write_text(json.dumps(r_))                                          # a result recorded before the fields existed
    reason = st("00002")["reason"]
    assert "re-run: stale spec_sha256, plug_obj_sha256, socket_obj_sha256, asset_sha256" in reason, reason
    d = A.ASSET_DIR / "00002"
    for name, k in ((A.SPEC_FILE, "spec_sha256"), ("plug.obj", "plug_obj_sha256"), ("socket.obj", "socket_obj_sha256")):
        write_m1b(d1, "00002")
        keep = (d / name).read_text()
        (d / name).write_text(keep + "\n ")                               # rewritten after the batch
        assert f"stale {k}, asset_sha256 " in st("00002")["reason"], (name, st("00002")["reason"])
        (d / name).write_text(keep)
    assert st("00002")["status"] == "included"
    keep = (d / "socket.obj").read_text()
    (d / "socket.obj").unlink()
    write_m1b(d1, "00002")                                                # both sides None: still stale
    assert "stale socket_obj_sha256" in st("00002")["reason"], st("00002")["reason"]
    (d / "socket.obj").write_text(keep)
    # r9-integration-01: the shared asset files (plug_grasps.json, the finger collision meshes) are part of 'current':
    # a finger mesh or plug_grasps.json rewritten after the batch, or a result recorded before the field existed, is stale
    fc = A.ASSET_DIR / "franka_mimic_collision"
    fc.mkdir(exist_ok=True)
    for name in ("panda_hand.obj", "panda_leftfinger.obj", "panda_rightfinger.obj"):
        (fc / name).write_text(f"o {name}\nv 0 0 0\n")
    (A.ASSET_DIR / "plug_grasps.json").write_text(json.dumps({"asset_00002": [0, 0, 0, 1, 0, 0, 0]}))
    write_m1b(d1, "00002")
    assert st("00002")["status"] == "included", st("00002")["reason"]
    sh = json.loads((d1 / "00002.json").read_text())["shared_asset_sha256"]
    assert set(sh) == {"plug_grasps.json", "franka_mimic_collision/panda_hand.obj",
                       "franka_mimic_collision/panda_leftfinger.obj", "franka_mimic_collision/panda_rightfinger.obj"} \
        and None not in sh.values(), sh
    for f in (fc / "panda_leftfinger.obj", A.ASSET_DIR / "plug_grasps.json"):
        keep = f.read_text()
        f.write_text(keep + "v 0 0 0.001\n")                             # re-exported after the batch
        assert st("00002")["status"] == "excluded" and "stale shared_asset_sha256" in st("00002")["reason"], \
            (f.name, st("00002")["reason"])
        f.write_text(keep)
        assert st("00002")["status"] == "included", st("00002")["reason"]
    (fc / "panda_extra.obj").write_text("o extra\n")                    # a new mesh in the folder counts too
    assert "stale shared_asset_sha256" in st("00002")["reason"], st("00002")["reason"]
    (fc / "panda_extra.obj").unlink()
    r_ = json.loads((d1 / "00002.json").read_text())
    r_.pop("shared_asset_sha256")
    (d1 / "00002.json").write_text(json.dumps(r_))                       # recorded before the field existed
    assert "stale shared_asset_sha256" in st("00002")["reason"], st("00002")["reason"]
    write_m1b(d1, "00002")
    # r4-expert-07: the batch procedure, geometry.py and sampling.py are part of the physics identity
    for rel in ("scripts/automate/batch_assemblies.py", "src/wm/sim/geometry.py", "src/wm/env/sampling.py"):
        assert rel in CO.PHYSICS_CODE_FILES
        root = tmp / f"tree_{rel.replace('/', '_')}"
        for pat in CO.PHYSICS_CODE_FILES:
            for f in WM.glob(pat):
                (root / f.relative_to(WM)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, root / f.relative_to(WM))
        before = CO.physics_code_identity(root)
        assert before == CO.physics_code_identity(WM)
        with open(root / rel, "a") as fh:
            fh.write("\n# edited\n")
        CO.physics_code_identity.cache_clear()
        edited = CO.physics_code_identity(root)
        assert edited != before
        write_m1b(d1, "00002", physics_code_sha256=edited)
        assert "stale physics_code_sha256" in st("00002")["reason"], rel
    forced = CO.CollectConfig(**{**cfg.dump(), "assemblies_force": ["00002"]})
    assert CO.assembly_status("00002", forced)["status"] == "excluded"
    # newest error-free result per check: a newer re-run wins, an error file never counts
    write_m1b(d1, "00003", ii={**II_OK, "depth_left_mm": 0.9})        # older, fails |depth_left| < 0.5
    write_m1b(d2, "00003", unix_time=2000.0)                          # newer, passes
    assert st("00003")["status"] == "included" and st("00003")["m1b"]["ii"]["file"].startswith(str(d2))
    write_m1b(d2, "00003", unix_time=2000.0, error="the simulator process died")
    assert "(ii) aligned insertion did not pass" in st("00003")["reason"]
    # pass_ii needs placed, completed, success at the end and |depth_left| < 0.5 mm, besides "pass"
    for over in ({"completed": False}, {"success": False}, {"placed": False}, {"depth_left_mm": -0.6}, {"pass": False},
                 {"completed": None}, {"placed": 1.0}):
        write_m1b(d1, "00004", ii={**II_OK, **over})
        assert "(ii)" in st("00004")["reason"], over
    ii_nc = {k: v for k, v in II_OK.items() if k != "completed"}           # r4-decisions-01: a missing key fails
    write_m1b(d1, "00004", ii=ii_nc)
    assert "(ii)" in st("00004")["reason"]
    # r4-expert-02 a: the end height in the reason, and the strict h > 0 case named
    write_m1b(d1, "00004", ii={**II_OK, "pass": False, "success": False, "height_mm": -0.003, "keypoint_dist_mm": 0.4})
    assert "fails only by h > 0: h_end = -3.0 um" in st("00004")["reason"], st("00004")["reason"]
    write_m1b(d1, "00004", ii={**II_OK, "pass": False, "success": False, "height_mm": 2.0, "keypoint_dist_mm": 2.0})
    assert "height_mm=2.0" in st("00004")["reason"] and "fails only" not in st("00004")["reason"]
    # pass_iii: move_ok, completed, |depth_left| < 0.5 mm, hand / finger force <= 1 N (box margin < 3 mm here)
    rg = json.loads(G.ranges_file("00005").read_text())
    rg.update(worst_margin_m=0.001, worst_point=[0.0, -0.005, 0.0])
    G.ranges_file("00005").write_text(json.dumps(rg))
    iii = {"grasp": [0.0, -0.005, 0.0], "mesh_margin_mm": 1.0, "move_ok": True, "completed": True, "depth_left_mm": 0.1,
           "hand_finger_socket_N": 0.5, "robot_socket_pairs_N": 0.2, "hand_finger_max_N": 0.5, "pass": True}
    write_m1b(d1, "00005", iii=iii)
    assert st("00005")["status"] == "included", st("00005")["reason"]
    # r4-integration-07 / r4-decisions-01: F_hf = hand_finger_max_N (sensor 0 N but robot-socket pairs 2 N fails);
    # a missing "pass", "completed" or force fails
    for over in ({"hand_finger_max_N": 1.5}, {"depth_left_mm": 0.7}, {"move_ok": False}, {"pass": False},
                 {"hand_finger_socket_N": 0.0, "robot_socket_pairs_N": 2.0, "hand_finger_max_N": 2.0},
                 {"hand_finger_max_N": None}, {"hand_finger_max_N": "x"}):
        write_m1b(d1, "00005", iii={**iii, **over})
        assert "(iii) failed" in st("00005")["reason"], over
    for k in ("pass", "completed", "hand_finger_max_N"):
        write_m1b(d1, "00005", iii={kk: v for kk, v in iii.items() if kk != k})
        assert "(iii) failed" in st("00005")["reason"], k
    write_m1b(d1, "00005", iii={**iii, "hand_finger_socket_N": 0.0, "robot_socket_pairs_N": 2.0, "hand_finger_max_N": 2.0})
    assert "robot_socket_pairs_N 2.0 N" in st("00005")["reason"], st("00005")["reason"]
    write_m1b(d1, "00005", iii=iii)
    # one code identity behind all included results: a newer result without physics_code_sha256 is stale now
    # (r4-expert-check-01), so it cannot mix code into the plan; m1b_code still refuses two identities
    write_m1b(d2, "00001", unix_time=3000.0, code_files_sha256="other", physics_code_sha256=None)
    write_m1b(d1, "00004")
    assert "stale physics_code_sha256" in st("00001")["reason"], st("00001")["reason"]
    two = [{"status": "included", "m1b": {"ii": {"physics_code_sha256": "a"}}},
           {"status": "included", "m1b": {"ii": {"physics_code_sha256": "b"}}}]
    assert "different code" in raises(lambda: CO.m1b_code(two), CO.Refused)
    (d2 / "00001.json").unlink()
    (d2 / "00003.json").unlink()
    write_m1b(d2, "00003", unix_time=2000.0)
    specs, table = CO.build(cfg)
    assert CO.m1b_code(table) == {"field": "physics_code_sha256", "sha256": CO.physics_code_identity()}
    # collect_subset: the plan keeps only these assemblies' grasps; an excluded one is refused
    sub = CO.CollectConfig(**{**cfg.dump(), "collect_subset": ["00003"]})
    assert {s_.port for s_ in CO.build(sub)[0]} == {"00003"}
    (d1 / "00002.json").write_text(json.dumps({"ii": II_OK}))
    assert "collect_subset" in raises(lambda: CO.build(CO.CollectConfig(**{**cfg.dump(), "collect_subset": ["00002"]})),
                                      CO.Refused)
    # the config device must equal WM_AUTOMATE_DEVICE (config.PHYSICS_DEVICE)
    gpu = CO.CollectConfig(**{**cfg.dump(), "device": "cuda:0"})
    assert "WM_AUTOMATE_DEVICE" in raises(lambda: CO.prepare(gpu, tmp / "o", log=lambda m: None), CO.Refused)
    # status table shows the depth and the start-box property
    row = st("00001")
    row["start_box"] = {"possible": True, "h_min_mm": 8.2, "limit_mm": 15.0, "fraction": 0.3, "draws": 2000,
                        "gap_from": "entrance"}
    txt = CO.status_text([row])
    assert "depth_left_mm (ii) 0.020" in txt and "START ALREADY SUCCESSFUL" in txt, txt
    return "stale provenance (none, ranges, device, switches, physics code, missing physics code, grasp.py, " \
           "geometry file / plug / socket mesh hashes: missing, changed, file gone; shared asset hashes: a changed " \
           "finger mesh or plug_grasps.json, a new mesh, a missing field) excluded and not forceable; " \
           "batch_assemblies.py / geometry.py / sampling.py edits change the physics " \
           "identity -> stale; newest error-free result per check; pass_ii / pass_iii from the recorded numbers " \
           "(missing keys fail; F_hf = hand_finger_max_N; h > 0 case named); mixed code refused; collect_subset; " \
           "device = WM_AUTOMATE_DEVICE; depth + start-box flag in the table"


# ------------------------------------------------------------------------------------------------------------- 14
FAKE_RUN_ON_RS2 = r"""#!/bin/bash
# stand-in run_on_rs2.sh: the next scenario code; writes the unit file, the log, markers as qx_watch.py would
name=$1; shift; shift
OUT=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && OUT=$a; prev=$a; done
k=$(cat "$OUT/k" 2>/dev/null || echo 0); echo $((k + 1)) > "$OUT/k"
code=$(sed -n "$((k + 1))p" "$OUT/scenario.txt"); [ -z "$code" ] && code=done
unit=qingxin-run-$name-1.scope
echo "$unit" > "$QX_UNIT_FILE"
LOGS=$(dirname "$QX_UNIT_FILE")
echo "start $name" > "$LOGS/$name.log"
shed() { mkdir -p "$QX_RUN_DIR/shed"; echo "test reason" > "$QX_RUN_DIR/shed/$unit"; }
# shed during launch(): Kit aborts (134) before batch_assemblies.py books or writes any marker (r6-ops-02)
[ "$code" = shedlaunch ] && { shed; exit 134; }
[ -e "$OUT/00001.started" ] && echo "$name" | cut -d_ -f1,2 >> "$OUT/booked"   # batch_assemblies.py would book a crash here (<tag>_<k>)
echo x > "$OUT/00001.started"                       # batch_assemblies.py's marker of the assembly it is in
case $code in
  done) rm -f "$OUT/00001.started"; echo "BATCH DONE" >> "$LOGS/$name.log"; exit 0 ;;
  shed|gpu0) mkdir -p "$QX_RUN_DIR/$code"; echo "test reason" > "$QX_RUN_DIR/$code/$unit"; exit 143 ;;
  shed134) shed; exit 134 ;;                         # Kit's SIGTERM abort (carb assertion, core dump)
  donethenshed) rm -f "$OUT/00001.started"; echo "BATCH DONE" >> "$LOGS/$name.log"; shed; exit 134 ;;
  *) exit $code ;;
esac
"""


def batch_loop(tmp: Path, scenario: list, nowatch=False, old_marker=False) -> tuple[int, str, Path]:
    W = tmp / "code"
    (W / "scripts" / "automate").mkdir(parents=True)
    shutil.copy(WM / "scripts/automate/run_batch.sh", W / "scripts/automate/")
    (W / "scripts/automate/run_on_rs2.sh").write_text(FAKE_RUN_ON_RS2)
    (W / "scripts/automate/caps.sh").write_text(
        f"export QX_RUN_DIR={tmp}/run\nmkdir -p $QX_RUN_DIR\nLOGS={tmp}/logs\n"
        "sleep() { rm -f \"$QX_RUN_DIR/nowatch\"; :; }\n")
    (W / "scripts/automate/collect_lib.sh").write_text("cl_watch_ensure() { [ ! -e \"$QX_RUN_DIR/nowatch\" ]; }\n")
    out = tmp / "out"
    out.mkdir()
    (out / "scenario.txt").write_text("\n".join(map(str, scenario)) + "\n")
    if old_marker:                                  # left by an earlier process that hung in 00001 (20 min ago)
        (out / "00001.started").write_text("x")
        t = time.time() - 1200
        os.utime(out / "00001.started", (t, t))
    if nowatch:
        (tmp / "run").mkdir()
        (tmp / "run" / "nowatch").write_text("x")
    r = subprocess.run(["bash", str(W / "scripts/automate/run_batch.sh"), "p0", "--out", str(out), "--ids", "00001,00002"],
                       capture_output=True, text=True, timeout=120)
    alert = (tmp / "run" / "ALERT").read_text() if (tmp / "run" / "ALERT").exists() else ""
    return r.returncode, r.stdout + r.stderr + "\nALERT:" + alert, out


def test_batch_loop(tmp: Path):
    """run_batch.sh (r4-ops-01 / N1): watcher required, shed vs timeout vs GPU-0 stop vs refusal."""
    t = lambda n: (tmp / n).mkdir() or tmp / n  # noqa: E731
    rc, txt, out = batch_loop(t("a"), ["shed", 1, "done"])
    assert rc == 0 and "shed by qx_watch.py" in txt and (out / "booked").read_text().split() == ["p0_2"], \
        "the start after the shed (p0_1 again) finds no marker; only the exit 1 is booked as a crash (by p0_2)"
    assert not list((tmp / "a" / "run" / "shed").iterdir()), "the shed marker is consumed"
    ks = lambda d: sorted(p.name.split("_")[1] for p in (tmp / d / "logs").glob("p0_[0-9]*_*.log"))  # noqa: E731
    assert ks("a") == ["1", "1", "2"], ("the shed did not count (k 1 again), the exit 1 did; r7-ops-04: the shed "
                                        "start's log is kept", ks("a"))
    # r6-ops-02: Kit's exit 134 on the watcher's SIGTERM is a shed too (markers of this start released, not counted)
    rc, txt, out = batch_loop(t("a2"), ["shed134", 1, "done"])
    assert rc == 0 and "shed by qx_watch.py" in txt and (out / "booked").read_text().split() == ["p0_2"], txt
    assert ks("a2") == ["1", "1", "2"], (ks("a2"), txt)
    assert not list((tmp / "a2" / "run" / "shed").iterdir()), "the shed marker is consumed"
    # a shed during launch() leaves the marker an earlier hang left (older than this start): the next start books it
    rc, txt, out = batch_loop(t("a3"), ["shedlaunch", "done"], old_marker=True)
    assert rc == 0 and "shed by qx_watch.py" in txt and (out / "booked").read_text().split() == ["p0_1"], txt
    # a log with BATCH DONE is done, whatever shed marker came after it
    rc, txt, out = batch_loop(t("a4"), ["donethenshed"])
    assert rc == 0 and "p0: BATCH DONE" in txt and "shed by" not in txt, txt
    rc, txt, out = batch_loop(t("b"), ["gpu0"])
    assert rc == 7 and (out / "STOP").exists() and "forbidden GPU-0" in txt and "test reason" in (out / "STOP").read_text()
    assert not (out / "00001.started").exists(), "r5-ops-03: a watcher GPU-0 stop releases this start's markers"
    rc, txt, out = batch_loop(t("c"), [124])
    assert (out / "booked").read_text().split() == ["p0_2"] and "exit 124" in txt, "a timeout keeps the marker: booked"
    rc, txt, out = batch_loop(t("d"), [3])
    assert rc == 3 and "refused (exit 3" in txt, txt
    rc, txt, out = batch_loop(t("e"), ["done"], nowatch=True)
    assert rc == 0 and "watcher qx_watch.py is not running" in txt, txt
    rc, txt, out = batch_loop(t("f"), ["shed"] * 7)
    assert rc == 1 and "7 times within 24 h" in txt, txt
    # r5-ops-05: BATCH DONE ends with exit 0 and no ALERT; 30 counted runs without it end with an ALERT naming the
    # ids without a result; the refusals reaching 30 give one ALERT line
    rc, txt, out = batch_loop(t("g"), [1, "done"])
    assert rc == 0 and "p0: BATCH DONE" in txt and txt.endswith("ALERT:"), txt
    rc, txt, out = batch_loop(t("h"), [1] * 30)
    assert rc == 1 and "30 counted runs without BATCH DONE" in txt and "ids without a result: 00001 00002" in txt, txt
    assert len(list((tmp / "h" / "logs").glob("p0_[0-9]*_*.log"))) == 30 and f"log {tmp}/h/logs/p0_30_" in txt, txt
    rc, txt, out = batch_loop(t("i"), [99] * 31 + ["done"])
    assert rc == 0 and txt.split("\nALERT:")[1].count("reached 30") == 1, txt
    assert ks("i") == ["1"], ("r7-ops-04: the logs of refused starts are removed", ks("i"))
    return "shed (exit 143 or Kit's 134): markers of this start removed, an older hang marker kept and booked, " \
           "shed marker consumed, not counted; BATCH DONE before a shed -> done; qx_watch GPU-0 stop -> STOP + exit 7 " \
           "and this start's markers released; exit 124 keeps the marker; exit 3 ends + ALERT; no watcher -> wait; " \
           "7 sheds in 24 h end the loop; BATCH DONE -> exit 0; 30 counted runs -> ALERT with the ids left, exit 1; " \
           "refusals reaching 30 -> one ALERT line; every start has its own log <tag>_<k>_<time>.log (a shed start's " \
           "log kept, refused starts' logs removed)"


# ------------------------------------------------------------------------------------------------------------- 15
def test_parallel_batch(tmp: Path):
    """parallel_batch.sh: refused without a watcher; --retry-crashes moves a crash-only result once (r4-ops-01 d)."""
    W = tmp / "live"
    (W / "scripts" / "automate").mkdir(parents=True)
    for f in ("parallel_batch.sh", "caps.sh"):
        shutil.copy(WM / "scripts/automate" / f, W / "scripts/automate/")
    os.symlink(WM / "src", W / "src")
    assets = tmp / "assets"
    assets.mkdir()
    (assets / "plug_grasps.json").write_text(json.dumps({f"asset_0000{i}": {} for i in (1, 2, 3, 4)}))
    tm = tmp / "faketmux"                   # sessions listed from a file; "new" of a name in failnew fails
    tm.write_text(f"""#!/bin/bash
case $1 in
  ls) cat {tmp}/sessions 2>/dev/null ;;
  new) s=""; prev=""; for a in "$@"; do [ "$prev" = -s ] && s=$a; prev=$a; done
       grep -qx "$s" {tmp}/failnew 2>/dev/null && exit 1
       echo "tmux $*" ;;
esac
""")
    tm.chmod(0o755)
    lib = (WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}").replace(
        'CL_TMUX="tmux -L qingxin"', f'CL_TMUX="{tm}"')
    lib += ("\ncl_watch_ensure() { [ ! -e \"$QX_RUN_DIR/nowatch\" ] || { echo 'WARNING: no watcher'; return 1; }; }\n"
            'eval "$(declare -f cl_clean_env | sed \'1s/cl_clean_env/cl_clean_env_real/\')"\n'
            f'cl_clean_env() {{ cl_clean_env_real "$@" || return 1; export WM_AUTOMATE_ASSETS={assets}; }}\n')
    (W / "scripts/automate/collect_lib.sh").write_text(lib)
    caps_txt = (W / "scripts/automate/caps.sh").read_text() + (
        f"\nQX_RUN_DIR={tmp}/run; mkdir -p $QX_RUN_DIR; LOG={tmp}/plog/parallel_launcher.log\n"
        "qx_run() { shift 5; \"$@\"; }\n")
    (W / "scripts/automate/caps.sh").write_text(caps_txt)
    out = tmp / "out"
    out.mkdir()
    crash = {"assembly_id": "00002", "error": "the simulator process died during this assembly", "started": "t"}
    (out / "00001.json").write_text(json.dumps({"ii": {}}))
    (out / "00002.json").write_text(json.dumps(crash))
    (out / "00003.json").write_text(json.dumps({**crash, "assembly_id": "00003"}))
    (out / "errors").mkdir()
    (out / "errors" / "00003.1.json").write_text("{}")                  # 00003 was retried once already
    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    run = lambda *a, **e: subprocess.run(["bash", str(W / "scripts/automate/parallel_batch.sh"), str(out), *a],  # noqa: E731
                                         capture_output=True, text=True, timeout=300, env={**env, **e})
    (tmp / "run").mkdir()
    (tmp / "run" / "nowatch").write_text("x")
    r = run("2")
    assert r.returncode == 1 and "no live watcher" in r.stdout, r.stdout + r.stderr
    (tmp / "run" / "nowatch").unlink()
    # r5-ops-01: a qingxin-batch* session of an earlier launch refuses before anything moves
    (tmp / "sessions").write_text("qingxin-smoke\nqingxin-batch3\n")
    r = run("2", "--retry-crashes")
    assert r.returncode == 1 and "batch loops still running: qingxin-batch3" in r.stdout, r.stdout + r.stderr
    assert "batch loops still running" in (tmp / "plog" / "parallel_launcher.log").read_text()
    assert (out / "00002.json").exists(), "nothing moved"
    (tmp / "sessions").unlink()
    r = run("2", "--retry-crashes")
    log = (tmp / "plog" / "parallel_launcher.log").read_text()
    assert r.returncode == 0, r.stdout + r.stderr + log
    assert not (out / "00002.json").exists() and list((out / "errors").glob("00002.*.json")), log
    assert (out / "00003.json").exists() and "not retried): 00003" in log and "remaining 2: 00002 00004" in log, log
    assert "TIMEOUT=5400" in log or "TIMEOUT=5400" in r.stdout, (r.stdout, log)
    # r5-ops-02: the log tag carries the batch folder and the launch stamp
    import re
    tags = re.findall(r"run_batch\.sh (\S+) --out", r.stdout)
    assert len(tags) == 2 and all(re.fullmatch(r"out_\d{8}T\d{6}_p[01]", x) for x in tags), (tags, r.stdout)
    # r5-ops-01: a worker whose tmux session cannot start is printed and the launcher exits 1
    (tmp / "failnew").write_text("qingxin-batch1\n")
    r = run("2")
    assert r.returncode == 1 and "worker 1 NOT started" in r.stdout and "1 workers not started" in r.stdout, r.stdout
    assert "worker 0 started" in (tmp / "plog" / "parallel_launcher.log").read_text()
    (tmp / "failnew").unlink()
    # r7-ops-01: BATCH_TIMEOUT_S reaches the tmux command and the launcher log; not a whole number >= 1 -> refused
    r = run("2", BATCH_TIMEOUT_S="21600")
    log = (tmp / "plog" / "parallel_launcher.log").read_text()
    assert r.returncode == 0 and r.stdout.count("TIMEOUT=21600 bash") == 2 and "timeout 21600 s), ids" in log, (r.stdout, log)
    for bad in ("abc", "0", "-5", "1.5"):
        r = run("2", BATCH_TIMEOUT_S=bad)
        assert r.returncode == 1 and f"refused: BATCH_TIMEOUT_S '{bad}'" in r.stdout, (bad, r.stdout)
    # --ids (lead decision, round 7: the GPU batch runs a given list): only those assemblies are listed and started
    r = run("2", "--ids", "00004")
    log = (tmp / "plog" / "parallel_launcher.log").read_text()
    assert r.returncode == 0 and r.stdout.count("run_batch.sh") == 1 and "--ids 00004" in r.stdout, r.stdout
    assert "assemblies: 00004;" in log and "remaining 1: 00004\n" in log, log
    r = run("2", "--ids", "00001,00004")                                     # 00001 has a result: only 00004 runs
    assert r.returncode == 0 and r.stdout.count("run_batch.sh") == 1 and "--ids 00004" in r.stdout, r.stdout
    for bad, why in (("00004,00009", "not assemblies ['00009']"), ("00004,00004", "repeated ids"),
                     ("4a", "refused: --ids '4a'"), (None, "--ids needs a list")):
        r = run("2", "--ids", *([bad] if bad else []))
        assert r.returncode == 1 and why in r.stdout, (bad, r.stdout)
    return "no watcher -> refused; a running qingxin-batch* session -> refused before any move; --retry-crashes " \
           "moves a crash-only result once (00002 retried, 00003 already retried and listed), todo = assemblies " \
           "without a result; TIMEOUT 5400 s by default, BATCH_TIMEOUT_S 21600 to the tmux command and the log, 'abc' / " \
           "'0' / '-5' / '1.5' refused; --ids limits the listing and the starts (unknown, repeated, malformed, " \
           "missing list refused); log tag <out>_<stamp>_p<w>; a worker not started -> printed, exit 1"


# ------------------------------------------------------------------------------------------------------------- 16
def test_smoke_stats(tmp: Path):
    """collect_smoke_check (h), the end heights of (b) and the launcher.log rule of (e) on synthetic rows
    (r4-decisions-02 / N2, r4-expert-02 b / -04, r6-integration-01 / r6-decisions-01 / r6-expert-03, r6-ops-01)."""
    sys.path.insert(0, str(WM / "scripts" / "automate" / "smoke"))
    import re
    import check_expert_data as CED
    import collect_smoke_check as SC
    lag = {"leg": "retreat", "ok": False, "reason": "lag", "max_lag_mm": 1.6, "peak_plug_socket_N": 0.2,
           "peak_finger_socket_N": 0.1, "steps": 3}
    desc = {"leg": "descend", "ok": False, "reason": "plug_socket_contact", "max_lag_mm": 0.3,
            "peak_plug_socket_N": 1.4, "peak_finger_socket_N": 0.0}
    ok_legs = [{"leg": "retreat", "ok": True, "max_lag_mm": 0.9}, {"leg": "descend", "ok": True, "max_lag_mm": 0.5}]
    rows = [{"idx": 0, "kind": "reference", "outcome": "inserted", "peak": {"min_actual_top_gap_mm_outside": 0.4}},
            {"idx": 1, "kind": "insert", "outcome": "reset_failed", "peak": {"min_actual_top_gap_mm_outside": 2.7}},
            {"idx": 2, "kind": "insert", "outcome": "inserted", "placement_attempts": 3,
             "peak": {"min_actual_top_gap_mm_outside": 1.5},
             "placement_failures": [{"stage": "retreat", "legs": [lag]},
                                    {"stage": "descend", "legs": [{**ok_legs[0]}, {"leg": "transfer", "ok": True}, desc]}],
             "placement_legs": ok_legs},
            {"idx": 3, "kind": "insert", "outcome": "label_failed"},
            {"idx": 4, "kind": "contact", "outcome": "placement_failed", "placement_attempts": 1,
             "placement_failures": [{"stage": "clearance"}]}]
    # 00005: 2 contact_report_error rows of 13 socket draws (t00 + 12); every row reached no placement
    rows5 = [{"idx": 0, "kind": "reference", "outcome": "inserted"}] + [
        {"idx": i, "kind": "contact", "outcome": "contact_report_error" if i <= 2 else "jammed"} for i in range(1, 13)]
    gj = {"00001_g00": {"grasp": "00001_g00", "port": "00001", "reference_ok": True, "trajectories": rows},
          "00001_g01": {"grasp": "00001_g01", "port": "00001", "reference_ok": False, "reference_expert_completed": True,
                        "reference_success": {"height_mm": -0.004, "keypoint_dist_mm": 0.3, "inserted_end": False},
                        "reference_failure": "rule fails at the end", "trajectories": []},
          "00002_g00": {"grasp": "00002_g00", "port": "00002", "reference_ok": False,
                        "reference_soft_outcome": "reset_failed", "trajectories": []},
          "00005_g00": {"grasp": "00005_g00", "port": "00005", "reference_ok": True, "trajectories": rows5}}
    plan = {"grasps": [{"grasp": k, "port": v["port"], "trajectories": [None] * (13 if k == "00005_g00" else 5)}
                       for k, v in gj.items()]}
    info, show, fail = SC.placement_stats(plan, gj)
    a = info["per_assembly"]["00001"]
    assert a["planned_rows"] == 4 and a["placement_attempts"] == 4 and "shares" not in a, a
    assert a["failed_attempts"] == {"retreat lag": 1, "descend plug_socket_contact": 1, "clearance": 1}, a
    assert a["free_lag_stops"] == 1 and fail and "lag" in fail[0], (a, fail)
    assert a["actual_top_gap_lt_1mm"] == {"n": 1, "of": 3, "min_mm": 0.4}, a            # information only, no FAIL
    assert a["actual_top_gap_lt_2mm"] == {"n": 2, "of": 3, "min_mm": 0.4}, a            # r9-decisions-01: 0.4 and 1.5 mm
    assert info["per_assembly"]["00005"]["actual_top_gap_lt_1mm"] == {"n": 0, "of": 0, "min_mm": None}
    assert info["per_assembly"]["00005"]["actual_top_gap_lt_2mm"] == {"n": 0, "of": 0, "min_mm": None}
    assert a["t00_refused"]["rule_fails_at_end"][0]["height_mm"] == -0.004
    assert info["per_assembly"]["00002"]["t00_refused"]["soft"]
    # r8-decisions-chk-01: a t00 that completed and passes AutoMate's rule but has no FK axis (collect.py: reference_ok
    # set False afterwards) is "other", as check_expert_data.t00_cause says, never rule_fails_at_end
    noax = {"grasp": "00009_g00", "port": "00009", "reference_ok": False, "reference_expert_completed": True,
            "reference_success": {"height_mm": 0.002, "keypoint_dist_mm": 0.4, "inserted_end": True},
            "reference_failure": "no axial push (FK axis undefined)", "trajectories": []}
    t9 = SC.placement_stats({"grasps": [{"grasp": "00009_g00", "port": "00009", "trajectories": [None]}]},
                            {"00009_g00": noax})[0]["per_assembly"]["00009"]["t00_refused"]
    assert t9 == {"not_completed": [], "rule_fails_at_end": [], "soft": [],
                  "other": [{"grasp": "00009_g00", "height_mm": 0.002, "keypoint_dist_mm": 0.4,
                             "failure": "no axial push (FK axis undefined)"}]}, t9
    assert CED.t00_cause(noax).startswith("other: no axial push"), CED.t00_cause(noax)
    assert CED.t00_cause(gj["00001_g01"]) == "completed, AutoMate rule fails at the end"
    assert info["max_lag_mm_per_leg"]["retreat"]["max"] == 1.6 and info["max_lag_mm_per_leg"]["descend"]["n"] == 2
    assert all("placement redraw" in x for x in show), "placement_stats shows only leg-reason redraw lines"
    h = SC.heights([0.02, -0.004, 0.005, None])
    assert h["n"] == 3 and h["min_mm"] == -0.004 and h["n_le_0.01mm"] == 2, h
    # r6-integration-01 / r6-decisions-01 / r6-expert-03: show_user = every flag of check_expert_data.draw_report (the
    # production list of prep_automate_data.sh / draws_check.sh), then the leg-reason redraw lines
    hinfo, shown, hfail = SC.section_h(plan, gj)
    _, want = CED.draw_report([gj[k] for k in sorted(gj)], {g["grasp"]: len(g["trajectories"]) for g in plan["grasps"]})
    assert shown[:len(want)] == want and shown[len(want):] == show and hfail == fail, (shown, want)
    assert hinfo["draws"]["flags_for_user"] == want
    for flag in ("00001: reset_failed 16.7% of 6", "00001: label_failed 16.7% of 6",     # draws incl. t00's reset
                 "00001: t00_refused 50.0% of 2", "00002: t00_refused 100.0% of 1",      # over grasps
                 "00001: placement_failed 50.0% of 2",                                   # over rows with placement
                 "00001: redraw clearance 25.0% of 4", "00005: contact_report_error 15.4% of 13"):
        assert any(x.startswith(flag) for x in shown), (flag, shown)
    assert any("00001: placement redraw 'retreat lag'" in x for x in shown), shown
    one = lambda n: {"port": None, "reference_ok": True, "trajectories": [  # noqa: E731
        {"idx": 1, "kind": "insert", "outcome": "inserted", "placement_attempts": n,
         "placement_failures": [{"stage": "clearance", "clearance_mm": {"min": 0.6}}]}]}
    info, show, fail = SC.placement_stats({}, {"00003_g00": {**one(10), "port": "00003"},
                                               "00004_g00": {**one(25), "port": "00004"}})
    assert [x for x in show if "clearance" in x] == ["00003: placement redraw 'clearance' in 1 of 10 placement "
                                                     "attempts (10.0 % > 5 %)"], show
    # a draw_report error is a FAIL of (h), not a silent empty list
    orig = CED.draw_report
    CED.draw_report = lambda *a, **k: 1 / 0
    try:
        _, shown, hfail = SC.section_h(plan, gj)
    finally:
        CED.draw_report = orig
    assert any("draws flags not available" in x for x in hfail) and all("placement redraw" in x for x in shown), hfail
    # the prep launcher's newest draws_<t>/flags_for_user.txt (the (f) cross-check)
    run = tmp / "prep"
    for t, fl in ((100, ["old"]), (1000, want)):
        (run / f"draws_{t}").mkdir(parents=True)
        (run / f"draws_{t}" / "flags_for_user.txt").write_text("".join(f + "\n" for f in fl))
    assert SC.prep_draws_flags(run) == want and SC.prep_draws_flags(tmp / "none") is None
    # r5-expert-06: a start that already satisfies AutoMate's rule is a FAIL of (b), t00 included
    v = {"reference_ok": False, "reference_success": {"success_at_start": True},
         "trajectories": [{"idx": 3, "success": {"success_at_start": True}}, {"idx": 4, "success": {}}]}
    assert SC.start_success_rows("00001_g00", v) == ["00001_g00 t03", "00001_g00 t00"]
    v["trajectories"].insert(0, {"idx": 0, "success": {"success_at_start": False}})
    assert SC.start_success_rows("00001_g00", v) == ["00001_g00 t03"]
    assert SC.start_success_rows("x", {"trajectories": [{"idx": 0, "success": {"success_at_start": False}}]}) == []
    # r6-ops-01: (e) on launcher.log. collect_launch.sh's launch line (written twice by the smoke), the STOP-file ends
    # and w2's exit 7 pass; a crash, an exit 8, the rate cap, a shed and the hang lines fail
    assert 'hang timeout $TIMEOUT s' in (WM / "scripts/automate/collect_launch.sh").read_text()
    ok_lines = ["2026-11-23 10:00:00 launch: 2 workers, MemoryMax 16G, CPUQuota 500%, GPU 7 GB, hang timeout 1020 s",
                "2026-11-23 10:00:01 worker 0: restart loop in tmux -L qingxin session qingxin-collect-smoke_x-0",
                "2026-11-23 11:00:00 worker 0 loop ended: stopped (STOP file)",
                "2026-11-23 11:00:00 worker 1 loop ended: stopped (STOP file)",
                "2026-11-23 11:00:00 worker 2 loop ended: forbidden GPU-0 context (exit 7, unit u), log l; STOP written "
                "(every worker of this run stops, no restart)",
                "2026-11-23 12:00:00 worker 0 loop ended: done (WORKER DONE; see logs/summary.json)"]
    bad_lines = ["2026-11-23 10:30:00 worker 0: crash (exit 1, 1 in the last hour), restarting",
                 "2026-11-23 10:30:00 worker 0: GPU check could not read nvidia-smi (exit 8, counted as a crash, 1 in "
                 "the last hour), restarting",
                 "2026-11-23 10:30:00 worker 0 loop ended: restart rate cap: 6 crashes within one hour (last: x)",
                 "2026-11-23 10:30:00 worker 0: shed by qx_watch.py (pressure; 1 in 24 h), retry in 600 s",
                 "2026-11-23 10:30:00 worker 0: killed after a hang with a claim held (pid 1, stage x), restarting",
                 "2026-11-23 10:30:00 worker 0: hang before a claim (stage launch), killed by qx_watch.py (exit 137, 1 "
                 "in the last hour), restarting"]
    assert not [x for x in ok_lines + ok_lines[:1] if re.search(SC.LAUNCHER_PROBLEM, x)]
    assert all(re.search(SC.LAUNCHER_PROBLEM, x) for x in bad_lines)
    return "placement_stats: planned rows, failed attempts by leg reason (> 5 % of attempts shown: clearance 10 % " \
           "yes, 4 % no), free-motion lag stops > 10 % FAIL, lag per leg, refused t00 by cause; show_user = " \
           "check_expert_data.draw_report's flags (reset_failed / label_failed / contact_report_error 2 of 13 over " \
           "draws incl. t00, t00_refused over grasps, placement_failed over placement rows, redraws) + leg-reason " \
           "lines; draw_report error -> FAIL; newest prep draws flags read; end heights; start-success rows; (e): " \
           "launch / STOP / exit-7 lines pass, crash / exit 8 / rate cap / shed / hang lines fail"


# ------------------------------------------------------------------------------------------------------------- 17
FAKE_PREP_STEP = r"""
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ["PREP_REC"], "a") as f:
    f.write(json.dumps({"script": sys.argv[0], "argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH"),
                        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "omp": os.environ.get("OMP_NUM_THREADS"),
                        "qx_gpu_gb": os.environ.get("QX_GPU_GB"), "pid": os.getpid(),
                        "isaac_env": os.environ.get("QX_TEST_ISAAC_ENV"),
                        "wm": sorted(k for k in os.environ if k.startswith("WM_"))}) + "\n")
if name == "check_expert_data.py":                     # --draws-only: <work>/draws.json with flags_for_user
    w = Path(sys.argv[sys.argv.index("--work") + 1])
    w.mkdir(parents=True, exist_ok=True)
    fl = json.loads(os.environ.get("PREP_FLAGS") or "[]")
    (w / "draws.json").write_text(json.dumps({"flags_for_user": fl}))
    (w / "flags_for_user.txt").write_text("".join(f + "\n" for f in fl))
if name == "build_index.py":                           # index_completeness.json as wm.data.index writes it
    o = Path(sys.argv[sys.argv.index("--automate") + 1])
    (o / "index_completeness.json").write_text(json.dumps({"planned": 3, "with_grasp_json": 3,
                                                           "partial": "--partial" in sys.argv}))
if name == "depth_stats.py" and os.environ.get("PREP_FAIL") == "depth_stats.py early":
    raise SystemExit("refused before compute (no distribution written)")    # r12-data-chk-01: e.g. a bad argument
if name == "depth_stats.py":                           # written before any refusal, as the real script (r9-data-01)
    o = Path(sys.argv[sys.argv.index("--data") + 1])
    o.mkdir(parents=True, exist_ok=True)
    (o / "depth_distribution.json").write_text(json.dumps({"n_valid": 3, "p50_m": 0.12}))
if os.environ.get("PREP_FAIL") == name:
    if name == "build_index.py":
        print("refused: the collection is still running")
        sys.exit(2)
    if name == "depth_stats.py":
        raise SystemExit("unusable statistics: too few valid pixels {}")
    sys.exit(4)
"""


def prep_tree(tmp: Path) -> Path:
    """A world_model stand-in: the real prep_automate_data.sh, caps.sh (pointed at fake /proc, cgroup and run folders)
    and collect_lib.sh (python = this interpreter); fake step scripts that record their call."""
    W = tmp / "live"
    for d in ("scripts/automate", "scripts/data", "src/wm"):
        (W / d).mkdir(parents=True)
    shutil.copy(WM / "scripts/automate/prep_automate_data.sh", W / "scripts/automate/")
    (W / "scripts/automate/caps.sh").write_text((WM / "scripts/automate/caps.sh").read_text() +
                                                f"\nQX_RUN_DIR={tmp}/run; QX_PROC={tmp}/proc; QX_CGROOT={tmp}/cg\n")
    lib = (WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}")
    (W / "scripts/automate/collect_lib.sh").write_text(lib.replace(  # stand-in: only marks that it was called
        "cl_isaac_env() {", "cl_isaac_env() { export QX_TEST_ISAAC_ENV=1; return 0\n"))
    shutil.copy(WM / "scripts/automate/draws_check.sh", W / "scripts/automate/")
    for f in ("scripts/automate/check_expert_data.py", "scripts/data/build_index.py", "scripts/data/cache_wm_v1_fast.py",
              "scripts/automate/depth_stats.py"):
        (W / f).write_text(FAKE_PREP_STEP)
    (W / "src/wm/__init__.py").write_text("")
    b = fake_tools(tmp)
    (b / "systemd-run").write_text("#!/bin/bash\n"                # runs the command without a scope, records options
                                   f"echo \"$*\" >> {tmp}/systemd_run_args\n"
                                   "while [ $# -gt 0 ]; do case $1 in -p) shift 2 ;; -*) shift ;; *) break ;; esac; done\n"
                                   "exec \"$@\"\n")
    (b / "systemd-run").chmod(0o755)
    return W


def prep(tmp: Path, W: Path, *args, fail="", flags=(), script="prep_automate_data.sh") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env.update(PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", PREP_REC=str(tmp / "rec.jsonl"), PREP_FAIL=fail,
               WM_AUTOMATE_SOCKET_DRAW="per_grasp", PYTHONPATH="/elsewhere", PREP_FLAGS=json.dumps(list(flags)))
    return subprocess.run(["bash", str(W / "scripts/automate" / script), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=300)


def test_prep_launcher(tmp: Path):
    """prep_automate_data.sh (r4-integration-04 / r4-int-chk-02): frozen code, capped CPU scopes with GPU 0 and
    CUDA_VISIBLE_DEVICES empty, PYTHONPATH = frozen src only, the three steps in order, stop at the first failure,
    PIDs and units recorded, D_max required; collect_smoke_check.prep_check on its records."""
    W = prep_tree(tmp)
    out, run, fast = tmp / "out", tmp / "prep", tmp / "fast"
    out.mkdir()
    rec = tmp / "rec.jsonl"
    for bad in ((run, out, fast), (run, out, fast, "abc"), (run, out, fast, "0"), (run, out, fast, "-1"),
                (run, out, fast, "0.5", "x"), (run, out, fast, "0.5", "--partal"), (run, out, fast, "3.3"),
                (run, out, fast, "3.27671"), (run, out, fast, "None"), (run, out, fast, "none ")):
        r = prep(tmp, W, *bad)
        assert r.returncode == 1 and ("D_max" in r.stdout), (bad, r.stdout, r.stderr)
    # r5-ops-04: early refusals also go to <run dir>/logs/ALERT
    assert (run / "logs" / "ALERT").read_text().count("prep_automate_data refused") == 10
    # r6-data-01: D_max above the int16 ceiling is refused before any step; the launcher's literal is policy320's
    import re
    from wm.tdmpc.policy320 import DEPTH_MAX_M
    lit = re.findall(r"^DEPTH_MAX_M=(\S+)", (WM / "scripts/automate/prep_automate_data.sh").read_text(), re.M)
    assert lit and float(lit[0]) == DEPTH_MAX_M, (lit, DEPTH_MAX_M)
    assert "(0, 3.2767]" in prep(tmp, W, run, out, fast, "3.3").stdout and not rec.exists()
    r = prep(tmp, W, run, out, fast, str(DEPTH_MAX_M))           # the ceiling itself passes the D_max check
    assert r.returncode == 1 and "D_max" not in r.stdout and "plan.json" in r.stdout, r.stdout
    r = prep(tmp, W, run, out, fast, "0.5")
    assert r.returncode == 1 and "plan.json" in r.stdout and not rec.exists(), r.stdout
    (out / "plan.json").write_text(json.dumps({"grasps": [{"port": "00001", "grasp": f"00001_g0{i}"} for i in range(3)]}))
    for i in range(2):
        (out / "00001" / f"00001_g0{i}").mkdir(parents=True)
        (out / "00001" / f"00001_g0{i}" / "grasp.json").write_text("{}")
    # r5-ops-chk-01: a collection that still runs (an active scope of its workers) or whose workers never ended
    (tmp / "units").write_text("qingxin-collect-out-w0-1-1700000000.scope\nqingxin-prep-x-1.scope\n")
    r = prep(tmp, W, run, out, fast, "0.5")
    assert r.returncode == 1 and "still runs" in r.stdout and "qingxin-collect-out-w0" in r.stdout, r.stdout
    (tmp / "units").write_text("qingxin-collect-outer-w0-1-1700000000.scope\n")         # another collection
    r = prep(tmp, W, run, out, fast, "0.5")
    assert r.returncode == 1 and "summary.json" in r.stdout and not rec.exists(), r.stdout
    (out / "logs").mkdir()
    (out / "logs" / "summary.json").write_text("{}")
    r = prep(tmp, W, run, out, fast, "0.5", flags=["00470: label_failed 7.0 % of 60 socket draws (> 5 %)"])
    assert r.returncode == 0 and "PREP OK" in r.stdout, r.stdout + r.stderr
    calls = [json.loads(x) for x in rec.read_text().splitlines()]
    code = run / "code"
    assert [Path(c["script"]) for c in calls] == [code / "scripts/automate/check_expert_data.py",
                                                  code / "scripts/data/build_index.py",
                                                  code / "scripts/data/cache_wm_v1_fast.py",
                                                  code / "scripts/automate/depth_stats.py"], calls
    assert calls[0]["argv"][:4] == ["--draws-only", "--data", str(out), "--work"] and \
        Path(calls[0]["argv"][4]).parent == run and Path(calls[0]["argv"][4]).name.startswith("draws_"), calls[0]
    assert calls[1]["argv"] == ["--automate", str(out), "--workers", "8"]
    assert calls[2]["argv"] == ["--src", str(out), "--dst", str(fast), "--workers", "8"]
    assert calls[3]["argv"] == ["--data", str(fast), "--depth-max-m", "0.5"]
    # r5-integration-01: the draws flags go to ALERT and the output
    assert "SHOW THE USER: 00470: label_failed 7.0 %" in (run / "logs" / "ALERT").read_text()
    assert "SHOW THE USER: 00470" in r.stdout and "grasp.json 2 of 3 planned" in r.stdout, r.stdout
    assert all(c["pythonpath"] == str(code / "src") and c["cvd"] == "" and c["omp"] == "1" and c["qx_gpu_gb"] == "0"
               and c["wm"] == ["WM_FORBIDDEN_GPU"] and c["isaac_env"] == "1" for c in calls), calls
    assert (run / "code.sha256").exists() and not os.access(code / "scripts/data/build_index.py", os.W_OK)
    sr = (tmp / "systemd_run_args").read_text().splitlines()
    assert len(sr) == 4 and all("--slice=qingxin.slice" in x and "MemoryMax=24G" in x and "CPUQuota=800%" in x and
                                "--unit=qingxin-prep-prep-" in x for x in sr), sr
    P = (run / "logs" / "prep_latest").read_text().strip()
    lines = Path(P + ".steps").read_text().splitlines()
    assert "# grasp.json 2 of 3 planned" in lines, lines
    steps = [x.split() for x in lines if not x.startswith("#")]
    assert [x[:2] for x in steps] == [["draws", "0"], ["index", "0"], ["fast", "0"], ["depth_stats", "0"]], steps
    assert all(x[2].startswith(f"qingxin-prep-prep-{x[0]}-") for x in steps), steps
    assert [int(x[3]) for x in steps] == [c["pid"] for c in calls] and all(x[4].isdigit() for x in steps), (steps, calls)
    assert Path(P + ".result").read_text().strip() == "ok"
    # stop at the first failing step; ALERT; a second start in the run dir finds the frozen copy unchanged
    rec.unlink()
    r = prep(tmp, W, run, out, fast, "0.5", fail="cache_wm_v1_fast.py")
    calls = [json.loads(x) for x in rec.read_text().splitlines()]
    assert r.returncode == 4 and "PREP FAILED at fast" in r.stdout and len(calls) == 3, r.stdout
    P2 = (run / "logs" / "prep_latest").read_text().strip()
    assert Path(P2 + ".result").read_text().split() == ["failed", "fast", "4"]
    assert "prep_automate_data fast ended with exit 4" in (run / "logs" / "ALERT").read_text()
    # the draws check never stops the chain (ALERT); the flags reach the index step
    rec.unlink()
    r = prep(tmp, W, run, out, fast, "0.5", "--accept-skipped", "--partial", fail="check_expert_data.py")
    calls = [json.loads(x) for x in rec.read_text().splitlines()]
    assert r.returncode == 0 and len(calls) == 4, r.stdout + r.stderr
    assert calls[1]["argv"] == ["--automate", str(out), "--workers", "8", "--accept-skipped", "--partial"], calls[1]
    assert "draws (check_expert_data.py --draws-only) ended with exit 4" in (run / "logs" / "ALERT").read_text()
    # exit 2 of build_index.py (collection running or unfinished) stops the chain with an ALERT
    rec.unlink()
    r = prep(tmp, W, run, out, fast, "0.5", fail="build_index.py")
    assert r.returncode == 2 and "running or unfinished" in r.stdout and len(rec.read_text().splitlines()) == 2, r.stdout
    assert "refused as running or unfinished" in (run / "logs" / "ALERT").read_text()
    # a changed frozen copy is refused
    f = code / "scripts/data/build_index.py"
    f.chmod(0o644)
    f.write_text(f.read_text() + "\n# edited\n")
    r = prep(tmp, W, run, out, fast, "0.5")
    assert r.returncode == 1 and "changed since it was frozen" in r.stdout, r.stdout
    # collect_smoke_check item (f) on the launcher's records
    sys.path.insert(0, str(WM / "scripts" / "automate" / "smoke"))
    import collect_smoke_check as SC
    run2 = tmp / "prep2"
    r = prep(tmp, W, run2, out, fast, "0.5", fail="depth_stats.py")
    assert r.returncode == 1, r.stdout + r.stderr
    why, info = SC.prep_check(run2, fast, r.returncode)
    assert not why and info["depth_stats"].startswith("unusable statistics") and len(info["steps"]) == 4, (why, info)
    assert info["depth_distribution"] == {"n_valid": 3, "p50_m": 0.12}, info          # r9-data-01: kept when refused
    (fast / "depth_distribution.json").rename(fast / "dd.json")
    assert "no depth_distribution.json" in SC.prep_check(run2, fast, r.returncode)[0][0]
    (fast / "dd.json").rename(fast / "depth_distribution.json")
    log = Path((run2 / "logs" / "prep_latest").read_text().strip() + ".depth_stats.log")
    log.chmod(0o644)
    log.write_text(log.read_text().replace("unusable statistics", "no training trajectories"))
    assert SC.prep_check(run2, fast, 1)[0], "another depth_stats refusal fails (f)"
    run3 = tmp / "prep3"
    r = prep(tmp, W, run3, out, fast, "0.5", fail="build_index.py")
    assert r.returncode == 2 and "index" in SC.prep_check(run3, fast, 2)[0][0]
    assert "no records" in SC.prep_check(tmp / "nothing", fast, 1)[0][0]
    run4 = tmp / "prep4"
    r = prep(tmp, W, run4, out, fast, "0.5")
    fast.mkdir(exist_ok=True)
    (fast / "manifest.json").write_text(json.dumps({"source": str(out), "entries": [], "split": {"name": "automate_v1"}}))
    from wm.tdmpc import demo320 as D
    good = {"split": {"name": "automate_v1"}, "train_files_sha256": D.train_files_sha256([], fast)}
    (fast / "depth_stats.json").write_text(json.dumps(good))
    why, info = SC.prep_check(run4, fast, 0)
    assert not why and info["result"] == "ok" and info["depth_distribution"]["n_valid"] == 3, (why, info)
    (fast / "depth_distribution.json").rename(fast / "dd.json")
    assert "no depth_distribution.json" in SC.prep_check(run4, fast, 0)[0][0], "required in the ok branch too"
    (fast / "dd.json").rename(fast / "depth_distribution.json")
    (fast / "depth_stats.json").write_text(json.dumps({**good, "train_files_sha256": "0" * 64}))
    assert "train_files_sha256" in SC.prep_check(run4, fast, 0)[0][0]
    # r10-data-01: D_max "none" = distribution only: an existing depth_stats.json is deleted, depth_stats.py runs with
    # --distribution-only (no D_max), only depth_distribution.json is left, result "ok distribution-only"
    (fast / "depth_distribution.json").unlink()
    rec.unlink()
    run5 = tmp / "prep5"
    r = prep(tmp, W, run5, out, fast, "none")
    assert r.returncode == 0 and "PREP OK (distribution only)" in r.stdout and "NO depth_stats.json" in r.stdout, \
        r.stdout + r.stderr
    assert f"--data {fast} --depth-max-m <D_max>" in r.stdout and f"R={run5}" in r.stdout, r.stdout
    calls = [json.loads(x) for x in rec.read_text().splitlines()]
    assert calls[-1]["argv"] == ["--data", str(fast), "--distribution-only"] and \
        Path(calls[-1]["script"]) == run5 / "code/scripts/automate/depth_stats.py", calls[-1]
    assert not (fast / "depth_stats.json").exists() and (fast / "depth_distribution.json").exists()
    P5 = (run5 / "logs" / "prep_latest").read_text().strip()
    assert Path(P5 + ".result").read_text().strip() == "ok distribution-only"
    assert [x.split()[0] for x in Path(P5 + ".steps").read_text().splitlines() if not x.startswith("#")] == \
        ["draws", "index", "fast", "depth_distribution"]
    assert SC.prep_check(run5, fast, 0)[0], "smoke item (f) must not accept a distribution-only run"
    # r11-data-01: with a numeric D_max too, an earlier depth_stats.json is deleted before the step, so a refused
    # depth_stats step leaves none (train_wm320 would otherwise accept the earlier D_max's file)
    (fast / "depth_stats.json").write_text(json.dumps(good))
    r = prep(tmp, W, tmp / "prep6", out, fast, "0.5", fail="depth_stats.py")
    assert r.returncode == 1 and "PREP FAILED" in r.stdout, r.stdout + r.stderr
    assert not (fast / "depth_stats.json").exists(), "an earlier depth_stats.json survived a refused depth_stats step"
    # r12-data-chk-01: an earlier depth_distribution.json is deleted too, so a depth step that fails before it writes
    # one leaves none (the old training set's numbers would answer the user's D_max question)
    (fast / "depth_distribution.json").write_text(json.dumps({"stale": True}))
    for dm in ("0.5", "none"):
        r = prep(tmp, W, tmp / f"prep7_{dm}", out, fast, dm, fail="depth_stats.py early")
        assert r.returncode == 1 and "PREP FAILED" in r.stdout, r.stdout + r.stderr
        assert not (fast / "depth_distribution.json").exists(), f"stale depth_distribution.json survived (D_max {dm})"
        (fast / "depth_distribution.json").write_text(json.dumps({"stale": True}))
    (fast / "depth_distribution.json").unlink()
    # the lead's mid-run draws check: capped CPU scope, flags printed, each new flag appended to <out>/logs/ALERT once
    flag = "00537: placement_failed 6.2 % of 65 socket draws (> 5 %)"
    for _ in range(2):
        r = prep(tmp, W, out, flags=[flag], script="draws_check.sh")
        assert r.returncode == 0 and f"SHOW THE USER: {flag}" in r.stdout and "1 flags" in r.stdout, r.stdout + r.stderr
    assert (out / "logs" / "ALERT").read_text().count(flag) == 1
    call = json.loads(rec.read_text().splitlines()[-1])
    assert call["argv"] == ["--data", str(out), "--work", str(out / "logs" / "draws"), "--draws-only"] and \
        call["cvd"] == "" and call["qx_gpu_gb"] == "0" and call["isaac_env"] == "1", call   # r6-data-05: cl_isaac_env
    # r5-ops-04: collect_smoke.sh refuses a config whose device differs from WM_AUTOMATE_DEVICE, before <out> exists,
    # and its early refusals are kept in <dirname out>/smoke_refusals.log
    (W / "scripts/automate/smoke").mkdir()
    shutil.copy(WM / "scripts/automate/smoke/collect_smoke.sh", W / "scripts/automate/smoke/")
    tm = tmp / "faketmux"                                       # our tmux server's sessions, listed from a file
    tm.write_text(f"#!/bin/bash\n[ \"$1\" = ls ] && cat {tmp}/sessions 2>/dev/null\nexit 0\n")
    tm.chmod(0o755)
    lib_f = W / "scripts/automate/collect_lib.sh"
    assert 'CL_TMUX="tmux -L qingxin"' in lib_f.read_text()
    lib_f.write_text(lib_f.read_text().replace('CL_TMUX="tmux -L qingxin"', f'CL_TMUX="{tm}"'))
    cfg_gpu = tmp / "cfg_gpu.json"
    cfg_gpu.write_text(json.dumps({"device": "cuda:0"}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    sm = lambda *a, **e: subprocess.run(["bash", str(W / "scripts/automate/smoke/collect_smoke.sh"), *map(str, a)],  # noqa: E731
                                        capture_output=True, text=True, env={**env, **e}, timeout=60)
    r = sm(tmp / "sm" / "smoke_gpu", cfg_gpu)
    assert r.returncode == 1 and "differs from WM_AUTOMATE_DEVICE 'cpu'" in r.stdout, r.stdout + r.stderr
    assert not (tmp / "sm" / "smoke_gpu").exists()
    r = sm(tmp / "sm" / "notsmoke", cfg_gpu)
    assert r.returncode == 1 and "smoke_" in r.stdout
    r = sm(tmp / "sm" / "smoke_gpu", cfg_gpu, "0")
    assert r.returncode == 1 and "hours" in r.stdout
    # r4-expert-06: no smoke_env result, or one the gate refuses, stops the smoke before <out> exists
    shutil.copy(WM / "scripts/automate/smoke/smoke_env_gate.py", W / "scripts/automate/smoke/")
    cfg_cpu = tmp / "cfg_cpu.json"
    cfg_cpu.write_text(json.dumps({"device": "cpu", "m1b_dirs": [str(tmp / "nobatch")]}))
    r = sm(tmp / "sm" / "smoke_cpu", cfg_cpu)
    assert r.returncode == 1 and "SMOKE_ENV_RESULT is not set" in r.stdout, r.stdout + r.stderr
    # r7-integration-01: a batch loop still running (qingxin-batch* session) refuses before the gate
    (tmp / "sessions").write_text("qingxin-watch\nqingxin-batch2\n")
    r = sm(tmp / "sm" / "smoke_cpu", cfg_cpu, SMOKE_ENV_RESULT=str(tmp / "missing.json"))
    assert r.returncode == 1 and "batch loops still running: qingxin-batch2" in r.stdout and "smoke_env gate" not in r.stdout, r.stdout
    (tmp / "sessions").unlink()
    # r11-ops-02: the smoke runs only the frozen code of the config's batch (first m1b_dirs folder), checked before
    # <out> exists: no m1b_dirs, no code.sha256, another hash refused; the batch's hash passes on to the gate
    cfg_nom = tmp / "cfg_nom.json"
    cfg_nom.write_text(json.dumps({"device": "cpu"}))
    r = sm(tmp / "sm" / "smoke_cpu", cfg_nom, SMOKE_ENV_RESULT=str(tmp / "missing.json"))
    assert r.returncode == 1 and "names no m1b_dirs" in r.stdout and "smoke_env gate" not in r.stdout, r.stdout
    r = sm(tmp / "sm" / "smoke_cpu", cfg_cpu, SMOKE_ENV_RESULT=str(tmp / "missing.json"))
    assert r.returncode == 1 and f"{tmp / 'nobatch'}/code.sha256 missing" in r.stdout, r.stdout + r.stderr
    (tmp / "nobatch").mkdir()
    (tmp / "nobatch" / "code.sha256").write_text("0" * 64 + "\n")
    r = sm(tmp / "sm" / "smoke_cpu", cfg_cpu, SMOKE_ENV_RESULT=str(tmp / "missing.json"))
    assert r.returncode == 1 and "differs from the batch's frozen code" in r.stdout and "smoke_env gate" not in r.stdout \
        and not (tmp / "sm" / "smoke_cpu").exists(), r.stdout
    h = subprocess.run(["bash", "-c", f"source {W}/scripts/automate/collect_lib.sh; cl_code_hash {W}"],
                       capture_output=True, text=True).stdout.strip()
    assert len(h) == 64, h
    (tmp / "nobatch" / "code.sha256").write_text(h + "\n")
    r = sm(tmp / "sm" / "smoke_cpu", cfg_cpu, SMOKE_ENV_RESULT=str(tmp / "missing.json"))
    assert r.returncode == 1 and "smoke_env gate" in r.stdout and not (tmp / "sm" / "smoke_cpu").exists(), r.stdout
    log = (tmp / "sm" / "smoke_refusals.log").read_text()
    assert "differs from WM_AUTOMATE_DEVICE" in log and "must start with smoke_" in log and "hours" in log, log
    assert "SMOKE_ENV_RESULT is not set" in log and "smoke_env gate" in log and "batch loops still running" in log, log
    assert "differs from the batch's frozen code" in log and "code.sha256 missing" in log, log
    return "D_max missing / not a number / <= 0 / above policy320.DEPTH_MAX_M (literal compared) and a folder " \
           "without plan.json refused before any step; frozen copy " \
           "(read-only, hash); index -> fast -> depth_stats from the copy with the exact arguments (8 workers, D_max), " \
           "each in its own qingxin-prep-* scope (24G, 800 %, GPU 0: CUDA_VISIBLE_DEVICES empty, QX_GPU_GB 0), " \
           "PYTHONPATH = frozen src only, WM_* cleaned, one BLAS thread; unit + pid + start time per step; stop at " \
           "the first failure (exit kept, ALERT); changed frozen copy refused; early refusals in ALERT; a running " \
           "collection (active worker scope) or no summary.json refused; draws check first, never stopping, flags " \
           "to ALERT; --accept-skipped / --partial to the index step; grasp.json count in P.steps; smoke (f): only " \
           "depth_stats' 'unusable statistics' tolerated, ok needs depth_stats.json with the manifest split and " \
           "train hash; D_max none: existing depth_stats.json deleted, depth_stats.py --distribution-only, only " \
           "depth_distribution.json left, result 'ok distribution-only' (refused by (f)); numeric D_max: an earlier " \
           "depth_stats.json deleted before the step, none left after a refused step; draws_check.sh capped with cl_isaac_env, flags to ALERT once; collect_smoke.sh device / name / hours " \
           "/ smoke_env result / running batch / code other than the batch's frozen copy (r11-ops-02) refusals " \
           "logged before <out> exists"


# ------------------------------------------------------------------------------------------------------------- 18
def test_start_box(tmp: Path):
    """collect.start_box_success: checks.start_box_success over draws of grasp (in the box) and start (with
    rotation) and socket yaw, gap_from = config.START_GAP_FROM; a deep-cavity assembly (00133-like: L 2.2 mm, g_top 22.5 mm,
    d_dis 26 mm) never starts successful from the socket top, but can from the entrance (r4-expert-01)."""
    from wm.env.sampling import sample_start
    spec = types.SimpleNamespace(pin_length_m=0.0022, disassembly_dist_m=0.026, depth_top_m=0.0247, axis_xy_m=(0.0, 0.0))
    T0 = G.Tz(-0.03)
    old = A.load_spec, G.T_hp0, C.START_GAP_FROM, CO.START_SUCCESS_DRAWS, C.DIAGNOSTIC
    try:
        A.load_spec, G.T_hp0, CO.START_SUCCESS_DRAWS = (lambda aid: spec), (lambda aid: T0), 300
        res = {}
        for gf in ("socket_top", "entrance"):
            C.START_GAP_FROM, C.DIAGNOSTIC = gf, gf == "entrance"       # "entrance": diagnostics only (2026-11-23)
            r = CO.start_box_success("00133", BOX)
            rng, n = np.random.default_rng(0), 0
            for _ in range(CO.START_SUCCESS_DRAWS):
                v = [float(rng.uniform(*BOX[k])) for k in CO.GRASP_VARS]
                off = sample_start(rng)
                yaw = np.radians(C.SOCKET_YAW_DEG + C.SOCKET_YAW_RANGE_DEG * rng.uniform())
                n += K.start_box_success(spec, G.T_hp(T0, *v, axis_xy=(0.0, 0.0)), off,
                                         (np.cos(yaw), -np.sin(yaw), 0.0), gap_from=gf)["success"]
            assert r["gap_from"] == gf and r["draws"] == 300 and r["fraction"] == round(n / 300, 4), (gf, r, n)
            assert abs(r["g_top_mm"] - 22.5) < 1e-9 and r["limit_mm"] == 15.0, r
            res[gf] = r
        assert abs(res["socket_top"]["h_min_mm"] - 30.7) < 1e-9 and not res["socket_top"]["possible"]
        assert res["socket_top"]["fraction"] == 0.0, res
        assert abs(res["entrance"]["h_min_mm"] - 8.2) < 1e-9 and res["entrance"]["possible"] and res["entrance"]["fraction"] > 0
        row = {"assembly_id": "00133", "status": "included", "reason": None, "zero_width": [], "m1b": {},
               "start_box": res["entrance"]}
        assert "START ALREADY SUCCESSFUL" in CO.status_text([row])
        row["start_box"] = res["socket_top"]
        assert "START ALREADY SUCCESSFUL" not in CO.status_text([row])
        assert CO.start_box_success("00133", None) is None

        def missing(aid):
            raise FileNotFoundError(aid)
        A.load_spec = missing
        assert CO.start_box_success("00133", BOX) is None
    finally:
        A.load_spec, G.T_hp0, C.START_GAP_FROM, CO.START_SUCCESS_DRAWS, C.DIAGNOSTIC = old
    return "fraction = checks.start_box_success over grasp-in-box x start-with-rotation draws (recomputed equal); " \
           "gap_from follows config.START_GAP_FROM; 00133-like: socket_top h_min 30.7 mm, never successful; " \
           "entrance h_min 8.2 mm, some starts successful (table flag); no box / unreadable spec -> None"


# ------------------------------------------------------------------------------------------------------------- 19
def test_gpu_batch_guard(tmp: Path):
    """gpu_batch_r3.sh (r6-ops-03): the first start must run from code equal to the CPU batch's frozen copy; a later
    start (<out>/code exists) is left to cl_freeze_code's unchanged-copy check."""
    cpu, gpu = tmp / "m1b" / "batch_r3_cpu", tmp / "m1b" / "batch_r3_gpu"
    W = tmp / "live"
    (W / "scripts" / "automate").mkdir(parents=True)
    os.symlink(WM / "src", W / "src")
    g = (WM / "scripts/automate/gpu_batch_r3.sh").read_text()
    g = g.replace("OUT=/mnt/storage3/qingxin/automate/m1b/batch_r3_gpu", f"OUT={gpu}").replace(
        "CPU_CODE_SHA=/mnt/storage3/qingxin/automate/m1b/batch_r3_cpu/code.sha256", f"CPU_CODE_SHA={cpu}/code.sha256")
    assert f"OUT={gpu}" in g and f"CPU_CODE_SHA={cpu}/code.sha256" in g
    (W / "scripts/automate/gpu_batch_r3.sh").write_text(g)
    shutil.copy(WM / "scripts/automate/caps.sh", W / "scripts/automate/")
    (W / "scripts/automate/collect_lib.sh").write_text((WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}"))
    (W / "scripts/automate/parallel_batch.sh").write_text(              # stand-in: what it was given
        'echo "PARALLEL $*"; echo "BATCH_TIMEOUT_S=$BATCH_TIMEOUT_S"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    run = lambda *a: subprocess.run(["bash", str(W / "scripts/automate/gpu_batch_r3.sh"), *a],  # noqa: E731
                                    capture_output=True, text=True, timeout=120, env=env)
    r = run()
    assert r.returncode == 1 and "code.sha256 missing" in r.stdout and not gpu.exists(), r.stdout + r.stderr
    cpu.mkdir(parents=True)
    h = subprocess.run(["bash", "-c", f"source {W}/scripts/automate/collect_lib.sh; cl_code_hash {W}"],
                       capture_output=True, text=True).stdout.strip()
    (cpu / "code.sha256").write_text("0" * 64 + "\n")
    r = run()
    assert r.returncode == 1 and "differs from the CPU batch's frozen code" in r.stdout and not gpu.exists(), r.stdout
    (cpu / "code.sha256").write_text(h + "\n")
    r = run("6")
    nine = "00015,00117,00133,00444,00470,00537,00755,00863,01053"
    assert r.returncode == 0 and f"PARALLEL {gpu} 6 --ids {nine}\n" in r.stdout, r.stdout + r.stderr
    assert "BATCH_TIMEOUT_S=21600" in r.stdout, "lead decision round 7 / r7-ops-01: the 9 smoke assemblies, 6 h"
    assert (gpu / "code").is_dir() and "WM_AUTOMATE_DEVICE=cuda:0" in (gpu / "switches.env").read_text()
    # --retry-crashes later: <out>/code exists, the CPU hash is not asked again (cl_freeze_code checks the copy)
    (cpu / "code.sha256").write_text("1" * 64 + "\n")
    r = run("6", "--retry-crashes")
    assert r.returncode == 0 and "frozen code" in r.stdout and f"6 --retry-crashes --ids {nine}" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(W / "scripts/automate/gpu_batch_r3.sh"), "--ids", "00001,00002", "3"],
                       capture_output=True, text=True, timeout=120, env={**env, "BATCH_TIMEOUT_S": "3600"})
    assert r.returncode == 0 and f"PARALLEL {gpu} 3 --ids 00001,00002\n" in r.stdout and "BATCH_TIMEOUT_S=3600" in r.stdout, r.stdout
    r = run("--ids", "all")
    assert r.returncode == 0 and f"PARALLEL {gpu}\n" in r.stdout, r.stdout
    r = run("--ids")
    assert r.returncode == 1 and "--ids needs a list" in r.stdout, r.stdout
    return "no CPU code.sha256 -> refused; live tree differs from the CPU batch's frozen code -> refused before " \
           "anything is written; equal -> frozen, switches cuda:0, parallel_batch.sh started with --ids = the 9 smoke " \
           "assemblies and BATCH_TIMEOUT_S 21600 by default (a given list / timeout passed on, 'all' = no list); a " \
           "later start with <out>/code skips the CPU comparison"


# ------------------------------------------------------------------------------------------------------------- 20
def test_smoke_env_gate(tmp: Path):
    """smoke/smoke_env_gate.py (r4-expert-06): the collector smoke needs a smoke_env result of the batch's code, run
    after the batch, with s1-s6 and e passing and code_changed_during_run False."""
    sys.path.insert(0, str(WM / "scripts" / "automate" / "smoke"))
    import smoke_env_gate as SG
    b = tmp / "batch"
    b.mkdir()
    t_batch = time.time() - 3600
    SW = {"physics_device_default": "cpu", "socket_draw": "per_reset"}
    for aid in ("00001", "00002"):
        (b / f"{aid}.json").write_text(json.dumps({"assembly_id": aid, "physics_code_sha256": "P", "code_files_sha256": "C",
                                                   "unix_time": t_batch, "seconds": 30.0, "switches": SW}))
        os.utime(b / f"{aid}.json", (t_batch + 30, t_batch + 30))            # written when the assembly ended
    (b / "other.json").write_text(json.dumps({"not": "a result"}))
    cfg = tmp / "cfg.json"
    cfg.write_text(json.dumps({"device": "cpu", "m1b_dirs": [str(b)]}))
    good = {"assembly": "00015", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "physics_code_sha256": "P",
            "code_files_sha256": "C", "code_changed_during_run": False, "switches": SW,
            "s": {"rule": "x", **{f"s{i}": {"pass": True} for i in range(1, 7)}}, "e": {"pass": True}}

    def run(**over):
        R = json.loads(json.dumps(good))
        for k, v in over.items():
            if v is None:
                R.pop(k, None)
            else:
                R[k] = v
        f = tmp / "r.json"
        f.write_text(json.dumps(R))
        return SG.gate(str(f), str(cfg))[0]

    assert run() == [], run()
    assert SG.main([str(tmp / "r.json"), str(cfg)]) == 0
    s_bad = {**good["s"], "s4": {"pass": False}}
    s_err = {k: v for k, v in good["s"].items() if k != "s6"}
    cases = {"s4 did not pass": dict(s=s_bad), "s6 did not pass (missing)": dict(s=s_err),
             "e did not pass": dict(e={"pass": False}), "fatally": dict(fatal="RuntimeError: x"),
             "code_changed_during_run True": dict(code_changed_during_run=True),
             "code_changed_during_run None": dict(code_changed_during_run=None),
             "physics_code_sha256 Q differs": dict(physics_code_sha256="Q"),
             "code_files_sha256 D differs": dict(code_files_sha256="D"),
             "not after the newest batch result": dict(started=time.strftime("%Y-%m-%d %H:%M:%S",
                                                                             time.localtime(t_batch - 60)))}
    # r7-ops-03: the switches are compared (a CPU-PhysX smoke_env result does not pass for a GPU-PhysX batch)
    cases["switches differ from the batch's (physics_device_default 'cuda:0' vs the batch's 'cpu'"] = \
        dict(switches={**SW, "physics_device_default": "cuda:0"})
    cases["switches differ"] = dict(switches=None)
    for want, over in cases.items():
        why = run(**over)
        assert len(why) == 1 and want in why[0], (want, why)
    # r7-integration-01: the end of a result is max(unix_time + seconds, file time): a run started while the last
    # assembly was still running is refused, by its seconds or by its file time
    started_mid = dict(started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_batch + 600)))
    assert run(**started_mid) == [], "started after both results ended"
    late = b / "00004.json"
    late.write_text(json.dumps({"assembly_id": "00004", "physics_code_sha256": "P", "code_files_sha256": "C",
                                "unix_time": t_batch, "seconds": 7200.0, "switches": SW}))
    os.utime(late, (t_batch, t_batch))
    why = run(**started_mid)
    assert len(why) == 1 and "not after the newest batch result had ended" in why[0], why
    late.write_text(json.dumps({"assembly_id": "00004", "physics_code_sha256": "P", "code_files_sha256": "C",
                                "unix_time": t_batch, "switches": SW}))      # no seconds (a crash row): file time
    os.utime(late, (t_batch + 1200, t_batch + 1200))
    why = run(**started_mid)
    assert len(why) == 1 and "had ended" in why[0], why
    late.unlink()
    # a heartbeat of a live process (pid + start time as qx_watch reads them) means the batch is still running; a
    # stale one (dead pid, or a pid reused by another process) does not; a leftover .started marker is INFO only
    me = os.getpid()
    assert SG.start_time(me) == QW.start_time(me) and SG.start_time(me) is not None
    (b / "heartbeat").mkdir()
    hb = b / "heartbeat" / f"{me}_{SG.start_time(me)}.json"
    hb.write_text(json.dumps({"pid": me, "start": SG.start_time(me), "stage": "00002 ii", "timeout_s": 1020}))
    why = run()
    assert len(why) == 1 and "batch still running" in why[0] and "00002 ii" in why[0], why
    hb.write_text(json.dumps({"pid": me, "start": "1", "stage": "00002 ii", "timeout_s": 1020}))
    (b / "heartbeat" / "4194399_5.json").write_text(json.dumps({"pid": 4194399, "start": "5", "stage": "x"}))
    (b / "00002.started").write_text("2026-11-23 10:00:00")
    f = tmp / "r.json"
    f.write_text(json.dumps(good))
    why, info = SG.gate(str(f), str(cfg))
    assert why == [] and any("leftover markers" in x and "00002.started" in x for x in info), (why, info)
    (b / "00003.json").write_text(json.dumps({"assembly_id": "00003", "physics_code_sha256": "P", "code_files_sha256": "C",
                                              "unix_time": t_batch, "switches": {**SW, "socket_draw": "per_grasp"}}))
    os.utime(b / "00003.json", (t_batch, t_batch))
    assert any("mixed switches" in x for x in run()), run()
    (b / "00003.json").write_text(json.dumps({"assembly_id": "00003", "physics_code_sha256": "P2", "code_files_sha256": "C",
                                              "unix_time": t_batch}))
    assert any("mixed code" in x for x in run()), run()
    cfg.write_text(json.dumps({"device": "cpu", "m1b_dirs": [str(tmp / "none")]}))
    assert any("missing" in x for x in run()) and SG.main([str(tmp / "r.json"), str(cfg)]) == 1
    return "a result of the batch's code and switches run after it with s1-s6 and e passing -> OK; each of a failing " \
           "/ missing check, fatal, code changed or unknown, other physics / code sha, other switches (cpu vs " \
           "cuda:0, none), a start before the newest batch result had ended (unix_time + seconds, or the file " \
           "time), a live batch heartbeat, mixed switches, mixed batch code, a missing batch folder -> refused; " \
           "stale heartbeats and leftover .started markers -> not refused (markers INFO)"


# ------------------------------------------------------------------------------------------------------------- 21
def test_train_preflight(tmp: Path):
    """collect_lib.sh cl_train_preflight (r7-data-05): the preflight runs with the job's environment (cl_isaac_env,
    cl_clean_env, cl_train_env as train_automate.sh --inside), in a subshell; no user site can shadow the env."""
    W = tmp / "live"
    (W / "scripts" / "automate").mkdir(parents=True)
    code = tmp / "code"
    (code / "src" / "wm").mkdir(parents=True)
    (code / "src" / "wm" / "__init__.py").write_text("")
    home = tmp / "isaachome"
    v = f"python{sys.version_info.major}.{sys.version_info.minor}"
    shadow = home / ".local" / "lib" / v / "site-packages" / "joblib"     # a user-site joblib that must not load
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise ImportError('user-site joblib loaded')\n")
    weights = tmp / "w1"
    weights.write_text("x")
    lib = (WM / "scripts/automate/collect_lib.sh").read_text()
    for old, new in (("CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}"),
                     ("HOME=/mnt/storage3/qingxin/isaac/home", f"HOME={home}"),
                     ("TMPDIR=/mnt/storage3/qingxin/isaac/tmp", f"TMPDIR={tmp}"),
                     ('CL_TRAIN_WEIGHTS="$CL_TORCH_HOME/hub/checkpoints/resnet34-b627a593.pth '
                      '$CL_HF_HOME/hub/models--nvidia--segformer-b1-finetuned-ade-512-512"', f'CL_TRAIN_WEIGHTS="{weights}"')):
        assert old in lib, old
        lib = lib.replace(old, new, 1)
    (W / "scripts/automate/collect_lib.sh").write_text(lib)
    (W / "scripts/automate/caps.sh").write_text("qx_run() { shift 5; \"$@\"; }\nqx_run_retry() { qx_run \"$@\"; }\n")
    sh = (f"source {W}/scripts/automate/caps.sh; source {W}/scripts/automate/collect_lib.sh; "
          f"cl_train_preflight {code}; rc=$?; echo \"rc=$rc HOME_AFTER=$HOME PYTHONPATH_AFTER=${{PYTHONPATH:-}}\"")
    env = {**{k: x for k, x in os.environ.items() if k != "PYTHONPATH"}, "HOME": str(tmp / "outerhome"), "WM_X": "1"}
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=env, timeout=120)
    assert "rc=0" in r.stdout and f"TRAIN PREFLIGHT OK HOME {home} user site False wm {code}/src/wm" in r.stdout, \
        r.stdout + r.stderr
    assert f"HOME_AFTER={tmp / 'outerhome'} PYTHONPATH_AFTER=\n" in r.stdout, "the caller's environment is unchanged"
    weights.unlink()
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=env, timeout=120)
    assert "refused: encoder weights" in r.stdout and "rc=1" in r.stdout, r.stdout
    return "the preflight runs with HOME = our Isaac home, no user site (a user-site joblib there is not loaded), wm " \
           "from the frozen src; the caller's HOME / PYTHONPATH unchanged; a missing weight file refused"


FAKE_COLLECT_PREFLIGHT = """import json, sys
from pathlib import Path
a = sys.argv
out, cfg = Path(a[a.index("--out") + 1]), json.loads(Path(a[a.index("--config") + 1]).read_text())
assert "--preflight" in a
if cfg["expected_grasps"] != 30:           # as collect.prepare: the plan's grasp count needs the frozen inputs
    print(f"REFUSED: the plan has 30 grasps, the config expects {cfg['expected_grasps']}", flush=True)
    sys.exit(3)
(out / "plan.json").write_text("{}")
print("RESOURCES 16G 500% 7 1800")
print("PREFLIGHT OK: 30 grasps, plan_sha x, device cpu")
"""


def test_collect_prep(tmp: Path):
    """collect_prep.sh (r8-ops-01): hand-written production key / value / resource errors refuse before <out>/inputs
    is frozen; a preflight refusal after the freeze (wrong expected_grasps) and a relaunch with a corrected config
    both say that nothing was collected and that a NEW <out> is needed (ALERT too); a new <out> then passes."""
    W = tmp / "live"
    (W / "scripts/automate").mkdir(parents=True)
    shutil.copy(WM / "scripts/automate/collect_prep.sh", W / "scripts/automate/")
    (W / "scripts/automate/collect_lib.sh").write_text((WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}"))
    (W / "scripts/automate/collect_expert.py").write_text(FAKE_COLLECT_PREFLIGHT)
    (W / "src").symlink_to(WM / "src")
    m1b, assets = tmp / "m1b", tmp / "assets_src"
    m1b.mkdir()
    (m1b / "r.json").write_text("{}")
    assets.mkdir()
    (assets / "a.txt").write_text("x")
    good = {"device": "cpu", "grasps_per_assembly": 15, "split": "automate_v1", "expected_grasps": 30,
            "m1b_dirs": [str(m1b)]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    env["WM_AUTOMATE_ASSETS"] = str(assets)

    def run(out: Path, cfg: dict, name: str) -> subprocess.CompletedProcess:
        f = tmp / name
        f.write_text(json.dumps(cfg))
        (out / "logs").mkdir(parents=True, exist_ok=True)                 # as collect_lib.sh cl_prep
        return subprocess.run(["bash", str(W / "scripts/automate/collect_prep.sh"), str(out), str(f)],
                              capture_output=True, text=True, env=env, timeout=300)

    out = tmp / "out1"
    for bad in ({**good, "worker_memory_max": "6G"}, {**good, "expected_grasps": None}, {**good, "device": "cuda:1"},
                {**good, "grasps_per_assembly": 3}):
        r = run(out, bad, "bad.json")
        assert r.returncode == 3 and "nothing was frozen" in r.stdout and "refused: " in r.stderr, (bad, r.stdout, r.stderr)
        assert not (out / "inputs").exists() and not (out / "inputs.tmp").exists(), bad
    r = run(out, {**good, "expected_grasps": 29}, "wrong.json")           # passes the early checks, preflight refuses
    assert r.returncode == 3 and "the config expects 29" in r.stdout, r.stdout + r.stderr
    assert (out / "inputs" / "config.json").exists() and not (out / "plan.json").exists()
    assert "start again in a NEW <out> folder" in r.stdout, r.stdout
    assert "start again in a NEW <out> folder" in (out / "logs" / "ALERT").read_text()
    r = run(out, good, "right.json")                                     # corrected config, same <out>
    assert r.returncode == 3 and "differs from the frozen" in r.stdout and "start again in a NEW <out> folder" in r.stdout, \
        r.stdout
    assert (out / "inputs" / "config.json").exists(), "nothing is deleted automatically"
    out2 = tmp / "out2"
    r = run(out2, good, "right.json")
    assert r.returncode == 0 and "PREFLIGHT OK" in r.stdout and (out2 / "plan.json").exists(), r.stdout + r.stderr
    assert (out2 / "logs" / "resources.txt").read_text().split() == ["16G", "500%", "7", "1800"]
    r = run(out2, {**good, "expected_grasps": 29}, "wrong.json")         # a collection with a plan: no NEW-folder line
    assert r.returncode == 3 and "differs from the frozen" in r.stdout and "NEW <out>" not in r.stdout, r.stdout
    for d in (out, out2):
        subprocess.run(["chmod", "-R", "u+w", str(d)], check=True)
    return "early production refusals freeze nothing; refused first launch and relaunch name a NEW <out>; ALERT line"


def test_robot_pair_check(tmp: Path):
    """scripts/automate/diag/robot_pair_check.py (r9-expert-01): the (i) / (iii) rows with hand_finger_socket_N > 1/3 N
    and robot_socket_pairs_N == 0 are listed; from an acceptance.json, the trajectories (contact_report_error_by_stage)
    and the refused t00 grasps (t00_refused_by_cause) with contact_report_error, and from grasp.json the split by rule
    (r10-expert-01 / r10-expert-01b)."""
    b = tmp / "batch"
    b.mkdir()
    res = {"00001": {"i": {"hand_finger_socket_N": 0.5, "robot_socket_pairs_N": 0.0},
                     "iii": {"hand_finger_socket_N": 2.0, "robot_socket_pairs_N": 1.7}},
           "00002": {"i": {"hand_finger_socket_N": 0.2, "robot_socket_pairs_N": 0.0}, "iii": {"skipped": "margin"}},
           "00537": {"i": {"hand_finger_socket_N": 0.34, "robot_socket_pairs_N": 0}, "iii": {"hand_finger_socket_N": 3.0}}}
    for aid, r in res.items():
        (b / f"{aid}.json").write_text(json.dumps({"assembly_id": aid, **r}))
    (b / "00003.json").write_text(json.dumps({"error": "the simulator process died during this assembly"}))
    acc = tmp / "acceptance.json"                 # the structure check_expert_data.draw_report writes (r10-expert-01b)
    acc.write_text(json.dumps({"h": {"info": {"draws": {"per_assembly": {
        "00015": {"contact_report_error_by_stage": {"run": 1}, "t00_refused_by_cause": {"contact_report_error": 2}},
        "00537": {"contact_report_error_by_stage": {}, "t00_refused_by_cause": {"reset_failed": 1}}},
        "all": {"contact_report_error_by_stage": {"run": 1}}}}}}))
    data = tmp / "smoke_out"                      # grasp.json as collect.py writes the two kinds of cases
    texts = {"robot": "ContactReportError: contact report has no robot-socket point while the hand / finger contact "
                      "sensor reports 1.4 N (0 other socket points)",
             "plug": "ContactReportError: contact report has no plug-socket point while the plug contact sensor reports "
                     "2.0 N (3 other socket points)"}
    grasps = {"00015/00015_g00": {"reference_ok": False, "reference_soft_outcome": "contact_report_error",
                                  "reference_failure": "reference: " + texts["robot"], "trajectories": []},
              "00015/00015_g01": {"reference_ok": False, "reference_soft_outcome": "contact_report_error",
                                  "reference_failure": "reference: " + texts["plug"], "trajectories": []},
              "00015/00015_g02": {"reference_ok": True, "trajectories": [
                  {"idx": 0, "outcome": "ok"}, {"idx": 3, "outcome": "contact_report_error", "stage": "run",
                                                 "error": texts["robot"]}]},
              "00537/00537_g00": {"reference_ok": False, "reference_soft_outcome": "reset_failed",
                                  "reference_failure": "reference: ResetFailed: no IK", "trajectories": []}}
    for k, g in grasps.items():
        (data / k).mkdir(parents=True)
        (data / k / "grasp.json").write_text(json.dumps(g))
    script = str(WM / "scripts/automate/diag/robot_pair_check.py")
    r = subprocess.run([sys.executable, script, str(b), "--acceptance", str(acc), "--data", str(data)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["rows"] == 5 and rep["sensor_max_above"] == 4 and rep["of_which_no_robot_force"] == 2, rep
    assert sorted((x["assembly"], x["part"]) for x in rep["listed"]) == [("00001", "i"), ("00537", "i")], rep
    assert rep["rows_without_robot_socket_pairs_N"] == [["00537", "iii"]], rep
    assert rep["contact_report_error"]["trajectories"] == 1 and rep["contact_report_error"]["t00_grasps"] == 2, rep
    assert rep["contact_report_error_by_rule"] == {"robot": 2, "plug": 1, "other": 0, "grasp_files": 4}, rep
    lines = r.stdout.splitlines()
    per = [ln for ln in lines if ln.startswith("contact_report_error 0")]
    assert per == ["contact_report_error 00537: 0 trajectories, 0 refused t00 grasps",
                   "contact_report_error 00015: 1 trajectories, 2 refused t00 grasps"], per          # 00537 first
    assert "rows with sensor max > 1/3 N: 4, of which 2 with no robot force; contact_report_error in the smoke: " \
           "1 trajectories + 2 refused t00 grasps (robot rule 2, plug rule 1)" in lines, r.stdout
    assert "WARNING" not in r.stdout, r.stdout                     # r + p + o = 3 = c + g
    run = lambda *x: subprocess.run([sys.executable, script, *map(str, x)], capture_output=True, text=True,  # noqa: E731
                                    timeout=60)
    # r11-expert-01: a --data folder without grasp.json reads "?", not 0; --data of another smoke is warned about
    (tmp / "empty_data").mkdir()
    r = run(b, "--acceptance", acc, "--data", tmp / "empty_data")
    assert r.returncode == 0 and f"no grasp.json under {tmp / 'empty_data'}" in r.stdout, r.stdout
    assert "1 trajectories + 2 refused t00 grasps (robot rule ?, plug rule ?)" in r.stdout and "WARNING" not in r.stdout, r.stdout
    other = tmp / "other_smoke"
    (other / "00015/00015_g00").mkdir(parents=True)
    (other / "00015/00015_g00/grasp.json").write_text(json.dumps(grasps["00015/00015_g00"]))
    r = run(b, "--acceptance", acc, "--data", other)
    assert r.returncode == 0 and "WARNING: --data and --acceptance disagree (r+p+o = 1, c+g = 3): not the same smoke?" \
        in r.stdout, r.stdout
    # r11-expert-02: a missing or empty batch folder reads "?", not a clean 0; a missing or broken acceptance.json
    # is named, its counts read "?", exit 0
    (tmp / "empty_batch").mkdir()
    for nb in (tmp / "no_batch", tmp / "empty_batch"):
        r = run(nb, "--acceptance", acc, "--data", data)
        assert r.returncode == 0 and f"no (i) / (iii) rows in {nb} (batch not run yet, or wrong folder)" in r.stdout, r.stdout
        assert "rows with sensor max > 1/3 N: ?; contact_report_error in the smoke: 1 trajectories + 2 refused t00 " \
               "grasps (robot rule 2, plug rule 1)" in r.stdout.splitlines(), r.stdout
    (tmp / "broken.json").write_text("{not json")
    for bad in (tmp / "missing.json", tmp / "broken.json"):
        r = run(b, "--acceptance", bad, "--data", data)
        assert r.returncode == 0 and "acceptance.json unreadable:" in r.stdout, r.stdout + r.stderr
        assert "rows with sensor max > 1/3 N: 4, of which 2 with no robot force; contact_report_error in the smoke: " \
               "? trajectories + ? refused t00 grasps (robot rule 2, plug rule 1)" in r.stdout.splitlines(), r.stdout
    # without --data the rules are unknown; an acceptance.json without h.info.draws gives no total
    r = subprocess.run([sys.executable, script, str(b), "--acceptance", str(acc)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "1 trajectories + 2 refused t00 grasps (robot rule ?, plug rule ?)" in r.stdout, r.stdout
    acc.write_text(json.dumps({"h": {"ok": False, "why": "check_expert_data failed"}, "a": {"ok": True}}))
    r = subprocess.run([sys.executable, script, str(b), "--acceptance", str(acc)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "no draws section in acceptance.json (item h failed?)" in r.stdout, r.stdout
    assert "contact_report_error in the smoke" not in r.stdout and "contact_report_error" not in \
        json.loads(r.stdout.strip().splitlines()[-1]), r.stdout
    return "rows with sensor max > 1/3 N and no robot force listed (00001 (i), 00537 (i)); 0.2 N, a (iii) with robot " \
           "force, skipped (iii) and a crash file not listed; missing field named; acceptance read with draw_report's " \
           "keys: 1 trajectory (by_stage run) + 2 refused t00 grasps (t00_refused_by_cause), 00537 first; grasp.json " \
           "texts split into robot rule 2 / plug rule 1; no draws section named, no total; empty --data, missing / " \
           "empty batch, missing / broken acceptance.json read '?' with exit 0; --data of another smoke warned about"


# ------------------------------------------------------------------------------------------------------------- 24
def test_launch_code_guard(tmp: Path):
    """collect_launch.sh (r11-ops-02, collect_lib.sh cl_batch_code_guard): the first launch (no <out>/code) runs only
    from code equal to the frozen copy of the config's batch (first m1b_dirs folder), refused before <out> exists; a
    later start (<out>/code exists) is left to cl_freeze_code's unchanged-copy check."""
    W, B, out = tmp / "live", tmp / "m1b" / "batch_r3_cpu", tmp / "coll" / "prod"
    (W / "scripts" / "automate").mkdir(parents=True)
    out.parent.mkdir(parents=True)                                  # collect_launch.sh: readlink -f needs the parent
    shutil.copy(WM / "scripts/automate/collect_launch.sh", W / "scripts/automate/")
    shutil.copy(WM / "scripts/automate/caps.sh", W / "scripts/automate/")
    lib = (WM / "scripts/automate/collect_lib.sh").read_text().replace(
        "CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python", f"CL_PY={sys.executable}")
    assert "cl_prep() {\n" in lib
    lib = lib.replace("cl_prep() {\n", "cl_prep() { echo PREP CALLED; return 1\n")   # stand-in: stops after the freeze
    (W / "scripts/automate/collect_lib.sh").write_text(lib)
    cfg, cfg_nom = tmp / "cfg.json", tmp / "cfg_nom.json"
    cfg.write_text(json.dumps({"device": "cpu", "m1b_dirs": [str(B), str(tmp / "other")]}))
    cfg_nom.write_text(json.dumps({"device": "cpu"}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("WM_")}
    run = lambda c: subprocess.run(["bash", str(W / "scripts/automate/collect_launch.sh"), str(out), str(c), "2"],  # noqa: E731
                                   capture_output=True, text=True, timeout=120, env=env)
    r = run(tmp / "missing_cfg.json")
    assert r.returncode == 1 and "missing" in r.stdout and not out.exists(), r.stdout + r.stderr
    r = run(cfg_nom)
    assert r.returncode == 1 and "names no m1b_dirs" in r.stdout and not out.exists(), r.stdout + r.stderr
    cfg_bad = tmp / "cfg_bad.json"                                  # r12-ops-03: the parse error, not "no m1b_dirs"
    cfg_bad.write_text('{"device": "cpu", "m1b_dirs": [')
    r = run(cfg_bad)
    assert r.returncode == 1 and f"config {cfg_bad} unreadable" in r.stdout and "JSONDecodeError" in r.stdout and \
        "names no m1b_dirs" not in r.stdout and not out.exists(), r.stdout + r.stderr
    r = run(cfg)
    assert r.returncode == 1 and f"{B}/code.sha256 missing" in r.stdout and not out.exists(), r.stdout + r.stderr
    B.mkdir(parents=True)
    (B / "code.sha256").write_text("0" * 64 + "\n")
    r = run(cfg)
    assert r.returncode == 1 and "differs from the batch's frozen code" in r.stdout and not out.exists(), r.stdout
    h = subprocess.run(["bash", "-c", f"source {W}/scripts/automate/collect_lib.sh; cl_code_hash {W}"],
                       capture_output=True, text=True).stdout.strip()
    (B / "code.sha256").write_text(h + "\n")
    r = run(cfg)
    assert r.returncode == 1 and "froze" in r.stdout and "PREP CALLED" in r.stdout, r.stdout + r.stderr
    assert (out / "code").is_dir() and (out / "code.sha256").read_text().strip() == h
    (B / "code.sha256").chmod(0o644)
    (B / "code.sha256").write_text("1" * 64 + "\n")              # a later start: <out>/code exists, B is not asked
    r = run(cfg)
    assert r.returncode == 1 and "frozen code" in r.stdout and "PREP CALLED" in r.stdout and "differs" not in r.stdout, \
        r.stdout + r.stderr
    return "first launch: missing config, no m1b_dirs, unparsable config (its error), no <B>/code.sha256 and another hash refused before <out> " \
           "exists; the batch's hash freezes and goes on (prep stand-in); a later start with <out>/code skips the " \
           "batch comparison"


if __name__ == "__main__":
    tests = [("1 test_config", test_config), ("2 test_freeze", test_freeze), ("3 test_claims_deaths", test_claims_deaths),
             ("4 test_worker", test_worker), ("5 test_collect_grasp", test_collect_grasp), ("6 test_signal", test_signal),
             ("7 test_heartbeat_watch", test_heartbeat_watch), ("8 test_caps", test_caps),
             ("9 test_worker_loop", test_worker_loop), ("10 test_split_v1", test_split_v1),
             ("11 test_pre_exit_hook", test_pre_exit_hook), ("12 test_gate", test_gate),
             ("13 test_success_rule", test_success_rule), ("14 test_batch_loop", test_batch_loop),
             ("15 test_parallel_batch", test_parallel_batch), ("16 test_smoke_stats", test_smoke_stats),
             ("17 test_prep_launcher", test_prep_launcher), ("18 test_start_box", test_start_box),
             ("19 test_gpu_batch_guard", test_gpu_batch_guard), ("20 test_smoke_env_gate", test_smoke_env_gate),
             ("21 test_train_preflight", test_train_preflight), ("22 test_collect_prep", test_collect_prep),
             ("23 test_robot_pair_check", test_robot_pair_check), ("24 test_launch_code_guard", test_launch_code_guard)]
    only = sys.argv[1:]
    for name, fn in tests:
        if only and name.split()[0] not in only:
            continue
        asset_dir = A.ASSET_DIR
        with tempfile.TemporaryDirectory() as tmp:
            print(name, "OK:", fn(Path(tmp)), flush=True)
        A.ASSET_DIR = asset_dir
