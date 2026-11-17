"""The parts of scripts/eval/eval_policy320.py that need no simulator: grasp selection, the cluster standard error,
the accounting of placement failures / unusable grasps / crashes, when a test counts as complete.

    python world_model/tests/test_eval_stats.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
import eval_policy320 as E  # noqa: E402


def test_pick():
    assert E.pick(297, None) == list(range(297)) and E.pick(10, 50) == list(range(10))
    p = E.pick(297, 40)
    assert len(p) == 40 and p[0] == 0 and p[-1] == 296 and max(np.diff(p)) <= 8, "evenly spread, the last grasp included"
    return "40 of 297: first, last and even spacing"


def test_cluster_rate():
    rng = np.random.default_rng(0)
    est, ps = [], []
    for _ in range(1500):                         # 100 grasps x 2 starts, the grasp decides most of the outcome
        q = rng.beta(0.6, 0.9, 100)
        s = rng.random((100, 2)) < q[:, None]
        r = E.cluster_rate(s.reshape(-1), np.repeat(np.arange(100), 2))
        est.append(r["se"]); ps.append(r["success"])
    true_sd, naive = float(np.std(ps)), float(np.sqrt(np.mean(ps) * (1 - np.mean(ps)) / 200))
    assert abs(np.mean(est) / true_sd - 1) < 0.08, (np.mean(est), true_sd)
    assert naive < 0.9 * true_sd, "the formula for independent starts would be too small here"
    one = E.cluster_rate([1, 0, 1], ["a", "a", "a"])
    assert one["success"] == 2 / 3 and one["se"] is None and E.cluster_rate([], [])["success"] is None
    return f"estimated {np.mean(est):.4f} against the real spread {true_sd:.4f} (independent-starts formula: {naive:.4f})"


def test_accounting():
    tmp = Path(tempfile.mkdtemp(prefix="e320_"))
    try:
        def g(name, port="G00", pitch=-30.0, pos=-0.015):
            return {"grasp": name, "port": port, "pitch_deg": pitch, "grasp_pos_m": pos}

        def write(gr, body):
            f = tmp / gr["port"] / f"{gr['grasp']}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"grasp": gr["grasp"], "port": gr["port"], **body}))

        att = lambda ok: {"attempt": {"success": ok, "stopped": "stalled"}}     # noqa: E731
        gs = [g("a"), g("b", pitch=-40.0), g("c", port="G46"), g("d", port="G46"), g("e"), g("f")]
        write(gs[0], {"starts": [att(True), att(False)]})
        write(gs[1], {"starts": [att(True), {"placement_failed": True}]})
        write(gs[2], {"starts": [att(True), att(True)]})
        write(gs[3], {"starts": [att(False), att(False)]})
        write(gs[4], {"starts": [], "error": "reference insertion failed", "error_kind": "setup"})
        write(gs[5], {"starts": [att(True)], "error": "RuntimeError: x", "error_kind": "crash"})
        s, complete = E.summarise(tmp, gs, 2, {})
        assert not complete and s["crashed_grasps"] == 1 and s["unusable_grasps"] == 1 and s["placement_failed_starts"] == 1
        assert s["planned_starts"] == 12 and s["judged_starts"] == 7, "the crashed grasp's attempts are not counted"
        assert s["groups"]["all"]["success"] == 4 / 7 and s["groups"]["unseen_shape"]["clusters"] == 1 and s["groups"]["unseen_shape"]["se"] is None
        assert s["groups"]["interior_strip"]["starts"] == 1 and s["groups"]["train_band"]["clusters"] == 1
        assert [E.grasp_state(tmp / x["port"] / f"{x['grasp']}.json") for x in gs] == ["done"] * 4 + ["setup", "crash"]
        (tmp / "G00" / "f.json").unlink()
        assert not E.summarise(tmp, gs, 2, {})[1], "a missing grasp is not complete either"
        write(gs[5], {"starts": [att(False), att(False)]})
        s, complete = E.summarise(tmp, gs, 2, {})
        assert complete and s["judged_starts"] == 9 and "success" in E.show(s)
        return "placement failures, unusable grasps and crashes are counted; complete only without crashes"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pool_does_not_hang():
    """45 grasps on 2 workers (more than 10 per worker): every job runs and the call returns. Without the simulator
    the workers fail to initialise, so every grasp gets a crash file: exactly what the accounting must then see."""
    tmp = Path(tempfile.mkdtemp(prefix="e320p_"))
    try:
        jobs = [({"grasp": f"g{i}", "port": "G00"}, str(tmp / "G00" / f"g{i}.json"), 1, i, 5, "free") for i in range(45)]
        args = type("A", (), {"workers": 2, "policy": "none", "jpeg_roundtrip": "on"})()
        E.run_pool(jobs, args)
        states = [E.grasp_state(tmp / "G00" / f"g{i}.json") for i in range(45)]
        assert states == ["crash"] * 45, states
        return "45 jobs on 2 workers all ran (as crashes: no simulator here) and the pool returned"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for t in (test_pick, test_cluster_rate, test_accounting, test_pool_does_not_hang):
        print(t.__name__, "OK:", t(), flush=True)
