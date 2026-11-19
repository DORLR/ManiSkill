"""CPU tests of the ablation pieces of 2026-11-19 (attempt320 "servo" / "random", scripts/analysis/ablation320.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from wm.online.common320 import RANDOM_EDGES_MM, gap_bin, servo_move  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402


def test_gap_bins():
    assert RANDOM_EDGES_MM == (0.0, 2.0, 6.0)
    assert [gap_bin(g) for g in (-1.4, -0.01, 0.0, 1.99, 2.0, 5.9, 6.0, 40.0)] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("axis", [[0.99, 0.13, -0.0002], [0.2, -0.95, 0.1], [0.0, 0.3, 0.95]])
def test_servo_move_cancels_the_lateral_label(axis):
    rng = np.random.default_rng(0)
    axis = np.asarray(axis) / np.linalg.norm(axis)
    Hs = np.eye(4)
    Hs[:3, 3] = [0.5, 0.1, 0.3]
    meta = {"H_star": Hs, "axis": axis, "pin_length_m": 0.015}
    for _ in range(20):
        hand = np.eye(4)
        hand[:3, 3] = Hs[:3, 3] + rng.normal(scale=0.003, size=3)
        lab = WD.decision_labels(meta, hand[None], np.array([0]))["probe"][0]          # along, x, y [mm], tilt
        mv = servo_move(float(lab[1]), float(lab[2]), axis, max_mm=100.0)            # no clipping
        moved = hand.copy()
        moved[:3, 3] += mv
        lab2 = WD.decision_labels(meta, moved[None], np.array([0]))["probe"][0]
        assert abs(lab2[1]) < 1e-3 and abs(lab2[2]) < 1e-3                          # lateral cancelled [mm]
        assert abs(lab2[0] - lab[0]) < 1e-3                                          # along unchanged
        short = servo_move(float(lab[1]), float(lab[2]), axis, max_mm=1.5)
        assert np.linalg.norm(short) * 1000 <= 1.5 + 1e-9
        assert np.dot(short, mv) > 0                                                 # same direction


def test_servo_move_zero():
    assert np.all(servo_move(0.0, 0.0, np.array([1.0, 0, 0]), 1.5) == 0)


def _write_arm(d: Path, rows):
    """rows: (grasp, port, pitch, pos_m, [(idx, start, success, per_decision)])."""
    for grasp, port, pitch, pos, starts in rows:
        f = d / port / f"{grasp}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"grasp": grasp, "port": port, "pitch_deg": pitch, "grasp_pos_m": pos,
                                 "starts": [{"idx": i, "start": st, "attempt": {"success": ok, "interventions": 0, "per_decision": pdl}}
                                            for i, st, ok, pdl in starts]}))


def test_pbins_and_compare(tmp_path):
    import ablation320 as A
    pd_wm = [{"k": 0, "g": 10.0, "lat": 1.0, "offered": True, "pick": 0},
             {"k": 1, "g": 3.0, "lat": 1.0, "offered": True, "pick": 4},
             {"k": 2, "g": 1.0, "lat": 1.0, "offered": True, "pick": 2},
             {"k": 3, "g": 1.0, "lat": 1.0, "offered": True, "pick": 0},
             {"k": 4, "g": -1.0, "lat": 0.2, "offered": True, "pick": 0},
             {"k": 5, "g": -2.0, "lat": 0.2, "offered": False, "pick": 0}]
    wm, ar = tmp_path / "wm", tmp_path / "arm"
    _write_arm(wm, [("G46_g00", "G46", -30.0, -0.015, [(0, {"a": 1}, True, pd_wm), (1, {"a": 2}, True, pd_wm)]),
                    ("G01_g00", "G01", -40.0, -0.015, [(0, {"a": 3}, False, pd_wm), (1, {"a": 4}, True, pd_wm)])])
    _write_arm(ar, [("G46_g00", "G46", -30.0, -0.015, [(0, {"a": 1}, False, None), (1, {"a": 9}, False, None)]),
                    ("G01_g00", "G01", -40.0, -0.015, [(0, {"a": 3}, True, None), (1, {"a": 4}, False, None)])])
    r = A.pbins(str(wm))
    assert r["offered"] == [4, 8, 4, 4] and r["corrected"] == [0, 4, 4, 0]
    assert r["p"] == [0.0, 0.5, 1.0, 0.0]
    res = A.compare(f"wm={wm}", [f"arm={ar}"])
    p = res["arms"]["arm"]["paired_vs_ref"]
    assert p["n"] == 3                                   # the start with a different pose is left out
    assert (p["arm_wins"], p["ref_wins"]) == (1, 2)
    assert res["arms"]["wm"]["moves_per_attempt"] == 2.0
    g = res["arms"]["arm"]["paired_vs_ref_by_group"]
    assert g["shape"]["n"] == 1 and g["angle"]["n"] == 2
    assert 0.0 < p["mcnemar_p"] <= 1.0
    assert A.show(res)


def test_pbins_refuses_old_runs(tmp_path):
    import ablation320 as A
    _write_arm(tmp_path / "old", [("G01_g00", "G01", -30.0, -0.015, [(0, {"a": 1}, True, None)])])
    with pytest.raises(SystemExit):
        A.pbins(str(tmp_path / "old"))


def test_counterfactual_metrics():
    import counterfactual320 as C
    R = [0.0] + [0.1 * i for i in range(1, 13)]          # candidate 12 is truly best, every copy beats the proposal
    S = [0.0] + [0.1 * i for i in range(1, 13)]          # the model ranks exactly as the truth
    p = {"g": 1.0, "pick": 12, "S": S, "G_img": S, "R": R + [0.5], "servo": True, "restore_dev_mm": 0.0, "first_step_dev_mm": 0.01}
    r = C.per_probe(p, eps=0.05)
    assert r["bin"] == 1 and r["corrected"] and r["top1"] and abs(r["rho_S"] - 1.0) < 1e-9 and r["regret"] == 0.0
    assert r["better_wm"] and not r["worse_wm"]
    assert r["better_random"] == 1.0                     # every R_i = 0.1 i beats R_0 = 0 by more than 0.05
    q = dict(p, pick=0, S=[1.0] + [0.0] * 12)
    r0 = C.per_probe(q, eps=0.05)
    assert not r0["corrected"] and r0["room"] and not r0["top1"]
    a = C.agg([r, r0])
    assert a["probes"] == 2 and a["corrected_share"] == 0.5 and a["better_wm_n"] == 1


def test_fidelity_phases_and_summary():
    import fidelity_factual320 as F
    ph = F.phase_of(np.array([5.0, 2.0, 0.0, -1.0, -5.0, -14.0]), np.array([0, 0, 0, 0, 0, 1.0]))
    assert ph.tolist() == [0, 1, 1, 2, 2, 3]
    H, per = 4, 60                                         # 60 trajectories x 6 windows: 120 face windows >= MIN_WINDOWS
    base_phase = np.array([0, 1, 1, 2, 2, 3])
    N = per * len(base_phase)
    rng = np.random.default_rng(0)
    R = {k: np.abs(rng.normal(0.5, 0.05, (N, H))) for k in ("lat_wm", "lat_enc", "al_wm", "al_enc", "al_zero", "al_cv", "rew_true", "rew_img")}
    R["lat_zero"], R["lat_cv"] = R["lat_wm"] + 1.0, R["lat_wm"] + 2.0
    R["ok"] = np.ones((N, H), bool)
    R["phase"] = np.tile(np.repeat(base_phase[None], per, 0).reshape(-1)[:, None], (1, H))
    R["prog_true"] = np.zeros((N, H))
    R["prog_wm"] = np.full((N, H), 2.0)
    R["done_lab"] = (R["phase"] == 3).astype(float)
    R["done_img"] = R["done_lab"].copy()
    R["gap_j"] = np.where(R["phase"] >= 2, -5.0, 5.0)
    R["traj"] = np.tile(np.arange(per)[:, None], (1, len(base_phase))).reshape(-1)
    R["push"] = np.ones(N, bool)
    s = F.summarise(R, H, N)
    assert s["pass"]["face"] == {"k1": True, "k2": True, "k3": True} and s["pass_overall"]
    f = s["by_phase"]["face"]
    assert f["through_board_share_k1"] == 1.0 and f["blocked_decisions"] == 2 * per
    R2 = dict(R, lat_cv=R["lat_wm"] - 0.1)                 # constant velocity better than the model: fails
    assert not F.summarise(R2, H, N)["pass_overall"]
    R3 = dict(R, ok=np.zeros((N, H), bool))                # empty cells fail, never count as a pass
    assert F.summarise(R3, H, N)["pass_overall"] is False
    x = np.r_[np.full(50, -1.0), np.full(50, -0.5)]
    assert F.boot_upper(x, np.repeat(np.arange(10), 10)) < 0


def test_cluster_signflip_and_same_start():
    import ablation320 as A
    assert A.cluster_signflip_p([]) == 1.0
    assert abs(A.cluster_signflip_p([1, 1, 1, 1]) - 2 / 16) < 1e-12            # only all + or all - reach |T| = 4
    assert abs(A.cluster_signflip_p([1, -1, 0]) - 1.0) < 1e-12
    p_mc = A.cluster_signflip_p([1.0] * 20)                                     # Monte Carlo branch: about 2 / 2^20
    assert p_mc < 1e-3
    x = {"start": "s", "hand": [0.0, 0.0, 0.0]}
    assert A.same_start(x, {"start": "s", "hand": [0.3, 0.0, 0.0]})
    assert not A.same_start(x, {"start": "s", "hand": [0.6, 0.0, 0.0]})
    assert A.same_start(x, {"start": "s", "hand": None}) and not A.same_start(x, {"start": "t", "hand": None})


def test_groups_and_combo_grasps(tmp_path):
    import ablation320 as A
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    import make_combo_grasps as M
    from wm.tdmpc import demo320 as D
    assert A.group_of("G01", -30.0, -0.015) == "none"
    assert A.group_of("G46", -40.0, -0.020) == "shape+angle+position"
    assert A.group_of("G01", -40.0, -0.020) == "angle+position"
    assert A.group_of("G55", -30.0, -0.020) == "shape+position"
    n = 26
    gs = M.make([f"G{k:02d}" for k in range(0, 40) if f"G{k:02d}" not in ("G46", "G47", "G48")], n, 0)
    assert len(gs) == 8 * n and len({g["grasp"] for g in gs}) == len(gs)
    by = {(g["condition"], g["pair_index"]): g for g in gs}
    for i in range(n):
        a, b = by[("none", i)], by[("angle", i)]
        assert a["port"] == b["port"] and a["grasp_pos_m"] == b["grasp_pos_m"] and a["episode_seed"] == b["episode_seed"]
        assert -42 < b["pitch_deg"] < -38 and not (-42 <= a["pitch_deg"] <= -38)
        c = by[("shape+position", i)]
        assert c["pitch_deg"] == a["pitch_deg"] and c["port"] == by[("shape", i)]["port"] and -21 < c["grasp_pos_m"] * 1000 < -19
    for g in gs:
        assert A.group_of(g["port"], g["pitch_deg"], g["grasp_pos_m"]) == g["condition"]
        sp = D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"])
        assert (sp in ("train", "select")) == (g["condition"] == "none") and sp != "test"
        assert -55 < g["pitch_deg"] < -25 and -28 < g["grasp_pos_m"] * 1000 < -12        # never the outer bands
        assert g["axis"] is None and g["episode_seed"] >= M.SEED0
        assert A.cell_of(g["grasp"]) == (M.CELLS.index(g["condition"]), g["pair_index"])
    assert len({g["port"] for g in gs if g["condition"] == "shape"}) == 13
    # cells(): two pairs per factor contrast, one start with a different pose left out
    d = tmp_path / "cells"
    for g in gs:
        if g["pair_index"] > 1:
            continue
        ok = g["condition"] == "none" or g["pair_index"] == 0
        f = d / g["port"] / f"{g['grasp']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"grasp": g["grasp"], "port": g["port"], "pitch_deg": g["pitch_deg"], "grasp_pos_m": g["grasp_pos_m"],
                                 "starts": [{"idx": 0, "start": {"s": g["pair_index"]}, "attempt": {"success": ok}}]}))
    r = A.cells({"wm": str(d)})["arms"]["wm"]
    assert r["cells"]["none"]["success"] == 1.0 and r["cells"]["angle"]["success"] == 0.5
    e = r["factor_effects"]["none -> angle (+angle)"]
    assert e["pairs"] == 2 and abs(e["mean_diff"] + 0.5) < 1e-12
    assert A.show_cells(A.cells({"wm": str(d)}))


def test_holm_and_t_interval():
    import ablation320 as A
    adj = A.holm([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-12 and abs(adj[2] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12
    lo, hi = A.t_ci([1, 0, 1, 1, 0, 1], ["a", "a", "b", "b", "c", "c"])
    assert 0.0 <= lo < 4 / 6 < hi <= 1.0
