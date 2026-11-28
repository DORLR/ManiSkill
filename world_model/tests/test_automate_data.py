"""AutoMate data path on a CPU (no simulator, no GPU): split, leakage and labels of the training side.
  1 split: ManiSkill entries / grasps get split320's result unchanged; AutoMate ones the stored condition
    (train / select / heldout / heldout_grasp / heldout_shape / test); "unassigned", a missing and an unknown
    condition raise SplitUndecided (a ValueError) unless allow_undecided (then "unassigned"; an unknown one still
    raises); load_entries follows it, also for an AutoMate grasp whose pitch / position lie in ManiSkill's bands;
    check_automate_conditions; group_of_entry refuses AutoMate
  2 fast copy without simulator truth: convert keeps only FAST_META_AUTOMATE of an AutoMate meta (eval_socket_pose,
    truth, peak, episode_seed, rng, disassembly_dist_m, file, tilt_definition ... dropped, eval_rel / eval_tip arrays
    not copied; expert_completed and automate_success_any kept and in the entry); entry_of refuses an AutoMate meta
    with another key; an AutoMate trajectory of condition "test" is not converted (unless allow_test); ManiSkill meta
    copied whole
  3 train_files_sha256: changes with a rewritten file (size), a changed condition, another source or another split
    identity; else stable; a manifest without a split identity gives the hash of the formula before 2026-11-23
  4 depth statistics: a held-out AutoMate trajectory is not counted; an undecided split is refused; the manifest's
    split identity, completeness and the converter's code record are carried; D_max above 3.2767 m (int16 ceiling),
    0 and nan refused before any picture is read, and by policy320.check_depth_settings; --distribution-only
    writes the same distribution and never a depth_stats.json (r10-data-01)
  5 train_wm320: online_entries refuses held-out and undecided AutoMate entries, and AutoMate entries whose
    split_sha256 is missing or differs from the fast copy's split identity; a pool entry of the other simulator
    (ManiSkill under an AutoMate fast copy, AutoMate under a ManiSkill one) is refused; ManiSkill pool entries under a
    ManiSkill fast copy unchanged
  6 labels: ManiSkill bit-identical to the formula (full_angle + last_step termination, no done_valid key); AutoMate
    default = symmetric_angle using symmetry_n (matches checks.tilt_spin across round / None / integer n on tilted
    axes; missing/invalid symmetry_n and unknown tilt_definition refused) + first_seated termination (BONUS, term=1 and
    zero terminal potential on first seated k_term, post-terminal reward=0, transition/value masks stop at n_eff while
    done_valid retains stationary post-terminal seated frames; effective_decisions filters k >= n_eff); 7-joint slice
  7 AutoMate index: only grasps with grasp.json; rows with "file" name their file, rows without (label_failed,
    placement_failed, error) are skipped; an older grasp.json without "file" by t<idx>_<kind>.npz minus the no-file
    outcomes; a listed missing file raises; split "unassigned"; expert_completed / automate_success_any carried;
    completeness (r5-data-01): a planned grasp without grasp.json of an assembly that is not skipped, a live worker
    heartbeat and a claim of a live process (pid and start time) are refused before anything is written; a skipped
    assembly's missing grasps need accept_skipped; partial indexes anyway; index_completeness.json bound to index.json
  8 encoder environment (subprocess): unset $HF_HOME, $HF_HOME in the account's home, and a SegFormer cache with
    config.json only are refused before training
  9 imitation set (option A of 2026-11-22): bc = policy kind AND outcome "inserted" AND (ManiSkill OR
    expert_completed); an AutoMate entry without expert_completed refused; ManiSkill unchanged
  10 check_expert_data: check_file on a stored trajectory with collect.py's meta: success after control steps only
    (start state excluded), success_any / success_end / first_success_step / start_success, the recorded
    automate_success_any compared; outcome_counts (success_any_not_completed, completed_not_success_end, bc,
    failure kinds) and per_assembly; draw_report (reset_failed / label_failed / t00 refusals by cause, placement redraws
    by reason, shares and the > 5 % flags, end heights with the counts <= 0.01 mm and <= 0); exit_reasons and
    counted_build_failures (the exit-code rules, start_success_rows among them); --draws-only on grasp.json fixtures
    whose t*.npz are unreadable: nothing but grasp.json / plan.json / spec files read, draws.json flags equal
    draw_report's, a start-state row flagged, band_above_top_mm and band_above_entrance_mm (r7-expert-02: listed in
    "all" above 2 mm, never a flag) from the frozen spec, contact jammed-with-success
    counted, exit 0; random_truncated (r6-expert-01: random rows stopped at the step budget or before a waypoint,
    over the random rows that ran, flagged above 5 %; draws-only carries it); contact_report_error above 5 % flagged;
    hand / finger rows by kind and waypoint redraws (r12-expert-01, decision 7): information lines, never a flag
  11 cache_wm_v1_fast on an AutoMate collection (index.build_automate, plan.json + split file): the split recomputed
    from the condition (an index "split" field saying "train" for a test grasp does not copy it), test skipped,
    --pins not needed (refused), the manifest carries the split identity; an undecided condition, a mixed index and
    a collection without a split file are refused; the manifest carries the index's completeness record (missing or
    describing another index.json: refused); ManiSkill data without --pins refused; resume: a kept file of another
    condition / plan_sha or older than its source is converted again, a reused --dst (no or another source.json) is
    refused, a split file that is not the plan's (edited list, other plan_sha) and a source of another plan refused;
    converter code (r6-data-04): manifest "code" holds this tree's md5 of cache_wm_v1_fast.CODE_FILES, source.json the
    list of writers, an earlier writer with other md5 listed under earlier_writers
  12 compare_physics_device.trajectory_numbers: completed, success_any / end, first_success_step, seat_ok;
    compare_batches (GPU vs CPU milestone-1b batches per assembly: pass (i)-(iii), depth / height differences,
    provenance apart from the device switch, the asset hashes spec / plug_obj / socket_obj included, r6-ops-03) and
    the collections' split-document check
  13 launch side: train_wm320.cap_gpu_memory and gpu_check_encoders320's QX_GPU_GB / --out / D_max rules, provenance
    versions, train_wm320.automate_provenance (fast_copy_code from the manifest; no offset label: the checkpoint's offset_mode), one MIN_SELECT_GRASPS, build_index.py --automate (unfinished refused with exit 2, --partial recorded), the
    up-front AutoMate refusals of eval_policy320 / eval_ppo_state / make_combo_grasps / ablation320
Needs numpy, scipy, PIL, torch (the Isaac Lab environment has them).   python world_model/tests/test_automate_data.py
"""

import json
import os
import pwd
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts" / "train"))
sys.path.insert(0, str(HERE.parent / "scripts" / "data"))
sys.path.insert(0, str(HERE.parent / "scripts" / "automate"))

from wm.data import index as IX  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402


def raises(f, exc=Exception, text: str = "") -> str:
    try:
        f()
    except exc as e:
        msg = str(e.code) if isinstance(e, SystemExit) else str(e)
        assert text in msg, msg
        return msg
    raise AssertionError(f"no {exc.__name__}")


def am(grasp="00015_g00", condition="train", **kw) -> dict:
    return {"grasp": grasp, "port": grasp[:5], "condition": condition, "simulator": "automate", "pitch_deg": -40.0,
            "grasp_pos_m": -0.020, "kind": "insert", "outcome": "inserted", "expert_completed": True, "steps": 40,
            "n_images": 3, **kw}


# ------------------------------------------------------------------------------------------------------------- 1
def test_split():
    g = np.random.default_rng(0)
    ports = ["A", "B", "G3", "G46", "G55", "G7", "G145"]
    n = 0
    for i in range(3000):                                  # ManiSkill: split320 unchanged, bands included
        e = {"port": ports[i % len(ports)], "pitch_deg": float(g.uniform(-60, -20)), "grasp_pos_m": float(g.uniform(-0.03, -0.01)),
             "grasp": f"G{i}_g{i % 7:02d}", "condition": "train"}
        assert D.split_of_entry(e) == D.split320(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"])
        assert D.group_of_entry(e) == D.group320(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"])
        n += 1
    for c, s in (("train", "train"), ("select", "select"), ("heldout", "heldout"), ("heldout_grasp", "heldout"),
                 ("heldout_shape", "heldout"), ("test", "test")):
        assert D.split_of_entry(am(condition=c)) == s, c
    # an AutoMate grasp in ManiSkill's held-out bands (pitch -40, position -20 mm) is still "train" by its condition
    assert D.split320("00015", -40.0, -0.020, "00015_g00") == "heldout" and D.split_of_entry(am()) == "train"
    for bad in (am(condition="unassigned"), {k: v for k, v in am().items() if k != "condition"}, am(condition="weird")):
        raises(lambda: D.split_of_entry(bad), D.SplitUndecided)
        assert issubclass(D.SplitUndecided, ValueError)
    assert D.split_of_entry(am(condition="unassigned"), allow_undecided=True) == "unassigned"
    raises(lambda: D.split_of_entry(am(condition="weird"), allow_undecided=True), D.SplitUndecided, "unknown")
    raises(lambda: D.group_of_entry(am()), ValueError, "no reporting groups")
    D.check_automate_conditions([{"grasp": "A_g00", "port": "A"}, am()])
    for bad in (am(condition="unassigned"), am(condition="weird")):
        raises(lambda: D.check_automate_conditions([am(), bad]), SystemExit)
    tmp = Path(tempfile.mkdtemp(prefix="am_split_"))
    ents = [am("00015_g00", "heldout_grasp"), am("00015_g01", "train"), am("00016_g00", "select"), am("00017_g00", "test")]
    (tmp / "manifest.json").write_text(json.dumps({"entries": ents}))
    assert [e["split"] for e in D.load_entries(tmp)] == ["heldout", "train", "select", "test"]
    (tmp / "manifest.json").write_text(json.dumps({"entries": ents + [am("00018_g00", "unassigned")]}))
    raises(lambda: D.load_entries(tmp), ValueError, "no decided split")
    assert D.load_entries(tmp, allow_undecided=True)[-1]["split"] == "unassigned"
    return f"{n} ManiSkill grasps: split320 / group320 unchanged; AutoMate: stored condition, undecided / unknown refused"


# ------------------------------------------------------------------------------------------------------------- 2
def raw_trajectory(path: Path, meta: dict, n_img: int = 3, T: int = 16):
    g = np.random.default_rng(1)
    depth = g.integers(0, 5000, (n_img, 320, 320)).astype(np.uint16)
    jpeg = np.frombuffer(b"\xff\xd8" + bytes(100) * n_img, np.uint8)
    np.savez(path, meta=np.asarray(json.dumps(meta)), img_step=np.arange(n_img) * 4, depth=depth, rgb_jpeg=jpeg,
             rgb_offsets=np.linspace(0, len(jpeg), n_img + 1).astype(np.int64), qpos=g.normal(size=(T + 1, 9)),
             hand=np.tile(np.eye(4), (T + 1, 1, 1)), action=g.normal(size=(T, 7)), eval_rel=g.normal(size=(T + 1, 7)),
             eval_tip=g.normal(size=(T + 1, 5)))


def test_fast_meta(tmp: Path):
    truthy = {"eval_socket_pose": [[1, 0, 0, 0.03]], "truth": {"depth_left_mm": 0.1, "seated": True}, "episode_seed": 7,
              "seed_offset": 3, "rng": {"entropy": 0}, "peak": {"plug_socket_N": 3.0}, "stages": [], "start_error_mm": 0.1,
              "depth_left_mm": 0.2, "placement_failures": [], "tip_centre_m": [0, 0, -0.02], "wall_s": 3.0,
              "physics_device": "cuda:0", "eval_tip_columns": ["depth_left_mm"], "offset": {}, "waypoints": None,
              "disassembly_dist_m": 0.03, "file": "t01_insert.npz", "tilt_definition": "full_angle",
              "success": {"inserted_any_step": True, "first_success_step": 3}}
    keep = {"grasp": "00015_g00", "port": "00015", "condition": "train", "kind": "insert", "outcome": "inserted", "steps": 16,
            "idx": 1, "H_star": np.eye(4).tolist(), "axis": [0, 0, -1.0], "pin_length_m": 0.02, "pitch_deg": 3.0,
            "grasp_pos_m": 0.001, "spin_deg": 2.0, "u": {"dtheta": 0.5}, "simulator": "automate", "symmetry_n": 4,
            "start": {"g0": 10}, "schema": 1, "assembly_id": "00015", "config_sha": "c", "plan_sha": "p",
            "expert_completed": False, "automate_success_any": True}
    raw = tmp / "raw.npz"
    raw_trajectory(raw, {**keep, **truthy})
    e = D.convert(raw, tmp / "fast" / "t.npz", 0.02, "t.npz")
    with np.load(tmp / "fast" / "t.npz") as z:
        meta, files = json.loads(str(z["meta"])), set(z.files)
    assert set(meta) <= set(D.FAST_META_AUTOMATE), sorted(set(meta) - set(D.FAST_META_AUTOMATE))
    assert not set(truthy) & set(meta), sorted(set(truthy) & set(meta))
    assert {k: meta[k] for k in keep} == keep and not any(f.startswith("eval_") for f in files), files
    assert e["simulator"] == "automate" and e["condition"] == "train"
    assert e["expert_completed"] is False and e["automate_success_any"] is True
    raises(lambda: D.entry_of({**meta, "eval_socket_pose": 1}, "t.npz", 3), ValueError, "allow-list")
    raises(lambda: D.entry_of({**meta, "tilt_definition": "full_angle"}, "t.npz", 3), ValueError, "allow-list")
    raw_trajectory(tmp / "test.npz", {**keep, **truthy, "condition": "test"})
    raises(lambda: D.convert(tmp / "test.npz", tmp / "fast" / "x.npz", 0.02, "x.npz"), ValueError, "condition 'test'")
    assert not (tmp / "fast" / "x.npz").exists()
    assert D.convert(tmp / "test.npz", tmp / "fast" / "x.npz", 0.02, "x.npz", allow_test=True)["condition"] == "test"
    ms = {k: v for k, v in {**keep, **truthy}.items() if k not in ("simulator",)}             # ManiSkill: whole meta
    raw_trajectory(tmp / "ms.npz", ms)
    D.convert(tmp / "ms.npz", tmp / "fast" / "m.npz", 0.02, "m.npz")
    with np.load(tmp / "fast" / "m.npz") as z:
        mm = json.loads(str(z["meta"]))
    assert mm == {**ms, "pin_length_m": 0.02, "format": D.FORMAT, "source": "m.npz"}
    em = D.entry_of(mm, "m.npz", 3)
    assert not set(em) & {"simulator", "expert_completed", "automate_success_any"}, em
    return (f"AutoMate fast meta {len(meta)} keys, none of {sorted(truthy)}; expert_completed / automate_success_any "
            "kept; no eval_* array; condition test not converted; ManiSkill meta and entry unchanged")


# ------------------------------------------------------------------------------------------------------------- 3
def test_hash(tmp: Path):
    root = tmp / "h"
    root.mkdir()
    for i in range(3):
        (root / f"t{i}.npz").write_bytes(bytes(100 + i))
    ents = [am(f"00015_g0{i}", path=f"t{i}.npz") for i in range(3)]
    (root / "manifest.json").write_text(json.dumps({"source": "/a", "entries": ents}))
    h0 = D.train_files_sha256(ents, root)
    assert h0 == D.train_files_sha256(list(reversed(ents)), root)
    (root / "t1.npz").write_bytes(bytes(150))
    h1 = D.train_files_sha256(ents, root)
    ents2 = [dict(ents[0], condition="select")] + ents[1:]
    h2 = D.train_files_sha256(ents2, root)
    (root / "manifest.json").write_text(json.dumps({"source": "/b", "entries": ents}))
    h3 = D.train_files_sha256(ents, root)
    import hashlib                                           # without a split identity: the formula of before
    rows = sorted([e["path"], int(e["n_images"]), int(e["steps"]), e.get("condition"), (root / e["path"]).stat().st_size]
                  for e in ents)
    assert h3 == hashlib.sha256(json.dumps({"source": "/b", "entries": rows}).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"source": "/b", "entries": ents, "split": {"sha256": "s1"}}))
    h4 = D.train_files_sha256(ents, root)
    (root / "manifest.json").write_text(json.dumps({"source": "/b", "entries": ents, "split": {"sha256": "s2"}}))
    h5 = D.train_files_sha256(ents, root)
    assert len({h0, h1, h2, h3, h4, h5}) == 6
    return "rewritten file, changed condition, other source, other split file: six different hashes; no split = old formula"


# ------------------------------------------------------------------------------------------------------------- 4
def depth_stats_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("depth_stats", HERE.parent / "scripts" / "automate" / "depth_stats.py")
    DS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DS)
    return DS


def test_depth_stats(tmp: Path):
    DS = depth_stats_module()
    root = tmp / "ds"
    root.mkdir()
    g = np.random.default_rng(2)
    pics = {}
    ents = []
    for name, cond, lo in (("tr", "train", 700), ("ho", "heldout_grasp", 4000)):
        d = [g.integers(lo, lo + 500, (320, 320)).astype(np.uint16) for _ in range(2)]
        blobs = [D.encode_depth(x) for x in d]
        np.savez(root / f"{name}.npz", depth_png=np.concatenate(blobs), depth_offsets=np.cumsum([0] + [len(b) for b in blobs]))
        pics[name] = d
        ents.append(am(f"0001{len(ents)}_g00", cond, path=f"{name}.npz", n_images=2, pin_length_m=0.02))
    comp = {"planned": 2, "with_grasp_json": 2, "partial": False}
    code = {"root": "/x/code", "md5": {"src/wm/tdmpc/demo320.py": "d" * 32}, "earlier_writers": []}
    (root / "manifest.json").write_text(json.dumps({"entries": ents, "split": {"name": "automate_v1", "sha256": "abc"},
                                                    "completeness": comp, "code": code}))
    st = DS.compute(str(root), 0.5)
    assert st["split"] == {"name": "automate_v1", "sha256": "abc"} and st["completeness"] == comp, st
    assert st["code"] == code, st                                     # r6-data-04: the converter's record carried
    v = np.concatenate([x.ravel() for x in pics["tr"]]).astype(np.float64) * 1e-4
    assert st["n_entries"] == 1 and st["n_valid_pixels"] == v.size and abs(st["depth_mean_m"] - v.mean()) < 1e-12
    assert st["train_files_sha256"] == D.train_files_sha256([e for e in D.load_entries(root) if e["split"] == "train"], root)
    # r9-data-01: the unclipped distribution of the training pictures (own file, and in depth_stats.json)
    raw = np.concatenate([x.ravel() for x in pics["tr"]])
    dd = json.loads((root / "depth_distribution.json").read_text())
    assert dd["n_valid"] == raw.size and dd["train_files_sha256"] == st["train_files_sha256"] and dd["split"] == st["split"]
    for q, k in ((50, "p50_m"), (90, "p90_m"), (95, "p95_m"), (99, "p99_m"), (99.9, "p99_9_m")):
        assert abs(dd[k] - float(np.percentile(raw, q, method="inverted_cdf")) * 1e-4) < 1e-12, (k, dd[k])
    assert dd["share_above_m"] == {"0.3": 0.0, "0.5": 0.0, "1": 0.0, "3.2767": 0.0}, dd
    assert st["distribution_unclipped"] == {k: v for k, v in dd.items() if k not in ("format", "data", "created", "what")}
    # r10-data-01: distribution only (no D_max): the same numbers, only depth_distribution.json, never depth_stats.json
    def main_with(*argv):
        old, sys.argv = sys.argv, ["depth_stats.py", *argv]
        try:
            DS.main()
        finally:
            sys.argv = old
    (root / "depth_distribution.json").unlink()
    main_with("--data", str(root), "--distribution-only")
    d2 = json.loads((root / "depth_distribution.json").read_text())
    assert {k: v for k, v in d2.items() if k != "created"} == {k: v for k, v in dd.items() if k != "created"}, d2
    assert not (root / "depth_stats.json").exists(), "distribution-only wrote a statistics file"
    assert DS.compute(str(root), None, distribution_only=True)["n_valid"] == dd["n_valid"]
    raises(lambda: DS.compute(str(root), 0.5, distribution_only=True), SystemExit, "takes no D_max")
    raises(lambda: main_with("--data", str(root), "--distribution-only", "--depth-max-m", "0.5"), SystemExit,
           "takes no --depth-max-m")
    raises(lambda: main_with("--data", str(root)), SystemExit, "--depth-max-m is required")
    assert not (root / "depth_stats.json").exists()
    # a wide distribution (0 = invalid, up to 65535 = 6.5535 m) whose statistics are refused: the file is still written
    wide = tmp / "ds_wide"
    wide.mkdir()
    d = [g.integers(0, 65536, (320, 320)).astype(np.uint16) for _ in range(2)]
    blobs = [D.encode_depth(x) for x in d]
    np.savez(wide / "tr.npz", depth_png=np.concatenate(blobs), depth_offsets=np.cumsum([0] + [len(b) for b in blobs]))
    (wide / "manifest.json").write_text(json.dumps({"entries": [am("00010_g00", "train", path="tr.npz", n_images=2,
                                                                    pin_length_m=0.02)],
                                                    "split": {"name": "automate_v1", "sha256": "abc"}}))
    keep, DS.check_depth_settings = DS.check_depth_settings, lambda *a: "refused for the test"
    try:
        raises(lambda: DS.compute(str(wide), 3.2767), SystemExit, "unusable statistics: refused for the test")
    finally:
        DS.check_depth_settings = keep
    raw = np.concatenate([x.ravel() for x in d])
    raw = raw[raw > 0]
    dw = json.loads((wide / "depth_distribution.json").read_text())
    assert dw["n_valid"] == raw.size, dw
    for q, k in ((50, "p50_m"), (90, "p90_m"), (95, "p95_m"), (99, "p99_m"), (99.9, "p99_9_m")):
        assert abs(dw[k] - float(np.percentile(raw, q, method="inverted_cdf")) * 1e-4) < 1e-12, (k, dw[k])
    for x, key in ((0.3, "0.3"), (0.5, "0.5"), (1.0, "1"), (3.2767, "3.2767")):
        assert abs(dw["share_above_m"][key] - float(np.mean(raw > round(x * 1e4)))) < 1e-12, (key, dw)
    assert 0.4 < dw["share_above_m"]["3.2767"] < 0.6, dw
    # r5-data-new-01: D_max above the int16 ceiling (3.2767 m) refused before any picture is read; 3.2767 itself passes
    from wm.tdmpc import policy320 as P
    calls = []
    keep, DS.D.decode_depth = DS.D.decode_depth, lambda *a: calls.append(a)
    try:
        for bad in (3.3, 6.5535, 0.0, float("nan")):
            raises(lambda: DS.compute(str(root), bad), SystemExit, "must lie in (0, 3.2767]")
    finally:
        DS.D.decode_depth = keep
    assert not calls, "pictures read before the D_max refusal"
    assert "int16 ceiling" in P.check_depth_settings(3.3, 0.3, 0.06, -6.0)
    assert P.check_depth_settings(P.DEPTH_MAX_M, 0.3, 0.06, -6.0) is None
    (root / "manifest.json").write_text(json.dumps({"entries": ents + [am("00019_g00", "unassigned", path="tr.npz")]}))
    raises(lambda: DS.compute(str(root), 0.5), SystemExit, "no decided split")
    return (f"held-out trajectory not counted (mean {st['depth_mean_m']:.4f} m = training pixels only); unclipped "
            "distribution (n_valid, p50-p99.9 = numpy inverted_cdf, shares above 0.3 / 0.5 / 1 / 3.2767 m) written to "
            "depth_distribution.json also when the statistics are refused; --distribution-only writes the same "
            "numbers and no depth_stats.json, refuses a D_max, and a normal run without D_max is refused; completeness, converter code "
            "carried; D_max > 3.2767 m / 0 / nan refused before any picture is read (check_depth_settings too); undecided refused")


# ------------------------------------------------------------------------------------------------------------- 5
def test_online_entries(tmp: Path):
    import train_wm320 as T
    for cond, text in (("heldout_grasp", "split heldout"), ("unassigned", "no decided split"), ("test", "split test")):
        pool = tmp / f"pool_{cond}"
        (pool / "r0").mkdir(parents=True)
        (pool / "r0" / "DONE").write_text("")
        (pool / "r0" / "manifest.json").write_text(json.dumps({"entries": [am(condition=cond, path="r0/x.npz", split_sha256="S")]}))
        raises(lambda: T.online_entries(pool, "S"), SystemExit, text)
    pool = tmp / "pool_train"
    (pool / "r0").mkdir(parents=True)
    (pool / "r0" / "DONE").write_text("")
    (pool / "r0" / "manifest.json").write_text(json.dumps({"entries": [am(condition="train", path="r0/x.npz", split_sha256="S")]}))
    out, rounds = T.online_entries(pool, "S")
    assert len(out) == 1 and out[0]["split"] == "train" and rounds == ["r0"] and Path(out[0]["path"]).is_absolute()
    cwd0 = os.getcwd()
    try:
        os.chdir(tmp)
        out_rel, _ = T.online_entries(Path("pool_train"), "S")
        assert len(out_rel) == 1 and Path(out_rel[0]["path"]).is_absolute() and out_rel[0]["path"] == out[0]["path"]
    finally:
        os.chdir(cwd0)
    raises(lambda: T.online_entries(pool, "other"), SystemExit, "carries the split S")       # another split identity
    raises(lambda: T.online_entries(pool), SystemExit, "is AutoMate, the fast copy is ManiSkill")   # ManiSkill fast copy
    (pool / "r0" / "manifest.json").write_text(json.dumps({"entries": [am(condition="train", path="r0/x.npz")]}))
    raises(lambda: T.online_entries(pool, "S"), SystemExit, "carries the split None")        # an entry without it
    ms = tmp / "pool_ms"                                  # ManiSkill pool entries: no split key needed (unchanged)
    (ms / "r0").mkdir(parents=True)
    (ms / "r0" / "DONE").write_text("")
    (ms / "r0" / "manifest.json").write_text(json.dumps({"entries": [{"path": "r0/x.npz", "port": "G3", "grasp": "G3_g01",
                                                                      "pitch_deg": -30.0, "grasp_pos_m": -0.01, "steps": 40,
                                                                      "outcome": "seated_fk"}]}))
    out, _ = T.online_entries(ms)
    assert len(out) == 1 and out[0]["bc"], out
    raises(lambda: T.online_entries(ms, "S"), SystemExit, "is ManiSkill, the fast copy is AutoMate")  # r7-data-01
    return ("online pool: held-out, test and undecided AutoMate entries refused, a training entry kept only with the fast "
            "copy's split sha256 (another and a missing one refused); a pool of the other simulator refused both ways "
            "(an AutoMate entry under a ManiSkill fast copy, a ManiSkill entry under an AutoMate one); ManiSkill "
            "entries under a ManiSkill fast copy unchanged")


# ------------------------------------------------------------------------------------------------------------- 6
def test_labels():
    from scipy.spatial.transform import Rotation
    g = np.random.default_rng(3)
    hand = np.tile(np.eye(4), (6, 1, 1))
    hand[:, :3, :3] = Rotation.from_rotvec(g.normal(size=(6, 3)) * 0.05).as_matrix()
    hand[:, :3, 3] = g.normal(size=(6, 3)) * 0.003
    Hs = np.eye(4)
    Hs[:3, :3] = Rotation.from_rotvec([0.1, -0.05, 0.2]).as_matrix()
    meta = {"H_star": Hs.tolist(), "pin_length_m": 0.02, "axis": [0.0, 0.0, -1.0], "grasp": "x"}
    steps = np.arange(6)
    lab = WD.decision_labels(meta, hand, steps)
    R = np.einsum("ij,tjk->tik", Hs[:3, :3].T, hand[steps, :3, :3])                   # the formula of git HEAD
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    e = hand[steps, :3, 3] - Hs[:3, 3]
    B = WD._basis(np.asarray(meta["axis"]))
    along, lx, ly = (e @ B.T).T
    dist = np.linalg.norm(e, axis=1)
    ref = {"depth": np.clip(0.02 - dist, 0, 0.02), "seated": (dist < WD.SEATED_M) & (ang < WD.SEATED_DEG),
           "phi": -(np.abs(along) * 1000 + np.hypot(lx, ly) * 1000 + ang),
           "probe": np.stack([along * 1000, lx * 1000, ly * 1000, ang], 1).astype(np.float32)}
    for k in ref:
        assert np.array_equal(lab[k], ref[k]), k
    am_meta = {**meta, "simulator": "automate", "symmetry_n": 4}
    la_full = WD.decision_labels({**am_meta, "tilt_definition": "full_angle"}, hand, steps)
    assert all(np.array_equal(la_full[k], lab[k]) for k in lab)
    hs = np.tile(np.eye(4), (2, 1, 1))
    hs[:, :3, :3] = Rotation.from_euler("z", 90 + 3, degrees=True).as_matrix()     # a quarter turn + 3 deg about the axis
    m0 = {**am_meta, "H_star": np.eye(4).tolist()}
    sym4 = WD.decision_labels(m0, hs, np.arange(2))["probe"][:, 3]                # AutoMate default: symmetric_angle
    assert np.allclose(sym4, 3, atol=1e-4), sym4                                  # quarter turn removed for n=4 -> 3 deg
    sym_round = WD.decision_labels({**m0, "symmetry_n": "round"}, hs, np.arange(2))["probe"][:, 3]
    assert np.allclose(sym_round, 0, atol=1e-4), sym_round                        # round plug: pure axial spin -> 0 deg
    full = WD.decision_labels({**m0, "tilt_definition": "full_angle"}, hs, np.arange(2))["probe"][:, 3]
    assert np.allclose(full, 93, atol=1e-4), full                                 # explicit full_angle keeps 93 deg
    assert WD.TILT_DEFINITIONS == ("full_angle", "symmetric_angle") and WD.TERM_MODES == ("last_step", "first_seated")
    for m in ({**am_meta, "tilt_definition": "axis_tilt"}, {**m0, "switches": {"tilt_definition": "axis_tilt"}}):
        raises(lambda: WD.decision_labels(m, hs, np.arange(2)), ValueError, "tilt_definition")
    raises(lambda: WD.decision_labels({**meta, "simulator": "automate"}, hs, np.arange(2)), ValueError, "symmetry_n")
    for bad_sym in (0, -1, False, True, 3.5, float("nan"), "square"):
        raises(lambda s=bad_sym: WD.decision_labels({**m0, "symmetry_n": s}, hs, np.arange(2)), ValueError, "symmetry_n")
    # verify _symmetric_angle_deg against checks.tilt_spin on tilted non-unit axis and float symmetry_n (e.g. 4.0)
    from wm.sim_automate import checks as K
    tilt_axis = np.array([1.0, -2.0, 3.0])                                        # non-unit tilted insertion axis
    B_t = WD._basis(tilt_axis)
    Hs_t = np.eye(4)
    Hs_t[:3, :3] = B_t.T                                                          # col 0 = unit insertion axis
    R_loc = Rotation.from_rotvec(g.normal(size=(12, 3)) * 0.7).as_matrix()
    hand_t = np.tile(np.eye(4), (12, 1, 1))
    hand_t[:, :3, :3] = np.einsum("ij,tjk->tik", Hs_t[:3, :3], R_loc)             # rel rot in socket frame
    q_xyzw = Rotation.from_matrix(R_loc).as_quat()
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    for sym in ("round", None, 1, 2, 4, 4.0, 6):
        got = WD.decision_labels({"H_star": Hs_t.tolist(), "pin_length_m": 0.03, "axis": tilt_axis.tolist(),
                                  "grasp": "t", "simulator": "automate", "symmetry_n": sym},
                                 hand_t, np.arange(12))["probe"][:, 3]
        ref_sym = K.tilt_spin(q_wxyz, int(sym) if isinstance(sym, float) else sym)["tilt_symmetric_deg"]
        assert np.allclose(got, ref_sym, atol=1e-4), (sym, np.max(np.abs(got - ref_sym)))
    # empty steps safe
    assert WD.decision_labels(m0, hs, np.array([], np.int64))["probe"].shape == (0, 4)
    raises(lambda: WD.decision_labels({**meta, "axis": None}, hand, steps), ValueError, "no insertion axis")
    # trajectory(): 7-joint slice (ManiSkill last_step unchanged) + AutoMate first_seated termination & done_valid
    T = 16
    z = {"action": g.normal(size=(T, 7)), "img_step": np.arange(0, T + 1, 4), "qpos": g.normal(size=(T + 1, 9)),
         "hand": np.tile(np.eye(4), (T + 1, 1, 1))}
    tr = WD.trajectory(z, {**meta, "kind": "insert", "H_star": np.eye(4).tolist()}, 0.3, 0.967)
    assert tr["q"].shape == (3, 7) and np.array_equal(tr["q"], z["qpos"][[0, 8, 16], :7].astype(np.float32))
    assert "term_mode" not in tr and tr["term"].tolist() == [0.0, 1.0]            # ManiSkill: last_step unchanged
    # AutoMate n=6 trajectory: first seated at d=3 (k_term=2), stays seated at d=4,5, bounces out at d=6
    n6 = 6
    T6 = 8 * n6
    hand6 = np.tile(np.eye(4), (T6 + 1, 1, 1))
    z_pos = [0.015, 0.008, 0.003, 0.0005, 0.0001, 0.0001, 0.003]                  # d=3,4,5 seated (<1.5mm); d=6 unseated
    for d_i, zp in enumerate(z_pos):
        hand6[8 * d_i, 2, 3] = zp
    z6 = {"action": g.normal(size=(T6, 7)) * 0.01, "img_step": np.arange(0, T6 + 1, 4),
          "qpos": g.normal(size=(T6 + 1, 9)), "hand": hand6}
    am_tr_meta = {"H_star": np.eye(4).tolist(), "pin_length_m": 0.03, "axis": [0.0, 0.0, -1.0],
                  "grasp": "00015_g00", "kind": "insert", "simulator": "automate", "symmetry_n": "round"}
    assert WD.effective_decisions(z6, am_tr_meta) == 3                            # k_term=2 -> n_eff=3
    tr_am = WD.trajectory(z6, am_tr_meta, 0.3, 0.967, offset_mode="gap")
    assert tr_am["term_mode"] == "first_seated" and tr_am["end_seated"] is True
    assert tr_am["term"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert np.allclose(tr_am["reward"][3:], 0.0) and tr_am["reward"][2] > 10.0
    s0, top = np.full(WD.ACTION_DIM, 0.05, np.float32), np.full(WD.ACTION_DIM, 0.5, np.float32)
    # straddling window k=2 (k_term): transitions stop at n_eff=3, done_valid keeps stationary d=4,5 (excludes bounced d=6)
    w2 = WD.window(tr_am, 2, s0, top)
    assert w2["valid"].tolist() == [1.0, 0.0, 0.0, 0.0] and w2["term"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert w2["obs_valid"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]                  # d=2,3 only for imagined/offset
    assert w2["done_valid"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]                 # d=2,3,4,5 (d=6 bounced -> 0!)
    assert w2["v_valid"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert w2["a_next_valid"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    # exact boundary k=n_eff=3: all masks 0 (no crash / negative slice)
    w3 = WD.window(tr_am, 3, s0, top)
    assert all(w3[k].sum() == 0.0 for k in ("valid", "obs_valid", "done_valid", "v_valid", "a_next_valid"))
    # ManiSkill trajectory on identical motion keeps legacy last_step (no done_valid key)
    ms_tr = WD.trajectory(z6, {k: v for k, v in am_tr_meta.items() if k not in ("simulator", "symmetry_n")}, 0.3, 0.967)
    assert "term_mode" not in ms_tr and ms_tr["end_seated"] is False and ms_tr["term"].sum() == 0.0
    assert "done_valid" not in WD.window(ms_tr, 2, s0, top)
    # AutoMate random kind never gets first_seated term
    rnd_am = WD.trajectory(z6, {**am_tr_meta, "kind": "random"}, 0.3, 0.967, offset_mode="gap")
    assert rnd_am["term"].sum() == 0.0 and WD.effective_decisions(z6, {**am_tr_meta, "kind": "random"}) == n6
    return ("ManiSkill labels = HEAD formula bit for bit (last_step, no done_valid); AutoMate symmetric_angle matches "
            "checks.tilt_spin (tilted axis, round/None/int/float), invalid/missing symmetry_n refused; AutoMate "
            "first_seated termination + done_valid stationary seated supervision + effective_decisions verified")


# ------------------------------------------------------------------------------------------------------------- 7
def test_index(tmp: Path):
    root = tmp / "coll"
    gdir = root / "00015" / "00015_g00"
    gdir.mkdir(parents=True)
    meta = am(pin_length_m=0.021, condition="unassigned")
    for name, kind in (("t00_reference.npz", "reference"), ("t01_insert.npz", "insert"), ("t05_perturbed.npz", "perturbed")):
        raw_trajectory(gdir / name, {**meta, "kind": kind})
    rows = [{"idx": 0, "kind": "reference", "outcome": "inserted", "file": "t00_reference.npz"},
            {"idx": 1, "kind": "insert", "outcome": "inserted", "file": "t01_insert.npz"},
            {"idx": 2, "kind": "insert", "outcome": "placement_failed"},
            {"idx": 3, "kind": "insert", "outcome": "label_failed"},
            {"idx": 4, "kind": "perturbed", "outcome": "reset_failed"},
            {"idx": 5, "kind": "perturbed", "outcome": "error"}]
    (gdir / "grasp.json").write_text(json.dumps({"grasp": "00015_g00", "trajectories": rows}))
    other = root / "00016" / "00016_g00"                       # unfinished: no grasp.json
    other.mkdir(parents=True)
    raw_trajectory(other / "t00_reference.npz", {**am("00016_g00"), "kind": "reference"})
    old = root / "00017" / "00017_g00"                         # an older grasp.json without "file"
    old.mkdir(parents=True)
    raw_trajectory(old / "t00_reference.npz", {**am("00017_g00", "unassigned", pin_length_m=0.01), "kind": "reference",
                                                "expert_completed": True, "automate_success_any": True})
    (old / "grasp.json").write_text(json.dumps({"grasp": "00017_g00", "trajectories": [
        {"idx": 0, "kind": "reference", "outcome": "inserted"}, {"idx": 1, "kind": "insert", "outcome": "label_failed"},
        {"idx": 2, "kind": "insert", "outcome": "placement_failed"}, {"idx": 3, "kind": "insert", "outcome": "error"}]}))
    plan_of(root, ["00015_g00", "00016_g00", "00017_g00"])
    comp = _completeness(root)                                 # 00016_g00 unfinished: refused unless --partial
    es = IX.build_automate(root, workers=1, partial=True)
    assert sorted(Path(e["path"]).name for e in es) == ["t00_reference.npz", "t00_reference.npz", "t01_insert.npz"], es
    assert all(e["simulator"] == "automate" and e["split"] == "unassigned" for e in es)
    assert sorted(e["pin_length_m"] for e in es) == [0.01, 0.021, 0.021]
    e17 = [e for e in es if e["grasp"] == "00017_g00"][0]
    assert e17["expert_completed"] is True and e17["automate_success_any"] is True
    assert json.loads((root / "index.json").read_text()) == es
    rec = json.loads((root / IX.COMPLETENESS_FILE).read_text())
    assert rec["index_sha256"] == IX.index_sha256(root) and rec["partial"] and rec["partial_flag"], rec
    assert rec["planned"] == 3 and rec["with_grasp_json"] == 2 and rec["undone_not_skipped"] == ["00016_g00"], rec
    (gdir / "grasp.json").write_text(json.dumps({"grasp": "00015_g00", "trajectories": rows + [
        {"idx": 6, "kind": "insert", "outcome": "inserted", "file": "t06_insert.npz"}]}))
    raises(lambda: IX.build_automate(root, workers=1, partial=True), FileNotFoundError, "t06_insert.npz")
    assert json.loads((root / IX.COMPLETENESS_FILE).read_text()) == rec           # the failed build left the pair as it was
    return ("3 files indexed: rows with 'file' only (placement_failed, label_failed, reset_failed, error skipped), an "
            "older grasp.json by name minus the no-file outcomes, grasp without grasp.json left out; a listed missing file "
            f"raises; {comp}")


def plan_of(root: Path, grasps: list[str], **extra):
    """A plan.json with the planned grasps (the keys the completeness check and check_expert_data read)."""
    (root / "plan.json").write_text(json.dumps({**extra, "grasps": [
        {"grasp": g, "port": g[:5], "trajectories": [{"idx": i} for i in range(12)]} for g in grasps]}))


def _completeness(root: Path) -> str:
    """r5-data-01 / r5-ops-chk-01 (index side): build_automate refuses a running or unfinished collection before
    reading or writing anything; --accept-skipped / --partial; live heartbeats and claims by pid AND start time."""
    from wm.sim_automate import collect as AC
    raises(lambda: IX.build_automate(root, workers=1), IX.Incomplete, "00016_g00")       # not skipped, no grasp.json
    assert not (root / "index.json").exists() and not (root / IX.COMPLETENESS_FILE).exists()
    deaths = root / "deaths"                                   # 00016 skipped after DEATHS_MAX process deaths
    deaths.mkdir()
    (deaths / "00016.json").write_text(json.dumps([{"grasp": "00016_g00", "pid": i, "start": "1", "kind": "crash"}
                                                   for i in range(AC.DEATHS_MAX)]))
    m = raises(lambda: IX.build_automate(root, workers=1), IX.Incomplete, "--accept-skipped")
    assert "00016" in m and "skipped" in m, m
    rec = IX.check_completeness(root, accept_skipped=True)
    assert not rec["partial"] and rec["undone_of_skipped"] == ["00016_g00"] and "00016" in rec["skipped_assemblies"], rec
    me, start = os.getpid(), AC._start_time(os.getpid())
    hb = root / "heartbeat"                                    # a live worker (this process) and a dead one
    hb.mkdir()
    (hb / f"{me}_{start}.json").write_text(json.dumps({"pid": me, "start": start}))
    (hb / "1_x.json").write_text(json.dumps({"pid": 1, "start": "not-a-start-time"}))
    m = raises(lambda: IX.check_completeness(root, accept_skipped=True), IX.Incomplete, "still running")
    assert "1 live worker heartbeats" in m, m
    (hb / f"{me}_{start}.json").unlink()
    claim = root / "00016" / "00016_g00" / ".claim"            # a grasp held by a live process (pid + start time)
    claim.write_text(f"attempt 1 pid {me} start {start} time x")
    m = raises(lambda: IX.check_completeness(root, accept_skipped=True), IX.Incomplete, "still running")
    assert "1 grasps held by a live worker" in m, m
    claim.write_text(f"attempt 1 pid {me} start 0 time x")     # the same pid, another start time: a dead claim
    assert not IX.check_completeness(root, accept_skipped=True)["held_by_live_worker"]
    claim.unlink()
    rec = IX.check_completeness(root, partial=True)
    assert rec["partial"] is True and rec["partial_flag"], rec    # skipped grasps missing and not accepted: partial
    rec = IX.check_completeness(root, accept_skipped=True, partial=True)
    assert rec["partial"] is False and rec["accept_skipped"], rec  # accepted: not partial
    import shutil
    shutil.rmtree(deaths)
    shutil.rmtree(hb)
    return ("completeness: unfinished grasp refused, skipped assembly needs --accept-skipped, a live heartbeat / claim "
            "(pid and start time) refused, a dead one not; nothing written on refusal")


# ------------------------------------------------------------------------------------------------------------- 8
def test_encoder_env(tmp: Path):
    code = ("import sys; sys.path[:0] = [{src!r}, {tr!r}]; import train_wm320 as T; T.check_encoder_env('segformer_b1'); "
            "print('PASSED')").format(src=str(HERE.parent / "src"), tr=str(HERE.parent / "scripts" / "train"))
    home = pwd.getpwuid(os.getuid()).pw_dir
    fake = tmp / "hf"
    rev = "a775b4f5dcb2179e57006dca0e268281306412ef"
    snap = fake / "hub" / "models--nvidia--segformer-b1-finetuned-ade-512-512" / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    base = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "TORCH_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
    out = []
    for env, text in (({"TORCH_HOME": str(tmp / "t")}, "set $HF_HOME"),
                      ({"HF_HOME": str(Path(home) / ".cache" / "huggingface"), "TORCH_HOME": str(tmp / "t")}, "home"),
                      ({"HF_HOME": str(fake), "TORCH_HOME": str(tmp / "t")}, "pytorch_model.bin")):
        r = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, timeout=600)
        msg = r.stdout + r.stderr
        assert r.returncode != 0 and "PASSED" not in msg and text in msg, msg[-2000:]
        out.append(text)
    return f"refused: {out}"


# ------------------------------------------------------------------------------------------------------------- 9
def test_bc(tmp: Path):
    root = tmp / "bc"
    root.mkdir()
    ents = [am("00015_g00", kind="insert", outcome="inserted", expert_completed=True),          # imitated
            am("00015_g01", kind="perturbed", outcome="inserted", expert_completed=False),     # AutoMate success, expert failed
            am("00015_g02", kind="reference", outcome="inserted", expert_completed=True),
            am("00015_g03", kind="contact", outcome="reached_depth", expert_completed=True),
            am("00015_g04", kind="insert", outcome="expert_failed", expert_completed=False),
            {"grasp": "G3_g00", "port": "G3", "condition": "train", "pitch_deg": -50.0, "grasp_pos_m": -0.025,
             "kind": "insert", "outcome": "inserted", "steps": 40, "n_images": 3, "path": "m.npz"},  # ManiSkill: no key
            {"grasp": "G3_g01", "port": "G3", "condition": "train", "pitch_deg": -50.0, "grasp_pos_m": -0.025,
             "kind": "contact", "outcome": "inserted", "steps": 40, "n_images": 3, "path": "m2.npz"}]
    (root / "manifest.json").write_text(json.dumps({"entries": ents}))
    assert [e["bc"] for e in D.load_entries(root)] == [True, False, True, False, False, True, False]
    old = {k: v for k, v in am("00015_g00", kind="insert", outcome="inserted").items() if k != "expert_completed"}
    (root / "manifest.json").write_text(json.dumps({"entries": [old]}))
    raises(lambda: D.load_entries(root), ValueError, "expert_completed")
    (root / "manifest.json").write_text(json.dumps({"entries": [{**old, "outcome": "expert_failed"}]}))
    assert D.load_entries(root)[0]["bc"] is False                   # not inserted: no completion flag needed
    return "bc = policy kind AND inserted AND expert_completed (AutoMate); ManiSkill unchanged; missing flag refused"


# ------------------------------------------------------------------------------------------------------------ 10
def stored_trajectory(path: Path, meta: dict, heights_mm: list[float]):
    """A trajectory as collect.py stores it (wm.data.store.save), eval_rel with the plug straight above the socket at
    the given heights (identity rotation: keypoint distance = height)."""
    from wm.data import store
    from wm.sim_automate import config as C
    T, N = len(heights_mm) - 1, None
    img = list(range(0, T + 1, 4)) + ([T] if T % 4 else [])
    N = len(img)
    g = np.random.default_rng(4)
    G3 = C.R_G[:3, :3]
    rel = np.zeros((T + 1, 7))
    rel[:, 3] = 1.0
    rel[:, :3] = np.stack([G3.T @ np.array([0.0, 0.0, h * 1e-3]) for h in heights_mm])
    arrays = {"rgb": g.integers(0, 255, (N, 320, 320, 3)).astype(np.uint8),
              "depth": g.integers(1, 5000, (N, 320, 320)).astype(np.uint16), "seg": np.zeros((N, 320, 320), np.uint8),
              "cam_T_world": np.tile(np.eye(4), (N, 1, 1)), "img_step": np.asarray(img, np.int64), "K": np.eye(3),
              "qpos": g.normal(size=(T + 1, 9)) * 0.1, "hand": np.tile(np.eye(4), (T + 1, 1, 1)),
              "action": g.normal(size=(T, 7)) * 0.1, "phase": np.zeros(T, np.int8), "eval_rel": rel,
              "eval_tip": np.zeros((T + 1, 9), np.float32)}
    store.save(path, arrays, {**meta, "steps": T})
    return arrays


def test_check_expert_data(tmp: Path):
    import check_expert_data as CE
    from wm.data import store
    data = tmp / "ced"
    gdir = data / "00015" / "00015_g00"
    gdir.mkdir(parents=True)
    keep_dir, CE.A.ASSET_DIR = CE.A.ASSET_DIR, tmp / "no_assets"   # no spec file: the meta's own values (collect.py's)
    try:
        return _check_expert_data(CE, store, data, gdir) + "; " + _draws_only(CE, tmp)
    finally:
        CE.A.ASSET_DIR = keep_dir


def _check_expert_data(CE, store, data, gdir):
    base = {**am(), "assembly_id": "00015", "pin_length_m": 0.02, "tip_centre_m": [0.0, 0.0, -0.02], "symmetry_n": 4,
            "disassembly_dist_m": 0.02, "H_star": np.eye(4).tolist(), "axis": [0.0, 0.0, -1.0], "failure": None,
            "eval_tip_columns": ["success", "height_mm"], "peak": {"finger_socket_N": 0.0}}
    # start 5 mm (the rule holds at the start: never counted), 30, 30, 5 (first success after step 3), end 30 mm
    cases = {"t01_insert.npz": ({"outcome": "inserted", "expert_completed": False, "automate_success_any": True,
                                 "failure": "ExpertFailure: lateral guard: 0.4 mm"}, [5, 30, 30, 5, 30]),
             "t02_insert.npz": ({"outcome": "inserted", "expert_completed": True, "automate_success_any": True}, [30, 30, 10, 5, 0.05]),
             "t03_perturbed.npz": ({"outcome": "expert_failed", "expert_completed": True, "automate_success_any": False,
                                    "kind": "perturbed"}, [5, 30, 30, 30, 30]),
             "t00_reference.npz": ({"outcome": "inserted", "expert_completed": True, "kind": "reference"}, [30, 10, 0.04])}
    for name, (m, hs) in cases.items():
        stored_trajectory(gdir / name, {**base, "kind": "insert", **m}, hs)
    ref = {}
    for name in ("t01_insert.npz", "t00_reference.npz", "t03_perturbed.npz"):
        with np.load(gdir / name) as z:
            sig = {k: (str(z[k].dtype), z[k].ndim, z[k].shape[1:]) for k in z.files if k != "meta" and not k.startswith("eval_")}
        meta = store.read_meta(gdir / name)
        ref[meta["kind"]] = (sig, meta, name)
    files = [CE.check_file(gdir / n, data, ref, None)[0] for n in sorted(cases)]
    f = {Path(x["file"]).name: x for x in files}
    t1 = f["t01_insert.npz"]
    assert t1["keys_equal"] and t1["counts_ok"] and t1["img_step_ok"] and t1["spec_source"] == "meta", t1
    assert t1["success_any"] and not t1["success_end"] and t1["first_success_step"] == 3 and t1["start_success"], t1
    t3 = f["t03_perturbed.npz"]
    assert not t3["success_any"] and t3["start_success"] and t3["first_success_step"] is None, t3   # start never counted
    t2 = f["t02_insert.npz"]
    assert t2["success_any"] and t2["success_end"] and t2["first_success_step"] == 2 and abs(t2["h_end_mm"] - 0.05) < 1e-9
    c = CE.outcome_counts(files)
    assert c["insert_perturbed"] == 3 and c["outcome_inserted_vs_success_any_differ"] == 0, c
    assert c["recorded_vs_recomputed_success_any_differ"] == 0 and c["success_any_not_completed"] == 1, c
    assert c["completed_not_success_end"] == 1 and c["completed_not_success_any"] == 1 and c["bc"] == 1, c
    assert c["success_any_not_end"] == 1 and c["start_success"] == 2, c
    assert c["failure_kinds"] == {"ExpertFailure: lateral guard": 1}, c
    pa = CE.per_assembly(files)["00015"]
    assert pa["references"] == 1 and pa["references_completed_and_success_end"] == 1 and pa["first_success_step_min_max"] == [1, 3], pa
    assert abs(pa["min_h_end_mm_completed"] - 0.04) < 1e-9 and pa["completed_not_success_end"] == 1, pa
    d = _draws(CE)
    x = _exit_codes(CE, data)
    return (f"start state never counted (t03: start 5 mm, success_any False), first success after step 3; counts {c}; {d}; {x}")


def _grasp_json(grasp: str, ref_ok=True, ref_completed=True, ref_h=0.02, soft=None, rows=()) -> dict:
    """A grasp.json as collect.collect_grasp writes it (only the keys draw_report reads)."""
    g = {"grasp": grasp, "reference_ok": ref_ok, "reference_expert_completed": ref_completed,
         "reference_success": {"inserted_end": ref_ok, "height_mm": ref_h, "keypoint_dist_mm": 1.0},
         "trajectories": [{"idx": 0, "kind": "reference", "outcome": "inserted"}] if ref_ok else []}
    if soft:
        g.update(reference_soft_outcome=soft, reference_expert_completed=None, reference_success=None)
    g["trajectories"] += list(rows)
    return g


def _row(idx, outcome="inserted", attempts=1, failures=(), lab_h=0.02, lab_completed=True, lab_end=True, stage=None):
    t = {"idx": idx, "kind": "insert", "outcome": outcome, "socket_index": idx,
         "label_reference": {"accepted": outcome != "label_failed", "ok": lab_end, "completed": lab_completed,
                             "inserted_end": lab_end, "success": {"inserted_end": lab_end, "height_mm": lab_h}}}
    if outcome == "reset_failed":
        t.pop("label_reference")
    if outcome == "contact_report_error":
        t["stage"] = stage
    if outcome not in ("reset_failed", "label_failed", "contact_report_error") or stage == "placement":
        t.update(placement_attempts=attempts, placement_failures=list(failures))
    return t


def _draw_grasps() -> list[dict]:
    """The grasp.json fixtures of the draws tests: 00015 with every kind of refusal and redraw, 00133 clean."""
    lag_leg = {"leg": "retreat", "ok": False, "reason": "lag", "max_lag_mm": 1.6, "peak_plug_socket_N": 0.2,
               "peak_finger_socket_N": 0.1}
    desc = [{"leg": "retreat", "ok": True, "reason": None}, {"leg": "transfer", "ok": True, "reason": None},
            {"leg": "descend", "ok": False, "reason": "plug_socket_contact", "max_lag_mm": 0.4}]
    rows = [_row(1, failures=[{"stage": "retreat", "legs": [lag_leg]}], attempts=2),
            _row(2, failures=[{"stage": "descend", "legs": desc}, {"stage": "clearance", "clearance_mm": {"min": 0.5}},
                              {"stage": "start_error", "start_error_mm": 0.8, "legs": desc[:2]}],
                 attempts=3, outcome="placement_failed"),
            _row(3, outcome="reset_failed"),
            _row(4, outcome="label_failed", lab_end=False, lab_h=-0.003),
            _row(5, outcome="contact_report_error", stage="placement", attempts=1)]
    rows += [_row(i, lab_h=0.0025 + i * 1e-3) for i in range(6, 14)]             # 0.0085, 0.0095, 0.0105, ... mm
    gs = [_grasp_json("00015_g00", rows=rows),
          _grasp_json("00015_g01", ref_ok=False, ref_h=-0.002),                    # completed, h <= 0: refused
          _grasp_json("00015_g02", ref_ok=False, ref_completed=False, ref_h=3.0),  # did not complete
          _grasp_json("00015_g03", ref_ok=False, soft="reset_failed"),            # t00 reset without IK
          _grasp_json("00133_g00", rows=[_row(i) for i in range(1, 14)])]
    return gs


def _draws(CE):
    gs = _draw_grasps()
    rep, flags = CE.draw_report(gs, {"00015_g01": 13, "00015_g02": 13, "00015_g03": 13})
    a = rep["per_assembly"]["00015"]
    c = a["counts"]
    assert c["grasps"] == 4 and c["draws"] == 4 + 13 and c["reset_failed"] == 2 and c["label_failed"] == 1, c
    assert c["t00_refused"] == 3 and c["attempts"] == 2 + 3 + 1 + 8 and c["placement_rows"] == 1 + 1 + 1 + 8, c
    assert a["t00_refused_by_cause"] == {"completed, AutoMate rule fails at the end": 1, "expert did not complete": 1,
                                         "reset_failed": 1}, a["t00_refused_by_cause"]
    assert a["trajectories_lost_with_t00"] == 39 and a["refused_completed_with_h_le_0"] == 2, a
    assert a["redraws_by_reason"] == {"retreat:lag": 1, "descend:plug_socket_contact": 1, "clearance": 1, "start_error": 1}, a
    assert abs(a["shares"]["reset_failed"] - 2 / 17) < 1e-12 and abs(a["shares"]["redraw descend:plug_socket_contact"] - 1 / 14) < 1e-12
    assert a["contact_report_error_by_stage"] == {"placement": 1} and a["lag_stops"]["low_force"] == 1, a
    assert a["label_failed_by"] == {"completed=True inserted_end=False": 1}, a
    h = a["end_height_label"]
    assert h["n"] == 12 and h["le_0mm"] == 1 and h["le_0.01mm"] == 3 and abs(h["min_mm"] + 0.003) < 1e-12, h
    assert a["end_height_t00"]["le_0mm"] == 1 and a["end_height_t00"]["n"] == 3, a["end_height_t00"]
    b = rep["per_assembly"]["00133"]
    assert not any(v for v in b["shares"].values()) and not [f for f in flags if f.startswith("00133")], (b, flags)
    for name in ("reset_failed", "t00_refused", "placement_failed", "redraw descend:plug_socket_contact", "label_failed",
                 "contact_report_error"):                                 # contact_report_error 1 of 17 = 5.9 %
        assert any(f.startswith(f"00015: {name} ") for f in flags), (name, flags)
    assert "00015: contact_report_error 5.9% of 17 (> 5%)" in flags, flags
    assert a["start_success_rows"] == [] and c["start_success_rows"] == 0 and a["band_above_top_mm"] is None, a
    assert a["random_truncated"] == {"n": 0, "of": 0, "share": None}, a["random_truncated"]
    # r6-expert-01: random rows stopped at the step budget or before a waypoint, share flagged above 5 %
    rnd = [{"idx": 1, "kind": "random", "outcome": "expert_failed", "expert_completed": False,
            "failure": "ExpertFailure: step_budget_800"},
           {"idx": 2, "kind": "random", "outcome": "expert_failed", "expert_completed": False,
            "failure": "ExpertFailure: random_motion_1_did_not_reach_target"},
           {"idx": 3, "kind": "random", "outcome": "done", "expert_completed": True, "failure": None},
           {"idx": 4, "kind": "random", "outcome": "expert_failed", "expert_completed": False, "failure": "ExpertFailure: jam"},
           {"idx": 5, "kind": "random", "outcome": "placement_failed"},                  # the expert never ran
           {"idx": 6, "kind": "insert", "outcome": "expert_failed", "expert_completed": False,
            "failure": "ExpertFailure: step_budget_800"}]                                 # not a random row
    rr, rf = CE.draw_report([_grasp_json("00320_g00", rows=rnd)])
    assert rr["per_assembly"]["00320"]["random_truncated"] == {"n": 2, "of": 4, "share": 0.5}, rr["per_assembly"]["00320"]
    assert rr["all"]["random_truncated"] == {"n": 2, "of": 4, "share": 0.5}, rr["all"]
    assert rf == ["00320: random_truncated 50.0% of 4 (> 5%)"], rf
    ok = [{**t, "outcome": "done", "expert_completed": True, "failure": None} for t in rnd[2:3]] * 19 + rnd[:1]
    ok = [{**t, "idx": i + 1} for i, t in enumerate(ok)]                       # 1 of 20 = 5 %: not above
    rr, rf = CE.draw_report([_grasp_json("00004_g00", rows=ok)])
    assert rr["per_assembly"]["00004"]["shares"]["random_truncated"] == 0.05 and rf == [], (rr, rf)
    assert rr["all"]["show_user_info"] == [] and rr["per_assembly"]["00004"]["finger_contact_rows_by_kind"] == {}
    # r12-expert-01 / decision 7 of 2026-11-23: hand / finger rows by kind and waypoint redraws, information only
    hfr = [{"idx": 1, "kind": "random", "outcome": "expert_failed", "expert_completed": False, "finger_contact": True,
            "failure": "ExpertFailure: socket_contact_force_guard", "waypoint_redraws": []},
           {"idx": 2, "kind": "random", "outcome": "expert_failed", "expert_completed": False,
            "failure": "ExpertFailure: random_motion_0_nominal_command_guard: hand_finger_command_clearance",
            "waypoint_redraws": [{"waypoint": 0}, {"waypoint": 2}]},
           {"idx": 3, "kind": "random", "outcome": "done", "expert_completed": True, "failure": None,
            "waypoint_redraws": [{"waypoint": 1}]},
           {"idx": 4, "kind": "insert", "outcome": "expert_failed", "expert_completed": False, "finger_contact": True,
            "failure": "ExpertFailure: socket_penetration_guard"},
           {"idx": 5, "kind": "contact", "outcome": "expert_failed", "expert_completed": False,
            "failure": "ExpertFailure: continuous_approach_nominal_command_guard: hand_finger_command_clearance"},
           {"idx": 6, "kind": "perturbed", "outcome": "inserted", "expert_completed": True, "finger_contact": False,
            "failure": None},
           {"idx": 7, "kind": "random", "outcome": "placement_failed"}]
    rr, rf = CE.draw_report([_grasp_json("00422_g00", rows=hfr), _grasp_json("00015_g01", rows=hfr[5:6])])
    a = rr["per_assembly"]["00422"]
    assert a["finger_contact_rows_by_kind"] == {"random": 1, "insert": 1}, a
    assert a["guard_stopped_rows_by_kind"] == {"random": 1, "insert": 1}, a
    assert a["hand_finger_command_clearance_rows_by_kind"] == {"random": 1, "contact": 1}, a
    assert a["waypoint_redraws"] == {"rejected_draws": 3, "rows_with_redraw": 2, "of": 3}, a
    assert rr["all"]["hand_finger_command_clearance_rows_by_kind"] == {"random": 1, "contact": 1}, rr["all"]
    assert rr["all"]["waypoint_redraws"] == {"rejected_draws": 3, "rows_with_redraw": 2, "of": 3}, rr["all"]
    info = rr["all"]["show_user_info"]
    assert len(info) == 1 and info[0].startswith("00422 (information)") and "hand_finger_command_clearance" in info[0]
    assert not any("hand_finger" in f or "finger_contact" in f for f in rf), ("information, not a flag", rf)
    assert rr["per_assembly"]["00015"]["guard_stopped_rows_by_kind"] == {}, rr["per_assembly"]["00015"]
    # r13-integration-01 / r13-expert-01: a per-reset label reference or a t00 reference stopped by the decision-7 rule
    # is counted under "label" / "t00" (information; the row / grasp is still a label_failed / t00_refused draw)
    hf_txt = "approach_nominal_command_guard: hand_finger_command_clearance"
    lab_row = {**_row(2, outcome="label_failed", lab_completed=False, lab_end=False), "kind": "contact"}
    lab_row["label_reference"]["failure"] = "ExpertFailure: " + hf_txt
    other_lab = _row(3, outcome="label_failed", lab_completed=False, lab_end=False)
    other_lab["label_reference"]["failure"] = "ExpertFailure: socket_contact_force_guard"
    g_t00 = _grasp_json("00741_g01", ref_ok=False, ref_completed=False)
    g_t00["reference_failure"] = "expert did not complete: ExpertFailure: " + hf_txt
    g_lab = _grasp_json("00741_g00", rows=[lab_row, other_lab])
    rr, rf = CE.draw_report([g_lab, g_t00, _grasp_json("00422_g00", rows=hfr)])
    a = rr["per_assembly"]["00741"]
    assert a["hand_finger_command_clearance_rows_by_kind"] == {"label": 1, "t00": 1}, a
    assert a["counts"]["label_failed"] == 2 and a["counts"]["t00_refused"] == 1, a["counts"]
    assert rr["all"]["hand_finger_command_clearance_rows_by_kind"] == {"random": 1, "contact": 1, "label": 1, "t00": 1}
    assert rr["per_assembly"]["00422"]["hand_finger_command_clearance_rows_by_kind"] == {"random": 1, "contact": 1}
    info = rr["all"]["show_user_info"]
    assert any(x.startswith("00741 (information)") and "'label': 1" in x and "'t00': 1" in x for x in info), info
    assert not any("hand_finger" in f for f in rf), ("information, not a flag", rf)
    return (f"draws: 00015 flagged ({len(flags)} flags, e.g. {flags[0]}; contact_report_error 5.9 % among them), 00133 "
            "clean; random_truncated 2 of 4 random rows that ran flagged, 1 of 20 (5 %) not; hand / finger rows by "
            "kind (finger contact, guard stops, hand_finger_command_clearance, incl. label / t00 references) and "
            "waypoint redraws: information lines, no flag")


def _exit_codes(CE, data):
    ok = {"readers": [{"file": "a"}], "load_entries": {"n": 1}, "plan": {"grasps": 1, "without_grasp_json": [],
          "counted_build_failures": []}, "outcome_counts": {"outcome_inserted_vs_success_any_differ": 0,
                                                            "recorded_vs_recomputed_success_any_differ": 0}}
    assert CE.exit_reasons(ok, []) == []
    cases = [({"readers": [{"file": "a", "error": "KeyError: x"}]}, "reader"),
             ({"load_entries": {"error": "SplitUndecided"}}, "load_entries"),
             ({"outcome_counts": {"outcome_inserted_vs_success_any_differ": 1}}, "outcome_inserted_vs_success_any_differ"),
             ({"outcome_counts": {"recorded_vs_recomputed_success_any_differ": 2}}, "recorded_vs_recomputed"),
             ({"plan": {"error": "no plan.json"}}, "no plan.json"),
             ({"plan": {"without_grasp_json": ["x_g00"]}}, "without grasp.json"),
             ({"plan": {"counted_build_failures": ["build_failures/00015.json"]}}, "counted build failures"),
             ({"start_success_rows": ["00015/00015_g00/t03_perturbed.npz"]}, "start state satisfies AutoMate's rule")]
    for change, text in cases:
        why = CE.exit_reasons({**ok, **change}, [])
        assert why and text in why[0], (change, why)
    assert "format / counts" in CE.exit_reasons(ok, [{"file": "t01.npz"}])[0]
    bf = data / "build_failures"
    bf.mkdir()
    (bf / "00015.json").write_text(json.dumps([{"counted": False, "error": "network"}]))
    assert CE.counted_build_failures(data) == []                       # network / GPU memory failures do not count
    (bf / "00016.json").write_text(json.dumps([{"counted": False}, {"error": "x"}]))     # a row without the key counts
    assert CE.counted_build_failures(data) == ["build_failures/00016.json"]
    return "exit 1 for reader, load_entries, outcome-count, format, plan and counted build-failure problems only"


def spec_json(aid: str, d_dis_m: float, depth_top_m: float, pin_length_m: float = 0.02) -> dict:
    """An assets.AssemblySpec as its spec file holds it (only d_dis, L_top and L matter here)."""
    return {"assembly_id": aid, "pin_length_m": pin_length_m, "clearance_m": 1e-4, "disassembly_dist_m": d_dis_m,
            "entrance_z_m": 0.02, "depth_top_m": depth_top_m, "socket_top_z_m": 0.03, "plug_bottom_z_m": 0.0,
            "plug_top_z_m": 0.04, "seated_offcentre_m": 0.0, "min_gap_m": 1e-4, "fingertip_z_m": 0.03,
            "plug_xy_m": [0.01, 0.01], "symmetry_n": 4, "category": "rect", "fit": {}, "axis_xy_m": [0.0, 0.0],
            "tip_centre_m": [0.0, 0.0, 0.0]}


def _draws_only(CE, tmp: Path) -> str:
    """r5-integration-01: --draws-only reads grasp.json + plan.json only (no trajectory file is opened, no reference,
    kinematics, fast copy or media), writes draws.json whose flags equal draw_report's, exits 0 with flags raised;
    r5-expert-06: a row whose start state satisfies AutoMate's rule is flagged; r5-expert-07: contact rows ending
    jammed / guard with AutoMate success counted, band_above_top_mm from the frozen spec files; r7-expert-02:
    band_above_entrance_mm per assembly, the "all" list above 2 mm, no flag."""
    from wm.data import store
    data, work = tmp / "draws_only", tmp / "draws_only_work"
    gs = _draw_grasps()
    gs[-1]["trajectories"] += [
        {"idx": 14, "kind": "contact", "outcome": "jammed", "automate_success_any": True, "socket_index": 14},
        {"idx": 15, "kind": "contact", "outcome": "guard:lateral", "automate_success_any": False, "socket_index": 15},
        {"idx": 16, "kind": "contact", "outcome": "reached_depth", "automate_success_any": True, "socket_index": 16},
        {"idx": 17, "kind": "random", "outcome": "expert_failed", "expert_completed": False, "socket_index": 17,
         "failure": "ExpertFailure: step_budget_800"}]                               # r6-expert-01
    gs[-1]["trajectories"][1]["success"] = {"success_at_start": True}
    for g in gs:
        gdir = data / g["grasp"][:5] / g["grasp"]
        gdir.mkdir(parents=True)
        (gdir / "grasp.json").write_text(json.dumps(g))
        (gdir / "t01_insert.npz").write_bytes(b"not an npz")            # unreadable: must never be opened
    (data / "00133" / "00133_g01").mkdir()
    (data / "00133" / "00133_g01" / "grasp.json").write_text('{"grasp": "00133_g01", "traj')   # being written
    planned = [g["grasp"] for g in gs] + ["00133_g01", "00133_g02"]
    plan_of(data, planned)
    sf = data / "inputs" / "assets" / "00133" / CE.A.SPEC_FILE           # the collection's frozen spec: band 1.5 mm
    sf.parent.mkdir(parents=True)
    sf.write_text(json.dumps(spec_json("00133", 0.0135, 0.012, 0.0022)))    # band above the entrance 11.3 mm
    opened = []

    def trap(*a, **k):
        opened.append(a[:1])
        raise AssertionError(f"opened {a[:1]}")
    keep = (np.load, store.load, store.read_meta, CE.D.convert, CE.load_ref)
    np.load = store.load = store.read_meta = CE.D.convert = CE.load_ref = trap
    try:
        rep = CE.main(["--draws-only", "--data", str(data), "--work", str(work), "--ref", str(tmp / "no_ref")])
    finally:
        np.load, store.load, store.read_meta, CE.D.convert, CE.load_ref = keep
    assert not opened and sorted(p.name for p in work.iterdir()) == ["draws.json", "flags_for_user.txt"], (opened, list(work.iterdir()))
    assert (work / "flags_for_user.txt").read_text().splitlines() == rep["flags_for_user"]
    disk = json.loads((work / "draws.json").read_text())
    keep_dir, CE.A.ASSET_DIR = CE.A.ASSET_DIR, data / "inputs" / "assets"
    try:
        want, flags = CE.draw_report(gs, {g: 12 for g in planned}, CE.specs_of(["00015", "00133"]))
    finally:
        CE.A.ASSET_DIR = keep_dir
    assert disk["flags_for_user"] == flags == rep["flags_for_user"], (disk["flags_for_user"], flags)
    assert json.loads(json.dumps(want["per_assembly"], default=float)) == disk["per_assembly"]
    b = disk["per_assembly"]["00133"]
    assert b["start_success_rows"] == ["00133_g00 t01"] and any("00133: 1 trajectories start" in f for f in flags), flags
    assert abs(b["band_above_top_mm"] - 1.5) < 1e-9 and disk["per_assembly"]["00015"]["band_above_top_mm"] is None, b
    assert b["counts"]["contact_rows"] == 3 and b["counts"]["contact_no_insert_with_automate_success"] == 1, b["counts"]
    assert b["random_truncated"] == {"n": 1, "of": 1, "share": 1.0} and "00133: random_truncated 100.0% of 1 (> 5%)" in flags
    assert disk["all"]["random_truncated"]["n"] == 1, disk["all"]
    assert disk["all"]["band_above_top"]["above_top"] == 1 and "within up to 1.50 mm" in disk["all"]["band_above_top"]["text"]
    assert abs(b["band_above_entrance_mm"] - 11.3) < 1e-9 and disk["per_assembly"]["00015"]["band_above_entrance_mm"] is None, b
    be = disk["all"]["band_above_entrance"]                                          # r7-expert-02: information only
    assert be["assemblies"] == {"00133": b["band_above_entrance_mm"]} and "00133 11.3 mm" in be["text"], be
    assert not any("entrance" in f for f in flags), flags
    keep_dir, CE.A.ASSET_DIR = CE.A.ASSET_DIR, data / "inputs" / "assets"            # 2 mm or less: not listed
    try:
        sf.write_text(json.dumps(spec_json("00133", 0.0135, 0.012, 0.012)))     # 1.5 mm
        small, _ = CE.draw_report(gs, None, CE.specs_of(["00133"]))
    finally:
        CE.A.ASSET_DIR = keep_dir
    assert small["all"]["band_above_entrance"] == {"listed_above_mm": 2.0, "assemblies": {}, "text": None}, small["all"]
    assert disk["plan"]["without_grasp_json"] == ["00133_g02"] and len(disk["grasp_json_unreadable"]) == 1, disk["plan"]
    assert disk["completeness"]["undone_not_skipped"] == ["00133_g02"], disk["completeness"]   # g01's file exists
    return (f"draws-only: no trajectory opened, draws.json flags = draw_report's ({len(flags)}), start-state row flagged, "
            "band 1.5 mm from the frozen spec, band above the entrance 11.3 mm listed (1.5 mm not), no flag, "
            "jammed-with-success counted, unfinished grasps information only, exit 0")


# ------------------------------------------------------------------------------------------------------------ 11
def test_cache(tmp: Path):
    import hashlib
    import cache_wm_v1_fast as CF
    root = tmp / "coll_cache"
    conds = {"00015_g00": "train", "00015_g01": "select", "00016_g00": "test", "00017_g00": "heldout_shape"}
    for grasp, cond in conds.items():
        gdir = root / grasp[:5] / grasp
        gdir.mkdir(parents=True)
        raw_trajectory(gdir / "t01_insert.npz", {**am(grasp, cond, pin_length_m=0.02 + 0.001 * int(grasp[3:5]), idx=1,
                                                         expert_completed=True, automate_success_any=True, plan_sha="P")})
        (gdir / "grasp.json").write_text(json.dumps({"grasp": grasp, "trajectories": [
            {"idx": 1, "kind": "insert", "outcome": "inserted", "file": "t01_insert.npz"}]}))
    document = {"rule": "automate_v1", "test": ["00016"]}
    split_doc = json.dumps({**document, "plan_sha": "P"})                # as collect.freeze writes it
    (root / "split_automate_v1.json").write_text(split_doc)
    plan = {"plan_sha": "P", "split": {"name": "automate_v1", "file": "split_automate_v1.json", "document": document}}
    plan_of(root, list(conds), **plan)
    IX.build_automate(root, workers=1)
    idx = json.loads((root / "index.json").read_text())
    for e in idx:                                        # a stale "split" field must not decide (the test grasp says train)
        e["split"] = "train"
    (root / "index.json").write_text(json.dumps(idx))
    rec_f = root / IX.COMPLETENESS_FILE
    rec = json.loads(rec_f.read_text())
    assert rec["partial"] is False and rec["planned"] == rec["with_grasp_json"] == 4, rec

    def run(dst, *extra):
        old, sys.argv = sys.argv, ["cache_wm_v1_fast.py", "--src", str(root), "--dst", str(dst), "--workers", "1", *extra]
        try:
            CF.main()
        finally:
            sys.argv = old
    dst = tmp / "fast_cache"
    raises(lambda: run(tmp / "f_stale_rec"), SystemExit, "describes another index.json")   # index.json edited above
    assert not (tmp / "f_stale_rec").exists()
    rec["index_sha256"] = IX.index_sha256(root)
    rec_f.write_text(json.dumps(rec))
    rec_f.rename(rec_f.with_suffix(".away"))
    raises(lambda: run(tmp / "f_no_rec"), SystemExit, "no index_completeness.json")
    rec_f.with_suffix(".away").rename(rec_f)
    run(dst)
    man = json.loads((dst / "manifest.json").read_text())
    assert man["completeness"] == rec, man.get("completeness")
    assert sorted(e["grasp"] for e in man["entries"]) == ["00015_g00", "00015_g01", "00017_g00"], man
    assert man["skipped"]["test_ports"] == 1 and not (dst / "00016").exists()
    assert man["split"] == {"name": "automate_v1", "file": "split_automate_v1.json",
                            "sha256": hashlib.sha256(split_doc.encode()).hexdigest(), "plan_sha": "P"}, man["split"]
    assert all(e["expert_completed"] is True and e["automate_success_any"] is True for e in man["entries"])
    assert abs({e["grasp"]: e["pin_length_m"] for e in man["entries"]}["00017_g00"] - 0.037) < 1e-12
    code = CF.converter_code()                                          # r6-data-04
    assert man["code"] == {**code, "earlier_writers": []}, man.get("code")
    assert set(code["md5"]) == set(CF.CODE_FILES) and code["md5"]["scripts/data/cache_wm_v1_fast.py"] == \
        hashlib.md5(Path(CF.__file__).read_bytes()).hexdigest(), code
    assert json.loads((dst / "source.json").read_text()) == {"source": str(root.resolve()), "split": man["split"],
                                                             "code": [code]}
    resumed = _cache_resume(CF, root, dst, run, plan, document)
    pins = tmp / "pins.json"
    pins.write_text("{}")
    raises(lambda: run(tmp / "f2", "--pins", str(pins)), SystemExit, "--pins is for ManiSkill")
    (root / "index.json").write_text(json.dumps(idx + [{**idx[0], "simulator": None}]))
    raises(lambda: run(tmp / "f3"), SystemExit, "build_automate")
    (root / "index.json").write_text(json.dumps(idx + [{**idx[0], "condition": "unassigned"}]))
    raises(lambda: run(tmp / "f4"), SystemExit, "no decided split")
    (root / "index.json").write_text(json.dumps(idx))
    (root / "split_automate_v1.json").unlink()
    raises(lambda: run(tmp / "f5"), SystemExit, "missing")
    (root / "plan.json").unlink()
    raises(lambda: run(tmp / "f6"), SystemExit, "split identity")
    ms = tmp / "ms_src"                                   # ManiSkill: --pins still required
    ms.mkdir()
    (ms / "index.json").write_text(json.dumps([{"path": "x.npz", "port": "G3", "grasp": "G3_g00", "condition": "train",
                                                "kind": "insert", "n_images": 1, "split": "train"}]))
    old, sys.argv = sys.argv, ["cache_wm_v1_fast.py", "--src", str(ms), "--dst", str(tmp / "f7")]
    try:
        raises(CF.main, SystemExit, "--pins is required")
    finally:
        sys.argv = old
    return ("test grasp skipped by its condition (index field ignored), split identity and completeness in the manifest "
            "(a missing or stale completeness record refused), pins from the "
            f"entries; {resumed}; --pins, a mixed index, an undecided split, a missing split file / plan refused; "
            "ManiSkill needs --pins")


def _cache_resume(CF, root, dst, run, plan, document):
    """r4-data-02 / r4-data-08: a kept fast file must hold the plan_sha and the condition and be newer than its source;
    --dst is tied to one collection and split; the split file must be the plan's."""
    import os
    f = dst / "00015" / "00015_g00" / "t01_insert.npz"
    with np.load(f) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    arrays["meta"] = np.asarray(json.dumps({**meta, "condition": "heldout_shape", "plan_sha": "OLD"}))
    np.savez(f, **arrays)                                       # an old file of another plan / split at the same path
    run(dst)
    man = json.loads((dst / "manifest.json").read_text())
    assert {e["grasp"]: e["condition"] for e in man["entries"]}["00015_g00"] == "train", man["entries"]
    with np.load(f) as z:
        assert json.loads(str(z["meta"]))["plan_sha"] == "P"     # converted again
    src = root / "00015" / "00015_g00" / "t01_insert.npz"
    old_ns = os.stat(src).st_mtime_ns - 10**9
    os.utime(f, ns=(old_ns, old_ns))                             # the copy older than its source (source rewritten)
    run(dst)
    assert os.stat(f).st_mtime_ns > old_ns                       # converted again
    other = dst.parent / "fast_reused"                           # a non-empty --dst without source.json
    other.mkdir()
    (other / "stray.npz").write_text("x")
    raises(lambda: run(other), SystemExit, "source.json is missing")
    mark = json.loads((dst / "source.json").read_text())
    (dst / "source.json").write_text(json.dumps({**mark, "split": {**mark["split"], "sha256": "0" * 64}}))
    raises(lambda: run(dst), SystemExit, "use a new --dst")      # another split identity
    old_writer = {"root": "/old/code", "md5": {**mark["code"][0]["md5"], "src/wm/tdmpc/demo320.py": "0" * 32}}
    (dst / "source.json").write_text(json.dumps({**mark, "code": [old_writer]}))    # files of an older converter kept
    run(dst)
    man = json.loads((dst / "manifest.json").read_text())
    assert man["code"]["earlier_writers"] == [old_writer["md5"]] and man["code"]["md5"] == mark["code"][0]["md5"], man["code"]
    assert json.loads((dst / "source.json").read_text())["code"] == [old_writer, mark["code"][0]]
    run(dst)                                                     # the same converter again: nothing appended
    assert json.loads((dst / "source.json").read_text())["code"] == [old_writer, mark["code"][0]]
    (dst / "source.json").write_text(json.dumps(mark))
    sf = root / "split_automate_v1.json"
    good = sf.read_text()
    sf.write_text(json.dumps({**document, "test": ["00017"], "plan_sha": "P"}))     # hand-edited list
    raises(lambda: run(dst.parent / "f_edit"), SystemExit, "differs from")
    sf.write_text(json.dumps({**document, "plan_sha": "OTHER"}))                     # a stale file of another plan
    raises(lambda: run(dst.parent / "f_stale"), SystemExit, "plan_sha")
    sf.write_text(good)
    g1 = root / "00015" / "00015_g01" / "t01_insert.npz"
    with np.load(g1) as z:
        arrays = {k: z[k] for k in z.files}
    keep_meta = str(arrays["meta"])
    arrays["meta"] = np.asarray(json.dumps({**json.loads(keep_meta), "plan_sha": "Q"}))
    np.savez(g1, **arrays)                                       # a source trajectory of another plan
    raises(lambda: run(dst.parent / "f_src"), SystemExit, "failed")
    arrays["meta"] = np.asarray(keep_meta)
    np.savez(g1, **arrays)
    return ("resume: a kept file of another condition / plan_sha converted again, a newer source converted again; a "
            "reused --dst without source.json or with another split refused; an older converter listed under "
            "earlier_writers; an edited or stale split file and a source of another plan refused")


# ------------------------------------------------------------------------------------------------------------ 12
def test_compare_device(tmp: Path):
    import types
    import compare_physics_device as CP
    spec = types.SimpleNamespace(tip_centre_m=(0.0, 0.0, -0.02), symmetry_n=4, disassembly_dist_m=0.02)
    from wm.sim_automate import config as C
    keep, CP.A.load_spec = CP.A.load_spec, (lambda aid: spec)
    try:
        return _compare_device(CP, C) + "; " + _compare_batches(CP, tmp)
    finally:
        CP.A.load_spec = keep


def _compare_device(CP, C):
    G3 = C.R_G[:3, :3]

    def arrays(hs):
        rel = np.zeros((len(hs), 7))
        rel[:, 3] = 1.0
        rel[:, :3] = np.stack([G3.T @ np.array([0.0, 0.0, h * 1e-3]) for h in hs])
        return {"eval_rel": rel}
    a = CP.trajectory_numbers(arrays([30, 10, 0.05]), {"assembly_id": "x", "expert_completed": True, "outcome": "inserted"})
    b = CP.trajectory_numbers(arrays([30, 10, 3.0]), {"assembly_id": "x", "expert_completed": False, "outcome": "inserted",
                                                       "failure": "ExpertFailure: jam"})
    assert a["seat_ok"] and a["success_any"] and a["first_success_step"] == 1 and a["completed"], a
    assert not b["seat_ok"] and b["success_any"] and b["success_end"] and not b["completed"], b
    return "seat_ok = completed AND |depth left| < 0.5 mm AND success at the end; a jam 3 mm short is AutoMate success but not seat_ok"


def _compare_batches(CP, tmp: Path) -> str:
    """r4-decisions-04: per-assembly comparison of a GPU and a CPU milestone-1b batch, and the plans' split check."""
    prov = {"ranges_sha256": "r", "grasp_py_sha256": "g", "physics_code_sha256": "p", "spec_sha256": "s",
            "plug_obj_sha256": "po", "socket_obj_sha256": "so", "shared_asset_sha256": {"plug_grasps.json": "pg"},
            "switches": {"socket_draw": "per_reset", "physics_device_default": "cpu"}}

    def res(dev, ii_pass=True, depth=0.01, h=0.02, iii=None, **kw):
        sw = {**prov["switches"], "physics_device_default": dev}
        return {**prov, "switches": sw, "physics_device": dev, "i": {"pass": True},
                "ii": {"pass": ii_pass, "placed": True, "completed": True, "push_end": "done", "depth_left_mm": depth,
                       "height_mm": h, "success_end": ii_pass, "success_any": True},
                "iii": iii if iii is not None else {"skipped": "no box"}, **kw}
    ga, gb = tmp / "batch_gpu", tmp / "batch_cpu"
    ga.mkdir()
    gb.mkdir()
    rows = {"00015": (res("cuda:0"), res("cpu")),
            "00133": (res("cuda:0", ii_pass=False, depth=0.7, h=-0.001), res("cpu")),
            "00444": (res("cuda:0", iii={"pass": False, "depth_left_mm": 0.1, "height_mm": 0.03, "success_end": True,
                                         "hand_finger_max_N": 2.0}),
                      res("cpu", iii={"pass": True, "depth_left_mm": 0.1, "height_mm": 0.03, "success_end": True,
                                      "hand_finger_max_N": 0.5}, ranges_sha256="other")),
            "00117": ({"error": "the simulator process died during this assembly", **prov}, res("cpu")),
            "00320": (res("cuda:0"), res("cpu", plug_obj_sha256="other"))}               # r6-ops-03: another plug mesh
    for aid, (a, b) in rows.items():
        (ga / f"{aid}.json").write_text(json.dumps(a))
        (gb / f"{aid}.json").write_text(json.dumps(b))
    (ga / "00559.json").write_text(json.dumps(res("cuda:0")))          # only in the GPU batch
    (ga / "00015.started").write_text("x")                              # markers are not results
    sm, rr = CP.compare_batches(ga, gb)
    assert sm["assemblies_in_both"] == 5 and sm["only_in_a"] == ["00559"] and sm["only_in_b"] == [], sm
    assert sm["pass_ii_or_iii_differs"] == ["00117", "00133", "00444"], sm
    assert sm["different"]["pass_ii"] == 2 and sm["different"]["pass_iii"] == 2 and sm["different"]["error"] == 1, sm
    assert sm["different"]["ii_depth_left_mm"] == 2 and abs(sm["max_abs_diff_mm"]["ii_depth_left_mm"] - 0.69) < 1e-9, sm
    assert sm["provenance_differs"] == {"00444": ["ranges_sha256"], "00320": ["plug_obj_sha256"]}, \
        sm["provenance_differs"]                                                    # device switch ignored
    assert ["cuda:0", "cpu"] in sm["physics_device"], sm["physics_device"]
    # collections: the two plans' split documents
    ca, cb = tmp / "coll_gpu", tmp / "coll_cpu"
    for d, test in ((ca, ["00016"]), (cb, ["00017"])):
        (d / "00015" / "00015_g00").mkdir(parents=True)
        (d / "plan.json").write_text(json.dumps({"split": {"document": {"test": test}}}))
    (cb / "00015" / "00015_g00" / "grasp.json").write_text(json.dumps({"grasp": "00015_g00"}))
    cs, _ = CP.compare(ca, cb)
    assert not cs["plan_split_equal"] and "grasp name only" in cs["compared_by"] and cs["grasps_only_in_b"] == ["00015_g00"], cs
    (cb / "plan.json").write_text((ca / "plan.json").read_text())
    assert CP.compare(ca, cb)[0]["plan_split_equal"]
    out = tmp / "cmp.json"
    CP.main(["--batch-a", str(ga), "--batch-b", str(gb), "--out", str(out)])
    assert json.loads(out.read_text())["batches"]["summary"]["assemblies_in_both"] == 5
    raises(lambda: CP.main(["--a", str(ca)]), SystemExit, "--a with --b")
    return (f"batches: pass (ii) / (iii) differ on {sm['pass_ii_or_iii_differs']}, provenance differences listed apart "
            "from the device switch (a different plug mesh listed); collections: unequal split documents reported, compared by grasp name only")


# ------------------------------------------------------------------------------------------------------------ 13
def _argv(main, argv):
    old, sys.argv = sys.argv, argv
    try:
        return main()
    finally:
        sys.argv = old


def test_launch_side(tmp: Path):
    """r4-data-05 (GPU cap), r4-data-04 (versions), r4-integration-04 (build_index --automate), r4-data-07 (refusals),
    r4-data-01 (one MIN_SELECT_GRASPS)."""
    import torch
    import train_wm320 as T
    keep = os.environ.pop("QX_GPU_GB", None)
    try:
        cuda = torch.device("cuda")
        raises(lambda: T.cap_gpu_memory(cuda, True, False), SystemExit, "QX_GPU_GB")        # AutoMate needs the budget
        assert T.cap_gpu_memory(cuda, False, False) is None                                 # ManiSkill unchanged
        assert T.cap_gpu_memory(torch.device("cpu"), True, False) is None
        for v, text in (("abc", "not a number"), ("1", "nothing left"), ("0.5", "nothing left")):
            os.environ["QX_GPU_GB"] = v
            raises(lambda: T.cap_gpu_memory(cuda, True, False), SystemExit, text)
        os.environ["QX_GPU_GB"] = "8"
        raises(lambda: T.cap_gpu_memory(cuda, True, True), SystemExit, "torchrun")
        import gpu_check_encoders320 as G
        base = ["gpu_check_encoders320.py", "--depth-max-m", "0.5", "--depth-mean-m", "0.3", "--depth-std-m", "0.06"]
        raises(lambda: _argv(G.main, base), SystemExit, "--out is required")               # r5-data-05
        raises(lambda: _argv(G.main, base + ["--out", str(tmp / "no_dir" / "g.json")]), SystemExit, "not writable")
        base += ["--out", str(tmp / "gpu_check.json")]
        raises(lambda: _argv(G.main, base + ["--mem-gb", "7.5"]), SystemExit, "--mem-gb 7.5 must lie in (0, QX_GPU_GB - 1.0 = 7.0]")
        raises(lambda: _argv(G.main, base[:2] + ["3.3"] + base[3:]), SystemExit, "int16 ceiling")   # r5-data-new-01
        os.environ.pop("QX_GPU_GB")
        raises(lambda: _argv(G.main, base), SystemExit, "QX_GPU_GB=None")
        assert not (tmp / "gpu_check.json").exists()
    finally:
        os.environ.pop("QX_GPU_GB", None)
        if keep is not None:
            os.environ["QX_GPU_GB"] = keep
    pv = T.provenance(None)
    assert {"sklearn", "joblib", "threadpoolctl"} <= set(pv["versions"]) and "sklearn_file" in pv, pv
    assert T.MIN_SELECT_GRASPS == D.MIN_SELECT_GRASPS == 20
    ap = T.automate_provenance({"completeness": {"partial": False}, "code": {"md5": {"a": "b"}}}, True)   # r6-data-04
    assert ap == {"completeness": {"partial": False}, "allow_partial": True, "fast_copy_code": {"md5": {"a": "b"}}}, ap
    assert T.automate_provenance({}, False)["fast_copy_code"] is None
    # build_index --automate: this tree's wm, the AutoMate index; refused without plan.json or with both forms
    import build_index as BI
    root = tmp / "coll_bi"
    gdir = root / "00015" / "00015_g00"
    gdir.mkdir(parents=True)
    raw_trajectory(gdir / "t01_insert.npz", {**am("00015_g00", "train", pin_length_m=0.02, idx=1)})
    (gdir / "grasp.json").write_text(json.dumps({"grasp": "00015_g00", "trajectories": [
        {"idx": 1, "kind": "insert", "outcome": "inserted", "file": "t01_insert.npz"}]}))
    raises(lambda: _argv(BI.main, ["build_index.py", "--automate", str(root), "--workers", "1"]), SystemExit, "no plan.json")
    plan_of(root, ["00015_g00", "00015_g01"])                  # 00015_g01 unfinished: refused (exit 2), then --partial
    raises(lambda: _argv(BI.main, ["build_index.py", "--automate", str(root), "--workers", "1"]), SystemExit, "2")
    assert not (root / "index.json").exists()
    _argv(BI.main, ["build_index.py", "--automate", str(root), "--workers", "1", "--partial"])
    assert json.loads((root / IX.COMPLETENESS_FILE).read_text())["partial"] is True
    plan_of(root, ["00015_g00"])
    _argv(BI.main, ["build_index.py", "--automate", str(root), "--workers", "1"])
    idx = json.loads((root / "index.json").read_text())
    assert len(idx) == 1 and idx[0]["simulator"] == "automate" and idx[0]["split"] == "train", idx
    assert json.loads((root / IX.COMPLETENESS_FILE).read_text())["partial"] is False
    raises(lambda: _argv(BI.main, ["build_index.py", str(root), "--partial"]), SystemExit, "for --automate")
    raises(lambda: _argv(BI.main, ["build_index.py", str(root), "--automate", str(root)]), SystemExit, "either")
    raises(lambda: _argv(BI.main, ["build_index.py"]), SystemExit, "either")
    # evaluation / analysis scripts refuse AutoMate up front
    sys.path.insert(0, str(HERE.parent / "scripts" / "eval"))
    sys.path.insert(0, str(HERE.parent / "scripts" / "analysis"))
    import ablation320 as AB
    import eval_policy320 as EP
    import eval_ppo_state as PS
    (gdir / "grasp.json").write_text(json.dumps({**am("00015_g00", "heldout_shape"), "reference_ok": True}))
    out = tmp / "ev_out"
    raises(lambda: _argv(EP.main, ["eval_policy320.py", "--grasps", str(root), "--split", "heldout", "--policy", "x",
                                   "--out", str(out)]), SystemExit, "1 AutoMate grasps")
    raises(lambda: _argv(PS.main, ["eval_ppo_state.py", "--grasps", str(root), "--split", "heldout", "--model", "x",
                                   "--vecnorm", "x", "--scale", "x", "--out", str(out)]), SystemExit, "1 AutoMate grasps")
    import make_combo_grasps as MC                                                       # r5-data-04
    raises(lambda: _argv(MC.main, ["make_combo_grasps.py", "--grasps", str(root), "--out", str(out)]), SystemExit,
           "1 AutoMate grasps")
    assert not out.exists()
    arm = tmp / "arm"
    (arm / "00015").mkdir(parents=True)
    (arm / "00015" / "00015_g00.json").write_text(json.dumps({"grasp": "00015_g00", "port": "00015", "starts": []}))
    raises(lambda: AB.load_arm(str(arm)), SystemExit, "AutoMate result")
    raises(lambda: AB.load_cells(str(arm)), SystemExit, "AutoMate result")
    (arm / "00015" / "00015_g00.json").write_text(json.dumps({"grasp": "G3_g00", "port": "G3", "condition": "heldout_grasp",
                                                              "starts": []}))
    raises(lambda: AB.load_arm(str(arm)), SystemExit, "AutoMate result")
    (arm / "00015" / "00015_g00.json").write_text(json.dumps({"grasp": "G3_g00", "port": "G3", "pitch_deg": -30.0,
                                                              "grasp_pos_m": -0.01, "starts": []}))
    assert AB.load_arm(str(arm)) == {}                                         # a ManiSkill file still loads
    return ("QX_GPU_GB: AutoMate on CUDA refused without it, bad values and torchrun refused, ManiSkill unchanged; "
            "gpu_check refuses --mem-gb above QX_GPU_GB - 1 and a missing QX_GPU_GB; provenance holds sklearn / joblib / "
            "threadpoolctl and the sklearn file, AutoMate provenance the fast copy's converter code; gpu_check needs a writable --out and refuses D_max > 3.2767 m; one "
            "MIN_SELECT_GRASPS; build_index --automate (an unfinished collection refused, --partial recorded); "
            "eval_policy320, eval_ppo_state, make_combo_grasps and ablation320 refuse AutoMate up front")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="automate_data_"))
    import shutil
    try:
        print("1 test_split OK:", test_split(), flush=True)
        print("2 test_fast_meta OK:", test_fast_meta(tmp), flush=True)
        print("3 test_hash OK:", test_hash(tmp), flush=True)
        print("4 test_depth_stats OK:", test_depth_stats(tmp), flush=True)
        print("5 test_online_entries OK:", test_online_entries(tmp), flush=True)
        print("6 test_labels OK:", test_labels(), flush=True)
        print("7 test_index OK:", test_index(tmp), flush=True)
        print("8 test_encoder_env OK:", test_encoder_env(tmp), flush=True)
        print("9 test_bc OK:", test_bc(tmp), flush=True)
        print("10 test_check_expert_data OK:", test_check_expert_data(tmp), flush=True)
        print("11 test_cache OK:", test_cache(tmp), flush=True)
        print("12 test_compare_device OK:", test_compare_device(tmp), flush=True)
        print("13 test_launch_side OK:", test_launch_side(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
