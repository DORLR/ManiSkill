"""Check recorded AutoMate expert data against wm_v1 and the existing readers; save wrist pictures and a GIF. No Isaac.

    python -u check_expert_data.py --data /mnt/storage3/qingxin/automate/expert_smoke/data \
        --ref /mnt/storage2/qingxin/data/wm_v1/A/A_g00 --work /mnt/storage3/qingxin/automate/expert_smoke/check
    python -u check_expert_data.py --draws-only --data <collection out> --work <folder>      (cheap; also mid-run)
--draws-only (r5-integration-01, the row-22 condition of 2026-11-23): reads only <data>/*/*/grasp.json and
<data>/plan.json (and, for the information numbers below, the frozen asset spec files and the build_failures /
deaths / heartbeat / .claim bookkeeping); it opens no trajectory file, loads no wm_v1 reference, no kinematics, and
writes no fast copy or media. It runs the draws, heights and start-state blocks below and the plan block, with planned
grasps without grasp.json as information only (the run may still be going: wm.data.index.automate_completeness facts
are added), writes <work>/draws.json ({"per_assembly", "all", "flags_for_user", "plan", "completeness", ...}) and
<work>/flags_for_user.txt (one flag per line, empty when none: for a shell caller's ALERT lines), prints
flags_for_user and exits 0 (also when a flag is raised: the flags are for the user, not a failure of the run).
Unreadable grasp.json files (e.g. one being written) are listed and skipped.
--data is a collection's output folder. The asset specs (pin length, tip centre, symmetry, disassembly distance) are
read from the collection's frozen <data>/inputs/assets when it exists (else $WM_AUTOMATE_ASSETS or the default
folder); a trajectory whose assembly has no spec file there falls back to the meta's own values (spec_source "meta").
Checks per trajectory file (results in <work>/check.json):
  format   wm.data.store.load reads it; the array keys and dtypes equal those of the wm_v1 file of the same kind in
           the --ref grasp directory, apart from AutoMate's own truth arrays, whose names start with "eval_" (eval_tip;
           any other extra array is a mismatch); array ranks and per-axis sizes (images 320 x 320, 9 joints, 7 targets, 4 x 4
           poses, 7-vector truth) equal; the meta has every key of that file's meta; schema version equal
  counts   qpos / hand / eval_rel T + 1, action / phase T, pictures N with img_step = 0, 4, 8, ... (+ the last step)
  values   hand = Pinocchio FK of qpos (wm.sim_automate.kinematics) [mm]; cam_T_world = inverse of FK hand x hand-eye
           mount at the picture steps [mm]; actions finite and inside the joint limits, largest step-to-step change
           [rad]; depth: fraction of pixels with a return, range [mm]; segmentation class fractions
  truth    AutoMate's success rule (the one success criterion, decision 1 of 2026-11-22; checks.py): with h_t, k_t the
           plug height and mean keypoint distance in row t of eval_rel (row 0 = the start state, row t = after control
           step t) and d_dis the assembly's disassembly distance,
               s_t = [0 < h_t < d_dis AND k_t < 15 mm],   evaluated for t = 1..T only (never the start state),
               success_any = OR_{t=1..T} s_t,   success_end = s_T,   first_success_step = min t with s_t,
               start_success = s_0 (information: the start state already satisfies the rule)
           (checks.success_steps); truth_end = checks.truth_from_rel_spec at row T (success, height_mm,
           keypoint_dist_mm, and the information numbers depth_left_mm, lateral_mm, tilt / spin / angle); the last
           eval_tip row; the recorded meta automate_success_any against the recomputed success_any
  outcome  counts over insert / perturbed trajectories (outcome "inserted" = success_any; the imitation set needs
           outcome "inserted" AND expert_completed, demo320.is_bc):
               outcome_inserted_vs_success_any_differ   should be 0
               recorded_vs_recomputed_success_any_differ should be 0
               success_any_not_completed  AutoMate success, but the expert did not complete (not imitated)
               completed_not_success_end  the expert completed but s_T is False (show the user before launch if > 0)
               completed_not_success_any
               success_any_not_end         met the rule after some step, not at the end (information)
               start_success               s_0 True (information)
               bc                          would be imitated
               failure_kinds               counts of the expert's failure text up to its second ':'
               peak_finger_socket_gt_1N, success_end_spin_ge_4_deg (information)
           per assembly: trajectories, the smallest h_T [mm] over completed ones, completed_not_success_end,
           start_success and the first_success_step range; references (t00): completed and success_end
  readers  wm.tdmpc.demo320.convert (the fast copy the 320 px policy and world model read; allow_test: this check's
           copy is a scratch folder under --work), demo320.policy_sample, demo320.load_entries on a manifest written as
           scripts/data/cache_wm_v1_fast.py writes it (allow_undecided: while the AutoMate split is undecided its
           entries are reported as split "unassigned"), and wm.tdmpc.wmdata320.trajectory (decision labels from FK, H*,
           axis and L: ends seated or not)
  plan     every grasp of <data>/plan.json has a grasp.json, and no build_failures/*.json holds a counted failure
           (counted true or missing, as collect.py counts them; an assembly is skipped after repeated ones)
  draws    per assembly, from every grasp.json (draw_report; also for collect_smoke_check). A socket draw is one reset
           of a grasp: the t00 reset plus one per planned trajectory reached (per-reset socket draw); n_draws counts
           them, n_attempts the placement attempts (each start drawn, the first and every redraw):
               reset_failed share   = (planned rows "reset_failed" + t00 resets without IK) / n_draws
               label_failed share   = planned rows "label_failed" / n_draws
               contact_report_error share = planned rows "contact_report_error" (any stage) / n_draws
               t00_refused share    = grasps whose t00 reference was refused / grasps, split by cause: "expert did
                                      not complete", "completed, AutoMate's rule fails at the end" (with h <= 0 counted
                                      apart: a seated plug a few micrometres below h = 0 fails the strict h > 0),
                                      reset_failed / contact_report_error (soft t00 outcomes), other; with the
                                      planned trajectories lost with them (plan.json)
               redraw share per reason = placement failures with that reason / n_attempts; reason = the failed
                                      stage and, for a guarded leg, its stop reason: clearance, retreat:lag,
                                      retreat:socket_force, retreat:socket_penetration, retreat:ik, retreat:settle,
                                      transfer:ik_or_settle, descend:plug_socket_contact (the 1 N outside rule),
                                      descend:lag, ..., start_error (all legs ok, start missed by > 0.5 mm / 0.2 deg),
                                      re-home reset
               placement_failed share = rows "placement_failed" / rows that reached placement
               start_success_rows    = rows (any idx, t00 included) whose recorded success.success_at_start is True,
                                      plus a refused t00 whose reference_success.success_at_start is True: under the
                                      start rule "socket_top" no start can satisfy AutoMate's rule (the band reaches at
                                      most 1.99 mm above the socket top, starts lie >= 6 mm above it), so any such row
                                      means a wrong start (g_top not applied, a stale spec); listed in flags_for_user
                                      (r5-expert-06)
               contact trials       (information, r5-expert-07) contact rows, and those ending "jammed" or "guard:..."
                                      with automate_success_any True (the rule met without insertion)
               band_above_top_mm    (information, r5-expert-07; needs the assembly's spec file) min(d_dis, 15 mm) -
                                      L_top: how far above the socket top AutoMate's success region reaches (> 0: a
                                      plug resting on the socket top within this height counts as success); "all"
                                      gives the assemblies with a positive value and the largest
               band_above_entrance_mm (information, r7-expert-02; needs the spec file) min(d_dis, 15 mm) - L: how
                                      far above the entrance of the tight fit (depth L) AutoMate's rule is already met
                                      (the plug root within 15 mm of seated). > 0: a plug can satisfy the rule, so
                                      S_any of insert / perturbed runs ("inserted") and automate_success_any of contact
                                      trials can become True, before the tight fit is reached, e.g. a plug jammed at
                                      the entrance (00133: 12.8 mm). "all" lists the assemblies above
                                      BAND_ENTRANCE_LIST_MM = 2 mm, largest first, to be shown to the user next to
                                      contact_no_insert_with_automate_success_share. No flag (AutoMate's rule is the
                                      user's approved, unchanged criterion); the imitation set is not affected (bc
                                      needs expert_completed, whose final_verify needs depth_left < 0.5 mm)
               random_truncated     (information, r6-expert-01) random-motion rows whose expert ran (the row has
                                      expert_completed) and stopped at the step budget or before a waypoint: failure
                                      text starting "ExpertFailure: step_budget" or containing "_did_not_reach_target"
                                      (expert.py _step / _move_outside; at wm_v1's 2.7 mm/s, _advance halves steps on
                                      curved TCP paths, as wm_v1's did). Recorded truncated as outcome "expert_failed";
                                      per assembly {"n", "of" (random rows that ran), "share"}, "all" the same; the
                                      share is also in "shares" and flagged above 5 % like the others (lead, round 6)
               hand / finger rows   (information, r12-expert-01 and the user's decision 7 of 2026-11-23, no flag)
                                      per assembly, rows counted by kind (insert, perturbed, contact, random; t00 is
                                      the reference and has no row, see hand_finger_command_clearance for it):
                                      finger_contact_rows_by_kind   rows with finger_contact (peak hand / finger -
                                                                    socket force > 1 N, collect.finger_info)
                                      guard_stopped_rows_by_kind    rows whose failure text contains
                                                                    socket_contact_force_guard or
                                                                    socket_penetration_guard (the 80 N / 0.5 mm guards)
                                      hand_finger_command_clearance_rows_by_kind   rows whose failure text contains
                                                                    hand_finger_command_clearance (decision 7: a nominal
                                                                    outside command's hand / finger < 1 mm from the
                                                                    socket; expert_failed); also the unrecorded
                                                                    references the rule stopped: "label" = label_failed
                                                                    rows whose label_reference.failure contains it,
                                                                    "t00" = grasps whose reference_failure contains it
                                                                    (these are also in label_failed / t00_refused, whose
                                                                    keys do not name the cause)
                                      waypoint_redraws  {"rejected_draws", "rows_with_redraw", "of"}: random rows
                                                        whose waypoints were redrawn (collect.redraw_waypoints), of the
                                                        random rows that reached the run
                                      "all" has the sums and "show_user_info": one line per assembly with any such
                                      row or redraw, for the user next to flags_for_user (collect_smoke_check adds
                                      them to show_user); they are not flags (no share limit)
           plus contact_report_error rows by stage, label_failed rows by (completed, inserted_end) with their end
           heights, and the lag stops of failed legs (max_lag_mm, and how many had plug and finger forces <= 1 N;
           legs of successful placements are not stored by the collector). Any share above DRAW_FLAG_SHARE (5 %,
           round-4 lead resolution) on any assembly is listed in flags_for_user: shown to the user before launch
           (it does not change the exit code).
  heights  the end heights h_T of the t00 references and of the per-reset label references (grasp.json
           reference_success / label_reference.success height_mm), overall and per assembly: n, min, p5, median, the
           count with h_T <= 0.01 mm (H_EDGE_MM) and with h_T <= 0 (fails AutoMate's strict 0 < h)
Exit code 1 (after check.json is written) when: a file fails the format / count checks (keys, counts, img_step, dtype /
rank / size, missing meta keys) or cannot be read; any reader (demo320.convert / policy_sample, wmdata320.trajectory)
raised; load_entries raised; outcome_inserted_vs_success_any_differ > 0; recorded_vs_recomputed_success_any_differ
> 0; any trajectory file of any kind has start_success True (start_success_rows: its start state already satisfies
AutoMate's rule, impossible under "socket_top" starts, r5-expert-06); plan.json is missing, a planned grasp has no
grasp.json, or a build_failures file holds a counted failure. --draws-only always exits 0.
The reasons are listed in check.json "exit_reasons".
Media: rgb + depth pictures of every 10th picture of one trajectory per kind, and a GIF (rgb | depth) of one insert.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wm.data import schema, store  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as W  # noqa: E402


def load_ref(ref_dir: str | Path) -> dict:
    """kind -> (array signature, meta, path) of the first wm_v1 file of each kind in a wm_v1 grasp directory."""
    ref = {}
    for rp in sorted(Path(ref_dir).glob("t*.npz")):
        m = store.read_meta(rp)
        if m["kind"] not in ref:
            with np.load(rp) as z:
                ref[m["kind"]] = ({k: (str(z[k].dtype), z[k].ndim, z[k].shape[1:]) for k in z.files if k != "meta"}, m, str(rp))
    return ref


def depth_rgb(d: np.ndarray) -> np.ndarray:
    v = d.astype(np.float32) * 0.1                                         # mm
    ok = v > 0
    lo, hi = (np.percentile(v[ok], 1), np.percentile(v[ok], 99)) if ok.any() else (0, 1)
    x = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.stack([x, 1 - np.abs(2 * x - 1), 1 - x], -1)
    out[~ok] = 0
    return (out * 255).astype(np.uint8)


def spec_of(meta: dict):
    """(spec, source): the assembly's AssemblySpec from the asset folder, or (no spec file) the meta's own values."""
    try:
        return A.load_spec(meta["assembly_id"]), "assets"
    except FileNotFoundError:
        if all(k in meta for k in ("tip_centre_m", "symmetry_n", "disassembly_dist_m")):
            return types.SimpleNamespace(tip_centre_m=tuple(meta["tip_centre_m"]), symmetry_n=meta["symmetry_n"],
                                         disassembly_dist_m=float(meta["disassembly_dist_m"])), "meta"
        raise


def truth_of(eval_rel: np.ndarray, meta: dict) -> dict:
    """The truth block of check_file (module docstring, truth)."""
    spec, source = spec_of(meta)
    rel = np.asarray(eval_rel, np.float64)
    s = checks.success_steps(rel, spec)
    t = checks.truth_from_rel_spec(rel[-1], spec)
    end = {k: (bool(v) if k in ("success", "seated") else float(v)) for k, v in t.items()}
    out = {"spec_source": source, "truth_end": end, "success_end": s["success_end"], "success_any": s["success_any"],
           "first_success_step": s["first_success_step"], "start_success": s["start_success"],
           "h_end_mm": end["height_mm"], "depth_left_end_mm": end["depth_left_mm"]}
    if source == "assets" and "disassembly_dist_m" in meta:
        out["meta_disassembly_dist_differs"] = bool(abs(float(meta["disassembly_dist_m"]) - spec.disassembly_dist_m) > 1e-9)
    return out


def check_file(p: Path, data: Path, ref: dict, kin) -> tuple[dict, dict, dict]:
    """One trajectory file (module docstring); kin: wm.sim_automate.kinematics.Kinematics, or None (no FK values)."""
    out = {"file": str(p.relative_to(data))}
    ref_raw, ref_meta, out["ref_file"] = ref[store.read_meta(p)["kind"]]
    with np.load(p) as z:
        raw = {k: (str(z[k].dtype), z[k].ndim, z[k].shape[1:]) for k in z.files if k != "meta"}
    own = {k for k in set(raw) - set(ref_raw) if k.startswith("eval_")}          # AutoMate's own truth arrays (eval_tip)
    out["keys_equal"] = set(raw) - own == set(ref_raw)
    out["key_diff"] = sorted((set(raw) - own) ^ set(ref_raw))
    out["own_truth_arrays"] = sorted(own)
    out["dtype_rank_size_mismatch"] = {k: [raw[k], ref_raw[k]] for k in raw if k in ref_raw and k != "rgb_jpeg"
                                       and k != "rgb_offsets" and raw[k] != ref_raw[k]}
    a, meta = store.load(p)
    out["meta_missing_keys"] = sorted(set(ref_meta) - set(meta))
    out["meta_extra_keys"] = sorted(set(meta) - set(ref_meta))
    out["schema"] = meta.get("schema")
    T, N = len(a["action"]), len(a["img_step"])
    out.update(kind=meta["kind"], outcome=meta.get("outcome"), steps=T, pictures=N, assembly_id=meta.get("assembly_id"),
               expert_completed=meta.get("expert_completed"), automate_success_any_meta=meta.get("automate_success_any"))
    out["counts_ok"] = bool(len(a["qpos"]) == len(a["hand"]) == len(a["eval_rel"]) == T + 1 and len(a["phase"]) == T
                            and len(a["rgb"]) == len(a["depth"]) == len(a["seg"]) == len(a["cam_T_world"]) == N
                            and meta.get("steps") == T)
    exp = list(range(0, T + 1, schema.IMAGE_EVERY))
    if exp[-1] != T:
        exp.append(T)
    out["img_step_ok"] = bool(np.array_equal(a["img_step"], exp))
    out["rgb_shape"], out["depth_shape"] = list(a["rgb"].shape[1:]), list(a["depth"].shape[1:])
    act = a["action"].astype(np.float64)
    if kin is not None:
        idx = np.unique(np.linspace(0, T, 12).astype(int))
        fk = np.stack([kin.tcp(q) for q in a["qpos"][idx].astype(np.float64)])
        out["hand_vs_fk_max_mm"] = float(np.abs(fk[:, :3, 3] - a["hand"][idx, :3, 3]).max() * 1e3)
        cam = np.stack([np.linalg.inv(kin.link("panda_hand", a["qpos"][s].astype(np.float64)) @ C.T_HAND_CAM_CV)
                        for s in a["img_step"][:: max(1, N // 6)]])
        out["cam_T_world_vs_fk_max_mm"] = float(np.abs(cam[:, :3, 3] - a["cam_T_world"][:: max(1, N // 6), :3, 3]).max() * 1e3)
        out["action_in_limits"] = bool(((act >= kin.lower[:7] - 1e-6) & (act <= kin.upper[:7] + 1e-6)).all())
    out["action_finite"] = bool(np.isfinite(act).all())
    out["action_max_step_rad"] = float(np.abs(np.diff(act, axis=0)).max()) if T > 1 else 0.0
    out["action_vs_next_qpos_max_rad"] = float(np.abs(act - a["qpos"][1:, :7]).max())
    d = a["depth"]
    out["depth_return_fraction"] = float((d > 0).mean())
    out["depth_mm_p1_p99"] = [float(np.percentile(d[d > 0], 1) * 0.1), float(np.percentile(d[d > 0], 99) * 0.1)] if (d > 0).any() else None
    out["seg_fraction"] = {c: float((a["seg"] == i).mean()) for i, c in enumerate(schema.SEG_CLASSES)}
    out["phases"] = {schema.PHASES[int(k)]: int(v) for k, v in zip(*np.unique(a["phase"], return_counts=True))}
    try:
        out.update(truth_of(a["eval_rel"], meta))
    except (FileNotFoundError, KeyError) as exc:              # data recorded before 2026-11-22 (no spec / ids)
        out.update(truth_end={"error": f"{type(exc).__name__}: {exc}"}, success_end=None, success_any=None,
                   first_success_step=None, start_success=None, h_end_mm=None, depth_left_end_mm=None)
    if "eval_tip" in a:
        cols = meta.get("eval_tip_columns") or []
        out["eval_tip_end"] = {c: float(v) for c, v in zip(cols, np.asarray(a["eval_tip"][-1], np.float64))}
    pk = meta.get("peak") or {}
    out["failure"] = meta.get("failure")
    out["peak_finger_socket_N"] = pk.get("finger_socket_N")
    out["H_star_in_meta"] = bool(np.asarray(meta["H_star"]).shape == (4, 4)) and meta.get("axis") is not None
    return out, a, meta


def completed(f: dict) -> bool:
    """The expert completed: the meta's expert_completed, or (older data) no failure recorded."""
    c = f.get("expert_completed")
    return bool(c) if c is not None else f.get("failure") is None


def failure_kind(text) -> str | None:
    return None if text is None else ":".join(str(text).split(":")[:2]).strip()


def outcome_counts(files: list[dict]) -> dict:
    """The outcome counts of the module docstring over insert / perturbed trajectories."""
    ins = [f for f in files if "error" not in f and f["kind"] in ("insert", "perturbed")]
    known = [f for f in ins if f["success_any"] is not None]
    return {
        "insert_perturbed": len(ins),
        "without_truth": len(ins) - len(known),
        "outcome_inserted_vs_success_any_differ": sum((f["outcome"] == "inserted") != f["success_any"] for f in known),
        "recorded_vs_recomputed_success_any_differ": sum(f["automate_success_any_meta"] is not None
                                                         and bool(f["automate_success_any_meta"]) != f["success_any"] for f in known),
        "success_any_not_completed": sum(f["success_any"] and not completed(f) for f in known),
        "completed_not_success_end": sum(completed(f) and not f["success_end"] for f in known),
        "completed_not_success_any": sum(completed(f) and not f["success_any"] for f in known),
        "success_any_not_end": sum(f["success_any"] and not f["success_end"] for f in known),
        "start_success": sum(bool(f["start_success"]) for f in known),
        "bc": sum(f["outcome"] == "inserted" and completed(f) for f in ins),
        "failure_kinds": dict(Counter(failure_kind(f["failure"]) for f in ins if f["failure"] is not None)),
        "peak_finger_socket_gt_1N": sum((f["peak_finger_socket_N"] or 0.0) > 1.0 for f in ins),
        "success_end_spin_ge_4_deg": sum(bool(f["success_end"]) and f["truth_end"].get("spin_deg", 0.0) >= 4.0 for f in known)}


def per_assembly(files: list[dict]) -> dict:
    """Per assembly (module docstring): smallest end height of completed insert / perturbed / reference trajectories,
    completed_not_success_end, start_success, first_success_step range, references completed / success_end."""
    by = defaultdict(list)
    for f in files:
        if "error" not in f and f.get("success_any") is not None:
            by[f["assembly_id"]].append(f)
    out = {}
    for aid, fs in sorted(by.items()):
        rec = [f for f in fs if f["kind"] in ("insert", "perturbed", "reference")]
        done = [f for f in rec if completed(f)]
        fss = [f["first_success_step"] for f in rec if f["first_success_step"] is not None]
        refs = [f for f in fs if f["kind"] == "reference"]
        out[aid] = {"trajectories": len(fs),
                    "min_h_end_mm_completed": min((f["h_end_mm"] for f in done), default=None),
                    "completed_not_success_end": sum(not f["success_end"] for f in done),
                    "start_success": sum(bool(f["start_success"]) for f in fs),
                    "first_success_step_min_max": [min(fss), max(fss)] if fss else None,
                    "references": len(refs), "references_completed_and_success_end":
                        sum(completed(f) and bool(f["success_end"]) for f in refs)}
    return out


DRAW_FLAG_SHARE = 0.05           # round-4 lead resolution: a rate above 5 % on any assembly is shown to the user
H_EDGE_MM = 0.01                 # an end height at or below this sits at the edge of AutoMate's strict h > 0
LAG_LOW_FORCE_N = 1.0            # a lag stop with plug and finger forces at or below this is a pure tracking stop


def assembly_of_grasp(g: dict) -> str:
    return str(g.get("assembly_id") or str(g.get("grasp", "")).rsplit("_g", 1)[0])


def redraw_reason(f: dict) -> str:
    """The reason of one placement failure (module docstring, draws)."""
    st = f.get("stage") or "start_error"
    failed = [x for x in (f.get("legs") or []) if x.get("leg") == st and not x.get("ok")]
    r = failed[0].get("reason") if failed else None
    return f"{st}:{r}" if r else str(st)


def height_stats(hs: list) -> dict:
    """n, min, p5, median and the counts h <= H_EDGE_MM and h <= 0 of end heights [mm] (module docstring, heights)."""
    v = np.asarray([h for h in hs if h is not None], np.float64)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "min_mm": float(v.min()), "p5_mm": float(np.percentile(v, 5)),
            "median_mm": float(np.median(v)), "le_0.01mm": int((v <= H_EDGE_MM).sum()), "le_0mm": int((v <= 0.0).sum())}


def t00_cause(g: dict) -> str | None:
    """None if the t00 reference was accepted, else its cause (module docstring, draws)."""
    if g.get("reference_ok"):
        return None
    soft = g.get("reference_soft_outcome")
    if soft:
        return str(soft)
    rs = g.get("reference_success") or {}
    if g.get("reference_expert_completed") is False:
        return "expert did not complete"
    if g.get("reference_expert_completed") is True and rs.get("inserted_end") is False:
        return "completed, AutoMate rule fails at the end"
    if g.get("error") and "reference_ok" not in g:
        return "grasp error before the reference was judged"
    return "other: " + str(g.get("reference_failure") or g.get("error") or "no reason")[:80]


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


BAND_KEYPOINT_M = 0.015          # AutoMate's keypoint-distance bound of the success rule (checks.py, k < 15 mm)
BAND_ENTRANCE_LIST_MM = 2.0      # band_above_entrance_mm above this is listed in "all" (information, no flag)
CONTACT_NO_INSERT = ("jammed", "guard:")   # contact-trial outcomes without insertion (expert.py contact outcomes)
RANDOM_TRUNCATED = ("ExpertFailure: step_budget", "_did_not_reach_target")   # prefix / substring (random_truncated)
GUARD_STOPS = ("socket_contact_force_guard", "socket_penetration_guard")      # the 80 N / 0.5 mm guards (expert._step)
HAND_FINGER_COMMAND = "hand_finger_command_clearance"     # decision 7 of 2026-11-23 (expert._send failure text)


def hand_finger_rows(rows: list[dict], grasps: list[dict] = ()) -> dict:
    """Per-kind counts of the module docstring's hand / finger rows over grasp.json trajectory rows (information);
    hand_finger_command_clearance also counts the unrecorded references stopped by it: per-reset label references of
    rows (label_reference.failure) under "label", t00 references of grasps (reference_failure) under "t00"."""
    fail = lambda t: str(t.get("failure") or "")  # noqa: E731
    rnd = [t for t in rows if t.get("kind") == "random" and "waypoint_redraws" in t]
    hfc = Counter(str(t.get("kind")) for t in rows if HAND_FINGER_COMMAND in fail(t))
    hfc.update({"label": sum(HAND_FINGER_COMMAND in str((t.get("label_reference") or {}).get("failure") or "")
                             for t in rows),
                "t00": sum(HAND_FINGER_COMMAND in str(g.get("reference_failure") or "") for g in grasps)})
    return {"finger_contact_rows_by_kind": dict(Counter(str(t.get("kind")) for t in rows if t.get("finger_contact"))),
            "guard_stopped_rows_by_kind": dict(Counter(str(t.get("kind")) for t in rows
                                                       if any(x in fail(t) for x in GUARD_STOPS))),
            "hand_finger_command_clearance_rows_by_kind": {k: v for k, v in hfc.items() if v},
            "waypoint_redraws": {"rejected_draws": sum(len(t["waypoint_redraws"] or []) for t in rnd),
                                 "rows_with_redraw": sum(bool(t["waypoint_redraws"]) for t in rnd), "of": len(rnd)}}


def random_truncated(t: dict) -> bool:
    """A random-motion row that stopped at the step budget or before a waypoint (module docstring, random_truncated)."""
    f = str(t.get("failure") or "")
    return f.startswith(RANDOM_TRUNCATED[0]) or RANDOM_TRUNCATED[1] in f


def start_success_rows(g: dict) -> list[str]:
    """'<grasp> tNN' of the rows whose start state satisfies AutoMate's rule (module docstring, start_success_rows)."""
    rows = g.get("trajectories", [])
    out = [f"{g.get('grasp')} t{int(t.get('idx', 0)):02d}" for t in rows
           if (t.get("success") or {}).get("success_at_start")]
    if not any(t.get("idx", 0) == 0 for t in rows) and (g.get("reference_success") or {}).get("success_at_start"):
        out.append(f"{g.get('grasp')} t00")
    return out


def band_above_top_mm(spec) -> float:
    """min(d_dis, 15 mm) - L_top [mm] (module docstring, band_above_top_mm)."""
    return (min(float(spec.disassembly_dist_m), BAND_KEYPOINT_M) - float(spec.depth_top_m)) * 1e3


def band_above_entrance_mm(spec) -> float:
    """min(d_dis, 15 mm) - L [mm] (module docstring, band_above_entrance_mm)."""
    return (min(float(spec.disassembly_dist_m), BAND_KEYPOINT_M) - float(spec.pin_length_m)) * 1e3


def draw_report(grasps: list[dict], planned: dict | None = None, specs: dict | None = None) -> tuple[dict, list[str]]:
    """Per assembly the draw, refusal, redraw and end-height numbers of the module docstring (draws, heights), from
    grasp.json dicts; planned: {grasp key: planned trajectories} (plan.json) for the trajectories lost with a refused
    t00; specs: {assembly id: AssemblySpec} for band_above_top_mm and band_above_entrance_mm (an assembly without
    one gets None).
    -> (report {"per_assembly": ..., "all": ...}, flags_for_user)."""
    by = defaultdict(list)
    for g in grasps:
        by[assembly_of_grasp(g)].append(g)
    per, flags = {}, []
    tot = Counter()
    all_h = {"t00": [], "label": []}
    hf_all = {k: Counter() for k in hand_finger_rows([])}
    hf_lines = []
    for aid, gs in sorted(by.items()):
        c, reasons, causes, cre, lab = Counter(), Counter(), Counter(), Counter(), Counter()
        h_t00, h_lab, lags, lost = [], [], [], 0
        refused_h_le_0 = 0
        start_rows = []
        for g in gs:
            c["grasps"] += 1
            start_rows += start_success_rows(g)
            for t in g.get("trajectories", []):
                if t.get("kind") == "contact":
                    c["contact_rows"] += 1
                    o = str(t.get("outcome"))
                    c["contact_no_insert_with_automate_success"] += bool(
                        t.get("automate_success_any") and (o == "jammed" or o.startswith(CONTACT_NO_INSERT[1])))
                if t.get("kind") == "random" and "expert_completed" in t:
                    c["random_rows"] += 1
                    c["random_truncated"] += random_truncated(t)
            c["grasp_errors"] += bool(g.get("error"))
            c["draws"] += 1                                          # the t00 reset
            cause = t00_cause(g)
            rs = g.get("reference_success") or {}
            if rs.get("height_mm") is not None:
                h_t00.append(float(rs["height_mm"]))
            if cause is not None:
                causes[cause] += 1
                c["t00_refused"] += 1
                c["reset_failed"] += cause == "reset_failed"
                lost += int((planned or {}).get(g.get("grasp"), 0))
                if g.get("reference_expert_completed") is True and rs.get("height_mm") is not None \
                        and float(rs["height_mm"]) <= 0.0:
                    refused_h_le_0 += 1
            for t in g.get("trajectories", []):
                if t.get("idx", 0) == 0:
                    continue
                c["draws"] += 1
                o = t.get("outcome")
                c["reset_failed"] += o == "reset_failed"
                if o == "contact_report_error":
                    cre[str(t.get("stage") or "run")] += 1
                    c["contact_report_error"] += 1
                lr = t.get("label_reference") or {}
                if (lr.get("success") or {}).get("height_mm") is not None:
                    h_lab.append(float(lr["success"]["height_mm"]))
                if o == "label_failed":
                    c["label_failed"] += 1
                    lab[f"completed={lr.get('completed')} inserted_end={lr.get('inserted_end')}"] += 1
                    hh = (lr.get("success") or {}).get("height_mm")
                    if lr.get("completed") is not False and lr.get("inserted_end") is False and hh is not None \
                            and float(hh) <= 0.0:
                        refused_h_le_0 += 1
                if "placement_attempts" in t:
                    c["placement_rows"] += 1
                    c["attempts"] += int(t["placement_attempts"])
                    c["placement_failed"] += o == "placement_failed"
                    for f in t.get("placement_failures") or []:
                        reasons[redraw_reason(f)] += 1
                        for x in f.get("legs") or []:
                            if x.get("reason") == "lag":
                                lags.append((x.get("leg"), x.get("max_lag_mm"),
                                             (x.get("peak_plug_socket_N") or 0.0) <= LAG_LOW_FORCE_N
                                             and (x.get("peak_finger_socket_N") or 0.0) <= LAG_LOW_FORCE_N))
        c["start_success_rows"] = len(start_rows)
        shares = {"reset_failed": _share(c["reset_failed"], c["draws"]),
                  "label_failed": _share(c["label_failed"], c["draws"]),
                  "contact_report_error": _share(c["contact_report_error"], c["draws"]),
                  "t00_refused": _share(c["t00_refused"], c["grasps"]),
                  "placement_failed": _share(c["placement_failed"], c["placement_rows"]),
                  "random_truncated": _share(c["random_truncated"], c["random_rows"]),
                  **{f"redraw {r}": _share(k, c["attempts"]) for r, k in sorted(reasons.items())}}
        lag_mm = [v for _, v, _ in lags if v is not None]
        hf = hand_finger_rows([t for g in gs for t in g.get("trajectories", [])], gs)
        for k, v in hf.items():
            hf_all[k].update(v)
        if any(hf[k] for k in ("finger_contact_rows_by_kind", "guard_stopped_rows_by_kind",
                               "hand_finger_command_clearance_rows_by_kind")) or hf["waypoint_redraws"]["rejected_draws"]:
            wr = hf["waypoint_redraws"]
            hf_lines.append(f"{aid} (information): finger_contact rows {hf['finger_contact_rows_by_kind']}, 80 N / 0.5 mm "
                            f"guard stops {hf['guard_stopped_rows_by_kind']}, hand_finger_command_clearance failures "
                            f"{hf['hand_finger_command_clearance_rows_by_kind']}, random waypoints redrawn in "
                            f"{wr['rows_with_redraw']} of {wr['of']} random rows ({wr['rejected_draws']} rejected draws)")
        per[aid] = {"counts": dict(c), "shares": shares, "t00_refused_by_cause": dict(causes), **hf,
                    "trajectories_lost_with_t00": lost if planned is not None else None,
                    "refused_completed_with_h_le_0": refused_h_le_0,
                    "label_failed_by": dict(lab), "contact_report_error_by_stage": dict(cre),
                    "redraws_by_reason": dict(reasons),
                    "lag_stops": {"n": len(lags), "low_force": sum(lf for *_, lf in lags),
                                  "by_leg": dict(Counter(str(leg) for leg, *_ in lags)),
                                  "max_lag_mm_p50_p95_max": [float(np.percentile(lag_mm, q)) for q in (50, 95, 100)]
                                  if lag_mm else None},
                    "end_height_t00": height_stats(h_t00), "end_height_label": height_stats(h_lab),
                    "start_success_rows": start_rows,
                    "contact_no_insert_with_automate_success_share": _share(c["contact_no_insert_with_automate_success"],
                                                                            c["contact_rows"]),
                    "band_above_top_mm": None if (specs or {}).get(aid) is None else band_above_top_mm(specs[aid]),
                    "band_above_entrance_mm": None if (specs or {}).get(aid) is None
                    else band_above_entrance_mm(specs[aid]),
                    "random_truncated": {"n": c["random_truncated"], "of": c["random_rows"],
                                         "share": _share(c["random_truncated"], c["random_rows"])}}
        if start_rows:
            flags.append(f"{aid}: {len(start_rows)} trajectories start in a state that already satisfies AutoMate's rule "
                         f"(impossible under socket_top starts: a wrong start), e.g. {start_rows[:3]}")
        for name, v in shares.items():
            if v is not None and v > DRAW_FLAG_SHARE:
                n = c["grasps"] if name == "t00_refused" else c["placement_rows"] if name == "placement_failed" else \
                    c["random_rows"] if name == "random_truncated" else \
                    c["attempts"] if name.startswith("redraw") else c["draws"]
                flags.append(f"{aid}: {name} {v:.1%} of {n} (> {DRAW_FLAG_SHARE:.0%})")
        tot.update(c)
        tot["refused_completed_with_h_le_0"] += refused_h_le_0
        all_h["t00"] += h_t00
        all_h["label"] += h_lab
    band = {aid: v["band_above_top_mm"] for aid, v in per.items() if v["band_above_top_mm"] is not None}
    above = {aid: b for aid, b in band.items() if b > 0}
    ent = sorted(((v["band_above_entrance_mm"], aid) for aid, v in per.items()
                  if v["band_above_entrance_mm"] is not None and v["band_above_entrance_mm"] > BAND_ENTRANCE_LIST_MM),
                 reverse=True)
    allr = {"counts": dict(tot), "end_height_t00": height_stats(all_h["t00"]),
            "end_height_label": height_stats(all_h["label"]),
            "refused_share_completed_with_h_le_0": _share(tot["refused_completed_with_h_le_0"],
                                                         tot["t00_refused"] + tot["label_failed"]),
            "contact_no_insert_with_automate_success_share": _share(tot["contact_no_insert_with_automate_success"],
                                                                    tot["contact_rows"]),
            "random_truncated": {"n": tot["random_truncated"], "of": tot["random_rows"],
                                 "share": _share(tot["random_truncated"], tot["random_rows"])},
            **{k: dict(v) for k, v in hf_all.items()}, "show_user_info": hf_lines,
            "band_above_top": {"assemblies_with_spec": len(band), "above_top": len(above),
                               "largest_mm": max(above.values(), default=None),
                               "text": (f"for {len(above)} of {len(band)} assemblies AutoMate's rule counts a plug resting "
                                        f"on the socket top within up to {max(above.values()):.2f} mm as success")
                               if above else None},
            "band_above_entrance": {"listed_above_mm": BAND_ENTRANCE_LIST_MM,
                                    "assemblies": {aid: b for b, aid in ent},
                                    "text": ("AutoMate's rule is met up to this height above the entrance of the tight "
                                             "fit, so a success there (insert / perturbed 'inserted' = S_any, contact "
                                             "automate_success_any) does not mean the tight fit was reached; the "
                                             "imitation set needs expert_completed: "
                                             + ", ".join(f"{aid} {b:.1f} mm" for b, aid in ent))
                                    if ent else None}}
    return {"per_assembly": per, "all": allr}, flags


def counted_build_failures(data: Path) -> list[str]:
    """build_failures/*.json files holding a counted failure (a row with counted true or without the key, as collect.py
    and collect_smoke_check count them; an unreadable file counts)."""
    out = []
    for f in sorted(Path(data).glob("build_failures/*.json")):
        try:
            rows = json.loads(f.read_text())
        except (OSError, ValueError):
            rows = [{}]
        rows = rows if isinstance(rows, list) else [rows]
        if any(not isinstance(r, dict) or r.get("counted", True) for r in rows):
            out.append(str(f.relative_to(data)))
    return out


def exit_reasons(report: dict, bad: list) -> list[str]:
    """Why the check ends with exit code 1 (module docstring); empty: exit 0."""
    why = [f"format / counts: {b['file']}" for b in bad[:20]]
    if len(bad) > 20:
        why.append(f"... {len(bad)} files in all")
    why += [f"reader: {r['file']}: {r['error']}"[:300] for r in report["readers"] if "error" in r][:20]
    if "error" in report.get("load_entries", {}):
        why.append(f"load_entries: {report['load_entries']['error']}"[:300])
    oc = report.get("outcome_counts", {})
    for k in ("outcome_inserted_vs_success_any_differ", "recorded_vs_recomputed_success_any_differ"):
        if oc.get(k):
            why.append(f"{k} = {oc[k]}")
    ss = report.get("start_success_rows") or []
    if ss:
        why.append(f"start state satisfies AutoMate's rule: {len(ss)} trajectories (e.g. {ss[:3]})")
    pl = report.get("plan", {})
    if pl.get("error"):
        why.append(pl["error"])
    if pl.get("without_grasp_json"):
        why.append(f"{len(pl['without_grasp_json'])} planned grasps without grasp.json")
    if pl.get("counted_build_failures"):
        why.append(f"counted build failures: {pl['counted_build_failures']}")
    return why


def save_media(work: Path, a: dict, tag: str, gif: bool):
    from PIL import Image
    imgs = []
    for j in range(len(a["rgb"])):
        both = np.concatenate([a["rgb"][j], depth_rgb(a["depth"][j])], 1)
        imgs.append(Image.fromarray(both))
        if j % 10 == 0 or j == len(a["rgb"]) - 1:
            imgs[-1].save(work / "media" / f"{tag}_{j:03d}.png")
    if gif:
        imgs[0].save(work / "media" / f"{tag}.gif", save_all=True, append_images=imgs[1:], duration=200, loop=0)


def read_grasp_jsons(data: Path) -> tuple[list[dict], list[str]]:
    """Every <data>/*/*/grasp.json as a dict, and the files that could not be read (e.g. being written)."""
    gjs, bad = [], []
    for gj in sorted(data.glob("*/*/grasp.json")):
        try:
            gjs.append(json.loads(gj.read_text()))
        except (OSError, ValueError) as exc:
            bad.append(f"{gj.relative_to(data)}: {type(exc).__name__}: {exc}"[:200])
    return gjs, bad


def plan_block(data: Path) -> tuple[dict, list | None]:
    """(report "plan", plan.json grasps or None): planned grasps, those without grasp.json, build-failure files."""
    plan_f = data / "plan.json"
    if not plan_f.exists():
        return {"error": f"no {plan_f}"}, None
    planned = json.loads(plan_f.read_text()).get("grasps", [])
    return {"grasps": len(planned),
            "without_grasp_json": [g["grasp"] for g in planned if not (data / g["port"] / g["grasp"] / "grasp.json").exists()],
            "build_failures": sorted(str(f.relative_to(data)) for f in data.glob("build_failures/*.json")),
            "counted_build_failures": counted_build_failures(data)}, planned


def specs_of(aids) -> dict:
    """{assembly id: AssemblySpec} from the asset spec files (json only); an assembly without one is left out."""
    out = {}
    for aid in aids:
        try:
            out[aid] = A.load_spec(aid)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            pass
    return out


def draws_only(data: Path, work: Path) -> dict:
    """--draws-only (module docstring): grasp.json + plan.json only; writes <work>/draws.json; never fails."""
    gjs, unreadable = read_grasp_jsons(data)
    try:
        plan, planned = plan_block(data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        plan, planned = {"error": f"plan.json unreadable: {type(exc).__name__}: {exc}"[:300]}, None
    try:                                                      # read-only facts (live workers, skipped assemblies)
        from wm.data.index import automate_completeness
        comp = automate_completeness(data)
    except Exception as exc:
        comp = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    aids = sorted({assembly_of_grasp(g) for g in gjs})
    draws, flags = draw_report(gjs, None if planned is None else {g["grasp"]: len(g.get("trajectories", [])) for g in planned},
                               specs_of(aids))
    rep = {"mode": "draws-only", "data": str(data), "assets": str(A.ASSET_DIR), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "grasp_json_read": len(gjs), "grasp_json_unreadable": unreadable, "plan": plan, "completeness": comp,
           "per_assembly": draws["per_assembly"], "all": draws["all"], "flags_for_user": flags}
    work.mkdir(parents=True, exist_ok=True)
    for name, text in (("draws.json", json.dumps(rep, indent=1, default=float)),
                       ("flags_for_user.txt", "".join(f + "\n" for f in flags))):
        tmp = work / f"{name}.tmp{os.getpid()}"
        tmp.write_text(text)
        os.replace(tmp, work / name)
    print(json.dumps({"grasp_json_read": len(gjs), "grasp_json_unreadable": unreadable[:5],
                      "planned": plan.get("grasps"), "without_grasp_json": len(plan.get("without_grasp_json") or []),
                      "skipped_assemblies": comp.get("skipped_assemblies"), "live_workers": len(comp.get("live_workers") or []),
                      "band_above_top": draws["all"]["band_above_top"]["text"],
                      "band_above_entrance": draws["all"]["band_above_entrance"]["text"],
                      "random_truncated": draws["all"]["random_truncated"], "draws_json": str(work / "draws.json"),
                      "flags_for_user": flags}, indent=1, default=float))
    return rep


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", default="/mnt/storage2/qingxin/data/wm_v1/A/A_g00", help="a wm_v1 grasp directory")
    ap.add_argument("--work", required=True)
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--draws-only", action="store_true", help="grasp.json + plan.json only (module docstring); exit 0")
    args = ap.parse_args(argv)
    data, work = Path(args.data), Path(args.work)
    if (data / "inputs" / "assets").is_dir():                 # the collection's frozen asset specs
        A.ASSET_DIR = data / "inputs" / "assets"
    if args.draws_only:
        return draws_only(data, work)
    (work / "media").mkdir(parents=True, exist_ok=True)
    from wm.sim_automate.kinematics import Kinematics
    ref, kin = load_ref(args.ref), Kinematics()
    report = {"ref": {k: v[2] for k, v in ref.items()}, "assets": str(A.ASSET_DIR), "files": [], "readers": [], "grasps": []}
    fast = work / "fast"
    fast.mkdir(parents=True, exist_ok=True)
    entries, shown, gjs = [], set(), []
    for gj in sorted(data.glob("*/*/grasp.json")):
        g = json.loads(gj.read_text())
        gjs.append(g)
        report["grasps"].append({k: g.get(k) for k in ("grasp", "reference_ok", "reference_failure", "error", "seconds",
                                                        "timing", "memory", "H_star_fk_vs_model_mm_deg", "axis_vs_true_deg")}
                                | {"outcomes": dict(Counter(t.get("outcome") for t in g.get("trajectories", [])))})
        for p in sorted(gj.parent.glob("t*.npz")):
            try:
                res, a, meta = check_file(p, data, ref, kin)
            except Exception as exc:
                report["files"].append({"file": str(p), "error": f"{type(exc).__name__}: {exc}"})
                continue
            report["files"].append(res)
            kind = meta["kind"]
            if kind not in shown and not args.no_media:
                shown.add(kind)
                save_media(work, a, f"{meta['grasp']}_{p.stem}", gif=kind == "insert")
            rel = str(p.relative_to(data))
            r = {"file": rel}
            try:
                e = D.convert(p, fast / rel, meta["pin_length_m"], rel, allow_test=True)
                entries.append(e)
                s = D.policy_sample(fast / rel, min(5, e["n_images"] - 1))
                r["policy_sample"] = {k: list(v.shape) for k, v in s.items()}
                with np.load(fast / rel) as z:
                    m2 = json.loads(str(z["meta"]))
                    tr = W.trajectory(z, m2, shaping=0.3, gamma=0.967)
                r["wmdata320"] = None if tr is None else {"decisions": tr["n"], "end_seated_fk": tr["end_seated"],
                                                           "final_depth_label_mm": float(tr["depth"][-1] * 1e3),
                                                           "reward_sum": float(tr["reward"].sum())}
            except Exception as exc:
                r["error"] = f"{type(exc).__name__}: {exc}"
            report["readers"].append(r)
    (fast / "manifest.json").write_text(json.dumps({"format": D.FORMAT, "source": str(data), "skipped": {}, "entries": entries}))
    try:
        le = D.load_entries(fast, allow_undecided=True)                  # the split may still be undecided while collecting
        report["load_entries"] = {"n": len(le), "splits": {s: sum(e["split"] == s for e in le) for s in sorted({e["split"] for e in le})},
                                  "bc": sum(e["bc"] for e in le)}
    except Exception as exc:
        report["load_entries"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["outcome_counts"] = outcome_counts(report["files"])
    report["per_assembly"] = per_assembly(report["files"])
    report["start_success_rows"] = [f["file"] for f in report["files"] if "error" not in f and f.get("start_success")]
    report["plan"], planned = plan_block(data)
    report["draws"], report["flags_for_user"] = draw_report(
        gjs, None if planned is None else {g["grasp"]: len(g.get("trajectories", [])) for g in planned},
        specs_of(sorted({assembly_of_grasp(g) for g in gjs})))
    bad = [f for f in report["files"] if "error" in f or not (f["keys_equal"] and f["counts_ok"] and f["img_step_ok"]
                                                             and not f["dtype_rank_size_mismatch"] and not f["meta_missing_keys"])]
    report["exit_reasons"] = exit_reasons(report, bad)
    (work / "check.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({"files": len(report["files"]), "bad": [b["file"] for b in bad],
                      "reader_errors": [r for r in report["readers"] if "error" in r],
                      "load_entries": report["load_entries"], "outcome_counts": report["outcome_counts"],
                      "per_assembly": report["per_assembly"],
                      "draws": {aid: {"shares": v["shares"], "t00_refused_by_cause": v["t00_refused_by_cause"],
                                      "end_height_t00": v["end_height_t00"], "end_height_label": v["end_height_label"]}
                                for aid, v in report["draws"]["per_assembly"].items()},
                      "draws_all": report["draws"]["all"], "flags_for_user": report["flags_for_user"],
                      "start_success_rows": report["start_success_rows"][:20],
                      "plan": {k: v if k != "without_grasp_json" else v[:20] for k, v in report["plan"].items()},
                      "exit_reasons": report["exit_reasons"]}, indent=1, default=float))
    if report["exit_reasons"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
