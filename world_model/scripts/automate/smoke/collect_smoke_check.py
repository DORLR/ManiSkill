"""Acceptance checks of the collector smoke (collect_smoke.sh phase 4; finding r3-ops-01). CPU only, no Isaac.

    WM_AUTOMATE_ASSETS=<out>/inputs/assets python -u collect_smoke_check.py --out <smoke out>

Items (PASS / FAIL; numbers marked INFO are reported for the user and the lead, they decide nothing here):
  (a) completeness: every planned grasp has grasp.json, none has "error"; no process death (deaths/), no counted
      build failure; logs/summary.json done == grasps; every grasp.json claim_attempt == 1 (the exit 7 and the STOP of
      phase 2 released their claims; neither counted as an attempt)
  (b) per-reset socket draw: per grasp, resets >= 1 + planned trajectories; every planned trajectory row has
      socket_index == its file index and (unless reset_failed / contact_report_error) a label_reference with
      "accepted"; label_failed rows have accepted False. INFO: reference_ok, label_failed, inserted,
      expert_completed, success_at_start, first_success_step, contact outcomes; end heights h [mm] of the accepted
      and refused t00 references and of the per-reset label references: min, p5, count h <= 0.01 mm (AutoMate's
      h > 0 is strict; r4-expert-02). FAIL (r5-expert-06): any trajectory row (t00 included) whose success
      .success_at_start is True, or a refused t00 whose reference_success.success_at_start is True: starts lie
      >= 6 mm above the socket top (b = L + g_top + g0) and AutoMate's success band reaches at most 2 mm above it, so
      a start that already satisfies the rule means a wrong start (g_top not applied, a stale spec)
  (c) files and meta: every row with "file" exists, every t*.npz is listed; eval_tip has shape (T + 1, 9) with
      eval_tip_columns = checks.TRUTH_COLUMNS; meta plan_sha / config_sha / switches equal plan.json's; insert /
      perturbed: outcome "inserted" iff automate_success_any, and automate_success_any = OR_{t >= 1} s_t recomputed
      from eval_rel (checks.truth_from_rel_spec); contact never "inserted"; t00 accepted only with expert_completed and
      success at the end; every grasp.json usd_vs_m1b == "equal" (the scene's plug / socket / plug_sensor USD layers
      are those of the milestone-1b (ii) result)
  (d) resources (from grasp.json memory, per worker pid): GPU-0 memory of our process <= 50 MiB at every grasp (FAIL
      otherwise); INFO: memory.peak, GPU-1 MiB, seconds per grasp, worker_gpu_gb_recommended = ceil(1.3 x G1) (the
      production config's worker_gpu_gb, also when below the default collect.WORKER_GPU_GB = 7: caps.sh charges the
      declaration, so n = min(6, floor((F - 12 GB) / worker_gpu_gb)) workers fit at GPU-1 free memory F; r7-ops-05,
      r11-ops-01) with G1 = the
      largest GPU-1 memory of one worker, max(grasp.json memory at the end of each grasp, the 60 s samples of
      smoke/samples.txt for the pids of our qingxin-collect-* units), and the production time estimate (15 grasps x
      included assemblies, 4 and 6 workers)
  (e) operations: smoke/samples.txt: watch.alive at most 180 s old at every sample, no pid of our units on GPU 0
      beyond graphics <= 50 MiB, no heartbeat file older than HANG_TIMEOUT_S / 2 at a sample, and no grasp.json
      timing.heartbeat_max_gap_s above HANG_TIMEOUT_S / 2 (INFO: the largest gap per stage); launcher.log has no
      line matching LAUNCHER_PROBLEM ("crash", "rate cap", "shed by", or "hang" not followed by " timeout": the
      launch line of collect_launch.sh names the "hang timeout" and is not a problem, r6-ops-01); ALERT lines and
      caps refusals (INFO, listed)
  (f) readers, through the production launcher scripts/automate/prep_automate_data.sh (frozen copy in
      smoke/check/prep/code, capped qx_run scopes, GPU 0): build_index.py --automate, cache_wm_v1_fast.py (into
      smoke/check/fast) and depth_stats.py (D_max = --depth-max-m, a smoke value only; the production value is the
      user's choice, not decided yet) in sequence, after the launcher's draws check (non-failing; the index step
      also checks the collection is finished and complete); check_expert_data.py (--no-media) exits 0 (its
      exit_reasons listed). PASS when the launcher ends "ok" with the steps draws, index, fast, depth_stats and the
      depth_stats.json it wrote carries the manifest's split and the train_files_sha256 of the training entries
      selected as train_wm320.py selects them, or when its only failure is the depth_stats step with
      check_depth_settings' refusal ("unusable statistics", too few pictures in 36 grasps); index / fast must pass. Every step's unit and pid are listed (INFO).
      In both cases <fast>/depth_distribution.json must exist (depth_stats.py writes it before any refusal; r9-data-01):
      the unclipped depth distribution of the training pictures (n_valid, p50 / p90 / p95 / p99 / p99.9 in m, share
      above 0.3 / 0.5 / 1.0 / 3.2767 m), copied into the INFO for the user's D_max question. check_expert_data's
      depth_mm_p1_p99 covers every condition (held-out and test too) and must not be used for D_max.
      The launcher's draws step must have written flags_for_user.txt equal to the flags of (h) (draw_report on the
      same grasp.json files; a cross-check of the production path, r6-integration-01)
  (g) exit 7: smoke/w2.end.phase2 names exit 7, smoke/STOP.g has the hook line ("forbidden context on GPU 0 ...
      released (not a death)"), no deaths/ file, the grasp whose claim the hook released has grasp.json without error
  (h) placement and rejection per assembly (INFO, r4-decisions-02 / -N2, r4-expert-04): "draws" = the per-assembly
      report of check_expert_data.draw_report (the production definitions: shares, refused t00 by cause, trajectories
      lost with a refused t00, refused_completed_with_h_le_0, end heights, redraws by reason, lag stops); from
      placement_stats: planned rows, outcomes, placement attempts and failed attempts by leg reason (clearance;
      retreat / transfer / descend with lag, socket_force, socket_penetration, ik, settle, plug_socket_contact,
      ik_or_settle; start_error; re-home reset); max_lag_mm p50 / p95 / max per leg over all attempts; refused t00
      references by cause with h and keypoint distance (soft / not_completed / rule_fails_at_end / other, the
      causes of check_expert_data.t00_cause: e.g. 'no axial push (FK axis undefined)' is other, r8-decisions-chk-01); actual_top_gap_lt_1mm (information only, no FAIL; deviation
      row 24, r7-decisions-01): {"n": trajectory rows, t00 included, whose expert peak min_actual_top_gap_mm_outside
      is < 1 mm (they would have tripped Chongyu's 1 mm actual tip-gap guard before insertion), "of": rows with that
      value, "min_mm"}; actual_top_gap_lt_2mm the same with 2 mm (r9-decisions-01, rows 24 / 32: Chongyu's 2 mm gap
      on every outside command; counted from the simulated plug pose, every outside step including the state after
      the last one, not from his FK-modelled plug, so it is an indication, not his exact rule). SHOW THE USER (before the launch; not a FAIL; r5-expert-05,
      r6-integration-01 / r6-decisions-01 / r6-expert-03): every flag of check_expert_data.draw_report, the same list
      that prep_automate_data.sh and draws_check.sh write to ALERT (per assembly: reset_failed, label_failed and
      contact_report_error > 5 % of the socket draws, t00's reset included; t00_refused > 5 % of the grasps, a refused
      t00 dropping the whole grasp; placement_failed > 5 % of the rows that reached placement; random_truncated > 5 %
      of the random-motion rows whose expert ran (stopped at the step budget or before a waypoint, r6-expert-01);
      each redraw reason "<stage>:<reason>" > 5 % of the placement attempts; any start already satisfying AutoMate's rule), followed by
      placement_stats' lines for a failed-attempt leg reason (clearance, "<leg> <reason>", start_error, re-home reset)
      > 5 % of the placement attempts, followed by draw_report's hand / finger information lines ("all"
      show_user_info: finger_contact rows, 80 N / 0.5 mm guard stops and hand_finger_command_clearance failures by
      kind, random-waypoint redraws; r12-expert-01, decision 7 of 2026-11-23). FAIL: more than 10 % of an assembly's placement attempts stop on "lag" with
      peak_plug_socket_N and peak_finger_socket_N <= 1 N (free-motion lag: the speed / threshold numbers go to the
      user before anything is changed); draw_report raising an error.
Writes <out>/smoke/acceptance.json (with "show_user"); exit 0 only if no item fails.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from wm.data import store  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import collect as CO  # noqa: E402

PLANNED_SOFT = ("reset_failed", "contact_report_error")
SHOW_USER_SHARE = 0.05          # a placement-redraw leg reason above this: show the user (row 22)
# (e): problem lines of launcher.log; "hang timeout <n> s" of collect_launch.sh's launch line is not one (r6-ops-01)
LAUNCHER_PROBLEM = r"crash|rate cap|shed by|\bhang\b(?! timeout)"
LAG_FAIL_SHARE = 0.10           # free-motion lag stops above this share of an assembly's placement attempts: FAIL
LAG_FREE_N = 1.0                # a lag stop counts as free motion when both peak forces of its leg are <= 1 N
ACTUAL_GAP_INFO_MM = 1.0        # Chongyu's actual tip-gap guard before insertion (information only, row 24)
# Chongyu's 2 mm FK-sample gap on every outside command (information only, rows 24 / 32, r9-decisions-01). Counted on
# the same peak (simulated plug pose, every outside step including the state after the last one), not on the
# FK-modelled plug his rule uses, and not on peak min_command_clearance_mm (that includes rejected noisy candidates)
ACTUAL_GAP_INFO2_MM = 2.0


def pct(xs, q):
    return None if not xs else round(float(np.percentile(np.asarray(xs, float), q)), 4)


def heights(hs: list) -> dict:
    hs = [float(h) for h in hs if h is not None]
    return {"n": len(hs), "min_mm": min(hs) if hs else None, "p5_mm": pct(hs, 5),
            "n_le_0.01mm": sum(h <= 0.01 for h in hs)}


def start_success_rows(key: str, v: dict) -> list[str]:
    """'<grasp> tNN' of the rows whose start state satisfies AutoMate's rule ((b), r5-expert-06): any row with
    success.success_at_start True, and a refused t00 (no t00 row) with reference_success.success_at_start True."""
    rows = v.get("trajectories") or []
    out = [f"{key} t{int(t.get('idx', 0)):02d}" for t in rows if (t.get("success") or {}).get("success_at_start")]
    if not any(int(t.get("idx", 0)) == 0 for t in rows) and (v.get("reference_success") or {}).get("success_at_start"):
        out.append(f"{key} t00")
    return out


def placement_stats(plan: dict, gj: dict) -> tuple[dict, list, list]:
    """(h): per assembly planned rows and outcomes, failed attempts by leg reason, lag per leg; -> (info, the
    placement-redraw lines for the user, fails). The draw-loss shares and their flags are draw_report's (draws_report)."""
    per, show, fail = {}, [], []
    lag = {}
    for key, v in gj.items():
        if not v:
            continue
        a = per.setdefault(v.get("port") or key.split("_g")[0], {
            "planned_rows": 0, "outcomes": {}, "placement_attempts": 0, "failed_attempts": {}, "free_lag_stops": 0,
            "t00_refused": {"not_completed": [], "rule_fails_at_end": [], "soft": [], "other": []},
            "actual_top_gap_lt_1mm": {"n": 0, "of": 0, "min_mm": None},
            "actual_top_gap_lt_2mm": {"n": 0, "of": 0, "min_mm": None}})
        if v.get("reference_ok") is False:
            rs = v.get("reference_success") or {}
            row = {"grasp": key, "height_mm": rs.get("height_mm"), "keypoint_dist_mm": rs.get("keypoint_dist_mm"),
                   "failure": v.get("reference_failure")}
            if v.get("reference_soft_outcome"):
                a["t00_refused"]["soft"].append(row)
            elif v.get("reference_expert_completed") is False:
                a["t00_refused"]["not_completed"].append(row)
            elif v.get("reference_expert_completed") is True and rs.get("inserted_end") is False:
                a["t00_refused"]["rule_fails_at_end"].append(row)
            else:                                   # e.g. completed and the rule holds, but no FK axis (as t00_cause)
                a["t00_refused"]["other"].append(row)
        for t in v.get("trajectories", []):
            g = (t.get("peak") or {}).get("min_actual_top_gap_mm_outside")
            if isinstance(g, (int, float)) and not isinstance(g, bool):         # information only (row 24)
                for name, lim in (("actual_top_gap_lt_1mm", ACTUAL_GAP_INFO_MM), ("actual_top_gap_lt_2mm", ACTUAL_GAP_INFO2_MM)):
                    gap = a[name]
                    gap["of"] += 1
                    gap["n"] += int(g < lim)
                    gap["min_mm"] = float(g) if gap["min_mm"] is None else min(gap["min_mm"], float(g))
            if t["idx"] == 0:
                continue
            a["planned_rows"] += 1
            o = t.get("outcome")
            a["outcomes"][o] = a["outcomes"].get(o, 0) + 1
            a["placement_attempts"] += int(t.get("placement_attempts") or 0)
            legs_all = list(t.get("placement_legs") or [])
            for f in t.get("placement_failures") or []:
                legs = f.get("legs") or []
                legs_all += legs
                bad = [x for x in legs if not x.get("ok")]
                why = f.get("stage") or "?"
                if bad:
                    why = f"{bad[0].get('leg')} {bad[0].get('reason')}"
                    if bad[0].get("reason") == "lag" and float(bad[0].get("peak_plug_socket_N") or 0) <= LAG_FREE_N \
                            and float(bad[0].get("peak_finger_socket_N") or 0) <= LAG_FREE_N:
                        a["free_lag_stops"] += 1
                a["failed_attempts"][why] = a["failed_attempts"].get(why, 0) + 1
            for x in legs_all:
                if x.get("max_lag_mm") is not None:
                    lag.setdefault(x.get("leg"), []).append(float(x["max_lag_mm"]))
    for aid, a in sorted(per.items()):
        n_att = max(a["placement_attempts"], 1)
        for why, cnt in sorted(a["failed_attempts"].items()):
            if a["placement_attempts"] and cnt / n_att > SHOW_USER_SHARE:
                show.append(f"{aid}: placement redraw '{why}' in {cnt} of {a['placement_attempts']} placement attempts "
                            f"({100 * cnt / n_att:.1f} % > 5 %)")
        if a["placement_attempts"] and a["free_lag_stops"] / n_att > LAG_FAIL_SHARE:
            fail.append(f"{aid}: {a['free_lag_stops']} of {a['placement_attempts']} placement attempts stopped on "
                        f"'lag' with both peak forces <= {LAG_FREE_N} N (> 10 %)")
    info = {"per_assembly": per, "max_lag_mm_per_leg": {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95),
                                                           "max": max(v)} for k, v in lag.items()}}
    return info, show, fail


def draws_report(plan: dict, gj: dict) -> tuple[dict, list]:
    """check_expert_data.draw_report on the grasp.json dicts, with the planned trajectories of plan.json and the specs
    of A.ASSET_DIR, as check_expert_data.py --draws-only (prep_automate_data.sh, draws_check.sh) calls it;
    -> (report, flags_for_user)."""
    sys.path.insert(0, str(ROOT / "scripts" / "automate"))
    import check_expert_data as CED
    port = {g["grasp"]: str(g.get("port")) for g in plan.get("grasps", [])}
    gs = [gj[k] for k in sorted(gj, key=lambda k: (port.get(k, ""), k)) if gj[k]]   # check_expert_data's file order
    planned = {g["grasp"]: len(g.get("trajectories", [])) for g in plan.get("grasps", [])}
    return CED.draw_report(gs, planned, CED.specs_of(sorted({CED.assembly_of_grasp(v) for v in gs})))


def section_h(plan: dict, gj: dict) -> tuple[dict, list, list]:
    """(h) -> (info, show_user, fails): show_user = draw_report's flags, then placement_stats' redraw lines, then
    draw_report's hand / finger information lines ("all" show_user_info; r12-expert-01, decision 7 of 2026-11-23)."""
    info, show, fail = placement_stats(plan, gj)
    try:
        rep, flags = draws_report(plan, gj)
    except Exception as exc:                       # the production definitions must be computable on the smoke
        return info, show, fail + [f"draws flags not available: check_expert_data.draw_report: "
                                   f"{type(exc).__name__}: {exc}"[:300]]
    info["draws"] = {"per_assembly": rep["per_assembly"], "all": rep["all"], "flags_for_user": flags}
    extra = flags + [x for x in show if x not in flags]
    return info, extra + [x for x in rep["all"].get("show_user_info", []) if x not in extra], fail


def prep_draws_flags(run: Path) -> list | None:
    """flags_for_user.txt of the newest <run>/draws_<unix time> folder of prep_automate_data.sh (None if none)."""
    ds = sorted((d for d in run.glob("draws_*") if d.name[6:].isdigit()), key=lambda d: int(d.name[6:]))
    if not ds or not (ds[-1] / "flags_for_user.txt").exists():
        return None
    return [x for x in (ds[-1] / "flags_for_user.txt").read_text().splitlines() if x]


def prep_check(run: Path, fast: Path, rc: int) -> tuple[list, dict]:
    """Item (f) on the records of prep_automate_data.sh (run dir <run>, exit code rc): problems and INFO."""
    why, info = [], {"launcher_exit": rc}
    try:
        P = Path((run / "logs" / "prep_latest").read_text().strip())
        steps = [ln.split() for ln in Path(f"{P}.steps").read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        result = Path(f"{P}.result").read_text().split()
    except OSError as exc:
        return [f"prep_automate_data.sh left no records in {run / 'logs'} (exit {rc}): {exc}"], info
    info["steps"] = [{"step": s[0], "exit": int(s[1]), "unit": s[2], "pid": s[3], "start_ticks": s[4]}
                     for s in steps if len(s) == 5]
    info["result"] = " ".join(result)
    # r9-data-01: the unclipped training-split depth distribution (for the user's D_max question) must be on disk in
    # both branches below; depth_stats.py writes it before check_depth_settings can refuse
    try:
        info["depth_distribution"] = json.loads((fast / "depth_distribution.json").read_text())
    except (OSError, ValueError) as exc:
        info["depth_distribution"] = None
        why.append(f"no depth_distribution.json in {fast} (depth_stats.py writes it before any refusal): {exc}")
    if result == ["ok"]:
        if rc != 0:
            why.append(f"prep_automate_data.sh: result ok but exit {rc}")
        if [s["step"] for s in info["steps"]] != ["draws", "index", "fast", "depth_stats"]:
            why.append(f"prep_automate_data.sh: steps {info['steps']}")
        sys.path.insert(0, str(ROOT / "scripts" / "automate"))
        import depth_stats as DS
        from wm.tdmpc import demo320 as D
        try:
            st = json.loads((fast / "depth_stats.json").read_text())
        except (OSError, ValueError) as exc:
            return why + [f"no depth_stats.json in {fast}: {exc}"], info
        want_split = json.loads((fast / "manifest.json").read_text()).get("split")
        want_sha = D.train_files_sha256(DS.train_entries(str(fast)), str(fast))
        info["depth_stats"] = {k: st.get(k) for k in ("n_entries", "n_pictures", "depth_max_m", "depth_mean_m",
                                                      "depth_std_m", "train_files_sha256", "split")}
        if st.get("split") != want_split or st.get("train_files_sha256") != want_sha:
            why.append(f"depth_stats: split {st.get('split')} vs manifest {want_split}, train_files_sha256 "
                       f"{st.get('train_files_sha256')} vs {want_sha}")
        return why, info
    log = Path(f"{P}.depth_stats.log")
    txt = log.read_text() if log.exists() else ""
    refusal = next((ln for ln in txt.splitlines() if ln.startswith("unusable statistics")), None)
    info["depth_stats"] = (refusal or txt[-500:])[:500]
    # only check_depth_settings' refusal (depth_stats.py's SystemExit text) of the last step is acceptable: 36 grasps
    # can be too few pictures
    if not (result[:2] == ["failed", "depth_stats"] and rc != 0 and refusal is not None):
        why.append(f"prep_automate_data.sh: {' '.join(result)} (exit {rc}; logs {P}.*)")
    return why, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-max-m", type=float, default=0.5,
                    help="D_max for the depth_stats check of (f): a smoke value only (production: the user's choice)")
    a = ap.parse_args()
    out = Path(a.out)
    S = out / "smoke"
    work = S / "check"
    work.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "plan.json").read_text())
    if (out / "inputs" / "assets").is_dir():       # the collection's frozen asset specs, as check_expert_data.main
        A.ASSET_DIR = out / "inputs" / "assets"
    res: dict = {}

    def item(k: str, ok: bool, why: list, info=None):
        res[k] = {"pass": bool(ok), "problems": why[:30], "info": info}
        print(f"({k}) {'PASS' if ok else 'FAIL'} {'; '.join(map(str, why[:5]))}", flush=True)
        if info is not None:
            print(f"    INFO {json.dumps(info, default=str)[:1500]}", flush=True)

    grasps = {g["grasp"]: g for g in plan["grasps"]}
    gj = {}
    for key, g in grasps.items():
        f = out / g["port"] / key / "grasp.json"
        gj[key] = json.loads(f.read_text()) if f.exists() else None

    # (a)
    why = [f"{k}: no grasp.json" for k, v in gj.items() if v is None]
    why += [f"{k}: error {v.get('error')!r}"[:200] for k, v in gj.items() if v and v.get("error")]
    deaths = sorted(glob.glob(str(out / "deaths" / "*.json")))
    why += [f"death record {d}" for d in deaths]
    counted = [f for f in glob.glob(str(out / "build_failures" / "*.json"))
               if any(r.get("counted", True) for r in json.loads(Path(f).read_text()))]
    why += [f"counted build failure {f}" for f in counted]
    summ = json.loads((out / "logs" / "summary.json").read_text()) if (out / "logs" / "summary.json").exists() else {}
    if summ.get("done") != len(grasps):
        why.append(f"summary.json done {summ.get('done')} != {len(grasps)}")
    why += [f"{k}: claim_attempt {v.get('claim_attempt')}" for k, v in gj.items() if v and v.get("claim_attempt") != 1]
    item("a", not why, why)

    # (b)
    why, info = [], {"reference_ok": 0, "grasps": 0, "label_failed": 0, "planned": 0, "outcomes": {},
                     "expert_completed_insert": 0, "insert_perturbed": 0, "success_at_start": 0, "first_success_step": []}
    h_t00 = {"accepted": [], "refused": []}
    h_label = {"accepted": [], "refused": []}
    for key, v in gj.items():
        if not v:
            continue
        info["grasps"] += 1
        info["reference_ok"] += bool(v.get("reference_ok"))
        h_t00["accepted" if v.get("reference_ok") else "refused"].append((v.get("reference_success") or {}).get("height_mm"))
        for t in v["trajectories"]:
            lr = t.get("label_reference")
            if lr:
                h_label["accepted" if lr.get("accepted") else "refused"].append((lr.get("success") or {}).get("height_mm"))
        why += [f"{r}: the start state already satisfies AutoMate's success rule (success_at_start; a wrong start)"
                for r in start_success_rows(key, v)]
        n_plan = len(grasps[key]["trajectories"]) - sum(t.get("outcome") == "reset_failed" for t in v["trajectories"])
        if v.get("reference_ok") and (v.get("timing") or {}).get("resets", 0) < 1 + n_plan:
            why.append(f"{key}: resets {(v.get('timing') or {}).get('resets')} < 1 + {n_plan}")
        for t in v["trajectories"]:
            if t["idx"] == 0:
                continue
            info["planned"] += 1
            o = t.get("outcome")
            info["outcomes"][f"{t['kind']}:{o}"] = info["outcomes"].get(f"{t['kind']}:{o}", 0) + 1
            if t.get("socket_index") != t["idx"]:
                why.append(f"{key} t{t['idx']:02d}: socket_index {t.get('socket_index')}")
            if o in PLANNED_SOFT or o == "error":
                continue
            lr = t.get("label_reference")
            if not lr or "accepted" not in lr:
                why.append(f"{key} t{t['idx']:02d}: no label_reference.accepted")
            elif (o == "label_failed") == bool(lr["accepted"]):
                why.append(f"{key} t{t['idx']:02d}: outcome {o} with accepted {lr['accepted']}")
            info["label_failed"] += o == "label_failed"
            if t["kind"] in ("insert", "perturbed") and o != "label_failed" and "success" in t:
                info["insert_perturbed"] += 1
                info["expert_completed_insert"] += bool(t.get("expert_completed"))
                info["success_at_start"] += bool(t["success"].get("success_at_start"))
                info["first_success_step"].append(t["success"].get("first_success_step"))
    info["end_height_t00"] = {k: heights(v) for k, v in h_t00.items()}
    info["end_height_label"] = {k: heights(v) for k, v in h_label.items()}
    item("b", not why, why, info)

    # (c)
    why, n_files = [], 0
    for key, v in gj.items():
        if not v:
            continue
        if v.get("usd_vs_m1b") != "equal":
            why.append(f"{key}: usd_vs_m1b {v.get('usd_vs_m1b')!r}")
        gdir = out / grasps[key]["port"] / key
        listed = {t["file"] for t in v["trajectories"] if t.get("file")}
        on_disk = {p.name for p in gdir.glob("t*.npz")}
        if listed != on_disk:
            why.append(f"{key}: listed {sorted(listed - on_disk)} missing, unlisted {sorted(on_disk - listed)}")
        spec = A.load_spec(grasps[key]["port"])
        for t in v["trajectories"]:
            if not t.get("file") or t["file"] not in on_disk:
                continue
            n_files += 1
            arrays, meta = store.load(gdir / t["file"], decode_rgb=False)
            T1 = len(arrays["eval_rel"])
            if tuple(meta.get("eval_tip_columns", ())) != tuple(K.TRUTH_COLUMNS) or \
                    arrays["eval_tip"].shape != (T1, len(K.TRUTH_COLUMNS)):
                why.append(f"{key}/{t['file']}: eval_tip {arrays['eval_tip'].shape} columns {meta.get('eval_tip_columns')}")
            for k in ("plan_sha", "config_sha", "switches"):
                if meta.get(k) != plan.get(k):
                    why.append(f"{key}/{t['file']}: meta {k} differs from plan.json")
            s = np.asarray(K.truth_from_rel_spec(arrays["eval_rel"].astype(np.float64), spec)["success"], bool)
            s_any = bool(s[1:].any())
            if meta.get("kind") in ("insert", "perturbed"):
                if (meta.get("outcome") == "inserted") != bool(meta.get("automate_success_any")) or \
                        bool(meta.get("automate_success_any")) != s_any:
                    why.append(f"{key}/{t['file']}: outcome {meta.get('outcome')}, automate_success_any "
                               f"{meta.get('automate_success_any')}, recomputed {s_any}")
            if meta.get("kind") == "contact" and meta.get("outcome") == "inserted":
                why.append(f"{key}/{t['file']}: contact outcome 'inserted'")
            if meta.get("kind") == "reference" and not (meta.get("expert_completed") and bool(s[-1])):
                why.append(f"{key}/{t['file']}: reference saved without completed AND success at the end")
    item("c", not why, why, {"files": n_files})

    # (d)
    why, per = [], {}
    for key, v in gj.items():
        if not v:
            continue
        m = v.get("memory") or {}
        g = m.get("gpu") or {}
        pid = (v.get("process") or {}).get("pid") or v.get("pid")
        r = per.setdefault(str(pid), {"grasps": 0, "memory_peak_gb": 0.0, "gpu1_mib": 0, "gpu0_mib": 0, "seconds": []})
        r["grasps"] += 1
        r["memory_peak_gb"] = max(r["memory_peak_gb"], m.get("memory_peak_gb") or 0.0)
        r["gpu1_mib"] = max(r["gpu1_mib"], g.get("gpu1_mib") or 0)
        r["gpu0_mib"] = max(r["gpu0_mib"], g.get("gpu0_mib") or 0)
        r["seconds"].append(v.get("seconds"))
        if (g.get("gpu0_mib") or 0) > 50:
            why.append(f"{key}: GPU-0 memory {g.get('gpu0_mib')} MiB > 50 MiB")
    secs = [x for r in per.values() for x in r["seconds"] if x]
    samples = (S / "samples.txt").read_text() if (S / "samples.txt").exists() else ""
    gpu1_samples = 0
    for blk in samples.split("=== ")[1:]:
        cpids = {p for u, line in re.findall(r"^unit (\S+) pids (.*)$", blk, re.M) if u.startswith("qingxin-collect-")
                 for p in line.split()}
        for line in blk.split("gpu1:")[-1].strip().splitlines() if "gpu1:" in blk else []:
            p = line.split()
            if len(p) == 3 and p[0] in cpids and p[2] != "?":
                gpu1_samples = max(gpu1_samples, float(p[2]))
    gpu1_grasp = max([r["gpu1_mib"] for r in per.values()] or [0])
    gpu1 = max(gpu1_grasp, gpu1_samples)
    n_incl = sum(r["status"] != "excluded" for r in plan["assemblies"])
    mean_s = float(np.mean(secs)) if secs else None
    info = {"per_worker": per, "gpu1_mib_max": {"grasp_json": gpu1_grasp, "samples": gpu1_samples},
            "worker_gpu_gb_recommended": math.ceil(1.3 * gpu1 * 1048576 / 1e9) if gpu1 else None,
            "seconds_per_grasp_mean": mean_s, "seconds_per_grasp_max": max(secs) if secs else None,
            "production_grasps": 15 * n_incl,
            "production_hours": {w: round(15 * n_incl * mean_s / w / 3600, 1) for w in (4, 6)} if mean_s else None,
            "production_hours_rule": "T_h = 15 x N_incl x t_g / (n x 3600), t_g = mean grasp.json seconds of this "
                                     "smoke (2 workers, after the 2026-11-23 speed changes); re-measure t_g from "
                                     "grasp.json seconds after 1 h of the production run at 4 workers (r5-ops-06)"}
    item("d", not why, why, info)

    # (e)
    why = []
    ages = [int(x) for x in re.findall(r"watch\.alive_age_s (\d+)", samples)]
    if not ages:
        why.append("no samples")
    why += [f"watch.alive {x} s old at a sample" for x in ages if x > 180]
    ours = set()
    for blk in samples.split("=== ")[1:]:
        ours = {p for line in re.findall(r"^unit \S+ pids (.*)$", blk, re.M) for p in line.split()}
        g0 = blk.split("gpu0:")[-1].split("gpu1:")[0]
        for line in g0.strip().splitlines():
            p = line.split()
            if len(p) == 3 and p[0] in ours and (set(p[1].split(",")) != {"G"} or p[2] == "?" or float(p[2]) > 50):
                why.append(f"sample: our pid {p[0]} on GPU 0 ({p[1]} {p[2]} MiB)")
    half = CO.HANG_TIMEOUT_S / 2
    hb_age = {}
    for name, age, stage in re.findall(r"^hb (\S+) age_s (\d+) stage (.*)$", samples, re.M):
        k = stage.split(" ")[0] or "?"
        hb_age[k] = max(hb_age.get(k, 0), int(age))
        if int(age) > half:
            why.append(f"sample: heartbeat {name} {age} s old (stage {stage!r}) > HANG_TIMEOUT_S / 2 = {half:.0f} s")
    gaps = {}
    for key, v in gj.items():
        for k, g in ((v or {}).get("timing") or {}).get("heartbeat_max_gap_s", {}).items():
            gaps[k] = max(gaps.get(k, 0.0), float(g))
            if float(g) > half:
                why.append(f"{key}: heartbeat gap {g} s in stage {k!r} > HANG_TIMEOUT_S / 2 = {half:.0f} s")
    launcher = (out / "logs" / "launcher.log").read_text() if (out / "logs" / "launcher.log").exists() else ""
    why += [ln for ln in launcher.splitlines() if re.search(LAUNCHER_PROBLEM, ln)]
    refusals = [ln for ln in launcher.splitlines() if "refused by the caps" in ln]
    alerts = (out / "logs" / "ALERT").read_text().splitlines() if (out / "logs" / "ALERT").exists() else []
    item("e", not why, why, {"alert_lines": alerts[-20:], "samples": len(ages), "caps_refusals": refusals[-10:],
                             "heartbeat_max_age_s_at_samples": hb_age, "heartbeat_max_gap_s": gaps})

    # (h) first computed here: (f) compares the prep launcher's draws flags with it
    hinfo, show, hfail = section_h(plan, gj)

    # (f)
    fast = work / "fast"
    r = subprocess.run(["bash", str(ROOT / "scripts/automate/prep_automate_data.sh"), str(work / "prep"), str(out),
                        str(fast), repr(float(a.depth_max_m))], capture_output=True, text=True, cwd=str(ROOT))
    (work / "prep_launcher.log").write_text(r.stdout + "\n" + r.stderr)
    why, finfo = prep_check(work / "prep", fast, r.returncode)
    pflags = prep_draws_flags(work / "prep")
    dflags = (hinfo.get("draws") or {}).get("flags_for_user")
    finfo["prep_draws_flags_for_user"] = pflags
    if pflags is None:
        why.append(f"the launcher's draws step wrote no flags_for_user.txt under {work / 'prep'}")
    elif dflags is not None and pflags != dflags:
        why.append(f"the launcher's draws flags {pflags[:5]} differ from draw_report's {dflags[:5]}")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/automate/check_expert_data.py"), "--data", str(out),
                        "--work", str(work / "ced"), "--no-media"], capture_output=True, text=True, env=env, cwd=str(ROOT))
    (work / "check_expert_data.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        why.append(f"check_expert_data exit {r.returncode} ({work / 'check_expert_data.log'})")
    ced = work / "ced" / "check.json"
    if ced.exists():
        finfo["check_expert_data_exit_reasons"] = (json.loads(ced.read_text()).get("exit_reasons") or [])[:20]
    item("f", not why, why, finfo)

    # (h)
    item("h", not hfail, hfail, hinfo)
    res["show_user"] = show
    for line in show:
        print(f"    SHOW THE USER BEFORE LAUNCH: {line}", flush=True)

    # (g)
    why = []
    w2 = (S / "w2.end.phase2").read_text() if (S / "w2.end.phase2").exists() else ""
    if "exit 7" not in w2:
        why.append(f"w2.end.phase2: {w2.strip()!r}")
    stop = (S / "STOP.g").read_text() if (S / "STOP.g").exists() else ""
    m = re.search(r"claim (\S+) released \(not a death\)", stop)
    if "forbidden context on GPU 0" not in stop or m is None:
        why.append(f"STOP.g has no hook line with a released claim: {stop.strip()[:300]!r}")
    else:
        g = Path(m.group(1)).parent / "grasp.json"
        if not g.exists() or json.loads(g.read_text()).get("error"):
            why.append(f"{g}: missing or error after the resume")
    if deaths:
        why.append(f"deaths recorded: {deaths}")
    item("g", not why, why)

    (S / "acceptance.json").write_text(json.dumps(res, indent=1, default=str))
    bad = [k for k, v in res.items() if k != "show_user" and not v["pass"]]
    print(f"SMOKE ACCEPTANCE {'PASS' if not bad else 'FAIL ' + ','.join(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
