"""Compare PhysX on the GPU and on the CPU (no Isaac needed): two milestone-1b batches and / or two collections of the
same assemblies (decision 4 of 2026-11-22: CPU PhysX, with a CPU-vs-GPU check on hard assemblies). The GPU batch
(gpu_batch_r3.sh) covers the 9 smoke assemblies by default (GPU PhysX is about 30 times slower per control step);
assemblies present in one batch only are listed as such, and the per-assembly results of the common ones are the check.

Run on rs2 only inside our caps (r10-ops-02): a caps.sh qx_run_retry scope (kind prep, 4G, 100 %, GPU 0; a caps
refusal is tried again every 60 s for up to QX_RETRY_MAX_S), from the CPU batch's frozen code, with the batch's
switches (wm.sim_automate.config reads WM_* from the shell). With B = /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu
and M = /mnt/storage3/qingxin/automate set in the launching shell:
    ( source $B/code/scripts/automate/caps.sh; source $B/code/scripts/automate/collect_lib.sh; cl_isaac_env;
      cl_clean_env $B/switches.env >/dev/null; cd $B/code;
      qx_run_retry prep cmpdev 4G 100% 0 $CL_PY -u scripts/automate/compare_physics_device.py \
        --batch-a $M/m1b/batch_r3_gpu --batch-b $B --out $M/m1b/compare_r3_gpu_cpu.json )
(add --a <GPU collection out> --b <CPU collection out> to compare two collections too). Never start it with a bare
python: that runs outside the caps (no scope, no MemoryMax) and with whatever WM_* the shell holds.

Batches (--batch-a / --batch-b: folders of batch_assemblies.py results <id>.json): per assembly in both,
  error            the result's "error" (a process crash or a failed scene build)
  pass_i, pass_ii, pass_iii   the batch's own pass of (i), (ii), (iii) ((iii) "skipped" when it has no box point
                   with margin < 3 mm; None when the check is missing)
  ii: depth_left_mm, height_mm (h_end), success_end, success_any, push_end, placed, completed
  iii: depth_left_mm, height_mm, success_end, hand_finger_max_N
  provenance       the fields that must be equal for a fair comparison: ranges_sha256, grasp_py_sha256,
                   physics_code_sha256, the geometry and mesh hashes spec_sha256, plug_obj_sha256, socket_obj_sha256
                   (collect.asset_identity, r6-ops-03), shared_asset_sha256 (plug_grasps.json and the finger
                   collision meshes, collect.shared_asset_identity, r9-integration-01) and switches apart from
                   physics_device_default; listed per
                   assembly when they differ or are missing (provenance_differs), and physics_device of both
Summary: assemblies in both / only in one; per field the number of assemblies whose value differs (|depth_left| or
|h_end| difference > H_TOL_MM counts as different); the assemblies whose (ii) or (iii) pass differs (the inclusion
gate could differ between the devices) and the largest differences.

Collections (--a / --b, e.g. the GPU and the CPU collector smoke on 00015, 00117, 00133 and 00444): first whether
the two plan.json split documents are equal (plan_split_equal; if not, the grasps are compared by name only, and the
summary says so), the grasps only in one of them; then for every grasp present in both, per trajectory file present
in both:
  completed          the expert completed (meta expert_completed; older data: no failure recorded)
  failure            the expert's failure text
  success_any, success_end, first_success_step, start_success
                     AutoMate's rule (checks.success_steps: s_t = [0 < h_t < d_dis AND k_t < 15 mm] after control
                     steps t = 1..T only; the start state is information)
  h_end_mm, depth_left_end_mm   plug height above the socket and depth left at the tip centre at the end
  seat_ok            completed AND |depth_left_end_mm| < SEAT_MM AND success_end (the physical "reaches seated" meaning
                     of the environment checks, lead resolution r3 of decisions-n1; a comparison number, not the task
                     criterion)
  plus the outcome, steps per phase, the largest plug-socket force and the hand path difference
  max_t |p_a(t) - p_b(t)| over the common steps [mm] (recorded FK hand).
Summary: per field the number of trajectories whose value differs between the two devices (|h_end| difference
> H_TOL_MM counts as different), and the grasp-level H* difference. Same plan and seeds, so differences come from the
physics device only. Asset specs come from <a>/inputs/assets when it exists (else $WM_AUTOMATE_ASSETS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.data import store  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks  # noqa: E402

SEAT_MM, H_TOL_MM = 0.5, 0.1
FIELDS = ("outcome", "completed", "failure", "success_any", "success_end", "first_success_step", "seat_ok")
BATCH_FIELDS = ("error", "pass_i", "pass_ii", "pass_iii", "ii_success_end", "ii_success_any", "ii_push_end",
                "iii_success_end")
BATCH_MM = ("ii_depth_left_mm", "ii_height_mm", "iii_depth_left_mm", "iii_height_mm")
PROVENANCE_EQUAL = ("ranges_sha256", "grasp_py_sha256", "physics_code_sha256", "spec_sha256", "plug_obj_sha256",
                    "socket_obj_sha256", "shared_asset_sha256")     # shared_asset_sha256: r9-integration-01
DEVICE_SWITCH = "physics_device_default"        # the one switch that differs between the two batches by design


def _pass(x):
    if not isinstance(x, dict):
        return None
    if "skipped" in x:
        return "skipped"
    return x.get("pass")


def _num(x, k):
    try:
        return float(x[k])
    except (TypeError, KeyError, ValueError):
        return None


def batch_numbers(res: dict) -> dict:
    """The per-assembly numbers of one batch result (module docstring, batches)."""
    ii, iii = res.get("ii"), res.get("iii")
    ii = ii if isinstance(ii, dict) else {}
    iii = iii if isinstance(iii, dict) and "skipped" not in iii else {}
    return {"error": res.get("error"), "pass_i": _pass(res.get("i")), "pass_ii": _pass(res.get("ii")),
            "pass_iii": _pass(res.get("iii")), "ii_success_end": ii.get("success_end"),
            "ii_success_any": ii.get("success_any"), "ii_push_end": ii.get("push_end"),
            "ii_placed": ii.get("placed"), "ii_completed": ii.get("completed"),
            "ii_depth_left_mm": _num(ii, "depth_left_mm"), "ii_height_mm": _num(ii, "height_mm"),
            "iii_success_end": iii.get("success_end"), "iii_depth_left_mm": _num(iii, "depth_left_mm"),
            "iii_height_mm": _num(iii, "height_mm"), "iii_hand_finger_max_N": _num(iii, "hand_finger_max_N"),
            "physics_device": res.get("physics_device")}


def provenance_differs(ra: dict, rb: dict) -> list[str]:
    bad = [k for k in PROVENANCE_EQUAL if ra.get(k) != rb.get(k) or ra.get(k) is None]
    sa = {k: v for k, v in (ra.get("switches") or {}).items() if k != DEVICE_SWITCH}
    sb = {k: v for k, v in (rb.get("switches") or {}).items() if k != DEVICE_SWITCH}
    if sa != sb or not sa:
        bad.append("switches (apart from " + DEVICE_SWITCH + ")")
    return bad


def _read(f: Path):
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError) as exc:
        return {"error": f"unreadable result: {type(exc).__name__}: {exc}"}
    return r if isinstance(r, dict) else {"error": "not a json object"}


def compare_batches(A_dir: Path, B_dir: Path) -> tuple[dict, list[dict]]:
    """Per assembly of two milestone-1b batch folders (module docstring, batches)."""
    ids_a = {f.stem for f in A_dir.glob("*.json") if f.stem.isdigit()}        # <assembly id>.json only
    ids_b = {f.stem for f in B_dir.glob("*.json") if f.stem.isdigit()}
    rows = []
    summary = {"assemblies_in_both": 0, "only_in_a": sorted(ids_a - ids_b), "only_in_b": sorted(ids_b - ids_a),
               "different": {f: 0 for f in (*BATCH_FIELDS, *BATCH_MM)}, "pass_ii_or_iii_differs": [],
               "provenance_differs": {}, "max_abs_diff_mm": {f: 0.0 for f in BATCH_MM}, "physics_device": None}
    devices = set()
    for aid in sorted(ids_a & ids_b):
        ra, rb = _read(A_dir / f"{aid}.json"), _read(B_dir / f"{aid}.json")
        na, nb = batch_numbers(ra), batch_numbers(rb)
        devices.add((na["physics_device"], nb["physics_device"]))
        summary["assemblies_in_both"] += 1
        for f in BATCH_FIELDS:
            summary["different"][f] += int(na[f] != nb[f])
        for f in BATCH_MM:
            if na[f] is None or nb[f] is None:
                summary["different"][f] += int((na[f] is None) != (nb[f] is None))
                continue
            d = abs(na[f] - nb[f])
            summary["different"][f] += int(d > H_TOL_MM)
            summary["max_abs_diff_mm"][f] = max(summary["max_abs_diff_mm"][f], d)
        if na["pass_ii"] != nb["pass_ii"] or na["pass_iii"] != nb["pass_iii"]:
            summary["pass_ii_or_iii_differs"].append(aid)
        pd = provenance_differs(ra, rb)
        if pd:
            summary["provenance_differs"][aid] = pd
        rows.append({"assembly_id": aid, **{k: [na[k], nb[k]] for k in na}, "provenance_differs": pd})
    summary["physics_device"] = sorted([list(d) for d in devices], key=str)
    return summary, rows


def trajectory_numbers(arrays: dict, meta: dict) -> dict:
    """The per-trajectory numbers of the module docstring for one file."""
    spec = A.load_spec(meta["assembly_id"])
    rel = np.asarray(arrays["eval_rel"], np.float64)
    s = checks.success_steps(rel, spec)
    t = checks.truth_from_rel_spec(rel[-1], spec)
    c = meta.get("expert_completed")
    completed = bool(c) if c is not None else meta.get("failure") is None
    dl = float(t["depth_left_mm"])
    return {"outcome": meta.get("outcome"), "completed": completed, "failure": meta.get("failure"),
            "success_any": s["success_any"], "success_end": s["success_end"],
            "first_success_step": s["first_success_step"], "start_success": s["start_success"],
            "h_end_mm": float(t["height_mm"]), "depth_left_end_mm": dl,
            "seat_ok": bool(completed and abs(dl) < SEAT_MM and s["success_end"])}


def plan_split(d: Path):
    f = d / "plan.json"
    if not f.exists():
        return None
    return (json.loads(f.read_text()).get("split") or {}).get("document")


def compare(A_dir: Path, B_dir: Path) -> tuple[dict, list[dict]]:
    rows = []
    sa, sb = plan_split(A_dir), plan_split(B_dir)
    ga_names = {p.relative_to(A_dir) for p in A_dir.glob("*/*/grasp.json")}
    gb_names = {p.relative_to(B_dir) for p in B_dir.glob("*/*/grasp.json")}
    summary = {"plan_split_equal": sa == sb and sa is not None,
               "compared_by": "grasp name, same split" if sa == sb and sa is not None else
                              "grasp name only: the two plans' split documents differ (or one is missing)",
               "grasps_only_in_a": sorted(str(p.parent.name) for p in ga_names - gb_names),
               "grasps_only_in_b": sorted(str(p.parent.name) for p in gb_names - ga_names),
               "trajectories": 0, "different": {f: 0 for f in (*FIELDS, "h_end_mm")}, "grasps": []}
    for ga in sorted(A_dir.glob("*/*/grasp.json")):
        gb = B_dir / ga.relative_to(A_dir)
        if not gb.exists():
            continue
        ja, jb = json.loads(ga.read_text()), json.loads(gb.read_text())
        summary["grasps"].append({"grasp": ja["grasp"], "seconds": [ja.get("seconds"), jb.get("seconds")],
                                  "reference_ok": [ja.get("reference_ok"), jb.get("reference_ok")],
                                  "memory_peak_gb": [(ja.get("memory") or {}).get("memory_peak_gb"),
                                                     (jb.get("memory") or {}).get("memory_peak_gb")],
                                  "H_star_diff_mm": float(np.linalg.norm(np.asarray(ja["H_star"])[:3, 3] - np.asarray(jb["H_star"])[:3, 3]) * 1e3)
                                  if ja.get("H_star") and jb.get("H_star") else None})
        for pa in sorted(ga.parent.glob("t*.npz")):
            pb = gb.parent / pa.name
            if not pb.exists():
                continue
            aa, ma = store.load(pa, decode_rgb=False)
            ab, mb = store.load(pb, decode_rgb=False)
            na, nb = trajectory_numbers(aa, ma), trajectory_numbers(ab, mb)
            n = min(len(aa["hand"]), len(ab["hand"]))
            r = {"grasp": ma["grasp"], "file": pa.stem, "steps": [ma["steps"], mb["steps"]],
                 **{k: [na[k], nb[k]] for k in na},
                 "peak_plug_socket_N": [(ma.get("peak") or {}).get("plug_socket_N"), (mb.get("peak") or {}).get("plug_socket_N")],
                 "stages": [[(s["phase"], s["steps"]) for s in ma.get("stages", [])], [(s["phase"], s["steps"]) for s in mb.get("stages", [])]],
                 "hand_path_max_diff_mm": float(np.abs(aa["hand"][:n, :3, 3] - ab["hand"][:n, :3, 3]).max() * 1e3)}
            summary["trajectories"] += 1
            for f in FIELDS:
                summary["different"][f] += int(na[f] != nb[f])
            summary["different"]["h_end_mm"] += int(abs(na["h_end_mm"] - nb["h_end_mm"]) > H_TOL_MM)
            rows.append(r)
    return summary, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="collection output folder (GPU)")
    ap.add_argument("--b", help="collection output folder (CPU)")
    ap.add_argument("--batch-a", help="milestone-1b batch folder (GPU), e.g. m1b/batch_r3_gpu")
    ap.add_argument("--batch-b", help="milestone-1b batch folder (CPU), e.g. m1b/batch_r3_cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if (args.a is None) != (args.b is None) or (args.batch_a is None) != (args.batch_b is None):
        raise SystemExit("give --a with --b and --batch-a with --batch-b")
    if args.a is None and args.batch_a is None:
        raise SystemExit("nothing to compare: give --batch-a / --batch-b and / or --a / --b")
    out = {}
    if args.batch_a:
        bs, brows = compare_batches(Path(args.batch_a), Path(args.batch_b))
        for r in brows:
            print(f"{r['assembly_id']} pass i/ii/iii {r['pass_i']} {r['pass_ii']} {r['pass_iii']} "
                  f"ii depth_left_mm {r['ii_depth_left_mm']} h_end_mm {r['ii_height_mm']} success_end {r['ii_success_end']} "
                  f"success_any {r['ii_success_any']} error {r['error']}")
        print(json.dumps({"batches": bs}, indent=1))
        out["batches"] = {"summary": bs, "rows": brows}
    if args.a:
        A_dir, B_dir = Path(args.a), Path(args.b)
        if (A_dir / "inputs" / "assets").is_dir():
            A.ASSET_DIR = A_dir / "inputs" / "assets"
        summary, rows = compare(A_dir, B_dir)
        for r in rows:
            print(f"{r['grasp']} {r['file']:14s} completed {r['completed']} success_any {r['success_any']} seat_ok {r['seat_ok']} "
                  f"h_end_mm {[round(v, 3) for v in r['h_end_mm']]} first_success {r['first_success_step']} "
                  f"failure {r['failure']} path_diff_mm {r['hand_path_max_diff_mm']:.2f}")
        print(json.dumps({"collections": summary}, indent=1))
        out["collections"] = {"summary": summary, "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
