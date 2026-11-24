"""Milestone-1b table for all assemblies: mesh analysis (analysis.json) + simulation batch (batch/<id>.json).

Run on rs2 only inside our caps (r10-ops-02): a caps.sh qx_run_retry scope (kind prep, 4G, 100 %, GPU 0; a caps
refusal is tried again every 60 s for up to QX_RETRY_MAX_S), from the CPU batch's frozen code, with the batch's
switches (wm.sim_automate.config reads WM_* from the shell); the table is redirected inside the scope, so the retry
lines still reach the terminal. With B = /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu and
M = /mnt/storage3/qingxin/automate set in the launching shell:
    ( source $B/code/scripts/automate/caps.sh; source $B/code/scripts/automate/collect_lib.sh; cl_isaac_env;
      cl_clean_env $B/switches.env >/dev/null; cd $B/code;
      qx_run_retry prep m1breport 4G 100% 0 bash -c 'exec "$0" -u scripts/automate/report_m1b.py --root "$1" \
        --batch batch_r3_cpu > "$2"' "$CL_PY" $M/m1b $M/m1b/table_r3_cpu.md )
Never start it with a bare python: that runs outside the caps (no scope, no MemoryMax) and with whatever WM_* the
shell holds.
Simulation columns: (i) / (ii) / (iii) pass under batch_assemblies.py's rules (environment checks, |depth_left| <
0.5 mm), with AutoMate's success after the last step (S_end) and after any step (S_any) shown next to (ii) / (iii).
"stale" lists the provenance fields of a result that differ from the current files and switches (collect.stale_fields,
the inclusion gate's own comparison, physics_code_sha256 included): a stale result must be re-run, it is never evidence
for the current code. The summary line also counts (r4-expert-02 d): results whose plug ended at h <= 0.01 mm in (i)
/ (ii) / (iii) (from height_mm, so results written before (i) recorded h_end_le_0.01mm count too; AutoMate's h > 0 is
strict), (ii) results that fail only by h > 0 (|h_end| < 0.05 mm, h <= 0, everything else seated), results whose
compared USD layers were not all read (usd_compared_ok False), and lists the assemblies that have only an error result.
It also lists every assembly with a part ending at h <= 0.01 mm, with those heights and whether its (ii) / (iii) pass
(the parts of the inclusion gate; r6-expert-02). In (i) / (iii) the plug is placed by move_to (2 mm waypoints at
20 mm/s, then settling), which can end up to about 15 um below h = 0 (batch2: 01053 -15.2, 00860 -13.8, 00471 -8.9,
00016 -3.2 um in (iii), 00755 -2.7 um in (i)). The expert's 3 mm/s push lags its command (the 0.3 mm
insert throttle) and settles onto h = 0 from above (gravity off): measured h_end 26-104 um, never below, over the 186
expert_smoke ends (reference, insert and perturbed runs) on 00015 / 00083 / 01102, including 01102 whose plug rests on
the socket and 00015 with 16 mm free below; smoke_env s4 +94 um (r7-expert-01). Refusals because h <= 0 are therefore
expected to be rare; they are still counted (a refused t00 drops the grasp, a refused label one socket draw;
refused_completed_with_h_le_0). Watch these assemblies' t00_refused / label_failed in the smoke's draws report (smoke
00755 / 01053) and in the first production draws_check.
Depth check: agree if |d_dis - L| <= 4 mm. Otherwise the reason is classified from the lift sweep g(d) (fit.py):
    lead-in      |d_dis - L_top| <= 4 mm: above the tight fit the plug is still inside a looser part of the socket
                 (counterbore / chamfer / wide mouth, gap g > 2c) for L_top - L mm; d_dis counts it
    early-free   d_dis < L - 4 mm: AutoMate's disassembly distance is shorter than the tight fit
    other        anything else (gap profile printed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402

A_DIR = A.ASSET_DIR


def current(r: dict) -> dict:
    """The analysis row with the CURRENT geometry and ranges (assets.SPEC_FILE, grasp.RANGES_FILE) in place of the
    values stored when analysis.json was written; marks rows whose files are missing."""
    aid = r["assembly_id"]
    try:
        spec = A.load_spec(aid).as_dict()
    except FileNotFoundError as exc:
        return {"assembly_id": aid, "error": str(exc)}
    try:
        rg = G.load_ranges(aid)
    except FileNotFoundError as exc:
        rg = {"box": None, "reason": str(exc), "spin_limit_deg": G.spin_limit_deg(spec["symmetry_n"])}
    return {**r, **spec, "ranges": rg}


def changed_n(rows) -> list:
    """Assemblies whose symmetry n differs between geometry_v4.json and the current SPEC_FILE."""
    out = []
    for r in rows:
        f4 = A_DIR / r["assembly_id"] / "geometry_v4.json"
        if "error" not in r and f4.exists():
            n4 = json.loads(f4.read_text())["symmetry_n"]
            if n4 != r["symmetry_n"]:
                out.append(f"{r['assembly_id']} ({n4} -> {r['symmetry_n']})")
    return out


def _sha(f) -> str | None:
    f = Path(f)
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def stale_fields(res: dict, aid: str, device: str | None = None) -> list:
    """collect.stale_fields (the gate's comparison: ranges_sha256, grasp_py_sha256, physics_device, switches, the
    asset hashes spec_sha256 / plug_obj_sha256 / socket_obj_sha256 / asset_sha256, shared_asset_sha256 and
    physics_code_sha256; a missing field counts as different) with device (default config.PHYSICS_DEVICE)."""
    import types
    from wm.sim_automate import collect as CO
    return CO.stale_fields(res, aid, types.SimpleNamespace(device=device or C.PHYSICS_DEVICE))


def fails_only_by_height(x: dict) -> bool:
    """(ii) failed although placed, completed, seated (|depth_left| < 0.5 mm) and keypoints close: only h > 0 broke."""
    try:
        h, k, d = float(x["height_mm"]), float(x["keypoint_dist_mm"]), float(x["depth_left_mm"])
    except (KeyError, TypeError, ValueError):
        return False
    return (x.get("pass") is not True and x.get("placed") is True and x.get("completed") is True and abs(d) < 0.5
            and h <= 0.0 and abs(h) < 0.05 and k < 15.0)


def low_end_heights(b: dict, limit_mm: float = 0.01) -> dict:
    """{part: height_mm} of the parts (i), (ii), (iii) of one batch result whose plug ended at h <= limit_mm (from
    height_mm, which every part records through checks.truth; skipped parts and missing values are left out)."""
    out = {}
    for part in ("i", "ii", "iii"):
        x = b.get(part)
        h = x.get("height_mm") if isinstance(x, dict) else None
        if isinstance(h, (int, float)) and math.isfinite(h) and h <= limit_mm:
            out[part] = float(h)
    return out


def part_state(b: dict, part: str) -> str:
    """pass / FAIL / skipped / missing of one part of a batch result."""
    x = b.get(part)
    if not isinstance(x, dict):
        return "missing"
    return "skipped" if "skipped" in x else "pass" if x.get("pass") else "FAIL"


def gap_at(sweep, d):
    best = None
    for x, g in sweep:
        if x <= d + 1e-9:
            best = g
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/storage3/qingxin/automate/m1b")
    ap.add_argument("--batch", default="batch_r3_cpu")
    ap.add_argument("--legacy-merge", action="store_true",
                    help="also merge batch_fix/ and batch_iii/ (the 2026-11-21 batches only)")
    args = ap.parse_args()
    rows = [current(r) for r in json.loads((Path(args.root) / "analysis.json").read_text())]
    ch = changed_n(rows)
    print(f"Geometry {A.SPEC_FILE}, ranges {G.RANGES_FILE}. Symmetry n changed since geometry_v4 for {len(ch)} "
          f"assemblies: {', '.join(ch)}\n")
    hdr = ["id", "c mm", "L mm", "d_dis mm", "depth check", "n", "category", "size mm", "dtheta deg", "dp mm",
           "dpsi deg", "(i) H*", "(ii) insert", "(iii) worst grasp in sim", "stale"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    stats = {"i": 0, "ii": 0, "iii": 0, "n": 0, "stale": 0, "i_h_le_0.01": 0, "ii_h_le_0.01": 0, "iii_h_le_0.01": 0,
             "ii_only_h": 0, "usd_not_ok": 0}
    error_only, low_h = [], []
    for r in rows:
        aid = r["assembly_id"]
        if "error" in r:
            print(f"| {aid} | analysis error: {r['error']} |")
            continue
        L, dd, Lt = r["pin_length_m"] * 1e3, r["disassembly_dist_m"] * 1e3, r["depth_top_m"] * 1e3
        f = r["fit"]
        if abs(dd - L) <= 4:
            chk = "agree"
        elif abs(dd - Lt) <= 4:
            chk = f"lead-in: {Lt - L:.1f} mm looser part above the tight fit (g({L + 1:.0f} mm) = {gap_at(f['sweep_coarse'], (L + 1) / 1e3) * 1e3 if gap_at(f['sweep_coarse'], (L + 1) / 1e3) is not None else float('inf'):.1f} mm)"
        elif dd < L - 4:
            chk = f"early-free: d_dis < L (L_top {Lt:.1f})"
        else:
            chk = f"other (L_top {Lt:.1f})"
        n = r["symmetry_n"]
        size = f"{f.get('equiv_diameter_m', float('nan')) * 1e3:.1f} (long {f.get('long_side_m', float('nan')) * 1e3:.1f})"
        cat = r["category"] + (" (12-gon, may be a tessellated cylinder)" if n == 12 and (f.get("radial_spread") or 1) < 0.04 else "")
        rg = r["ranges"]
        if rg.get("box"):
            b = rg["box"]
            th, dp, ps = (f"[{b['dtheta_deg'][0]:.0f}, {b['dtheta_deg'][1]:.0f}]",
                          f"[{b['dp_m'][0] * 1e3:.1f}, {b['dp_m'][1] * 1e3:.1f}]",
                          f"[{b['dpsi_deg'][0]:.1f}, {b['dpsi_deg'][1]:.1f}]")
        else:
            th = dp = ps = "none: " + rg.get("reason", "?")[:60]
        bf = Path(args.root) / args.batch / f"{aid}.json"
        i = ii = iii = "not run"
        stale = ""
        if bf.exists():
            b = json.loads(bf.read_text())
            fix = Path(args.root) / "batch_fix" / f"{aid}.json"          # reruns in a fresh process
            if args.legacy_merge and fix.exists():
                b = json.loads(fix.read_text())
            b3 = Path(args.root) / "batch_iii" / f"{aid}.json"           # (iii) with the final ranges
            if args.legacy_merge and b3.exists():
                b.pop("iii", None)
                x3 = json.loads(b3.read_text())
                if "iii" in x3:
                    b["iii"] = x3["iii"]
                elif "error" in x3:
                    b["iii"] = {"skipped": "ERROR " + x3["error"][:60]}
            stats["n"] += 1
            st = stale_fields(b, aid)
            stale = ", ".join(st)
            stats["stale"] += bool(st)
            if "i" in b:
                x = b["i"]
                i = ("pass" if x["pass"] else "FAIL") + f" (depth {x['depth_left_mm']:.2f}, pen {x['max_penetration_mm']:.2f}" \
                    f" [obj overlap {r.get('obj_overlap_mm', float('nan')):.2f}], hand/finger {x['hand_finger_socket_N']:.1f} N" \
                    f"{'' if x['move_ok'] else ', IK failed'})"
                stats["i"] += x["pass"]
            if "ii" in b:
                x = b["ii"]
                ii = ("pass" if x["pass"] else "FAIL") + f" (depth {x['depth_left_mm']:.2f}, lat {x['lateral_mm']:.2f}, " \
                     f"tilt {x['tilt_deg']:.2f}, S_end {x.get('success_end', '?')}, S_any {x.get('success_any', '?')}, " \
                     f"{x['push_end']}{'' if x['placed'] else ', start not reached'})"
                stats["ii"] += x["pass"]
                stats["ii_only_h"] += fails_only_by_height(x)
            if "iii" in b:
                x = b["iii"]
                iii = x.get("skipped") or (("pass" if x.get("pass") else "FAIL" if "pass" in x else "no rule") +
                                           f" ({x.get('hand_finger_max_N', x['hand_finger_socket_N']):.1f} N at "
                                           f"{[round(v, 4) for v in x['grasp']]}, mesh margin {x['mesh_margin_mm']:.1f} mm, "
                                           f"depth {x['depth_left_mm']:.2f}, S_end {x.get('success_end', '?')})")
                stats["iii"] += bool(x.get("pass"))
            low = low_end_heights(b)
            for part in low:
                stats[f"{part}_h_le_0.01"] += 1
            if low:
                gate = ", ".join(f"{p} {part_state(b, p)}" for p in ("ii", "iii"))
                low_h.append(f"{aid} " + ", ".join(f"({p}) {h * 1e3:.1f} um" for p, h in low.items()) + f" [{gate}]")
            stats["usd_not_ok"] += b.get("usd_compared_ok") is not True
            if "error" in b and not any(k in b for k in ("i", "ii", "iii")):
                error_only.append(aid)
            if "error" in b:
                i = i if "i" in b else "ERROR " + b["error"][:80]
                ii = ii if "ii" in b else "ERROR"
        print(f"| {aid} | {r['clearance_m'] * 1e3:.3f} | {L:.1f} | {dd:.1f} | {chk} | {n} | {cat} | {size} | {th} | {dp} | "
              f"{ps} | {i} | {ii} | {iii} | {stale} |")
    print(f"\nsimulated {stats['n']}, (i) pass {stats['i']}, (ii) pass {stats['ii']}, (iii) pass {stats['iii']}, "
          f"stale results {stats['stale']}")
    print(f"end height h <= 0.01 mm: (i) {stats['i_h_le_0.01']}, (ii) {stats['ii_h_le_0.01']}, (iii) "
          f"{stats['iii_h_le_0.01']}; (ii) failing only by h > 0: {stats['ii_only_h']}; compared USD layers not all "
          f"read: {stats['usd_not_ok']}")
    print(f"assemblies with a part ending at h <= 0.01 mm ({len(low_h)}; watch their t00_refused / label_failed in "
          f"draws_check): {'; '.join(low_h) or '-'}")
    print(f"error-only results ({len(error_only)}): {', '.join(error_only) or '-'}")


if __name__ == "__main__":
    main()
