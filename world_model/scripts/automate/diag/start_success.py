"""Which assemblies can already satisfy AutoMate's success rule at the start, before any motion (no Isaac; numpy only).

AutoMate evaluates success after each control step only, so a start that satisfies the rule is not counted as a
success by itself (checks.success_steps excludes eval_rel[0]); but the first control step from such a start can count.
Per assembly with a grasp box (grasp.RANGES_FILE), N random draws (seeded, reproducible) of
    grasp   (dtheta, dp, dpsi) uniform in the verified box          (grasp.T_hp with the switch's spin axis)
    start   wm.env.sampling.sample_start (tip gap g0 6-18 mm, lateral +-10 mm, rotations +-6 deg), g0 measured as
            config.START_GAP_FROM says (default "socket_top", the user's decision 2026-11-23: from the higher of
            {socket top, entrance}; "entrance" only with WM_AUTOMATE_DIAGNOSTIC=1)
    socket  yaw uniform in [0, 10] deg (AutoMate's noise; it turns the robot base x axis in the socket frame, which
            sets the lateral axes of the start frame, frames.frame)
are tested with checks.start_box_success (success_now of the plug pose at the start: 0 < h < d_dis and mean keypoint
distance < 15 mm). Reported: p_start_success = successes / N, and the aligned bound
    aligned_g0_min_success = (b_min < min(d_dis, 15 mm)),   b_min = frames.start_back(L, 6 mm, g_top)
(b_min = L + g_top + 6 mm under "socket_top", L + 6 mm under "entrance"; an aligned start at the smallest tip gap has
h = keypoint distance = b_min, the plug root being the socket root when seated). The collector's FK H* differs from
the model H* used here by 0.04-0.12 mm (smoke runs).

    python -u start_success.py --out /mnt/storage3/qingxin/automate/m1b/diag/start_success.json [--n 4000] [--ids a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.env.sampling import sample_start  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import frames as FR  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402


def assembly(aid: str, n: int, seed: int) -> dict:
    rg = G.load_ranges(aid)
    if rg.get("box") is None:
        return {"assembly_id": aid, "box": None}
    spec = A.load_spec(aid)
    T0 = G.T_hp0(aid)
    axis_xy = G.spin_axis_xy(spec)
    box = rg["box"]
    rng = np.random.default_rng([seed, int(aid)])
    L, d = float(spec.pin_length_m), float(spec.disassembly_dist_m)
    g_top = K.g_top_of(spec)
    b_min = FR.start_back(L, 0.006, g_top)                                  # aligned start at g0 = 6 mm (switch)
    hits, h_min, kd_min = 0, np.inf, np.inf
    for _ in range(n):
        v = [rng.uniform(*box[k]) if box[k][1] > box[k][0] else box[k][0] for k in ("dtheta_deg", "dp_m", "dpsi_deg")]
        yaw = np.radians(C.SOCKET_YAW_DEG + C.SOCKET_YAW_RANGE_DEG * rng.uniform())
        base_x = np.array([np.cos(yaw), -np.sin(yaw), 0.0])                 # R_z(yaw)^T e_x
        r = K.start_box_success(spec, G.T_hp(T0, *v, axis_xy=axis_xy), sample_start(rng), base_x)
        hits += r["success"]
        h_min, kd_min = min(h_min, r["height_mm"]), min(kd_min, r["keypoint_dist_mm"])
    return {"assembly_id": aid, "L_mm": L * 1e3, "g_top_mm": g_top * 1e3, "d_dis_mm": d * 1e3, "n": n,
            "start_successes": int(hits), "p_start_success": hits / n, "aligned_start_back_mm": b_min * 1e3,
            "aligned_g0_min_success": bool(b_min < min(d, K.CLOSE_ERROR_THRESH_M)),
            "min_height_mm": float(h_min), "min_keypoint_dist_mm": float(kd_min)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = a.ids.split(",") if a.ids else A.assembly_ids()
    rows = []
    for aid in ids:
        try:
            rows.append(assembly(aid, a.n, a.seed))
        except FileNotFoundError as exc:
            rows.append({"assembly_id": aid, "error": str(exc)})
    Path(a.out).write_text(json.dumps({"switches": C.switches(), "ranges_file": G.RANGES_FILE, "spec_file": A.SPEC_FILE,
                                       "n_draws": a.n, "seed": a.seed, "rows": rows}, indent=1))
    pos = [r for r in rows if r.get("start_successes") or r.get("aligned_g0_min_success")]
    print(f"{len([r for r in rows if r.get('n')])} assemblies with a box; start can satisfy the rule for {len(pos)}:")
    for r in sorted(pos, key=lambda r: -r["p_start_success"]):
        print(f"  {r['assembly_id']}  L {r['L_mm']:.1f} mm  d_dis {r['d_dis_mm']:.1f} mm  {r['start_successes']} of "
              f"{r['n']} starts ({100 * r['p_start_success']:.1f} %)  aligned g0 = 6 mm: {r['aligned_g0_min_success']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
