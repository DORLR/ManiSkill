"""Choose the expert's approach speed: approach + alignment from far starts at several speed limits (rs2, GPU 1).

    python -u probe_expert_speed.py --assembly 00015 --out /mnt/storage3/qingxin/automate/expert_smoke/probe
For each speed v (m/s; angular limit 15 deg/s * v / 15 mm/s; acceleration 50 mm/s^2, 60 deg/s^2) and each start
(corners of the start box: g0 = 18 mm, lateral (+-10, +-10) mm, rotations +-6 deg), unrecorded transfer to the start,
then Execution.approach_align(stand-off) and the axial insertion. Reported per run: steps of the approach segment,
steps until admission (true plug within 0.3 mm / 0.2 deg of the stand-off and still for 5 steps), the largest lag of
the FK hand behind the commanded TCP [mm], the smallest commanded plug-socket clearance [mm], the true plug at
admission, insertion steps and the final truth, wall seconds per control step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.sim_automate.app import launch, shutdown  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--assembly", default="00015")
ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/expert_smoke/probe")
ap.add_argument("--speeds", default="0.010,0.015,0.020")
ap.add_argument("--grasp", default="0,0,0", help="d_theta deg, d_p m, d_psi deg")
args = ap.parse_args()
app = launch(headless=True, cameras=False)

import numpy as np  # noqa: E402

from wm.env.sampling import Offset  # noqa: E402
from wm.sim_automate import checks  # noqa: E402
from wm.sim_automate import expert as E  # noqa: E402
from wm.sim_automate.frames import frame  # noqa: E402
from wm.sim_automate.robot import Robot, grasp_kwargs  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
res = {"assembly": args.assembly, "grasp": args.grasp, "runs": []}
code = 0
try:
    r = Robot(args.assembly, camera=False)
    gr = [float(v) for v in args.grasp.split(",")]
    r.reset(0, **grasp_kwargs(gr), socket_index=0)
    ex = E.Expert(r)
    # reference (default speed) gives H*
    t0 = time.time()
    ref = ex.run("reference", None, seed_offset=0)
    res["reference"] = {k: ref.get(k) for k in ("completed", "failure_reason", "steps", "wall_s", "peak")}
    res["reference"]["truth"] = checks.truth(r)
    res["reference"]["stages"] = ref["stages"]
    H = r.hand_pose()
    F = frame(r.axis(), r.kin.base_T[:3, 0])
    L = r.port_spec.pin_length_m
    starts = [Offset(0.018, (0.010, 0.010), (6.0, -6.0, 6.0)), Offset(0.018, (-0.010, 0.010), (-6.0, 6.0, -6.0)),
              Offset(0.012, (0.0, -0.010), (0.0, 0.0, 6.0))]
    for v in [float(x) for x in args.speeds.split(",")]:
        E.APPROACH_SPEED_M_S = v
        E.APPROACH_ANG_SPEED_DEG_S = 15.0 * v / 0.015
        for si, st in enumerate(starts):
            T = ex.start_pose(H, F, L, st, ex.geom.g_top)
            clr = ex.start_clearance(T)
            mv = ex.move_to_start(T)
            row = {"speed_m_s": v, "start": st.as_dict(), "start_clearance_mm": clr["min"] * 1e3,
                   "transfer": {k: mv[k] for k in ("ok", "stage", "steps", "seconds", "start_error_mm")}}
            if not mv["ok"]:
                res["runs"].append(row)
                continue
            lag = []

            def on_step(phase, q, info, _x=None):
                pass
            exe = E.Execution(r, ex.geom, None, np.random.default_rng(si))
            orig_step = exe._step

            def step(q7, _orig=orig_step, _exe=exe):
                _orig(q7)
                if _exe.cmd is not None:
                    lag.append(float(np.linalg.norm(_exe.tcp(_exe.cmd)[:3, 3] - r.hand_pose()[:3, 3])))
            exe._step = step
            ts = time.time()
            try:
                exe.approach_align(exe.stand_off())
                st0 = exe.stages[-1]
                row.update(approach_align_steps=st0["steps"], admission_step=st0.get("admission_step"),
                           at_admission={"top_gap_mm": st0["top_gap_mm"], "lateral_mm": st0["lateral_mm"],
                                         "tilt_deg": st0["tilt_deg"]})
                n_seg = len(lag)
                exe.insert(noise=False)
                row.update(insert_steps=exe.stages[-2]["steps"], verify_steps=exe.stages[-1]["steps"], ok=True)
            except E.ExpertFailure as exc:
                row.update(ok=False, failure=str(exc), stages=exe.stages)
            row.update(max_lag_mm=max(lag) * 1e3 if lag else None, steps=exe.steps,
                       s_per_step=(time.time() - ts) / max(exe.steps, 1), truth=checks.truth(r), peak=exe.peak)
            res["runs"].append(row)
            print(json.dumps(row, default=float)[:600], flush=True)
            (OUT / f"probe_{args.assembly}.json").write_text(json.dumps(res, indent=1, default=float))
except Exception:
    import traceback
    res["fatal"] = traceback.format_exc()
    print(res["fatal"], flush=True)
    code = 1
(OUT / f"probe_{args.assembly}.json").write_text(json.dumps(res, indent=1, default=float))
print("PROBE DONE", flush=True)
shutdown(code)
