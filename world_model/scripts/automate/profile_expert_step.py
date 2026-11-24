"""Where the wall time of one recorded control step goes (rs2, GPU 1): physics vs rendering vs IK / python.

    python -u profile_expert_step.py --assembly 00015 --out /mnt/storage3/qingxin/automate/expert_smoke/profile
Runs the reference insertion and one insert trajectory (with its unrecorded transfer) exactly as the collector does
(Recorder on, pictures every 4 steps), with timers around (CUDA synchronised before each timer stops):
    physics    AutomateScene.step (6 PhysX steps of 1/120 s, write_data_to_sim, scene.update)
    render     Robot.frame (camera pose, sim.render, camera update, copies to the CPU, crop, segmentation classes)
    ik         Robot._ik (Pinocchio damped least squares)
    fcl        hpp-fcl collide + distance of the expert's guards and the start check
    snapshot   the expert's truth read-out per step (poses, contact forces, FK)
    record     the recorder's per-step state (qpos, FK hand, truth) without the picture
    other      the rest (python of the expert, numpy)
--device cpu runs PhysX on the CPU (rendering stays on the GPU), to see whether a single environment steps faster there.
Reported: totals and per control step, separately for recorded steps and for transfer steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.sim_automate.app import launch, shutdown  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--assembly", default="00015")
ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/expert_smoke/profile")
ap.add_argument("--device", default="cuda:0", help="PhysX device: cuda:0 (default) or cpu")
args = ap.parse_args()
app = launch(headless=True, cameras=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wm.env.sampling import Offset  # noqa: E402
from wm.sim_automate import expert as E  # noqa: E402
from wm.sim_automate.frames import frame  # noqa: E402
from wm.sim_automate.record import Recorder  # noqa: E402
from wm.sim_automate.robot import Robot  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
T = defaultdict(float)
N = defaultdict(int)
depth = [0]


def timed(name, fn):
    def wrap(*a, **k):
        depth[0] += 1
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if depth[0] == 1:                    # count only the outermost timer (no double counting)
                T[name] += time.perf_counter() - t
                N[name] += 1
            depth[0] -= 1
    return wrap


res = {"assembly": args.assembly, "device": args.device}
code = 0
try:
    r = Robot(args.assembly, device=args.device)
    r.sc.step = timed("physics", r.sc.step)
    r.frame = timed("render", r.frame)
    r._ik = timed("ik", r._ik)
    E.PlugGeometry._dist = timed("fcl", E.PlugGeometry._dist)
    E.Execution._snapshot = timed("snapshot", E.Execution._snapshot)
    rec = Recorder(r)
    rec._sample_orig = rec._sample

    def sample(image, _r=rec):
        t = time.perf_counter()
        depth[0] += 1
        try:
            _r.buf["qpos"].append(_r.robot.qpos().astype(np.float32))
            _r.buf["hand"].append(_r.robot.hand_pose().astype(np.float32))
            from wm.sim_automate.record import truth_rel
            _r.buf["eval_rel"].append(truth_rel(_r.robot))
            torch.cuda.synchronize()
            T["record"] += time.perf_counter() - t
            N["record"] += 1
        finally:
            depth[0] -= 1
        if image:
            _r._image()
    rec._sample = sample
    r.reset(0, clamp_deg=0.0, dp_m=0.0, spin_deg=0.0, socket_index=0)
    ex = E.Expert(r)
    phases = {}
    for name in ("reference", "transfer", "insert"):
        T.clear()
        N.clear()
        s0, t0 = r.sc.steps, time.perf_counter()
        if name == "reference":
            rec.begin()
            out = ex.run("reference", rec.on_step, seed_offset=0)
            rec.end()
        elif name == "transfer":
            F = frame(r.axis(), r.kin.base_T[:3, 0])
            Ts = ex.start_pose(r.hand_pose(), F, r.port_spec.pin_length_m, Offset(0.015, (0.008, -0.006), (3.0, -4.0, 2.0)), ex.geom.g_top)
            ex.start_clearance(Ts)
            out = ex.move_to_start(Ts)
        else:
            rec.begin()
            out = ex.run("insert", rec.on_step, seed_offset=1)
            rec.end()
        wall = time.perf_counter() - t0
        steps = r.sc.steps - s0
        known = sum(T.values())
        phases[name] = {"steps": steps, "wall_s": wall, "s_per_step": wall / max(steps, 1),
                        "ms_per_step": {k: 1e3 * v / max(steps, 1) for k, v in T.items()},
                        "calls": dict(N), "other_ms_per_step": 1e3 * (wall - known) / max(steps, 1),
                        "ok": out.get("completed", out.get("ok"))}
        print(name, json.dumps(phases[name], default=float), flush=True)
    res["phases"] = phases
except Exception:
    import traceback
    res["fatal"] = traceback.format_exc()
    print(res["fatal"], flush=True)
    code = 1
(OUT / f"profile_{args.assembly}_{args.device.replace(':', '')}.json").write_text(json.dumps(res, indent=1, default=float))
print("PROFILE DONE", flush=True)
shutdown(code)
