"""Cheap check before collection: what does the native wrist camera see near
the pre-insertion pose, and how long do the expert, a move and a render take?

Usage: python scripts/probe_view.py --out results/probe
"""

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from paf.env_adapter import Robot
from paf.geometry import make_T, port_axes, pose_from_state, relative_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--ports", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--pitches", nargs="+", type=float, default=[-60.0, -40.0, -20.0])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for port in args.ports:
        robot = Robot(port)
        for pitch in args.pitches:
            row = {"port": port, "pitch": pitch}
            t0 = time.time()
            robot.reset(seed=1000 + int(-pitch), pitch_deg=pitch, grasp_pos_m=-0.020)
            row["reset_s"] = time.time() - t0
            t0 = time.time()
            ins = robot.expert_insert()
            row.update(expert_s=time.time() - t0, expert_ok=ins["completed"], expert_steps=ins["steps"],
                       failure=ins["failure"], phases=sorted({p for p, _ in ins["poses"]}))
            if not ins["completed"]:
                log.append(row)
                print(row, flush=True)
                continue
            ins_poses = [H for p, H in ins["poses"] if p == "axial_push"]
            H_star = robot.hand_pose()
            axis = ins_poses[-1][:3, 3] - ins_poses[0][:3, 3]
            row["insert_stroke_mm"] = float(np.linalg.norm(axis) * 1000)
            R_P = port_axes(axis)
            s = robot.port_spec.pin_length_m + 0.003
            H_pre = make_T(H_star[:3, :3], H_star[:3, 3] - s * R_P[:, 0])
            t0 = time.time()
            ok = robot.move_to(H_pre)
            row["retreat_s"] = time.time() - t0
            row["retreat_ok"] = ok
            views = {
                "q0": np.zeros(6),
                "back10": np.array([-0.010, 0, 0, 0, 0, 0]),
                "y+5": np.array([-0.003, 0.005, 0, 0, 0, 0]),
                "z+5": np.array([-0.003, 0, 0.005, 0, 0, 0]),
                "rz+5deg": np.array([-0.003, 0, 0, 0, 0, np.radians(5)]),
                "ry+5deg": np.array([-0.003, 0, 0, 0, np.radians(5), 0]),
            }
            for name, q in views.items():
                t0 = time.time()
                ok = robot.move_to(pose_from_state(q, H_pre, R_P))
                tm = time.time() - t0
                t0 = time.time()
                obs = robot.observe()
                tr = time.time() - t0
                qa = relative_state(obs.H, H_pre, R_P)
                row[f"{name}_move_s"], row[f"{name}_render_s"], row[f"{name}_ok"] = tm, tr, ok
                row[f"{name}_err_mm"] = float(np.linalg.norm(qa[:3] - q[:3]) * 1000)
                stem = f"{port}_p{int(pitch)}_{name}"
                imageio.imwrite(out / f"{stem}.png", obs.rgb)
                d = obs.depth_mm.astype(np.float32)
                valid = d > 0
                dv = np.zeros_like(d)
                if valid.any():
                    lo, hi = np.percentile(d[valid], [1, 99])
                    dv = np.clip((d - lo) / max(hi - lo, 1), 0, 1) * valid
                imageio.imwrite(out / f"{stem}_depth.png", (dv * 255).astype(np.uint8))
            t0 = time.time()
            for _ in range(20):
                robot._command(robot.qpos(), 1)
            row["physics_step_s"] = (time.time() - t0) / 20
            log.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("_s") or "expert" in k}), flush=True)
        robot.env.close()
    (out / "probe.json").write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
