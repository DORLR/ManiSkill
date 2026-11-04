"""Bounded ABCD scripted rollouts with wrist video and geometric/contact audits.

This is a privileged expert diagnostic, not a learned-policy benchmark.
Each assigned case is recorded even if reset or execution fails.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def json_value(value):
    """Normalize native/NumPy scalars, including booleans in expert metadata."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ports", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--profiles", nargs="+", choices=("clean", "mixed"), default=["clean", "mixed"])
    parser.add_argument("--seed", type=int, default=10043)
    parser.add_argument("--pitch", type=float, default=-35.)
    parser.add_argument("--grasp-position", type=float, default=-.020)
    parser.add_argument("--hole-offset", type=float, nargs=2, default=(0., 0.))
    parser.add_argument("--tracking-bias-gain", type=float, default=.75)
    parser.add_argument("--terminal-condition", choices=("insertion_geometry", "legacy_tcp"), default="insertion_geometry")
    args = parser.parse_args()
    if len(args.ports) != len(set(args.ports)) or len(args.profiles) != len(set(args.profiles)):
        parser.error("Ports and profiles must not contain duplicates")
    return args


def main():
    args = parse_args()
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    import mani_skill.envs
    from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _raw, _sp, _pose_error
    from mani_skill.envs.tasks.tabletop.wminsertion_v2.metrics import geometry_from_poses
    from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2 import (
        ContinuousScriptedPolicyConfig, RandomContinuousScriptedPolicy,
    )
    from mani_skill.utils import common

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    assigned = [(port, profile) for port in args.ports for profile in args.profiles]
    manifest = {"assigned_cases": len(assigned), "cases": records,
                "kind": "privileged_scripted_diagnostic", "status": "running",
                "requested": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}

    def save():
        target = args.output_dir / "manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(json_value(manifest), indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    save()
    for port, profile in assigned:
        started = time.perf_counter()
        case_id = f"{port}_{profile}"
        folder = args.output_dir / case_id
        folder.mkdir()
        row = {"case": case_id, "port_type": port, "profile": profile, "seed": args.seed,
               "reset_ok": False, "expert_completed": False, "depth_confirmed": False,
               "failure_reason": None, "steps": 0, "sim_seconds": 0.,
               "video": None, "frames": {}, "finite": True,
               "max_socket_penetration_mm": 0., "max_socket_contact_force_n": 0.,
               "max_h_translation_mm": 0., "max_h_rotation_deg": 0.,
               "max_joint_velocity_rad_s": 0., "physics_substeps": 0,
               "board_slot_violation_substeps": 0, "nonfinite_substeps": 0,
               "close_errors": []}
        records.append(row)
        env = writer = trace = None
        phase = "reset"
        first_phase = set()
        try:
            env = gym.make("WMInsertionChargerRandomized-v2", port_type=port,
                           obs_mode="rgbd", control_mode="pd_joint_pos", sim_backend="cpu",
                           num_envs=1, seed=args.seed, max_episode_steps=800)
            _, info = env.reset(seed=args.seed, options={
                "grasp_pitch_deg": args.pitch, "grasp_position_m": args.grasp_position,
                "hole_offset_yz_m": args.hole_offset})
            base = env.unwrapped
            row.update(reset_ok=True, reset_audit=info["charger_v2"])
            row["port_geometry"] = base.port_spec.as_dict()
            row["port_geometry_sha256"] = base.port_spec.sha256
            reference_h = _sp(base.agent.tcp.pose).inv() * _sp(base.charger.pose)
            original_hook = base._after_simulation_step

            def substep():
                original_hook()
                row["physics_substeps"] += 1
                q = common.to_numpy(base.agent.robot.get_qpos())
                v = common.to_numpy(base.agent.robot.get_qvel())
                finite = bool(np.isfinite(q).all() and np.isfinite(v).all()
                              and np.isfinite(_raw(base.charger.pose)).all()
                              and np.isfinite(_raw(base.agent.tcp.pose)).all())
                row["finite"] &= finite
                if not finite:
                    row["nonfinite_substeps"] += 1
                    return
                row["max_joint_velocity_rad_s"] = max(row["max_joint_velocity_rad_s"], float(np.abs(v).max()))
                h = _pose_error(_sp(base.agent.tcp.pose).inv() * _sp(base.charger.pose), reference_h)
                row["max_h_translation_mm"] = max(row["max_h_translation_mm"], h["translation_mm"])
                row["max_h_rotation_deg"] = max(row["max_h_rotation_deg"], h["rotation_deg"])
                geometry = geometry_from_poses(base.charger.pose, base.goal_pose, base.port_spec)
                row["board_slot_violation_substeps"] += geometry["pin_in_board_slot_fit"] is False
                socket_force = 0.
                for contact in base.scene.get_contacts():
                    if not any("receptacle" in body.entity.name for body in contact.bodies):
                        continue
                    penetration = max((max(0., -float(p.separation) * 1000) for p in contact.points), default=0.)
                    force = float(np.linalg.norm(np.sum([np.asarray(p.impulse) for p in contact.points], axis=0)) * base.sim_freq)
                    row["max_socket_penetration_mm"] = max(row["max_socket_penetration_mm"], penetration)
                    socket_force += force
                row["max_socket_contact_force_n"] = max(row["max_socket_contact_force_n"], socket_force)

            base._after_simulation_step = substep
            writer = imageio.get_writer(folder / "wrist.mp4", fps=10, codec="libx264", macro_block_size=None)
            trace = (folder / "trace.jsonl").open("x")

            def frame(label=None, to_video=False):
                obs = base.get_obs()
                rgb = np.asarray(common.to_numpy(obs["sensor_data"]["hand_camera"]["rgb"]))[0]
                assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
                if to_video:
                    writer.append_data(rgb)
                if label and label not in row["frames"]:
                    path = folder / f"{label}.png"
                    Image.fromarray(rgb).save(path)
                    row["frames"][label] = {"path": f"{case_id}/{path.name}", "step": row["steps"],
                                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

            frame("reset", True)

            def on_step(current_phase, action, step_info):
                nonlocal phase
                phase = current_phase
                row["steps"] += 1
                if not row["finite"]:
                    return "non_finite_simulation_state"
                g = geometry_from_poses(base.charger.pose, base.goal_pose, base.port_spec)
                metadata = step_info["scripted_policy"]
                trace.write(json.dumps(json_value({"step": row["steps"], "phase": phase, "action": action.tolist(),
                                        "geometry": g, "tcp_pose": _raw(base.agent.tcp.pose),
                                        "tracking": {k: metadata[k] for k in (
                                            "q_nominal", "q_noisy", "q_sent", "q_actual_pre",
                                            "tracking_bias_estimate_rad", "tracking_compensation_rad",
                                            "tracking_estimate_updated", "tracking_calibration_samples",
                                            "tracking_applied_scale", "tracking_estimate_frozen_in_insertion")
                                            if k in metadata}}), allow_nan=False) + "\n")
                label = phase if phase not in first_phase else None
                first_phase.add(phase)
                if row["steps"] % 2 == 0 or label:
                    frame(label, row["steps"] % 2 == 0)
                row["sim_seconds"] = row["steps"] / float(base.control_freq)
                if not row["finite"] or row["max_socket_penetration_mm"] > 3.:
                    return "nonfinite_or_socket_penetration_over_3mm"
                if row["max_h_translation_mm"] > 5. or row["max_h_rotation_deg"] > 5.:
                    return "fixed_grasp_constraint_violation"
                if row["steps"] % 100 == 0:
                    save()

            teacher = RandomContinuousScriptedPolicy(ContinuousScriptedPolicyConfig(
                profile=profile, tracking_bias_gain=args.tracking_bias_gain,
                terminal_condition=args.terminal_condition))
            result = teacher.run(env, on_step=on_step)
            row.update(expert_completed=bool(result["completed"]), failure_reason=result.get("failure_reason"),
                       policy_result=result, final_geometry=geometry_from_poses(base.charger.pose, base.goal_pose, base.port_spec))
            g = row["final_geometry"]
            row["geometric_insertion"] = bool(g["depth_ok"] and g["pin_slot_fit"])
            row["depth_confirmed"] = bool(row["expert_completed"] and g["depth_ok"] and g["pin_slot_fit"])
            row["native_pose_success"] = bool(common.to_numpy(base.evaluate()["success"]).reshape(-1)[0])
            frame("final", True)
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            for obj, name in ((writer, "video"), (trace, "trace"), (env, "environment")):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception as exc:
                        row["close_errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            video = folder / "wrist.mp4"
            if video.exists() and video.stat().st_size > 0:
                row["video"] = f"{case_id}/wrist.mp4"
            row["stability_audit_pass"] = bool(
                row["reset_ok"] and row["finite"] and row["physics_substeps"] > 0
                and row["max_socket_penetration_mm"] <= .5
                and row["max_socket_contact_force_n"] <= 80.
                and row["max_h_translation_mm"] <= .25 and row["max_h_rotation_deg"] <= .5
                and row["max_joint_velocity_rad_s"] <= 20.)
            row["stability_thresholds"] = {"socket_penetration_mm": .5, "socket_force_n": 80.,
                "H_translation_mm": .25, "H_rotation_deg": .5, "joint_velocity_rad_s": 20.}
            row["wall_seconds"] = time.perf_counter() - started
            save()
            print(json.dumps({k: row[k] for k in ("case", "reset_ok", "expert_completed", "depth_confirmed", "steps", "failure_reason")}), flush=True)
    manifest["status"] = "complete"
    manifest["depth_confirmed_cases"] = sum(r["depth_confirmed"] for r in records)
    save()


if __name__ == "__main__":
    main()
