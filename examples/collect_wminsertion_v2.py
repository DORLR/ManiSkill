"""Record a bounded batch of scripted WMInsertion-v2 demonstrations."""

import argparse
from dataclasses import asdict, is_dataclass
import json
import sys
from pathlib import Path

ENV_ID = "WMInsertionChargerRandomized-v2"
SCENE_KEYS = (
    "port_type",
    "port_geometry",
    "port_geometry_sha256",
    "native_package_versions",
    "achieved_grasp_pitch_deg",
    "grasp_position_m",
    "hole_offset_yz_m",
)


def _attempt_row(seed, failure_reason=None):
    return {
        "seed": seed,
        "trajectory_id": None,
        "expert_completed": False,
        "native_pose_success": None,
        "final_geometry": None,
        "depth_confirmed": False,
        "failure_reason": failure_reason,
        "steps": 0,
        "actions": [],
    }


def collect_attempt(
    env, teacher, seed, trajectory_id, save_failures=False, *, geometry_from_poses=None
):
    """Keep the outcome of every assigned reset, even if no HDF5 group is saved."""
    row = _attempt_row(seed)
    try:
        _, info = env.reset(seed=seed)
    except Exception as error:
        row["failure_reason"] = f"reset: {type(error).__name__}: {error}"
        return row
    audit = info.get("charger_v2", {})
    row["scene"] = {key: audit[key] for key in SCENE_KEYS if key in audit}

    def on_step(phase, _action, step_info):
        metadata = step_info["scripted_policy"]
        row["actions"].append(
            {
                "phase": phase,
                "use_for_bc": bool(metadata["use_for_bc"]),
                "bc_segment_id": metadata["bc_segment_id"],
            }
        )
        success = step_info["success"]
        row["native_pose_success"] = bool(
            success.item() if hasattr(success, "item") else success
        )

    teacher_failed = False
    try:
        result = teacher.run(env, on_step=on_step)
        row["steps"] = result["steps"]
        row["expert_completed"] = bool(result["completed"])
        row["failure_reason"] = result.get("failure_reason")
    except Exception as error:
        row["steps"] = len(row["actions"])
        row["failure_reason"] = f"expert: {type(error).__name__}: {error}"
        teacher_failed = True

    if geometry_from_poses is not None:
        try:
            base = env.unwrapped
            row["final_geometry"] = geometry_from_poses(
                base.charger.pose, base.goal_pose, base.port_spec
            )
        except Exception as error:
            row["final_geometry_error"] = f"{type(error).__name__}: {error}"

    if teacher_failed:
        env.flush_trajectory(save=False)
        return row

    if len(row["actions"]) != row["steps"]:
        row["failure_reason"] = "control_step_metadata_mismatch"
        row["expert_completed"] = False
        env.flush_trajectory(save=False)
        return row

    geometry = row["final_geometry"]
    row["depth_confirmed"] = bool(
        row["expert_completed"] and geometry and geometry["depth_ok"] and geometry["pin_slot_fit"]
    )
    # Retain the original eligibility rule independently of the geometry audit.
    completed = row["expert_completed"] and row["native_pose_success"] is True
    if not completed and not row["failure_reason"]:
        row["failure_reason"] = "expert_or_native_pose_check_failed"
    save = row["steps"] > 0 and (completed or save_failures)
    env.flush_trajectory(save=save)
    if save:
        row["trajectory_id"] = trajectory_id
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10043)
    parser.add_argument("--obs-mode", choices=("rgb", "rgbd"), default="rgbd")
    parser.add_argument("--profile", choices=("clean", "mixed"), default="clean")
    parser.add_argument("--port-type", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--tracking-bias-gain", type=float, default=.75)
    parser.add_argument("--terminal-condition", choices=("insertion_geometry", "legacy_tcp"), default="insertion_geometry")
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args(argv)
    if args.attempts < 1 or not 0 <= args.seed <= args.seed + args.attempts - 1 < 2**32:
        parser.error("attempts must be positive and all assigned seeds must fit in uint32")
    return args


def run_collection(args, make_env, record_episode, teacher, geometry_from_poses):
    env_kwargs = dict(
        port_type=args.port_type,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        num_envs=1,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        schema_version=1,
        env_id=ENV_ID,
        env_kwargs=env_kwargs,
        first_seed=args.seed,
        assigned_attempts=args.attempts,
        expert_profile=args.profile,
        expert_config=asdict(teacher.config) if is_dataclass(getattr(teacher, "config", None)) else None,
        save_failures=args.save_failures,
        default_save_eligibility="expert_completed_and_native_pose_success",
        depth_confirmation="expert_completed_and_depth_ok_and_pin_slot_fit",
    )
    (args.output_dir / "collection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    env = None
    fatal_error = None
    saved = completed = depth_confirmed = 0
    try:
        with (args.output_dir / "attempts.jsonl").open("x", encoding="utf-8") as ledger:
            try:
                env = make_env(ENV_ID, **env_kwargs)
                env = record_episode(
                    env,
                    output_dir=args.output_dir,
                    trajectory_name="trajectory",
                    save_video=False,
                    save_on_reset=False,
                    source_type="motionplanning",
                    source_desc="WMInsertion-v2 scripted expert",
                )
            except Exception as error:
                fatal_error = f"constructor: {type(error).__name__}: {error}"
            for seed in range(args.seed, args.seed + args.attempts):
                if fatal_error is not None:
                    row = _attempt_row(seed, fatal_error)
                else:
                    try:
                        row = collect_attempt(
                            env, teacher, seed, saved, args.save_failures,
                            geometry_from_poses=geometry_from_poses,
                        )
                    except Exception as error:
                        reason = f"recording: {type(error).__name__}: {error}"
                        row = _attempt_row(seed, reason)
                        fatal_error = f"not_attempted_after_{reason}"
                saved += row["trajectory_id"] is not None
                completed += row["expert_completed"] and row["native_pose_success"] is True
                depth_confirmed += row["depth_confirmed"]
                ledger.write(json.dumps(row, allow_nan=False) + "\n")
                ledger.flush()
    finally:
        if env is not None:
            env.close()
    if fatal_error is not None:
        print(fatal_error, file=sys.stderr)
    print(
        f"{completed}/{args.attempts} expert completions, "
        f"{depth_confirmed}/{args.attempts} depth confirmed, {saved} trajectories saved"
    )
    return int(fatal_error is not None)


def main():
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.envs.tasks.tabletop.wminsertion_v2.metrics import geometry_from_poses
    from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2 import (
        ContinuousScriptedPolicyConfig,
        RandomContinuousScriptedPolicy,
    )
    from mani_skill.utils.wrappers.record import RecordEpisode

    config = ContinuousScriptedPolicyConfig(profile=args.profile, tracking_bias_gain=args.tracking_bias_gain,
                                            terminal_condition=args.terminal_condition)
    teacher = RandomContinuousScriptedPolicy(config)
    return run_collection(args, gym.make, RecordEpisode, teacher, geometry_from_poses)


if __name__ == "__main__":
    raise SystemExit(main())
