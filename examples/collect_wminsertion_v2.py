"""Record a bounded batch of scripted WMInsertion-v2 demonstrations."""

import argparse
import json
from pathlib import Path

ENV_ID = "WMInsertionChargerRandomized-v2"
SCENE_KEYS = (
    "native_package_versions",
    "achieved_grasp_pitch_deg",
    "grasp_position_m",
    "hole_offset_yz_m",
)


def collect_attempt(env, teacher, seed, trajectory_id, save_failures=False):
    """Keep the outcome of every assigned reset, even if no HDF5 group is saved."""
    row = {
        "seed": seed,
        "trajectory_id": None,
        "expert_completed": False,
        "native_pose_success": None,
        "failure_reason": None,
        "steps": 0,
        "actions": [],
    }
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

    try:
        result = teacher.run(env, on_step=on_step)
        row["steps"] = result["steps"]
        row["expert_completed"] = bool(result["completed"])
        row["failure_reason"] = result.get("failure_reason")
    except Exception as error:
        row["steps"] = len(row["actions"])
        row["failure_reason"] = f"expert: {type(error).__name__}: {error}"
        env.flush_trajectory(save=False)
        return row

    if len(row["actions"]) != row["steps"]:
        row["failure_reason"] = "control_step_metadata_mismatch"
        row["expert_completed"] = False
        env.flush_trajectory(save=False)
        return row

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
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args(argv)
    if args.attempts < 1 or not 0 <= args.seed <= args.seed + args.attempts - 1 < 2**32:
        parser.error("attempts must be positive and all assigned seeds must fit in uint32")
    return args


def main():
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2 import (
        ContinuousScriptedPolicyConfig,
        RandomContinuousScriptedPolicy,
    )
    from mani_skill.utils.wrappers.record import RecordEpisode

    config = ContinuousScriptedPolicyConfig(profile=args.profile)
    teacher = RandomContinuousScriptedPolicy(config)
    env_kwargs = dict(
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
        save_failures=args.save_failures,
    )
    (args.output_dir / "collection.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    env = gym.make(ENV_ID, **env_kwargs)
    saved = completed = 0
    try:
        env = RecordEpisode(
            env,
            output_dir=args.output_dir,
            trajectory_name="trajectory",
            save_video=False,
            save_on_reset=False,
            source_type="motionplanning",
            source_desc="WMInsertion-v2 scripted expert",
        )
        with (args.output_dir / "attempts.jsonl").open("x", encoding="utf-8") as ledger:
            for seed in range(args.seed, args.seed + args.attempts):
                row = collect_attempt(env, teacher, seed, saved, args.save_failures)
                saved += row["trajectory_id"] is not None
                completed += row["expert_completed"] and row["native_pose_success"] is True
                ledger.write(json.dumps(row, allow_nan=False) + "\n")
                ledger.flush()
    finally:
        env.close()
    print(f"{completed}/{args.attempts} expert completions; {saved} trajectories saved")


if __name__ == "__main__":
    main()
