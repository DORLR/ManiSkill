"""Test attempt accounting without a renderer or simulator."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

collector = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/collect_wminsertion_v2.py")
)
collect_attempt = collector["collect_attempt"]
parse_args = collector["parse_args"]
run_collection = collector["run_collection"]


class Recorder:
    def __init__(self, reset_error=False):
        self.reset_error = reset_error
        self.flushes = []

    def reset(self, seed):
        if self.reset_error:
            raise RuntimeError("IK reset failed")
        return {}, {"charger_v2": {"grasp_position_m": -0.02}}

    def flush_trajectory(self, *, save):
        self.flushes.append(save)


class Teacher:
    def __init__(self, completed, native_success, steps=2):
        self.completed = completed
        self.native_success = native_success
        self.steps = steps

    def run(self, env, on_step):
        for phase, segment in (("perturbation", None), ("recovery", 1)):
            on_step(
                phase,
                None,
                {
                    "success": self.native_success,
                    "scripted_policy": {
                        "use_for_bc": segment is not None,
                        "bc_segment_id": segment,
                    },
                },
            )
        return {"completed": self.completed, "steps": self.steps}


class TestCollection(unittest.TestCase):
    def test_pose_only_success_does_not_admit_aborted_expert(self):
        env = Recorder()
        row = collect_attempt(env, Teacher(False, True), 10043, 0)
        self.assertIsNone(row["trajectory_id"])
        self.assertEqual(env.flushes, [False])
        self.assertEqual(row["steps"], 2)

    def test_action_metadata_keeps_perturbation_before_recovery(self):
        env = Recorder()
        row = collect_attempt(env, Teacher(True, True), 10043, 3)
        self.assertEqual(row["trajectory_id"], 3)
        self.assertEqual(env.flushes, [True])
        self.assertEqual([a["use_for_bc"] for a in row["actions"]], [False, True])
        self.assertEqual([a["bc_segment_id"] for a in row["actions"]], [None, 1])

    def test_reset_failure_keeps_seed_without_flushing_previous_attempt(self):
        env = Recorder(reset_error=True)
        row = collect_attempt(env, Teacher(True, True), 10043, 0)
        self.assertEqual(row["seed"], 10043)
        self.assertIn("IK reset failed", row["failure_reason"])
        self.assertIsNone(row["trajectory_id"])
        self.assertEqual(env.flushes, [])

    def test_missing_action_metadata_prevents_saving_even_with_failures_enabled(self):
        env = Recorder()
        row = collect_attempt(env, Teacher(True, True, steps=3), 10043, 0, True)
        self.assertEqual(row["failure_reason"], "control_step_metadata_mismatch")
        self.assertIsNone(row["trajectory_id"])
        self.assertEqual(env.flushes, [False])

    def test_save_failures_preserves_partial_trajectory_without_relabeling(self):
        env = Recorder()
        row = collect_attempt(env, Teacher(False, False), 10043, 4, True)
        self.assertEqual(row["trajectory_id"], 4)
        self.assertFalse(row["expert_completed"])
        self.assertEqual(env.flushes, [True])

    def test_depth_failure_is_reported_independently_of_legacy_save(self):
        env = Recorder()
        env.unwrapped = SimpleNamespace(
            charger=SimpleNamespace(pose=object()), goal_pose=object(), port_spec=object()
        )
        geometry = {"depth_ok": False, "pin_slot_fit": True, "axial_error_mm": 0.25}
        score = Mock(return_value=geometry)
        row = collect_attempt(env, Teacher(True, True), 10043, 0, geometry_from_poses=score)
        score.assert_called_once_with(
            env.unwrapped.charger.pose, env.unwrapped.goal_pose, env.unwrapped.port_spec
        )
        self.assertTrue(row["expert_completed"])
        self.assertTrue(row["native_pose_success"])
        self.assertEqual(row["final_geometry"], geometry)
        self.assertFalse(row["depth_confirmed"])
        self.assertEqual(row["trajectory_id"], 0)
        self.assertEqual(env.flushes, [True])

    def test_constructor_failure_records_every_assigned_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "failed_collection"
            args = parse_args([
                "--output-dir", str(output_dir), "--attempts", "3", "--seed", "17",
                "--port-type", "D",
            ])
            make_env = Mock(side_effect=RuntimeError("constructor reset IK failed"))
            record_episode, teacher, score = Mock(), Mock(), Mock()
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                status = run_collection(args, make_env, record_episode, teacher, score)
            rows = [
                json.loads(line)
                for line in (output_dir / "attempts.jsonl").read_text().splitlines()
            ]
            metadata = json.loads((output_dir / "collection.json").read_text())

        self.assertEqual(status, 1)
        self.assertEqual(metadata["assigned_attempts"], 3)
        self.assertEqual(metadata["default_save_eligibility"], "expert_completed_and_native_pose_success")
        make_env.assert_called_once_with(collector["ENV_ID"], **metadata["env_kwargs"])
        record_episode.assert_not_called()
        teacher.run.assert_not_called()
        score.assert_not_called()
        self.assertEqual([row["seed"] for row in rows], [17, 18, 19])
        for row in rows:
            self.assertEqual(row["failure_reason"], "constructor: RuntimeError: constructor reset IK failed")
            self.assertIsNone(row["trajectory_id"])
            self.assertFalse(row["depth_confirmed"])
            self.assertEqual(row["steps"], 0)

    def test_fatal_recording_error_marks_remaining_seeds_unattempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "failed_recording"
            args = parse_args(["--output-dir", str(output_dir), "--attempts", "3"])
            env = Recorder()
            env.flush_trajectory = Mock(side_effect=OSError("HDF5 write failed"))
            env.close = Mock()
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                status = run_collection(
                    args, Mock(return_value=env), Mock(return_value=env),
                    Teacher(True, True), None,
                )
            rows = [
                json.loads(line)
                for line in (output_dir / "attempts.jsonl").read_text().splitlines()
            ]

        self.assertEqual(status, 1)
        self.assertEqual([row["seed"] for row in rows], [10043, 10044, 10045])
        self.assertTrue(rows[0]["failure_reason"].startswith("recording: OSError"))
        self.assertTrue(all(row["failure_reason"].startswith("not_attempted_after_recording:")
                            for row in rows[1:]))
        self.assertTrue(all(row["trajectory_id"] is None for row in rows))
        env.flush_trajectory.assert_called_once_with(save=True)
        env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
