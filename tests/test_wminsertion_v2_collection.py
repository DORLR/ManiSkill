"""Test attempt accounting without a renderer or simulator."""

from pathlib import Path
import runpy
import unittest

collect_attempt = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/collect_wminsertion_v2.py")
)["collect_attempt"]


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


if __name__ == "__main__":
    unittest.main()
