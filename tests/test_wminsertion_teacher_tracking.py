"""Regression tests for the teacher's tracking compensation without a scene."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.wminsertion_v2.common import _pose_error
from mani_skill.envs.tasks.tabletop.wminsertion_v2.ports import get_port_spec
from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.continuous import (
    ContinuousScriptedPolicy,
    ContinuousScriptedPolicyConfig,
    _ContinuousExecution,
    _PolicyFailure,
)


METRES_PER_RAD = .1


def make_execution(*, profile="clean", gain=.75, insertion=False, **config_overrides):
    x = -.10 if insertion else -.18
    command = np.array([x, 0., 0., 0., 0., 0., 0., .022, .022], dtype=np.float32).astype(float)
    robot = SimpleNamespace(qpos=torch.tensor(command[None]), qvel=torch.zeros((1, 9)))
    robot.get_qpos = lambda: robot.qpos
    robot.get_qvel = lambda: robot.qvel
    base = SimpleNamespace(
        agent=SimpleNamespace(robot=robot, tcp=SimpleNamespace()),
        port_spec=get_port_spec("D"),
        control_timestep=.05,
    )
    policy = ContinuousScriptedPolicy(
        ContinuousScriptedPolicyConfig(
            profile=profile, tracking_bias_gain=gain, tracking_bias_filter=.25,
            **config_overrides,
        )
    )
    execution = _ContinuousExecution(policy, SimpleNamespace(unwrapped=base), None)
    yaw = math.sqrt(.5)
    execution.goal = sapien.Pose([.03, -.02, .10], [yaw, 0., 0., yaw])
    execution.h = sapien.Pose([.002, .003, 0.])
    execution._initialize_pin_geometry()
    execution.require_outside_wall = not insertion
    execution.insertion_active = insertion
    execution.last_sent_arm = command[:7].copy()
    execution.joint_limits = np.tile([-1., 1.], (9, 1))
    execution.planner = SimpleNamespace(
        check_for_self_collision=lambda **kwargs: False,
        check_for_env_collision=lambda **kwargs: False,
    )
    execution.metrics.update(
        tcp_endpoint_linear_velocity_m_s=0.,
        tcp_endpoint_angular_velocity_deg_s=0.,
        socket_contact_force_n=0.,
        actual_minimum_pin_corner_tip_gap_mm=4.,
        actual_transverse_error_mm=0.,
        actual_goal_attitude_error_deg=0.,
        all_pin_slot_fit=True,
        actual_transverse_velocity_m_s=0.,
        actual_axial_velocity_m_s=0.,
        minimum_slot_margin_mm=.5,
    )

    def linear_fk(q):
        return execution.goal * sapien.Pose(q[:3] * METRES_PER_RAD) * execution.h.inv()

    def linear_ik(target, seed):
        relative = execution.goal.inv() * target * execution.h
        solution = seed.copy()
        solution[:3] = relative.p / METRES_PER_RAD
        return solution

    execution._fk_tcp = linear_fk
    execution._exact_ik = linear_ik
    base.agent.tcp.pose = linear_fk(command)
    return execution, command


def set_terminal_pose(execution, relative):
    execution.base.charger = SimpleNamespace(pose=execution.goal * relative)
    execution.metrics.update(
        tcp_error=_pose_error(execution._tcp(relative), execution._tcp(sapien.Pose())),
        arm_joint_velocity_rad_s=0.,
        all_pin_slot_fit=execution._slot_geometry(relative)["all_pin_slot_fit"],
    )


class TestTeacherTracking(unittest.TestCase):
    def test_clean_and_recovery_calibrate_from_previous_sent_minus_actual(self):
        for profile, phase in (("clean", "continuous_approach"), ("mixed", "recovery_align")):
            with self.subTest(profile=profile):
                execution, command = make_execution(profile=profile)
                if profile == "mixed":
                    execution._begin("perturbation_stochastic")
                execution._begin(phase)
                execution.last_sent_arm += [.004, -.002, .001, 0., 0., 0., 0.]
                near = sapien.Pose([-.017, 0., 0.])
                self.assertFalse(execution._readiness(near)[0])

                samples = execution.config.settle_consecutive_steps
                for _ in range(samples):
                    execution._compensate_tracking(command)
                expected = (1. - .75 ** samples) * np.array([.003, -.002, .001, 0., 0., 0., 0.])
                np.testing.assert_allclose(execution.drive_bias, expected, atol=1e-12)
                self.assertEqual(execution.drive_bias_samples, samples)
                self.assertTrue(execution.tracking_metadata["tracking_estimate_updated"])
                self.assertTrue(execution._readiness(near)[0])

                execution.base.agent.robot.qvel[0, 0] = .03
                execution._compensate_tracking(command)
                np.testing.assert_allclose(execution.drive_bias, expected, atol=1e-12)
                self.assertFalse(execution._readiness(near)[0])

    def test_insertion_freezes_estimate_and_projects_out_axial_bias(self):
        execution, command = make_execution(insertion=True)
        execution._begin("axial_push")
        bias = np.array([.002, .001, 0., 0., 0., 0., 0.])
        execution.drive_bias = bias.copy()
        execution.drive_bias_samples = 6
        execution.last_sent_arm[0] -= .002

        sent, offset, diagnostics = execution._compensate_tracking(command)
        nominal = execution.goal.inv() * execution._fk_tcp(command) * execution.h
        corrected = execution.goal.inv() * execution._fk_tcp(sent) * execution.h
        self.assertAlmostEqual(float(corrected.p[0]), float(nominal.p[0]), delta=1e-7)
        self.assertAlmostEqual(float(corrected.p[1] - nominal.p[1]), .000075, delta=1e-7)
        self.assertGreater(offset[1], 0.)
        self.assertTrue(diagnostics["fk_partial_pin_slot_safe"])
        np.testing.assert_array_equal(execution.drive_bias, bias)
        self.assertEqual(execution.drive_bias_samples, 6)
        self.assertFalse(execution.tracking_metadata["tracking_estimate_updated"])
        self.assertTrue(execution.tracking_metadata["tracking_estimate_frozen_in_insertion"])

    def test_projection_ik_failure_uses_reduced_or_zero_compensation(self):
        for failures, expected_scale in ((1, .5), (4, 0.)):
            with self.subTest(failures=failures):
                execution, command = make_execution(insertion=True)
                execution._begin("final_verify")
                execution.drive_bias[1] = .002
                linear_ik = execution._exact_ik

                def fail_then_solve(target, seed):
                    if execution._exact_ik.call_count <= failures:
                        raise _PolicyFailure("projection_ik_failed")
                    return linear_ik(target, seed)

                execution._exact_ik = Mock(side_effect=fail_then_solve)
                sent, offset, diagnostics = execution._compensate_tracking(command)
                self.assertEqual(execution._exact_ik.call_count, min(failures + 1, 4))
                self.assertEqual(execution.tracking_metadata["tracking_applied_scale"], expected_scale)
                self.assertTrue(diagnostics["fk_partial_pin_slot_safe"])
                if expected_scale:
                    self.assertAlmostEqual(offset[1], .75 * .002 * expected_scale, delta=1e-6)
                else:
                    np.testing.assert_array_equal(sent, command)
                    np.testing.assert_array_equal(offset, np.zeros(7))

    def test_zero_gain_sends_nominal_command_without_projection_ik(self):
        execution, command = make_execution(gain=0.)
        execution._begin("continuous_approach")
        execution.drive_bias[1] = .002
        execution._exact_ik = Mock(return_value=command.copy())
        execution._step = Mock()

        sent = execution._send(sapien.Pose([-.018, 0., 0.]))
        execution._exact_ik.assert_called_once()
        np.testing.assert_array_equal(execution._step.call_args.args[0], command)
        np.testing.assert_array_equal(sent, command)
        np.testing.assert_array_equal(execution.action_metadata["q_nominal"], command[:7])
        np.testing.assert_array_equal(execution.action_metadata["q_sent"], command[:7])
        np.testing.assert_array_equal(execution.action_metadata["tracking_compensation_rad"], np.zeros(7))
        self.assertTrue(execution._readiness(sapien.Pose([-.017, 0., 0.]))[0])


class TestTeacherTerminalCondition(unittest.TestCase):
    def test_d_insertion_geometry_accepts_valid_depth_outside_tcp_ball(self):
        relative = sapien.Pose([-.000287, .000227, 0.])
        execution, _ = make_execution(insertion=True)
        set_terminal_pose(execution, relative)
        ready, evidence = execution._terminal_ready()

        self.assertTrue(ready)
        self.assertEqual(evidence["terminal_condition"], "insertion_geometry")
        self.assertFalse(evidence["strict_tcp_converged"])
        self.assertGreater(execution.metrics["tcp_error"]["translation_mm"], .3)
        self.assertTrue(all(evidence["checks"].values()))
        self.assertEqual(evidence["geometry"]["pin_length_m"], .014)
        self.assertAlmostEqual(evidence["geometry"]["axial_error_mm"], -.287, places=4)

        legacy, _ = make_execution(insertion=True, terminal_condition="legacy_tcp")
        set_terminal_pose(legacy, relative)
        self.assertFalse(legacy._terminal_ready()[0])
        set_terminal_pose(legacy, sapien.Pose([-.0001, 0., 0.]))
        self.assertTrue(legacy._terminal_ready()[0])

    def test_depth_overtravel_rejected_even_with_strict_tcp_convergence(self):
        relative = sapien.Pose([.00025, 0., 0.])
        execution, _ = make_execution(insertion=True)
        set_terminal_pose(execution, relative)
        ready, evidence = execution._terminal_ready()

        self.assertFalse(ready)
        self.assertFalse(evidence["checks"]["axial_depth"])
        self.assertTrue(evidence["checks"]["pin_slot_fit"])
        self.assertTrue(evidence["strict_tcp_converged"])

    def test_failed_geometry_or_stability_gate_blocks_completion(self):
        half_angle = math.radians(.4) / 2
        poses = {
            "pin_slot_fit": sapien.Pose([-.0003, .00051, 0.]),
            "orientation": sapien.Pose(
                [-.0003, 0., 0.], [math.cos(half_angle), math.sin(half_angle), 0., 0.]
            ),
        }
        for failed_check, relative in poses.items():
            with self.subTest(failed_check=failed_check):
                execution, _ = make_execution(insertion=True)
                set_terminal_pose(execution, relative)
                ready, evidence = execution._terminal_ready()
                self.assertFalse(ready)
                self.assertFalse(evidence["checks"][failed_check])

        velocities = (
            ("tcp_velocity", "tcp_endpoint_linear_velocity_m_s", .0006),
            ("angular_velocity", "tcp_endpoint_angular_velocity_deg_s", .6),
            ("joint_velocity", "arm_joint_velocity_rad_s", .03),
        )
        for failed_check, metric, value in velocities:
            with self.subTest(failed_check=failed_check):
                execution, _ = make_execution(insertion=True)
                set_terminal_pose(execution, sapien.Pose([-.0003, 0., 0.]))
                execution.metrics[metric] = value
                ready, evidence = execution._terminal_ready()
                self.assertFalse(ready)
                self.assertFalse(evidence["checks"][failed_check])
                self.assertTrue(evidence["checks"]["axial_depth"])
                self.assertTrue(evidence["checks"]["pin_slot_fit"])


if __name__ == "__main__":
    unittest.main()
