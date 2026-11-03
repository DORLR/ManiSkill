"""Focused checks for the CPU-physics, wrist-RGBD insertion task."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from itertools import product

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.wminsertion_v2.geometry import (
    mathematical_self_check,
    sample_hole_offset,
    sample_pitch,
    validate_hole_offset,
    sample_grasp_position,
    DEFAULT_SEED,
)
from mani_skill.utils import common


def test_socket_partition_and_paired_sampling():
    assert mathematical_self_check()["passed"]
    for seed in range(10):
        assert sample_pitch(seed) == sample_pitch(seed)
        assert sample_hole_offset(seed, False) == (0.0, 0.0)
        assert all(abs(x) <= 0.020 for x in sample_hole_offset(seed))
        assert sample_pitch(seed, enabled=False) == -35.0
        assert sample_grasp_position(seed, enabled=False) == -0.020
    positions = [sample_grasp_position(seed) for seed in range(1000)]
    assert (
        len(set(positions)) == 1000
    )  # continuous positions, not front/middle/rear bins
    assert min(positions) < -0.028 and max(positions) > -0.012


def test_holes_cannot_leave_the_calibrated_region():
    with pytest.raises(ValueError):
        validate_hole_offset((0.021, 0.0))


def test_custom_sampling_ranges():
    for seed in range(100):
        assert -45 <= sample_pitch(seed, (-45, -25)) <= -25
        offset = sample_hole_offset(
            seed, offset_ranges=((-0.01, 0.01), (-0.002, 0.003))
        )
        assert -0.01 <= offset[0] <= 0.01 and -0.002 <= offset[1] <= 0.003
        assert -0.025 <= sample_grasp_position(seed, (-0.025, -0.015)) <= -0.015


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA rendering")
def test_wrist_rgb_depth_and_fixed_grasp():
    env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="rgbd")
    try:
        obs, _ = env.reset(seed=DEFAULT_SEED)
        assert tuple(obs["sensor_data"]) == ("hand_camera",)
        wrist = obs["sensor_data"]["hand_camera"]
        rgb, depth = common.to_numpy(wrist["rgb"]), common.to_numpy(wrist["depth"])
        assert rgb.shape == (1, 256, 256, 3) and rgb.dtype == np.uint8
        assert depth.shape == (1, 256, 256, 1) and depth.dtype == np.int16
        assert np.count_nonzero(depth > 0) > 100
        assert env.spec.max_episode_steps == 800
        qpos = common.to_numpy(env.unwrapped.agent.robot.get_qpos()).reshape(-1)
        action = np.r_[qpos[:7], -1.0].astype(np.float32)
        for _ in range(20):
            env.step(action)
        drift = env.unwrapped.audit()["achieved_H_drift"]
        assert drift["translation_mm"] <= 0.25
        assert drift["rotation_deg"] <= 0.5
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA rendering")
def test_three_independent_switches_and_fixed_board():
    board = plug = None
    for pitch_on, hole_on, position_on in product((False, True), repeat=3):
        env = gym.make(
            "WMInsertionChargerRandomized-v2",
            obs_mode="state",
            grasp_pitch_enabled=pitch_on,
            hole_offset_enabled=hole_on,
            grasp_position_enabled=position_on,
        )
        try:
            row = env.unwrapped.audit()
            assert row["episode_seed"] == DEFAULT_SEED
            assert row["reset_ik_seed"] == DEFAULT_SEED
            assert row["requested_grasp_pitch_deg"] == sample_pitch(
                DEFAULT_SEED, enabled=pitch_on
            )
            assert row["hole_offset_yz_m"] == list(
                sample_hole_offset(DEFAULT_SEED, hole_on)
            )
            assert row["grasp_position_m"] == sample_grasp_position(
                DEFAULT_SEED, enabled=position_on
            )
            if board is None:
                board, plug = (
                    row["initial_receptacle_pose"],
                    row["initial_charger_pose"],
                )
            np.testing.assert_allclose(row["initial_receptacle_pose"], board, atol=1e-7)
            np.testing.assert_allclose(
                row["initial_charger_pose"][:3], plug[:3], atol=1e-6
            )
            assert (
                row["board_outer_boundary_fixed"]
                and not row["whole_board_translation_applied"]
            )
        finally:
            env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA rendering")
def test_pitch_and_continuous_grip_calibration_extremes():
    env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="state")
    try:
        board = env.unwrapped.audit()["initial_receptacle_pose"]
        for angle, position in product((-60.0, -40.0, -20.0), (-0.030, -0.020, -0.010)):
            _, info = env.reset(
                seed=DEFAULT_SEED,
                options={
                    "grasp_pitch_deg": angle,
                    "grasp_position_m": position,
                    "hole_offset_yz_m": [-0.020, 0.020],
                },
            )
            row = info["charger_v2"]
            assert abs(row["achieved_grasp_pitch_deg"] - angle) <= 0.2
            assert row["grip_center_in_charger_m"][0] == position
            np.testing.assert_allclose(row["initial_receptacle_pose"], board, atol=1e-7)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA rendering")
def test_default_seed_replay_and_advancing_unseeded_resets():
    env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="state")
    try:
        assert env.unwrapped.audit()["episode_seed"] == DEFAULT_SEED
        env.reset(seed=DEFAULT_SEED)
        first = env.unwrapped.audit()
        env.reset(seed=DEFAULT_SEED)
        replay = env.unwrapped.audit()
        for key in (
            "requested_grasp_pitch_deg",
            "hole_offset_yz_m",
            "grasp_position_m",
            "initial_qpos",
        ):
            assert first[key] == replay[key]
        env.reset()
        unseeded_a = env.unwrapped.audit()["episode_seed"]
        env.reset()
        unseeded_b = env.unwrapped.audit()["episode_seed"]
        assert unseeded_a != unseeded_b
    finally:
        env.close()
