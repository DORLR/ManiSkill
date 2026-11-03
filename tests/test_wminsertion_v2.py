"""Focused checks for the CPU-physics, wrist-RGBD insertion task."""

import gymnasium as gym
import numpy as np
import pytest
import torch

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.wminsertion_v2.geometry import (
    mathematical_self_check,
    sample_hole_offset,
    sample_pitch,
    validate_hole_offset,
)
from mani_skill.utils import common


def test_socket_partition_and_paired_sampling():
    assert mathematical_self_check()["passed"]
    for seed in range(10):
        assert sample_pitch(seed) == sample_pitch(seed)
        assert sample_hole_offset(seed, False) == (0.0, 0.0)
        assert all(abs(x) <= 0.005 for x in sample_hole_offset(seed))


def test_holes_cannot_leave_the_calibrated_region():
    with pytest.raises(ValueError):
        validate_hole_offset((0.006, 0.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA rendering")
def test_wrist_rgb_depth_and_fixed_grasp():
    env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="rgbd")
    try:
        obs, _ = env.reset(seed=1420000)
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
