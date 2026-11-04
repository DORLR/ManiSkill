"""Read the native wrist RGB and depth from one v2 reset."""

import gymnasium as gym
import mani_skill.envs


def main():
    env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="rgbd", seed=10043)
    try:
        obs, info = env.reset(seed=10043)
        wrist = obs["sensor_data"]["hand_camera"]
        rgb = wrist["rgb"]  # uint8 [1, 256, 256, 3]
        depth_mm = wrist["depth"]  # int16 [1, 256, 256, 1]
        depth_m = depth_mm.float() / 1000
        print("RGB:", rgb.shape, rgb.dtype)
        print("Depth (mm):", depth_mm.shape, depth_mm.dtype)
        print("Depth range (m):", depth_m.min().item(), depth_m.max().item())
    finally:
        env.close()


if __name__ == "__main__":
    main()
