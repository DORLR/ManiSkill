# Preheld charger insertion v2

`WMInsertionChargerRandomized-v2` extends `PlugCharger-v1` on ManiSkill 3.0.0b4
(`652ad9353c0223507a938f0e8d990dd6f1c771ad`), using SAPIEN 3.0.0b1.

The charger starts preheld at its housing middle on the table, directly in front
of a fixed socket. A zero-DoF articulation child of `panda_hand` preserves the
achieved grasp. Reset performs no physical pickup or simulation steps.

- Achieved grasp pitch: continuous uniform **−45° to −25°**.
- Both holes translate together in board-local Y/Z: **±5 mm per axis**.
  Set `hole_offset_enabled=False` for the fixed-hole condition.
- Pin/hole dimensions stay native: **0.5 mm clearance per side**.
- Native wrist mount/FOV, **256×256 RGB and depth**; no base observation camera.
  Native depth is **int16 millimetres**. `obs_mode="rgb"` remains available.
- One CPU-physics scene; native joint-position/gripper control, **20 Hz** control,
  **100 Hz** physics and an **800-control** horizon.
- The original Franka URDF, collision clearances, contact offsets and drives are
  retained. The builder adds a fixed charger child; position solver iterations
  are raised to at least 128 for finger/charger contact stability.
- White visual boxes follow the shifted collision openings; native coplanar
  gold hole decorations are omitted.

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make("WMInsertionChargerRandomized-v2", obs_mode="rgbd")
obs, info = env.reset(seed=1420000)
rgb = obs["sensor_data"]["hand_camera"]["rgb"]
depth_mm = obs["sensor_data"]["hand_camera"]["depth"]
env.close()
```

For calibration, reset accepts `grasp_pitch_deg` within the active range and
`hole_offset_yz_m` within ±0.005 m. Scene geometry is rebuilt at every reset.
The raw state modes and `info["charger_v2"]` contain simulator truth for auditing;
RGB-D observations expose robot proprioception and wrist measurements.

This environment retains native pose-only `info["success"]`. A scripted expert,
dataset collection and additional depth/stability benchmark scoring are separate
from this environment change.
