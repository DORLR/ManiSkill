# Preheld charger insertion v2

`WMInsertionChargerRandomized-v2` extends `PlugCharger-v1` on ManiSkill 3.0.0b4
(`652ad9353c0223507a938f0e8d990dd6f1c771ad`), using SAPIEN 3.0.0b1.

The charger starts preheld on the table directly in front of a fixed white
backboard. A zero-DoF articulation child of `panda_hand` preserves the achieved
grasp. Reset performs no physical pickup or simulation steps.

## Configurable randomization

All three factors are enabled by default, use continuous uniform sampling, and
have independent deterministic RNG streams.

| Factor | Enable key | Range key and default | Disabled value |
| --- | --- | --- | --- |
| Grasp pitch | `grasp_pitch_enabled` | `pitch_range_deg=(-60., -20.)` | `fixed_grasp_pitch_deg=-35.` |
| Hole-pair position | `hole_offset_enabled` | `hole_offset_range_yz_m=((-0.020, 0.020), (-0.020, 0.020))` | Y=Z=0 |
| Housing grasp location | `grasp_position_enabled` | `grasp_position_range_m=(-0.030, -0.010)` | `fixed_grasp_position_m=-0.020` |

The white backboard actor and its outer boundary stay fixed. Only the two
openings translate together in a **40 x 40 mm** board-local Y/Z square; the
collision geometry, visible openings and goal follow that translation. The
whole socket board does not move. Configured regions that put either opening
outside the board are rejected.

Grasp position is the continuous local-X coordinate of the grasp center on the
**40 mm housing**, measured from the pin root at X=0. More negative means rear;
less negative means front. The default spans rear/middle/front without discrete
bins and leaves 10 mm at each housing end. It changes the grasp transform, not
the charger's world spawn. The vertical grasp-center offset remains 1.5 mm.

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "WMInsertionChargerRandomized-v2",
    obs_mode="rgbd",
    seed=10043,
    grasp_pitch_enabled=True,
    pitch_range_deg=(-60., -20.),
    hole_offset_enabled=True,
    hole_offset_range_yz_m=((-0.020, 0.020), (-0.020, 0.020)),
    grasp_position_enabled=True,
    grasp_position_range_m=(-0.030, -0.010),
)
obs, info = env.reset(seed=10043)
rgb = obs["sensor_data"]["hand_camera"]["rgb"]
depth_mm = obs["sensor_data"]["hand_camera"]["depth"]
env.close()
```

The constructor seed defaults to **10043** and initializes the episode RNG.
Subsequent `reset()` calls advance that RNG; `reset(seed=10043)` explicitly
replays the same scene. Reset IK uses the constructor seed unless a separate
`reset_ik_seed` is supplied. Reset calibration options are `grasp_pitch_deg`,
`grasp_position_m`, and `hole_offset_yz_m`, inside their configured ranges.

## Fixed task settings

- Pin/hole dimensions stay native: **0.5 mm clearance per side**.
- Native wrist mount/FOV, **256x256 RGB and depth**; no base observation camera.
  Depth is **int16 millimetres**. `obs_mode="rgb"` remains available.
- One CPU-physics scene; native joint-position/gripper control, **20 Hz** control,
  **100 Hz** physics and an **800-control** horizon.
- The original Franka URDF, contact offsets and drives are retained. The builder
  adds a fixed charger child; position solver iterations are at least 128.
- White visual boxes follow the real collision openings; native coplanar gold
  hole decorations are omitted. Scene geometry is rebuilt at every reset.

The raw state modes and `info["charger_v2"]` contain simulator truth for auditing;
RGB-D observations expose robot proprioception and wrist measurements. Native
pose-only `info["success"]` is retained. Scripted policy, data collection and
additional benchmark scoring remain separate changes.
