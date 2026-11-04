# Preheld charger insertion v2

`WMInsertionChargerRandomized-v2` extends `PlugCharger-v1` on ManiSkill 3.0.0b4
(`652ad9353c0223507a938f0e8d990dd6f1c771ad`), using SAPIEN 3.0.0b1.

The charger starts preheld on the table directly in front of a fixed white
backboard. A zero-DoF articulation child of `panda_hand` preserves the achieved
grasp. Reset performs no physical pickup or simulation steps.

## Port types

Select a matched plug and socket with the `port_type` constructor argument.
The default `"A"` uses the native geometry. B, C, and D use the corresponding
synthetic connector geometries.

| Port | Pin layout | Full pin cross-section (Y × Z) | Pin length |
| --- | --- | --- | --- |
| A | Two narrow flat pins | 1.5 × 6.4 mm each | 16 mm |
| B | One wide rectangular blade | 12 × 3.6 mm | 18 mm |
| C | Three square pins in a triangle | 3.2 × 3.2 mm each | 20 mm |
| D | Two asymmetric horizontal and vertical blades | 6.4 × 2.4 mm and 2.4 × 6.4 mm | 14 mm |

The audit includes the selected port, its dimension record and hash, and each
opening's center and half sizes in the board frame. All four types have
**0.5 mm clearance per side**. Each port's fixed spawn places its housing on
the table with a 12 mm projected axial gap from the pin tips to the board.

## Configurable randomization

All three factors are enabled by default, use continuous uniform sampling, and
have independent deterministic RNG streams.

| Factor | Enable key | Range key and default | Disabled value |
| --- | --- | --- | --- |
| Grasp pitch | `grasp_pitch_enabled` | `pitch_range_deg=(-60., -20.)` | `fixed_grasp_pitch_deg=-35.` |
| Aperture pattern position | `hole_offset_enabled` | `hole_offset_range_yz_m=((-0.020, 0.020), (-0.020, 0.020))` | Y=Z=0 |
| Housing grasp location | `grasp_position_enabled` | `grasp_position_range_m=(-0.030, -0.010)` | `fixed_grasp_position_m=-0.020` |

The white backboard actor and its outer boundary stay fixed for each port
type. The matched openings translate together in a **40 x 40 mm** board-local
Y/Z square, and the collision apertures, visible apertures, and goal follow
that translation. Configured regions that put any opening outside the board
are rejected.

Grasp position is the continuous local-X coordinate of the grasp center on the
housing, measured from the pin root at X=0. More negative means rear and less
negative means front. The default leaves 10 mm at each end of the 40 mm A
housing. All port types use the same absolute X range, so the relative
placement varies with housing length. The position changes the grasp
transform while the charger world spawn stays fixed for the selected type.
The vertical grasp-center offset is 1.5 mm.

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "WMInsertionChargerRandomized-v2",
    port_type="A",
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
pose-only `info["success"]` is retained.

## Insertion geometry diagnostics

`geometry_from_poses(charger_pose, goal_pose, port_spec)` in the task's
`metrics` module reports static geometry from simulator poses. Pass the
current `base.goal_pose`, which includes the aperture offset and the
180-degree rotation about board Z. Goal-frame +X points into the board.

`depth_ok` requires a plug-root axial coordinate from −1.0 to +0.2 mm,
every pin box's minimum X at least −1.0 mm, and every leading face's
minimum X at least its pin length minus 1.0 mm. The resulting leading-face
thresholds for A/B/C/D are 15/17/19/13 mm. Per-pin slot fit compares the
transformed pin against its matched opening. `pin_slot_fit` uses the full
pin and an extended slot, while `pin_in_board_slot_fit` restricts the
calculation to the finite board slab. The latter is null when no pin
intersects the slab.

Physical validation uses pose, contacts, grasp drift, and stability evidence
in addition to static geometry. The standalone geometry tests exercise the
native A thresholds and all four matched aperture patterns.

```bash
python -m unittest discover -s tests -p test_wminsertion_ports.py
```

## Scripted demonstrations

Install the motion planning dependencies for the pinned runtime, then run:

```bash
python examples/collect_wminsertion_v2.py --output-dir demos/wminsertion-v2 \
    --attempts 1 --seed 10043 --obs-mode rgbd --port-type A
```

The expert reads the simulator socket goal and achieved grasp transform at
reset. It lifts the preheld charger, moves laterally at a safe standoff,
aligns outside the board and inserts through the native joint-position
controller. It is a demonstration generator with access to simulator state.
The default `clean` profile follows a nominal path; `--profile mixed` adds
seeded perturbations and recovery.

The scripted expert defaults to `tracking_bias_gain=0.75` to compensate
steady joint following error. During alignment outside the board, it filters
the difference between the previous sent arm command and measured joint
angles. A calibration sample requires measured endpoint speed below
0.3 mm/s and low angular speed, joint speed and socket contact. With
compensation enabled, insertion admission requires five consecutive
qualifying calibration samples by default, plus the existing alignment
checks. The control loop advances the calibration from measured conditions.

The estimate is held during insertion. The expert projects each
compensated target to the uncompensated axial coordinate before resolving
IK, bounds the joint offset to 0.003 rad per arm joint, and checks the
command against the joint, FK spacing and clearance guards. This changes
the expert's joint commands through the existing position controller. The
actuator settings, simulator physics and native pose success rule are the
same.

The default `terminal_condition="insertion_geometry"` applies to
`final_verify` for all four port types. It requires `depth_ok` and
`pin_slot_fit` on the measured pose, a transverse error at most 0.3 mm,
and an orientation error at most 0.2°. It also checks endpoint linear
speed at most 0.5 mm/s, endpoint angular speed at most 0.5°/s, and maximum
arm joint speed at most 0.02 rad/s. By default all checks must pass for
five consecutive control samples. The expert records the original
TCP-based convergence and slot-fit test separately as `strict_tcp_converged`.

Both the collector and diagnostic accept `--tracking-bias-gain 0` to
disable compensation and `--terminal-condition legacy_tcp` to select the
previous TCP stopping rule. The two options can be set independently.

Collection assigns exactly `--attempts` consecutive seeds. The ManiSkill
recorder writes `trajectory.h5` and its standard JSON metadata. The
`attempts.jsonl` ledger records every assigned outcome, including reset
failures and unsaved attempts. Each saved entry refers to its HDF5 group by
`trajectory_id`; per-action phase, behavior-cloning mask and segment ID
align with the HDF5 actions. In the HDF5, observation `t` is the input for
action `t`; the last observation has no action.

By default only trajectories that finish the expert's guarded motion and
satisfy the final native pose flag are saved. `--save-failures` also keeps
nonempty partial trajectories with complete action metadata. Simulator
states in the HDF5 are for replay. The ledger records final static geometry
and a separate `depth_confirmed` flag, which requires expert completion,
`depth_ok` and `pin_slot_fit`. A physical validation run also audits
stability over the simulated steps.

## Nominal scripted validation

Run the eight fixed cases on the pinned runtime.

```bash
python examples/validate_wminsertion_ports.py --output-dir demos/wminsertion-ports \
    --ports A B C D --profiles clean mixed --seed 10043 \
    --pitch -35 --grasp-position -0.020 --hole-offset 0 0
```

Each run below assigned A/B/C/D with both profiles, at the same fixed
pitch, grasp position and aperture offset.

| Diagnostic | Expert completed with depth confirmed | Final pose met static depth and slot checks |
| --- | ---: | ---: |
| Initial uncompensated teacher, legacy TCP stop (001) | 2/8 | 4/8 |
| Calibrated teacher, legacy TCP stop (004) | 6/8 | 8/8 |
| Calibrated teacher, common insertion geometry stop (008) | 8/8 | 8/8 |

In 001, both C poses met the static checks after the teacher aborted at
the transverse-position guard. In 004, both D cases reached the time limit
during `final_verify` after meeting the static checks. In 008, all eight
cases passed the separate substep stability audit and the five consecutive
terminal checks. The two D cases completed at 729 and 718 control steps
with `strict_tcp_converged=false` at termination. No board-slab slot
violation was recorded in the 008 substep audits.

These scripted results cover the fixed configuration above. The wider
pitch, grasp and aperture ranges define the sampling configuration for
further collection.
