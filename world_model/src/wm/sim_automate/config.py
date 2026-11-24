"""Constants of the AutoMate scene (milestone 1). Units: metres, radians unless the name says deg / mm.

Frames
    world   the environment frame of env i (Isaac Lab world minus scene.env_origins[i]); the robot base
            (panda_link0) sits at its origin with identity rotation, as in AutoMate.
    hand    panda_hand body;  tcp = hand * Trans(0, 0, 0.1034) (link panda_hand_tcp of our ManiSkill Panda);
            hand_pose() returns the tcp pose, as wm.sim.env_adapter.Robot.hand_pose() does.
    socket  root of AutoMate's fixed asset (socket.usd); the seated plug pose equals the socket pose and the
            insertion direction is -z_socket (wm.sim_automate.assets).

Decisions of the user, 2026-11-22 (deviation table; "approved" where it differs from AutoMate's original):
    success          AutoMate's own check_plug_inserted_in_socket, unchanged (checks.py); the ManiSkill-style
                     depth / lateral / tilt / spin numbers are recorded as information only        (= AutoMate)
    socket pose      a new draw at every reset, AutoMate's randomize_fixed_initial_state (SOCKET_DRAW "per_reset")
                                                                                                    (= AutoMate)
    grasp spin       about the plug's own axis through the tight-fit centroid (GRASP_SPIN_AXIS "plug_axis");
                     AutoMate has no spin variable                                  (approved by the user 2026-11-22)
    physics device   CPU PhysX for collection (PHYSICS_DEVICE "cpu"); AutoMate runs PhysX on the GPU; GPU remains an
                     option and a CPU-vs-GPU comparison on hard assemblies is a check   (approved by the user 2026-11-22)
    grasp-box margin hand / finger - socket distance >= 0.5 mm at H* (grasp.MARGIN_M, was 1 mm), to be confirmed in
                     simulation for the assemblies that had no box               (approved by the user 2026-11-22)
    00855            excluded: plug wider than the socket collar in the CAD (grasp.EXCLUDED)
                                                                                    (approved by the user 2026-11-22)
    camera           unchanged: ManiSkill's D405 mount, intrinsics and 320 x 320 crop  (approved by the user 2026-11-22)
    GPU 0            our processes may hold on GPU 0 only a graphics context (nvidia-smi type G) of at most 50 MiB per
                     process (Kit's renderer enumerates every GPU through Vulkan), never a compute context (type C or
                     C+G); app.assert_off_forbidden_gpu enforces it                 (decided by the user 2026-11-22)
Decisions of the user, 2026-11-23 (deviation table audit_am_r4/deviations_table.md, rows as numbered there):
    start            (row 14, r4-expert-01 option B) the start box relative to H*: gap g0 6-18 mm measured from the
                     socket top, lateral +-10 mm, +-6 deg per axis; back-off along a  b = L + g_top + g0  with
                     g_top = max(L_top, L) - L (START_GAP_FROM "socket_top", default and APPROVED); every start is
                     outside the socket, as every start of AutoMate's evaluation (Insertion, assembly_env.py:786-811,
                     SBC off in evaluation :80-81: plug aligned at the seated pose, lifted to d_dis + 10 mm, xy noise
                     +-10 mm, fixed grasp). "entrance" (b = L + g0, the box until 2026-11-23, under which 25.7 % of
                     00133's starts, 2.1 % of 01036's and 0.2 % of 00559's already satisfied AutoMate's success rule;
                     diag/start_success.py, the round-4 audit's own draws gave 35.6 / 3.4 / 0.5 %) is refused at
                     import unless WM_AUTOMATE_DIAGNOSTIC=1. Recomputed under "socket_top" (rs2 2026-11-23,
                     diag/start_success.py, 20,000 draws for each of the 91 assemblies with a box): 0 starts satisfy
                     AutoMate's rule                                                  (decided by the user 2026-11-23)
                     Information (r7-expert-02; AutoMate's rule unchanged): the rule holds once the plug is within
                     min(d_dis, 15 mm) of seated. Where that exceeds L, it holds while the tight-fit band is still
                     above the entrance, by min(d_dis, 15 mm) - L (geometry_v7): 00133 12.8 mm, 01036 6.6, 00559 6.2,
                     00731 / 00700 / 00320 / 00187 5.0, 01129 4.8, 00755 3.6, 01079 / 00256 2.4, 00138 / 00030 2.2 mm
                     (the other 87 <= 2 mm). On these assemblies "success" (S_any) of an insert / perturbed run and
                     automate_success_any of a contact trial can be True for a plug stuck above the tight fit; the
                     imitation set needs expert_completed (final_verify, depth_left < 0.5 mm), so it is not affected
    spin symmetry    (row 37, r4-decisions-06 option 2 on a named list) the 16 assemblies of
                     assets.SPIN_ROUND_2026_11_23 (15 with a 12-gon plug, and 00863; for each the physical check
                     plug-against-socket says round with zero overlap) use symmetry "round" (s = 15 deg) for the spin
                     box and the spin information columns; every other assembly keeps the n of fit.py; the split
                     category stays the plug's shape class (assets.SPEC_FILE geometry_v7, grasp.RANGES_FILE ranges_v5)
                                                                                      (decided by the user 2026-11-23)
    approved 2026-11-23 (keep ours; the user's answer to the deviation table, rows 9, 10, 11, 13, 15, 17, 20, 22):
    control          (row 9) joint-position PD at 20 Hz (DECIMATION 6 x 1/120 s), ManiSkill Panda gains (ARM_*),
                     absolute joint targets; AutoMate: task-space impedance with torques at 15 Hz (decimation 8).
                     Added after the answer (not on the page they approved; to be shown to the user with the smoke
                     results): effort limit ARM_EFFORT_LIMIT 100 N m on all 7 joints (ManiSkill's force limit) against
                     AutoMate's 87 N m (joints 1-4) and 12 N m (joints 5-7) (AM assembly_env_cfg.py actuators
                     panda_arm1 / panda_arm2); velocity limits, friction 0 and armature 0 kept from AutoMate
                     (scene.py:102-108)
    fingers          (row 10) held at the opening where they touch the plug (robot.Robot.reset; follows from the weld:
                     the plug is an articulation link and self-collisions are off); AutoMate: finger target 0
    contact report   (row 11) omni.physx contact processing switched on for the expert's guards
                     (scene.socket_contacts; it copies contact data out, the dynamics do not change); AutoMate / Isaac
                     Lab: off
    re-home          (row 13) after a failed placement the robot is reset with the same grasp and the same socket pose
                     of that trajectory (collect.py); a new start is drawn, not a new socket pose
    expert budget    (row 15) step_budget (expert.py) per recorded trajectory; AutoMate: episode_length_s 5 s = 75 env
                     steps at 15 Hz. Policies: N_max decisions (assets.py, row 16, approved 2026-11-21)
    success timing   (row 17) evaluated after every 20 Hz control step (6 physics steps); AutoMate: after every 15 Hz
                     env step (8 physics steps)
    assembly gate    (row 20) assemblies without a grasp box, 00855 and (ii) / (iii) failures are not collected
                     (plan.json lists each with its reason); AutoMate's list has 100 assemblies
    rejected draws   (row 22) a socket draw whose reset has no IK solution or whose unrecorded reference insertion fails
                     is not collected (reset_failed / label_failed, collect.py); AutoMate keeps every draw. Condition of
                     the approval: the per-assembly reset_failed / label_failed share is reported, and any share > 5 %
                     is shown to the user (contact_report_error, placement_failed and a refused t00 reference
                     (t00_refused: the whole grasp, its 12 planned trajectories; share of grasps) also drop draws; the
                     table lists them for the user under the same 5 % rule, added after the answer)
    acknowledged 2026-11-23 (row 21, n/a): AutoMate's observation noise of the socket position (1 mm), action bounds,
                     rewards and curriculum are not used (no state observation, no RL during collection)
Approved earlier (2026-11-21, task definition): plug welded to the hand (AutoMate: held by friction); in-hand grasp
variation (clamp angle, offset along and spin about the plug axis; AutoMate: one optimised grasp); policy horizon
N_max. Restored 2026-11-23: camera far clip 100 m as ManiSkill (decision 6, camera unchanged; it had been 10 m). The
expert's own deviations from Chongyu's expert (rows 23-36, 38 and 39) and their approvals are listed in expert.py;
restored 2026-11-23 (decision 4, after round 5): the contact-offset move and the random motion follow wm_v1's
_move_outside exactly (2.7 mm/s, 2 deg/s, no ramp, no hold; rows 27 and 38); approved 2026-11-23 (decision 7, row 39):
hand / finger >= 1 mm from the socket on nominal outside commands.
Switches read from the environment at import (all recorded by switches(); a collection freezes them in
<out>/inputs/switches.env): WM_AUTOMATE_SOCKET_DRAW, WM_AUTOMATE_SPIN_AXIS, WM_AUTOMATE_DEVICE, WM_PHYSX_THREADS,
WM_AUTOMATE_MOUNT, WM_AUTOMATE_WELD, WM_CAM_SCALE, WM_KIT_GPU, WM_FORBIDDEN_GPU, WM_AUTOMATE_START_GAP_FROM. A mount
other than "table", a camera scale other than 1, a forbidden GPU other than 0 or a start gap other than "socket_top"
is refused at import unless
WM_AUTOMATE_DIAGNOSTIC=1 (diagnostics only; such a process must never write collection data). APPROVED pins the
approved value of every switch except the PhysX device; refuse_unapproved() (called by batch_assemblies.py before the
app starts, and by the collector at freeze time and in its validation) refuses any other value outside diagnostics.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation

from ..sim import config as _MS   # ManiSkill experiment constants (camera stream, crops); plain numpy module

# ------------------------------------------------------------------ control: 20 Hz joint-position PD
PHYSICS_DT = 1.0 / 120.0           # AutoMate's sim dt (assembly_env_cfg.AssemblyEnvCfg.sim.dt), unchanged
DECIMATION = 6                     # 6 x 1/120 s = 0.05 s per control step (AutoMate: 8 -> 15 Hz)
CONTROL_DT = PHYSICS_DT * DECIMATION
# ManiSkill Panda (mani_skill/agents/robots/panda/panda.py, arm_pd_joint_pos of control mode "pd_joint_pos"):
# stiffness 1e3 N m/rad, damping 1e2 N m s/rad, force limit 100 N m (AutoMate: 87 / 12 N m, deviation row 9),
# absolute targets (normalize_action=False), PhysX implicit drive in force mode in both simulators. The gripper keeps
# AutoMate's actuator (7500, 173, 40 N); its target is the opening at which the fingers touch the plug (robot.Robot.reset), not AutoMate's 0.
ARM_STIFFNESS = 1.0e3
ARM_DAMPING = 1.0e2
ARM_EFFORT_LIMIT = 100.0

# ------------------------------------------------------------------ wrist camera (D405 stream of wm.sim.config)
CAM_SCALE = int(os.environ.get("WM_CAM_SCALE", "1"))   # as wm.env.robot: 2 renders 1280 x 960 (diagnostics)
CAM_W, CAM_H = _MS.CAM_W * CAM_SCALE, _MS.CAM_H * CAM_SCALE
CAM_HFOV_DEG = _MS.CAM_HFOV_DEG
DEPTH_MIN_M = _MS.DEPTH_MIN_MM / 1000.0
DEPTH_UNIT_M = 1e-4                # stored depth unit 0.1 mm, as wm.env.robot
CROP = tuple(CAM_SCALE * v for v in _MS.REL_CROP)      # rows r0:r1, cols c0:c1 -> 320 x 320 (x CAM_SCALE)
CLIP_NEAR, CLIP_FAR = 0.01, 100.0  # ManiSkill's panda_wristcam hand_camera: near 0.01, far 100 (decision 6)


def intrinsics(width: int = CAM_W, height: int = CAM_H, hfov_deg: float = CAM_HFOV_DEG) -> np.ndarray:
    """Pinhole K of the full stream: square pixels, fx = fy = (W/2) / tan(HFOV/2), principal point at the image
    centre (W/2, H/2), as SAPIEN builds it from the vertical FOV in wm.sim.env_adapter."""
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def _urdf_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()     # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    T[:3, 3] = xyz
    return T


# Camera mount of ManiSkill's panda_wristcam (mani_skill/assets/robots/panda/panda_v3.urdf, unchanged numbers):
#   panda_hand -realsense_joint-> camera_base_link: xyz (0.035, 0, 0.036), rpy (0, -1.5707, 3.1415)
#   camera_base_link -camera_link_joint-> camera_link: xyz (0, 0.02, 0.0115), rpy 0
# and the ManiSkill camera sits at camera_link with identity pose (SAPIEN camera axes: x forward, y left, z up,
# which is Isaac Lab's "world" camera convention).
T_HAND_CAMLINK = _urdf_T([0.035, 0.0, 0.036], [0.0, -1.5707, 3.1415]) @ _urdf_T([0.0, 0.02, 0.0115], [0.0, 0.0, 0.0])
# OpenCV axes (x right, y down, z forward) expressed in the SAPIEN camera axes
_R_SAPIEN_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_HAND_CAM_CV = T_HAND_CAMLINK.copy()
T_HAND_CAM_CV[:3, :3] = T_HAND_CAMLINK[:3, :3] @ _R_SAPIEN_CV
# goal / charger convention of the ManiSkill task: x axis = insertion direction. Socket and plug frames have the
# insertion direction along -z, so both get the same right factor R_G (x -> -z).
R_G = np.eye(4)
R_G[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
T_HAND_TCP = np.eye(4)
T_HAND_TCP[2, 3] = 0.1034          # panda_hand_tcp_joint of panda_v3.urdf

SEG_CLASSES = ("other", "plug", "robot", "board", "table")   # as wm.env.robot ("board" = the socket)

# ------------------------------------------------------------------ weld (scene.AutomateScene)
WELD_MODE = os.environ.get("WM_AUTOMATE_WELD", "articulation")   # "articulation" | "maximal"
WELD_INITIAL = np.eye(4)
WELD_INITIAL[2, 3] = 0.15          # placeholder frame before the first reset (plug 15 cm in front of the hand)

# ------------------------------------------------------------------ placement
# Socket mount = rotation of AutoMate's upright socket about the world y axis; the insertion direction is
#     a = R_y(phi) (0, 0, -1) = (cos theta, 0, -sin theta),  phi = -(90 deg - theta),  theta = MOUNT_TILT_DEG[mount]
# (theta = angle of a below the horizontal, a points away from the robot base).
#   "table"   theta = 90 (default, the agreed scope): AutoMate's upright socket at AutoMate's default position,
#             randomised as in AutoMate (socket_pose below), top-down insertion.
#   "tilted"  theta = 60 (option): the socket face looks up and towards the robot (axis 30 deg from vertical).
#   "wall"    theta = 0 (option): horizontal axis, as the ManiSkill board.
# Known consequence of "table": wm.collect.explore.frame() and wm.sim.geometry.port_axes() build their frame from
# gravity-up made orthogonal to the insertion axis, which is undefined (division by zero) for a vertical axis; the
# options exist for that reason (decision for the user, see the milestone-1 report).
# Limits of "wall": AutoMate's grasps hold the plug along its axis, so the hand must point along a. Pointing the Franka
# hand horizontally needs joint 6 near its upper limit; with the planar arm (j1 = j3 = j5 = 0) it is reachable only
# with the TCP at z >= 0.63 m, and grasp pitch offsets that tilt the hand downwards (d_pitch < 0) are not reachable
# there at all; the other IK solutions are twisted (j1 ~ +-1.5, j5 ~ +-2.7). Measured with kinematics.py, 2026-11-21.
# The same holds at theta = 45 deg for low sockets (j6 ~ 3.6 of 3.75 at the centre grasp). Search 2026-11-21: planar IK
# (|j1|, |j3|, |j5| < 0.3) for assemblies 00015 and 00318, grasp offsets d_pitch in {0, +-20 deg}, d_pos in {0,
# +-10 mm}, hand at H* and 5 cm + L before it, socket origin x in 0.60-0.75, z in 0.15-0.45: theta = 60 deg at
# (0.70, 0, 0.35) has every case solved with >= 0.59 rad to the nearest joint limit; theta = 45 deg needs z >= 0.35
# and gives 0.08-0.47 rad.
# Gravity acts on nothing that moves (robot links and the plug have gravity disabled as in AutoMate; the socket root
# is fixed), so the mount changes no physics, only the robot configuration and the camera background.
SOCKET_MOUNT = os.environ.get("WM_AUTOMATE_MOUNT", "table")
MOUNT_TILT_DEG = {"table": 90.0, "tilted": 60.0, "wall": 0.0}
SOCKET_POS = {"table": (0.6, 0.0, 0.05 + 0.1435),   # AutoMate default_root_state pos + fixed_asset_z_offset
              "tilted": (0.7, 0.0, 0.35),          # IK margin search (below)
              "wall": (0.7, 0.0, 0.7)}
SOCKET_QUAT = {m: tuple(np.roll(Rotation.from_euler("y", -(90.0 - t), degrees=True).as_quat(), 1))   # (w, x, y, z)
               for m, t in MOUNT_TILT_DEG.items()}
# Per-episode socket pose: AutoMate's own randomisation, restored 2026-11-22 (milestone 1 had removed it). As
# isaaclab_tasks direct/automate assembly_env.py randomize_fixed_initial_state with the Insertion task's values
# (assembly_tasks_cfg.py: fixed_asset_init_pos_noise 0.05 m per axis, fixed_asset_init_orn_deg 0, range 10 deg):
#     p = SOCKET_POS[mount] + (2 u_1:3 - 1) * 0.05 m,   yaw = 0 + 10 deg * u_6,   roll = pitch = 0
# (SOCKET_POS already holds AutoMate's default position + fixed_asset_z_offset; u_1..u_6 uniform in [0, 1), u_4, u_5
# drawn and unused as in AutoMate's torch.rand((n, 3)) of which only the yaw entry is kept). AutoMate replaces the
# orientation by the yaw; for the table mount (SOCKET_QUAT identity) that is the same as yaw o mount, which is what
# socket_pose applies for every mount. The random numbers come from the episode seed through their own stream:
#     numpy default_rng(SeedSequence(entropy=seed, spawn_key=(SOCKET_STREAM,)))
# The socket pose is simulator truth: it is recorded in the episode / meta (eval_socket_pose), never in model inputs.
SOCKET_POS_NOISE_M = (0.05, 0.05, 0.05)
SOCKET_YAW_DEG, SOCKET_YAW_RANGE_DEG = 0.0, 10.0
SOCKET_STREAM = 101
# How often the socket pose is drawn (config switch, environment variable WM_AUTOMATE_SOCKET_DRAW):
#   "per_reset"  (default; AutoMate's original) a new draw at every reset: reset number k of an episode seed (k = 0 the
#                reference insertion, k = trajectory index for the trajectories of a grasp) uses
#                    SeedSequence(entropy=seed, spawn_key=(SOCKET_STREAM,))       for k = 0
#                    SeedSequence(entropy=seed, spawn_key=(SOCKET_STREAM, k))    for k >= 1
#                so every trajectory has its own socket pose and needs its own reference insertion for the FK H* label
#                (expert.Expert.reference_label).
#   "per_grasp"  one draw per grasp (k ignored, always the k = 0 draw): the reference insertion, its trajectories and
#                every replay of that grasp share one pose (the behaviour of 2026-11-22).
SOCKET_DRAWS = ("per_reset", "per_grasp")
SOCKET_DRAW = os.environ.get("WM_AUTOMATE_SOCKET_DRAW", "per_reset")
assert SOCKET_DRAW in SOCKET_DRAWS, SOCKET_DRAW


def socket_seed_sequence(seed: int, index: int = 0, draw: str | None = None) -> np.random.SeedSequence:
    """The random stream of reset number `index` of episode seed `seed` (SOCKET_DRAW rule above)."""
    draw = draw or SOCKET_DRAW
    assert draw in SOCKET_DRAWS, draw
    k = 0 if draw == "per_grasp" else int(index)
    assert k >= 0, index
    key = (SOCKET_STREAM,) if k == 0 else (SOCKET_STREAM, k)
    return np.random.SeedSequence(entropy=int(seed), spawn_key=key)


def socket_pose(seed: int, mount: str, index: int = 0, draw: str | None = None) -> tuple[np.ndarray, dict]:
    """Socket pose (4x4, env frame) of reset number `index` of the episode seed `seed`, and the drawn values."""
    draw = draw or SOCKET_DRAW
    rng = np.random.default_rng(socket_seed_sequence(seed, index, draw))
    u_pos, u_orn = rng.random(3), rng.random(3)
    dp = (2.0 * u_pos - 1.0) * np.asarray(SOCKET_POS_NOISE_M)
    yaw = SOCKET_YAW_DEG + SOCKET_YAW_RANGE_DEG * float(u_orn[2])
    q = SOCKET_QUAT[mount]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ \
        Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = np.asarray(SOCKET_POS[mount]) + dp
    return T, {"seed": int(seed), "index": int(index), "draw": draw, "offset_m": dp.tolist(), "yaw_deg": yaw}


# ------------------------------------------------------------------ GPUs (app.py)
# Kit's render GPU (--/renderer/activeGpu) = the row of Kit's own GPU table (Vulkan enumeration, PCI order: row 0 = bus
# 16 = physical GPU 0, row 1 = bus ac = physical GPU 1 on rs2), NOT a CUDA-visible index. Measured on rs2 with
# CUDA_VISIBLE_DEVICES=1: 2026-11-22 (multi-GPU on, AppLauncher's activeGpu 0) Kit skipped row 0 ("CUDA in bad state")
# and fell back to row 1 ("Active: Yes: 0"); 2026-11-23 (multi-GPU off, activeGpu 0) "activeGpu 0 is not compatible
# with current foundation settings", "No device could be created": no renderer at all. Default 1 = physical GPU 1.
KIT_GPU = int(os.environ.get("WM_KIT_GPU", "1"))
# The GPU our processes must not compute on (physical index, nvidia-smi PCI order) and the graphics context accepted
# there (the user's decision 2026-11-22: type G only, at most 50 MiB per process).
FORBIDDEN_GPU = int(os.environ.get("WM_FORBIDDEN_GPU", "0"))
FORBIDDEN_GPU_GRAPHICS_MAX_MIB = 50
DIAGNOSTIC = os.environ.get("WM_AUTOMATE_DIAGNOSTIC", "0") == "1"
if not DIAGNOSTIC and (SOCKET_MOUNT, CAM_SCALE, FORBIDDEN_GPU) != ("table", 1, 0):
    raise RuntimeError(f"WM_AUTOMATE_MOUNT={SOCKET_MOUNT}, WM_CAM_SCALE={CAM_SCALE}, WM_FORBIDDEN_GPU={FORBIDDEN_GPU}: "
                       "only table / 1 / 0 outside diagnostics (set WM_AUTOMATE_DIAGNOSTIC=1 for a diagnostic run)")

# ------------------------------------------------------------------ switches (decided by the user 2026-11-22)
# The success criterion has no switch: AutoMate's check_plug_inserted_in_socket (checks.py), approved 2026-11-22.
# Axis of the grasp spin dpsi (grasp.py):
#   "plug_axis"   (default; approved by the user 2026-11-22) about the plug's own axis, the line through the tight-fit
#                 centroid (x_c, y_c) (spec text); "cad_origin" (option) about plug-frame z through the CAD origin
GRASP_SPIN_AXES = ("plug_axis", "cad_origin")
GRASP_SPIN_AXIS = os.environ.get("WM_AUTOMATE_SPIN_AXIS", "plug_axis")
assert GRASP_SPIN_AXIS in GRASP_SPIN_AXES, GRASP_SPIN_AXIS
# Socket draw: SOCKET_DRAW above, default "per_reset" = AutoMate's original (confirmed by the user 2026-11-22).
# PhysX device: "cpu" for collection (approved by the user 2026-11-22; AutoMate's own config runs PhysX on the GPU,
# so this differs from AutoMate); "cuda:0" (= GPU 1 under CUDA_VISIBLE_DEVICES=1) stays an option. The camera renders
# on the GPU either way.
PHYSICS_DEVICE = os.environ.get("WM_AUTOMATE_DEVICE", "cpu")
# CPU threads of PhysX (and Kit's task system) per process (app.launch): 5, to match the planned CPUQuota of 500 %
# per collection worker; env WM_PHYSX_THREADS overrides.
PHYSX_CPU_THREADS = int(os.environ.get("WM_PHYSX_THREADS", "5"))
# Where the start gap g0 of the start box is measured from (frames.start_pose; r4-expert-01, decided by the user
# 2026-11-23: option B):
#   "socket_top"  (default and APPROVED) from the higher of {socket top, entrance}: b = L + g_top + g0, every start
#                 outside the socket, as in AutoMate's evaluation
#   "entrance"    (diagnostics only, WM_AUTOMATE_DIAGNOSTIC=1) from the start of the tight fit: b = L + g0, the box
#                 until 2026-11-23; refused here at import and by frames.start_back outside diagnostics
START_GAP_FROMS = ("socket_top", "entrance")
START_GAP_FROM = os.environ.get("WM_AUTOMATE_START_GAP_FROM", "socket_top")
assert START_GAP_FROM in START_GAP_FROMS, START_GAP_FROM
if not DIAGNOSTIC and START_GAP_FROM != "socket_top":
    raise RuntimeError(f"WM_AUTOMATE_START_GAP_FROM={START_GAP_FROM}: only socket_top outside diagnostics (the user's "
                       "decision 2026-11-23; set WM_AUTOMATE_DIAGNOSTIC=1 for a diagnostic run)")


SPAWN_GAP_M = 0.012                # reset: plug aligned on the axis, its tip 12 mm outside the socket face
                                   # (ManiSkill expert's transport_tip_gap_m, wm.env.expert.SAFE_GAP_M)
SETTLE_STEPS = 5                   # control steps (0.25 s) with the fingers closing after a reset (AutoMate: 0.25 s)
Q_NOMINAL = np.array([0.0, 0.0, 0.0, -1.870, 0.0, 1.8675, 0.785398])   # AutoMate ctrl.reset_joints
# IK seeds for the reset pose: AutoMate's reset joints and planar arm configurations (j1 = j3 = j5 = 0, j7 = pi/4);
# the converged solution with the largest distance to the joint limits is used.
RESET_SEEDS = [Q_NOMINAL] + [np.array([0.0, j2, 0.0, j4, 0.0, j6, 0.785398])
                             for j2 in (0.0, 0.4) for j4 in (-2.3, -1.7) for j6 in (1.6, 2.2, 2.8, 3.3)]

# ------------------------------------------------------------------ grasp
# Grasp variables (shape = assembly id, clamp angle, offset along and spin about the plug axis) and their ranges:
# wm.sim_automate.grasp. The ManiSkill pitch / grasp-position ranges of wm.sim.config do not apply here.
# port -> assembly id. A port string that is itself an assembly id ("00015") is used directly.
PORT_TO_ASSEMBLY: dict[str, str] = {}
DEFAULT_ASSEMBLY = "00015"         # AutoMate's default Insertion task


def assembly_of(port: str) -> str:
    return PORT_TO_ASSEMBLY.get(port, port)


def switches() -> dict:
    """The values of the switches above (record them with every result: they change what is measured / drawn)."""
    return {"socket_draw": SOCKET_DRAW, "grasp_spin_axis": GRASP_SPIN_AXIS, "physics_device_default": PHYSICS_DEVICE,
            "physx_cpu_threads": PHYSX_CPU_THREADS, "socket_mount": SOCKET_MOUNT, "weld_mode": WELD_MODE,
            "cam_scale": CAM_SCALE, "kit_gpu": KIT_GPU, "forbidden_gpu": FORBIDDEN_GPU,
            "start_gap_from": START_GAP_FROM,
            "success_rule": "automate check_plug_inserted_in_socket",
            "success_steps": "after each control step (never the start state)"}


# The switch values a collection or a milestone-1b batch may run with (decisions of 2026-11-22: socket draw per reset,
# spin about the plug axis, camera unchanged, GPU 0 forbidden; KIT_GPU 1 measured 2026-11-23, see above; weld,
# PhysX threads = the CPUQuota plan of the caps; start_gap_from "socket_top" = option B of r4-expert-01, decided by
# the user 2026-11-23). The PhysX device is not pinned: CPU is the collection choice and GPU an approved option, used by the
# CPU-vs-GPU check (r4-decisions-04); the gate compares the device of the batch with the collection's.
APPROVED = {"socket_draw": "per_reset", "grasp_spin_axis": "plug_axis", "weld_mode": "articulation",
            "physx_cpu_threads": 5, "kit_gpu": 1, "socket_mount": "table", "cam_scale": 1, "forbidden_gpu": 0,
            "start_gap_from": "socket_top"}


def unapproved_switches(values: dict | None = None) -> dict:
    """{key: (value, approved value)} for every switch of APPROVED whose value differs (values = switches() when None;
    a key missing from values counts as differing). Empty = every pinned switch has its approved value."""
    v = switches() if values is None else values
    return {k: (v.get(k), a) for k, a in APPROVED.items() if v.get(k) != a}


def refuse_unapproved(where: str, values: dict | None = None):
    """Raise RuntimeError naming each unapproved switch (unapproved_switches), unless WM_AUTOMATE_DIAGNOSTIC=1."""
    bad = unapproved_switches(values)
    if bad and not DIAGNOSTIC:
        raise RuntimeError(f"{where}: switch values not approved: "
                           + ", ".join(f"{k}={v!r} (approved {a!r})" for k, (v, a) in sorted(bad.items()))
                           + "; unset the variables or set WM_AUTOMATE_DIAGNOSTIC=1 for a diagnostic run")

