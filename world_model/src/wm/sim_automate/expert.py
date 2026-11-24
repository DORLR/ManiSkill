"""Scripted demonstration expert for the AutoMate scene: the role of wm.env.expert (ManiSkill) on wm.sim_automate.

The expert stands in for a teleoperator, so it may use simulator truth (the socket pose, the plug pose in the hand,
the plug and socket meshes). Recorded model inputs never contain that truth (wm.data.schema).

Chongyu's phases (wminsertion_v2/{scripted, near_wall, stochastic, continuous, randomized}.py) are ported, not
imported: they need SAPIEN and MPlib, which the Isaac Lab environment does not have. Same structure, guards and
numbers unless listed under "Changed" below.

Frames (all simulator truth)
    G      goal frame = T_socket * R_G (x = a, the insertion direction; y, z lateral; robot.R_G)
    CH     plug pose in the same convention, T_plug * R_G;  rel = G^-1 CH  (identity = seated)
    h      TCP^-1 CH, frozen at the start of every trajectory (FK TCP, true plug pose), so TCP(rel) = G rel h^-1
    pivot  c = centre of the plug's lowest 1 mm (plug frame, charger axes). Poses of the plug are written as
           (t, R): rotation R about the pivot, then the pivot moved by t from its seated place:
           rel = [R, c + t - R c]. t_x < 0 is outside, (t_y, t_z) the lateral offset of the plug tip.
    gap    g(rel) = x_ent - max_v (rel v)_x over the plug's convex-hull vertices v: the plug's deepest point
           relative to the entrance (the start of the tight fit, assets.AssemblySpec.entrance_z_m); g = -L seated.
    top    g_top = max(L_top, L) - L >= 0: how far the higher of {socket top, entrance} is above the entrance.
           "Outside" poses of the expert (stand-off, contact offset, random waypoints) are given by their top gap
           g - g_top, so that a lateral offset never puts the plug into a lead-in chamfer above the entrance.

Phases of one recorded trajectory (labels of wm.data.schema.PHASES)
    approach   continuous_approach (recovery_align after a perturbation): the commanded pose moves from where the
               hand is to the aligned stand-off S (top gap NEAR_GAP_M = 3 mm, as Chongyu's 3 mm target gap) on a
               straight line (pivot) / geodesic (rotation) with a trapezoidal speed profile: FAST (APPROACH_*).
    align      same label: the command is held at S (exact pose from truth) with Chongyu's tracking compensation
               (filtered last-sent minus measured joints, gain 0.75, bounded 3 mrad, learned only while quiet and
               outside) until the TRUE plug is within 0.3 mm / 0.2 deg of S, inside the gap corridor and slow, for
               5 consecutive steps (timeout 100 steps).
    insert     axial_push: the pivot command moves along a at PUSH_SPEED_M_S = 3 mm/s (SLOW), never leading the
               true plug by more than 0.3 mm; true lateral error <= 0.3 mm and tilt <= 0.2 deg are checked before
               and after every step (Chongyu's actual insertion guards); then final_verify: the seated command is
               held until depth_left < 0.5 mm, lateral <= 0.3 mm, tilt <= 0.2 deg and the hand is still, 5 steps.
    perturbed  perturbation_stochastic: Chongyu's 56-step window from the start towards a 6 mm top gap, the
               command moving at Chongyu's close_speed (CLOSE_SPEED_M_S 3 mm/s, CLOSE_ANG_SPEED_DEG_S 2 deg/s;
               CY continuous.py:469, :321-335), with a drifting target (renewed every 5 steps, smoothing 0.25,
               amplitudes 0.8 mm along a, 1.5 mm lateral, 1.5 deg) and AR(1) joint noise after IK (sigma 1.3 mrad,
               bound 4 mrad, correlation 0.85; a noisy command whose plug comes within 2 mm of the socket is
               resampled, 12 attempts); then recovery_align,
               and the insertion with Chongyu's bounded insertion target noise (15 um along a, 40 um lateral,
               0.03 deg, AR 0.85, tapered to 10 % over the last 1 mm).
    contact    approach + align to S, contact_offset: move to the offset pose (top gap 3 mm, lateral 0-3 mm,
               0-3 deg; wm.env.sampling.sample_contact_offset) by wm_v1's _move_outside (Execution._move_outside:
               2.7 mm/s, 2 deg/s; restored 2026-11-23, deviation row 38), contact_push from the offset pose:
               straight push along a at 3 mm/s without the insertion guards, stopped when the command leads the true
               plug by 1.5 mm (jam), then 10 steps holding the last command. Outcome (the expert's procedure, not the success criterion):
               reached_depth (wm_v1's contact rule: depth_left at the plug CAD origin <= 1 mm; wm_v1 called it
               "inserted", renamed so that "inserted" means one rule only), jammed, or guard:<reason>.
    random     random_motion_i: moves through 3 waypoints in front of the socket (top gap 4-18 mm, lateral +-10 mm,
               +-6 deg) by wm_v1's _move_outside (Execution._move_outside, V1 env/expert.py:89-99): constant steps of
               Chongyu's _advance (CY continuous.py:321-335) at RANDOM_SPEED_M_S = 0.9 x close_speed = 2.7 mm/s and
               CLOSE_ANG_SPEED_DEG_S = 2 deg/s, no ramp, no hold at a waypoint (the next waypoint starts as soon as
               the command reaches it); the outside guards stay active.
Guards on every step: IK refused (no solution or a branch jump > 0.25 rad, robot._ik), a joint target moving more
than 0.025 rad in one step, Chongyu's two contact guards read from PhysX's contact report
(scene.AutomateScene.socket_contacts, per pair of collision shapes as SAPIEN gives them):
    socket_contact_force_guard   F > 80 N,  F = sum over every shape pair touching the socket (plug, hand, fingers,
                                 any robot link) of |sum of the pair's point impulses| / dt   (checks.pair_force_sum)
    socket_penetration_guard     max over those contact points of max(0, -separation) > 0.5 mm
before insertion (outside phases) a NOMINAL command's plug must stay >= 1 mm from the socket mesh (hpp-fcl distance of
the commanded pose only; noisy candidates 2 mm; no check at the measured configuration) and the TRUE plug must not touch
it (plug-socket F <= 1 N). Chongyu's 2 mm FK-sample gap on every outside command (CY continuous.py:303-315, :347, :385)
and his actual tip gap >= 1 mm are not ported (rows 24 / 32 below). Information: peak min_actual_top_gap_mm_outside =
min over the outside steps of the true plug's top gap (m["top_gap"], deepest plug point above the higher of socket top /
entrance) in mm. Hand / finger contact with the socket is recorded (peak hand_finger_socket_N, information only) and
limited through the 80 N sum, as in Chongyu's expert. Since the user's decision 7 of 2026-11-23 (deviation row 39) the
hand and both fingers of every NOMINAL outside command must also stay >= HAND_FINGER_COMMAND_CLEARANCE_M = 1 mm from the
socket mesh (hpp-fcl, grasp.FingerSocketCheck meshes, the commanded TCP pose tcp(rel), fingers at the closing opening
robot.q_finger, as start_clearance; checked in _send next to the plug's COMMAND_CLEARANCE_M; noisy and compensated
candidates keep their plug-only checks); a violation ends the trajectory as expert_failed with failure
"<phase>_nominal_command_guard: hand_finger_command_clearance". It applies wherever the expert is outside: approach,
align, perturbation window, recovery, contact offset, random motion, and the unrecorded reference (t00 / per-reset
label). Random waypoints that violate it at the waypoint itself are redrawn by collect.py (Expert.waypoint_clearance;
starts by start_clearance, which already holds the hand and fingers to 1 mm). Information: peak
min_command_hand_finger_clearance_mm = the smallest such distance over the trajectory's nominal outside commands (-1000
= intersecting).
Success (approved by the user 2026-11-22): AutoMate's check_plug_inserted_in_socket (checks.py), evaluated after
every control step, never on the start state (as AutoMate's _get_rewards); result "success" = after any step
(AutoMate's ep_succeeded), "success_end" = after the last step (False if no step was taken), "first_success_step" =
the first step with success (None if none). The expert's own convergence tests below (Chongyu's final_verify, the
contact rule) are its procedure, not the criterion; "completed" = the procedure finished.
Orientation checks (insertion guard, final_verify): the full rotation angle of the true plug relative to its seated
pose, 2 arccos|w| <= 0.2 deg, Chongyu's actual_goal_attitude_error_deg (the expert always aims at the one seated
pose its weld defines, so no symmetry enters). The plug's axis tilt and spin are recorded per stage (information).
Step budget per recorded trajectory: step_budget(L, g_top) = 800 + 8 * ceil(max(0, L + g_top - 20 mm) / 0.5 mm)
control steps. 800 is Chongyu's budget for the 20 mm charger pins; every further 0.5 mm of push travel (tight fit L
plus the lead-in g_top above it) adds 8 control steps = one 0.4 s decision, the rate of the episode limit N_max = 150
+ ceil(max(0, L - 20 mm) / 0.5 mm). At 3 mm/s, 0.5 mm takes 3.3 control steps, so 8 steps leave a factor 2.4 for the
0.3 mm lag throttle and settling. g_top is added (N_max has L only) because the push starts above max(socket top,
entrance).
The same budget caps random motion (rows 15 and 27 are decided; no rule changes here). At wm_v1's 2.7 mm/s a few
random-motion trajectories run into it: _advance halves a step whenever the TCP chord of a curved path is longer
than 0.135 mm (as Chongyu's _advance does, so wm_v1 behaved the same under its 800-step cap, CY scripted.py:221-222),
and legs limited by the TCP distance then take up to about twice the ideal number of commands. Measured in the
round-6 expert review (numpy re-run of the real _advance / _move_outside on ranges_v5 grasps and socket_top starts):
about 0.2 % of random-motion trajectories overall (5 of 2,730), up to about 4 % on 00004 and 00320 (150 draws each).
Such a trajectory stops at the budget before its third waypoint with failure_reason "ExpertFailure: step_budget_<N>"
(or "ExpertFailure: random_motion_<i>_did_not_reach_target" after 600 commands of one waypoint) and is recorded,
truncated, as expert_failed. The contact offset needs at most about 90 commands and is not at risk.

Unrecorded transfers (Expert.move_to_start; ManiSkill used MPlib, here only IK and joint interpolation):
    1. retreat: straight out along -a (lateral offset and attitude kept) until the plug's lowest point is
       TRANSFER_GAP_M = 22 mm above the higher of {socket top, entrance} (above every start of the 6-18 mm box), one
       IK target per control step at <= RETREAT_SPEED_M_S = 10 mm/s; stopped (placement failure) when the command
       leads the FK hand by > 1.5 mm (stuck plug), the socket force F exceeds 80 N or the penetration 0.5 mm;
    2. at that height, translate / rotate to the start's lateral offset and attitude (robot.move_to: IK waypoints of
       <= 5 mm / 2.5 deg, 2 control steps each);
    3. straight down along a to the start, guarded as leg 1 (one IK target per control step at <= 10 mm/s; lag
       1.5 mm, 80 N, 0.5 mm penetration) and, since it ends at the start above the socket, also by the outside rule
       of the recorded phases: the true plug must not touch the socket (plug-socket F <= OUTSIDE_CONTACT_N = 1 N).
       The descent ends at the start: config.START_GAP_FROM "socket_top" (the user's decision 2026-11-23, option B
       of r4-expert-01) puts it 6-18 mm above the higher of {socket top, entrance}, outside the socket, as every
       start of AutoMate's evaluation ("entrance", 6-18 mm above the entrance, is for diagnostics only). A brush is
       caught where it happens instead of only through the start error.
    Every leg ends by settling until the hand is within 0.1 mm / 0.05 deg of its target and still (at most 40
    steps); a leg that does not settle fails the placement. Each leg records its peak plug-socket and hand / finger -
    socket forces and penetration; a failure records the leg, the reason and the largest command-FK lag. Before a
    start is redrawn after a failed transfer, the robot is re-homed by robot.reset with the same grasp and seed
    (collect.py), so the next attempt starts from the same state as the first. The start is
    accepted only if the plug, hand and both fingers are >= START_CLEARANCE_M from the socket mesh at the start and
    the plug at every 2 mm of the descent (hpp-fcl, meshes of grasp.FingerSocketCheck); otherwise it is redrawn
    (placement attempt). Leg 1 reverses the motion that brought the plug where it is; legs 2-3 stay above the
    socket top except for the checked descent.

Per-reset socket (config.SOCKET_DRAW "per_reset", AutoMate's original): every reset draws a new socket pose, so the
FK H* label of a trajectory comes from its own reference insertion after that reset: Expert.reference_label() runs
the reference (unrecorded) and returns H* (FK hand at its end), the axis (FK hand during the axial push), the model
H* = T_socket T_hp^-1 and their difference. Cost (expert smoke 2026-11-22, 00015 / 00083 / 01102, 22 grasps): a
reference insertion takes 139-347 control steps; one per trajectory (+ the 5 settling steps of its reset) adds
12 x (ref + 5) = 1,728-4,224 control steps per grasp, +67 to +94 % of a grasp's control steps (recorded + transfer
2,310-4,851; median +78 %). The extra steps are unrecorded (no rendering), so the wall-time share is smaller.

Differences from Chongyu's ManiSkill expert (deviation table audit_am_r4/deviations_table.md, rows 23-36, 38 and 39; the
user's answer of 2026-11-23: approved, "keep ours", unless marked restored):
  * (row 23, approved 2026-11-23) speeds: approach and alignment FAST (15 mm/s, 50 mm/s^2, 15 deg/s, 60 deg/s^2)
    instead of Chongyu's 2.7 mm/s and 2 deg/s; insertion and contact push unchanged at 3 mm/s. Added after the
    user's answer (not on the page they approved; to be shown to the user with the smoke results): Chongyu's nominal
    target spacing guard (<= 0.5 mm and 0.25 deg between consecutive commands, CY continuous.py:340-343) is not
    ported (the fast approach would trip it);
  * (row 24, approved 2026-11-23) the outside guards use the plug and socket meshes (hpp-fcl) instead of the charger's
    pin-corner / slot model, and the actual-pose checks use the plug's tip centre (pivot). Added after the answer (shown
    with the smoke): nominal commands >= 1 mm mesh distance of the commanded pose only (COMMAND_CLEARANCE_M), noisy 2
    mm, no check at the measured configuration; Chongyu: 2 mm at 5 FK samples from the measured joints (the first: the
    FK-modelled plug now) to every outside command, nominal and noisy. The true plug is guarded by contact (F <=
    OUTSIDE_CONTACT_N = 1 N after every outside step); Chongyu's actual tip gap >= 1 mm before and after each step (CY
    near_wall.py:45, :183-184; stochastic.py:125-127) is not ported;
  * (row 25, approved 2026-11-23) MPlib's collision checks of the arm (self / environment) are not available; the arm
    stays above the socket;
  * (row 26, approved 2026-11-23) the axial push starts 3 mm above the higher of socket top and entrance (lead-in
    chamfers are pushed through at 3 mm/s, as the tight fit);
  * (row 27, RESTORED 2026-11-23) random waypoints move by wm_v1's _move_outside (Execution._move_outside: constant
    2.7 mm/s and 2 deg/s steps, no ramp, no hold); until 2026-11-23 the speed was drawn per waypoint in 3-15 mm/s, on
    a trapezoidal profile with 4 holding steps per waypoint. A "speeds" argument other than that constant is refused
    (Execution.execute). Still different from wm_v1: our _advance (row 30 (a), (b)). Every command also goes
    through the send path that all outside phases share, with the differences of rows 23 (no nominal spacing guard;
    it cannot trip at 0.135 mm steps), 24 / 32 (1 mm mesh distance of the commanded pose and the 1 N contact rule
    instead of the 2 mm FK-sample gap and the actual gap >= 1 mm; pending), 25, 28 and 29. Tracking compensation
    (row 31) is inert here, as in wm_v1: random motion runs no approach, so the bias stays zero;
  * (row 28, approved 2026-11-23) max_action_delta 0.025 rad per step: Chongyu's validation upper bound
    (stochastic.py:80); his default is 0.02;
  * (row 29, approved 2026-11-23) IK (robot._ik, also in the recorded phases): branch-jump limit 0.25 rad against the
    seed, tolerance relaxed in steps up to 1e-3 (about 1 mm / 1 mrad) with random restarts (Chongyu: ik_threshold
    1e-5, one seed, ik_max_joint_change_rad 0.15 against the measured joints; ours is wm_v1's sim/env_adapter rule,
    there used only for unrecorded transfers);
  * (row 30, RESTORED 2026-11-23) the perturbation window moves at Chongyu's close_speed, 3 mm/s and 2 deg/s
    (CY continuous.py:469); until 2026-11-23 it moved at the FAST speeds (15 mm/s, 15 deg/s). Our _advance (also used
    by _move_outside, rows 27 / 38) differs from Chongyu's in two points, never faster: the step fraction comes from
    max(pivot, TCP) distance instead of the TCP distance only, and the pivot is interpolated linearly instead of
    re-projected through _outside_pose with the interpolated gap (his 0.25 deg step acceptance against our 0.1 deg
    has no effect: the fraction caps rotation at 0.1 deg per step in both);
  * (row 31, approved 2026-11-23) tracking compensation is not applied in the perturbation window, the axial push,
    final_verify and the contact push (Chongyu keeps the frozen bias applied, with an axial-depth-preserving re-IK
    and a spacing check);
  * (row 32, approved 2026-11-23) nominal commands >= 1 mm on the hpp-fcl mesh distance of the commanded pose, noisy 2
    mm, no FK path samples, no check at the measured configuration (Chongyu: 2 mm tip gap to the wall plane on 5 FK
    samples from the measured joints to every outside command, nominal and noisy; added after the answer);
  * (row 33, approved 2026-11-23) alignment admission corridor [2.7, 6] mm of top gap (Chongyu: [3, 6] mm);
  * (rows 15 / 34, approved 2026-11-23) step budget scaled with the push travel (step_budget above), instead of a
    fixed 800;
  * (row 35, approved 2026-11-23) the retreat and descent legs of the unrecorded transfer are guarded (lag 1.5 mm,
    80 N, 0.5 mm penetration; the descent also by the 1 N outside contact rule; round 3, 2026-11-23) and every leg
    settles (ManiSkill: MPlib plan). Added after the user's answer (not on the page they approved; to be shown to
    the user with the smoke results, together with 00470's clearance-redraw share): the start-clearance redraw. A
    start is redrawn when the plug, the hand or a finger is closer than START_CLEARANCE_M = 1 mm to the socket mesh
    at the start, or the plug at any 2 mm sample of the descent (Expert.start_clearance; placement-failure stage
    "clearance"; at most collect.PLACEMENT_ATTEMPTS starts, then placement_failed). It filters the start
    distribution (wm_v1 has no such filter). Plug part (start and descent, mesh distance): on 00470 6.95 % of its
    starts have the plug's lowest hull point within 1 mm of the socket-top plane (round 5), and the rule redraws
    about 3 % (round-8 audit: 29 of 1,000 and 18 of 400 draws, pooled 3.4 %). Hand / finger part (round-12 audit,
    hpp-fcl, ranges_v5, socket_top): 00422 14 %, 00581 13 %, 00186 7 % of start draws, about 0 elsewhere; none of
    them is in the collector smoke. The recorded outside phases had no hand / finger clearance check until decision 7
    (row 39); before it, a commanded finger entered the socket on 22 / 18 / 5 % of random trajectories (00422 / 00581
    / 00186), 8 / 2 / 0 % of approaches, 2.5 % of 00422 contact offsets, and in rare single cases on 00615 / 00726 /
    00741, ending at the 80 N / 0.5 mm guards (expert_failed, truncated) or scraping the socket below 80 N. Shown with
    the smoke as information: on 00470 about 0.8 % of
    accepted starts (8 of 971) end the recorded approach, and about 0.3 % (1 of 382) the perturbation window, as
    expert_failed by the 1 mm nominal command guard (no contact);
  * (row 36, approved 2026-11-23) per-reset socket draws with one unrecorded reference insertion per trajectory
    (above; AutoMate's randomisation is the original, the extra reference is its cost);
  * (row 39, approved 2026-11-23, the user's decision 7) the hand and both fingers of every nominal outside command
    keep >= 1 mm hpp-fcl mesh distance from the socket (HAND_FINGER_COMMAND_CLEARANCE_M, next to the plug's
    COMMAND_CLEARANCE_M; Execution._send); a violating approach / align / perturbation / recovery / contact-offset /
    random-motion command fails the trajectory (expert_failed, "hand_finger_command_clearance"); random waypoints
    that violate it are redrawn (collect.py, Expert.waypoint_clearance). Chongyu's expert has no hand / finger
    clearance rule (MPlib checks the arm, row 25);
  * (row 38, RESTORED 2026-11-23) the contact-offset move uses wm_v1's _move_outside (2.7 mm/s, 2 deg/s, V1
    env/expert.py:95, :148), the same helper as random motion; until 2026-11-23 it ran at the FAST approach speeds
    (15 mm/s, 15 deg/s). Still different from wm_v1: our _advance (row 30 (a), (b)). Every command also goes through
    the send path that all outside phases share, with the differences of rows 23 (no nominal spacing guard; it
    cannot trip at 0.135 mm steps), 24 / 32 (1 mm mesh distance of the commanded pose and the 1 N contact rule
    instead of the 2 mm FK-sample gap and the actual gap >= 1 mm; pending), 25, 28, 29 and 31 (the bias learned in
    approach_align is added to the contact-offset commands without Chongyu's depth-preserving re-IK and spacing
    check).
  Restored to Chongyu's rule in round 2 (2026-11-22): the force guard per shape pair from the contact report (the
  contact sensor gave one net force per body pair, in which opposite walls cancel); the 0.5 mm penetration guard
  (the contact report gives the separations); the full rotation angle in the orientation checks (round 1 had used
  axis tilt and spin modulo symmetry); no separate 1 N hand / finger guard (round 1 had added one, also on the
  retreat, and made it a condition of the outcome "inserted"; the peak is now information only).
  Caveat of the force port: AutoMate's socket collides through one shape (smoke s5, rs2 2026-11-23, 00015: the only
  socket shape in contact is a_00015_socket/collisions), so the plug-socket contact is one shape pair and F for it is
  |net force| (it equalled the contact sensor's net force within 1e-6 N), as SAPIEN would give for a receptacle made of
  one shape: opposite walls cancel inside it. The per-point magnitude sum (an upper bound) is recorded next to it (peak
  plug_socket_points_N). Smoke s6 (same run): the 80 N guard tripped (143 N at the trip, 0.5 mm per step pushed into
  the face), the descent's 1 N outside rule and the 1.5 mm lag stop tripped.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..env.sampling import Offset
from ..sim.geometry import interpolate_T, pose_distance
from . import assets as A
from . import config as C
from . import frames
from . import grasp as G
from .checks import ContactReportError, ResetError, automate_success, tilt_spin, truth  # noqa: F401 (re-exported)
from .robot import R_G

StepCallback = Callable[[str, np.ndarray, dict], object]
DT = C.CONTROL_DT

# ------------------------------------------------------------------ speeds (user decision: approach fast, insert slow)
APPROACH_SPEED_M_S = 0.015
APPROACH_ACCEL_M_S2 = 0.05
APPROACH_ANG_SPEED_DEG_S = 15.0
APPROACH_ANG_ACCEL_DEG_S2 = 60.0
PUSH_SPEED_M_S = 0.003             # Chongyu's insertion_speed_m_s; also the contact push (wm.env.expert)
# Chongyu's close_speed (ContinuousScriptedPolicyConfig close_speed_m_s, close_angular_speed_deg_s; CY
# continuous.py:32, :34): the perturbation window (CY continuous.py:469; restored 2026-11-23, deviation row 30) and,
# at 0.9 x, wm_v1's random-motion speed (V1 env/expert.py:95; restored 2026-11-23, deviation row 27)
CLOSE_SPEED_M_S = 0.003
CLOSE_ANG_SPEED_DEG_S = 2.0
RANDOM_SPEED_M_S = 0.0027            # 0.9 x close_speed, wm_v1's _move_outside (random motion, contact offset)
MOVE_OUTSIDE_MAX_STEPS = 600         # wm_v1's _move_outside max_steps (V1 env/expert.py:89)
MOVE_OUTSIDE_TOL = 1e-3              # wm_v1's stop: TCP translation [mm] + rotation [deg] < 1e-3 (V1 env/expert.py:93)
# ------------------------------------------------------------------ geometry of the phases
NEAR_GAP_M = 0.003                 # Chongyu's target_tip_gap_m (top gap of the stand-off where the push starts)
STOCHASTIC_END_GAP_M = 0.006       # Chongyu's stochastic_end_tip_gap_m
SAFE_GAP_M = C.SPAWN_GAP_M         # 12 mm: the reset spawn (Chongyu's transport_tip_gap_m)
TRANSFER_GAP_M = 0.022             # unrecorded transfers pass this far above the socket top / entrance
# ------------------------------------------------------------------ Chongyu's tolerances and guards
POSITION_TOL_M = 0.0003
ORIENTATION_TOL_DEG = 0.2
LIN_VEL_TOL_M_S = 0.0005
ANG_VEL_TOL_DEG_S = 0.5
JOINT_VEL_TOL_RAD_S = 0.02
SETTLE_CONSECUTIVE = 5
SETTLE_TIMEOUT = 100
MAX_INSERT_LAG_M = 0.0003
MAX_FORCE_N = 80.0
MAX_PENETRATION_MM = 0.5           # Chongyu's max_socket_penetration_mm
OUTSIDE_CONTACT_N = 1.0            # the true plug must not touch the socket before the insertion
COMMAND_CLEARANCE_M = 0.001        # commanded plug - socket mesh distance before the insertion
# the user's decision 7 of 2026-11-23 (deviation row 39): the hand and both fingers of every NOMINAL outside command
# keep this hpp-fcl mesh distance from the socket too (grasp.FingerSocketCheck meshes, the closing opening
# robot.q_finger, as start_clearance); a violation fails the trajectory
# ("<phase>_nominal_command_guard: hand_finger_command_clearance")
HAND_FINGER_COMMAND_CLEARANCE_M = 0.001
NOISY_CLEARANCE_M = 0.002          # Chongyu's minimum_noisy_command_tip_gap_m
MAX_ACTION_DELTA_RAD = 0.025       # Chongyu's validation upper bound (stochastic.py:80); his default is 0.02
JAM_LAG_M = 0.0015                 # wm.env.expert.JAM_LAG_M
INSERTED_MARGIN_M = 0.001          # wm.env.expert.INSERTED_MARGIN_M
MAX_STEPS = 800                    # Chongyu's step budget per execution (20 mm pins); step_budget() scales it
# stochastic window (StochasticScriptedPolicyConfig) and noise (ContinuousScriptedPolicyConfig), unchanged
STOCHASTIC_STEPS, DRIFT_RENEW, DRIFT_SMOOTH = 56, 5, 0.25
DRIFT_AMP = np.array([0.0008, 0.0015, 0.0015, 1.5, 1.5, 1.5])
JOINT_NOISE_SIGMA, JOINT_NOISE_BOUND, JOINT_NOISE_CORR, NOISE_ATTEMPTS = 0.0013, 0.004, 0.85, 12
INS_NOISE_BOUNDS = np.array([0.000015, 0.00004, 0.00004, 0.03, 0.03, 0.03])
INS_NOISE_CORR, INS_NOISE_TAPER_M, INS_NOISE_FINAL = 0.85, 0.001, 0.10
TRACK_GAIN, TRACK_FILTER, TRACK_BOUND = 0.75, 0.15, 0.003
# ------------------------------------------------------------------ transfers
START_CLEARANCE_M = 0.001
TRANSFER_MAX_LIN, TRANSFER_MAX_ANG = 0.005, np.radians(2.5)
RETREAT_SPEED_M_S = 0.010          # leg 1 (out of the socket): <= 10 mm/s, as milestone-1b check (ii)'s push
TRANSFER_LAG_STOP_M = 0.0015       # command - FK hand distance that stops the retreat (wm.env.expert.JAM_LAG_M)
FINGER_CONTACT_INFO_N = 1.0        # meta finger_contact = peak hand / finger - socket force > 1 N (information only)
SETTLE_STEPS_TRANSFER = 40


def step_budget(L: float, g_top: float) -> int:
    """Control steps per recorded trajectory (module docstring): 800 + 8 ceil(max(0, L + g_top - 20 mm) / 0.5 mm)."""
    return MAX_STEPS + 8 * int(np.ceil(max(0.0, L + g_top - 0.020) / 0.0005 - 1e-9))


class ExpertFailure(RuntimeError):
    pass


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


euler_R = frames.euler_R          # static XYZ roll / pitch / yaw [deg] (Chongyu's euler2quat(*radians(rpy)))


class PlugGeometry:
    """Plug hull, entrance, pivot and the hpp-fcl clearance to the socket (simulator truth, per assembly)."""

    def __init__(self, robot):
        import trimesh
        spec = robot.port_spec
        plug = trimesh.load(A.fetch(f"{robot.assembly_id}/plug.obj"), force="mesh")
        V = np.asarray(plug.convex_hull.vertices, np.float64) @ R_G[:3, :3]     # charger axes: x = -z_plug
        self.V = V
        self.c = R_G[:3, :3].T @ np.asarray(spec.tip_centre_m, np.float64)    # assets.tip_centre in charger axes
        self.n = spec.symmetry_n
        self.L = float(spec.pin_length_m)
        self.x_ent = -float(spec.entrance_z_m)
        self.g_top = max(float(spec.depth_top_m), self.L) - self.L
        self.d_dis = float(spec.disassembly_dist_m)                 # AutoMate's success height (checks.py)
        self.fcl = robot._fingers                                   # hpp-fcl meshes: plug, socket, hand, fingers

    # -------------------------------------------------- (t, R) <-> rel
    def rel(self, t, R) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.c + np.asarray(t, np.float64) - R @ self.c
        return T

    def t_of(self, rel: np.ndarray) -> np.ndarray:
        return rel[:3, 3] - self.c + rel[:3, :3] @ self.c

    def gap(self, rel: np.ndarray) -> float:
        return self.x_ent - float((self.V @ rel[:3, :3].T + rel[:3, 3])[:, 0].max())

    def top_gap(self, rel: np.ndarray) -> float:
        return self.gap(rel) - self.g_top

    def outside(self, top_gap: float, yz, R: np.ndarray) -> np.ndarray:
        """Plug pose rotated by R about the pivot, pivot shifted by yz across a, deepest point top_gap above the
        higher of socket top / entrance (Chongyu's _outside_pose with the pivot instead of the charger origin)."""
        deepest = float(((self.V - self.c) @ R.T)[:, 0].max()) + self.c[0]
        tx = self.x_ent - (self.g_top + top_gap) - deepest
        return self.rel([tx, float(yz[0]), float(yz[1])], R)

    # -------------------------------------------------- clearance (hpp-fcl), socket frame
    def _dist(self, geom, T) -> float:
        f = self.fcl
        if f._collide(geom, T, f.sock):
            return -1.0
        return f._dist(geom, T, f.sock)

    def plug_clearance(self, rel: np.ndarray) -> float:
        """Distance [m] between the plug at `rel` and the socket mesh (-1 if they intersect)."""
        return self._dist(self.fcl.plug, R_G @ rel @ _inv(R_G))

    def hand_finger_clearance(self, T_socket_tcp: np.ndarray, q_finger: float) -> dict:
        """Hand and finger distances [m] to the socket (-1 if intersecting) for a TCP pose given in the socket frame,
        fingers at the opening q_finger (meshes of grasp.FingerSocketCheck)."""
        hand = T_socket_tcp @ _inv(G.Tz(G.TCP_Z))
        out = {"hand": self._dist(self.fcl.hand, hand)}
        for name, geom, s in (("left", self.fcl.left, 1.0), ("right", self.fcl.right, -1.0)):
            F = np.eye(4)
            F[:3, 3] = [0.0, s * q_finger, G.FINGER_JOINT_Z]
            out[name] = self._dist(geom, hand @ F)
        return out

    def full_clearance(self, T_socket_tcp: np.ndarray, T_tcp_plug: np.ndarray, q_finger: float) -> dict:
        """Plug, hand and finger distances to the socket for a TCP pose given in the socket frame."""
        return {"plug": self._dist(self.fcl.plug, T_socket_tcp @ T_tcp_plug),
                **self.hand_finger_clearance(T_socket_tcp, q_finger)}


class Execution:
    """One recorded trajectory (Chongyu's execution object): frozen goal and grasp, phases, guards."""

    def __init__(self, robot, geom: PlugGeometry, on_step: StepCallback | None, rng: np.random.Generator,
                 max_steps: int = MAX_STEPS):
        self.r, self.g, self.on_step, self.rng, self.max_steps = robot, geom, on_step, rng, max_steps
        self.G = robot.sc.socket_T(robot.env_id) @ R_G
        self.Gi = _inv(self.G)
        self.h = _inv(robot.hand_pose()) @ self.charger()
        self.hi = _inv(self.h)
        self.steps, self.phase, self.stages = 0, "initialization", []
        self.cmd = None                      # last commanded rel
        self.q_nominal = None                # last nominal IK solution (9)
        self.q_sent = None                   # last sent arm targets (7)
        self.outside, self.insertion, self.perturbation_seen = True, False, False
        self.bias, self.bias_samples = np.zeros(7), 0
        self.residual, self.ins_state = np.zeros(7), np.zeros(6)
        self.prev = None
        self.m = {}
        self.peak = {}
        self.success_any = False             # AutoMate's episode success over the guarded steps (checks.py)
        self.first_success_step = None       # first control step with success (the start state is not evaluated)
        self._snapshot()
        self.success_any, self.first_success_step = False, None
        # contact left over from the transfer is recorded separately; the peaks cover the guarded steps only
        self.start_contact = {k: self.m[k] for k in ("force", "robot_force", "total_force", "penetration_mm")}
        self.peak = {"plug_socket_N": 0.0, "finger_socket_N": 0.0, "total_socket_N": 0.0, "penetration_mm": 0.0,
                     "plug_socket_points_N": 0.0, "plug_socket_net_sensor_N": 0.0, "min_command_clearance_mm": None,
                     "min_actual_top_gap_mm_outside": None,       # information only (row 24, Chongyu's 1 mm gap)
                     "min_command_hand_finger_clearance_mm": None}    # information (decision 7, row 39)
        self.result = {"completed": False, "failure_reason": None, "steps": 0, "stages": self.stages,
                       "start_contact": self.start_contact}

    # -------------------------------------------------- truth
    def charger(self) -> np.ndarray:
        return self.r.sc.plug_T(self.r.env_id) @ R_G

    def actual(self) -> np.ndarray:
        return self.Gi @ self.charger()

    def tcp(self, rel: np.ndarray) -> np.ndarray:
        return self.G @ rel @ self.hi

    def rel_of_q(self, q9: np.ndarray) -> np.ndarray:
        return self.Gi @ self.r.kin.tcp(q9) @ self.h

    def _snapshot(self):
        r = self.r
        rel = self.actual()
        t = self.g.t_of(rel)
        H = r.hand_pose()
        x, y, z, w = Rotation.from_matrix(rel[:3, :3]).as_quat()
        ts = tilt_spin(np.array([w, x, y, z]), self.g.n)
        c = r.sc.guard_contacts(r.env_id)
        G3 = R_G[:3, :3]
        au = automate_success(G3 @ rel[:3, 3], G3 @ rel[:3, :3] @ G3.T, self.g.d_dis)
        self.success_any = self.success_any or bool(au["success"])
        if au["success"] and self.first_success_step is None:
            self.first_success_step = self.steps
        m = {"success": bool(au["success"]), "height_mm": float(au["height_mm"]),
             "keypoint_dist_mm": float(au["keypoint_dist_mm"]),"t": t, "R": rel[:3, :3].copy(), "rel": rel, "top_gap": self.g.top_gap(rel),
             "lateral": float(np.hypot(t[1], t[2])), "angle": float(ts["angle_deg"]),
             "axis_tilt": float(ts["tilt_deg"]), "spin": float(ts["spin_deg"]),
             "force": c["plug_N"], "robot_force": c["robot_N"], "total_force": c["total_N"],
             "penetration_mm": c["penetration_mm"], "plug_points_N": c["plug_points_N"],
             "sensor_plug_N": c["sensor_plug_net_N"],
             "qvel": float(np.abs(r.get_qvel()[:7]).max()), "H": H}
        if self.prev is None:
            m.update(v_axial=0.0, v_lat=0.0, w=0.0, v_tcp=0.0, w_tcp=0.0)
        else:
            dtv = (t - self.prev["t"]) / DT
            m.update(v_axial=float(dtv[0]), v_lat=float(np.hypot(dtv[1], dtv[2])),
                     w=_angle_deg(self.prev["R"].T @ rel[:3, :3]) / DT)
            d_t, d_r = pose_distance(H, self.prev["H"])
            m.update(v_tcp=d_t / DT, w_tcp=float(np.degrees(d_r)) / DT)
        self.prev = self.m = m
        if self.peak:
            pk = self.peak
            for key, val in (("plug_socket_N", m["force"]), ("finger_socket_N", m["robot_force"]),
                             ("total_socket_N", m["total_force"]), ("penetration_mm", m["penetration_mm"]),
                             ("plug_socket_points_N", m["plug_points_N"]), ("plug_socket_net_sensor_N", m["sensor_plug_N"])):
                pk[key] = max(pk[key], val)

    # -------------------------------------------------- stages
    def _begin(self, phase: str, **info):
        if self.stages and "end_step" not in self.stages[-1]:
            self._end()
        self.phase = phase
        if phase.startswith("perturbation"):
            self.perturbation_seen = True
        self.stages.append({"phase": phase, "start_step": self.steps, **info})

    def _end(self):
        s = self.stages[-1]
        s.update(end_step=self.steps, steps=self.steps - s["start_step"], top_gap_mm=self.m["top_gap"] * 1e3,
                 lateral_mm=self.m["lateral"] * 1e3, angle_deg=self.m["angle"], axis_tilt_deg=self.m["axis_tilt"],
                 spin_deg=self.m["spin"])

    # -------------------------------------------------- one control step
    def _step(self, q7: np.ndarray):
        if self.steps >= self.max_steps:
            raise ExpertFailure(f"step_budget_{self.max_steps}")
        q7 = np.asarray(q7, np.float64)[:7].astype(np.float32).astype(np.float64)
        self.r._command(q7, 1)
        self.steps += 1
        self.q_sent = q7.copy()
        self._snapshot()
        if self.outside:                         # information only: Chongyu's actual-gap guard is not ported (row 24)
            gap_mm, mg = float(self.m["top_gap"]) * 1e3, self.peak.get("min_actual_top_gap_mm_outside")
            self.peak["min_actual_top_gap_mm_outside"] = gap_mm if mg is None else min(mg, gap_mm)
        if self.on_step is not None:
            self.on_step(self.phase, q7.copy(), {})
        if self.m["total_force"] > MAX_FORCE_N:
            raise ExpertFailure("socket_contact_force_guard")
        if self.m["penetration_mm"] > MAX_PENETRATION_MM:
            raise ExpertFailure("socket_penetration_guard")
        if self.outside and self.m["force"] > OUTSIDE_CONTACT_N:
            raise ExpertFailure("actual_plug_socket_contact_before_insertion")

    def _validate(self, q9: np.ndarray, rel_cmd: np.ndarray | None, clearance: float | None) -> str | None:
        r = self.r
        arm = q9[:7]
        if not np.isfinite(arm).all():
            return "non_finite_command"
        if np.any(arm < r.kin.lower[:7]) or np.any(arm > r.kin.upper[:7]):
            return "joint_limit"
        prev = r.qpos()[:7] if self.q_sent is None else self.q_sent
        if np.abs(arm - prev).max() > MAX_ACTION_DELTA_RAD:
            return "per_action_joint_delta"
        if clearance is not None and self.outside:
            rel = self.rel_of_q(q9) if rel_cmd is None else rel_cmd
            d = self.g.plug_clearance(rel)
            mc = self.peak["min_command_clearance_mm"]
            self.peak["min_command_clearance_mm"] = d * 1e3 if mc is None else min(mc, d * 1e3)
            if d < clearance:
                return "commanded_plug_near_socket_before_insertion"
        return None

    def _ik(self, rel: np.ndarray) -> np.ndarray:
        seed = self.r.qpos() if self.q_nominal is None else self.q_nominal
        q = self.r._ik(self.tcp(rel), seed)
        if q is None:
            raise ExpertFailure(f"{self.phase}_ik_failed")
        q = q.copy()
        q[:7] = q[:7].astype(np.float32).astype(np.float64)
        return q

    def _compensate(self, q: np.ndarray) -> np.ndarray:
        """Chongyu's tracking compensation: filtered (last sent - measured) joints, learned while quiet outside."""
        m = self.m
        quiet = (self.q_sent is not None and self.outside and not self.insertion
                 and self.phase in ("continuous_approach", "recovery_align") and m["qvel"] < 0.02
                 and m["v_tcp"] < 0.0003 and m["w_tcp"] < 0.5 and m["force"] < 1.0)
        if quiet:
            err = np.clip(self.q_sent - self.r.qpos()[:7], -TRACK_BOUND, TRACK_BOUND)
            self.bias = (1 - TRACK_FILTER) * self.bias + TRACK_FILTER * err
            self.bias_samples += 1
        elif self.outside:
            self.bias_samples = 0
        out = q.copy()
        if not np.any(self.bias):
            return out
        for scale in (1.0, 0.5, 0.25, 0.125):
            cand = q.copy()
            cand[:7] = (q[:7] + scale * TRACK_GAIN * self.bias).astype(np.float32).astype(np.float64)
            if self._validate(cand, None, COMMAND_CLEARANCE_M) is None:
                return cand
        return out

    def command_hand_finger_clearance(self, rel: np.ndarray) -> float:
        """Smallest hand / finger - socket mesh distance [m] (-1 if intersecting) of the commanded TCP pose tcp(rel)
        (socket frame: R_G rel h^-1), fingers at the closing opening robot.q_finger (decision 7, row 39)."""
        d = self.g.hand_finger_clearance(R_G @ rel @ self.hi, self.r.q_finger)
        return min(d.values())

    def _send(self, rel: np.ndarray, approach_noise: bool = False, insertion_noise: bool = False,
              compensate: bool = True) -> np.ndarray:
        q = self._ik(rel)
        reason = self._validate(q, rel, COMMAND_CLEARANCE_M)
        if reason:
            raise ExpertFailure(f"{self.phase}_nominal_command_guard: {reason}")
        if self.outside:                         # decision 7 of 2026-11-23 (row 39): the nominal command only
            d = self.command_hand_finger_clearance(rel)
            mc = self.peak["min_command_hand_finger_clearance_mm"]
            self.peak["min_command_hand_finger_clearance_mm"] = d * 1e3 if mc is None else min(mc, d * 1e3)
            if d < HAND_FINGER_COMMAND_CLEARANCE_M:
                raise ExpertFailure(f"{self.phase}_nominal_command_guard: hand_finger_command_clearance")
        sent, sent_rel = q, rel
        if approach_noise:
            for _ in range(NOISE_ATTEMPTS):
                prop = JOINT_NOISE_CORR * self.residual + np.sqrt(1 - JOINT_NOISE_CORR ** 2) * \
                    self.rng.normal(0.0, JOINT_NOISE_SIGMA, 7)
                if np.abs(prop).max() > JOINT_NOISE_BOUND:
                    continue
                cand = q.copy()
                cand[:7] = (q[:7] + prop).astype(np.float32).astype(np.float64)
                if self._validate(cand, None, NOISY_CLEARANCE_M) is None:
                    sent, sent_rel = cand, self.rel_of_q(cand)
                    break
            self.residual = sent[:7] - q[:7]
        if insertion_noise:
            remaining = max(0.0, -float(self.g.t_of(rel)[0]))
            taper = INS_NOISE_FINAL + (1 - INS_NOISE_FINAL) * min(1.0, remaining / INS_NOISE_TAPER_M)
            for _ in range(NOISE_ATTEMPTS):
                state = INS_NOISE_CORR * self.ins_state + (1 - INS_NOISE_CORR) * self.rng.uniform(-1, 1, 6)
                n = state * INS_NOISE_BOUNDS * taper
                t = self.g.t_of(rel) + n[:3]
                if t[0] > 0.0:
                    continue
                cand_rel = self.g.rel(t, rel[:3, :3] @ euler_R(n[3:]))
                try:
                    cand = self._ik(cand_rel)
                except ExpertFailure:
                    continue
                if self._validate(cand, cand_rel, None) is None:
                    sent, sent_rel, self.ins_state = cand, cand_rel, state
                    break
        if compensate:
            sent = self._compensate(sent)
        self._step(sent[:7])
        self.cmd, self.q_nominal = sent_rel, q
        return sent

    # -------------------------------------------------- motions
    def _segment(self, target: np.ndarray, vmax: float, wmax_deg: float, acc: float, angacc_deg: float,
                 noise_free: bool = True):
        """Commanded rel from self.cmd (or the true pose) to target: pivot on a straight line, rotation geodesic,
        trapezoidal profile limited by the pivot and TCP translation speeds and the rotation speed."""
        start = self.actual() if self.cmd is None else self.cmd
        t0, t1 = self.g.t_of(start), self.g.t_of(target)
        d_piv = float(np.linalg.norm(t1 - t0))
        d_tcp = pose_distance(self.tcp(start), self.tcp(target))[0]
        d_lin = max(d_piv, d_tcp)
        d_rot = _angle_deg(start[:3, :3].T @ target[:3, :3])
        if d_lin < 1e-7 and d_rot < 1e-5:
            return
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        inf = float("inf")
        sdot_max = min(vmax / d_lin if d_lin > 0 else inf, wmax_deg / d_rot if d_rot > 0 else inf)
        sacc = min(acc / d_lin if d_lin > 0 else inf, angacc_deg / d_rot if d_rot > 0 else inf)
        s, sdot = 0.0, 0.0
        while s < 1.0 - 1e-9:
            sdot = min(sdot + sacc * DT, sdot_max, np.sqrt(2.0 * sacc * max(1.0 - s, 0.0)) + 0.5 * sacc * DT)
            s = min(1.0, s + max(sdot, 0.25 * sacc * DT) * DT)
            self._send(self.g.rel(t0 + s * (t1 - t0), slerp([s]).as_matrix()[0]))

    def _ready(self, target: np.ndarray) -> bool:
        m, g = self.m, self.g
        tt = g.t_of(target)
        gap_t = g.top_gap(target)
        return bool(np.hypot(*(m["t"][1:] - tt[1:])) <= POSITION_TOL_M
                    and _angle_deg(target[:3, :3].T @ m["R"]) <= ORIENTATION_TOL_DEG
                    and gap_t - POSITION_TOL_M <= m["top_gap"] <= gap_t + 0.003
                    and m["v_lat"] <= LIN_VEL_TOL_M_S and m["w"] <= ANG_VEL_TOL_DEG_S
                    and -LIN_VEL_TOL_M_S <= m["v_axial"] <= PUSH_SPEED_M_S
                    and self.bias_samples >= SETTLE_CONSECUTIVE)

    def approach_align(self, target: np.ndarray):
        """approach (fast) to the stand-off `target`, then align: hold it until the true plug is there."""
        phase = "recovery_align" if self.perturbation_seen else "continuous_approach"
        self._begin(phase, kind="approach_align")
        self._segment(target, APPROACH_SPEED_M_S, APPROACH_ANG_SPEED_DEG_S, APPROACH_ACCEL_M_S2,
                      APPROACH_ANG_ACCEL_DEG_S2)
        consecutive, t0 = 0, self.steps
        while consecutive < SETTLE_CONSECUTIVE:
            if self.steps - t0 >= SETTLE_TIMEOUT:
                raise ExpertFailure("moving_actual_alignment_admission_timeout")
            self._send(target)
            consecutive = consecutive + 1 if self._ready(target) else 0
        self.stages[-1]["admission_step"] = self.steps
        self._end()

    def _insert_guard(self):
        if self.m["lateral"] > POSITION_TOL_M:
            raise ExpertFailure("actual_insertion_transverse_error_guard")
        if self.m["angle"] > ORIENTATION_TOL_DEG:
            raise ExpertFailure("actual_insertion_attitude_guard")

    def insert(self, noise: bool):
        """axial_push at 3 mm/s (throttled to lead the true plug by <= 0.3 mm), then final_verify."""
        self.outside, self.insertion = False, True
        x = float(self.g.t_of(self.cmd)[0])
        self._begin("axial_push", start_top_gap_mm=self.m["top_gap"] * 1e3, throttled=0)
        while x < -1e-10:
            self._insert_guard()
            actual_x = float(self.m["t"][0])
            scheduled = min(0.0, x + PUSH_SPEED_M_S * DT)
            nxt = max(x, min(scheduled, actual_x + MAX_INSERT_LAG_M))
            self.stages[-1]["throttled"] += int(nxt < scheduled)
            x = nxt
            self._send(self.g.rel([x, 0.0, 0.0], np.eye(3)), insertion_noise=noise, compensate=False)
            self._insert_guard()
        self._begin("final_verify")
        consecutive = 0
        for _ in range(SETTLE_TIMEOUT):
            self._send(np.eye(4), compensate=False)
            m = self.m
            depth_left = -float(m["rel"][0, 3])
            ready = (depth_left < 0.0005 and m["lateral"] <= POSITION_TOL_M and m["angle"] <= ORIENTATION_TOL_DEG
                     and m["v_tcp"] <= LIN_VEL_TOL_M_S and m["w_tcp"] <= ANG_VEL_TOL_DEG_S
                     and m["qvel"] <= JOINT_VEL_TOL_RAD_S)
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= SETTLE_CONSECUTIVE:
                self._end()
                return
        raise ExpertFailure("final_actual_pose_or_velocity_convergence_timeout")

    def perturbation(self):
        """Chongyu's stochastic window (perturbation_stochastic), from the true pose towards a 6 mm top gap."""
        g = self.g
        self._begin("perturbation_stochastic")
        source = self.actual()
        t_src, gap0 = g.t_of(source), g.top_gap(source)
        drift, endpoint = np.zeros(6), np.zeros(6)
        prev = source if self.cmd is None else self.cmd
        for k in range(STOCHASTIC_STEPS):
            if k % DRIFT_RENEW == 0:
                endpoint = self.rng.uniform(-DRIFT_AMP, DRIFT_AMP)
            drift += DRIFT_SMOOTH * (endpoint - drift)
            frac = (k + 1) / STOCHASTIC_STEPS
            gap = gap0 + frac * (STOCHASTIC_END_GAP_M - gap0) + drift[0]
            yz = (1 - frac) * t_src[1:] + drift[1:3]
            desired = g.outside(gap, yz, euler_R(drift[3:]))
            cmd = self._advance(prev, desired, CLOSE_SPEED_M_S, CLOSE_ANG_SPEED_DEG_S)
            self._send(cmd, approach_noise=True, compensate=False)
            prev = cmd
        self._end()

    def _advance(self, a: np.ndarray, b: np.ndarray, speed_m_s: float, ang_speed_deg_s: float) -> np.ndarray:
        """One step from a towards b, at most speed_m_s * DT (pivot and TCP) and ang_speed_deg_s * DT (Chongyu's
        _advance, CY continuous.py:321-335): the fraction f from the whole distance, then halved (at most 24 times)
        until the step itself is within the limit (the TCP moves on an arc, so f of the chord can exceed it)."""
        ta, tb = self.g.t_of(a), self.g.t_of(b)
        d_lin = max(float(np.linalg.norm(tb - ta)), pose_distance(self.tcp(a), self.tcp(b))[0])
        d_rot = _angle_deg(a[:3, :3].T @ b[:3, :3])
        f = min(1.0, speed_m_s * DT / max(d_lin, 1e-12), ang_speed_deg_s * DT / max(d_rot, 1e-12))
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))
        for _ in range(24):
            point = self.g.rel(ta + f * (tb - ta), slerp([f]).as_matrix()[0])
            step = max(f * float(np.linalg.norm(tb - ta)), pose_distance(self.tcp(a), self.tcp(point))[0])
            if step <= speed_m_s * DT + 1e-9 and _angle_deg(a[:3, :3].T @ point[:3, :3]) <= ang_speed_deg_s * DT + 1e-7:
                return point
            f *= 0.5
        raise ExpertFailure("continuous_target_spacing_limit_failed")

    def _pose_error_mm(self, a: np.ndarray, b: np.ndarray) -> float:
        """wm_v1's _pose_error_mm (V1 env/expert.py:101-106): TCP translation [mm] + TCP rotation [deg] (1 deg counted
        as 1 mm) between two rel poses."""
        ta, tb = self.tcp(a), self.tcp(b)
        d_t, d_r = pose_distance(ta, tb)
        return d_t * 1000.0 + float(np.degrees(d_r))

    def _move_outside(self, phase: str, target: np.ndarray, **info):
        """wm_v1's _move_outside (V1 env/expert.py:89-99), the motion of the contact offset and of every random
        waypoint (restored 2026-11-23, deviation rows 27 and 38): from the last command (or the true pose), steps of
        _advance at RANDOM_SPEED_M_S = 2.7 mm/s and CLOSE_ANG_SPEED_DEG_S = 2 deg/s until the command is within
        MOVE_OUTSIDE_TOL of target; at most MOVE_OUTSIDE_MAX_STEPS commands, else "<phase>_did_not_reach_target".
        No ramp and no hold."""
        self._begin(phase, speed_m_s=RANDOM_SPEED_M_S, ang_speed_deg_s=CLOSE_ANG_SPEED_DEG_S, **info)
        command = self.actual() if self.cmd is None else self.cmd
        for _ in range(MOVE_OUTSIDE_MAX_STEPS):
            if self._pose_error_mm(command, target) < MOVE_OUTSIDE_TOL:
                break
            command = self._advance(command, target, RANDOM_SPEED_M_S, CLOSE_ANG_SPEED_DEG_S)
            self._send(command)
        else:
            raise ExpertFailure(f"{phase}_did_not_reach_target")
        self._end()

    def contact_push(self, start: np.ndarray) -> dict:
        """Straight push along a from `start` at 3 mm/s without the insertion guards; stop on a jam."""
        self.outside, self.insertion = False, False
        t = self.g.t_of(start)
        R = start[:3, :3].copy()
        self._begin("contact_push")
        x, outcome, guard = float(t[0]), "inserted", None
        try:
            while x < 0.0:
                if x - float(self.m["t"][0]) > JAM_LAG_M:
                    outcome = "jammed"
                    break
                x = min(0.0, x + PUSH_SPEED_M_S * DT)
                self._send(self.g.rel([x, t[1], t[2]], R), compensate=False)
            for _ in range(10):
                self._step(self.q_sent)
        except ExpertFailure as exc:
            outcome, guard = f"guard:{str(exc)[:60]}", str(exc)
        self._end()
        depth_left = -float(self.actual()[0, 3])
        tr = truth(self.r)
        # outcome: the wm_v1 contact rule (depth at the plug CAD origin <= 1 mm, no tilt / lateral check), named
        # "reached_depth" (wm_v1: "inserted"); it is not the success criterion, which is AutoMate's (result "success":
        # after any guarded step; "success_end")
        return {"outcome": "reached_depth" if depth_left <= INSERTED_MARGIN_M else outcome,
                "depth_left_mm": depth_left * 1000.0, "guard_reason": guard, "truth": tr}

    def stand_off(self) -> np.ndarray:
        return self.g.outside(NEAR_GAP_M, np.zeros(2), np.eye(3))

    def offset_rel(self, off: Offset) -> np.ndarray:
        """Contact offsets and random waypoints: Offset(top gap, lateral of the pivot, static XYZ about the pivot)."""
        return self.g.outside(off.tip_gap_m, off.yz_m, euler_R(off.rpy_deg))

    # -------------------------------------------------- entry
    def execute(self, kind: str, perturbed: bool = False, **kw) -> dict:
        """kind "random" takes waypoints (Offset list); "speeds", if given, must be None or RANDOM_SPEED_M_S for every
        waypoint (the speed is wm_v1's constant since 2026-11-23, not a draw)."""
        if kind == "random" and kw.get("speeds") is not None:
            sp = [float(v) for v in kw["speeds"]]
            if len(sp) != len(kw["waypoints"]) or any(abs(v - RANDOM_SPEED_M_S) > 1e-12 for v in sp):
                raise ValueError(f"random-motion speeds {sp}: the speed is the constant RANDOM_SPEED_M_S = "
                                 f"{RANDOM_SPEED_M_S} m/s (wm_v1, the user's decision 2026-11-23); pass speeds=None")
        info = {}
        t0 = time.time()
        try:
            if kind in ("insert", "perturbed", "reference"):
                if kind == "perturbed":
                    self.perturbation()
                self.approach_align(self.stand_off())
                self.insert(noise=kind == "perturbed")
            elif kind == "contact":                                      # wm_v1: align, _move_outside, push
                self.approach_align(self.stand_off())
                target = self.offset_rel(kw["offset"])
                self._move_outside("contact_offset", target)            # row 38, restored 2026-11-23
                info = self.contact_push(target)
            elif kind == "random":                                       # row 27, restored 2026-11-23
                for i, wp in enumerate(kw["waypoints"]):
                    self._move_outside(f"random_motion_{i}", self.offset_rel(wp))
            else:
                raise ValueError(kind)
            self.result["completed"] = True
        except ExpertFailure as exc:
            self.result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            if self.stages and "end_step" not in self.stages[-1]:
                self._end()
                self.stages[-1]["failure"] = str(exc)
        pk = dict(self.peak)
        pk["finger_contact"] = bool(pk["finger_socket_N"] > FINGER_CONTACT_INFO_N)     # information only
        self.result.update(steps=self.steps, last_phase=self.phase, wall_s=round(time.time() - t0, 1),
                           peak=pk, success=bool(self.success_any),
                           success_end=bool(self.m["success"]) if self.steps > 0 else False,
                           first_success_step=self.first_success_step, **info)
        return self.result


class Expert:
    """Places the hand (not recorded) and runs recorded expert trajectories (the API of wm.env.expert.Expert)."""

    def __init__(self, robot, transfer: str = "move"):
        assert transfer in ("move", "teleport"), transfer
        self.robot, self.transfer = robot, transfer
        self.geom = PlugGeometry(robot)

    # ------------------------------------------------- recorded trajectories
    def run(self, kind: str, on_step: StepCallback, seed_offset, **kw) -> dict:
        """seed_offset: an int (wm_v1) or a numpy SeedSequence (collect.py's expert-noise stream) for the noise."""
        ss = seed_offset if isinstance(seed_offset, np.random.SeedSequence) else int(seed_offset)
        ex = Execution(self.robot, self.geom, on_step, np.random.default_rng(ss),
                       max_steps=step_budget(self.geom.L, self.geom.g_top))
        return ex.execute(kind, **kw)

    def reference_label(self, seed_offset, on_step: StepCallback | None = None) -> dict:
        """The FK H* label of the current reset (config.SOCKET_DRAW "per_reset": one per trajectory): an unrecorded
        reference insertion from the reset spawn (kind "reference", the same procedure as the recorded t00), then
            H_star            FK hand pose at its end (wm_v1's taught pose)
            axis              direction of the FK hand from the first to the last step of the axial push (wm_v1's
                              episode._axis_from), None if the push had fewer than 2 steps
            H_star_model      T_socket T_hp^-1 (simulator truth), H_star_fk_vs_model_mm_deg = [mm, deg] between them
            completed         the expert finished (Chongyu's final_verify passed)
            success_end       AutoMate's success (checks.py) after the last step
            success_any, first_success_step   AutoMate's episode success over the steps (information)
            ok                completed AND success_end: the label counts only then (as wm_v1's reference_ok, which
                              required completion, plus AutoMate's rule at the end; lead's resolution 2026-11-23)
            failure           the expert's failure reason, or why ok is False; truth (end), steps, peak
        on_step (optional) is called as in run() (e.g. to count steps); nothing is recorded here."""
        r = self.robot
        hands = []

        def cb(phase, q7, info):
            if phase == "axial_push":
                hands.append(r.hand_pose()[:3, 3].copy())
            return on_step(phase, q7, info) if on_step is not None else None
        res = self.run("reference", cb, seed_offset)
        tr = truth(r)
        H, Hm = r.hand_pose(), r.H_star()
        dt = float(np.linalg.norm(H[:3, 3] - Hm[:3, 3]))
        dr = float(np.degrees(np.arccos(np.clip((np.trace(H[:3, :3].T @ Hm[:3, :3]) - 1) / 2, -1, 1))))
        axis = None
        if len(hands) >= 2 and np.linalg.norm(hands[-1] - hands[0]) > 0:
            axis = (hands[-1] - hands[0]) / np.linalg.norm(hands[-1] - hands[0])
        completed, success_end = bool(res.get("completed")), bool(res.get("success_end"))
        ok = completed and success_end
        failure = res.get("failure_reason") or (None if ok else "expert completed but AutoMate's success does not hold "
                                                "after the last step")
        return {"ok": ok, "completed": completed, "success_end": success_end, "success_any": bool(res.get("success")),
                "first_success_step": res.get("first_success_step"), "H_star": H.tolist(),
                "axis": None if axis is None else axis.tolist(), "H_star_model": Hm.tolist(),
                "H_star_fk_vs_model_mm_deg": [dt * 1e3, dr], "truth": tr, "steps": int(res["steps"]),
                "peak": res.get("peak"), "failure": failure, "socket_index": r.episode.get("socket_index"),
                "socket_pose": r.episode.get("socket_pose")}

    # ------------------------------------------------- start poses (task definition, section 6)
    @staticmethod
    def start_pose(H_star: np.ndarray, F: np.ndarray, L: float, off: Offset, g_top: float | None = None,
                   gap_from: str | None = None) -> np.ndarray:
        """TCP pose of a start: H* moved back along a by L + g_top + g0 (config.START_GAP_FROM "socket_top", the
        default and approved value, the user's decision 2026-11-23; g_top required: self.geom.g_top or
        checks.g_top_of(spec), a missing g_top raises) or by L + g0 ("entrance", diagnostics only), shifted by yz
        along F's y, z columns (the goal frame's lateral axes, F = frames.frame(a) = [a, -e2, e1]) and rotated about
        the TCP by the static XYZ angles rpy expressed in F (frames.start_pose)."""
        return frames.start_pose(H_star, F, L, off, g_top, gap_from)

    # ------------------------------------------------- unrecorded placement
    def _socket_frame(self, T: np.ndarray) -> np.ndarray:
        return _inv(self.robot.sc.socket_T(self.robot.env_id)) @ T

    def waypoint_clearance(self, off: Offset) -> float:
        """Smallest hand / finger - socket mesh distance [m] (-1 if intersecting) at the commanded pose of a random
        waypoint (Execution.offset_rel), with the grasp's nominal TCP - plug transform robot.T_tcp_plug and the closing
        opening robot.q_finger, as start_clearance. collect.py redraws a waypoint below HAND_FINGER_COMMAND_CLEARANCE_M
        (decision 7 of 2026-11-23, row 39). It does not depend on the socket pose."""
        r, g = self.robot, self.geom
        rel = g.outside(off.tip_gap_m, off.yz_m, euler_R(off.rpy_deg))
        d = g.hand_finger_clearance(R_G @ rel @ _inv(R_G) @ _inv(r.T_tcp_plug), r.q_finger)
        return min(d.values())

    def start_clearance(self, T_start: np.ndarray) -> dict:
        """Clearances [m] of plug / hand / fingers at the start and the plug's smallest clearance on the descent."""
        r, g = self.robot, self.geom
        a = r.axis()
        out = g.full_clearance(self._socket_frame(T_start), r.T_tcp_plug, r.q_finger)
        rel = _inv(r.sc.socket_T(r.env_id) @ R_G) @ T_start @ r.T_tcp_plug @ R_G
        lift = max(0.0, TRANSFER_GAP_M - g.top_gap(rel))
        desc = []
        for d in np.arange(0.0, lift + 1e-9, 0.002):
            T = T_start.copy()
            T[:3, 3] -= d * a
            desc.append(g._dist(g.fcl.plug, self._socket_frame(T) @ r.T_tcp_plug))
        out["descent_plug"] = min(desc) if desc else out["plug"]
        out["min"] = min(out.values())
        return out

    def _true_rel(self) -> np.ndarray:
        r = self.robot
        return _inv(r.sc.socket_T(r.env_id) @ R_G) @ r.sc.plug_T(r.env_id) @ R_G

    def _move(self, T: np.ndarray, settle_steps: int = SETTLE_STEPS_TRANSFER) -> bool:
        return self.robot.move_to(T, max_lin=TRANSFER_MAX_LIN, max_ang=TRANSFER_MAX_ANG, settle_steps=settle_steps)

    def _forces(self) -> dict:
        r = self.robot
        return r.sc.guard_contacts(r.env_id)

    def _retreat(self, T_goal: np.ndarray, plug_contact_max_n: float | None = None) -> dict:
        """Legs 1 and 3: straight line to T_goal at <= RETREAT_SPEED_M_S, one IK target per control step, with the lag,
        force and penetration guards (module docstring) and, when plug_contact_max_n is given, the outside rule
        plug-socket F <= plug_contact_max_n; then settling (robot.move_to).
        -> {"ok", "reason", "steps", max_lag_mm, peak_plug_socket_N, peak_finger_socket_N, peak_penetration_mm}."""
        r = self.robot
        T0 = r.hand_pose()
        dt, dr = pose_distance(T0, T_goal)
        n = max(1, int(np.ceil(max(dt / (RETREAT_SPEED_M_S * DT), dr / TRANSFER_MAX_ANG))))
        seed, lag, fp, ff, pen, reason = r.qpos(), 0.0, 0.0, 0.0, 0.0, None
        for k in range(1, n + 1):
            cmd = interpolate_T(T0, T_goal, k / n)
            q = r._ik(cmd, seed)
            if q is None:
                reason = "ik"
                break
            r._command(q, 1)
            seed = q
            lag = max(lag, float(np.linalg.norm(cmd[:3, 3] - r.hand_pose()[:3, 3])))
            c = self._forces()
            fp, ff, pen = max(fp, c["plug_N"]), max(ff, c["robot_N"]), max(pen, c["penetration_mm"])
            if lag > TRANSFER_LAG_STOP_M:
                reason = "lag"
            elif c["total_N"] > MAX_FORCE_N:
                reason = "socket_force"
            elif c["penetration_mm"] > MAX_PENETRATION_MM:
                reason = "socket_penetration"
            elif plug_contact_max_n is not None and c["plug_N"] > plug_contact_max_n:
                reason = "plug_socket_contact"
            if reason:
                break
        if reason is None and not self._move(T_goal):
            reason = "settle"
        return {"ok": reason is None, "reason": reason, "steps": n, "max_lag_mm": lag * 1e3, "peak_plug_socket_N": fp,
                "peak_finger_socket_N": ff, "peak_penetration_mm": pen}

    def move_to_start(self, T_start: np.ndarray) -> dict:
        """Unrecorded transfer to T_start (module docstring). -> {"ok", "steps", "seconds", "stage", ...}."""
        r, g = self.robot, self.geom
        t0, s0 = time.time(), r.sc.steps
        a = r.axis()
        out = {"ok": False, "stage": None}
        if self.transfer == "teleport":
            out.update(self._teleport(T_start))
        else:
            up = max(0.0, TRANSFER_GAP_M - g.top_gap(self._true_rel()))
            T = r.hand_pose()
            T[:3, 3] -= up * a
            rel_s = _inv(r.sc.socket_T(r.env_id) @ R_G) @ T_start @ r.T_tcp_plug @ R_G
            T_high = T_start.copy()
            T_high[:3, 3] -= max(0.0, TRANSFER_GAP_M - g.top_gap(rel_s)) * a
            legs = []
            leg = self._retreat(T)
            legs.append({"leg": "retreat", **leg})
            if leg["ok"]:
                ok = self._move(T_high)
                c = self._forces()
                dt_, _ = pose_distance(r.hand_pose(), T_high)
                legs.append({"leg": "transfer", "ok": ok, "reason": None if ok else "ik_or_settle",
                             "end_error_mm": dt_ * 1e3, "plug_socket_N": c["plug_N"], "finger_socket_N": c["robot_N"],
                             "penetration_mm": c["penetration_mm"]})
                if ok:
                    leg = self._retreat(T_start, plug_contact_max_n=OUTSIDE_CONTACT_N)
                    dt_, _ = pose_distance(r.hand_pose(), T_start)
                    legs.append({"leg": "descend", **leg, "end_error_mm": dt_ * 1e3})
            out["legs"] = legs
            failed = [x for x in legs if not x["ok"]]
            if failed:
                out["stage"] = failed[0]["leg"]
            else:
                out["ok"] = True
        dt, dr = pose_distance(r.hand_pose(), T_start)
        out.update(steps=r.sc.steps - s0, seconds=round(time.time() - t0, 1), start_error_mm=dt * 1e3,
                   start_error_deg=float(np.degrees(dr)))
        out["ok"] = bool(out["ok"] and dt < 5e-4 and np.degrees(dr) < 0.2)
        return out

    def _teleport(self, T_start: np.ndarray) -> dict:
        """Option (not the default): joints written to the start configuration at rest, as a reset does."""
        r = self.robot
        q = r._ik(T_start, r.qpos())
        if q is None:
            q = r.reset_ik(T_start)
        if q is None:
            return {"ok": False, "stage": "ik"}
        q[7:9] = r.q_finger
        plug_T = r.kin.link("panda_hand", q) @ C.T_HAND_TCP @ r.T_tcp_plug
        r.sc.write_state(r.env_id, q, plug_T, np.r_[q[:7], r.q_finger, r.q_finger])
        r._q_cmd = q[:7].copy()
        r._command(q[:7], 10)
        return {"ok": True}
