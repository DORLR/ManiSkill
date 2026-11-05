"""Insertion demonstrations and rollouts of a learned insertion policy.

Demonstrator (stand-in for teleoperation): it knows the pre-insertion pose
H_pre of this grasp (from the one insertion, FK only), corrects the lateral and
rotational offset in small steps, then advances along the insertion axis until
the pins are fully inserted. Executed actions get small noise while the
recorded label is the clean action (DART, Laskey et al. 2017), so the policy
sees and learns to correct small deviations.

Action = hand motion for one step in port axes, sigma units (1 mm, 1 deg),
executed with apply_port_motion (rotation about the hand position).
Observation = RGB crop + coloured point cloud (paf.perception) + proprio
(hand motion since the episode start, in port axes, sigma units; FK only).
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .geometry import apply_port_motion, relative_state
from .perception import preprocess

STEP_LAT, STEP_ROT, STEP_ADV = 0.3, 0.3, 0.5   # sigma units per step (mm / deg)
ALIGN_TOL_LAT, ALIGN_TOL_ROT = 0.15, 0.15
MAX_STEPS = 80
DART_NOISE = 0.08                                 # sigma units, executed only


def demo_action(u, stroke_sigma):
    """Clean demonstrator action for true state u (sigma units)."""
    a = np.zeros(6)
    lat, rot = u[1:3], u[3:]
    a[1:3] = np.clip(-lat, -STEP_LAT, STEP_LAT)
    a[3:] = np.clip(-rot, -STEP_ROT, STEP_ROT)
    aligned = np.all(np.abs(lat) < ALIGN_TOL_LAT) and np.all(np.abs(rot) < ALIGN_TOL_ROT)
    if aligned or u[0] > 0.5:  # once the pins are entering, keep pushing while correcting
        a[0] = min(STEP_ADV, stroke_sigma - u[0])
    elif u[0] < -0.5:
        a[0] = min(STEP_ADV, -u[0] - 0.5)  # approach to just outside the board while aligning
    return a


def proprio(H, H0, R_P):
    """Hand motion since the episode start, in port axes, sigma units (FK only)."""
    return relative_state(H, H0, R_P) / C.SIGMA


def inserted(u, stroke_sigma):
    return u[0] >= stroke_sigma - 1.0


def run_demo(robot, H_pre, R_P, R_P_est, q_start, rng):
    """One demonstration from q_start (metres/radians). Returns frames and success."""
    from .geometry import pose_from_state
    stroke = (robot.port_spec.pin_length_m + C.GAP0) / C.SIGMA_T
    if not robot.move_to(pose_from_state(q_start, H_pre, R_P)):
        return None
    H0 = robot.hand_pose()
    frames = {"rgb": [], "pc": [], "prop": [], "act": [], "u": []}
    ok = False
    for _ in range(MAX_STEPS):
        obs = robot.observe()
        u = relative_state(obs.H, H_pre, R_P) / C.SIGMA
        if inserted(u, stroke):
            ok = True
            break
        crop, pc = preprocess(obs, R_P_est, rng)
        a = demo_action(u, stroke)
        frames["rgb"].append(crop)
        frames["pc"].append(pc.astype(np.float16))
        frames["prop"].append(proprio(obs.H, H0, R_P_est).astype(np.float32))
        frames["act"].append(a.astype(np.float32))
        frames["u"].append(u.astype(np.float32))
        a_exec = a + rng.normal(size=6) * DART_NOISE
        if not robot.move_to(apply_port_motion(obs.H, a_exec * C.SIGMA, R_P), settle_steps=10):
            break
    return {k: np.asarray(v) for k, v in frames.items()}, ok


def rollout_policy(robot, policy, H_pre, R_P, rng, max_steps=MAX_STEPS):
    """Run a learned insertion policy from the current hand pose. H_pre and R_P are
    used ONLY to score (insertion depth); the policy sees images + proprio."""
    stroke = (robot.port_spec.pin_length_m + C.GAP0) / C.SIGMA_T
    H0 = robot.hand_pose()
    policy.reset()
    for t in range(max_steps):
        obs = robot.observe()
        u_true = relative_state(obs.H, H_pre, R_P) / C.SIGMA
        if inserted(u_true, stroke):
            return {"inserted": True, "steps": t, "u_end": u_true.tolist()}
        a = policy.act(obs, H0, rng)
        if not robot.move_to(apply_port_motion(obs.H, a * C.SIGMA, policy.R_P), settle_steps=10):
            break
    u_true = relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA
    return {"inserted": bool(inserted(u_true, stroke)), "steps": max_steps, "u_end": u_true.tolist()}
