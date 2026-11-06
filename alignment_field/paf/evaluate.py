"""Closed-loop test: can the field pull OOD starts into the ID set?

Per test grasp (never in training): one expert insertion gives H_pre and the
port axes R_P by FK. These are used ONLY to place the hand at the OOD start
and to score the end state; the method (paf.policy) never receives them.

Methods run from identical starts:
  field     estimated state from point cloud + RGB, learned field
  field_pc  estimated state from the point-cloud head only (ablation)
  oracle    true state (FK vs H_pre) through the same field (upper bound)
  none      no alignment, push straight from the OOD start (lower bound)
Every episode ends with the same fixed straight push (success = insertion).
"""

from __future__ import annotations

import time
import traceback

import numpy as np

from . import config as C
from .collect import establish_reference
from .geometry import apply_port_motion, pose_from_state, relative_state


# Control baseline target: 2 mm before the pre-insertion pose, aligned (the same
# point the policy's in-distribution reference start P_id is drawn around).
POINT_TARGET = np.array([-2.0, 0, 0, 0, 0, 0])


def sample_ood_start(rng):
    t_lat = rng.uniform(*C.OOD_T, 2) * rng.choice([-1, 1], 2)
    t = np.r_[rng.uniform(*C.OOD_TX), t_lat]
    r = rng.uniform(-C.OOD_R, C.OOD_R, 3)
    return np.r_[t, r]


def run_loop(robot, policy, H_pre, R_P, method, rng):
    """Returns the trajectory of true and estimated states (sigma units)."""
    traj = []
    small = 0
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    for it in range(C.LOOP_MAX_ITERS):
        obs = robot.observe()
        u_true = relative_state(obs.H, H_pre, R_P) / C.SIGMA
        oracle = method in ("oracle", "point_oracle")
        if oracle:
            u_est, p_id = u_true, float("nan")
        else:
            u_est, p_id = policy.estimate_state(obs, rng)
        if method in ("point", "point_oracle"):
            v, du = policy.point_step(u_est, POINT_TARGET)
        else:
            v, du = policy.field_step(u_est)
        target = apply_port_motion(obs.H, du * C.SIGMA, R_P) if oracle else policy.target_pose(obs, du)
        traj.append({"it": it, "u_true": u_true.tolist(), "u_est": u_est.tolist(), "p_id": p_id,
                     "v_norm": float(np.linalg.norm(v)), "step_norm": float(np.linalg.norm(du))})
        small = small + 1 if np.linalg.norm(du) < 0.5 else 0
        if small >= 2:
            break
        # Safety clamp, identical for all methods: never command the hand more
        # than 1 sigma (1 mm) past the pre-insertion depth in the state the
        # method believes it is in (tips stay >= 2 mm outside the board).
        if u_est[0] + du[0] > 1.0:
            du = du.copy()
            du[0] = 1.0 - u_est[0]
            target = apply_port_motion(obs.H, du * C.SIGMA, R_P) if oracle else policy.target_pose(obs, du)
        if not robot.move_to(target):
            traj[-1]["ik_fail"] = True
            break
    return traj


def eval_grasp(robot, policy_by_method, spec):
    rng = np.random.default_rng(spec["seed"])
    t0 = time.time()
    for attempt in range(3):
        seed = spec["seed"] + 7919 * attempt
        robot.reset(seed, spec["pitch"], spec["grasp_pos"])
        H_pre, R_P, ins = establish_reference(robot)
        if H_pre is not None:
            break
    res = {**spec, "episode_seed": seed, "expert_ok": H_pre is not None, "episodes": []}
    if H_pre is None:
        return res
    stroke = robot.port_spec.pin_length_m + C.GAP0
    for s in range(spec["n_starts"]):
        q0 = sample_ood_start(rng)
        for method, policy in policy_by_method.items():
            ep = {"start": s, "method": method, "q0_cmd": q0.tolist()}
            try:
                if not robot.move_to(pose_from_state(q0, H_pre, R_P)):
                    ep["failure"] = "start_ik"
                    res["episodes"].append(ep)
                    continue
                obs = robot.observe()
                ep["u0_true"] = (relative_state(obs.H, H_pre, R_P) / C.SIGMA).tolist()
                if method != "none":
                    if method != "oracle":
                        policy.begin_episode(obs, rng)
                    ep["traj"] = run_loop(robot, policy, H_pre, R_P, method, rng)
                ep["u_final_true"] = (relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA).tolist()
                push = robot.push_along(R_P[:, 0], stroke, C.PUSH_STEP, C.PUSH_JAM_LAG, C.PUSH_SUCCESS_MARGIN)
                ep["push_success"] = push["success"]
                ep["advanced_mm"] = push["advanced_mm"]
            except Exception as exc:
                ep["failure"] = f"{type(exc).__name__}: {exc}"
                ep["traceback"] = traceback.format_exc()[-1500:]
            res["episodes"].append(ep)
            robot.move_to(H_pre)  # back to a safe pose between episodes
    res["seconds"] = time.time() - t0
    return res
