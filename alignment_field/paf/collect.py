"""Training data for one grasp ("config"): insert once, retreat, sweep, push.

Labels come only from forward kinematics (Back to the Manifold uses the
robot's own displacements the same way):
  * H*    hand pose at the end of one successful insertion (expert here,
          teleoperation on the real robot)
  * axis  direction of the hand motion during the insertion stroke
  * H_pre H* moved back along the axis by pin length + GAP0
  * q     relative_state(measured hand pose, H_pre, port axes)
ID vs OOD (SID): ID samples are drawn near H_pre, OOD samples from a wider,
partly disjoint box; push trials from near-ID states give the success label
that defines the in-distribution set M used by the field.
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np

from . import config as C
from .geometry import make_T, port_axes, pose_from_state, relative_state
from .demo import run_demo
from .perception import port_axes_from_obs, preprocess


def establish_reference(robot):
    """One insertion -> (H_pre, R_P, expert info). FK only."""
    ins = robot.expert_insert()
    if not ins["completed"]:
        return None, None, ins
    H_star = robot.hand_pose()
    stroke = [H for p, H in ins["poses"] if p == "axial_push"]
    axis = stroke[-1][:3, 3] - stroke[0][:3, 3]
    R_P = port_axes(axis)
    back = robot.port_spec.pin_length_m + C.GAP0
    H_pre = make_T(H_star[:3, :3], H_star[:3, 3] - back * R_P[:, 0])
    if not robot.move_to(H_pre):
        return None, None, {**ins, "failure": "retreat_ik"}
    return H_pre, R_P, ins


def _sample_q(rng, kind):
    if kind == "near":
        t = [rng.uniform(-C.NEAR_TX, 0), rng.uniform(-C.NEAR_T, C.NEAR_T), rng.uniform(-C.NEAR_T, C.NEAR_T)]
        r = rng.uniform(-C.NEAR_R, C.NEAR_R, 3)
    elif kind == "wide":
        t = [rng.uniform(-C.WIDE_TX, 0), rng.uniform(-C.WIDE_T, C.WIDE_T), rng.uniform(-C.WIDE_T, C.WIDE_T)]
        r = rng.uniform(-C.WIDE_R, C.WIDE_R, 3)
    else:  # push
        t = [0.0, rng.uniform(-C.PUSH_T, C.PUSH_T), rng.uniform(-C.PUSH_T, C.PUSH_T)]
        r = rng.uniform(-C.PUSH_R, C.PUSH_R, 3)
    return np.r_[t, r]


def collect_config(robot, spec: dict) -> dict:
    rng = np.random.default_rng(spec["seed"])
    t_start = time.time()
    # The expert occasionally hits the environment's 800-step limit; retry the
    # same grasp with a new episode seed (new hole offset), at most 3 times.
    for attempt in range(3):
        seed = spec["seed"] + 7919 * attempt
        info = robot.reset(seed, spec["pitch"], spec["grasp_pos"])
        hole = info["charger_v2"].get("hole_offset_yz_m") if "charger_v2" in info else None
        H_pre, R_P, ins = establish_reference(robot)
        if H_pre is not None:
            break
    meta = {**spec, "episode_seed": seed, "expert_attempts": attempt + 1, "expert_ok": ins["completed"],
            "expert_failure": ins.get("failure"), "expert_steps": ins["steps"],
            "hole_offset_yz_m_metadata_only": hole}
    if H_pre is None:
        return {"meta": meta}
    first = robot.observe()
    R_P_est, frac = port_axes_from_obs(first, rng)
    if R_P_est is None:
        meta["failure"] = "ransac"
        return {"meta": meta}
    meta["ransac_inlier_frac"] = frac
    meta["port_axis_error_deg"] = float(np.degrees(np.arccos(np.clip(R_P_est[:, 0] @ R_P[:, 0], -1, 1))))
    kinds = ["near"] * int(0.3 * spec["n_sweep"]) + ["wide"] * (spec["n_sweep"] - int(0.3 * spec["n_sweep"]))
    kinds += ["push"] * spec["n_push"]
    rng.shuffle(kinds)
    rec = {k: [] for k in ("rgb", "pc", "q", "q_cmd", "kind", "push", "advanced_mm")}
    raw = spec.get("store_raw", False)  # full RGB-D + camera/hand poses for the relational net
    if raw:
        rec.update({k: [] for k in ("rgb_full", "depth_full", "K", "cam_T_world", "H")})
    stroke = robot.port_spec.pin_length_m + C.GAP0
    for kind in kinds:
        q_cmd = _sample_q(rng, kind)
        if not robot.move_to(pose_from_state(q_cmd, H_pre, R_P)):
            continue
        obs = robot.observe()
        crop, pc = preprocess(obs, R_P_est, rng)
        rec["rgb"].append(crop)
        rec["pc"].append(pc.astype(np.float16))
        rec["q"].append(relative_state(obs.H, H_pre, R_P))
        rec["q_cmd"].append(q_cmd)
        rec["kind"].append({"near": 0, "wide": 1, "push": 2}[kind])
        if raw:
            rec["rgb_full"].append(obs.rgb[..., :3])
            rec["depth_full"].append(obs.depth_mm)
            rec["K"].append(obs.K.astype(np.float32))
            rec["cam_T_world"].append(obs.cam_T_world.astype(np.float32))
            rec["H"].append(obs.H.astype(np.float32))
        if kind == "push":
            res = robot.push_along(R_P[:, 0], stroke, C.PUSH_STEP, C.PUSH_JAM_LAG, C.PUSH_SUCCESS_MARGIN)
            rec["push"].append(int(res["success"]))
            rec["advanced_mm"].append(res["advanced_mm"])
        else:
            rec["push"].append(-1)
            rec["advanced_mm"].append(np.nan)
    # insertion demonstrations from near the pre-insertion pose (v4, learned insertion policy)
    demo = {k: [] for k in ("demo_rgb", "demo_pc", "demo_prop", "demo_act", "demo_u", "demo_id")}
    demo_ok = []
    for k in range(spec.get("n_demo", 0)):
        q0 = np.r_[rng.uniform(-0.004, 0.0), rng.uniform(-0.0015, 0.0015, 2), rng.uniform(-1, 1, 3) * np.radians(1.5)]
        res = run_demo(robot, H_pre, R_P, R_P_est, q0, rng)
        if res is None:
            continue
        fr, ok = res
        demo_ok.append(bool(ok))
        if ok and len(fr["act"]):
            for key in ("rgb", "pc", "prop", "act", "u"):
                demo["demo_" + key].append(fr[key])
            demo["demo_id"].append(np.full(len(fr["act"]), k))
        # back out straight along the axis before the next demonstration
        u = relative_state(robot.hand_pose(), H_pre, R_P)
        u[0] = -0.003
        robot.move_to(pose_from_state(u, H_pre, R_P))
    meta["demo_ok"] = demo_ok
    meta["n"] = len(rec["q"])
    meta["seconds"] = time.time() - t_start
    out = {"meta": meta, "H_pre": H_pre, "R_P": R_P, "R_P_est": R_P_est}
    out.update({k: np.asarray(v) for k, v in rec.items()})
    out.update({k: np.concatenate(v) for k, v in demo.items() if v})
    return out


def save(path, data):
    meta = data.pop("meta")
    np.savez_compressed(path, meta=json.dumps(meta, default=float), **data)
    return meta


def safe_collect(robot, spec):
    try:
        return collect_config(robot, spec)
    except Exception as exc:  # keep the worker alive; record the failure
        return {"meta": {**spec, "expert_ok": False, "failure": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-2000:]}}
