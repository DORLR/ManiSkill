"""Insertion demonstrations from Chongyu's scripted expert (stand-in for teleoperation).

Each demonstration is one full run of the expert in its "mixed" profile, which
perturbs the plug near the board and then recovers, aligns and pushes in. Only
the part near the board is recorded: phases perturbation_stochastic,
recovery_align and axial_push. Every STRIDE control ticks we store
  * the RGB crop (paf.perception.rgb_crop) and the 7 arm joint angles
  * the expert's commanded joint targets of the following ticks
The expert reads simulator truth to decide its motion (as a human teleoperator
sees the hole); nothing it reads is stored, only images, joint angles and its
joint commands (what a teleoperation log contains on a real robot).

IL label for a frame at tick t: an action chunk A_k = target[t + STRIDE*(k+1)] - q_t,
k = 0..H-1 (joint targets relative to the current joint angles, radians).
"""

from __future__ import annotations

import numpy as np

from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2 import (
    ContinuousScriptedPolicyConfig,
    RandomContinuousScriptedPolicy,
)

from .perception import rgb_crop

RECORD = ("perturbation_stochastic", "recovery_align", "axial_push")
STRIDE = 2
H_CHUNK = 8


def record_expert_demo(robot, seed, pitch, grasp_pos):
    robot.reset(seed, pitch, grasp_pos)
    targets, phases, frames = [], [], []

    def cb(phase, action, _info):
        targets.append(np.asarray(action[:7], np.float64).copy())
        phases.append(phase)
        t = len(targets) - 1
        if phase in RECORD and t % STRIDE == 0:
            obs = robot.observe()
            frames.append((t, rgb_crop(obs), obs.qpos[:7].astype(np.float64)))

    res = RandomContinuousScriptedPolicy(ContinuousScriptedPolicyConfig(profile="mixed")).run(robot.env, on_step=cb)
    if not res.get("completed") or not frames:
        return None, {"completed": bool(res.get("completed")), "failure": res.get("failure_reason")}
    T = np.asarray(targets)
    last = len(T) - 1
    rgb, qpos, act = [], [], []
    for t, crop, q in frames:
        idx = [min(t + STRIDE * (k + 1), last) for k in range(H_CHUNK)]
        rgb.append(crop)
        qpos.append(q.astype(np.float32))
        act.append((T[idx] - q[None]).astype(np.float32))
    return ({"rgb": np.asarray(rgb), "qpos": np.asarray(qpos), "act": np.asarray(act)},
            {"completed": True, "ticks": len(T), "frames": len(frames)})
