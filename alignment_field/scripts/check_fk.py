"""Check that hand_pose() (FK of joint angles) equals the simulator's TCP pose.

python scripts/check_fk.py
"""
import numpy as np

from paf.env_adapter import Robot
from paf.geometry import pose_distance

r = Robot("A")
r.reset(seed=11, pitch_deg=-40.0, grasp_pos_m=-0.02)
errs = []
for k in range(5):
    q = r.qpos()
    q[:7] += np.random.default_rng(k).normal(size=7) * 0.05
    r.base.agent.robot.set_qpos(q[None].astype(np.float32))
    errs.append(pose_distance(r.hand_pose(), r.hand_pose_sim()))
print("max FK vs sim TCP: %.2e m, %.2e rad" % (max(e[0] for e in errs), max(e[1] for e in errs)))
