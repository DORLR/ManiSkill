"""Diagnostic: where does the closed-loop handoff error come from? (point servo, A-D)
Per step: estimation error (u_est - u_true), tracking error (reached vs commanded), final error."""
import numpy as np
from paf import config as C
from paf.env_adapter import Robot
from paf.collect import establish_reference
from paf.evaluate import run_loop, sample_ood_start, POINT_TARGET
from paf.policy import FieldPolicy
from paf.geometry import pose_from_state, relative_state
fp = FieldPolicy("runs_v4/fieldrel_sid", estimate="rel")
rows = []
for port in "ABCD":
    robot = Robot(port)
    for i in range(2):
        rng = np.random.default_rng(777000 + 10 * ord(port) + i)
        p, g = C.sample_train_grasp(rng)
        robot.reset(777000 + i, p, g)
        H_pre, R_P, _ = establish_reference(robot)
        if H_pre is None:
            continue
        fp.begin_episode(robot.observe(), rng)
        ax_err = float(np.degrees(np.arccos(np.clip(fp.R_P[:, 0] @ R_P[:, 0], -1, 1))))
        for s in range(3):
            if not robot.move_to(pose_from_state(sample_ood_start(rng), H_pre, R_P)):
                continue
            tr = run_loop(robot, fp, H_pre, R_P, "point", rng)
            ut = np.array([t["u_true"] for t in tr]); ue = np.array([t["u_est"] for t in tr])
            du = POINT_TARGET - ue; n = np.linalg.norm(du, axis=1, keepdims=True); du = du * np.minimum(1, 3 / n)
            track = ut[1:] - (ut[:-1] + du[:-1])
            final = relative_state(robot.hand_pose(), H_pre, R_P) / C.SIGMA
            late = slice(-8, None)
            r = dict(port=port, axis_err_deg=ax_err, iters=len(tr),
                     est_lat_err_late=float(np.median(np.linalg.norm((ue - ut)[late, 1:3], axis=1))),
                     est_lat_err_first=float(np.linalg.norm((ue - ut)[0, 1:3])),
                     est_bias_late=np.round((ue - ut)[late, 1:3].mean(0), 2).tolist(),
                     track_lat_late=float(np.median(np.linalg.norm(track[late, 1:3], axis=1))) if len(track) else None,
                     step_late=float(np.median(np.linalg.norm(du[late], axis=1))),
                     final_lat=float(np.linalg.norm(final[1:3])), final_rot=float(np.linalg.norm(final[3:])))
            rows.append(r); print(r, flush=True)
    robot.env.close()
for k in ("axis_err_deg", "iters", "est_lat_err_late", "est_lat_err_first", "track_lat_late", "step_late", "final_lat", "final_rot"):
    print("MEDIAN", k, round(float(np.median([r[k] for r in rows if r[k] is not None])), 2))
