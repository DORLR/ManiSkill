"""Pinocchio forward / inverse kinematics of the Franka in the AutoMate scene (no Isaac import).

The model is franka_kinematics.urdf (joint origins of Isaac Sim's franka_description, which franka_mimic.usd is
built from; the match with the simulator is measured, scripts/automate/smoke_env.py "fk"). Joint vector q is the
9-vector of wm.env.robot.Robot.qpos(): 7 arm joints [rad] then the two finger joints [m].
All poses are 4x4 matrices in the robot base frame (panda_link0); Kinematics.base_T maps them to the world.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

URDF = Path(__file__).with_name("franka_kinematics.urdf")
LINKS = ("panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6",
         "panda_link7", "panda_link8", "panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_hand_tcp",
         "panda_fingertip_centered")


def _se3_to_T(M) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(M.rotation)
    T[:3, 3] = np.asarray(M.translation)
    return T


class _LinkPose:
    """What MPlib's PinocchioModel.get_link_pose returns: .p (3,) and .q (w, x, y, z)."""

    def __init__(self, T: np.ndarray):
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.p, self.q = T[:3, 3].copy(), np.array([w, x, y, z])


class Kinematics:
    def __init__(self, base_T: np.ndarray | None = None, urdf: Path = URDF):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.base_T = np.eye(4) if base_T is None else np.asarray(base_T, np.float64)
        self.frame_ids = {n: self.model.getFrameId(n) for n in LINKS}
        # Pinocchio's configuration order is the URDF tree order; ours is joint1..7, finger1, finger2.
        # (model.names is not used: inside Isaac Sim its std::vector<std::string> has no Python converter.)
        order = ["panda_joint%d" % k for k in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        jids = [self.model.getJointId(n) for n in order]
        assert self.model.nq == 9 and all(0 < j < self.model.njoints for j in jids), jids
        self.idx = np.array([self.model.joints[j].idx_q for j in jids])
        self.lower = np.asarray(self.model.lowerPositionLimit)[self.idx]
        self.upper = np.asarray(self.model.upperPositionLimit)[self.idx]
        self._last_q = None

    # ------------------------------------------------------------ FK
    def _q(self, q9) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self.idx] = np.asarray(q9, np.float64)[:9]
        return q

    def compute(self, q9) -> None:
        self.pin.framesForwardKinematics(self.model, self.data, self._q(q9))
        self._last_q = np.asarray(q9, np.float64)[:9].copy()

    def link(self, name: str, q9=None, world: bool = True) -> np.ndarray:
        """4x4 pose of `name` for q9 (or the last computed q), world frame unless world=False (base frame)."""
        if q9 is not None:
            self.compute(q9)
        T = _se3_to_T(self.data.oMf[self.frame_ids[name]])
        return self.base_T @ T if world else T

    def tcp(self, q9) -> np.ndarray:
        return self.link("panda_hand_tcp", q9)

    # ------------------------------------------------------------ IK
    def ik(self, T_world: np.ndarray, seed_q9, frame: str = "panda_hand_tcp", tol: float = 1e-6, iters: int = 200,
           damping: float = 1e-6, n_restarts: int = 0, rng: np.random.Generator | None = None,
           max_dev: float | None = None) -> np.ndarray | None:
        """Damped least squares on the 6-D pose error log(T(q)^-1 T_goal) (LOCAL frame Jacobian), arm joints only,
        clipped to the joint limits. Starts at seed_q9, then at n_restarts random configurations; returns the
        converged solution (|error| < tol, position m and rotation rad stacked) closest to the seed (max |dq_arm|),
        or None. Fingers are copied from the seed. `max_dev` refuses solutions further than that from the seed."""
        pin = self.pin
        Tg = np.linalg.inv(self.base_T) @ np.asarray(T_world, np.float64)
        goal = pin.SE3(Tg[:3, :3].copy(), Tg[:3, 3].copy())
        fid = self.frame_ids[frame]
        seed = np.asarray(seed_q9, np.float64)[:9]
        rng = rng or np.random.default_rng(0)
        starts = [seed] + [np.r_[rng.uniform(self.lower[:7], self.upper[:7]), seed[7:9]] for _ in range(n_restarts)]
        best = None
        arm = self.idx[:7]
        for q9 in starts:
            q = self._q(q9)
            for _ in range(iters):
                pin.framesForwardKinematics(self.model, self.data, q)
                err = pin.log6(self.data.oMf[fid].actInv(goal)).vector
                if np.linalg.norm(err) < tol:
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.LOCAL)[:, arm]
                dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
                q[arm] = np.clip(q[arm] + dq, self.lower[:7], self.upper[:7])
            else:
                pin.framesForwardKinematics(self.model, self.data, q)
                err = pin.log6(self.data.oMf[fid].actInv(goal)).vector
                if np.linalg.norm(err) >= tol:
                    continue
            sol = q[self.idx].copy()
            sol[7:9] = seed[7:9]
            dev = np.abs(sol[:7] - seed[:7]).max()
            if max_dev is not None and dev > max_dev:
                continue
            if best is None or dev < np.abs(best[:7] - seed[:7]).max():
                best = sol
            if dev < 1e-3:
                break
        return best


class PinocchioModelShim:
    """The two methods of MPlib's PinocchioModel that wm.data.kinematics.HandFK and the ManiSkill adapter call."""

    def __init__(self, kin: Kinematics, names: list[str]):
        self.kin, self.names = kin, names

    def compute_forward_kinematics(self, q9) -> None:
        self.kin.compute(q9)

    def get_link_pose(self, idx: int) -> _LinkPose:
        return _LinkPose(self.kin.link(self.names[idx], world=False))   # base frame, as MPlib
