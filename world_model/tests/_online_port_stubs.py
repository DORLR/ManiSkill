"""Stand-ins for the AutoMate port tests (tests/test_online_port_*.py): a deterministic toy arm (forward kinematics,
first-order joint tracking, a planner shim for wm.data.kinematics.HandFK) and a deterministic stand-in for
wm.online.agent320.WMAgent (numpy only). Nothing here is a model of either simulator: it only has to drive every branch
of the attempt loop the same way twice."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

Q0 = np.array([0.1, -0.3, 0.2, -1.9, 0.1, 1.7, 0.6, 0.02, 0.02])


def toy_fk_local(q9: np.ndarray) -> np.ndarray:
    """4x4 TCP pose (robot base frame) of a smooth, full-rank toy arm."""
    q = np.asarray(q9, np.float64)[:7]
    T = np.eye(4)
    T[:3, 3] = [0.45 + 0.12 * np.sin(q[0]) + 0.05 * q[1] + 0.01 * q[6],
                0.05 + 0.10 * np.cos(q[2]) + 0.04 * q[3] - 0.02 * q[0],
                0.30 + 0.08 * np.sin(q[4]) + 0.03 * q[5] + 0.015 * q[1]]
    T[:3, :3] = Rotation.from_rotvec([0.4 * q[5] - 0.6, 0.3 * q[6] + 0.1 * q[2], 0.2 * q[0] + 0.05 * q[3]]).as_matrix()
    return T


class _LinkPose:
    def __init__(self, T):
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.p, self.q = T[:3, 3].copy(), np.array([w, x, y, z])


class PinShim:
    """compute_forward_kinematics / get_link_pose of MPlib's PinocchioModel (as wm.data.kinematics.HandFK uses it)."""

    def __init__(self):
        self.T = None

    def compute_forward_kinematics(self, q9):
        self.T = toy_fk_local(q9)

    def get_link_pose(self, idx):
        return _LinkPose(self.T)


class PlannerShim:
    def __init__(self):
        self.pinocchio_model = PinShim()
        self.user_link_names = ["panda_link0", "panda_hand_tcp"]


class BasePose:
    def __init__(self, T):
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        self.p, self.q = T[:3, 3][None].copy(), np.array([[w, x, y, z]])


BASE_T = np.eye(4)
BASE_T[:3, 3] = [-0.6, 0.0, 0.0]


class ToyArm:
    """Joints, joint tracking, FK hand, state save / restore. `pull`: a joint-space drift per control step that the
    commands have to fight (so that the hand also stalls and the loop's back-off runs)."""

    def __init__(self, pin_length_m: float = 0.016):
        self.q = Q0.copy()
        self.v = np.zeros(9)
        self.steps = 0
        self.L = pin_length_m

    def qpos(self) -> np.ndarray:
        return self.q.copy()

    def hand_pose(self) -> np.ndarray:
        return BASE_T @ toy_fk_local(self.q)

    def _command(self, q7, n: int = 1):
        q7 = np.asarray(q7, np.float64)[:7]
        for _ in range(int(n)):
            old = self.q.copy()
            d = q7 - self.q[:7]
            self.q[:7] = self.q[:7] + 0.6 * d - 0.002 * np.tanh(50.0 * d)        # lag and a little stiction
            self.v = (self.q - old) / 0.05
            self.steps += 1

    def get_state(self) -> dict:
        return {"q": self.q.copy(), "v": self.v.copy(), "steps": self.steps}

    def set_state(self, s: dict):
        self.q, self.v, self.steps = s["q"].copy(), s["v"].copy(), s["steps"]

    def picture(self, h: int = 24, w: int = 24):
        """A deterministic picture of the joints: rgb (h, w, 3) uint8, depth (h, w) float [m]."""
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        p = self.hand_pose()[:3, 3]
        base = 127 + 120 * np.sin(xx * 0.3 + 40 * p[0]) * np.cos(yy * 0.25 + 40 * p[1])
        rgb = np.stack([base, np.roll(base, 3, 0), np.roll(base, 5, 1)], -1)
        depth = 0.12 + 0.03 * np.sin(xx * 0.2 + 30 * p[2]) + 0.0003 * yy
        depth[0, 0] = 0.05                                                       # below the D405 minimum range
        return np.clip(rgb, 0, 255).astype(np.uint8), depth


def truth7(arm: ToyArm, H_goal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """A truth 7-vector (plug in the goal frame, x along the insertion axis, 0 = seated) of the toy arm."""
    from wm.tdmpc.wmdata320 import _basis
    B = _basis(np.asarray(axis, np.float64))
    e = B @ (arm.hand_pose()[:3, 3] - H_goal[:3, 3])
    x, y, z, w = Rotation.from_matrix(H_goal[:3, :3].T @ arm.hand_pose()[:3, :3]).as_quat()
    return np.r_[e, w, x, y, z].astype(np.float32)


class StubAgent:
    """agent320.WMAgent's interface, deterministic in its inputs and its call count. kind:
        "stall"    proposals shrink to nothing after 12 decisions (the hand stops: stall, back-off, re-position)
        "seat"     the done head says seated from the 9th decision on
    offset read-out: gap = 7 - 0.6 x decisions [mm] (the in-hole rule fires), lateral from the picture."""

    def __init__(self, kind: str = "stall", offset_mode: str = "gap"):
        self.kind, self.offset_mode = kind, offset_mode
        self.gamma, self.shaping, self.horizon, self.step = 0.967, 0.3, 2, 4321
        self.n = 0

    def encode(self, rgb, depth, q):
        self.n += 1
        return np.r_[float(np.asarray(rgb, np.float64).mean()) / 255.0, float(np.asarray(depth, np.float64).mean()) / 1e4,
                     np.asarray(q, np.float64)[:7], self.n]

    def propose(self, z):
        s = 0.004 if (self.kind != "stall" or self.n < 12) else 1e-7
        k = np.arange(16)[:, None] + np.arange(7)[None]
        return s * np.sin(0.7 * k + z[0] * 3.0 + z[2]) * (1 + 0.05 * k / 16)

    def done(self, z):
        if self.kind == "seat":
            return 0.9 if self.n >= 9 else 0.2
        return float(0.25 + 0.2 * np.sin(z[0] * 10.0))

    def values(self, z):
        return np.array([np.sin(z[1] * 100.0), np.cos(z[0] * 7.0)])

    def offset_mm(self, z):
        return np.array([7.0 - 0.6 * self.n, 1.2 * np.sin(z[0] * 20.0), 0.9 * np.cos(z[1] * 50.0), 1.0])

    def gap_mm(self, z, pin_length_mm=None):
        if self.offset_mode == "gap":
            if pin_length_mm is not None:
                raise ValueError("zero-privilege")
            return float(self.offset_mm(z)[0])
        return float(-self.offset_mm(z)[0] - pin_length_mm)

    def plan_mppi(self, z, Wq, prev_mean, eval_mode, noise=False):
        """agent320.WMAgent.plan_mppi's interface, deterministic: u0 from the latent, mean shifted as TD-MPC2's."""
        u0 = np.clip(np.array([np.sin(z[0] * 5.0), np.cos(z[1] * 30.0), 0.3]), -1, 1)
        noisy = bool(noise) and not eval_mode
        mean = np.tile(u0, (3, 1)) if prev_mean is None else np.vstack([prev_mean[1:], u0])
        return {"u0": u0, "dq": np.asarray(Wq, np.float64) @ u0, "u_seq": np.tile(u0, (3, 1)), "move_hand_mm": u0 * 1.5, "mean": mean,
                "std0": np.full(3, 0.5), "value_max": 1.0, "value_pi": 0.5, "noisy": noisy}

    def mppi_gate(self, z, u_seq, Wq, margin):
        """agent320.WMAgent.mppi_gate's interface, deterministic: both outcomes occur along a run."""
        adv = np.array([0.4, 0.9, 1.1, 0.8, 0.7]) * (1.0 if self.n % 3 else -1.0)
        return {"ok": bool(adv.mean() > margin and adv.min() > 0), "adv": adv}

    def score(self, z, deltas, parts: bool = False):
        d = np.asarray(deltas, np.float64)
        a = d.reshape(len(d), -1)
        base = -20.0 * (a ** 2).sum(1) + 200.0 * (a @ np.sin(np.arange(a.shape[1]) + z[0]))
        tot = np.stack([base, base * 0.9 + 0.01], 1)
        return (tot, base * 0.5) if parts else tot
