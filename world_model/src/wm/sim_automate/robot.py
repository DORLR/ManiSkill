"""Robot: the AutoMate scene behind the interface that wm.collect / wm.online use on wm.env.robot.Robot.

Same names, semantics, units and frames as wm.sim.env_adapter.Robot + wm.env.robot.Robot (ManiSkill):
    reset(seed, *, clamp_deg, dp_m, spin_deg, socket_index)  new episode; grasp = offsets of AutoMate's grasp
                    (grasp.py), keyword-only and without defaults (NOT ManiSkill's reset(seed, pitch_deg, grasp_pos_m):
                    there the second and third values are absolute pitch / position, here offsets, and a third offset,
                    the spin, exists; a call with positional grasp values raises TypeError); the grasp must lie in the
                    assembly's verified box (grasp.load_ranges) unless allow_outside=True (recorded). socket_index =
                    the reset number k of config.socket_pose (a new socket pose per k when config.SOCKET_DRAW is
                    "per_reset"; required there, ValueError without it; "per_grasp": default 0). reset_from_meta(g)
                    replays a recorded grasp / trajectory. A reset that cannot produce its state raises
                    checks.ResetError (a RuntimeError): no IK solution for the spawn, socket pose not taken.
    qpos() (9,) rad / m; get_qvel() (9,); hand_pose() 4x4 TCP pose from Pinocchio FK of qpos(), world frame
    observe() full 640 x 480 stream (Obs); frame() the 320 x 320 crop (Frame), depth uint16 in 0.1 mm
    _command(q7, n)  joint-position targets held for n control steps of 0.05 s (fingers closed)
    _ik(T, seed_q) -> (9,) or None, refusing branch jumps > 0.25 rad;  move_to(T, ...) as the ManiSkill adapter,
                    except that it returns False when the settle tolerance is not reached (ManiSkill: True once IK works)
                    and that "still" also holds when the FK hand pose moved < STILL_M / STILL_RAD over one control
                    step (Isaac's joint_vel can stay at a stale non-zero value, 2026-11-24)
    port_spec.pin_length_m (tight-fit depth), port_spec.clearance_m, max_decisions, lateral_frame(), grasp_ranges()
    base.get_state_dict() / base.set_state_dict(); base._planner() (FK / IK only), base.agent.robot.get_qvel(),
    base.agent.robot.pose, base.goal_pose, base.charger.pose, base.control_timestep, base.scene.update_render()
    _camera() / _twin: the camera internals wm.collect.branch.FullRecorder reads
Truth (evaluation only): truth_rel() = the 7-vector of wm.data.recorder._truth_rel (plug in the goal frame, x along
the insertion direction, 0 = seated, negative = outside), base.goal_pose / base.charger.pose in that convention.
Taught pose (simulation-free): H_star() = TCP pose with the plug seated, from the socket pose and the weld.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..sim.geometry import interpolate_T, pose_distance
from . import assets as A
from .checks import ResetError
from . import config as C
from . import frames
from . import grasp as G
from .kinematics import Kinematics, PinocchioModelShim, _LinkPose
from .scene import AutomateScene

R_G = C.R_G                    # goal / charger convention (config.R_G): x axis = insertion direction
STILL_M = 1e-6                 # move_to stillness by measured pose: < 1 um and < 0.001 deg in one control step (0.05 s);
                               # about 23 x stricter in translation and 6 x in rotation than 2e-3 rad/s on joint 6
                               # (axis to TCP about 0.23 m: 2e-3 rad/s ~ 23 um and 0.006 deg per control step)
STILL_RAD = float(np.radians(0.001))


def _trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


class Pose:
    """The part of sapien.Pose that our code uses: p, q (w, x, y, z), inv(), *, to_transformation_matrix()."""

    def __init__(self, T: np.ndarray):
        self.T = np.asarray(T, np.float64)
        x, y, z, w = Rotation.from_matrix(self.T[:3, :3]).as_quat()
        self.p, self.q = self.T[:3, 3].copy(), np.array([w, x, y, z])

    def inv(self) -> "Pose":
        return Pose(np.linalg.inv(self.T))

    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.T @ other.T)

    def to_transformation_matrix(self) -> np.ndarray:
        return self.T.copy()


@dataclass
class Obs:                     # as wm.sim.env_adapter.Obs
    rgb: np.ndarray            # (H, W, 3) uint8, full stream
    depth_mm: np.ndarray       # (H, W) int16 millimetres, 0 = no return / below the D405 minimum range
    K: np.ndarray
    cam_T_world: np.ndarray    # world -> camera (OpenCV), from FK and the hand-eye mount
    H: np.ndarray
    qpos: np.ndarray


@dataclass
class Frame:                   # as wm.env.robot.Frame
    rgb: np.ndarray            # (320, 320, 3) uint8
    depth: np.ndarray          # (320, 320) uint16, units of DEPTH_UNIT_M, 0 = no return
    seg: np.ndarray            # (320, 320) uint8 index into SEG_CLASSES
    K: np.ndarray              # intrinsics of the crop
    cam_T_world: np.ndarray
    qpos: np.ndarray
    hand: np.ndarray


# ---------------------------------------------------------------------------------------------- facades
class _PlannerShim:
    """FK / IK part of MPlib's Planner (as built by the ManiSkill task's _planner()). No motion planning."""

    def __init__(self, robot: "Robot"):
        self.robot = robot
        self.user_link_names = list(robot.kin.frame_ids)
        self.pinocchio_model = PinocchioModelShim(robot.kin, self.user_link_names)

    def _transform_goal_to_wrt_base(self, pose) -> "Pose":
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
        T[:3, 3] = pose.p
        return Pose(np.linalg.inv(self.robot.kin.base_T) @ T)

    def IK(self, goal_base, start_qpos, n_init_qpos: int = 20, threshold: float = 1e-3, return_closest: bool = False):
        T = self.robot.kin.base_T @ goal_base.to_transformation_matrix()
        sol = self.robot.kin.ik(T, start_qpos, tol=threshold, n_restarts=max(0, n_init_qpos - 1))
        return ("Success", sol) if sol is not None else ("IK Failed", None)

    def plan_pose(self, *a, **k) -> dict:
        return {"status": "plan_pose is not available in wm.sim_automate (no collision-aware planner)"}


class _ArticulationShim:
    def __init__(self, robot: "Robot"):
        self.robot = robot

    def get_qpos(self) -> np.ndarray:
        return self.robot.qpos()[None]

    def get_qvel(self) -> np.ndarray:
        return self.robot.get_qvel()[None]

    @property
    def pose(self):
        P = Pose(self.robot.kin.base_T)
        P.p, P.q = P.p[None], P.q[None]          # batched like ManiSkill's Pose (code indexes [0])
        return P


class _Agent:
    def __init__(self, robot: "Robot"):
        self.robot = _ArticulationShim(robot)
        self._r = robot

    @property
    def tcp(self):
        class _L:
            pass
        link = _L()
        link.pose = Pose(self._r.hand_pose_sim())
        return link


class _SceneShim:
    def update_render(self):
        pass                                  # the camera renders in _CamShim.capture()


class _Charger:
    def __init__(self, robot):
        self._r = robot

    @property
    def pose(self) -> Pose:
        return Pose(self._r.sc.plug_T(self._r.env_id) @ R_G)


class _Base:
    """The parts of the ManiSkill env (robot.base) that our code reads."""

    def __init__(self, robot: "Robot"):
        self._r = robot
        self.agent = _Agent(robot)
        self.scene = _SceneShim()
        self.charger = _Charger(robot)
        self.control_timestep = C.CONTROL_DT
        self.port_spec = robot.port_spec
        self._planner_shim = None

    def _planner(self):
        if self._planner_shim is None:
            self._planner_shim = _PlannerShim(self._r)
        return self._planner_shim

    @property
    def goal_pose(self) -> Pose:
        return Pose(self._r.sc.socket_T(self._r.env_id) @ R_G)

    def get_state_dict(self) -> dict:
        return self._r.get_state()

    def set_state_dict(self, state: dict):
        self._r.set_state(state)


class _Env:
    def __init__(self, robot: "Robot"):
        self.unwrapped = robot.base

    def close(self):
        pass                                  # the app is closed by the script (wm.sim_automate.app.shutdown)


class _CamShim:
    """ManiSkill Camera as wm.collect.branch.FullRecorder uses it: capture(), get_picture('Color')."""

    def __init__(self, robot: "Robot"):
        self._r = robot

    def capture(self):
        self._r._render()

    def get_picture(self, name: str) -> np.ndarray:
        if name != "Color":
            raise KeyError(name)
        rgb = self._r._last["rgb"]
        return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1)[None]


class _CompShim:
    """SAPIEN RenderCameraComponent: intrinsics of the full stream, OpenCV extrinsics (world -> camera, 3 x 4)."""

    def __init__(self, robot: "Robot"):
        self._r = robot

    def get_intrinsic_matrix(self) -> np.ndarray:
        return C.intrinsics()

    def get_extrinsic_matrix(self) -> np.ndarray:
        return self._r.cam_T_world()[:3, :]


class _TwinShim:
    """The float-depth twin camera: get_picture('Position')[..., 2] = -depth (OpenGL camera axes), 0 = no return."""

    def __init__(self, robot: "Robot"):
        self._r = robot

    def take_picture(self):
        pass                                  # rendered together with the colour in _CamShim.capture()

    def get_picture(self, name: str) -> np.ndarray:
        if name != "Position":
            raise KeyError(name)
        d = self._r._last["depth"]
        K = C.intrinsics()
        v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
        y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
        return np.stack([x, -y, -d, (d > 0).astype(np.float64)], -1).astype(np.float32)


def grasp_kwargs(v) -> dict:
    """(clamp deg, dp m, spin deg) -> the keyword arguments of Robot.reset."""
    clamp_deg, dp_m, spin_deg = (float(x) for x in v)
    return {"clamp_deg": clamp_deg, "dp_m": dp_m, "spin_deg": spin_deg}


# ---------------------------------------------------------------------------------------------- robot
class Robot:
    def __init__(self, port_type: str, seed: int = 10043, num_envs: int = 1, camera: bool = True, seg: bool = True,
                 mount: str | None = None, device: str | None = None, weld: str | None = None):
        # One env only: every method acts on env_id 0, and an "articulation" weld restart puts every env's socket back
        # to its initial pose while reset re-places only env_id.
        if num_envs != 1:
            raise ValueError(f"wm.sim_automate.robot.Robot supports num_envs = 1 only (got {num_envs})")
        self.port_type = port_type
        self.assembly_id = C.assembly_of(port_type)
        self.port_spec = A.load_spec(self.assembly_id)
        self.mount = mount or C.SOCKET_MOUNT
        T0 = C.T_HAND_TCP @ G.T_hp0(self.assembly_id)                 # weld built at the nominal grasp (no restart)
        self.sc = AutomateScene(self.assembly_id, num_envs, camera, seg, self.mount, device or C.PHYSICS_DEVICE,
                                weld or C.WELD_MODE, T0)
        self.env_id = 0
        self.kin = Kinematics(base_T=self.sc.robot_base_T(self.env_id))
        self.T_hp0 = G.T_hp0(self.assembly_id)                         # AutoMate's grasp, plug in the TCP frame
        self.T_tcp_plug = self.T_hp0.copy()
        self._fingers = G.FingerSocketCheck(self.assembly_id)
        self.q_finger, self.finger_note = self._fingers.closing([0.0, 0.0, 0.0])
        self.base = _Base(self)
        self.env = _Env(self)
        self.planner = self.base._planner()
        self._cam, self._comp, self._twin = _CamShim(self), _CompShim(self), _TwinShim(self)
        self._last = None
        self._seg_lut = {}
        self.episode = None
        self._q_cmd = None

    # ------------------------------------------------------------------ grasp (wm.sim_automate.grasp)
    def in_hand(self, clamp_deg: float = 0.0, dp_m: float = 0.0, spin_deg: float = 0.0) -> np.ndarray:
        """Plug pose in the TCP frame: T_hp = R_y(dtheta) T_hp0 Trans_z(dp) T_c R_z(dpsi) T_c^-1 (grasp.py; dp > 0 =
        fingers closer to the plug tip; T_c = Trans(grasp.spin_axis_xy): the plug axis or the CAD origin, switch
        config.GRASP_SPIN_AXIS)."""
        return G.T_hp(self.T_hp0, float(clamp_deg), float(dp_m), float(spin_deg), axis_xy=G.spin_axis_xy(self.port_spec))

    def axis(self) -> np.ndarray:
        """Insertion direction (unit, world): -z of the socket."""
        return -self.sc.socket_T(self.env_id)[:3, 2]

    def lateral_frame(self) -> np.ndarray:
        """Columns [a, -e2, e1] of the approved lateral-axes rule (frames.py), robot base x as the fallback."""
        return frames.frame(self.axis(), self.kin.base_T[:3, 0])

    def H_star(self) -> np.ndarray:
        """TCP pose with the plug seated: H* = T_socket * T_hp^-1 (seated plug pose = socket pose)."""
        return self.sc.socket_T(self.env_id) @ np.linalg.inv(self.T_tcp_plug)

    @property
    def max_decisions(self) -> int:
        """N_max = 150 + ceil(max(0, L - 20 mm) / 0.5 mm) decisions of 0.4 s."""
        return self.port_spec.max_decisions

    def grasp_ranges(self) -> dict:
        """Per-assembly ranges, held-out intervals and the normalisation of the three grasp variables."""
        return G.load_ranges(self.assembly_id)

    # ------------------------------------------------------------------ reset
    def check_grasp_in_box(self, clamp_deg: float, dp_m: float, spin_deg: float, tol: float = 1e-6) -> dict | None:
        """None if (clamp_deg, dp_m, spin_deg) lies in the assembly's verified box (grasp.load_ranges, within tol),
        else the box and the offending values (also when the assembly has no box)."""
        try:
            box = G.load_ranges(self.assembly_id).get("box")
        except FileNotFoundError as exc:
            return {"box": None, "reason": str(exc)}
        if box is None:
            return {"box": None, "reason": "no grasp box for this assembly"}
        v = {"dtheta_deg": clamp_deg, "dp_m": dp_m, "dpsi_deg": spin_deg}
        bad = {k: x for k, x in v.items() if not (box[k][0] - tol <= float(x) <= box[k][1] + tol)}
        return {"box": box, "outside": bad} if bad else None

    def reset(self, seed: int, *, clamp_deg: float, dp_m: float, spin_deg: float, socket_index: int | None = None,
              allow_outside: bool = False, gap_m: float = C.SPAWN_GAP_M) -> dict:
        """New episode with grasp (dtheta, dp, dpsi) = (clamp_deg, dp_m, spin_deg), offsets of AutoMate's grasp. The
        grasp is keyword-only, without defaults: ManiSkill code calls reset(seed, pitch_deg, grasp_pos_m) with absolute
        values and no spin, and such a call must fail (TypeError) rather than be read as offsets with spin 0. It must
        lie in the assembly's verified grasp box (ValueError otherwise) unless allow_outside=True, which is recorded
        in the episode (milestone-1b checks and smoke tests of grasps outside the box).
        The socket is placed at pose number socket_index of the seed (config.socket_pose: AutoMate's +-5 cm, yaw
        0-10 deg; "per_reset": a new pose per index, "per_grasp": the index is ignored), written after the weld so
        that a weld restart cannot undo it, and checked against PhysX after the settling steps. H_star(), axis(),
        lateral_frame(), the goal pose and the expert read the socket pose from the simulator, so they follow it.
        The plug is placed on the socket axis, aligned, its lowest point gap_m above the highest socket point
        (L_top + gap_m above seated); fingers close for SETTLE_STEPS control steps. Deterministic given the seed and
        socket_index. socket_index is required under "per_reset" (ValueError without it: the pose must never be
        guessed); under "per_grasp" it defaults to 0."""
        if socket_index is None:
            if C.SOCKET_DRAW == "per_reset":
                raise ValueError("reset: socket_index is required with config.SOCKET_DRAW 'per_reset' (pose number k of "
                                 "the seed); pass socket_index=0 for the first pose of a seed")
            socket_index = 0
        outside = self.check_grasp_in_box(clamp_deg, dp_m, spin_deg)
        if outside is not None and not allow_outside:
            raise ValueError(f"reset: grasp ({clamp_deg}, {dp_m}, {spin_deg}) of {self.assembly_id} is not in the "
                             f"verified box: {outside}; pass allow_outside=True for a deliberate check outside it")
        i = self.env_id
        self.T_tcp_plug = self.in_hand(clamp_deg, dp_m, spin_deg)
        T_hp = C.T_HAND_TCP @ self.T_tcp_plug                   # plug in the panda_hand frame (weld)
        restarted = self.sc.set_weld(i, T_hp)        # "articulation" weld: a new grasp restarts the simulation
        T_socket, socket_draw = C.socket_pose(seed, self.mount, socket_index)
        self.sc.set_socket_pose(i, T_socket)          # after the weld: a restart puts the socket back to its initial pose
        lift = max(self.port_spec.depth_top_m, self.port_spec.pin_length_m) + gap_m
        plug0 = self.sc.socket_T(i) @ _trans([0.0, 0.0, lift])
        tcp0 = plug0 @ np.linalg.inv(self.T_tcp_plug)
        q = self.reset_ik(tcp0)
        if q is None:
            raise ResetError(f"reset: no IK solution for the start pose of {self.assembly_id}")
        # The plug is a link of the robot articulation and Franka self-collisions are off (AutoMate), so fingers
        # driven to 0 would pass through the plug (measured: q = 0 at H* for 00015 / 00293, and finger-socket contact
        # for sleeve-like assemblies). The fingers are held at the opening where they touch the plug instead
        # (grasp.FingerSocketCheck.closing, PhysX collision shapes of franka_mimic.usd); the weld carries the load.
        self.q_finger, self.finger_note = self._fingers.closing([float(clamp_deg), float(dp_m), float(spin_deg)])
        q[7:9] = self.q_finger
        plug_T = self.kin.link("panda_hand", q) @ T_hp
        targets = np.r_[q[:7], self.q_finger, self.q_finger]
        self.sc.write_state(i, q, plug_T, targets)
        self._q_cmd = q[:7].copy()
        self.sc.step(C.SETTLE_STEPS)
        socket_check = self.sc.check_socket_pose(i, T_socket)     # a PhysX read now (raises on a failed teleport)
        self.episode = {"seed": int(seed), "clamp_deg": float(clamp_deg), "dp_m": float(dp_m),
                        "spin_deg": float(spin_deg), "assembly_id": self.assembly_id, "mount": self.mount,
                        "spawn_lift_m": lift, "socket_index": int(socket_index), "socket_pose": T_socket.tolist(),
                        "socket_draw": socket_draw, "socket_check": socket_check, "switches": C.switches(),
                        "grasp_outside_box": outside if allow_outside else None,
                        "finger_opening_m": self.q_finger, "finger_note": self.finger_note,
                        "T_tcp_plug": self.T_tcp_plug.tolist(), "sim_restarted": bool(restarted)}
        self.planner = self.base._planner()
        return {"automate": dict(self.episode), "H_star": self.H_star().tolist(), "axis": self.axis().tolist(),
                "max_decisions": self.max_decisions}

    def reset_from_meta(self, g: dict, socket_index: int | None = None) -> dict:
        """reset() with the grasp of a grasp.json / trajectory meta: episode_seed, pitch_deg (= clamp angle offset),
        grasp_pos_m (= offset along the plug axis), spin_deg, and the socket pose number: g["socket_index"] if the
        meta has it, else the argument socket_index. With config.SOCKET_DRAW "per_reset" one of the two is required
        (KeyError otherwise: the socket pose would be guessed); with "per_grasp" it defaults to 0 (same pose for
        every index). A meta without spin_deg (ManiSkill) raises KeyError."""
        if g.get("simulator", "automate") != "automate":
            raise ValueError(f"grasp {g.get('grasp')} is from simulator {g.get('simulator')!r}, not automate")
        k = g.get("socket_index", socket_index)
        if k is None:
            if C.SOCKET_DRAW == "per_reset":
                raise KeyError(f"grasp {g.get('grasp')}: no socket_index in the meta or the call (SOCKET_DRAW per_reset)")
            k = 0
        return self.reset(int(g["episode_seed"]), clamp_deg=float(g["pitch_deg"]), dp_m=float(g["grasp_pos_m"]),
                          spin_deg=float(g["spin_deg"]), socket_index=int(k))

    def reset_ik(self, T: np.ndarray) -> np.ndarray | None:
        """IK from config.RESET_SEEDS (then 40 random starts if none converges); the solution with the largest
        distance to the joint limits. Deterministic."""
        best, margin = None, -np.inf
        for seed in C.RESET_SEEDS:
            q = self.kin.ik(T, np.r_[seed, 0.04, 0.04])
            if q is not None:
                m = float(np.min(np.minimum(q[:7] - self.kin.lower[:7], self.kin.upper[:7] - q[:7])))
                if m > margin:
                    best, margin = q, m
        if best is None:
            best = self.kin.ik(T, np.r_[C.Q_NOMINAL, 0.04, 0.04], n_restarts=40, rng=np.random.default_rng(0))
        return best

    # ------------------------------------------------------------------ proprioception / FK
    def qpos(self) -> np.ndarray:
        return self.sc.joint_pos()[self.env_id]

    def get_qvel(self) -> np.ndarray:
        return self.sc.joint_vel()[self.env_id]

    def hand_pose(self) -> np.ndarray:
        """TCP (panda_hand + 0.1034 m along z) from Pinocchio FK of the measured joints, world frame."""
        return self.kin.tcp(self.qpos())

    def hand_pose_sim(self) -> np.ndarray:
        """Simulator panda_hand body pose * T_hand_tcp; only for checking hand_pose()."""
        return self.sc.body_T("panda_hand", self.env_id) @ C.T_HAND_TCP

    def cam_T_world(self, q9=None) -> np.ndarray:
        """world -> camera (OpenCV axes) from FK and the hand-eye mount."""
        q = self.qpos() if q9 is None else q9
        return np.linalg.inv(self.kin.link("panda_hand", q) @ C.T_HAND_CAM_CV)

    # ------------------------------------------------------------------ truth (evaluation only)
    def truth_rel(self) -> np.ndarray:
        """Plug in the goal frame, (x, y, z, qw, qx, qy, qz), float32, as wm.data.recorder._truth_rel."""
        rel = (self.base.goal_pose.inv() * self.base.charger.pose)
        return np.r_[rel.p, rel.q].astype(np.float32)

    # ------------------------------------------------------------------ camera
    def _render(self):
        T_cam = self.kin.link("panda_hand", self.qpos()) @ C.T_HAND_CAMLINK     # FK + mount, SAPIEN/"world" axes
        self._last = self.sc.render_camera(self.env_id, T_cam)
        return self._last

    def _camera(self):
        return self._cam, self._comp

    def _seg_classes(self, ids: np.ndarray, labels: dict) -> np.ndarray:
        lut = {}
        for k, path in labels.items():
            p = str(path)
            lut[int(k)] = 1 if "/HeldAsset" in p else 2 if "/Robot" in p else 3 if "/FixedAsset" in p else \
                4 if ("/Table" in p or "ground" in p.lower()) else 0
        out = np.zeros(ids.shape, np.uint8)
        for k, c in lut.items():
            if c:
                out[ids == k] = c
        return out

    def observe(self) -> Obs:
        f = self._render()
        depth_mm = np.round(f["depth"] * 1000.0)
        depth_mm[f["depth"] < C.DEPTH_MIN_M] = 0
        return Obs(rgb=f["rgb"], depth_mm=np.clip(depth_mm, 0, 32767).astype(np.int16), K=C.intrinsics(),
                   cam_T_world=self.cam_T_world(), H=self.hand_pose(), qpos=self.qpos())

    def frame(self) -> Frame:
        f = self._render()
        r0, r1, c0, c1 = C.CROP
        depth = f["depth"].copy()
        depth[depth < C.DEPTH_MIN_M] = 0.0
        d = np.clip(np.round(depth[r0:r1, c0:c1] / C.DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
        K = C.intrinsics()
        K[0, 2] -= c0
        K[1, 2] -= r0
        seg = self._seg_classes(f["seg_ids"][r0:r1, c0:c1], f["seg_labels"]) if "seg_ids" in f else \
            np.zeros(d.shape, np.uint8)
        return Frame(rgb=np.ascontiguousarray(f["rgb"][r0:r1, c0:c1]), depth=d, seg=seg, K=K,
                     cam_T_world=self.cam_T_world(), qpos=self.qpos(), hand=self.hand_pose())

    # ------------------------------------------------------------------ motion
    def _ik(self, T: np.ndarray, seed_q: np.ndarray) -> np.ndarray | None:
        """As wm.sim.env_adapter.Robot._ik: seed first, widen the search only if needed, closest branch, refuse
        jumps > 0.25 rad. Tolerances are on |log(T^-1 T_goal)| (m and rad stacked)."""
        seed_q = np.asarray(seed_q, np.float64)
        sol = None
        for n_restarts, tol in ((0, 1e-5), (19, 1e-5), (19, 1e-4), (49, 1e-3)):
            sol = self.kin.ik(T, seed_q, tol=tol, n_restarts=n_restarts, rng=np.random.default_rng(n_restarts))
            if sol is not None:
                break
        if sol is None or np.abs(sol[:7] - seed_q[:7]).max() > 0.25:
            return None
        return sol

    def _command(self, q7: np.ndarray, n: int = 1):
        """Hold joint targets q7 (rad) for n control steps (0.05 s each), fingers held on the plug."""
        q7 = np.asarray(q7, np.float64)[:7]
        t = np.tile(np.r_[q7, self.q_finger, self.q_finger], (self.sc.num_envs, 1))
        self.sc.set_targets(t)
        self._q_cmd = q7.copy()
        self.sc.step(n)

    def move_to(self, T_goal: np.ndarray, max_lin: float = 0.002, max_ang: float = np.radians(1.0),
                settle_steps: int = 40, tol_m: float = 1e-4, tol_rad: float = np.radians(0.05)) -> bool:
        """Procedure of wm.sim.env_adapter.Robot.move_to with two differences named below (waypoints of <= 2 mm /
        1 deg, 2 control steps each,
        then settle until within tolerance and still). False if IK fails or if the hand is not within tol_m / tol_rad
        and still after settle_steps (the ManiSkill adapter returned True in that case).
        Still = the reported joint speed max|qvel[:7]| < 2e-3 rad/s (the ManiSkill rule) OR the measured hand pose
        moved less than STILL_M and STILL_RAD over the last control step. The second test was added on 2026-11-24:
        in the CPU batch batch_r3_cpu the arm stops (hand pose unchanged to 0.1 um over 30+ control steps) while
        Isaac's joint_vel keeps reporting a constant, stale 2.4e-3 rad/s on one joint, so the speed rule alone never
        passed (move_ok 1 / 100, placed 0 / 100, against 98-99 / 100 in the 2026-11-21 batches batch / batch2).
        The same stale reading (identical 2.40e-3 rad/s, pose frozen) occurs with GPU PhysX (00015, 2026-11-24), so it
        is not CPU sleeping; what made the 2026-11-21 batches settle is not established."""
        start = self.hand_pose()
        dt, dr = pose_distance(start, T_goal)
        n = max(1, int(np.ceil(max(dt / max_lin, dr / max_ang))))
        seed_q = self.qpos()
        targets = []
        for k in range(1, n + 1):
            sol = self._ik(interpolate_T(start, T_goal, k / n), seed_q)
            if sol is None:
                return False
            targets.append(sol)
            seed_q = sol
        for sol in targets:
            self._command(sol, 2)
        prev = self.hand_pose()
        for _ in range(settle_steps):
            self._command(targets[-1], 1)
            now = self.hand_pose()
            dt, dr = pose_distance(now, T_goal)
            if dt < tol_m and dr < tol_rad and self._still(now, prev):
                return True
            prev = now
        return False

    def _still(self, now: np.ndarray, prev: np.ndarray) -> bool:
        """move_to's stillness test (docstring there): reported joint speed < 2e-3 rad/s, or the hand pose moved less
        than STILL_M / STILL_RAD between two consecutive control steps."""
        if np.abs(self.get_qvel()[:7]).max() < 2e-3:
            return True
        ds, da = pose_distance(now, prev)
        return bool(ds < STILL_M and da < STILL_RAD)

    # ------------------------------------------------------------------ save / restore
    def get_state(self) -> dict:
        """Scene, PD targets, last command, episode and the grasp (plug in the TCP frame, finger opening): a state
        restored from another grasp brings its own H_star(), finger targets and expert geometry with it."""
        return {"sim": self.sc.get_state(), "q_cmd": None if self._q_cmd is None else self._q_cmd.copy(),
                "episode": None if self.episode is None else dict(self.episode),
                "T_tcp_plug": self.T_tcp_plug.copy(), "q_finger": float(self.q_finger), "finger_note": self.finger_note}

    def set_state(self, st: dict):
        self.sc.set_state(st["sim"])
        self._q_cmd = None if st["q_cmd"] is None else st["q_cmd"].copy()
        self.episode = st["episode"]
        self.T_tcp_plug = np.array(st["T_tcp_plug"], np.float64)
        self.q_finger, self.finger_note = float(st["q_finger"]), st["finger_note"]
        if not np.allclose(self.sc.T_hand_plug[self.env_id], C.T_HAND_TCP @ self.T_tcp_plug, atol=1e-9):
            raise RuntimeError("set_state: the restored weld does not hold the restored grasp")
