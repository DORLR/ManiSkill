"""The AutoMate scene in Isaac Lab, changed only where our experiment needs it. Import after app.launch().

Kept from AutoMate (Isaac Lab v2.3.2 direct/automate, imported, never edited): the PhysX settings
(AssemblyEnvCfg.sim: TGS, 192 position / 1 velocity iterations, bounce 0.2, friction offset / correlation, one GPU
partition, default material friction 1.0), the robot USD franka_mimic.usd with its rigid-body / articulation /
collision properties, the gripper actuator, the plug (held asset) and socket (fixed asset) USDs with their SDF
collisions, masses and properties (socket: articulation with a fixed root), the table, ground and dome light, and the
two adjustments AssemblyEnv makes after start-up (+0.01 kg m^2 on every robot body inertia diagonal, friction 0.75 on
robot, plug and socket).

Changed (milestone 1 brief):
  * plug welded to panda_hand by a PhysX fixed joint (UsdPhysics.FixedJoint, excluded from the robot articulation so
    that its frame can be set per episode: measured, PhysX refuses runtime frame changes of articulation joints);
    the fingers are held where they touch the plug (AutoMate's gripper gains; the plug is an articulation link and
    self-collisions are off, so a target of 0 would close them through the plug); plug gravity stays disabled as in
    AutoMate.
  * arm: joint-position PD, ManiSkill Panda gains (config.ARM_*), 20 Hz control = 6 physics steps of 1/120 s;
    AutoMate's task-space impedance / torque path, rewards, curriculum, SDF reward meshes, soft-DTW and the 5 s
    episode limit are not used (this module never builds AssemblyEnv).
  * a wrist RGB-D camera (config: D405 stream, ManiSkill mount), rendered only when render_camera() is called; the
    camera prim is posed from FK of the measured joints and the hand-eye mount before each render;
    RTX anti-aliasing off (no DLSS upscaling / temporal reuse).
  * socket placement: AutoMate's per-episode randomisation (+-5 cm per axis, yaw 0-10 deg; config.socket_pose), drawn
    at every reset (config.SOCKET_DRAW) and written by set_socket_pose (robot.Robot.reset), also after a weld
    restart, which puts the socket back to its initial state. Robot.reset checks the pose after the settling steps
    (check_socket_pose): only then is socket_T a PhysX read; right after the write Isaac Lab returns the written
    buffer.
  * contact sensors (plug-socket, fingers-socket; robot links get the contact-report API for this) for checks and
    jam detection; they give ONE net force per (sensor body, filter body), opposite walls cancel in it.
  * the expert's force and penetration guards read PhysX's contact report (socket_contacts): per pair of collision
    shapes the contact points' impulses and separations, as SAPIEN's scene.get_contacts() gives them to Chongyu's
    guard; omni.physx contact processing is switched on for this (Isaac Lab switches it off by default; it only
    copies contact data out, the dynamics do not change).

Dynamics that differ from the ManiSkill task (deliberate, kept from AutoMate; not tuned):
  * gravity: off for the plug and the robot links (AutoMate's held asset and robot have disable_gravity); in ManiSkill
    the welded charger link has gravity on (wminsertion_v2 robot.py, disable_gravity = False). With the joint PD of
    stiffness 1e3 N m/rad the charger's weight caused sub-mrad droop there, so the effect is small, but it is not zero.
  * link inertia: AutoMate adds 0.01 kg m^2 to every robot link's inertia diagonal (to steady its torque controller,
    which is not used here); ManiSkill's Panda has no such addition. With damping 1e2 N m s/rad the extra inertia
    moves the fast pole of each joint's PD response by well under 1 ms; the slow pole (D / K = 0.1 s) is unchanged.
  * gripper drive: AutoMate's actuator (stiffness 7500, damping 173, 40 N) holding the fingers at the opening where
    they touch the plug, instead of ManiSkill's gripper gains (the plug is welded in both).
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.direct.automate.assembly_env_cfg import AssemblyEnvCfg
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR, Insertion
from pxr import Gf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation

from . import config as C
from .checks import ResetError, check_contact_report, force_magnitude_sum

AUTOMATE_FRICTION = 0.75     # AssemblyEnv._set_default_dynamics_parameters (held, fixed, robot)
AUTOMATE_INERTIA_ADD = 0.01  # AssemblyEnv._set_body_inertias


def T_from_pose7(p7) -> np.ndarray:
    """(x, y, z, qw, qx, qy, qz) -> 4x4."""
    p7 = np.asarray(p7, np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p7[4], p7[5], p7[6], p7[3]]).as_matrix()
    T[:3, 3] = p7[:3]
    return T


def pose7_from_T(T) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    return np.r_[np.asarray(T)[:3, 3], w, x, y, z]


@configclass
class AutomateSceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg = None
    table: AssetBaseCfg = None
    light: AssetBaseCfg = None
    robot: ArticulationCfg = None
    fixed_asset: ArticulationCfg = None
    held_asset: RigidObjectCfg = None
    plug_contact: ContactSensorCfg = None
    finger_contact: ContactSensorCfg = None
    wrist_cam: CameraCfg = None


def scene_cfg(assembly_id: str, num_envs: int, camera: bool, seg: bool, mount: str) -> AutomateSceneCfg:
    env_cfg, task = AssemblyEnvCfg(), Insertion()
    a = env_cfg.robot.actuators
    actuators = {   # AutoMate: stiffness 0, damping 0 (torques from its task-space controller); ours: joint PD
        "panda_arm1": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"], stiffness=C.ARM_STIFFNESS,
                                          damping=C.ARM_DAMPING, friction=0.0, armature=0.0,
                                          effort_limit_sim=C.ARM_EFFORT_LIMIT,
                                          velocity_limit=a["panda_arm1"].velocity_limit),
        "panda_arm2": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"], stiffness=C.ARM_STIFFNESS,
                                          damping=C.ARM_DAMPING, friction=0.0, armature=0.0,
                                          effort_limit_sim=C.ARM_EFFORT_LIMIT,
                                          velocity_limit=a["panda_arm2"].velocity_limit),
        "panda_hand": a["panda_hand"],
    }
    # contact reporting switched on for the robot links (AutoMate: off) so that finger-socket contact can be measured
    robot = env_cfg.robot.replace(prim_path="{ENV_REGEX_NS}/Robot", actuators=actuators,
                                  spawn=env_cfg.robot.spawn.replace(activate_contact_sensors=True))
    fixed = task.fixed_asset.replace(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        spawn=task.fixed_asset.spawn.replace(usd_path=f"{ASSET_DIR}/{assembly_id}/socket.usd"),
        init_state=task.fixed_asset.init_state.replace(pos=tuple(C.SOCKET_POS[mount]), rot=tuple(C.SOCKET_QUAT[mount])))
    held = task.held_asset.replace(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        spawn=task.held_asset.spawn.replace(usd_path=f"{ASSET_DIR}/{assembly_id}/plug.usd"))
    cfg = AutomateSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=False)
    cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.4)))
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)))
    cfg.light = AssetBaseCfg(prim_path="/World/Light",
                             spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    cfg.robot, cfg.fixed_asset, cfg.held_asset = robot, fixed, held
    cfg.plug_contact = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/HeldAsset/a_{assembly_id}_plug",
                                        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/FixedAsset/a_{assembly_id}_socket"],
                                        update_period=0.0, history_length=0)
    cfg.finger_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_(hand|leftfinger|rightfinger)",
                                          filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/FixedAsset/a_{assembly_id}_socket"],
                                          update_period=0.0, history_length=0)
    if camera:
        K = C.intrinsics()
        types = ["rgb", "distance_to_image_plane"] + (["instance_id_segmentation_fast"] if seg else [])
        cfg.wrist_cam = CameraCfg(
            # A world-level prim posed from FK at every render (render_camera): measured 2026-11-21, a camera prim
            # under Robot/panda_hand stayed at the hand's USD spawn pose (0.77 m / 45 deg away) instead of following
            # the simulated hand.
            prim_path="{ENV_REGEX_NS}/WristCam", update_period=0.0,
            height=C.CAM_H, width=C.CAM_W, data_types=types, colorize_instance_id_segmentation=False,
            depth_clipping_behavior="zero", update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=K.flatten().tolist(), width=C.CAM_W, height=C.CAM_H,
                clipping_range=(C.CLIP_NEAR, C.CLIP_FAR)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"))
    return cfg


class AutomateScene:
    """Simulation + scene of `num_envs` copies; all poses returned in each env's own frame (world - env origin)."""

    def __init__(self, assembly_id: str, num_envs: int = 1, camera: bool = True, seg: bool = True,
                 mount: str = C.SOCKET_MOUNT, device: str = C.PHYSICS_DEVICE, weld: str = C.WELD_MODE,
                 T_hand_plug0: np.ndarray | None = None):
        assert weld in ("articulation", "maximal"), weld
        self.assembly_id, self.num_envs, self.mount, self.device, self.weld_mode = assembly_id, num_envs, mount, device, weld
        env_cfg = AssemblyEnvCfg()
        # Anti-aliasing off: Isaac Lab's default DLSS renders 640 x 480 at 371 x 278 and reconstructs it from earlier
        # frames (temporal), which is wrong for single on-demand frames and for metric depth.
        render = sim_utils.RenderCfg(antialiasing_mode="Off")
        self.sim = sim_utils.SimulationContext(env_cfg.sim.replace(dt=C.PHYSICS_DT, device=device, render=render))
        self.scene = InteractiveScene(scene_cfg(assembly_id, num_envs, camera, seg, mount))
        self.robot, self.socket, self.plug = self.scene["robot"], self.scene["fixed_asset"], self.scene["held_asset"]
        self.contact = self.scene["plug_contact"]
        self.finger_contact = self.scene["finger_contact"]
        self.camera = self.scene["wrist_cam"] if camera else None
        T0 = np.array(C.WELD_INITIAL if T_hand_plug0 is None else T_hand_plug0, dtype=np.float64)
        self._make_welds(T0)
        self.T_hand_plug = [T0.copy() for _ in range(num_envs)]
        import carb
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)   # socket_contacts()
        self._usd_inertia = None
        self.inertia_checks = []
        self.sim.reset()
        self._automate_post_init()
        from .app import assert_off_forbidden_gpu
        assert_off_forbidden_gpu("after the scene build")
        self.regrasps = 0
        self.origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float64)
        self.arm_ids = list(range(7))
        self.hand_idx = self.robot.body_names.index("panda_hand")
        self.fingertip_idx = self.robot.body_names.index("panda_fingertip_centered")
        self.plug_body = f"a_{assembly_id}_plug"
        self.steps = 0

    # ------------------------------------------------------------------ construction
    def _make_welds(self, T0: np.ndarray):
        stage = self.sim.stage
        self.welds = []
        for i in range(self.num_envs):
            ns = f"/World/envs/env_{i}"
            hand = stage.GetPrimAtPath(f"{ns}/Robot/panda_hand")
            plug = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{ns}/HeldAsset")) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert hand.IsValid() and len(plug) == 1, (hand, plug)
            j = UsdPhysics.FixedJoint.Define(stage, f"{ns}/PlugWeld")
            j.CreateBody0Rel().SetTargets([hand.GetPath()])
            j.CreateBody1Rel().SetTargets([plug[0].GetPath()])
            x, y, z, w = Rotation.from_matrix(T0[:3, :3]).as_quat()
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in T0[:3, 3]]))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # "articulation": PhysX adds the plug to the robot articulation as a fixed link (exact, like the ManiSkill
            # task's wm_charger link); its frame can only change between a stop and a new start (set_weld).
            # "maximal": a maximal-coordinate joint solved iteratively; frame settable at runtime, but measured
            # 0.58 mm / 1.3 deg weld deformation at 44 N (smoke check a, 2026-11-21).
            j.CreateExcludeFromArticulationAttr().Set(self.weld_mode == "maximal")
            j.CreateCollisionEnabledAttr().Set(False)          # hand and plug do not collide (fingers still do)
            self.welds.append(j)

    def _automate_post_init(self):
        """AssemblyEnv's two adjustments, after every (re)start. The inertia addition is checked by reading back: a
        restart must have brought back the USD inertias (not the already raised ones), and after the addition every
        robot body must hold USD + 0.01 on its diagonal (exactly once). Raises otherwise; each check is logged in
        inertia_checks."""
        ids = torch.arange(self.num_envs)
        inertias = self.robot.root_physx_view.get_inertias()
        before = inertias.detach().cpu().numpy().astype(np.float64)
        if self._usd_inertia is None:
            self._usd_inertia = before.copy()
        elif np.abs(before - self._usd_inertia).max() > 1e-6:
            raise RuntimeError("inertia read-back: a restart did not restore the USD inertias "
                               f"(max difference {np.abs(before - self._usd_inertia).max():.3g} kg m^2)")
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += AUTOMATE_INERTIA_ADD
        for k, name in enumerate(self.robot.body_names):            # robot bodies only, not the welded plug link
            if "_plug" in name:
                offset[:, k] = 0.0
        self.robot.root_physx_view.set_inertias(inertias + offset, ids)
        after = self.robot.root_physx_view.get_inertias().detach().cpu().numpy().astype(np.float64)
        err = float(np.abs(after - (self._usd_inertia + offset.detach().cpu().numpy())).max())
        self.inertia_checks.append({"restart": len(self.inertia_checks), "max_err_kg_m2": err})
        if err > 1e-6:
            raise RuntimeError(f"inertia read-back: USD + {AUTOMATE_INERTIA_ADD} expected, max difference {err:.3g}")
        for asset in (self.plug, self.socket, self.robot):
            m = asset.root_physx_view.get_material_properties()
            m[..., 0] = AUTOMATE_FRICTION
            m[..., 1] = AUTOMATE_FRICTION
            asset.root_physx_view.set_material_properties(m, torch.arange(self.num_envs, device="cpu"))

    def usd_identity(self) -> dict:
        """The USD layers this stage loaded (Usd.Stage.GetUsedLayers: plug.usd, socket.usd, franka_mimic.usd, the table
        and every layer they reference, e.g. configuration/plug_sensor.usd), each with the sha256 of its bytes as
        Kit's asset client reads them (local cache or the Isaac nucleus):
            {"layers": {identifier: sha256 | "unreadable: <reason>"}, "usd_sha256": sha256 over the sorted pairs}
        The collection's plan hashes the .obj meshes; this records what PhysX actually loaded (the USDs still come
        from the nucleus, not from a frozen local mirror)."""
        import hashlib
        import omni.client
        layers = {}
        for layer in self.sim.stage.GetUsedLayers():
            ident = layer.identifier
            if layer.anonymous or ident.startswith("anon:"):
                continue
            try:
                res, _, content = omni.client.read_file(layer.realPath or ident)
                if res != omni.client.Result.OK:
                    raise OSError(str(res))
                layers[ident] = hashlib.sha256(bytes(memoryview(content))).hexdigest()
            except Exception as exc:                                  # noqa: BLE001 - recorded, never fatal
                layers[ident] = f"unreadable: {type(exc).__name__}: {exc}"[:200]
        h = hashlib.sha256()
        for k in sorted(layers):
            h.update(f"{k} {layers[k]}\n".encode())
        return {"layers": layers, "usd_sha256": h.hexdigest()}

    def close(self):
        """Release this scene so that another assembly can be built in the same app (new stage)."""
        import isaacsim.core.utils.stage as stage_utils
        self.sim._disable_app_control_on_stop_handle = True
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        stage_utils.create_new_stage()

    # ------------------------------------------------------------------ weld
    def set_weld(self, env: int, T_hand_plug: np.ndarray) -> bool:
        """Plug pose in the panda_hand frame, held by the fixed joint. "maximal": applies at the next physics step.
        "articulation": the simulation is stopped, the joint frame edited and the simulation started again (all
        envs' states are reset to the defaults; the caller writes the episode state afterwards). -> restarted?"""
        T_hand_plug = np.asarray(T_hand_plug, np.float64)
        if np.allclose(T_hand_plug, self.T_hand_plug[env], atol=1e-12):
            return False
        restart = self.weld_mode == "articulation"
        if restart:
            self.sim._disable_app_control_on_stop_handle = True   # a stop must not close the app
            self.sim.stop()
        p, (x, y, z, w) = T_hand_plug[:3, 3], Rotation.from_matrix(T_hand_plug[:3, :3]).as_quat()
        self.welds[env].GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in p]))
        self.welds[env].GetLocalRot0Attr().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        self.T_hand_plug[env] = T_hand_plug.copy()
        if restart:
            self.sim.reset()
            self._automate_post_init()
            self.regrasps += 1
        return restart

    # ------------------------------------------------------------------ state access (env frame)
    def _np(self, t) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    def joint_pos(self) -> np.ndarray:
        return self._np(self.robot.data.joint_pos)

    def joint_vel(self) -> np.ndarray:
        return self._np(self.robot.data.joint_vel)

    def body_T(self, name: str, env: int = 0) -> np.ndarray:
        p7 = self._np(self.robot.data.body_pose_w[env, self.robot.body_names.index(name)])
        p7[:3] -= self.origins[env]
        return T_from_pose7(p7)

    def plug_T(self, env: int = 0) -> np.ndarray:
        if self.weld_mode == "articulation":
            return self.body_T(self.plug_body, env)
        p7 = self._np(self.plug.data.root_pose_w[env])
        p7[:3] -= self.origins[env]
        return T_from_pose7(p7)

    def socket_T(self, env: int = 0) -> np.ndarray:
        p7 = self._np(self.socket.data.root_pose_w[env])
        p7[:3] -= self.origins[env]
        return T_from_pose7(p7)

    def robot_base_T(self, env: int = 0) -> np.ndarray:
        p7 = self._np(self.robot.data.root_pose_w[env])
        p7[:3] -= self.origins[env]
        return T_from_pose7(p7)

    def contact_force(self, env: int = 0) -> np.ndarray:
        """Force on the plug from the socket [N], world axes (contact sensor force matrix, filtered)."""
        f = self.contact.data.force_matrix_w
        return np.zeros(3) if f is None else self._np(f[env]).reshape(-1, 3).sum(0)

    def finger_socket_force(self, env: int = 0) -> float:
        """Largest norm of the socket's force on the hand or either finger [N] (milestone-1b checks only)."""
        f = self.finger_contact.data.force_matrix_w
        return 0.0 if f is None else float(np.linalg.norm(self._np(f[env]).reshape(-1, 3), axis=1).max())

    def plug_socket_force_n(self, env: int = 0) -> float:
        """Net plug-socket force magnitude [N] from the contact sensor (one vector per body pair; information)."""
        f = self.contact.data.force_matrix_w
        return 0.0 if f is None else force_magnitude_sum(self._np(f[env]))

    def finger_socket_force_sum(self, env: int = 0) -> float:
        """Sum over hand, left and right finger of the net force magnitude with the socket [N] (contact sensor;
        information)."""
        f = self.finger_contact.data.force_matrix_w
        return 0.0 if f is None else force_magnitude_sum(self._np(f[env]))

    def _path(self, i: int) -> str:
        p = self._paths.get(i) if hasattr(self, "_paths") else None
        if p is None:
            from pxr import PhysicsSchemaTools
            self._paths = getattr(self, "_paths", {})
            p = self._paths[i] = str(PhysicsSchemaTools.intToSdfPath(i))
        return p

    def socket_contacts(self, env: int = 0) -> dict:
        """Contacts of the socket in the last physics step, from PhysX's contact report (omni.physx
        get_contact_report: one header per pair of collision shapes, with its contact points' impulses [N s] and
        separations [m]). Chongyu's guard quantities (checks.py):
            plug_N   = sum over plug-socket shape pairs of |sum of the pair's impulses| / dt_physics
            robot_N  = the same over the pairs of any robot link (hand, fingers, ...) with the socket
            total_N  = plug_N + robot_N          (socket_contact_force_n: every contact with the receptacle)
            penetration_mm = max over those points of max(0, -separation) [mm]   (socket_penetration_mm)
        Information: *_points_N = sum of |impulse| over points / dt (upper bound), plug_net_N = |sum over all plug
        points| / dt (what the contact sensor reports), n_pairs, n_points, socket shapes seen."""
        from omni.physx import get_physx_simulation_interface
        from .checks import pair_force_sum, point_force_sum
        headers, data = get_physx_simulation_interface().get_contact_report()
        ns = f"/World/envs/env_{env}/"
        sock = ns + "FixedAsset"
        groups = {"plug": {}, "robot": {}}
        sep_min, shapes, net = 0.0, set(), np.zeros(3)
        for h in headers:
            n = int(h.num_contact_data)
            if n == 0:
                continue
            a0, a1 = self._path(int(h.actor0)), self._path(int(h.actor1))
            if a0.startswith(sock):
                other, sign, sc = a1, -1.0, int(h.collider0)
            elif a1.startswith(sock):
                other, sign, sc = a0, 1.0, int(h.collider1)
            else:
                continue
            kind = "plug" if other.startswith(ns + "HeldAsset") else "robot" if other.startswith(ns + "Robot") else None
            if kind is None:
                continue
            o = int(h.contact_data_offset)
            J = np.array([(data[k].impulse.x, data[k].impulse.y, data[k].impulse.z) for k in range(o, o + n)], np.float64)
            sep_min = min(sep_min, min(float(data[k].separation) for k in range(o, o + n)))
            key = (int(h.collider0), int(h.collider1))
            groups[kind][key] = np.vstack([groups[kind][key], J]) if key in groups[kind] else J
            shapes.add(self._path(sc))
            if kind == "plug":
                net += sign * J.sum(0)       # impulse on actor0; sign so that it is the socket's push on the plug
        dt = C.PHYSICS_DT
        out = {k + "_N": pair_force_sum(g.values(), dt) for k, g in groups.items()}
        out["plug_points"] = int(sum(len(J) for J in groups["plug"].values()))
        out["robot_points"] = int(sum(len(J) for J in groups["robot"].values()))
        out.update({k + "_points_N": point_force_sum(g.values(), dt) for k, g in groups.items()})
        out.update(total_N=out["plug_N"] + out["robot_N"], penetration_mm=max(0.0, -sep_min) * 1000.0,
                   plug_net_N=float(np.linalg.norm(net) / dt), n_pairs=sum(len(g) for g in groups.values()),
                   n_points=int(sum(len(J) for g in groups.values() for J in g.values())),
                   socket_shapes=sorted(shapes))
        return out

    def guard_contacts(self, env: int = 0) -> dict:
        """socket_contacts, checked against the contact sensors (checks.check_contact_report): raises
        checks.ContactReportError (a RuntimeError) if the report has no plug-socket contact point while the plug
        sensor sees more than 1 N between plug and socket, or no robot-socket point while the hand / finger sensor
        sees more than 1 N (contact processing not working). Each kind is checked on its own: hand / finger points
        must not hide a missing plug pair, nor plug points a missing robot pair (r8-expert-01)."""
        return check_contact_report(self.socket_contacts(env), self.plug_socket_force_n(env),
                                    self.finger_socket_force_sum(env))

    # ------------------------------------------------------------------ socket pose
    def set_socket_pose(self, env: int, T: np.ndarray):
        """Socket root pose (env frame), zero velocity, as AutoMate's randomize_fixed_initial_state writes it
        (write_root_state_to_sim, reset). Not checked here: until the next physics step Isaac Lab's root_pose_w
        returns the buffer just written (articulation.py write_root_link_pose_to_sim), so a read-back now proves
        nothing; check_socket_pose after a step does."""
        p7 = pose7_from_T(T)
        p7[:3] += self.origins[env]
        st = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        st[0, :7] = torch.tensor(p7, dtype=torch.float32, device=self.device)
        ids = torch.tensor([env], device=self.device)
        self.socket.write_root_state_to_sim(st, env_ids=ids)
        self.socket.reset(ids)

    def check_socket_pose(self, env: int, T: np.ndarray, tol_m: float = 1e-5, tol_rad: float = 1e-5) -> dict:
        """After at least one physics step: the socket pose read from PhysX equals T within tol (raises
        checks.ResetError, a RuntimeError, otherwise)."""
        got = self.socket_T(env)
        T = np.asarray(T, np.float64)
        dp = float(np.linalg.norm(got[:3, 3] - T[:3, 3]))
        dr = float(np.arccos(np.clip((np.trace(got[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        if dp > tol_m or dr > tol_rad:
            raise ResetError(f"socket pose after the step differs from the written one by {dp * 1e3:.4f} mm / "
                               f"{np.degrees(dr):.4f} deg: wrote {T.tolist()}, PhysX has {got.tolist()}")
        return {"socket_pose_err_m": dp, "socket_pose_err_rad": dr}

    # ------------------------------------------------------------------ writing state
    def write_state(self, env: int, q9: np.ndarray, plug_T: np.ndarray, targets9: np.ndarray | None = None):
        """Teleport: joints to q9 (zero velocity), plug to plug_T (env frame), PD targets to targets9 (default q9)."""
        ids = torch.tensor([env], device=self.device)
        q = torch.tensor(np.asarray(q9, np.float32)[None], device=self.device)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        tgt = q if targets9 is None else torch.tensor(np.asarray(targets9, np.float32)[None], device=self.device)
        self.robot.set_joint_position_target(tgt, env_ids=ids)
        if self.weld_mode == "maximal":                               # an articulation link follows the joints
            p7 = pose7_from_T(plug_T)
            p7[:3] += self.origins[env]
            self.plug.write_root_pose_to_sim(torch.tensor(p7[None], dtype=torch.float32, device=self.device), env_ids=ids)
            self.plug.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=ids)
            self.plug.reset(ids)
        self.robot.reset(ids)

    def set_targets(self, targets: np.ndarray):
        """PD targets (num_envs, 9): 7 arm joints [rad], 2 fingers [m]."""
        self.robot.set_joint_position_target(torch.tensor(np.asarray(targets, np.float32), device=self.device))

    def step(self, n_control: int = 1):
        """n control steps of DECIMATION physics steps each; no rendering."""
        for _ in range(n_control):
            for _ in range(C.DECIMATION):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=C.PHYSICS_DT)
            self.steps += 1

    # ------------------------------------------------------------------ camera
    def render_camera(self, env: int = 0, T_cam: np.ndarray | None = None) -> dict:
        """Render now; -> rgb (H, W, 3) uint8, depth (H, W) float64 metres to the image plane (0 = no return),
        seg_ids (H, W) int64 and their prim paths, camera pose read back from the simulator (for checks).
        T_cam: camera pose in the env frame, Isaac "world" camera axes (x forward, z up), set before rendering."""
        if T_cam is not None:
            x, y, z, w = Rotation.from_matrix(T_cam[:3, :3]).as_quat()
            pos = torch.tensor((T_cam[:3, 3] + self.origins[env])[None], dtype=torch.float32, device=self.device)
            quat = torch.tensor([[w, x, y, z]], dtype=torch.float32, device=self.device)
            self.camera.set_world_poses(pos, quat, env_ids=[env], convention="world")
        self.sim.render()
        if not getattr(self, "_gpu_checked", False):                # the first frame may open render contexts
            from .app import assert_off_forbidden_gpu
            assert_off_forbidden_gpu("after the first rendered frame")
            self._gpu_checked = True
        self.camera.update(0.0, force_recompute=True)
        out, info = self.camera.data.output, self.camera.data.info[env]
        res = {"rgb": out["rgb"][env, ..., :3].detach().cpu().numpy().astype(np.uint8),
               "depth": out["distance_to_image_plane"][env, ..., 0].detach().cpu().numpy().astype(np.float64)}
        if "instance_id_segmentation_fast" in out:
            res["seg_ids"] = out["instance_id_segmentation_fast"][env, ..., 0].detach().cpu().numpy().astype(np.int64)
            res["seg_labels"] = info.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
        p = self._np(self.camera.data.pos_w[env]) - self.origins[env]
        q = self._np(self.camera.data.quat_w_ros[env])                 # ROS = OpenCV axes (x right, y down, z fwd)
        res["cam_T_cv_sim"] = T_from_pose7(np.r_[p, q]) if np.linalg.norm(q) > 0.5 else None
        res["K_sim"] = self._np(self.camera.data.intrinsic_matrices[env])
        return res

    # ------------------------------------------------------------------ save / restore
    def get_state(self) -> dict:
        s = self.scene.get_state(is_relative=True)
        s = {k: {n: {f: t.clone() for f, t in e.items()} for n, e in v.items()} for k, v in s.items()}
        if self.weld_mode == "articulation":
            s["rigid_object"] = {}                                    # the plug is a robot link: its state is q
        return {"scene": s, "targets": self.robot.data.joint_pos_target.clone(),
                "T_hand_plug": [T.copy() for T in self.T_hand_plug], "steps": self.steps}

    def set_state(self, st: dict):
        for i, T in enumerate(st["T_hand_plug"]):
            if not np.array_equal(T, self.T_hand_plug[i]):
                self.set_weld(i, T)
        org = self.scene.env_origins
        for name, art in self.scene.articulations.items():          # as InteractiveScene.reset_to, without the plug
            a = st["scene"]["articulation"][name]                   # when it is a robot link
            pose = a["root_pose"].clone()
            pose[:, :3] += org
            art.write_root_pose_to_sim(pose)
            art.write_root_velocity_to_sim(a["root_velocity"].clone())
            art.write_joint_state_to_sim(a["joint_position"].clone(), a["joint_velocity"].clone())
            art.reset()
        for name, obj in self.scene.rigid_objects.items():
            if name in st["scene"]["rigid_object"]:
                o = st["scene"]["rigid_object"][name]
                pose = o["root_pose"].clone()
                pose[:, :3] += org
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(o["root_velocity"].clone())
                obj.reset()
        self.robot.set_joint_position_target(st["targets"].clone())
        self.scene.write_data_to_sim()
        self.steps = st["steps"]
