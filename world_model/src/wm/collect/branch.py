"""The branch set (2026-11-15, decided with the user): the policy's own attempts, several from the same start.

Why: the planner has to choose, in ONE state, between candidate actions (the policy's proposal and perturbed copies of it).
Every earlier set has one trajectory per start, so "this start is good / bad" and "this action is better than that one"
cannot be told apart. Here the same start is tried several times; going back to a start and trying again is also what the
robot does at run time after it was blocked, so the branches of a start are the retries of that start.

What is collected for one grasp (everything the robot does is possible on a real robot):
    reference   one insertion of this grasp (wm_v1's seed) gives the inserted hand pose H* and the axis a (FK only). On the
                real robot: a person seats the plug, the gripper closes, the joint angles are read.
    start       (not recorded) explore.Explorer.place: H* moved back along a by pin length + gap, shifted and rotated;
                the box of the policy's demonstrations (gap 6-18 mm, +-10 mm across, +-6 deg per axis). FK feedback only.
    branch      (recorded) the flow-matching policy acts every 0.4 s (16 joint targets, 8 executed) from its own images.
                The first N_PURE branches execute them as they are (different sampling seeds: the policy's own retries).
                The other branches add a displacement of the commanded hand pose to every executed target of a
                decision, in the frame (a, y, z): a bias drawn once per branch (across a, and for half of the branches
                a small rotation) plus a jitter drawn per decision. The policy's own targets are recorded next to the
                executed ones.
                Sizes (smoke run 2026-11-15, 180 branches, bias sigma 0.1 / 0.2 / 0.4 mm per decision): the policy's
                targets are relative to the measured joints, so a bias adds up over the ~18 decisions to the board;
                the true lateral offset at the board face went from 0.78 mm (median, policy alone) to 1.4 / 1.9 / 2.9 mm
                and success from 18/45 to 6/55, 5/42, 0/38. The sigmas are therefore 0.02 / 0.04 mm per decision
                (about 0.25-0.5 mm more at the face), jitter 0.03 mm, rotation bias 0.005 deg.
    same start  the simulator state after placement is saved and restored for every branch (measured deviation in the
                summary). On the real robot: the joint configuration is replayed with the same grasp.
    stop        seated (FK hand within 1 mm and 2 deg of H*), stalled (the FK hand moved less than 0.2 mm in the last 8
                decisions = 3.2 s: blocked; at run time the robot then goes back to the start and tries again), left
                (hand more than 30 mm across a from H*, or 3 mm past it), or 60 s. Only FK is used. (Smoke run: 1 of 29
                inserted branches had such a stall before it went in; failed branches stop at decision 62 (median)
                instead of 150.)
Labels come from FK and H* only: depth = pin length - |hand - H*| (clipped at 0); seated_fk = the branch ends within
1.5 mm and 2 deg of H*. The simulator's truth
(eval_rel, outcome) is stored next to them for comparison and for judging, never as a model input or training label.

Files: <out>/<port>/<grasp>/s<start>_b<branch>.npz (data/store.py format). Images (the user, 2026-11-15: 640 x 640): run
with WM_CAM_SCALE=2, the wrist camera is rendered at 1280 x 960 and the usual region around pins and holes is stored as
640 x 640 (rgb JPEG, depth uint16 0.1 mm; K of that crop), one every 4 control steps (5 Hz). The policy sees the same
picture reduced to its 320 x 320 input (rgb: mean of 2 x 2, depth: every second pixel, K scaled). Extra arrays: qvel,
action_policy (the policy's targets before the displacement), dec_step, dec_pos_m / dec_rot_deg (the displacement of
every decision), depth_fk_m.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..data import store
from ..data.kinematics import HandFK
from ..data.recorder import _truth_rel
from ..env import robot as _robot
from ..env.expert import Expert
from ..env.robot import Frame, Robot
from .explore import PLACEMENT_ATTEMPTS, Explorer, offset_pose, truth_outcome
from .._paths import use_alignment_field

use_alignment_field()
from paf import env_adapter as _ea  # noqa: E402
from paf.geometry import pose_distance  # noqa: E402

EXECUTE = 8                   # control steps per decision (0.4 s), as eval/agent.py
IMAGE_EVERY = 4               # 5 Hz images
MAX_DECISIONS = 150           # 60 s, the closed-loop protocol
STOP_SEATED_M = 0.001         # a branch stops when the FK hand is this close to H* (and within SEATED_DEG)
SEATED_M, SEATED_DEG = 0.0015, 2.0
# The label "seated" (meta: seated_fk) is the END of a branch within 1.5 mm and 2 deg of H*. Two smoke runs, 276 branches:
# truly inserted plugs end 0.3-1.15 mm from H* (H* is where the reference insertion happened to stop; some stall just
# above 1 mm), blocked ones 15 mm or more, tilted half-inserted ones 1.8-2.8 mm with more than 2 deg. Stopping at 1.5 mm
# cut branches that were still going in (first try of the full run), so the stop stays at 1 mm.
LEFT_M, PAST_M = 0.030, 0.003
START_BOX = {"gap_m": (0.006, 0.018), "lateral_m": 0.010, "rot_deg": 6.0}   # the demonstrations' start box
STALL_DECISIONS, STALL_M = 8, 0.0002
N_PURE = 2                    # branches that run the policy as it is
BIAS_SIGMAS_M = (0.00002, 0.00004)   # per-decision bias across a (one sigma per branch, drawn from these)
JITTER_M = 0.00003            # per-decision jitter, each axis
ROT_BIAS_DEG = 0.005          # per-decision rotation bias (each axis), half of the perturbed branches


def sample_start(rng: np.random.Generator) -> dict:
    b = START_BOX
    return {"gap_m": float(rng.uniform(*b["gap_m"])),
            "lateral_m": rng.uniform(-b["lateral_m"], b["lateral_m"], 2).tolist(),
            "rotvec_deg": rng.uniform(-b["rot_deg"], b["rot_deg"], 3).tolist()}


def sample_noise(rng: np.random.Generator, branch: int) -> dict:
    """The first N_PURE branches: the policy as it is. Others: bias across a (+ rotation for half of them) and jitter."""
    if branch < N_PURE:
        return {"bias_m": [0.0, 0.0, 0.0], "rot_bias_deg": [0.0, 0.0, 0.0], "jitter_m": 0.0, "sigma_m": 0.0}
    s = float(rng.choice(BIAS_SIGMAS_M))
    rot = rng.normal(0, ROT_BIAS_DEG, 3) if rng.random() < 0.5 else np.zeros(3)
    return {"bias_m": [0.0, *rng.normal(0, s, 2).tolist()], "rot_bias_deg": rot.tolist(), "jitter_m": JITTER_M, "sigma_m": s}


class FullRecorder:
    """Per control step: joints, joint velocities, FK hand, truth (eval only), executed and policy targets.
    Every IMAGE_EVERY steps: the full wrist frame. `frame` is the 320 crop of the latest picture (the policy's input)."""

    def __init__(self, robot: Robot):
        self.robot = robot

    def begin(self):
        self.t = 0
        self.buf = {k: [] for k in ("qpos", "qvel", "hand", "eval_rel", "action", "action_policy",
                                    "rgb", "depth", "img_step", "cam_T_world")}
        self._sample()
        self._image()

    def step(self, q_exec: np.ndarray, q_policy: np.ndarray):
        self.buf["action"].append(np.asarray(q_exec, np.float32)[:7])
        self.buf["action_policy"].append(np.asarray(q_policy, np.float32)[:7])
        self.t += 1
        self._sample()
        if self.t % IMAGE_EVERY == 0:
            self._image()

    def end(self) -> dict:
        if self.buf["img_step"][-1] != self.t:
            self._image()
        b = self.buf
        return {"qpos": np.asarray(b["qpos"], np.float32), "qvel": np.asarray(b["qvel"], np.float32),
                "hand": np.asarray(b["hand"], np.float32), "eval_rel": np.asarray(b["eval_rel"], np.float32),
                "action": np.asarray(b["action"], np.float32).reshape(-1, 7),
                "action_policy": np.asarray(b["action_policy"], np.float32).reshape(-1, 7),
                "rgb": np.asarray(b["rgb"], np.uint8), "depth": np.asarray(b["depth"], np.uint16),
                "img_step": np.asarray(b["img_step"], np.int32),
                "cam_T_world": np.asarray(b["cam_T_world"], np.float32), "K": np.asarray(self.K, np.float32)}

    def _sample(self):
        r = self.robot
        self.buf["qpos"].append(r.qpos().astype(np.float32))
        self.buf["qvel"].append(_ea._np(r.base.agent.robot.get_qvel())[0].astype(np.float32))
        self.buf["hand"].append(r.hand_pose().astype(np.float32))
        self.buf["eval_rel"].append(_truth_rel(r))

    def _image(self):
        """As Robot.frame() (the crop of the rendered stream, at the render scale); the policy gets the same picture
        reduced to its 320 x 320 input."""
        r, s = self.robot, _robot.CAM_SCALE
        cam, comp = r._camera()
        r.base.scene.update_render()
        cam.capture()
        r._twin.take_picture()
        r0, r1, c0, c1 = _robot.CROP
        rgb = np.ascontiguousarray(_ea._np(cam.get_picture("Color"))[0, r0:r1, c0:c1, :3]).astype(np.uint8)
        depth = -np.asarray(r._twin.get_picture("Position"))[r0:r1, c0:c1, 2].astype(np.float64)
        depth[depth < _robot.DEPTH_MIN_M] = 0.0
        d = np.clip(np.round(depth / _robot.DEPTH_UNIT_M), 0, 65535).astype(np.uint16)
        ext = np.eye(4)
        ext[:3, :] = np.asarray(comp.get_extrinsic_matrix())
        K = np.asarray(comp.get_intrinsic_matrix(), dtype=np.float64).copy()
        K[0, 2] -= c0
        K[1, 2] -= r0
        self.K = K
        self.buf["rgb"].append(rgb)
        self.buf["depth"].append(d)
        self.buf["img_step"].append(self.t)
        self.buf["cam_T_world"].append(ext.astype(np.float32))
        if s == 1:
            rgb_p, d_p, Kp = rgb, d, K
        else:
            h, w = rgb.shape[0] // s, rgb.shape[1] // s
            rgb_p = np.round(rgb.reshape(h, s, w, s, 3).astype(np.float32).mean((1, 3))).astype(np.uint8)
            d_p = np.ascontiguousarray(d[::s, ::s])
            Kp = K.copy()
            Kp[:2] /= s
        self.frame = Frame(rgb=rgb_p, depth=d_p, seg=np.zeros(d_p.shape, np.uint8), K=Kp, cam_T_world=ext,
                           qpos=r.qpos(), hand=r.hand_pose())
        self.frame.full = (rgb, d)          # the stored full picture, for a policy that reads it (wm/tdmpc/policy_resnet.py)


class Brancher(Explorer):
    def __init__(self, robot: Robot, H_star: np.ndarray, axis: np.ndarray):
        super().__init__(robot, H_star, axis)
        self.fk = HandFK(robot)
        self.expert = Expert(robot)

    def place(self, start: dict) -> str:
        """Explorer.place; when its straight waypoint path is refused (measured 2026-11-15 on two held-out grasps: 20 of 32
        starts, mostly those nearer than the 12 mm standoff), the same target (still H*, a, pin length only) is reached
        by the planner's path instead, as Expert.move_to_offset does for the demonstrations' starts. Without this the
        redrawn starts would lose the near ones."""
        stage = super().place(start)
        if stage == "ok":
            return stage
        r = self.robot
        self._joint_move(self.q_back)
        near = offset_pose(self.H_star, self.F, -(self.L + start["gap_m"]), start["lateral_m"], start["rotvec_deg"])
        if not self.expert._planned_move(near, tol_m=1e-3, tol_deg=0.5):
            self._joint_move(self.q_back)
            return stage + "+plan"
        cmd, q = near.copy(), None
        for _ in range(3):                                         # as Explorer.place: remove the steady-state error
            q = r._ik(cmd, r.qpos() if q is None else q)
            if q is None or np.abs(q[:7] - r.qpos()[:7]).max() > 0.05:
                self._joint_move(self.q_back)
                return stage + "+settle"
            r._command(q, 20)
            cmd[:3, 3] += near[:3, 3] - r.hand_pose()[:3, 3]
        self.cmd0, self.q0 = cmd, q
        self.start_error_mm = float(np.linalg.norm(near[:3, 3] - r.hand_pose()[:3, 3]) * 1000)
        return "ok" if self.start_error_mm < 0.5 else stage + "+far_off"

    # ------------------------------------------------------------ same start
    def snapshot(self):
        self._snap = self.robot.base.get_state_dict()
        self._snap_hand = self.robot.hand_pose()

    def restore(self) -> list[float]:
        """Back to the saved start. Returns the deviation of the hand from the saved one (mm, deg)."""
        self.robot.base.set_state_dict(self._snap)
        dt, dr = pose_distance(self.robot.hand_pose(), self._snap_hand)
        return [float(dt * 1000), float(np.degrees(dr))]

    # ------------------------------------------------------------ FK-only state
    def fk_state(self) -> dict:
        H = self.robot.hand_pose()
        rel = self.F.T @ (H[:3, 3] - self.H_star[:3, 3])
        dt, dr = pose_distance(H, self.H_star)
        return {"dist_m": float(dt), "angle_deg": float(np.degrees(dr)), "along_m": float(rel[0]),
                "across_m": float(np.linalg.norm(rel[1:]))}

    # ------------------------------------------------------------ one recorded branch
    def rollout(self, rec: FullRecorder, agent, seed: int, axes_seed: int, noise: dict, max_decisions: int) -> dict:
        import torch
        r, F = self.robot, self.F
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rng = np.random.default_rng(seed)
        rec.begin()
        if not agent.reset(rec.frame, axes_seed):
            return {"stopped": "no_board_plane", "decisions": 0}
        perturbed = noise["sigma_m"] > 0
        bias, rot_bias = np.asarray(noise["bias_m"]), np.asarray(noise["rot_bias_deg"])
        dec_step, dec_pos, dec_rot, missed, why, q_last = [], [], [], 0, "time", r.qpos()[:7]
        hands = [r.hand_pose()[:3, 3]]
        for k in range(max_decisions):
            chunk = np.asarray(agent.act(rec.frame)[0][:EXECUTE], np.float64)          # (8, 7) absolute joint targets
            eps = bias + (rng.normal(0, noise["jitter_m"], 3) if perturbed else 0.0)
            dec_step.append(rec.t)
            dec_pos.append(eps)
            dec_rot.append(rot_bias)
            if perturbed:
                fingers = np.tile(r.qpos()[7:9], (EXECUTE, 1))
                T = self.fk(np.c_[chunk, fingers])
                R_eps = Rotation.from_rotvec(F @ np.radians(rot_bias)).as_matrix()
            for j in range(EXECUTE):
                q = chunk[j]
                if perturbed:
                    Tj = T[j].copy()
                    Tj[:3, 3] += F @ eps
                    Tj[:3, :3] = R_eps @ Tj[:3, :3]
                    sol = r._ik(Tj, np.r_[chunk[j], fingers[j]])
                    if sol is None or np.abs(sol[:7] - chunk[j]).max() > 0.03:
                        missed += 1                                                   # no solution: the policy's own target
                    else:
                        q = sol[:7]
                q = q.astype(np.float32).astype(float)
                r._command(q, 1)
                rec.step(q, chunk[j])
                q_last = q
            s = self.fk_state()
            if s["dist_m"] < STOP_SEATED_M and s["angle_deg"] < SEATED_DEG:
                why = "seated"
                break
            if s["across_m"] > LEFT_M or s["along_m"] > PAST_M:
                why = "left"
                break
            hands.append(r.hand_pose()[:3, 3])
            if len(hands) > STALL_DECISIONS and np.linalg.norm(hands[-1] - hands[-1 - STALL_DECISIONS]) < STALL_M:
                why = "stalled"
                break
        for _ in range(4):                                                            # a short hold, as the other sets
            r._command(q_last, 1)
            rec.step(q_last, q_last)
        end = self.fk_state()
        return {"stopped": why, "decisions": len(dec_step), "ik_missed_steps": missed, "fk_end": end,
                "seated_fk": bool(end["dist_m"] < SEATED_M and end["angle_deg"] < SEATED_DEG),
                "dec_step": np.asarray(dec_step, np.int32), "dec_pos_m": np.asarray(dec_pos, np.float32),
                "dec_rot_deg": np.asarray(dec_rot, np.float32)}


def collect_grasp(robot: Robot, agent, g: dict, out: Path, n_starts: int, n_branches: int, seed: int,
                  max_decisions: int = MAX_DECISIONS) -> dict:
    """g: a wm_v1 grasp.json (port, grasp, condition, pitch_deg, grasp_pos_m, episode_seed, H_star, axis)."""
    t0 = time.time()
    gdir = out / g["port"] / g["grasp"]
    gdir.mkdir(parents=True, exist_ok=True)
    base = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    summary = {**base, "starts": [], "bytes": 0}
    try:
        rng = np.random.default_rng(seed)
        expert = Expert(robot)
        robot.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            summary["error"] = "reference insertion failed"
            return summary
        H_star = robot.hand_pose()
        dt, dr = pose_distance(H_star, np.asarray(g["H_star"]))
        summary["H_star_vs_wm_v1_mm_deg"] = [dt * 1000, float(np.degrees(dr))]
        ex = Brancher(robot, H_star, np.asarray(g["axis"]))
        if not ex.remember_standoff():
            summary["error"] = "standoff not reachable"
            return summary
        label = {"H_star": H_star.tolist(), "axis": list(g["axis"]), "pin_length_m": ex.L}
        for i in range(n_starts):
            fails = []
            for _ in range(PLACEMENT_ATTEMPTS):
                start = sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            s_entry = {"idx": i, "start": start, "placement_failures": fails, "branches": []}
            summary["starts"].append(s_entry)
            if stage != "ok":
                s_entry["outcome"] = "placement_failed"
                continue
            s_entry["start_error_mm"] = ex.start_error_mm
            ex.snapshot()
            for b in range(n_branches):
                noise = sample_noise(rng, b)
                bseed = int(seed + 1000 * i + b)
                dev = ex.restore()
                rec = FullRecorder(robot)
                res = ex.rollout(rec, agent, bseed, int(seed + 1000 * i), noise, max_decisions)
                arrays = rec.end()
                entry = {"idx": b, "seed": bseed, "noise": noise, "restore_dev_mm_deg": dev,
                         **{k: v for k, v in res.items() if not isinstance(v, np.ndarray)},
                         "steps": int(len(arrays["action"])), **truth_outcome(arrays["eval_rel"][-1])}
                s_entry["branches"].append(entry)
                if res["stopped"] == "no_board_plane":
                    continue
                for k in ("dec_step", "dec_pos_m", "dec_rot_deg"):
                    arrays[k] = res[k]
                d = np.linalg.norm(arrays["hand"][:, :3, 3] - H_star[:3, 3], axis=1)
                arrays["depth_fk_m"] = np.clip(ex.L - d, 0.0, ex.L).astype(np.float32)
                meta = {**base, **label, "kind": "branch", "start_idx": i, "start": start,
                        "start_error_mm": ex.start_error_mm, **entry}
                summary["bytes"] += store.save(gdir / f"s{i:02d}_b{b}.npz", arrays, meta)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-2000:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        (gdir / "grasp.json").write_text(json.dumps(summary))
    return summary
