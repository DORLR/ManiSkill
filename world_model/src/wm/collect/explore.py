"""Policy-independent contact exploration (v4, decided with the user 2026-11-10).

Why: the world model must predict what happens for actions no demonstration contains (CEM searches such
actions when a policy is improved), and it must not be trained on any policy's rollouts. So the robot
tries random things near and on the board and the real consequences are recorded.

Everything the robot does here is possible on a real robot (the user's condition):
    reference   one insertion of this grasp gives the inserted hand pose H* and the insertion axis a (forward
                kinematics of that insertion; already in wm_v1's grasp.json). The plug's pin length is a
                property of the plug in the hand (measured once).
    start       (not recorded) the hand goes to H* moved back along a by pin length + gap (2-6 mm in front
                of the board), shifted across a by 0-10 mm and rotated by 0-6 deg (random; 10 % aligned,
                40 % within 1 mm, 30 % 1-3 mm, 30 % 3-10 mm: dense where the outcome is decided). It always
                starts from the same remembered standoff configuration, and the arm's steady-state error
                is removed with FK feedback.
    maneuver    (recorded) 0-3 random segments: push / retreat along a slightly tilted axis, slide across
                it, small rotation, push + slide, rarely a long slide (10-60 mm, to the board's edge);
                straight-line hand motion at 2-6 mm/s through inverse kinematics. Segments are drawn
                before the trajectory starts: they do not depend on where the hole is.
    push        (recorded) straight push along a at 3 mm/s until the hand reaches the inserted depth
                (FK vs H*) or is blocked
    blocked     the commanded hand position leads the measured one (FK) by more than 1.5 mm: the segment
                stops and the command is held for 0.4-1.6 s (a force limit on a real robot)
The simulator's truth is used only for the outcome label (data/outcome.py) and for checks, as in wm_v1.

Files: <out>/<port>/<grasp>/tNN_explore.npz (data/store.py format; meta: start, segments, outcome).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..data import store
from ..data.outcome import diagnosis, inserted
from ..data.recorder import Recorder
from ..env.expert import Expert
from ..env.robot import Robot
from .._paths import use_alignment_field

use_alignment_field()
from paf.geometry import interpolate_T, pose_distance  # noqa: E402

DT = 0.05                    # control period (20 Hz)
BACK_GAP_M = 0.012           # standoff where the offset is applied (free space)
LAG_LIMIT_M = 0.0015         # blocked: commanded leads measured hand position by this much
PUSH_SPEED = 0.003           # final push, m/s (the expert's insertion speed)
DEPTH_CAP_M = 0.001          # never command the hand deeper than inserted + 1 mm
MAX_STEPS = 480              # 24 s per trajectory
PLACEMENT_ATTEMPTS = 5


def frame(axis: np.ndarray) -> np.ndarray:
    """Columns: a (insertion axis), y, z (up, perpendicular to a)."""
    a = axis / np.linalg.norm(axis)
    z = np.array([0.0, 0.0, 1.0]) - a[2] * a
    z /= np.linalg.norm(z)
    return np.stack([a, np.cross(z, a), z], 1)


def offset_pose(H_star: np.ndarray, F: np.ndarray, depth_m: float, lateral_m, rotvec_deg) -> np.ndarray:
    """H* moved by `depth_m` along a (negative = back), `lateral_m` (y, z) across it, and rotated about the
    hand by `rotvec_deg` (components along a, y, z)."""
    T = H_star.copy()
    T[:3, 3] = H_star[:3, 3] + F @ np.array([depth_m, lateral_m[0], lateral_m[1]])
    T[:3, :3] = Rotation.from_rotvec(F @ np.radians(rotvec_deg)).as_matrix() @ H_star[:3, :3]
    return T


def _unit(rng, n):
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def sample_start(rng: np.random.Generator) -> dict:
    """Start in front of the board; truth-free by construction (relative to H*)."""
    if rng.random() < 0.1:
        lat, rot = np.zeros(2), np.zeros(3)                       # aligned: a clean positive
    else:
        u, v = rng.random(), rng.random()
        lat = _unit(rng, 2) * (rng.uniform(0, 0.001) if u < 0.4 else rng.uniform(0.001, 0.003) if u < 0.7 else rng.uniform(0.003, 0.010))
        rot = _unit(rng, 3) * (rng.uniform(0, 1.0) if v < 0.5 else rng.uniform(1.0, 3.0) if v < 0.8 else rng.uniform(3.0, 6.0))
    return {"gap_m": float(rng.uniform(0.002, 0.006)), "lateral_m": lat.tolist(), "rotvec_deg": rot.tolist()}


def sample_segments(rng: np.random.Generator) -> list[dict]:
    """0-3 random maneuver segments, as motions of the commanded hand pose in the frame (a, y, z)."""
    segs = []
    for _ in range(int(rng.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15]))):
        kind = str(rng.choice(["push", "retreat", "slide", "rotate", "push_slide", "long_slide"],
                              p=[0.25, 0.2, 0.2, 0.1, 0.2, 0.05]))
        d, r = np.zeros(3), np.zeros(3)
        if kind in ("push", "push_slide"):
            tilt = _unit(rng, 2) * np.tan(np.radians(rng.uniform(0, 5.0)))      # push axis off a by up to 5 deg
            d = np.array([1.0, tilt[0], tilt[1]]) * rng.uniform(0.001, 0.008)
        if kind == "retreat":
            d = np.array([-rng.uniform(0.001, 0.006), 0.0, 0.0])
        if kind in ("slide", "push_slide"):
            d[1:] += _unit(rng, 2) * rng.uniform(0.0003, 0.004)
        if kind == "long_slide":
            d[1:] = _unit(rng, 2) * rng.uniform(0.010, 0.060)
        if kind == "rotate":
            r = _unit(rng, 3) * rng.uniform(0.3, 2.0)
        segs.append({"kind": kind, "move_m": d.tolist(), "rotvec_deg": r.tolist(),
                     "speed_m_s": float(rng.uniform(0.002, 0.006) if kind != "long_slide" else rng.uniform(0.006, 0.015)),
                     "hold_steps": int(rng.integers(8, 33))})
    return segs


class Explorer:
    def __init__(self, robot: Robot, H_star: np.ndarray, axis: np.ndarray):
        self.robot, self.H_star, self.F = robot, np.asarray(H_star, np.float64), frame(np.asarray(axis, np.float64))
        self.L = float(robot.port_spec.pin_length_m)

    def depth(self, T: np.ndarray) -> float:
        """Hand position along a relative to inserted (negative = in front)."""
        return float(self.F[:, 0] @ (T[:3, 3] - self.H_star[:3, 3]))

    # ------------------------------------------------------------ placement (not recorded)
    def remember_standoff(self) -> bool:
        """Right after the reference insertion: straight out to the aligned standoff and remember that joint
        configuration. Every placement starts from it, so the arm always stays on the same IK branch."""
        out = offset_pose(self.H_star, self.F, -(self.L + BACK_GAP_M), [0.0, 0.0], [0.0, 0.0, 0.0])
        if not self.robot.move_to(out):
            return False
        self.q_back = self.robot.qpos().copy()
        return True

    def _joint_move(self, q_goal: np.ndarray, max_step: float = 0.01):
        q0 = self.robot.qpos()
        n = max(1, int(np.ceil(np.abs(q_goal[:7] - q0[:7]).max() / max_step)))
        for k in range(1, n + 1):
            self.robot._command(q0[:7] + (q_goal[:7] - q0[:7]) * k / n, 1)
        self.robot._command(q_goal[:7], 30)

    def place(self, start: dict) -> str:
        """-> "ok" or the stage that failed. Uses FK feedback only (H*, a, pin length, joint angles)."""
        r = self.robot
        H = r.hand_pose()
        back = -(self.L + BACK_GAP_M)
        if self.depth(H) > back:                                   # straight out first (pins may be in the holes)
            out = H.copy()
            out[:3, 3] += self.F[:, 0] * (back - self.depth(H))
            r.move_to(out)                                         # if IK refuses, the joint move below still backs out
        self._joint_move(self.q_back)
        far = offset_pose(self.H_star, self.F, back, start["lateral_m"], start["rotvec_deg"])
        near = offset_pose(self.H_star, self.F, -(self.L + start["gap_m"]), start["lateral_m"], start["rotvec_deg"])
        if not r.move_to(far):
            return "far"
        if not r.move_to(near):
            self._joint_move(self.q_back)
            return "near"
        # the arm's steady-state tracking error (~0.2 mm) is removed with FK feedback, so the start is where
        # it was drawn; the recorded part then continues from the last *commanded* pose
        cmd = near.copy()
        q = None
        for _ in range(3):
            q = r._ik(cmd, r.qpos() if q is None else q)
            if q is None or np.abs(q[:7] - r.qpos()[:7]).max() > 0.05:
                self._joint_move(self.q_back)
                return "settle"
            r._command(q, 20)
            cmd[:3, 3] += near[:3, 3] - r.hand_pose()[:3, 3]
        self.cmd0, self.q0 = cmd, q
        self.start_error_mm = float(np.linalg.norm(near[:3, 3] - r.hand_pose()[:3, 3]) * 1000)
        return "ok"

    # ------------------------------------------------------------ recorded motion
    def _segment(self, rec: Recorder, phase: str, cmd: np.ndarray, target: np.ndarray, speed: float, q_seed, steps_left: int):
        """Move the commanded pose from cmd to target in a straight line. Returns (cmd, q_seed, steps, why)."""
        r = self.robot
        dt, dr = pose_distance(cmd, target)
        n = max(1, int(np.ceil(max(dt / (speed * DT), dr / (np.radians(2.0) * DT)))))
        start = cmd
        for k in range(1, n + 1):
            if k > steps_left:
                return cmd, q_seed, k - 1, "time"
            nxt = interpolate_T(start, target, k / n)
            if self.depth(nxt) > DEPTH_CAP_M:
                return cmd, q_seed, k - 1, "depth"
            q = r._ik(nxt, q_seed)
            if q is None or np.abs(q[:7] - q_seed[:7]).max() > 0.03:   # no solution, or a jump to another branch
                return cmd, q_seed, k - 1, "ik"
            q = q.copy()
            q[:7] = q[:7].astype(np.float32).astype(float)
            r._command(q, 1)
            rec.on_step(phase, q, {})
            cmd, q_seed = nxt, q
            if np.linalg.norm(cmd[:3, 3] - r.hand_pose()[:3, 3]) > LAG_LIMIT_M:
                return cmd, q_seed, k, "blocked"
        return cmd, q_seed, n, "done"

    def _hold(self, rec: Recorder, q, n: int) -> int:
        for _ in range(n):
            self.robot._command(q, 1)
            rec.on_step("explore_hold", q, {})
        return n

    def run(self, rec: Recorder, segments: list[dict], rng: np.random.Generator) -> dict:
        """The recorded part: maneuver segments, then the straight push. Returns the segment log."""
        r = self.robot
        cmd, q_seed, used, log = self.cmd0.copy(), self.q0.copy(), 0, []
        rec.begin()
        for seg in segments:
            target = cmd.copy()
            target[:3, 3] = cmd[:3, 3] + self.F @ np.asarray(seg["move_m"])
            target[:3, :3] = Rotation.from_rotvec(self.F @ np.radians(seg["rotvec_deg"])).as_matrix() @ cmd[:3, :3]
            cmd, q_seed, n, why = self._segment(rec, "explore_maneuver", cmd, target, seg["speed_m_s"], q_seed, MAX_STEPS - used)
            used += n
            log.append({"kind": seg["kind"], "steps": n, "end": why})
            if why in ("ik", "time"):
                return {"segments": log, "stopped": why}
            if why == "blocked":
                used += self._hold(rec, q_seed, min(seg["hold_steps"], MAX_STEPS - used))
        target = cmd.copy()
        target[:3, 3] = cmd[:3, 3] + self.F[:, 0] * (0.0 - self.depth(cmd))            # to the inserted depth
        cmd, q_seed, n, why = self._segment(rec, "explore_push", cmd, target, PUSH_SPEED, q_seed, MAX_STEPS - used)
        used += n
        log.append({"kind": "final_push", "steps": n, "end": why})
        self._hold(rec, q_seed, 10 if why != "blocked" else int(rng.integers(8, 33)))
        return {"segments": log, "stopped": why}


def truth_outcome(rel7: np.ndarray) -> dict:
    """Outcome label of the final state (simulator truth; label / statistics only)."""
    d = diagnosis(rel7)
    depth, lat, ang = float(d["depth_left_m"] * 1000), float(d["lateral_m"] * 1000), float(d["angle_deg"])
    if bool(inserted(rel7)):
        o = "inserted"
    elif depth <= 1.0:
        o = "beside_board" if lat > 2.0 else "tilted"
    else:
        o = "not_inserted"
    return {"outcome": o, "depth_left_mm": depth, "lateral_mm": lat, "angle_deg": ang}


def collect_grasp(robot: Robot, g: dict, out: Path, n_traj: int, seed: int) -> dict:
    """g: a wm_v1 grasp.json (port, grasp, condition, pitch_deg, grasp_pos_m, episode_seed, H_star, axis)."""
    t0 = time.time()
    gdir = out / g["port"] / g["grasp"]
    gdir.mkdir(parents=True, exist_ok=True)
    summary = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    summary.update(trajectories=[], bytes=0)
    try:
        rng = np.random.default_rng(seed)
        expert = Expert(robot)
        robot.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            summary["error"] = "reference insertion failed"
            return summary
        H_star = robot.hand_pose()
        dt, dr = pose_distance(H_star, np.asarray(g["H_star"]))
        summary["H_star_vs_wm_v1_mm_deg"] = [dt * 1000, float(np.degrees(dr))]     # same seed: should agree
        ex = Explorer(robot, H_star, np.asarray(g["axis"]))
        if not ex.remember_standoff():
            summary["error"] = "standoff not reachable"
            return summary
        base = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
        label = {"H_star": H_star.tolist(), "axis": list(g["axis"])}
        for i in range(n_traj):
            segs, fails = sample_segments(rng), []
            for _ in range(PLACEMENT_ATTEMPTS):                     # a start the arm cannot reach is redrawn
                start = sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            entry = {"idx": i, "kind": "explore", "start": start, "placement_failures": fails}
            if stage != "ok":
                entry["outcome"] = "placement_failed"
                summary["trajectories"].append(entry)
                continue
            entry["start_error_mm"] = ex.start_error_mm
            rec = Recorder(robot)
            res = ex.run(rec, segs, rng)
            arrays = rec.end()
            entry.update(res, planned=segs, steps=int(len(arrays["action"])), **truth_outcome(arrays["eval_rel"][-1]))
            if len(arrays["action"]) < 4:                           # nothing happened (IK failed at once)
                entry["outcome"] = "empty"
                summary["trajectories"].append(entry)
                continue
            meta = {**base, **label, **entry}
            summary["bytes"] += store.save(gdir / f"t{i:02d}_explore.npz", arrays, meta)
            summary["trajectories"].append({k: v for k, v in entry.items() if k != "planned"})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-1500:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        (gdir / "grasp.json").write_text(json.dumps(summary))
    return summary
