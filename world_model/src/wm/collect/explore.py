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
    contact     (recorded, `contact=True`, 2026-11-12) after the final push, 4-16 s more at the board. Why:
                closed-loop policies stay pressed against the board for many seconds and re-plan from where
                the hand is; the first exploration set ends 0.4-1.6 s after the block, so neither such
                states nor what any action does from them was in the data. The phase is a random sequence of
                    hold      the command stays where it is
                    lead      "decisions" of 4 / 8 / 16 commands, each a ramp (or a step) from the measured
                              hand position (FK) to that position + a lead of 0.1-1.5 mm along a (sometimes a
                              little across it, sometimes a small rotation): what a controller that re-plans
                              from its measured state commands while it keeps pushing. The lead never exceeds
                              1.5 mm (the force limit above).
                    search    the same with a lead across a that slides / alternates / circles / is random, or
                              walks an outward spiral (0.25-0.5 mm steps and pitch; the blind search of
                              peg-in-hole practice), so that some pressed states do end inserted
                    back_off  the same with a negative lead (lets go of the board), sometimes with a side step
                drawn before the trajectory starts; only FK feedback is used. Decisions start on multiples of
                8 control steps, as a policy's decisions do.
                Across a, the leads of a primitive refer either to the measured hand position of each
                decision (30 %), or (`anchored`, 70 %) to a position that starts at the hand and is moved by
                the leads themselves: a controller that keeps its place across a while it pushes. Measured on
                the first trials: without it the hand, pressed on the board and re-planned from where it is,
                slides away steadily (up to 26 mm in 15 s), because each decision accepts the slip.
                The phase stays within about 3 mm (across a) of where it began: a lead pointing further out
                is mirrored, and from outside a decision only returns (up to 1.2 mm back, hardly pushing).
                The hand's own positions only, not the hole's: the commands still say nothing about where
                the hole is. When the whole lead would exceed 1.5 mm, the part along a gives way.
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
CHUNK = 8                    # one decision of a closed-loop controller: 8 commands (0.4 s)
LEAD_MAX_M = 0.0015          # a decision's last command never leads the measured hand by more than this
WANDER_M = 0.0025            # the contact phase stays within this distance (across a) of where it began


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


def sample_start(rng: np.random.Generator, near: bool = False) -> dict:
    """Start in front of the board; truth-free by construction (relative to H*). `near` (contact set): fewer
    aligned and far starts, more within 3 mm, where pressing and searching can change the outcome."""
    a, b = (0.55, 0.85) if near else (0.4, 0.7)
    if rng.random() < (0.05 if near else 0.1):
        lat, rot = np.zeros(2), np.zeros(3)                       # aligned: a clean positive
    else:
        u, v = rng.random(), rng.random()
        lat = _unit(rng, 2) * (rng.uniform(0, 0.001) if u < a else rng.uniform(0.001, 0.003) if u < b else rng.uniform(0.003, 0.010))
        rot = _unit(rng, 3) * (rng.uniform(0, 1.0) if v < 0.5 else rng.uniform(1.0, 3.0) if v < 0.8 else rng.uniform(3.0, 6.0))
    return {"gap_m": float(rng.uniform(0.002, 0.006)), "lateral_m": lat.tolist(), "rotvec_deg": rot.tolist()}


def sample_segments(rng: np.random.Generator, p=(0.3, 0.3, 0.25, 0.15)) -> list[dict]:
    """0-3 random maneuver segments, as motions of the commanded hand pose in the frame (a, y, z)."""
    segs = []
    for _ in range(int(rng.choice([0, 1, 2, 3], p=list(p)))):
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


def sample_contact(rng: np.random.Generator) -> list[dict]:
    """The contact phase: primitives for 4-16 s (80-320 control steps). Every number is drawn here, before
    the trajectory starts. `leads_m`: per decision the lead (along a, y, z) of its last command over the
    measured hand position."""
    budget, used, prims = int(rng.integers(80, 321)), 0, []
    while used < budget:
        kind = str(rng.choice(["hold", "lead", "search", "back_off"], p=[0.15, 0.40, 0.30, 0.15]))
        n = int(rng.integers(1, 5)) if kind == "back_off" else int(rng.integers(2, 13))
        p = {"kind": kind, "chunk_steps": CHUNK, "chunks": n}
        if kind != "hold":
            p["chunk_steps"] = int(rng.choice([4, 8, 16], p=[0.15, 0.7, 0.15]))
            p["shape"] = "ramp" if rng.random() < 0.6 else "step"
            p["anchored"] = bool(rng.random() < 0.7)
            if kind == "lead":
                lat = _unit(rng, 2) * rng.uniform(0, 0.0006) if rng.random() < 0.4 else np.zeros(2)
                leads = np.tile(np.r_[rng.uniform(0.0001, LEAD_MAX_M), lat], (n, 1))
                if rng.random() < 0.15:                             # a small rotation per decision, <= 2 deg in all
                    p["rotvec_deg"] = (_unit(rng, 3) * min(rng.uniform(0.05, 0.3), 2.0 / n)).tolist()
            elif kind == "search":
                pattern = str(rng.choice(["slide", "wiggle", "circle", "random", "spiral"], p=[0.15, 0.2, 0.15, 0.2, 0.3]))
                d0, turn, mag = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.2) * rng.choice([-1, 1]), rng.uniform(0.0002, 0.0012)
                if pattern == "spiral":                             # the blind search of peg-in-hole practice: an outward
                    n = p["chunks"] = int(rng.integers(8, 25))      # spiral around where the hand is, lightly pressed
                    step, pitch, th, pts = rng.uniform(0.00025, 0.0005), rng.uniform(0.00025, 0.0005), 0.0, [np.zeros(2)]
                    for _ in range(n):
                        th += step / max(pitch * th / (2 * np.pi), step / 2)
                        pts.append(pitch * th / (2 * np.pi) * np.array([np.cos(d0 + turn / abs(turn) * th), np.sin(d0 + turn / abs(turn) * th)]))
                    leads = np.c_[np.full(n, rng.uniform(0.0001, 0.0006)), np.diff(np.array(pts), axis=0)]
                    p["anchored"] = True
                else:
                    ang = {"slide": np.full(n, d0), "wiggle": d0 + np.pi * np.arange(n), "circle": d0 + turn * np.arange(n),
                           "random": rng.uniform(0, 2 * np.pi, n)}[pattern]
                    leads = np.c_[np.full(n, rng.uniform(0.0001, 0.001)), mag * np.cos(ang), mag * np.sin(ang)]
                p["pattern"] = pattern
            else:
                lat = _unit(rng, 2) * rng.uniform(0, 0.001) if rng.random() < 0.5 else np.zeros(2)
                leads = np.tile(np.r_[-rng.uniform(0.0002, 0.001), lat], (n, 1))
            norm = np.linalg.norm(leads, axis=1, keepdims=True)
            p["leads_m"] = (leads * np.minimum(1.0, LEAD_MAX_M / np.maximum(norm, 1e-12))).tolist()
        used += n * p["chunk_steps"]
        prims.append(p)
    return prims


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

    def _hold(self, rec: Recorder, q, n: int, phase: str = "explore_hold") -> int:
        for _ in range(n):
            self.robot._command(q, 1)
            rec.on_step(phase, q, {})
        return n

    def _contact(self, rec: Recorder, cmd: np.ndarray, q_seed, plan: list[dict], used: int):
        """The contact phase (module docstring). Returns (steps, q_seed, log, why)."""
        r, start, log, why, miss, missed = self.robot, used, [], "done", 0, 0
        used += self._hold(rec, q_seed, min((-used) % CHUNK, MAX_STEPS - used), "explore_contact")   # the decision clock
        R_cmd, origin, Fl, ref = cmd[:3, :3].copy(), r.hand_pose()[:3, 3].copy(), self.F[:, 1:], np.zeros(2)
        for p in plan:
            m, done = p["chunk_steps"], 0
            for j in range(p["chunks"]):
                if used + m > MAX_STEPS:
                    why = "time"
                    break
                if p["kind"] == "hold":
                    used += self._hold(rec, q_seed, m, "explore_contact")
                    done += 1
                    continue
                base, lead = r.hand_pose()[:3, 3].copy(), np.asarray(p["leads_m"][j], np.float64)
                hand_lat = Fl.T @ (base - origin)                   # across a, from where the phase began
                if j == 0 or not p["anchored"]:
                    ref = hand_lat                                  # the position across a that the commands refer to
                n_ref = np.linalg.norm(ref)
                if n_ref > WANDER_M + 0.0005:                       # outside (slid there): a decision that only returns
                    lead = np.r_[min(lead[0], 0.0002), -ref / n_ref * min(n_ref, 0.0012)]
                elif np.linalg.norm(ref + lead[1:]) > WANDER_M and ref @ lead[1:] > 0:
                    u = ref / n_ref
                    lead[1:] -= 2 * (lead[1:] @ u) * u              # a lead pointing further out is mirrored
                lat = ref + lead[1:] - hand_lat                     # the last command's lead over the measured hand, across a
                lat *= min(1.0, LEAD_MAX_M / max(np.linalg.norm(lat), 1e-12))
                room = np.sqrt(max(LEAD_MAX_M ** 2 - lat @ lat, 0.0))     # the limit is on the whole lead: the part along a
                vec = np.r_[np.clip(lead[0], -room, room), lat]            # gives way, so the place across a is kept
                ref = hand_lat + lat
                if "rotvec_deg" in p:
                    R_cmd = Rotation.from_rotvec(self.F @ np.radians(p["rotvec_deg"])).as_matrix() @ R_cmd
                for k in range(1, m + 1):
                    nxt = np.eye(4)
                    nxt[:3, :3] = R_cmd
                    nxt[:3, 3] = base + self.F @ (vec * (k / m if p["shape"] == "ramp" else 1.0))
                    nxt[:3, 3] -= self.F[:, 0] * max(0.0, self.depth(nxt) - DEPTH_CAP_M)     # never deeper than the cap
                    q = r._ik(nxt, q_seed)
                    if q is None or np.abs(q[:7] - q_seed[:7]).max() > 0.03:
                        q = r._ik(nxt, r.qpos())                    # once more from the measured joints
                        if q is not None and np.abs(q[:7] - q_seed[:7]).max() > 0.03:
                            q = None
                    if q is None:                                   # no solution for this command: the last one is repeated
                        miss, missed, q = miss + 1, missed + 1, q_seed
                        if miss > m:
                            why = "ik"
                            break
                    else:
                        miss = 0
                    q = q.copy()
                    q[:7] = q[:7].astype(np.float32).astype(float)
                    r._command(q, 1)
                    rec.on_step("explore_contact", q, {})
                    q_seed, used = q, used + 1
                if why == "ik":
                    break
                done += 1
            log.append({"kind": p["kind"], "pattern": p.get("pattern"), "chunks": done, "end": why})
            if why != "done":
                break
        if log:
            log[-1]["ik_missed_steps"] = missed
        return used - start, q_seed, log, why

    def run(self, rec: Recorder, segments: list[dict], rng: np.random.Generator, contact: list[dict] | None = None) -> dict:
        """The recorded part: maneuver segments, the straight push, then (`contact`) the contact phase.
        Returns the segment log."""
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
        if contact is not None and why in ("blocked", "done", "depth"):
            n, q_seed, clog, cwhy = self._contact(rec, cmd, q_seed, contact, used)
            self._hold(rec, q_seed, 4, "explore_contact")
            return {"segments": log, "stopped": why, "contact": clog, "contact_steps": n, "contact_stopped": cwhy}
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


def collect_grasp(robot: Robot, g: dict, out: Path, n_traj: int, seed: int, contact: bool = False) -> dict:
    """g: a wm_v1 grasp.json (port, grasp, condition, pitch_deg, grasp_pos_m, episode_seed, H_star, axis).
    `contact`: the contact set -- starts nearer the hole, fewer maneuvers, and the contact phase after the push."""
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
            segs, fails = (sample_segments(rng, (0.5, 0.3, 0.15, 0.05)) if contact else sample_segments(rng)), []
            plan = sample_contact(rng) if contact else None
            for _ in range(PLACEMENT_ATTEMPTS):                     # a start the arm cannot reach is redrawn
                start = sample_start(rng, near=contact)
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
            res = ex.run(rec, segs, rng, plan)
            arrays = rec.end()
            entry.update(res, planned=segs, planned_contact=plan, steps=int(len(arrays["action"])), **truth_outcome(arrays["eval_rel"][-1]))
            if len(arrays["action"]) < 4:                           # nothing happened (IK failed at once)
                entry["outcome"] = "empty"
                summary["trajectories"].append(entry)
                continue
            meta = {**base, **label, **entry}
            summary["bytes"] += store.save(gdir / f"t{i:02d}_explore.npz", arrays, meta)
            summary["trajectories"].append({k: v for k, v in entry.items() if k not in ("planned", "planned_contact")})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-1500:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        (gdir / "grasp.json").write_text(json.dumps(summary))
    return summary
