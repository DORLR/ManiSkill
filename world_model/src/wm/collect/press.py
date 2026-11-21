"""The press set (2026-11-14, decided with the user): a policy that keeps commanding forward after the plug meets the board.

Why: the NWM-as-is run (runs/nwm_asis/plug_cdit_b, 20k and 40k steps) lets the picture keep changing when the command keeps
advancing from a blocked state (push check: picture change in 8 s, push 0.21 vs no motion 0.08, real camera 0.018). In every
other set the scripts re-plan the command from the measured hand, so when the hand is blocked the command stops too: "the
command advances" always comes with "the picture moves". In imagination there is no measured hand (NWM sees only images and
the action), so a policy's commands do keep advancing; the model must learn that a blocked hand stays where it is.

What a trajectory is (everything is drawn before the trajectory starts; no simulator truth is used for the motion):
    start     as the contact set (explore.Explorer.place: pins 2-6 mm in front of the board, shifted / rotated), but 65 %
              of the starts aligned (lateral <= 0.3 mm, rotation <= 0.5 deg) so that blocked and inserted are balanced
    requested the policy's command: a sequence of decisions (4 / 8 / 16 control steps), each a displacement of the
              commanded hand position in the frame (a, y, z), ramp or step. It is RECORDED as the trajectory's command
              (written as hand_cmd into <out>_ee, the file the NWM adapter reads), unbounded.
    executed  what the controller actually sends: the previous executed target + this step's requested displacement, then
              limited to lead the measured hand (FK) by at most 1.5 mm (along a gives way first; the same limit as every
              other set, so contact forces stay in the range already seen), and never deeper than inserted + 1 mm. The next
              displacement is added to the LIMITED target (no wind-up), so the hidden lead never exceeds 1.5 mm.
              A real robot does the same with a delta-pose interface and a safety clamp on the target.
    plan      1. press: forward along a at 0.5-1.5 mm per 0.4 s until the requested position is 0.7 x pin length to pin length + 10 mm
                 past where the pin tips meet the board (pin length is a property of the plug in the hand)
              2. (70 %) 1-4 more primitives until the 24 s budget: press_more (forward 2-8 s), hold, search (lateral
                 slide / wiggle / circle / spiral with a light forward press, as explore.sample_contact), retreat (back 2-10 mm
                 at 0.5-1.5 mm per 0.4 s)
Files: <out>/<port>/<grasp>/tNN_press.npz (data/store.py format, extra array req_pos (T, 3)) and <out>_ee/... (hand_cmd: the
requested poses, rotation of the executed command); meta: start, plan log, outcome (truth, label only).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from ..data import store
from ..data.recorder import Recorder
from ..env.expert import Expert
from ..env.robot import Robot
from .explore import CHUNK, DEPTH_CAP_M, LEAD_MAX_M, PLACEMENT_ATTEMPTS, Explorer, _unit, truth_outcome
from ..sim.geometry import pose_distance

MAX_STEPS = 480               # 24 s, as the other sets
PHASE = "explore_contact"     # phase label (bookkeeping only)


def sample_start(rng: np.random.Generator) -> dict:
    """65 % aligned (the press then inserts about 2 times in 3), 35 % as the contact set's near starts (usually blocked).
    Smoke check 2026-11-14 with 50 % aligned: 12 of 37 inserted; 65 % aims at about half."""
    gap = float(rng.uniform(0.002, 0.006))
    if rng.random() < 0.65:
        lat = _unit(rng, 2) * rng.uniform(0, 0.0003)
        rot = _unit(rng, 3) * rng.uniform(0, 0.5)
    else:
        u, v = rng.random(), rng.random()
        lat = _unit(rng, 2) * (rng.uniform(0.0005, 0.001) if u < 0.4 else rng.uniform(0.001, 0.003) if u < 0.8 else rng.uniform(0.003, 0.010))
        rot = _unit(rng, 3) * (rng.uniform(0, 1.0) if v < 0.5 else rng.uniform(1.0, 3.0) if v < 0.8 else rng.uniform(3.0, 6.0))
    return {"gap_m": gap, "lateral_m": lat.tolist(), "rotvec_deg": rot.tolist()}


def _decisions(rng, kind: str, n: int, m: int, v: float, lat_jitter: float = 0.0) -> dict:
    """n decisions of m steps; v = forward speed in m per 0.4 s (negative = back)."""
    d = np.zeros((n, 3))
    d[:, 0] = v * m / CHUNK
    if lat_jitter:
        d[:, 1:] = rng.normal(0, lat_jitter, (n, 2))
    return {"kind": kind, "chunk_steps": m, "chunks": n, "shape": "ramp" if rng.random() < 0.6 else "step", "deltas_m": d.tolist()}


def sample_plan(rng: np.random.Generator, pin_length_m: float, gap_m: float) -> list[dict]:
    m = int(rng.choice([4, 8, 16], p=[0.15, 0.7, 0.15]))
    v = float(rng.uniform(0.0005, 0.0015))
    travel = gap_m + float(rng.uniform(0.7 * pin_length_m, pin_length_m + 0.010))    # past where the pin tips meet the board
    # (smoke check 2026-11-14: 2-14 mm past the board was shorter than the pins (15-17 mm), so aligned starts stopped half in)
    n = int(np.ceil(travel / (v * m / CHUNK)))
    plan = [_decisions(rng, "press", n, m, v, 0.0001 if rng.random() < 0.3 else 0.0)]
    used = n * m
    if rng.random() < 0.7:
        for _ in range(int(rng.integers(1, 5))):
            if used >= MAX_STEPS:
                break
            kind = str(rng.choice(["press_more", "hold", "search", "retreat"], p=[0.3, 0.15, 0.3, 0.25]))
            m = int(rng.choice([4, 8, 16], p=[0.15, 0.7, 0.15]))
            if kind == "press_more":
                p = _decisions(rng, kind, int(rng.integers(5, 21)), m, float(rng.uniform(0.0005, 0.0015)))
            elif kind == "hold":
                p = _decisions(rng, kind, int(rng.integers(1, 6)), m, 0.0)
            elif kind == "retreat":
                vb = float(rng.uniform(0.0005, 0.0015))
                p = _decisions(rng, kind, max(1, int(np.ceil(rng.uniform(0.002, 0.010) / (vb * m / CHUNK)))), m, -vb)
            else:                                                 # as explore.sample_contact's search, plus a forward press
                pattern = str(rng.choice(["slide", "wiggle", "circle", "spiral"], p=[0.2, 0.25, 0.2, 0.35]))
                k = int(rng.integers(4, 17))
                d0, turn, mag = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.2) * rng.choice([-1, 1]), rng.uniform(0.0002, 0.001)
                if pattern == "spiral":
                    step, pitch, th, pts = rng.uniform(0.00025, 0.0005), rng.uniform(0.00025, 0.0005), 0.0, [np.zeros(2)]
                    for _ in range(k):
                        th += step / max(pitch * th / (2 * np.pi), step / 2)
                        pts.append(pitch * th / (2 * np.pi) * np.array([np.cos(d0 + np.sign(turn) * th), np.sin(d0 + np.sign(turn) * th)]))
                    lat = np.diff(np.array(pts), axis=0)
                else:
                    ang = {"slide": np.full(k, d0), "wiggle": d0 + np.pi * np.arange(k), "circle": d0 + turn * np.arange(k)}[pattern]
                    lat = mag * np.c_[np.cos(ang), np.sin(ang)]
                fwd = float(rng.uniform(0.0001, 0.0006)) * m / CHUNK
                p = {"kind": "search", "pattern": pattern, "chunk_steps": m, "chunks": k,
                     "shape": "ramp" if rng.random() < 0.6 else "step", "deltas_m": np.c_[np.full(k, fwd), lat].tolist()}
            used += p["chunks"] * p["chunk_steps"]
            plan.append(p)
    return plan


class Presser(Explorer):
    def press(self, rec: Recorder, plan: list[dict]) -> dict:
        """Run the plan from the placed start (self.cmd0, self.q0). Returns the log and the requested positions."""
        r, F = self.robot, self.F
        R_cmd = self.cmd0[:3, :3].copy()
        exe, req, q_seed = self.cmd0[:3, 3].copy(), self.cmd0[:3, 3].copy(), self.q0.copy()
        reqs, log, used, miss, missed, limited, why = [], [], 0, 0, 0, 0, "done"
        rec.begin()
        for p in plan:
            m, done = p["chunk_steps"], 0
            for j in range(p["chunks"]):
                if used + m > MAX_STEPS:
                    why = "time"
                    break
                delta = F @ np.asarray(p["deltas_m"][j], np.float64)
                for k in range(m):
                    step = delta / m if p["shape"] == "ramp" else (delta if k == 0 else np.zeros(3))
                    req, exe = req + step, exe + step
                    h = r.hand_pose()[:3, 3]
                    lead = F.T @ (exe - h)
                    lat = lead[1:] * min(1.0, LEAD_MAX_M / max(np.linalg.norm(lead[1:]), 1e-12))
                    room = np.sqrt(max(LEAD_MAX_M ** 2 - lat @ lat, 0.0))
                    new = h + F @ np.r_[np.clip(lead[0], -room, room), lat]
                    limited += int(np.linalg.norm(new - exe) > 1e-6)
                    exe = new
                    exe -= F[:, 0] * max(0.0, float(F[:, 0] @ (exe - self.H_star[:3, 3])) - DEPTH_CAP_M)
                    T = np.eye(4)
                    T[:3, :3], T[:3, 3] = R_cmd, exe
                    q = r._ik(T, q_seed)
                    if q is None or np.abs(q[:7] - q_seed[:7]).max() > 0.03:
                        q = r._ik(T, r.qpos())
                        if q is not None and np.abs(q[:7] - q_seed[:7]).max() > 0.03:
                            q = None
                    if q is None:                               # no solution: the last command is repeated
                        miss, missed, q = miss + 1, missed + 1, q_seed
                        if miss > m:
                            why = "ik"
                            break
                    else:
                        miss = 0
                    q = q.copy()
                    q[:7] = q[:7].astype(np.float32).astype(float)
                    r._command(q, 1)
                    rec.on_step(PHASE, q, {})
                    reqs.append(req.copy())
                    q_seed, used = q, used + 1
                if why == "ik":
                    break
                done += 1
            log.append({"kind": p["kind"], "pattern": p.get("pattern"), "chunks": done, "end": why})
            if why != "done":
                break
        for _ in range(4):                                      # as the other sets: a short hold at the end
            r._command(q_seed, 1)
            rec.on_step(PHASE, q_seed, {})
            reqs.append(req.copy())
        return {"log": log, "stopped": why, "steps": used, "ik_missed_steps": missed, "limited_steps": limited,
                "req": np.asarray(reqs), "R_cmd": R_cmd,
                "requested_past_executed_mm": float(F[:, 0] @ (req - exe) * 1000)}


def collect_grasp(robot: Robot, g: dict, out: Path, n_traj: int, seed: int) -> dict:
    """g: a wm_v1 grasp.json. Writes <out>/<port>/<grasp>/tNN_press.npz and the requested poses to <out>_ee/<same>."""
    t0 = time.time()
    gdir = out / g["port"] / g["grasp"]
    edir = Path(str(out) + "_ee") / g["port"] / g["grasp"]
    gdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
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
        summary["H_star_vs_wm_v1_mm_deg"] = [dt * 1000, float(np.degrees(dr))]
        ex = Presser(robot, H_star, np.asarray(g["axis"]))
        if not ex.remember_standoff():
            summary["error"] = "standoff not reachable"
            return summary
        base = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
        label = {"H_star": H_star.tolist(), "axis": list(g["axis"])}
        for i in range(n_traj):
            fails = []
            for _ in range(PLACEMENT_ATTEMPTS):
                start = sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            entry = {"idx": i, "kind": "press", "start": start, "placement_failures": fails, "pin_length_m": ex.L}
            if stage != "ok":
                entry["outcome"] = "placement_failed"
                summary["trajectories"].append(entry)
                continue
            plan = sample_plan(rng, ex.L, start["gap_m"])
            entry["start_error_mm"] = ex.start_error_mm
            rec = Recorder(robot)
            res = ex.press(rec, plan)
            arrays = rec.end()
            req, R_cmd = res.pop("req"), res.pop("R_cmd")
            entry.update(res, planned=plan, steps=int(len(arrays["action"])), **truth_outcome(arrays["eval_rel"][-1]))
            if len(arrays["action"]) < 4:
                entry["outcome"] = "empty"
                summary["trajectories"].append(entry)
                continue
            arrays["req_pos"] = req.astype(np.float32)
            meta = {**base, **label, **entry}
            summary["bytes"] += store.save(gdir / f"t{i:02d}_press.npz", arrays, meta)
            cmd = np.tile(np.eye(4), (len(req), 1, 1))
            cmd[:, :3, :3], cmd[:, :3, 3] = R_cmd, req
            np.savez(edir / f"t{i:02d}_press.npz", hand_cmd=cmd.astype(np.float32))
            summary["trajectories"].append({k: v for k, v in entry.items() if k != "planned"})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-1500:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        (gdir / "grasp.json").write_text(json.dumps(summary))
    return summary
