"""Closed-loop test, second part (the user, 2026-11-16: why go all the way back to the start?): after a blocked or
called-off attempt the robot backs off only a few decisions and tries again from there.

Back-off point: the joint targets commanded BACK_DECISIONS decisions before the pins reached the board face in this
attempt (about 2 s earlier; a place the attempt has been). Pins in the holes: first straight out along the port axis
to 2 mm in front of the face. The policy keeps the port axes it fitted at the start of the mode; only its picture
history is cleared. Everything else as closed_loop.py (same starts and seeds, so the modes of both files are paired).

Modes
  near4            up to 4 attempts                                   (against `retry`: success per attempt)
  near_time        attempts until TIME_BUDGET_S of robot time is used (against `retry`: success for the same time)
  near_abort_time  near_time + the model calls an attempt off early
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np

from ..collect import branch as B
from ..collect.branch import FullRecorder, sample_noise
from .closed_loop import MAX_ATTEMPTS, Scorer, Tester

MODES = ("near4", "near_time", "near_abort_time")
BACK_DECISIONS = 5
OUT_GAP_M = 0.002
TIME_BUDGET_S, MAX_NEAR_ATTEMPTS = 120.0, 16
STEP_S = 0.05


class KeepAxes:
    """The policy agent, but from the second attempt on `reset` keeps the fitted port axes and clears the history only."""

    def __init__(self, agent):
        self.agent, self.keep = agent, False

    def reset(self, frame, seed=0) -> bool:
        if self.keep and getattr(self.agent, "R_P", True) is not None:
            self.agent.context.clear()
            return True
        return self.agent.reset(frame, seed)

    def act(self, frame, n_samples=1):
        return self.agent.act(frame, n_samples)


class NearTester(Tester):
    def back_off(self, executed: list, at_face, decisions: int) -> tuple[int, float]:
        """-> control steps spent, FK distance of the hand to the board face along the axis afterwards [mm]."""
        r, E = self.robot, B.EXECUTE
        k = max((at_face if at_face is not None else decisions) - BACK_DECISIONS, 0)
        target, steps = executed[min(E * k, len(executed) - 1)], 0
        H = r.hand_pose()
        if self.depth(H) > -(self.L - 0.0005):                    # pins in the holes: straight out first
            out = H.copy()
            out[:3, 3] += self.F[:, 0] * (-(self.L + OUT_GAP_M) - self.depth(H))
            r.move_to(out)
            steps += 80
            if self.depth(r.hand_pose()) > -self.L:
                self._joint_move(self.q_back)
                steps += 60
            self._joint_move(target)
            steps += 40
        else:
            for q in reversed(executed[E * k:]):
                r._command(q, 1)
                steps += 1
            r._command(target, 20)
            steps += 20
        return steps, float((-self.L - self.depth(r.hand_pose())) * 1000)

    def run_mode_near(self, mode: str, agent, scorer: Scorer, seed: int, axes_seed: int, abort_below: float, max_decisions: int) -> dict:
        timed = mode in ("near_time", "near_abort_time")
        thr = abort_below if mode == "near_abort_time" else None
        proxy = KeepAxes(agent)
        rng = np.random.default_rng(seed + 77)
        out, steps = [], 0
        for i in range(MAX_NEAR_ATTEMPTS if timed else MAX_ATTEMPTS):
            noise = sample_noise(rng, i)
            left = max_decisions if not timed else max(int((TIME_BUDGET_S / STEP_S - steps) / B.EXECUTE), 1)
            res = self.attempt(FullRecorder(self.robot), proxy, scorer, seed + i, axes_seed, noise, None, thr, min(max_decisions, left))
            proxy.keep = True
            executed = res.pop("executed")
            steps += len(executed)
            res["sigma_m"] = noise["sigma_m"]
            out.append(res)
            if res["seated_fk"] or res["stopped"] == "no_board_plane":
                break
            if (timed and steps * STEP_S >= TIME_BUDGET_S) or (not timed and i == MAX_ATTEMPTS - 1):
                break
            used, gap = self.back_off(executed, res["at_face"], res["decisions"])
            res["backed_off_to_mm_before_face"] = gap
            steps += used
        return {"attempts": out, "success": out[-1]["outcome"] == "inserted", "ever_inserted": any(a["outcome"] == "inserted" for a in out),
                "seated_fk": out[-1]["seated_fk"], "control_steps": steps}


def test_grasp_near(robot, agent, scorer: Scorer, g: dict, out_file, n_starts: int, seed: int, modes, abort_below: float,
                    max_decisions: int = B.MAX_DECISIONS) -> dict:
    """As closed_loop.test_grasp (same reference insertion, same starts from the same seed), with the near modes."""
    from ..env.expert import Expert
    t0 = time.time()
    summary = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    summary["starts"] = []
    try:
        rng = np.random.default_rng(seed)
        expert = Expert(robot)
        robot.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            summary["error"] = "reference insertion failed"
            return summary
        ex = NearTester(robot, robot.hand_pose(), np.asarray(g["axis"]))
        if not ex.remember_standoff():
            summary["error"] = "standoff not reachable"
            return summary
        for i in range(n_starts):
            fails = []
            for _ in range(B.PLACEMENT_ATTEMPTS):
                start = B.sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            entry = {"idx": i, "start": start, "placement_failures": fails, "modes": {}}
            summary["starts"].append(entry)
            if stage != "ok":
                entry["outcome"] = "placement_failed"
                continue
            ex.snapshot()
            for mode in modes:
                dev = ex.restore()
                t1 = time.time()
                res = ex.run_mode_near(mode, agent, scorer, int(seed + 1000 * i), int(seed + 1000 * i), abort_below, max_decisions)
                entry["modes"][mode] = {**res, "restore_dev_mm": dev[0], "seconds": round(time.time() - t1, 1)}
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-2000:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary))
    return summary


def summarise_near(files) -> dict:
    rows = []
    for f in files:
        s = json.loads(open(f).read())
        rows += [st["modes"] for st in s["starts"] if st.get("modes")]
    rep = {"starts": len(rows)}
    for m in [m for m in MODES if rows and all(m in r for r in rows)]:
        x = np.array([r[m]["success"] for r in rows], float)
        e = {"success": float(x.mean()), "se": float(x.std() / len(x) ** 0.5),
             "ever_inserted": float(np.mean([r[m]["ever_inserted"] for r in rows])),
             "attempts_mean": float(np.mean([len(r[m]["attempts"]) for r in rows])),
             "seconds_of_robot_time_mean": float(np.mean([r[m]["control_steps"] for r in rows]) * STEP_S), "stopped": {},
             "success_by_attempt": {}}
        for r in rows:
            for i, a in enumerate(r[m]["attempts"]):
                key = f"{a['stopped']}/{a['outcome']}"
                e["stopped"][key] = e["stopped"].get(key, 0) + 1
            if r[m]["success"]:
                n = str(len(r[m]["attempts"]))
                e["success_by_attempt"][n] = e["success_by_attempt"].get(n, 0) + 1
        gaps = [a["backed_off_to_mm_before_face"] for r in rows for a in r[m]["attempts"] if "backed_off_to_mm_before_face" in a]
        if gaps:
            e["backed_off_to_mm_before_face_pct_10_50_90"] = np.percentile(gaps, [10, 50, 90]).tolist()
        rep[m] = e
    return rep
