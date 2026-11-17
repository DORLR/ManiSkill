"""Closed-loop test of a policy alone: one attempt per start, the demonstrations' start box (collect/branch.py).

Stops (changed 2026-11-17 after the code review). stop="free" (default): an attempt ends only on the time limit, on
a stall of the hand (branch.py's STALL_DECISIONS / STALL_M), or when the hand is more than LEFT_FREE_M from where the
attempt began. Nothing derived from the reference insertion of the test grasp decides when an attempt ends: the old
harness stopped as soon as the hand was "seated" against H*, which a new grasp does not have and which froze
successes. stop="reference" keeps the old stops for comparison, with the success criterion's 4 deg instead of 2.

The reference insertion is still run for every test grasp, for two things only: placing the start poses (the start
box is defined relative to the seated pose and the insertion axis) and the label `seated_fk`. The policy never sees
either. Success is judged on the simulator's truth after the final hold, with the one criterion (demo320.SUCCESS).
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import torch

from ..collect import branch as B
from ..collect.branch import Brancher, FullRecorder
from ..collect.explore import truth_outcome
from .demo320 import SUCCESS, is_success

LEFT_FREE_M = 0.06                     # start gap <= 18 mm + pins <= 20 mm + lateral <= 10 mm: a hand 6 cm away has left
SEATED_DEG = SUCCESS["angle_deg"]


class PolicyTester(Brancher):
    def attempt(self, agent, seed: int, max_decisions: int, stop: str = "free") -> dict:
        assert stop in ("free", "reference")
        r, E = self.robot, B.EXECUTE
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rec = FullRecorder(r)
        rec.begin()
        agent.reset()
        why, q_last, k = "time", r.qpos()[:7], 0
        hands = [np.array(r.hand_pose()[:3, 3])]
        for k in range(max_decisions):
            targets = np.asarray(agent.act(rec, 1), np.float64)[0][:E]
            for j in range(E):
                q = targets[j].astype(np.float32).astype(float)
                r._command(q, 1)
                rec.step(q, targets[j])
                q_last = q
            hand = np.array(r.hand_pose()[:3, 3])
            if stop == "reference":
                s = self.fk_state()
                if s["dist_m"] < B.STOP_SEATED_M and s["angle_deg"] < SEATED_DEG:
                    why = "seated"
                    break
                if s["across_m"] > B.LEFT_M or s["along_m"] > B.PAST_M:
                    why = "left"
                    break
            elif np.linalg.norm(hand - hands[0]) > LEFT_FREE_M:
                why = "left"
                break
            hands.append(hand)
            if len(hands) > B.STALL_DECISIONS and np.linalg.norm(hands[-1] - hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
                why = "stalled"
                break
        for _ in range(4):
            r._command(q_last, 1)
            rec.step(q_last, q_last)
        end = self.fk_state()                                  # label only
        truth = truth_outcome(rec.end()["eval_rel"][-1])
        return {"stopped": why, "decisions": k + 1, "seated_fk": bool(end["dist_m"] < B.SEATED_M and end["angle_deg"] < SEATED_DEG),
                "success": is_success(truth), **{key: float(truth[key]) for key in SUCCESS}}


def test_grasp_policy(robot, agent, g: dict, out_file, n_starts: int, seed: int, max_decisions: int = B.MAX_DECISIONS, stop: str = "free") -> dict:
    """All starts of one grasp. The file always gets written: with `error` and `error_kind` = "setup" (the reference
    insertion or the stand-off failed: a property of the grasp) or "crash" (an exception) when it did not run through."""
    from ..env.expert import Expert
    t0 = time.time()
    summary = {k: g[k] for k in ("grasp", "port", "condition", "pitch_deg", "grasp_pos_m", "episode_seed")}
    summary.update(stop=stop, starts=[], error="interrupted", error_kind="crash")     # stays if anything ends this function early
    try:
        rng = np.random.default_rng(seed)
        expert = Expert(robot)
        robot.reset(g["episode_seed"], g["pitch_deg"], g["grasp_pos_m"])
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            summary.update(error="reference insertion failed", error_kind="setup")
            return summary
        ex = PolicyTester(robot, robot.hand_pose(), np.asarray(g["axis"]))
        if not ex.remember_standoff():
            summary.update(error="standoff not reachable", error_kind="setup")
            return summary
        for i in range(n_starts):
            fails = []
            for _ in range(B.PLACEMENT_ATTEMPTS):
                start = B.sample_start(rng)
                stage = ex.place(start)
                if stage == "ok":
                    break
                fails.append(stage)
            entry = {"idx": i, "start": start, "placement_failures": fails}
            summary["starts"].append(entry)
            if stage != "ok":
                entry["placement_failed"] = True
                continue
            ex.snapshot()
            ex.restore()
            entry["attempt"] = ex.attempt(agent, int(seed + 1000 * i), max_decisions, stop)
        del summary["error"], summary["error_kind"]
    except Exception as exc:
        summary.update(error=f"{type(exc).__name__}: {exc}", error_kind="crash", trace=traceback.format_exc()[-2000:])
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary))
    return summary
