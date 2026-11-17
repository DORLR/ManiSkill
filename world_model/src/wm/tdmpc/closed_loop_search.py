"""Closed-loop test, third part (2026-11-16 evening): touch the board, notice the attempt fails, back off a little,
shift sideways, try again -- and compare WHERE to shift.

Measured before (scripts/analysis/probe_lateral_direction.py, held-out attempts): blocked attempts reach the board face
1.65 mm off (median; 80 % within 3 mm); a lateral difference made within 0.8 s of the face is kept by the policy
(92 %); V does not know which position is closer to the hole, but a linear readout of the latent gives the direction
of the lateral error within 45 deg in about half of the cases (chance: a quarter).

One mode = first attempt by the policy with the model's early stop (as mode `abort` of closed_loop.py); if it fails
after reaching the face, probes follow: straight back along the port axis to GAP_M in front of the face, sideways to a
candidate offset from where the hand first met the face, then the policy inserts again (early stop as before). Offsets:
6 directions at 1.2 mm, 12 at 2.4 mm; each is tried at most once; until seated (FK) or the time budget is used.
  search_blind    candidates in spiral order (inner ring, then outer; random starting angle and sense)
  search_dir      candidates ordered by the angle to the direction a linear readout of the latent gives, nearer ring
                  first. Readout fitted on training grasps against the simulator's lateral error of the plug
                  (scripts/analysis/fit_truth_readout.py; held-out: within 45 deg in 51 % at the face, chance 25 %)
  search_dir_fk   the same with the readout fitted against the joint-only label, the hand's offset from H*
                  (scripts/analysis/fit_lateral_readout.py; 42 %): that label is not the pins' offset when the hand is
                  rotated against H* (lever arm of the plug; it differs from the truth by 0.33 mm median, 1.15 mm at
                  the 90th percentile)
  search_oracle   candidates ordered by distance to the true correction (simulator; upper bound)
The policy keeps the port axes it fitted at the start; its picture history is cleared before every probe.
"""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import torch

from ..collect import branch as B
from ..collect.branch import FullRecorder
from scipy.spatial.transform import Rotation

from ..collect.explore import truth_outcome
from ..data import recorder as _rec
from .closed_loop import ABORT_COUNT, ABORT_FROM, Scorer, Tester
from .closed_loop_near import STEP_S, KeepAxes
from .data import FACE_MARGIN_M

MODES = ("search_blind", "search_dir", "search_dir_fk", "search_oracle")
# Ceiling checks (smoke test 2026-11-16: with 40 decisions per probe and the early stop, not even the oracle order
# inserted): ONE probe after the first attempt, no early stop, 80 decisions, the hand put
#   ceil_lat       at the true lateral correction, orientation as at the first contact, 0.8 mm in front of the face
#   ceil_pose      at the reference pose H* moved straight back to 0.8 mm in front of the face (position and orientation)
#   ceil_pose_far  the same, 5 mm in front of the face
CEILING_MODES = ("ceil_lat", "ceil_pose", "ceil_pose_far")
CEILING_DECISIONS = 80
GAP_M = 0.0008
RINGS = ((0.0012, 6), (0.0024, 12))
TIME_BUDGET_S, MAX_PROBES, PROBE_DECISIONS = 120.0, 18, 40


class Readout:
    """A lateral error (two numbers, mm) from the latent. `frame`: "goal" = (y, z) of the socket frame (fitted on the
    simulator's plug error), "axis" = (y, z) of the port-axis frame F (fitted on the hand's offset from H*)."""

    def __init__(self, path: str, frame: str):
        f = np.load(path)
        self.W, self.mu, self.sd, self.ym, self.frame = f["W"], f["mu"], f["sd"], f["ymean"], frame

    def __call__(self, z: torch.Tensor) -> np.ndarray:
        return ((z[0].double().cpu().numpy() - self.mu) / self.sd) @ self.W + self.ym


def ring_candidates(rng) -> list[np.ndarray]:
    phase, sense = rng.uniform(0, 2 * np.pi), rng.choice([-1.0, 1.0])
    return [np.array([r * np.cos(phase + sense * 2 * np.pi * i / n), r * np.sin(phase + sense * 2 * np.pi * i / n)])
            for r, n in RINGS for i in range(n)]


class SearchTester(Tester):
    def lateral(self, T: np.ndarray) -> np.ndarray:
        """Hand position across the axis relative to H* [m] (FK and H*: oracle arm and logs only)."""
        return (self.F.T @ (T[:3, 3] - self.H_star[:3, 3]))[1:]

    def goal_rotation(self) -> np.ndarray:
        q = np.asarray(_rec._copy_pose(self.robot.base.goal_pose).q, float).ravel()          # w, x, y, z
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def attempt_plain(self, agent, scorer: Scorer, readouts: dict | None, seed: int, axes_seed: int, abort_below: float,
                      max_decisions: int) -> dict:
        """The policy as it is, with the model's early stop. Also returns, for the first decision that starts at the
        face: the hand pose (`contact_T`), the plug's true lateral error there as a world vector (`contact_err`, for the
        oracle arm and the logs), and the readouts' estimates (mean of that decision and the next)."""
        r, E = self.robot, B.EXECUTE
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rec = FullRecorder(r)
        rec.begin()
        if not agent.reset(rec.frame, axes_seed):
            return {"stopped": "no_board_plane", "decisions": 0, "steps": 0, "seated_fk": False, "outcome": "not_inserted", "contact_T": None,
                    "contact_err": None, "preds": {}, "v_first8": [], "v_last": None}
        why, q_last, steps = "time", r.qpos()[:7], 0
        hands, at_face, low, contact_T, contact_err, v_log = [r.hand_pose()[:3, 3]], None, 0, None, None, []
        preds = {name: [] for name in (readouts or {})}
        k = 0
        for k in range(max_decisions):
            targets = np.asarray(agent.act(rec.frame, n_samples=1), np.float64)[0][:E]
            z = scorer.latent(rec)
            v0 = float(scorer.model.scalar(scorer.model.V(z))[..., 0].mean())
            v_log.append(round(v0, 2))
            if at_face is not None and k - at_face in (1, 2):
                if k - at_face == 1:
                    contact_T = r.hand_pose().copy()
                    contact_err = self.goal_rotation()[:, 1:] @ np.asarray(_rec._truth_rel(r)[1:3], float)
                for name, ro in (readouts or {}).items():
                    preds[name].append(ro(z))
            for j in range(E):
                q = targets[j].astype(np.float32).astype(float)
                r._command(q, 1)
                rec.step(q, targets[j])
                q_last = q
                steps += 1
            s = self.fk_state()
            if s["dist_m"] < B.STOP_SEATED_M and s["angle_deg"] < B.SEATED_DEG:
                why = "seated"
                break
            if s["across_m"] > B.LEFT_M or s["along_m"] > B.PAST_M:
                why = "left"
                break
            hands.append(r.hand_pose()[:3, 3])
            if len(hands) > B.STALL_DECISIONS and np.linalg.norm(hands[-1] - hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
                why = "stalled"
                break
            if at_face is None and s["along_m"] >= -(self.L + FACE_MARGIN_M):
                at_face = k
            if at_face is not None and k - at_face >= ABORT_FROM:
                low = low + 1 if v0 < abort_below else 0
                if low >= ABORT_COUNT:
                    why = "aborted"
                    break
        for _ in range(4):
            r._command(q_last, 1)
            rec.step(q_last, q_last)
            steps += 1
        end = self.fk_state()
        arrays = rec.end()
        return {"stopped": why, "decisions": k + 1, "steps": steps, "at_face": at_face,
                "seated_fk": bool(end["dist_m"] < B.SEATED_M and end["angle_deg"] < B.SEATED_DEG),
                **truth_outcome(arrays["eval_rel"][-1]), "contact_T": contact_T, "contact_err": contact_err,
                "preds": {name: np.mean(v, 0) for name, v in preds.items() if v}, "v_first8": v_log[:8], "v_last": v_log[-1]}

    def go_to(self, base_T: np.ndarray, offset_m: np.ndarray, gap: float = GAP_M) -> tuple[bool, int]:
        """Straight back along the axis (if the pins are at or in the board), then to `base_T` moved back to `gap` in
        front of the face and sideways by `offset_m`."""
        r, steps = self.robot, 0
        H = r.hand_pose()
        if self.depth(H) > -(self.L + gap):
            out = H.copy()
            out[:3, 3] += self.F[:, 0] * (-(self.L + gap) - self.depth(H))
            r.move_to(out)
            steps += 60
            if self.depth(r.hand_pose()) > -self.L:
                self._joint_move(self.q_back)
                steps += 60
        T = base_T.copy()
        T[:3, 3] += self.F[:, 0] * (-(self.L + gap) - self.depth(base_T)) + self.F[:, 1:] @ np.asarray(offset_m)
        ok = bool(r.move_to(T))
        return ok, steps + 60

    def run_mode_search(self, mode: str, agent, scorer: Scorer, readouts: dict, seed: int, axes_seed: int, abort_below: float,
                        max_decisions: int) -> dict:
        proxy = KeepAxes(agent)
        first = self.attempt_plain(proxy, scorer, readouts, seed, axes_seed, abort_below, max_decisions)
        proxy.keep = True
        contact_T, err, preds = first.pop("contact_T"), first.pop("contact_err"), first.pop("preds")
        out, steps, info = [first], first["steps"], {}
        if not first["seated_fk"] and contact_T is not None and first["stopped"] != "no_board_plane":
            cands = ring_candidates(np.random.default_rng(seed + 91))
            want = -(self.F[:, 1:].T @ err)                                   # the true correction of the hand, frame F [m]
            d_true = want / (np.linalg.norm(want) + 1e-12)
            info["contact_lateral_mm"] = float(np.linalg.norm(want) * 1000)
            info["hand_offset_from_H_star_mm"] = float(np.linalg.norm(self.lateral(contact_T)) * 1000)
            dirs = {}
            for name, p in preds.items():                                    # each readout's estimate as a direction in F
                v = self.F[:, 1:].T @ (self.goal_rotation()[:, 1:] @ p) if readouts[name].frame == "goal" else np.asarray(p)
                dirs[name] = -v / (np.linalg.norm(v) + 1e-12)
                info[f"readout_{name}_cos"] = float(dirs[name] @ d_true)
            use = {"search_dir": "truth", "search_dir_fk": "fk"}.get(mode)
            d_pred = dirs.get(use)
            ang = lambda c, d: float(np.degrees(np.arccos(np.clip(c @ d / np.linalg.norm(c), -1, 1))))
            if d_pred is not None:
                order = sorted(range(len(cands)), key=lambda i: (round(ang(cands[i], d_pred) / 31.0), np.linalg.norm(cands[i])))
            elif mode == "search_oracle":
                order = sorted(range(len(cands)), key=lambda i: np.linalg.norm(cands[i] - want))
            else:
                order = list(range(len(cands)))
            best = int(np.argmin([np.linalg.norm(c - want) for c in cands]))
            info["rank_of_nearest_candidate"] = order.index(best)
            info["nearest_candidate_miss_mm"] = float(np.linalg.norm(cands[best] - want) * 1000)
            if mode in CEILING_MODES:
                if mode == "ceil_lat":
                    ok, used = self.go_to(contact_T, want)
                else:
                    ok, used = self.go_to(self.H_star, np.zeros(2), GAP_M if mode == "ceil_pose" else 0.005)
                dt, dr = B.pose_distance(self.robot.hand_pose(), self.H_star)
                res = self.attempt_plain(proxy, scorer, None, seed + 1, axes_seed, -1e9, CEILING_DECISIONS)
                for key in ("contact_T", "contact_err", "preds"):
                    res.pop(key)
                res.update(placed=ok, lateral_before_mm=float(np.linalg.norm(self.F[:, 1:].T @ (self.goal_rotation()[:, 1:] @ np.zeros(2))) * 1000),
                           angle_to_H_star_deg_before=float(np.degrees(dr)))
                out.append(res)
                steps += used + res["steps"]
                order = []
            for n, i in enumerate(order[:MAX_PROBES]):
                if steps * STEP_S >= TIME_BUDGET_S:
                    break
                ok, used = self.go_to(contact_T, cands[i])
                steps += used
                if not ok:
                    continue
                res = self.attempt_plain(proxy, scorer, None, seed + 1 + n, axes_seed, abort_below, PROBE_DECISIONS)
                for key in ("contact_T", "contact_err", "preds"):
                    res.pop(key)
                res["offset_mm"] = (cands[i] * 1000).round(3).tolist()
                res["offset_miss_mm"] = float(np.linalg.norm(cands[i] - want) * 1000)
                steps += res["steps"]
                out.append(res)
                if res["seated_fk"]:
                    break
        return {"attempts": out, "success": out[-1]["outcome"] == "inserted", "ever_inserted": any(a["outcome"] == "inserted" for a in out),
                "seated_fk": out[-1]["seated_fk"], "control_steps": steps, **info}


def test_grasp_search(robot, agent, scorer: Scorer, readouts: dict, g: dict, out_file, n_starts: int, seed: int, modes,
                      abort_below: float, max_decisions: int = B.MAX_DECISIONS) -> dict:
    """As closed_loop.test_grasp (same reference insertion, same starts from the same seed), with the search modes."""
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
        ex = SearchTester(robot, robot.hand_pose(), np.asarray(g["axis"]))
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
                res = ex.run_mode_search(mode, agent, scorer, readouts, int(seed + 1000 * i), int(seed + 1000 * i), abort_below, max_decisions)
                entry["modes"][mode] = {**res, "restore_dev_mm": dev[0], "seconds": round(time.time() - t1, 1)}
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-2000:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary))
    return summary


def summarise_search(files) -> dict:
    rows = []
    for f in files:
        s = json.loads(open(f).read())
        rows += [st["modes"] for st in s["starts"] if st.get("modes")]
    rep = {"starts": len(rows)}
    modes = [m for m in MODES if rows and all(m in r for r in rows)]
    ok = {m: np.array([r[m]["success"] for r in rows], float) for m in modes}
    for m in modes:
        x = ok[m]
        searched = [r[m] for r in rows if len(r[m]["attempts"]) > 1]
        e = {"success": float(x.mean()), "se": float(x.std() / len(x) ** 0.5),
             "first_attempt_success": float(np.mean([r[m]["attempts"][0]["outcome"] == "inserted" for r in rows])),
             "ever_inserted": float(np.mean([r[m]["ever_inserted"] for r in rows])),
             "seconds_of_robot_time_mean": float(np.mean([r[m]["control_steps"] for r in rows]) * STEP_S),
             "starts_with_a_search": len(searched),
             "search_success": float(np.mean([r["success"] for r in searched])) if searched else None,
             "probes_mean": float(np.mean([len(r["attempts"]) - 1 for r in searched])) if searched else None,
             "probes_to_success_mean": float(np.mean([len(r["attempts"]) - 1 for r in searched if r["success"]])) if any(r["success"] for r in searched) else None,
             "success_within_probes": {str(n): float(np.mean([r["success"] and len(r["attempts"]) - 1 <= n for r in searched])) for n in (1, 2, 3, 6, 12, 18)} if searched else None,
             "rank_of_nearest_candidate_median": float(np.median([r["rank_of_nearest_candidate"] for r in searched])) if searched else None,
             "contact_lateral_mm_pct_25_50_75": np.percentile([r["contact_lateral_mm"] for r in searched], [25, 50, 75]).tolist() if searched else None,
             "stopped": {}}
        for name in ("truth", "fk"):
            cos = np.array([r[f"readout_{name}_cos"] for r in searched if f"readout_{name}_cos" in r])
            if len(cos):
                e[f"readout_{name}_direction"] = {"n": len(cos), "within_90deg": float(np.mean(cos > 0)), "within_45deg": float(np.mean(cos > 0.707)),
                                                  "within_30deg": float(np.mean(cos > 0.866))}
        for r in rows:
            for a in r[m]["attempts"]:
                key = f"{a['stopped']}/{a['outcome']}"
                e["stopped"][key] = e["stopped"].get(key, 0) + 1
        for ref in modes:
            if ref != m:
                d = x - ok[ref]
                e[f"minus_{ref}"] = {"diff": float(d.mean()), "se": float(d.std() / len(d) ** 0.5), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
        rep[m] = e
    return rep
