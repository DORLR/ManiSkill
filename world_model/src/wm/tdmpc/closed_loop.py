"""Closed-loop test of a branch model in the simulator: the policy inserts, the model may choose between the policy's
candidates and may call an attempt off; a blocked attempt is taken back to the start and tried again.

Everything the loop uses exists on the real robot: the wrist pictures, the joints, FK, the reference pose H* of the
grasp. The simulator's truth is read only to count the result. All modes of a start begin from the same saved
simulator state (so they are compared on equal terms); INSIDE a mode the robot goes back to the start the way the real
one would. Pins in the holes (FK depth): first straight out along the port axis to the standoff, as explore.place does
-- replaying the targets backwards cannot pull out a plug that sits in the hole crooked (first run, 2026-11-16: 27 of
67 such attempts stayed in the hole, 27-30 mm from the start). Pins not in the holes: the executed joint targets
replayed backwards. Then the start's joint command.

Modes (attempt = one run of the policy from the start until seated / stalled / left / time, as collect/branch.py):
  policy       one attempt, the policy as it is                                  (data: first branch of a start)
  retry        up to 4 attempts; attempts 3 and 4 with the small displacement of the branch set (data: any of 4)
  abort        retry + the model calls an attempt off early (v0 under a threshold twice in a row, from 3 decisions
               after the pins reach the board face on)
  select_q0    one attempt; at every decision the policy gives N_POLICY samples, N_SHIFT of them also shifted by a
  select_im2   displacement of the size the branch set contains; the model scores all, the best is executed.
               q0: Q(latent, action). im2: imagined reward of the two decisions the 16 targets cover + value of the
               imagined latent (the second action is taken relative to the last target of the first: an approximation)
  full         up to 4 attempts with select_im2 and the early stop
Scores are the mean over the value heads minus K_SD times their spread.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation  # noqa: F401  (kept for parity with collect/branch.py)

from ..collect import branch as B
from ..collect.branch import Brancher, FullRecorder, sample_noise
from ..collect.explore import BACK_GAP_M, truth_outcome
from .data import ACTION_DIM, FACE_MARGIN_M, scale_action
from .model import LatentWM, make_cfg

MODES = ("policy", "select_q0", "select_im2", "retry", "abort", "full")
MAX_ATTEMPTS = 4
N_POLICY, N_SHIFT = 4, 4
SHIFT_SIGMA_M = np.array([0.00003, 0.00005, 0.00005])     # along / across the axis; the branch set: bias 0.02-0.04 mm + jitter 0.03 mm
K_SD = 1.0
ABORT_FROM, ABORT_COUNT = 3, 2


class Scorer:
    def __init__(self, ckpt: str, device: str = "cuda"):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.stats, self.device = ck["stats"], device
        self.model = LatentWM(make_cfg(**ck["cfg"]), ck["stats"]).to(device).eval()
        self.model.load_state_dict(ck["model"])

    @torch.no_grad()
    def latent(self, rec: FullRecorder) -> torch.Tensor:
        """From the recorder's three latest pictures (0.2 s apart) and the joints at those instants."""
        n = len(rec.buf["rgb"])
        ids = [max(n - 3, 0), max(n - 2, 0), n - 1]
        rgb = torch.from_numpy(np.stack([rec.buf["rgb"][i] for i in ids]))[None].to(self.device).permute(0, 1, 4, 2, 3)
        depth = torch.from_numpy(np.stack([np.minimum(rec.buf["depth"][i], 32767).astype(np.int16) for i in ids]))[None].to(self.device)
        q = torch.from_numpy(np.stack([rec.buf["qpos"][rec.buf["img_step"][i]][:7] for i in ids]).astype(np.float32))[None].to(self.device)
        return self.model.encode(rgb, depth, q)

    @torch.no_grad()
    def score(self, z: torch.Tensor, deltas: np.ndarray) -> dict:
        """deltas (K, 2, 56): joint targets minus the joints they are relative to [rad]. -> v0 (2,) = mean and spread
        over the heads; q0, im2 (K, 2)."""
        a = torch.from_numpy(scale_action(deltas, self.stats)).to(self.device)
        s = self.model.scores(z.expand(len(a), -1), a, torch.full((len(a),), 2, device=self.device), hs=(1, 2))
        ms = lambda v: np.stack([v.mean(0).cpu().numpy(), v.std(0).cpu().numpy()], -1)
        return {"v0": ms(s["v0"])[0], "q0": ms(s["q0"]), "im2": ms(s["im2"])}


class Tester(Brancher):
    def snapshot(self):
        super().snapshot()
        self._snap_q = self.q0.copy()

    def go_back(self, executed: list) -> list[float]:
        """Back to the start (see the module docstring). Returns the hand's deviation from the start (mm, deg) and
        leaves the robot settled there."""
        r = self.robot
        H = r.hand_pose()
        if self.depth(H) > -(self.L - 0.0005):                    # pins at least 0.5 mm in the holes: straight out first
            back = -(self.L + BACK_GAP_M)
            out = H.copy()
            out[:3, 3] += self.F[:, 0] * (back - self.depth(H))
            r.move_to(out)
            if self.depth(r.hand_pose()) > -self.L:               # still in: the standoff's joint configuration pulls
                self._joint_move(self.q_back)
        else:
            for q in reversed(executed):
                r._command(q, 1)
        self._joint_move(self._snap_q)
        dt, dr = B.pose_distance(r.hand_pose(), self._snap_hand)
        return [float(dt * 1000), float(np.degrees(dr))]

    def _shifted(self, chunk8: np.ndarray, eps: np.ndarray) -> tuple[np.ndarray, int]:
        """The 8 targets with the commanded hand displaced by eps (frame of the axis); steps without an IK solution
        keep the policy's own target (as collect/branch.py)."""
        r = self.robot
        fingers = np.tile(r.qpos()[7:9], (len(chunk8), 1))
        T = self.fk(np.c_[chunk8, fingers])
        out, missed = chunk8.copy(), 0
        for j in range(len(chunk8)):
            Tj = T[j].copy()
            Tj[:3, 3] += self.F @ eps
            sol = r._ik(Tj, np.r_[chunk8[j], fingers[j]])
            if sol is None or np.abs(sol[:7] - chunk8[j]).max() > 0.03:
                missed += 1
            else:
                out[j] = sol[:7]
        return out, missed

    def attempt(self, rec: FullRecorder, agent, scorer: Scorer | None, seed: int, axes_seed: int, noise: dict,
                select: str | None, abort_below: float | None, max_decisions: int) -> dict:
        """One attempt; without `select` and `abort_below` it is collect/branch.py's rollout."""
        r, F, E = self.robot, self.F, B.EXECUTE
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rng = np.random.default_rng(seed)
        rec.begin()
        if not agent.reset(rec.frame, axes_seed):
            return {"stopped": "no_board_plane", "decisions": 0, "executed": []}
        perturbed = noise["sigma_m"] > 0
        bias, rot_bias = np.asarray(noise["bias_m"]), np.asarray(noise["rot_bias_deg"])
        why, q_last, executed, missed = "time", r.qpos()[:7], [], 0
        hands, at_face, low, picks, v_log = [r.hand_pose()[:3, 3]], None, 0, [], []
        k = 0
        for k in range(max_decisions):
            q_now = r.qpos()[:7].astype(np.float64)
            chunks = np.asarray(agent.act(rec.frame, n_samples=N_POLICY if select else 1), np.float64)     # (n, 16, 7)
            z = scorer.latent(rec) if (select or abort_below is not None) else None
            if select:
                cands = [c[:E] for c in chunks]
                for j in range(N_SHIFT):
                    sh, m = self._shifted(chunks[j % len(chunks)][:E], rng.normal(0, 1, 3) * SHIFT_SIGMA_M)
                    cands.append(sh)
                second = [chunks[j % len(chunks)][E:2 * E] - chunks[j % len(chunks)][E - 1] for j in range(len(cands))]
                deltas = np.stack([np.stack([(c - q_now).reshape(-1), s.reshape(-1)]) for c, s in zip(cands, second)]).astype(np.float32)
                sc = scorer.score(z, deltas)
                val = sc[select][:, 0] - K_SD * sc[select][:, 1]
                best = int(np.argmax(val))
                targets = cands[best]
                picks.append([best, float(val[best] - val[0]), float(val.std())])
                v0 = float(sc["v0"][0])
            else:
                targets = chunks[0][:E]
                if perturbed:
                    eps = bias + rng.normal(0, noise["jitter_m"], 3)
                    fingers = np.tile(r.qpos()[7:9], (E, 1))
                    T = self.fk(np.c_[targets, fingers])
                    R_eps = Rotation.from_rotvec(F @ np.radians(rot_bias)).as_matrix()
                    tg = targets.copy()
                    for j in range(E):
                        Tj = T[j].copy()
                        Tj[:3, 3] += F @ eps
                        Tj[:3, :3] = R_eps @ Tj[:3, :3]
                        sol = r._ik(Tj, np.r_[targets[j], fingers[j]])
                        if sol is None or np.abs(sol[:7] - targets[j]).max() > 0.03:
                            missed += 1
                        else:
                            tg[j] = sol[:7]
                    targets = tg
                v0 = None
                if z is not None:
                    d0 = np.stack([(targets - q_now).reshape(-1), np.zeros(ACTION_DIM)])[None].astype(np.float32)
                    v0 = float(scorer.score(z, d0)["v0"][0])
            v_log.append(v0)
            for j in range(E):
                q = targets[j].astype(np.float32).astype(float)
                r._command(q, 1)
                rec.step(q, chunks[0][j])
                executed.append(q)
                q_last = q
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
            if abort_below is not None and at_face is not None and k - at_face >= ABORT_FROM and v0 is not None:
                low = low + 1 if v0 < abort_below else 0
                if low >= ABORT_COUNT:
                    why = "aborted"
                    break
        for _ in range(4):
            r._command(q_last, 1)
            rec.step(q_last, q_last)
            executed.append(q_last)
        end = self.fk_state()
        arrays = rec.end()
        return {"stopped": why, "decisions": k + 1, "ik_missed_steps": missed, "at_face": at_face,
                "seated_fk": bool(end["dist_m"] < B.SEATED_M and end["angle_deg"] < B.SEATED_DEG),
                "picked_other": float(np.mean([p[0] != 0 for p in picks])) if picks else None,
                "score_gain_mean": float(np.mean([p[1] for p in picks])) if picks else None,
                "score_spread_mean": float(np.mean([p[2] for p in picks])) if picks else None,
                "v0_first_last": [v_log[0], v_log[-1]] if v_log and v_log[0] is not None else None,
                **truth_outcome(arrays["eval_rel"][-1]), "executed": executed}

    def run_mode(self, mode: str, agent, scorer: Scorer, seed: int, axes_seed: int, abort_below: float, max_decisions: int) -> dict:
        """From the saved start (restored by the caller). Ends when FK says seated or the attempts are used up."""
        select = {"select_q0": "q0", "select_im2": "im2", "full": "im2"}.get(mode)
        attempts = MAX_ATTEMPTS if mode in ("retry", "abort", "full") else 1
        thr = abort_below if mode in ("abort", "full") else None
        rng = np.random.default_rng(seed + 77)
        out, steps = [], 0
        for i in range(attempts):
            noise = sample_noise(rng, i if mode in ("retry", "abort") else 0)
            res = self.attempt(FullRecorder(self.robot), agent, scorer, seed + i, axes_seed, noise, select, thr, max_decisions)
            executed = res.pop("executed")
            steps += len(executed)
            res["sigma_m"] = noise["sigma_m"]
            out.append(res)
            if res["seated_fk"] or res["stopped"] == "no_board_plane" or i == attempts - 1:
                break
            res["back_dev_mm_deg"] = self.go_back(executed)
            steps += len(executed) + 30
        return {"attempts": out, "success": out[-1]["outcome"] == "inserted", "ever_inserted": any(a["outcome"] == "inserted" for a in out),
                "seated_fk": out[-1]["seated_fk"], "control_steps": steps}


def test_grasp(robot, agent, scorer: Scorer, g: dict, out_file, n_starts: int, seed: int, modes, abort_below: float,
               max_decisions: int = B.MAX_DECISIONS) -> dict:
    """g: a wm_v1 grasp.json. The reference insertion and the starts as collect/branch.py; every mode from every start."""
    import json
    import time
    import traceback
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
        ex = Tester(robot, robot.hand_pose(), np.asarray(g["axis"]))
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
                res = ex.run_mode(mode, agent, scorer, int(seed + 1000 * i), int(seed + 1000 * i), abort_below, max_decisions)
                entry["modes"][mode] = {**res, "restore_dev_mm": dev[0], "seconds": round(time.time() - t1, 1)}
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-2000:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(summary))
    return summary


def summarise(files) -> dict:
    """Per mode: success rate (truth at the end of the mode) with its standard error, and the paired difference to
    `policy` and to `retry` over the same starts."""
    import json
    rows = []
    for f in files:
        s = json.loads(open(f).read())
        rows += [st["modes"] for st in s["starts"] if st.get("modes")]
    rep = {"starts": len(rows)}
    modes = [m for m in MODES if rows and all(m in r for r in rows)]
    ok = {m: np.array([r[m]["success"] for r in rows], float) for m in modes}
    for m in modes:
        x = ok[m]
        e = {"success": float(x.mean()), "se": float(x.std() / max(len(x), 1) ** 0.5),
             "ever_inserted": float(np.mean([r[m]["ever_inserted"] for r in rows])),
             "fk_says_seated": float(np.mean([r[m]["seated_fk"] for r in rows])),
             "attempts_mean": float(np.mean([len(r[m]["attempts"]) for r in rows])),
             "seconds_of_robot_time_mean": float(np.mean([r[m]["control_steps"] for r in rows]) * 0.05),
             "stopped": {}}
        for r in rows:
            for a in r[m]["attempts"]:
                key = f"{a['stopped']}/{a['outcome']}"
                e["stopped"][key] = e["stopped"].get(key, 0) + 1
        back = [a["back_dev_mm_deg"][0] for r in rows for a in r[m]["attempts"] if "back_dev_mm_deg" in a]
        if back:
            e["back_dev_mm_pct_50_90_100"] = np.percentile(back, [50, 90, 100]).tolist()
        picked = [a["picked_other"] for r in rows for a in r[m]["attempts"] if a.get("picked_other") is not None]
        if picked:
            e["picked_other_than_first_sample"] = float(np.mean(picked))
        for ref in ("policy", "retry"):
            if ref in ok and ref != m:
                d = x - ok[ref]
                e[f"minus_{ref}"] = {"diff": float(d.mean()), "se": float(d.std() / len(d) ** 0.5),
                                     "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
        rep[m] = e
    return rep
