"""One attempt of the world model's agent (agent320.WMAgent) from a placed start, recorded in the fast copy's format.

Modes
  act      the built-in ACT head alone (no correction, no stop, no retry; the "free" stops of the policies'
           closed-loop test): the "policy alone" column
  wm       the run-time loop: candidates, correction when the model is sure, stop when seated, back off when blocked
  collect  "wm" plus exploration (online rounds, training grasps only)

Per decision of "wm"/"collect" (every 0.4 s):
  1. observation: the context pictures of the recorder (pictures 2d-6, 2d-4, 2d-2, 2d of the attempt, 0.4 s apart,
     the first repeated at the start, exactly the training windows' definition) through the store's JPEG codec,
     and the measured joints.
  2. seated: the done head says p > 0.5 at two decisions in a row; the second of them still executes the ACT
     proposal (a plug a little short of the seat is pushed once more), then the attempt ends.
  3. candidates: the ACT head's 8 executed targets, and 12 corrected copies: the hand moved by 0.5 or 1.5 mm along
     +-x, +-y, +-z of the HAND's own frame (FK of the measured joints; no H*, no axis), as a joint offset added to
     all 8 targets (damped least squares on the FK Jacobian, rotation held). Excluded directions (5) are left out.
  4. correction when the best corrected copy beats the proposal by more than `margin` on the mean of the value
     heads' scores AND on every head (agent320.score); otherwise the proposal as it is.
  5. blocked (agreed with the researcher 2026-11-17): the FK hand moved less than 0.2 mm over 8 decisions (as the
     policies' test) AND the done head says not seated (p < 0.5); and the remaining decisions leave room for the
     back-off, the re-position and a new try (2 + 1 + 8 + 1). Back-off: 5 mm in 2 decisions along the reverse of
     this try's motion (from where the try began; when that is under 1 mm, towards the attempt's free-space start;
     when that too is under 1 mm the attempt ends "stalled"). Re-position: `n_candidates` positions drawn
     uniformly in a disc of radius `radius_mm` in the plane across the way in (hand frame of the measured joints and
     the motion itself; no H*); each is imagined as one decision that moves the hand there, then the ACT head going
     on (agent320.score, `horizon` decisions); the best mean score is executed (collect: with probability
     `random_reposition` a random one instead, so the data holds re-positions of every kind and the model's choice
     can be compared with chance). Then the attempt goes on. The candidate direction
     most aligned with the model's own corrections of this try (in total; random candidates do not count) is
     excluded for the rest of the attempt (only when they add up to at least 0.5 mm); when nothing is excluded the
     next try would repeat the same approach (the agent is deterministic), so every back-off halves the correction
     margin of the attempt instead (the model corrects more readily). Otherwise, or after `max_retries` back-offs,
     the attempt ends "stalled".
  6. ends: seated (2), stalled (5), left (hand more than 6 cm from the start), 150 decisions in total (back-off
     and re-position decisions included).
Exploration in "collect" (labels may be used here: training grasps only, as the scripted sets did):
  - with probability `p_shift` per attempt, once, when the pin tips are first 0.5-4 mm in front of the board face
    (FK and H*), the hand is moved by 0.5-2 mm in a random direction across the insertion axis during that decision;
  - with probability `eps` per decision, while the pin tips are 0-6 mm in front of the face, a random candidate is
    executed instead of the chosen one.
Every executed decision (back-off ones included) is a decision of the stored trajectory, so the world model learns
from all of it; `dec_kind` records what each decision was (KINDS), so that imitation can keep the proposal and
correction decisions only. The last decision is followed by a 4-step hold (as every other set). Labels from FK and
H* only (wmdata320); `seated_fk` is taken at the last decision's end (step 8n, the instant of the trajectory's
terminal flag; the stored outcome), `seated_fk_after_hold` after the hold (as the policies' test reports it). The
simulator's truth goes to the attempt's summary, for reporting, never into the trajectory.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..collect import branch as B
from ..collect.branch import Brancher, FullRecorder
from ..collect.explore import truth_outcome
from ..tdmpc import demo320 as D
from ..tdmpc import wmdata320 as WD
from ..tdmpc.demo320 import is_success
from .common320 import jpeg, offset, unjpeg

EXECUTE = WD.EXECUTE
LEFT_M = 0.06                                  # as the policies' closed-loop test ("free" stops)
STEPS_MM = (0.5, 1.5)
BACKOFF_MM, BACKOFF_DECISIONS = 5.0, 2
DONE_P, DONE_TIMES, NOT_DONE_P = 0.5, 2, 0.5
MIN_TRY_MM, MIN_CORR_MM = 1.0, 0.5
KINDS = {"proposal": 0, "correction": 1, "random": 2, "shift": 3, "backoff": 4, "reposition": 5}


class OnlineTester(Brancher):
    # ------------------------------------------------------------ joint offsets that move the hand
    def jacobian(self, q9: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """(6, 7): hand position [m] and rotation vector [rad] (world frame) per joint [rad], by central differences of FK."""
        J = np.zeros((6, 7))
        for i in range(7):
            qp, qm = q9.copy(), q9.copy()
            qp[i] += eps
            qm[i] -= eps
            Tp, Tm = self.fk(np.stack([qp, qm]))
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            J[3:, i] = Rotation.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
        return J

    def face_gap_mm(self) -> float:
        """How far the pin tips are in front of the board face along the insertion axis [mm] (FK and H*; collect mode only)."""
        return -(self.depth(self.robot.hand_pose()) * 1000.0) - self.L * 1000.0

    # ------------------------------------------------------------ one attempt
    def attempt(self, agent, mode: str, seed: int, max_decisions: int = B.MAX_DECISIONS, margin: float = 0.5,
                max_retries: int = 3, p_shift: float = 0.5, eps: float = 0.1, n_candidates: int = 16, radius_mm: float = 3.0,
                random_reposition: float = 0.5, inside_mm: float | None = None) -> tuple[dict, dict]:
        """-> (summary, recording). `inside_mm` (2026-11-18, the user's decision; None = off): once the estimated pin-tip
        depth g_t = agent.gap_mm(z_t, L) < -inside_mm, the try is in the hole and only the proposal is executed
        (no correction, no random candidate, no exploration push) until a back-off starts a new try. The decision and g_t at entry are logged (`inside_at`, `inside_gap_mm`). The recording is FullRecorder.end() plus `rgb_jpeg_list` (the stored JPEG bytes)
        and `dec_kind` (one KINDS value per executed decision)."""
        import torch
        assert mode in ("act", "wm", "collect")
        r, rng = self.robot, np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rec = FullRecorder(r)
        rec.begin()
        jp: list[bytes] = []                                   # JPEG bytes of every recorded picture, in order
        dec: dict[int, np.ndarray] = {}                        # decoded pictures still needed by a context

        def picture(j: int) -> np.ndarray:
            while len(jp) <= j:
                jp.append(jpeg(rec.buf["rgb"][len(jp)]))
                if len(jp) - 1 >= j - 2 * (D.CONTEXT - 1):
                    dec[len(jp) - 1] = unjpeg(jp[-1])
            for i in [i for i in dec if i < j - 2 * (D.CONTEXT - 1)]:
                del dec[i]
            return dec[j] if j in dec else unjpeg(jp[j])

        kinds: list[int] = []

        def execute(targets: np.ndarray, policy: np.ndarray, kind: str) -> np.ndarray:
            q = targets[0]
            for j in range(EXECUTE):
                q = targets[j].astype(np.float32).astype(float)
                r._command(q, 1)
                rec.step(q, policy[j])
            kinds.append(KINDS[kind])
            return q

        hands, start_hand = [r.hand_pose()[:3, 3].copy()], r.hand_pose()[:3, 3].copy()
        try_start, try_corr = start_hand.copy(), np.zeros(3)
        excluded: list[np.ndarray] = []
        log = {"interventions": 0, "random_candidates": 0, "retries": 0, "shift": None, "done_p": [], "v_mean": [], "stall_states": [],
               "repositions": [], "inside_at": None, "inside_gap_mm": None}
        shifted, done_run, why, q_last, k = mode != "collect" or rng.random() >= p_shift, 0, "time", r.qpos()[:7], 0
        p, vm, m_now, inside = None, None, margin, False
        while k < max_decisions:
            j = len(rec.buf["rgb"]) - 1                         # the picture of this decision's start
            idx = D.context_indices(j)
            q9 = r.qpos().astype(np.float64)
            q = q9[:7].copy()
            z = agent.encode(np.stack([picture(i) for i in idx]), np.stack([rec.buf["depth"][i] for i in idx]), q)
            chunk = agent.propose(z)[:EXECUTE]                  # (8, 7) relative to q
            target, kind = chunk, "proposal"
            if mode != "act":
                p = agent.done(z)
                vm = float(agent.values(z).mean())
                log["done_p"].append(round(p, 3))
                log["v_mean"].append(round(vm, 3))
                done_run = done_run + 1 if p > DONE_P else 0
                if inside_mm is not None and not inside:
                    g = agent.gap_mm(z, self.L * 1000.0)
                    if g is not None and g < -inside_mm:
                        inside = True                           # for the rest of this try (a back-off starts a new one)
                        if log["inside_at"] is None:
                            log["inside_at"], log["inside_gap_mm"] = k, round(g, 3)
                if done_run >= DONE_TIMES:                      # one more push with the proposal, then the end
                    q_last = execute(q[None] + chunk, q[None] + chunk, "proposal")
                    k += 1
                    why = "seated"
                    break
                J = self.jacobian(q9)
                Rh = r.hand_pose()[:3, :3]
                dirs = [s * Rh[:, i] for i in range(3) for s in (1.0, -1.0)]
                dirs = [d for d in dirs if not any(d @ e > np.cos(np.radians(45.0)) for e in excluded)]
                moves = [d * mm / 1000.0 for d in dirs for mm in STEPS_MM] if not inside else []   # in the hole: none
                offs = [offset(J, mv) for mv in moves]
                cands = np.stack([chunk] + [chunk + o[None] for o in offs])
                pick = 0
                if len(offs):
                    sc = agent.score(z, cands)                 # (1 + n, num_q)
                    adv = sc[1:] - sc[0]
                    ok = (adv.mean(1) > m_now) & (adv.min(1) > 0)
                    if ok.any():
                        pick = 1 + int(np.argmax(np.where(ok, adv.mean(1), -np.inf)))
                model_pick = pick
                if mode == "collect":
                    gap = self.face_gap_mm()
                    if len(offs) and 0.0 <= gap < 6.0 and rng.random() < eps:
                        pick = int(rng.integers(1, len(cands)))
                        log["random_candidates"] += 1
                if pick and pick == model_pick:
                    log["interventions"] += 1
                target, kind = cands[pick], ("proposal" if pick == 0 else "correction" if pick == model_pick else "random")
                if pick and pick == model_pick:                # the model's own corrections only
                    try_corr = try_corr + moves[pick - 1]
                if mode == "collect" and not shifted and 0.5 <= gap <= 4.0 and not inside:   # no push in the hole
                    shifted = True
                    a = self.F[:, 0]
                    u = rng.normal(size=3)
                    u -= (u @ a) * a
                    u /= np.linalg.norm(u)
                    mm = float(rng.uniform(0.5, 2.0))
                    target, kind = target + offset(J, u * mm / 1000.0)[None], "shift"
                    log["shift"] = {"decision": k, "mm": round(mm, 3), "gap_mm": round(gap, 3)}
            q_last = execute(q[None] + target, q[None] + chunk, kind)
            k += 1
            hand = r.hand_pose()[:3, 3].copy()
            if np.linalg.norm(hand - start_hand) > LEFT_M:
                why = "left"
                break
            hands.append(hand)
            if len(hands) > B.STALL_DECISIONS and np.linalg.norm(hands[-1] - hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
                log["stall_states"].append({"decision": k, "done_p": None if p is None else round(p, 3), "v_mean": None if vm is None else round(vm, 3)})
                room = max_decisions - k >= BACKOFF_DECISIONS + 1 + B.STALL_DECISIONS + 1
                bad = mode != "act" and p < NOT_DONE_P
                d = hand - try_start
                if np.linalg.norm(d) < MIN_TRY_MM / 1000.0:
                    d = hand - start_hand
                if not (bad and room and log["retries"] < max_retries and np.linalg.norm(d) >= MIN_TRY_MM / 1000.0):
                    why = "stalled"
                    break
                log["retries"] += 1                            # back off along the reverse of the way in
                d = -d / np.linalg.norm(d)
                if np.linalg.norm(try_corr) >= MIN_CORR_MM / 1000.0 and dirs:
                    c = try_corr / np.linalg.norm(try_corr)
                    excluded.append(max(dirs, key=lambda x: float(x @ c)))
                else:
                    m_now = m_now / 2.0
                for _ in range(BACKOFF_DECISIONS):
                    q9 = r.qpos().astype(np.float64)
                    o = offset(self.jacobian(q9), d * BACKOFF_MM / BACKOFF_DECISIONS / 1000.0)
                    ramp = q9[None, :7] + o[None] * (np.arange(1, EXECUTE + 1)[:, None] / EXECUTE)
                    q_last = execute(ramp, ramp, "backoff")
                    k += 1
                if np.linalg.norm(r.hand_pose()[:3, 3] - start_hand) > LEFT_M:
                    why = "left"
                    break
                # re-position: candidate positions across the way in, each imagined, the best (or a random one) taken
                j = len(rec.buf["rgb"]) - 1
                idx = D.context_indices(j)
                q9 = r.qpos().astype(np.float64)
                q = q9[:7].copy()
                z = agent.encode(np.stack([picture(i) for i in idx]), np.stack([rec.buf["depth"][i] for i in idx]), q)
                ax = -d                                        # the way in
                e1 = np.cross(ax, [0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.cross(ax, [1.0, 0.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(ax, e1)
                rad = radius_mm / 1000.0 * np.sqrt(rng.uniform(0.0, 1.0, n_candidates))
                ang = rng.uniform(0.0, 2 * np.pi, n_candidates)
                pos = rad[:, None] * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
                Jr = self.jacobian(q9)
                ramp = np.arange(1, EXECUTE + 1)[:, None] / EXECUTE
                rp = np.stack([offset(Jr, mv)[None] * ramp for mv in pos])     # (N, 8, 7) relative to q
                sc = agent.score(z, rp).mean(1)
                best = int(np.argmax(sc))
                pick, how = best, "model"
                if mode == "collect" and rng.random() < random_reposition:
                    pick, how = int(rng.integers(n_candidates)), "random"
                q_last = execute(q[None] + rp[pick], q[None] + rp[pick], "reposition")
                k += 1
                log["repositions"].append({"decision": k, "how": how, "mm": [round(float(x) * 1000, 3) for x in pos[pick]],
                                           "score_pick": round(float(sc[pick]), 3), "score_best": round(float(sc[best]), 3)})
                hands = [r.hand_pose()[:3, 3].copy()]
                try_start, try_corr, done_run, inside = hands[0].copy(), np.zeros(3), 0, False
                if np.linalg.norm(hands[0] - start_hand) > LEFT_M:
                    why = "left"
                    break
        for _ in range(4):
            r._command(q_last, 1)
            rec.step(q_last, q_last)
        out = rec.end()
        picture(len(out["rgb"]) - 1)
        out["rgb_jpeg_list"], out["dec_kind"] = jp, np.asarray(kinds, np.int8)
        lab = WD.decision_labels({"H_star": self.H_star, "axis": self.F[:, 0], "pin_length_m": self.L}, out["hand"].astype(np.float64),
                                 np.array([EXECUTE * k, len(out["hand"]) - 1]))
        truth = truth_outcome(out["eval_rel"][-1])
        summary = {"mode": mode, "stopped": why, "decisions": k, "seated_fk": bool(lab["seated"][0]), "seated_fk_after_hold": bool(lab["seated"][1]),
                   "success": is_success(truth), **{key: float(truth[key]) for key in D.SUCCESS},
                   **{key: log[key] for key in ("interventions", "random_candidates", "retries", "shift", "stall_states", "repositions",
                                                "inside_at", "inside_gap_mm")},
                   "decision_kinds": {name: int((out["dec_kind"] == v).sum()) for name, v in KINDS.items()},
                   "done_p": log["done_p"], "v_mean": log["v_mean"], "model_step": agent.step}
        return summary, out


__all__ = ["OnlineTester", "KINDS"]
