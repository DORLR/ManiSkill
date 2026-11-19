"""One attempt of the world model's agent (agent320.WMAgent) from a placed start, recorded in the fast copy's format.

Modes
  act      the built-in ACT head alone (no correction, no stop, no retry; the "free" stops of the policies'
           closed-loop test): the "policy alone" column
  wm       the run-time loop: candidates, correction when the model is sure, stop when seated, back off when blocked
  collect  "wm" plus exploration (online rounds, training grasps only)
  servo    ablation (2026-11-19, the user's request): the loop of "wm" (same done stop, same in-hole rule, same
           stall end), but the correction comes from the offset read-out instead of imagination:
              a_t = a_0 + J^+( -clip_{1.5 mm}(o_x e_1 + o_y e_2) )   if not inside and |(o_x, o_y)| >= 0.5 mm,  else a_0
           o = agent.offset_mm(z) (lateral x, y [mm] in the label frame of wmdata320._basis(axis)), e_1, e_2 = rows 1, 2
           of _basis(axis). servo_axis "true": the grasp's true insertion axis (privileged: more than the "wm" agent
           knows); "motion": the axis estimated as the direction of the hand's motion over the last SERVO_RECENT
           decisions minus the servo's own moves in them (over the whole try while it has fewer decisions); when that
           is under 1 mm (blocked) the last estimate is kept; no move before a first estimate. The angle of the
           estimate to the true axis is logged per servo decision (`axis_err_deg`, reporting only).
           No imagination is used. Ablation modes run without back-off (max_retries 0): the back-off's re-position
           is chosen by imagination, so it is refused for them.
  random   ablation: the loop of "wm", but with probability p_b (random_p[b]) a candidate drawn uniformly from the
           corrected copies is executed instead of the proposal; b = the bin of the estimated pin-tip gap g_t
           (edges RANDOM_EDGES_MM: g < 0, 0-2, 2-6, >= 6 mm); p_b is meant to be the "wm" agent's own correction rate
           in that bin. No imagination is used.
Counterfactual probe (B2, 2026-11-19; mode "wm" only, probe_max > 0): at up to probe_max decisions per bin of g_t
(gap_bin) where candidates are offered and g_t < probe_gap_mm, the simulator state is saved and every candidate (the
proposal, the n_cand - 1 corrected copies, then the servo's move with the true axis when |(o_x, o_y)| >= 0.5 mm, then
the proposal once more as a noise check: `R0_repeat`) is really executed for one decision,
then the ACT head for horizon - 1 decisions (the look-ahead the model imagines), and the state is restored after each
(recorder, picture cache and decision kinds are cut back too). True return of a branch, from FK and H* labels as
wmdata320.trajectory's reward (the first seated decision is terminal):
    R = sum_t gamma^t c_t r_t,  r_t = 1000 * depth gain + shaping * (gamma * phi_{t+1} - phi_t) [+ 10 - shaping * gamma * phi_{t+1} if seated]
The summary's `probes` keeps per probe n_cand, the model's mean score S and imagined reward sum G of every candidate,
the true R (n_cand values, then the servo's when `servo`), the per-decision true rewards and seated flags of every
branch, seated, depth gain and lateral offset after the branch, the model's pick, and checks: the hand's deviation
after each restore, the repeated proposal's R and first-decision hand deviation from the first, and after the real
decision the deviation of the hand from the branch of the executed candidate.
The attempt itself goes on exactly as mode "wm".
Per agent decision of every mode but "act" the summary keeps g_t, the read-out's lateral |(o_x, o_y)|, whether
candidates were offered, and the pick (0 proposal, i a corrected copy, -1 the servo's move) in `per_decision`
(back-off and re-position decisions are not agent decisions and are not in it); for reporting only, also the FK
lateral offset from H* (`fk_lat` [mm], label frame) and the executed move of a non-proposal pick (`move_mm`: along,
x, y in the label frame of the true axis).

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

import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..collect import branch as B
from ..collect.branch import Brancher, FullRecorder
from ..collect.explore import truth_outcome
from ..tdmpc import demo320 as D
from ..tdmpc import wmdata320 as WD
from ..tdmpc.demo320 import is_success
from .common320 import RANDOM_EDGES_MM, gap_bin, jpeg, offset, servo_move, unjpeg

EXECUTE = WD.EXECUTE
LEFT_M = 0.06                                  # as the policies' closed-loop test ("free" stops)
STEPS_MM = (0.5, 1.5)
BACKOFF_MM, BACKOFF_DECISIONS = 5.0, 2
DONE_P, DONE_TIMES, NOT_DONE_P = 0.5, 2, 0.5
MIN_TRY_MM, MIN_CORR_MM = 1.0, 0.5
KINDS = {"proposal": 0, "correction": 1, "random": 2, "shift": 3, "backoff": 4, "reposition": 5}
MODES = ("act", "wm", "collect", "servo", "random")
SERVO_MIN_MM, SERVO_MAX_MM = 0.5, 1.5           # ablation "servo": dead band and largest move per decision
SERVO_RECENT = 4                                # ablation "servo", axis "motion": decisions of recent motion


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
                random_reposition: float = 0.5, inside_mm: float | None = None, servo_axis: str = "true",
                random_p: tuple | list | None = None, probe_max: int = 0, probe_gap_mm: float = 6.0) -> tuple[dict, dict]:
        """-> (summary, recording). `inside_mm` (2026-11-18, the user's decision; None = off): once the estimated pin-tip
        depth g_t = agent.gap_mm(z_t, L) < -inside_mm, the try is in the hole and only the proposal is executed
        (no correction, no random candidate, no exploration push) until a back-off starts a new try. The decision and g_t at entry are logged (`inside_at`, `inside_gap_mm`). The recording is FullRecorder.end() plus `rgb_jpeg_list` (the stored JPEG bytes)
        and `dec_kind` (one KINDS value per executed decision)."""
        import torch
        assert mode in MODES
        if mode in ("servo", "random") and inside_mm is None:
            raise ValueError(f"mode {mode} is an ablation of the in-hole agent: it needs inside_mm")
        if mode in ("servo", "random") and max_retries:
            raise ValueError(f"mode {mode} runs without back-off (its re-position would use imagination): max_retries must be 0")
        if mode == "servo" and servo_axis not in ("true", "motion"):
            raise ValueError(f"servo_axis must be 'true' or 'motion', not {servo_axis!r}")
        if mode == "random" and (random_p is None or len(random_p) != len(RANDOM_EDGES_MM) + 1
                                 or not all(0.0 <= float(x) <= 1.0 for x in random_p)):
            raise ValueError(f"mode random needs random_p: {len(RANDOM_EDGES_MM) + 1} probabilities in [0, 1], got {random_p!r}")
        if probe_max and mode != "wm":
            raise ValueError("the counterfactual probe runs with mode wm only")
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

        meta_l = {"H_star": self.H_star, "axis": self.F[:, 0], "pin_length_m": self.L}

        def true_return(hb: list) -> tuple[float, dict]:
            """FK/H* labels of a branch's hand poses at its decision boundaries -> (R, labels), as wmdata320.trajectory."""
            lab = WD.decision_labels(meta_l, np.stack(hb).astype(np.float64), np.arange(len(hb)))
            R, gam, sh, steps = 0.0, agent.gamma, agent.shaping, []
            for t in range(len(hb) - 1):
                rw = (lab["depth"][t + 1] - lab["depth"][t]) * 1000.0 + sh * (gam * lab["phi"][t + 1] - lab["phi"][t])
                steps.append(round(float(rw), 4))                 # plain step reward (no bonus): either convention offline
                if lab["seated"][t + 1]:
                    R += gam ** t * (rw + WD.BONUS - sh * gam * lab["phi"][t + 1])
                    break
                R += gam ** t * rw
            lab["r_steps"] = steps
            return float(R), lab

        def branch_all(q: np.ndarray, chunk: np.ndarray, targets: list) -> tuple[list, float]:
            """Every target (8, 7; relative to q) for one decision, then the ACT head for horizon - 1 decisions, each from
            the current simulator state, restored after each branch. -> (per target results, largest restore deviation mm)."""
            snap, hand0 = r.base.get_state_dict(), r.hand_pose().copy()
            lens = {key: len(v) for key, v in rec.buf.items()}
            t0, frame0, nk = rec.t, getattr(rec, "frame", None), len(kinds)
            res, dev = [], 0.0
            for tgt in targets:
                hb = [hand0]
                execute(q[None] + tgt, q[None] + chunk, "proposal")
                hb.append(r.hand_pose().copy())
                for _ in range(agent.horizon - 1):
                    jb = len(rec.buf["rgb"]) - 1
                    ib = D.context_indices(jb)
                    qb = r.qpos().astype(np.float64)[:7].copy()
                    zb = agent.encode(np.stack([picture(i) for i in ib]), np.stack([rec.buf["depth"][i] for i in ib]), qb)
                    cb = agent.propose(zb)[:EXECUTE]
                    execute(qb[None] + cb, qb[None] + cb, "proposal")
                    hb.append(r.hand_pose().copy())
                R, lab = true_return(hb)
                res.append({"R": R, "r_steps": lab["r_steps"], "seated_steps": [bool(x) for x in lab["seated"][1:]],
                            "seated": bool(lab["seated"][1:].any()), "depth_gain_mm": float((lab["depth"][-1] - lab["depth"][0]) * 1000.0),
                            "lat_mm": float(np.hypot(lab["probe"][-1, 1], lab["probe"][-1, 2])), "first_hand": hb[1][:3, 3].copy()})
                r.base.set_state_dict(snap)                    # back: simulator, recorder, picture cache, decision kinds
                for key, v in rec.buf.items():
                    del v[lens[key]:]
                rec.t, rec.frame = t0, frame0
                del jp[lens["rgb"]:]
                for i in [i for i in dec if i >= lens["rgb"]]:
                    del dec[i]
                del kinds[nk:]
                dev = max(dev, float(np.linalg.norm(r.hand_pose()[:3, 3] - hand0[:3, 3]) * 1000.0))
            return res, dev

        hands, start_hand = [r.hand_pose()[:3, 3].copy()], r.hand_pose()[:3, 3].copy()
        try_start, try_corr = start_hand.copy(), np.zeros(3)
        excluded: list[np.ndarray] = []
        log = {"interventions": 0, "random_candidates": 0, "retries": 0, "shift": None, "done_p": [], "v_mean": [], "stall_states": [],
               "repositions": [], "inside_at": None, "inside_gap_mm": None, "servo_moves": 0, "per_decision": [], "probes": []}
        shifted, done_run, why, q_last, k = mode != "collect" or rng.random() >= p_shift, 0, "time", r.qpos()[:7], 0
        Bt = WD._basis(np.asarray(self.F[:, 0], np.float64))   # label frame of the true axis: reporting only
        servo_hist, ax_last = [], None                         # servo moves per agent decision of this try; last axis
        probes_in_bin = [0] * (len(RANDOM_EDGES_MM) + 1)
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
                o = agent.offset_mm(z)                          # read-out (None without the head); g_t as agent.gap_mm
                g = None if o is None else float(-o[0] - self.L * 1000.0)
                if o is None and mode in ("servo", "random"):
                    raise ValueError(f"mode {mode} needs the offset read-out head")
                fk = WD.decision_labels(meta_l, r.hand_pose()[None].astype(np.float64), np.array([0]))["probe"][0]
                rec_d = {"k": k, "g": None if g is None else round(g, 3),
                         "lat": None if o is None else round(float(np.hypot(o[1], o[2])), 3), "offered": False, "pick": 0,
                         "fk_lat": [round(float(fk[1]), 3), round(float(fk[2]), 3)]}
                log["per_decision"].append(rec_d)
                if inside_mm is not None and not inside:
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
                cands = np.stack([chunk] + [chunk + of[None] for of in offs])
                rec_d["offered"] = bool(len(offs))
                pick = 0
                probing = bool(probe_max and len(offs) and g is not None and g < probe_gap_mm and probes_in_bin[gap_bin(g)] < probe_max)
                if len(offs) and mode in ("wm", "collect"):      # imagination: the agent's own correction rule
                    if probing:
                        sc, g_img = agent.score(z, cands, parts=True)
                    else:
                        sc = agent.score(z, cands)             # (1 + n, num_q)
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
                if mode == "random" and len(offs) and rng.random() < float(random_p[gap_bin(g)]):
                    pick = int(rng.integers(1, len(cands)))    # ablation: a corrected copy at random, no imagination
                    log["random_candidates"] += 1
                if pick and pick == model_pick:
                    log["interventions"] += 1
                target, kind = cands[pick], ("proposal" if pick == 0 else "correction" if pick == model_pick else "random")
                rec_d["pick"] = pick
                if pick:
                    rec_d["move_mm"] = [round(float(x), 3) for x in Bt @ moves[pick - 1] * 1000.0]
                if pick and pick == model_pick:                # the model's own corrections only
                    try_corr = try_corr + moves[pick - 1]
                if mode == "servo":
                    mv = np.zeros(3)
                    ax = None
                    if servo_axis == "true":
                        ax = np.asarray(self.F[:, 0], np.float64)
                    else:                                      # the way in, from the hand's own motion (no H*, no axis)
                        n_b = min(SERVO_RECENT, len(hands) - 1, len(servo_hist))
                        dm = (hands[-1] - hands[-1 - n_b] - np.sum(servo_hist[len(servo_hist) - n_b:], axis=0)) if n_b > 0 else np.zeros(3)
                        if np.linalg.norm(dm) >= MIN_TRY_MM / 1000.0:
                            ax_last = dm / np.linalg.norm(dm)
                        ax = ax_last                           # blocked: the last estimate
                    if not inside and float(np.hypot(o[1], o[2])) >= SERVO_MIN_MM and ax is not None:
                        # ablation: move by the read-out, no imagination (in the read-out's label frame _basis(ax))
                        rec_d["axis_err_deg"] = round(float(np.degrees(np.arccos(np.clip(ax @ self.F[:, 0], -1.0, 1.0)))), 2)
                        mv = servo_move(float(o[1]), float(o[2]), ax, SERVO_MAX_MM)     # [m]
                        target, kind = chunk + offset(J, mv)[None], "correction"
                        rec_d["pick"] = -1
                        rec_d["move_mm"] = [round(float(x), 3) for x in Bt @ mv * 1000.0]
                        log["servo_moves"] += 1
                    servo_hist.append(mv)
                probe_rec = None
                if probing:                                    # B2: every candidate really executed, then restored
                    t_p = time.time()
                    probes_in_bin[gap_bin(g)] += 1
                    targets = list(cands)
                    servo = float(np.hypot(o[1], o[2])) >= SERVO_MIN_MM
                    if servo:
                        targets.append(chunk + offset(J, servo_move(float(o[1]), float(o[2]), self.F[:, 0], SERVO_MAX_MM))[None])
                    targets.append(cands[0])                   # the proposal again: the simulator's own noise
                    br, dev = branch_all(q, chunk, targets)
                    dup = br.pop()
                    probe_rec = {"k": k, "g": round(g, 3), "lat": round(float(np.hypot(o[1], o[2])), 3), "pick": pick, "n_cand": len(cands),
                                 "S": [round(float(x), 4) for x in sc.mean(1)], "S_min_adv": [round(float(x), 4) for x in adv.min(1)],
                                 "G_img": [round(float(x), 4) for x in g_img], "R": [round(b["R"], 4) for b in br],
                                 "seated": [b["seated"] for b in br], "depth_gain_mm": [round(b["depth_gain_mm"], 3) for b in br],
                                 "lat_mm": [round(b["lat_mm"], 3) for b in br], "servo": servo, "restore_dev_mm": round(dev, 4),
                                 "r_steps": [b["r_steps"] for b in br], "seated_steps": [b["seated_steps"] for b in br],
                                 "R0_repeat": round(dup["R"], 4),
                                 "repeat_first_dev_mm": round(float(np.linalg.norm(dup["first_hand"] - br[0]["first_hand"]) * 1000.0), 4),
                                 "first_hand": br[pick]["first_hand"], "seconds": round(time.time() - t_p, 2)}
                    log["probes"].append(probe_rec)
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
            if mode != "act" and probe_rec is not None:       # the executed branch against the real decision
                probe_rec["first_step_dev_mm"] = round(float(np.linalg.norm(hand - probe_rec.pop("first_hand")) * 1000.0), 4)
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
                servo_hist = []
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
                                                "inside_at", "inside_gap_mm", "servo_moves", "per_decision", "probes")},
                   "servo_axis": servo_axis if mode == "servo" else None,
                   "random_p": [float(x) for x in random_p] if mode == "random" else None,
                   "decision_kinds": {name: int((out["dec_kind"] == v).sum()) for name, v in KINDS.items()},
                   "done_p": log["done_p"], "v_mean": log["v_mean"], "model_step": agent.step}
        return summary, out


__all__ = ["OnlineTester", "KINDS", "MODES", "RANDOM_EDGES_MM", "gap_bin"]
