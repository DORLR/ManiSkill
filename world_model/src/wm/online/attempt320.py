"""One attempt of the world model's agent (agent320.WMAgent) from a placed start, recorded in the fast copy's format.

Modes
  act      the built-in ACT head alone (no correction, no stop, no retry; the "free" stops of the policies'
           closed-loop test): the "policy alone" column
  act_done identical to "act" (no WM candidate corrections, no inside_mm, no retry), EXCEPT it uses the done head
           (done(z) > 0.5 twice consecutively) to stop "seated", matching the stopping rule of "wm"
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
Insertion time (2026-11-19, reporting only; never an input of the agent): after every executed agent decision of every
mode (back-off and re-position decisions included) the simulator-truth success criterion of the adapter (truth_step:
ManiSkill demo320.is_success; AutoMate check_plug_inserted_in_socket, wm.sim_automate.checks) is checked; the summary
keeps the number of decisions after which it first held (`first_success_decision`, None if never),
per executed decision whether it held (`truth_ok`) and the true depth left and lateral offset (`truth_depth_mm`,
`truth_lateral_mm`), and per main-loop agent decision [decisions executed before it, wall-clock ms of the agent's
computation from the pictures to the chosen action, GPU synchronised; the counterfactual probe excluded] (`latency_ms`).
Per agent decision of every mode but "act" the summary keeps g_t, the read-out's lateral |(o_x, o_y)|, whether
candidates were offered, and the pick (0 proposal, i a corrected copy, -1 the servo's move) in `per_decision`
(back-off and re-position decisions are not agent decisions and are not in it); for reporting only, also the FK
lateral offset from H* (`fk_lat` [mm], label frame) and the executed move of a non-proposal pick (`move_mm`: along,
x, y in the label frame of the true axis).
Gap g_t (the in-hole rule, the probe's and the ablation "random"'s bins, the logs): from the offset read-out o_t by
_read_agent_offset_and_gap (agent320.WMAgent.gap_mm). offset_mode "gap" (wmdata320): g_t = o_t[0], with no pin length
L, H* or simulator truth (self.L is not read during the decision, _ForbidPinLength); offset_mode "along" (legacy):
g_t = -o_t[0] - L. AutoMate (the user's decision 6 of 2026-11-23): an attempt with a checkpoint of any offset_mode
but "gap" is refused (common320.check_offset_mode).

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
  6. ends: seated (2), stalled (5), left (hand more than left_m from the start: ManiSkill 6 cm, AutoMate 6 cm + L),
     max_decisions in total (back-off and re-position decisions included; ManiSkill 150, AutoMate robot.max_decisions).
Planner "mppi" (the user's decisions of 2026-11-26; agent320.MPPI): in "wm" and "collect" the fixed candidates and their
margin rule are replaced by TD-MPC2's MPPI over hand moves around the proposal (1.5 mm per unit); "wm" plans until the
in-hole rule fires (inside_mm required), "collect" plans at every decision; in "collect" TD-MPC2's exploration noise is
NOT added (the user's decision of 2026-11-26: with noise at every step nearly no online decision could be imitated):
every executed plan is the model's correction ("correction"), the imitation head copies it from seated attempts as it
copied the fixed planner's corrections; exploration is the sparse scheme below, unchanged (eps: one of the 12 fixed
moves; one shift push); interventions count executed plans (a plan with the shift push added still counts, as with
the fixed planner).
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
Simulators (the user's decisions 1, 3, 6 and 9 of 2026-11-25): this module and wm.online.shared320 import no simulator;
the adapters wm.online.maniskill320 (ManiSkill, behaviour unchanged) and wm.online.automate320 (AutoMate) give the
recorder's reads, the "left" distance, the labels' simulator key and the truth (OnlineTester docstring).
"""

from __future__ import annotations

import time

import numpy as np

from . import shared320 as B                   # the simulator-free parts (were wm.collect.branch / explore)
from .shared320 import AttemptBase, FullRecorder
from ..tdmpc import demo320 as D
from ..tdmpc import wmdata320 as WD
from .agent320 import MPPI
from .common320 import RANDOM_EDGES_MM, check_offset_mode, gap_bin, jpeg, offset, servo_move, simulator_of_robot, unjpeg

EXECUTE = WD.EXECUTE
LEFT_M = 0.06                                  # as the policies' closed-loop test ("free" stops); the adapter's left_m:
                                               # ManiSkill LEFT_M, AutoMate LEFT_M + L (the user's decision 3 of 2026-11-25)
STEPS_MM = (0.5, 1.5)
BACKOFF_MM, BACKOFF_DECISIONS = 5.0, 2
DONE_P, DONE_TIMES, NOT_DONE_P = 0.5, 2, 0.5
MIN_TRY_MM, MIN_CORR_MM = 1.0, 0.5
KINDS = {"proposal": 0, "correction": 1, "random": 2, "shift": 3, "backoff": 4, "reposition": 5}
MODES = ("act", "act_done", "wm", "collect", "servo", "random")
SERVO_MIN_MM, SERVO_MAX_MM = 0.5, 1.5           # ablation "servo": dead band and largest move per decision
SERVO_RECENT = 4                                # ablation "servo", axis "motion": decisions of recent motion


def validate_attempt_args(mode: str, inside_mm: float | None = None, max_retries: int = 3, servo_axis: str = "true",
                          random_p: tuple | list | None = None, probe_max: int = 0) -> None:
    """Validate attempt() arguments without needing a simulator (also called by CPU unit tests)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "act_done" and inside_mm is not None:
        raise ValueError("mode act_done has no candidate corrections and does not accept inside_mm")
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


def _read_agent_offset_and_gap(agent, z, pin_length_m: float | None = None) -> tuple[np.ndarray | None, float | None]:
    """Read (offset_mm, gap_mm) from `agent` according to its offset_mode.
    In zero-privilege 'gap' mode, pin_length_m MUST be None (never read/passed at test time) and agent.gap_mm(z) is
    called with no pin_length_mm argument. In legacy 'along' mode, pin_length_m is passed to agent.gap_mm()."""
    omode = getattr(agent, "offset_mode", "along")
    if omode == "gap":
        if pin_length_m is not None:
            raise RuntimeError("zero-privilege violation: attempt320 must not pass/read self.L in offset_mode='gap'")
        g = agent.gap_mm(z)
        o = agent.offset_mm(z)
        return o, g
    if omode == "along":
        if pin_length_m is None:
            raise ValueError("legacy offset_mode='along' requires pin_length_m")
        o = agent.offset_mm(z)
        g = agent.gap_mm(z, pin_length_mm=float(pin_length_m) * 1000.0)
        return o, g
    raise ValueError(f"unknown offset_mode {omode!r}")


class _ForbidPinLength:
    """Context manager that raises RuntimeError if OnlineTester.L is accessed during test-time decision making."""

    def __init__(self, tester: "OnlineTester", active: bool):
        self.tester, self.active = tester, bool(active)

    def __enter__(self):
        if self.active:
            self.tester._forbid_L = True

    def __exit__(self, *exc):
        if self.active:
            self.tester._forbid_L = False


class OnlineTester(AttemptBase):
    """The attempt loop (module docstring). Simulator-free: an adapter (wm.online.maniskill320.ManiSkillTester,
    wm.online.automate320.AutomateTester) sets
        io           shared320.RecorderIO of its simulator (the recorder's reads)
        left_m       the "left" distance [m] from the start (ManiSkill LEFT_M; AutoMate LEFT_M + L)
        label_meta   extra keys of the labels' meta (AutoMate: simulator "automate"; ManiSkill: none)
    and defines truth_step(rel7) -> (success, depth_left_mm, lateral_mm) after a decision and truth_final(out) -> the
    summary's success fields (simulator truth: reporting only, never an input of the agent)."""

    @property
    def L(self) -> float:
        if getattr(self, "_forbid_L", False):
            raise RuntimeError("zero-privilege violation: attempt320 read self.L during test-time decision in offset_mode='gap'")
        return self._L_value

    @L.setter
    def L(self, value: float) -> None:
        self._L_value = value

    # ------------------------------------------------------------ the simulator's part (the adapter's)
    def recorder(self) -> FullRecorder:
        return FullRecorder(self.robot, self.io)

    def truth_step(self, rel7: np.ndarray) -> tuple[bool, float, float]:
        raise NotImplementedError("the simulator adapter defines truth_step")

    def truth_final(self, out: dict) -> dict:
        raise NotImplementedError("the simulator adapter defines truth_final")

    def face_gap_mm(self) -> float:
        """How far the pin tips are in front of the board face along the insertion axis [mm] (FK and H*; collect mode only)."""
        return -(self.depth(self.robot.hand_pose()) * 1000.0) - self.L * 1000.0

    # ------------------------------------------------------------ one attempt
    def attempt(self, agent, mode: str, seed: int, max_decisions: int = B.MAX_DECISIONS, margin: float = 0.5,
                max_retries: int = 3, p_shift: float = 0.5, eps: float = 0.1, n_candidates: int = 16, radius_mm: float = 3.0,
                random_reposition: float = 0.5, inside_mm: float | None = None, servo_axis: str = "true",
                random_p: tuple | list | None = None, probe_max: int = 0, probe_gap_mm: float = 6.0,
                planner: str = "fixed") -> tuple[dict, dict]:
        """-> (summary, recording). `inside_mm` (2026-11-18, the user's decision; None = off): once the estimated pin-tip
        depth g_t < -inside_mm (agent.gap_mm: offset_mode "gap" g_t = o[0], no pin length, required on AutoMate by the
        user's decision 6 of 2026-11-23; offset_mode "along" g_t = -o[0] - L), the try is in the hole and only the proposal is executed
        (no correction, no random candidate, no exploration push) until a back-off starts a new try; with planner "mppi" in
        mode "collect" the plans go on in the hole (corrections), still without random moves or the push. The decision and g_t at entry are logged (`inside_at`, `inside_gap_mm`). The recording is FullRecorder.end() plus `rgb_jpeg_list` (the stored JPEG bytes)
        and `dec_kind` (one KINDS value per executed decision)."""
        import torch
        validate_attempt_args(mode, inside_mm=inside_mm, max_retries=max_retries, servo_axis=servo_axis,
                              random_p=random_p, probe_max=probe_max)
        if planner not in ("fixed", "mppi"):
            raise ValueError(f"planner must be 'fixed' or 'mppi', not {planner!r}")
        if planner == "mppi" and (mode not in ("wm", "collect") or probe_max or max_retries or inside_mm is None):
            raise ValueError("planner mppi: modes wm / collect only, with inside_mm (the in-hole rule), without the probe "
                             "and without back-off (max_retries 0)")
        mppi_mean = None                                       # TD-MPC2 t0: the attempt's first plan starts from mean 0
        offset_mode = str(getattr(agent, "offset_mode", "along"))
        forbid_L = mode != "collect" and (offset_mode == "gap" or mode in ("act", "act_done"))
        check_offset_mode(offset_mode, simulator_of_robot(self.robot))   # AutoMate: offset_mode "gap" only (decision 6)
        r, rng = self.robot, np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rec = self.recorder()
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

        meta_l = {"H_star": self.H_star, "axis": self.F[:, 0], "pin_length_m": self.L, **self.label_meta}

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

        def truth_check(k_done: int):
            """Reporting only: the success criterion on the simulator's truth after k_done executed decisions."""
            ok, depth_mm, lat_mm = self.truth_step(rec.buf["eval_rel"][-1])
            log["truth_ok"].append(ok)
            log["truth_depth_mm"].append(round(float(depth_mm), 2))
            log["truth_lateral_mm"].append(round(float(lat_mm), 2))
            if log["first_success_decision"] is None and ok:
                log["first_success_decision"] = k_done

        hands, start_hand = [r.hand_pose()[:3, 3].copy()], r.hand_pose()[:3, 3].copy()
        try_start, try_corr = start_hand.copy(), np.zeros(3)
        excluded: list[np.ndarray] = []
        log = {"interventions": 0, "random_candidates": 0, "retries": 0, "shift": None, "done_p": [], "v_mean": [], "stall_states": [],
               "repositions": [], "inside_at": None, "inside_gap_mm": None, "servo_moves": 0, "per_decision": [], "probes": [],
               "first_success_decision": None, "truth_ok": [], "truth_depth_mm": [], "truth_lateral_mm": [], "latency_ms": []}
        shifted, done_run, why, q_last, k = mode != "collect" or rng.random() >= p_shift, 0, "time", r.qpos()[:7], 0
        Bt = WD._basis(np.asarray(self.F[:, 0], np.float64))   # label frame of the true axis: reporting only
        servo_hist, ax_last = [], None                         # servo moves per agent decision of this try; last axis
        probes_in_bin = [0] * (len(RANDOM_EDGES_MM) + 1)
        p, vm, m_now, inside = None, None, margin, False
        while k < max_decisions:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_dec, lat_ms = time.perf_counter(), None            # latency (reporting only): pictures -> chosen action
            j = len(rec.buf["rgb"]) - 1                         # the picture of this decision's start
            idx = D.context_indices(j)
            q9 = r.qpos().astype(np.float64)
            q = q9[:7].copy()
            probe_rec = None
            with _ForbidPinLength(self, forbid_L):
                z = agent.encode(np.stack([picture(i) for i in idx]), np.stack([rec.buf["depth"][i] for i in idx]), q)
                chunk = agent.propose(z)[:EXECUTE]                  # (8, 7) relative to q
                target, kind = chunk, "proposal"
                if mode == "act_done":
                    p = agent.done(z)
                    log["done_p"].append(round(p, 3))
                    done_run = done_run + 1 if p > DONE_P else 0
                    if done_run >= DONE_TIMES:                      # one more push with the proposal, then the end
                        q_last = execute(q[None] + chunk, q[None] + chunk, "proposal")
                        k += 1
                        truth_check(k)
                        why = "seated"
                        break
                elif mode != "act":
                    p = agent.done(z)
                    vm = float(agent.values(z).mean())
                    log["done_p"].append(round(p, 3))
                    log["v_mean"].append(round(vm, 3))
                    done_run = done_run + 1 if p > DONE_P else 0
                    o, g = _read_agent_offset_and_gap(agent, z, pin_length_m=None if offset_mode == "gap" else self.L)
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
                        truth_check(k)
                        why = "seated"
                        break
                    if planner == "mppi":                          # MPPI (agent320 MPPI): wm and collect only (checked above)
                        J = self.jacobian(q9)
                        Rh = r.hand_pose()[:3, :3]
                        # the in-hole rule switches MPPI off in the test mode "wm" only; collection plans always (decision of
                        # 2026-11-26); random moves and the shift push stay outside the hole, as with the fixed candidates
                        rec_d["offered"] = bool(mode == "collect" or not inside)
                        if rec_d["offered"]:
                            Wq = np.stack([offset(J, Rh[:, i] * MPPI["mm_per_unit"] / 1000.0) for i in range(3)], 1)  # (7, 3)
                            # no TD-MPC2 exploration noise, in collection either (the user's decision of 2026-11-26,
                            # replacing A/B): the executed plan is the model's correction, which the imitation head may
                            # copy as before; collection explores only by the sparse eps moves and the one shift push
                            plan = agent.plan_mppi(z, Wq, mppi_mean, eval_mode=(mode == "wm"), noise=False)
                            assert not plan["noisy"]
                            mppi_mean = plan["mean"]
                            mv_plan = Rh @ (plan["move_hand_mm"] / 1000.0)   # [m, world]
                            target, kind = chunk + plan["dq"][None], "correction"
                            rec_d.update(pick=1, u0=[round(float(x), 3) for x in plan["u0"]],
                                         std0=[round(float(x), 3) for x in plan["std0"]],
                                         v_max=round(plan["value_max"], 3), v_pi=round(plan["value_pi"], 3),
                                         move_mm=[round(float(x), 3) for x in Bt @ mv_plan * 1000.0])
                        if mode == "collect":
                            gap = self.face_gap_mm()
                            if not inside and 0.0 <= gap < 6.0 and rng.random() < eps:
                                # the collection's eps exploration exactly as with the fixed planner: one of the 12 fixed
                                # moves (0.5 / 1.5 mm along +-x, +-y, +-z of the hand frame), uniformly
                                moves_r = [s_ * Rh[:, i] * mm / 1000.0 for i in range(3) for s_ in (1.0, -1.0) for mm in STEPS_MM]
                                mv = moves_r[int(rng.integers(0, len(moves_r)))]
                                target, kind = chunk + offset(J, mv)[None], "random"
                                log["random_candidates"] += 1
                                rec_d["pick"] = -2
                                rec_d["move_mm"] = [round(float(x), 3) for x in Bt @ mv * 1000.0]
                        if kind == "correction":                   # the model's own plan was executed (not replaced)
                            log["interventions"] += 1
                            try_corr = try_corr + mv_plan
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        lat_ms = (time.perf_counter() - t_dec) * 1000.0
                        if mode == "collect" and not shifted and 0.5 <= gap <= 4.0 and not inside:   # no push in the hole
                            shifted = True
                            a = self.F[:, 0]
                            u = rng.normal(size=3)
                            u -= (u @ a) * a
                            u /= np.linalg.norm(u)
                            mm = float(rng.uniform(0.5, 2.0))
                            target, kind = target + offset(J, u * mm / 1000.0)[None], "shift"
                            log["shift"] = {"decision": k, "mm": round(mm, 3), "gap_mm": round(gap, 3)}
                    else:
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
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        lat_ms = (time.perf_counter() - t_dec) * 1000.0    # the choice is made; the probe below is not counted
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
            if lat_ms is None:                                 # mode act / act_done: the proposal is the choice
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lat_ms = (time.perf_counter() - t_dec) * 1000.0
            log["latency_ms"].append([k, round(lat_ms, 2)])
            q_last = execute(q[None] + target, q[None] + chunk, kind)
            if getattr(self, "side_capture", None) is not None:   # visual check only (run_wm320_attempts --side-video)
                self.side_capture(k)
            k += 1
            truth_check(k)
            hand = r.hand_pose()[:3, 3].copy()
            if mode not in ("act", "act_done") and probe_rec is not None:       # the executed branch against the real decision
                probe_rec["first_step_dev_mm"] = round(float(np.linalg.norm(hand - probe_rec.pop("first_hand")) * 1000.0), 4)
            if np.linalg.norm(hand - start_hand) > self.left_m:
                why = "left"
                break
            hands.append(hand)
            if len(hands) > B.STALL_DECISIONS and np.linalg.norm(hands[-1] - hands[-1 - B.STALL_DECISIONS]) < B.STALL_M:
                log["stall_states"].append({"decision": k, "done_p": None if p is None else round(p, 3), "v_mean": None if vm is None else round(vm, 3)})
                room = max_decisions - k >= BACKOFF_DECISIONS + 1 + B.STALL_DECISIONS + 1
                bad = mode not in ("act", "act_done") and p < NOT_DONE_P
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
                    truth_check(k)
                if np.linalg.norm(r.hand_pose()[:3, 3] - start_hand) > self.left_m:
                    why = "left"
                    break
                # re-position: candidate positions across the way in, each imagined, the best (or a random one) taken
                with _ForbidPinLength(self, forbid_L):
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
                truth_check(k)
                log["repositions"].append({"decision": k, "how": how, "mm": [round(float(x) * 1000, 3) for x in pos[pick]],
                                           "score_pick": round(float(sc[pick]), 3), "score_best": round(float(sc[best]), 3)})
                hands = [r.hand_pose()[:3, 3].copy()]
                try_start, try_corr, done_run, inside = hands[0].copy(), np.zeros(3), 0, False
                servo_hist = []
                if np.linalg.norm(hands[0] - start_hand) > self.left_m:
                    why = "left"
                    break
        for _ in range(4):
            r._command(q_last, 1)
            rec.step(q_last, q_last)
        out = rec.end()
        picture(len(out["rgb"]) - 1)
        out["rgb_jpeg_list"], out["dec_kind"] = jp, np.asarray(kinds, np.int8)
        lab = WD.decision_labels({"H_star": self.H_star, "axis": self.F[:, 0], "pin_length_m": self.L, **self.label_meta},
                                 out["hand"].astype(np.float64), np.array([EXECUTE * k, len(out["hand"]) - 1]))
        summary = {"mode": mode, "offset_mode": offset_mode, "stopped": why, "decisions": k, "seated_fk": bool(lab["seated"][0]),
                   "seated_fk_after_hold": bool(lab["seated"][1]),
                   **self.truth_final(out),
                   **{key: log[key] for key in ("interventions", "random_candidates", "retries", "shift", "stall_states", "repositions",
                                                "inside_at", "inside_gap_mm", "servo_moves", "per_decision", "probes",
                                                "first_success_decision", "truth_ok", "truth_depth_mm", "truth_lateral_mm",
                                                "latency_ms")},
                   "servo_axis": servo_axis if mode == "servo" else None,
                   "random_p": [float(x) for x in random_p] if mode == "random" else None,
                   "decision_kinds": {name: int((out["dec_kind"] == v).sum()) for name, v in KINDS.items()},
                   "done_p": log["done_p"], "v_mean": log["v_mean"], "model_step": agent.step}
        return summary, out


__all__ = ["OnlineTester", "KINDS", "MODES", "RANDOM_EDGES_MM", "gap_bin", "validate_attempt_args", "_read_agent_offset_and_gap"]
