"""AutoMate adapter of the online attempt (attempt320.OnlineTester) and one attempt on an AutoMate grasp (port plan
docs/2026-11-23_automate_online_port_plan.md, items 1-7 and 11; the user's decisions of 2026-11-25).

Importing this module needs no Isaac app (Isaac Lab is imported only by wm.sim_automate.robot / scene / expert, which
are imported inside the functions that need a live scene).

AutomateTester(robot, H_star, axis, fk, F): the attempt loop of attempt320 on wm.sim_automate.robot.Robot
    recorder     shared320.FullRecorder with AUTOMATE reads: joint velocities Robot.get_qvel(), the truth 7-vector
                 wm.sim_automate.record.truth_rel (= wm.data.recorder._truth_rel), and the picture of Robot.frame()
                 (the crop, depth, K and cam_T_world of the AutoMate collection's Recorder, byte for byte: the
                 collection records Robot.frame(); the float twin shim of the ManiSkill reads would round the depth
                 through float32, tests/test_online_port_automate_cpu.py T3)
    left_m       attempt320.LEFT_M + L = 6 cm + the tight-fit depth (the user's decision 3 of 2026-11-25)
    labels       the meta of WD.decision_labels carries simulator "automate" (labels, reward: unchanged from ManiSkill)
    max_decisions  robot.max_decisions = 150 + ceil(max(0, L - 20 mm) / 0.5 mm) (decision 6); any other value refused
    max_retries  0 only (decision 5: no back-off on AutoMate)
    probe_max    0 only: the counterfactual probe restores the simulator state, which on AutoMate needs the restore
                 check T8 first (plan item 1)
    success      AutoMate's own rule (decision 9; wm.sim_automate.checks, as the collector's automate_success):
                     s_t = [0 < h_t < d_dis AND k_t < 15 mm] after control step t = 1..T (row 0 = the placed start,
                     never counted); success = automate_success_any = OR_t s_t; success_end = s_T;
                     first_success_step = min t with s_t; first_success_decision = ceil(first_success_step / 8)
                     (a value decisions + 1: during the final 4-step hold); start_success = s_0
                 truth_ok per executed decision = s at the end of that decision (checks.truth_from_rel_spec);
                 truth_depth_mm / truth_lateral_mm and truth_info_end: ManiSkill-style numbers at the tip centre,
                 information only. start_success True means a wrong start (plan item 5: 0 of 1,820,000 starts of the
                 socket_top box satisfy the rule): StartSuccess is raised and the run stops (run_attempt checks the
                 placed start before the attempt; this check on row 0 stays as a second guard).
    gap          the agent's gap reads no L (offset_mode "gap" only, attempt320 / common320.check_offset_mode)
    collect mode's exploration windows (eps 0-6 mm, shift 0.5-4 mm) on face_gap_mm = g = s - L, the tip gap from the
                 tight-fit entrance (FK hand, the attempt's reference H* and L; decision 4: the ManiSkill definition)

run_attempt: one attempt on an AutoMate grasp (plan items 3, 4, 6), every step as the AutoMate collector
(wm.sim_automate.collect.collect_grasp) does it for one per-reset trajectory:
    seeds        attempt_streams: numpy SeedSequence(entropy = eval_seed, spawn_key = (int(assembly id), grasp index,
                 attempt index, purpose) [+ (round,) for an online round]); purposes P_* below, none of them a value
                 of the collector's STREAMS. Same seeds for every mode, independent of --grasps-n.
    reset        robot.reset(reset_seed, grasp of grasp.json, socket_index = 0): a new socket pose per attempt
    reference    Expert.reference_label(stream "reference"); accepted iff the expert completed AND AutoMate's rule
                 holds at its end AND the FK axis is defined (the collector's per-reset label rule, lead resolution
                 r3); otherwise "reference_failed" (not an attempt; no redraw of the socket)
    start        the collector's placement loop (Expert transfer TRANSFER = "move"; prepare refuses a collection whose
                 config.json transfer differs): first start sample_start(stream "start"), redraws from stream "place";
                 T_start = Expert.start_pose(H*, F, L, start, g_top, config.START_GAP_FROM) with F = frames.frame(FK
                 axis, robot base x) and g_top = checks.g_top_of(spec); a start closer than START_CLEARANCE_M (1 mm) is
                 redrawn ("clearance"); Expert.move_to_start; after a failed transfer the robot is re-homed by the same
                 reset; at most collect.PLACEMENT_ATTEMPTS (3); else "placement_failed" (not an attempt)
    start check  check_start right after the placement: AutoMate's rule s_0 on the placed state (record.truth_rel, the
                 value the attempt's row 0 would hold); s_0 true: the entry gets outcome "start_success" and
                 StartSuccess (carrying the entry) is raised: the run stops (plan item 5)
    attempt      AutomateTester.attempt from the placed start (attempt seed from stream "attempt")
A checks.ResetError of a reset gives "reset_failed", a checks.ContactReportError of the reference or the transfer
"contact_report_error" (the collector's soft outcomes; not attempts).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..sim_automate import checks as K
from ..sim_automate import config as C
from .attempt320 import EXECUTE, LEFT_M, OnlineTester
from .common320 import simulator_of_robot
from .shared320 import RecorderIO

# purposes of the attempt streams (spawn_key[3]); the collector's STREAMS are 0-6 (collect.STREAMS), never used here
P_SOCKET, P_START, P_PLACE, P_REFERENCE, P_ATTEMPT = 11, 12, 13, 14, 15
PURPOSES = {"socket": P_SOCKET, "start": P_START, "place": P_PLACE, "reference": P_REFERENCE, "attempt": P_ATTEMPT}
SOCKET_INDEX = 0                     # every attempt resets with its own seed and socket pose number 0 (plan item 4)
TRANSFER = "move"                    # Expert's unrecorded transfer; must equal the collection's config.json transfer


class StartSuccess(RuntimeError):
    """The placed start already satisfies AutoMate's rule: a wrong start (plan item 5); the run stops. entry: the
    attempt's entry (outcome "start_success") when it is known."""

    def __init__(self, msg: str, entry: dict | None = None):
        super().__init__(msg)
        self.entry = entry


# ---------------------------------------------------------------------------------------------- seeds (item 4)
def attempt_streams(eval_seed: int, assembly_id: str, gi: int, attempt: int, round_: int | None = None) -> dict:
    """The random streams of one attempt (module docstring): reset_seed (int), socket_index, SeedSequences start /
    place / reference, attempt_seed (int, the agent's generator and torch seed)."""
    def ss(p: int) -> np.random.SeedSequence:
        key = (int(assembly_id), int(gi), int(attempt), int(p)) + (() if round_ is None else (int(round_),))
        return np.random.SeedSequence(entropy=int(eval_seed), spawn_key=key)
    return {"reset_seed": int(ss(P_SOCKET).generate_state(1)[0]), "socket_index": SOCKET_INDEX,
            "start": ss(P_START), "place": ss(P_PLACE), "reference": ss(P_REFERENCE),
            "attempt_seed": int(ss(P_ATTEMPT).generate_state(1)[0])}


def check_plan_seeds(plan: dict, eval_seed: int, reset_seeds: list[int]) -> None:
    """SystemExit unless the attempts' resets (seed_a, 0) differ from every reset (grasp seed, k) of the collection
    (plan.json, ALL grasps: the collector resets with its grasp seed and k = 0..12), the evaluation entropy differs from
    the collection's (plan.json rng.entropy) and no purpose here is one of the collection's streams."""
    grasp_seeds = {int(g["seed"]) for g in plan["grasps"]}
    hit = sorted(set(int(s) for s in reset_seeds) & grasp_seeds)
    if hit:
        raise SystemExit(f"{len(hit)} attempt reset seeds equal a grasp seed of the collection (e.g. {hit[:3]}): their "
                         "socket poses would be the collection's; use another --seed")
    rng = plan.get("rng") or {}
    if rng.get("entropy") is not None and int(rng["entropy"]) == int(eval_seed):
        raise SystemExit(f"--seed {eval_seed} is the collection's entropy (plan.json rng.entropy): use another --seed")
    used = set(PURPOSES.values()) & {int(v) for v in (rng.get("streams") or {}).values()}
    if used:
        raise SystemExit(f"attempt stream purposes {sorted(used)} are streams of the collection")


def check_switches(plan: dict) -> dict:
    """SystemExit unless config.SOCKET_DRAW == "per_reset" == plan.json switches.socket_draw and config.START_GAP_FROM ==
    plan.json switches.start_gap_from (the test starts come from the collection's distribution, plan items 4, 6),
    every switch has its approved value (config.APPROVED, as the collector), config.switches() equals plan.json
    switches as a whole and config.PHYSICS_DEVICE equals plan.json physics_device (the collection's frozen
    switches.env). -> config.switches() (recorded)."""
    sw = plan.get("switches") or {}
    if C.SOCKET_DRAW != "per_reset" or sw.get("socket_draw") != "per_reset":
        raise SystemExit(f"socket draw: config {C.SOCKET_DRAW!r}, plan.json {sw.get('socket_draw')!r}; the online attempts "
                         "need 'per_reset' in both (a new socket pose per attempt)")
    if C.START_GAP_FROM != sw.get("start_gap_from"):
        raise SystemExit(f"start gap from: config {C.START_GAP_FROM!r}, plan.json {sw.get('start_gap_from')!r}: the starts "
                         "must come from the collection's start distribution")
    bad = C.unapproved_switches()
    if bad:
        raise SystemExit(f"switch values not approved: {bad}")
    now = C.switches()
    if now != sw:
        diff = sorted(k for k in set(now) | set(sw) if now.get(k) != sw.get(k))
        raise SystemExit(f"switches differ from plan.json's in {diff} (config {[now.get(k) for k in diff]}, plan.json "
                         f"{[sw.get(k) for k in diff]}): run with the collection's <collection>/inputs/switches.env")
    if C.PHYSICS_DEVICE != plan.get("physics_device"):
        raise SystemExit(f"physics device: config {C.PHYSICS_DEVICE!r}, plan.json {plan.get('physics_device')!r}: one "
                         "physics device per collection (the collection's switches.env)")
    return now


def check_inputs(grasps_dir, plan: dict) -> str:
    """SystemExit unless this process reads the collection's frozen assets: wm.sim_automate.assets.ASSET_DIR
    (WM_AUTOMATE_ASSETS) = plan.json input_roots.assets = <collection>/inputs/assets (resolved), collect.asset_files()
    = plan.json asset_files, and the collection's config.json transfer = automate320.TRANSFER. -> the asset folder."""
    from ..sim_automate import assets as AS
    from ..sim_automate import collect as CO
    grasps_dir = Path(grasps_dir)
    want = (plan.get("input_roots") or {}).get("assets")
    frozen = (grasps_dir / "inputs" / "assets").resolve()
    if not want or Path(want).resolve() != frozen:
        raise SystemExit(f"plan.json input_roots.assets {want!r} is not the collection's frozen {frozen}")
    if Path(AS.ASSET_DIR).resolve() != frozen:
        raise SystemExit(f"asset folder {AS.ASSET_DIR} (WM_AUTOMATE_ASSETS) is not the collection's frozen {frozen}: "
                         "export WM_AUTOMATE_ASSETS=<collection>/inputs/assets")
    if CO.asset_files() != plan.get("asset_files"):
        raise SystemExit(f"asset_files {CO.asset_files()} differ from plan.json's {plan.get('asset_files')}")
    try:
        transfer = json.loads((grasps_dir / "config.json").read_text()).get("transfer")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{grasps_dir / 'config.json'} unreadable: {exc}")
    if transfer != TRANSFER:
        raise SystemExit(f"the collection's transfer {transfer!r} differs from the online Expert's {TRANSFER!r}")
    return str(AS.ASSET_DIR)


# ---------------------------------------------------------------------------------------------- the adapter
def _picture(r):
    """Robot.frame(): the crop, depth (0.1 mm), K and cam_T_world that the collection's Recorder stores."""
    f = r.frame()
    return f.rgb, f.depth, f.K, np.asarray(f.cam_T_world, np.float64)


def _qvel(r) -> np.ndarray:
    return r.get_qvel()


def automate_io(beat=None) -> RecorderIO:
    """The AutoMate reads of the recorder. beat (optional): the worker's heartbeat, called (throttled by it) at every
    picture, so that the watcher sees a live worker during an attempt's decisions."""
    from ..sim_automate import robot as AR            # Isaac Lab (after the app's start)
    from ..sim_automate.record import truth_rel
    picture = _picture
    if beat is not None:
        def picture(r):
            beat()
            return _picture(r)
    return RecorderIO(qvel=_qvel, truth_rel=truth_rel, picture=picture, frame_cls=AR.Frame, cam_scale=C.CAM_SCALE)


# The strict criterion (the user's decision of 2026-11-26, "the real number"; AutoMate's own rule is the loose one and is
# kept beside it): at the attempt's last step, at the plug tip centre, depth left < 1.25 mm, full rotation angle < 4 deg,
# lateral < 2 mm (the thresholds of the ManiSkill experiment's success, demo320).
from ..tdmpc.demo320 import SUCCESS as _MS_SUCCESS    # the same numbers, one definition
STRICT = {"depth_mm": float(_MS_SUCCESS["depth_left_mm"]), "angle_deg": float(_MS_SUCCESS["angle_deg"]),
          "lateral_mm": float(_MS_SUCCESS["lateral_mm"])}


def truth_final(eval_rel: np.ndarray, spec) -> dict:
    """The summary's success fields of an attempt from its eval_rel (T + 1, 7) (module docstring, success)."""
    st = K.success_steps(np.asarray(eval_rel, np.float64), spec)
    end = K.truth_from_rel_spec(np.asarray(eval_rel[-1], np.float64), spec)
    strict = bool(end["depth_left_mm"] < STRICT["depth_mm"] and end["angle_deg"] < STRICT["angle_deg"]
                  and end["lateral_mm"] < STRICT["lateral_mm"])
    return {"success": bool(st["success_any"]), "automate_success_any": bool(st["success_any"]), "strict_success": strict,
            "success_end": bool(st["success_end"]), "first_success_step": st["first_success_step"],
            "start_success": bool(st["start_success"]), "height_mm_end": float(end["height_mm"]),
            "keypoint_dist_mm_end": float(end["keypoint_dist_mm"]),
            "truth_info_end": {k: float(end[k]) for k in ("depth_left_mm", "lateral_mm", "tilt_deg", "spin_deg",
                                                          "angle_deg", "tilt_symmetric_deg")}}


def band_above_top_mm(spec) -> float:
    """min(d_dis, 15 mm) - L_top [mm]: how far AutoMate's success region reaches above the socket top (plan item 5;
    > 0: a plug resting on the socket top within that height counts as success). Reported beside success figures."""
    return (min(float(spec.disassembly_dist_m), K.CLOSE_ERROR_THRESH_M) - float(spec.depth_top_m)) * 1000.0


class AutomateTester(OnlineTester):
    def __init__(self, robot, H_star: np.ndarray, axis: np.ndarray, fk, F: np.ndarray, io: RecorderIO | None = None,
                 beat=None):
        if simulator_of_robot(robot) != "automate":
            raise ValueError(f"AutomateTester needs wm.sim_automate's Robot, got {type(robot).__module__}.{type(robot).__name__}")
        super().__init__(robot, H_star, axis, fk=fk, F=F)
        self.io = automate_io(beat) if io is None else io
        self.left_m = LEFT_M + self.L                 # decision 3 of 2026-11-25 (environment side, not the agent's)
        self.label_meta = {"simulator": "automate"}
        self.spec = robot.port_spec

    def truth_step(self, rel7: np.ndarray) -> tuple[bool, float, float]:
        t = K.truth_from_rel_spec(np.asarray(rel7, np.float64), self.spec)
        return bool(t["success"]), float(t["depth_left_mm"]), float(t["lateral_mm"])

    def truth_final(self, out: dict) -> dict:
        return truth_final(out["eval_rel"], self.spec)

    def attempt(self, agent, mode: str, seed: int, max_decisions: int | None = None, max_retries: int = 0,
                probe_max: int = 0, **kw) -> tuple[dict, dict]:
        n_max = int(self.robot.max_decisions)
        if max_retries != 0:
            raise ValueError(f"max_retries {max_retries}: no back-off on AutoMate (the user's decision 5 of 2026-11-25)")
        if max_decisions is not None and int(max_decisions) != n_max:
            raise ValueError(f"max_decisions {max_decisions}: AutoMate's rule gives {n_max} for this assembly (decision 6)")
        if probe_max:
            raise ValueError("the counterfactual probe needs the AutoMate restore check (plan T8) first: probe_max 0")
        summary, out = super().attempt(agent, mode, seed, max_decisions=n_max, max_retries=0, probe_max=0, **kw)
        fs = summary["first_success_step"]
        summary["first_success_decision"] = None if fs is None else int(math.ceil(fs / EXECUTE))
        summary.update(max_decisions=n_max, left_m=self.left_m, band_above_top_mm=band_above_top_mm(self.spec))
        t = K.truth_from_rel_spec(np.asarray(out["eval_rel"], np.float64), self.spec)
        out["eval_tip"] = np.stack([np.asarray(t[k], np.float64) for k in K.TRUTH_COLUMNS], -1).astype(np.float32)
        if summary["start_success"]:
            raise StartSuccess(f"{self.robot.assembly_id}: the placed start satisfies AutoMate's rule (wrong start: "
                               f"g_top not applied, a stale spec or another START_GAP_FROM); the run stops")
        return summary, out


# ---------------------------------------------------------------------------------------------- one attempt
def grasp_kwargs(g: dict) -> dict:
    """The reset's grasp from an AutoMate grasp.json (pitch_deg = clamp offset, grasp_pos_m = dp, spin_deg)."""
    return {"clamp_deg": float(g["pitch_deg"]), "dp_m": float(g["grasp_pos_m"]), "spin_deg": float(g["spin_deg"])}


def reset_and_reference(robot, g: dict, idx: int, streams: dict, beat=None) -> dict:
    """Reset with the attempt's seed and socket pose number, then the reference (module docstring). -> a state dict
    {"entry", "reset" (the reset function, re-home), "expert", and on success "H", "axis", "F", "L", "g_top"};
    entry["outcome"] is set when there is no reference (reset_failed / contact_report_error / reference_failed)."""
    from ..sim_automate import collect as CO
    from ..sim_automate import frames
    from ..sim_automate.expert import Expert
    beat = beat or (lambda stage=None, **kw: None)
    kw = grasp_kwargs(g)
    seed, sidx = streams["reset_seed"], streams["socket_index"]
    entry = {"idx": idx, "reset_seed": seed, "socket_index": sidx, "attempt_seed": streams["attempt_seed"]}
    timing = {"reset_s": 0.0, "reference_s": 0.0, "reference_steps": 0, "transfer_s": 0.0, "transfer_steps": 0,
              "attempt_s": 0.0}
    entry["timing"] = timing

    def reset():
        t0 = time.time()
        robot.reset(seed, **kw, socket_index=sidx)
        timing["reset_s"] += time.time() - t0

    st = {"entry": entry, "reset": reset, "expert": None}
    beat(f"attempt {idx} reset")
    try:
        reset()
    except K.ResetError as exc:
        entry.update(outcome="reset_failed", error=f"{type(exc).__name__}: {exc}")
        return st
    ep = robot.episode
    entry["socket"] = {k: ep.get(k) for k in ("seed", "socket_index", "socket_pose", "socket_draw", "socket_check")}
    expert = st["expert"] = Expert(robot, transfer=TRANSFER)
    beat(f"attempt {idx} reference")
    t0 = time.time()
    try:
        lab = expert.reference_label(streams["reference"], on_step=lambda *a: beat())
    except K.ContactReportError as exc:
        timing["reference_s"] += time.time() - t0
        entry.update(outcome="contact_report_error", stage="reference", error=f"{type(exc).__name__}: {exc}")
        return st
    timing["reference_s"] += time.time() - t0
    timing["reference_steps"] += int(lab["steps"])
    lsucc = CO.automate_success(robot)
    completed = lab.get("completed")
    ok = (bool(lab["ok"]) and (completed is None or bool(completed)) and lsucc["inserted_end"]
          and lab["axis"] is not None)                      # the collector's label rule (collect_grasp, lead resolution r3)
    entry["reference"] = {"accepted": ok, "ok": bool(lab["ok"]), "completed": completed,
                          "inserted_end": lsucc["inserted_end"], "success": lsucc, "success_any": lab.get("success_any"),
                          "steps": int(lab["steps"]), "failure": lab.get("failure"), "truth": lab.get("truth"),
                          "peak": lab.get("peak"), "H_star_fk_vs_model_mm_deg": lab.get("H_star_fk_vs_model_mm_deg"),
                          "axis_defined": lab["axis"] is not None}
    if not ok:
        entry["outcome"] = "reference_failed"
        return st
    H, axis = np.asarray(lab["H_star"], np.float64), np.asarray(lab["axis"], np.float64)
    spec = robot.port_spec
    st.update(H=H, axis=axis, F=frames.frame(axis, robot.kin.base_T[:3, 0]), L=float(spec.pin_length_m),
              g_top=K.g_top_of(spec))
    entry.update(H_star=H.tolist(), axis=axis.tolist(), pin_length_m=st["L"], g_top_m=st["g_top"])
    return st


def place_start(robot, st: dict, streams: dict, beat=None) -> bool:
    """The collector's placement loop (collect_grasp, `for k in range(PLACEMENT_ATTEMPTS)`) from the state of
    reset_and_reference; records start, placement attempts / failures / legs and the start error in st["entry"];
    sets entry["outcome"] (contact_report_error / placement_failed) when no start was reached. -> placed."""
    from ..env.sampling import sample_start
    from ..sim_automate import collect as CO
    from ..sim_automate.expert import START_CLEARANCE_M, Expert
    beat = beat or (lambda stage=None, **kw: None)
    entry, expert, timing = st["entry"], st["expert"], st["entry"]["timing"]
    H, F, L, g_top = st["H"], st["F"], st["L"], st["g_top"]
    rng_start, rng_place = np.random.default_rng(streams["start"]), np.random.default_rng(streams["place"])
    start, placed, fails, k, mv, contact_fail = None, False, [], 0, None, None
    t0 = time.time()
    for k in range(CO.PLACEMENT_ATTEMPTS):
        beat(f"attempt {entry['idx']} placement {k + 1}")
        start = sample_start(rng_start) if k == 0 else sample_start(rng_place)
        T_start = Expert.start_pose(H, F, L, start, g_top, C.START_GAP_FROM)
        clr = expert.start_clearance(T_start)
        if clr["min"] < START_CLEARANCE_M:
            fails.append({"stage": "clearance", "clearance_mm": {a: round(b * 1e3, 2) for a, b in clr.items()}})
            continue
        try:
            mv = expert.move_to_start(T_start)
        except K.ContactReportError as exc:
            contact_fail = f"{type(exc).__name__}: {exc}"
            break
        timing["transfer_steps"] += mv["steps"]
        if mv["ok"]:
            placed = True
            break
        fails.append({"stage": mv.get("stage") or "start_error", "start_error_mm": mv["start_error_mm"], "legs": mv.get("legs")})
        try:
            st["reset"]()                                   # re-home (same grasp, seed and socket pose)
        except K.ResetError as exc:
            fails.append({"stage": "re-home reset", "error": f"{type(exc).__name__}: {exc}"})
            break
    timing["transfer_s"] += time.time() - t0
    entry.update(start=None if start is None else start.as_dict(), placement_attempts=k + 1, placement_failures=fails)
    if contact_fail is not None:
        entry.update(outcome="contact_report_error", stage="placement", error=contact_fail)
        return False
    if not placed:
        entry["outcome"] = "placement_failed"
        return False
    entry.update(start_error_mm=mv["start_error_mm"], start_error_deg=mv.get("start_error_deg"), placement_legs=mv.get("legs"),
                 start_hand_mm=[round(float(x) * 1000.0, 3) for x in robot.hand_pose()[:3, 3]])
    return True


def check_start(robot, entry: dict) -> None:
    """AutoMate's rule on the placed start (module docstring, start check): s_0 = checks.truth_from_rel_spec of
    record.truth_rel(robot), the value the attempt's recorder would store as row 0. s_0 true: entry["outcome"] =
    "start_success" (with the height and keypoint distance) and StartSuccess(entry) is raised."""
    from ..sim_automate.record import truth_rel
    t = K.truth_from_rel_spec(np.asarray(truth_rel(robot), np.float64), robot.port_spec)
    if bool(np.asarray(t["success"]).reshape(-1)[0]):
        entry.update(outcome="start_success",
                     start_truth={"height_mm": float(np.asarray(t["height_mm"]).reshape(-1)[0]),
                                  "keypoint_dist_mm": float(np.asarray(t["keypoint_dist_mm"]).reshape(-1)[0])})
        raise StartSuccess(f"{robot.assembly_id}: the placed start satisfies AutoMate's rule (wrong start: g_top not "
                           "applied, a stale spec or another START_GAP_FROM); the run stops", entry)


def run_attempt(robot, agent, fk, g: dict, idx: int, streams: dict, mode: str, attempt_kw: dict, beat=None) -> tuple[dict, dict | None, AutomateTester | None]:
    """One attempt (module docstring). -> (entry, recording or None, tester or None). The entry holds the reset seed
    and socket pose, the reference (accepted or why not), the start, the placement failures and timings, and either
    "attempt" (the summary) or "outcome" (reset_failed / contact_report_error / reference_failed / placement_failed).
    A successful start raises StartSuccess with the entry (outcome "start_success")."""
    beat = beat or (lambda stage=None, **kw: None)
    st = reset_and_reference(robot, g, idx, streams, beat)
    entry = st["entry"]
    if "outcome" in entry or not place_start(robot, st, streams, beat):
        return entry, None, None
    check_start(robot, entry)
    tester = AutomateTester(robot, st["H"], st["axis"], fk, st["F"], beat=beat)
    beat(f"attempt {idx} decisions")
    timing = entry["timing"]
    t0 = time.time()
    try:
        summ, out = tester.attempt(agent, mode, streams["attempt_seed"], **attempt_kw)
    except StartSuccess as exc:                         # the second guard (row 0 of the attempt)
        entry["outcome"] = "start_success"
        exc.entry = entry
        raise
    timing["attempt_s"] += time.time() - t0
    for key in ("reset_s", "reference_s", "transfer_s", "attempt_s"):
        timing[key] = round(timing[key], 2)
    entry["attempt"] = summ
    return entry, out, tester


__all__ = ["AutomateTester", "StartSuccess", "attempt_streams", "check_plan_seeds", "check_switches", "check_inputs",
           "run_attempt",
           "reset_and_reference", "place_start", "check_start", "TRANSFER",
           "truth_final", "band_above_top_mm", "grasp_kwargs", "PURPOSES"]
