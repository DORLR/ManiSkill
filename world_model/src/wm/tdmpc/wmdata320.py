"""Data of the 320 px world model (wm320.py): labels and training windows from the fast copy (demo320.py).
numpy + PIL only, so that it is tested without a GPU.

One step of the model = one decision of a policy: 8 control steps (0.4 s); decision d starts at control step 8d,
where picture 2d was taken. Observation of decision d = the pictures [2d-6, 2d-4, 2d-2, 2d] (0.4 s apart, the first
repeated at the start, exactly the policy's context) and the 7 joint angles measured at step 8d.
Action of decision d = the 8 executed joint targets minus the joints measured at step 8d (56 numbers).

Labels, from the joints (the recorded FK hand pose) and the grasp's seated hand pose H*, insertion axis and pin
length only; never from the simulator's truth (the fast copy does not even contain it):
  depth    pin length - |hand - H*|, clipped to [0, pin length]                          [m]
  seated   |hand - H*| < 1.5 mm and rotation to H* < 4 deg (the success criterion's 4 deg)
  phi      shaping potential -(|along| + |lateral| [mm] + rotation [deg, 1 deg counted as 1 mm]), along / lateral
           relative to the insertion axis
  reward   1000 * depth gain + shaping * (gamma * phi(next) - phi) + 10 at the terminal seated decision
           (potential-based shaping: the best behaviour is unchanged; its weight is a required training argument)
  term     ManiSkill ("last_step"): the last decision (step 8n) if seated there (unchanged, bit-identical to before).
           AutoMate ("first_seated"): the FIRST decision of a non-random trajectory whose next state is seated (FK):
           `+BONUS`, `term = 1`, and zero terminal shaping potential are placed on that first seated transition
           `k_term`, post-terminal rewards are zeroed, and transition/value masks stop at `n_eff = k_term + 1` while
           `done_valid` keeps stationary post-terminal seated frames (`done == 1`) for the real-observation `done` head
           so `DONE_TIMES = 2` sees bottomed-out stationary positives. Everything else is cut, not ended -- a jammed
           push too (decided with the researcher 2026-11-17). Labels version LABELS.
  rotation ManiSkill: the full rotation angle arccos((tr(H*_R^T hand_R) - 1) / 2), bit-identical to before ("full_angle").
           AutoMate: defaults to "symmetric_angle" using meta["symmetry_n"] (axis tilt for "round", min over n-fold
           rotations about the insertion axis for integer n >= 1, None -> n=1 matching checks.period_deg), so visually
           unobservable axial spin on symmetric plugs does not poison seated/done/+BONUS/phi labels. Explicit
           "full_angle" is also accepted; missing/invalid symmetry_n or unknown tilt_definition names are refused.
  probe    (along, lateral x, lateral y, rotation) relative to H* [mm, deg], with offset_mode (OFFSET_MODES; stored
           in the checkpoint as "offset_mode"):
             "along" (legacy; every checkpoint without the key): the first component is along, as before
             "gap"   (zero-privilege): the first component is the pin-tip gap g = -along - L [mm] (L = pin_length_m;
                     AutoMate: the tight-fit depth), g < 0: the tip is |g| past the entrance. L is used here only, to
                     build the training label; at test time the agent reads g = o[0] directly (agent320.WMAgent.gap_mm),
                     no L, H* or truth. AutoMate runs use "gap" only (train_wm320.py, agent320, attempt320 refuse
                     anything else; the user's decision 6 of 2026-11-23)
           The target of the world model's offset read-out head (--offset-weight, 2026-11-18: wm320.WM320._offset) and
           of the stand-alone policy's training-only offset read-out head (policy320.ResnetFMPolicy.off, "along"),
           never read at test time; gap_mm (whatever the mode) finds the decision before the face (face_items)
The bonus, the terminal reward and the seated label of the terminal next-observation all refer to the same instant
(step 8n on ManiSkill; step 8*(k_term + 1) on AutoMate).
"""

from __future__ import annotations

import numpy as np

from . import demo320 as D

EXECUTE, H = 8, 4                      # control steps per decision; decisions per training window
SEATED_M, SEATED_DEG, BONUS = 1.5e-3, 4.0, 10.0
LABELS = "wm320-labels-3"              # 3: only a seated end is terminal; a jam is cut (no fixed continuation)
TILT_DEFINITIONS = ("full_angle", "symmetric_angle")     # ManiSkill: full_angle; AutoMate default: symmetric_angle (symmetry_n)
TERM_MODES = ("last_step", "first_seated")               # ManiSkill: last_step; AutoMate: first_seated
OFFSET_MODES = ("along", "gap")        # dim 0 of probe: "along" = hand along axis (legacy); "gap" = pin-tip gap (-along - L, zero-privilege)

ACTION_DIM = EXECUTE * D.JOINTS


def _basis(axis: np.ndarray) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([a, u, np.cross(a, u)])


def tilt_definition(meta: dict) -> str:
    """The angle term of the labels: "full_angle" for ManiSkill, "symmetric_angle" by default for AutoMate (honouring
    an explicit tilt_definition in meta or switches if in TILT_DEFINITIONS; unknown names raise ValueError)."""
    if meta.get("simulator") != "automate":
        return "full_angle"
    t = meta.get("tilt_definition", (meta.get("switches") or {}).get("tilt_definition", "symmetric_angle"))
    if t not in TILT_DEFINITIONS:
        raise ValueError(f"{meta.get('grasp')}: tilt_definition {t!r} has no label here (known: {TILT_DEFINITIONS})")
    return t


def term_mode(meta: dict) -> str:
    """Termination alignment mode: "first_seated" for AutoMate, "last_step" for ManiSkill (unchanged)."""
    return "first_seated" if meta.get("simulator") == "automate" else "last_step"


def _symmetric_angle_deg(hand_R: np.ndarray, Hs_R: np.ndarray, B: np.ndarray, symmetry_n, grasp=None) -> np.ndarray:
    """Rotation angle [deg] from Hs_R to hand_R modulo n-fold symmetry about B[0] (the insertion axis).
    symmetry_n == 'round': pure axis tilt arccos((B R_rel B^T)_00).
    symmetry_n is None (unknown, matching checks.period_deg) or integer n >= 1: min over k=0..n-1 of full angle of
    R_x(2 pi k / n)^T (B R_rel B^T). Invalid values (bool, <= 0, non-integral float, unknown str) raise ValueError."""
    R_rel = np.einsum("tij,kj->tik", hand_R, Hs_R)                       # hand_R @ Hs_R^T (world-frame relative rot)
    R_ax = np.einsum("ij,tjk,lk->til", B, R_rel, B)                      # expressed in insertion-axis basis B
    if symmetry_n == "round":
        return np.degrees(np.arccos(np.clip(R_ax[:, 0, 0], -1.0, 1.0)))
    if symmetry_n is None:
        n = 1
    else:
        if isinstance(symmetry_n, bool) or not isinstance(symmetry_n, (int, float, np.integer, np.floating)):
            raise ValueError(f"{grasp}: symmetric_angle requires symmetry_n ('round', None, or int >= 1), got {symmetry_n!r}")
        nf = float(symmetry_n)
        if not np.isfinite(nf) or nf < 1.0 or nf != int(nf):
            raise ValueError(f"{grasp}: symmetric_angle requires symmetry_n ('round', None, or int >= 1), got {symmetry_n!r}")
        n = int(nf)
    if n == 1:
        return np.degrees(np.arccos(np.clip((np.trace(R_ax, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    a = 2.0 * np.pi * np.arange(n, dtype=np.float64) / n                 # (n,)
    tr = (R_ax[:, 0, 0:1]
          + np.cos(a)[None] * (R_ax[:, 1, 1:2] + R_ax[:, 2, 2:3])
          + np.sin(a)[None] * (R_ax[:, 2, 1:2] - R_ax[:, 1, 2:3]))
    return np.degrees(np.arccos(np.clip((tr.max(axis=1) - 1.0) / 2.0, -1.0, 1.0)))


def decision_labels(meta: dict, hand: np.ndarray, steps: np.ndarray, offset_mode: str = "along") -> dict:
    """Labels at the given control steps (hand: (T+1, 4, 4) recorded FK hand poses).

    offset_mode:
      - 'along' (legacy): probe[:, 0] = along_mm (hand offset from H* along insertion axis, <= 0 before H*).
      - 'gap' (zero-privilege): probe[:, 0] = gap_mm = -along_mm - L_mm (pin-tip distance in front of board face,
        positive in front of board, negative inside hole; computed using training metadata pin_length_m only).
    """
    if offset_mode not in OFFSET_MODES:
        raise ValueError(f"unknown offset_mode {offset_mode!r}; expected one of {OFFSET_MODES}")
    Hs, L = np.asarray(meta["H_star"], np.float64), float(meta["pin_length_m"])
    if meta.get("axis") is None:
        raise ValueError(f"{meta.get('grasp')}: no insertion axis in the meta (the reference push had no axial push); "
                         "such a grasp has no labels")
    B = _basis(np.asarray(meta["axis"], np.float64))
    e = hand[steps, :3, 3] - Hs[:3, 3]
    along, lx, ly = (e @ B.T).T
    tdef = tilt_definition(meta)
    if tdef == "symmetric_angle":
        if "symmetry_n" not in meta:
            raise ValueError(f"{meta.get('grasp')}: symmetric_angle requires 'symmetry_n' in meta")
        ang = _symmetric_angle_deg(hand[steps, :3, :3], Hs[:3, :3], B, meta["symmetry_n"], meta.get("grasp"))
    else:
        R = np.einsum("ij,tjk->tik", Hs[:3, :3].T, hand[steps, :3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    dist = np.linalg.norm(e, axis=1)
    lat = np.hypot(lx, ly)
    along_mm = along * 1000.0
    gap_mm = -along_mm - L * 1000.0
    dim0 = gap_mm if offset_mode == "gap" else along_mm
    return {"depth": np.clip(L - dist, 0, L), "seated": (dist < SEATED_M) & (ang < SEATED_DEG),
            "phi": -(np.abs(along) * 1000 + lat * 1000 + ang),
            "along_mm": along_mm.astype(np.float32), "gap_mm": gap_mm.astype(np.float32),
            "probe": np.stack([dim0, lx * 1000, ly * 1000, ang], 1).astype(np.float32)}


def effective_decisions(z, meta: dict) -> int:
    """Number of active decisions before termination (n_eff <= n) for window_items() filtering.
    ManiSkill and random trajectories use all n decisions; AutoMate non-random stops at first seated (k_term + 1)."""
    n = len(z["action"]) // EXECUTE
    if n < 1 or term_mode(meta) != "first_seated" or meta.get("kind") == "random":
        return n
    lab = decision_labels(meta, z["hand"], EXECUTE * np.arange(n + 1))
    hits = np.nonzero(lab["seated"][1:])[0]
    return int(hits[0]) + 1 if len(hits) else n


def trajectory(z, meta: dict, shaping: float, gamma: float, offset_mode: str = "along") -> dict | None:
    """Decision-level arrays of one fast-copy trajectory (an opened npz). None if it has no complete decision."""
    act, img, qpos, hand = z["action"], z["img_step"], z["qpos"], z["hand"]
    n = len(act) // EXECUTE
    if n < 1 or len(img) < 2 * n + 1 or not np.array_equal(img[:2 * n + 1], 4 * np.arange(2 * n + 1)):
        return None
    steps = EXECUTE * np.arange(n + 1)
    lab = decision_labels(meta, hand, steps, offset_mode=offset_mode)
    reward = np.diff(lab["depth"]) * 1000.0 + shaping * (gamma * lab["phi"][1:] - lab["phi"][:-1])
    term = np.zeros(n, np.float32)
    tmode = term_mode(meta)
    if tmode == "first_seated" and meta["kind"] != "random":
        hits = np.nonzero(lab["seated"][1:])[0]              # first decision whose next state is seated
        if len(hits):
            k_term = int(hits[0])
            term[k_term] = 1.0
            reward[k_term] += BONUS
            reward[k_term] -= shaping * gamma * lab["phi"][k_term + 1]   # potential 0 at terminal state
            if k_term + 1 < n:
                reward[k_term + 1:] = 0.0                    # post-terminal hold/extra-push masked out in window()
            end_seated = True
        else:
            end_seated = False
    else:
        end_seated = bool(lab["seated"][n])
        if end_seated:
            reward[-1] += BONUS
        if end_seated and meta["kind"] != "random":
            term[-1] = 1.0
            reward[-1] -= shaping * gamma * lab["phi"][n]    # potential 0 at a terminal state
    q = qpos[steps, :D.JOINTS]                           # the arm's 7 joints only (qpos[7:9]: fingers, never an input)
    assert q.shape[-1] == D.JOINTS == 7
    delta = (act[:EXECUTE * n].reshape(n, EXECUTE, D.JOINTS) - q[:-1][:, None]).reshape(n, ACTION_DIM)
    out = {"n": n, "delta": delta.astype(np.float32), "reward": reward.astype(np.float32), "term": term,
           "q": q.astype(np.float32), "done": lab["seated"].astype(np.float32),
           "depth": lab["depth"].astype(np.float32), "probe": lab["probe"], "gap_mm": lab["gap_mm"],
           "offset_mode": offset_mode, "end_seated": end_seated,
           "kind": meta["kind"], "outcome": meta.get("outcome")}
    if tmode == "first_seated":
        out["term_mode"] = "first_seated"
    return out


def scale_action(delta: np.ndarray, s0: np.ndarray, top: np.ndarray) -> np.ndarray:
    """asinh scaling to [-1, 1] (as wm/tdmpc/data.py): near the board the moves are ~2 % of the largest ones."""
    return np.clip(np.arcsinh(delta / s0) / np.arcsinh(top / s0), -1.0, 1.0).astype(np.float32)


def window(tr: dict, k: int, s0, top, warm_next: bool = True) -> dict:
    """The window of H decisions starting at decision k (0 <= k < n). Picture slots s = 0..H+3 hold the pictures
    2k - 6 + 2s (clamped to [0, 2n]); observation j uses slots j..j+3 and the joints of decision min(k+j, n).
    Under AutoMate first_seated termination (tr['term_mode'] == 'first_seated'), transition/value/imagined masks stop at
    n_eff = k_term + 1 while done_valid retains real post-terminal stationary seated observations (k + j <= n and
    done == 1) for the real-latent done classifier. ManiSkill windows are unchanged bit for bit."""
    n = tr["n"]
    first_seated = tr.get("term_mode") == "first_seated"
    if first_seated:
        hits = np.nonzero(tr["term"] > 0.5)[0]
        n_eff = int(hits[0]) + 1 if len(hits) else n
    else:
        n_eff = n
    slots = np.clip(2 * k - 6 + 2 * np.arange(H + 4), 0, 2 * n)
    dec = np.minimum(k + np.arange(H + 1), n)
    act = scale_action(tr["delta"], s0, top)
    ks = k + np.arange(H + 1)
    out = {"slots": slots, "q": tr["q"][dec], "done": tr["done"][dec], "fk_depth": tr["depth"][dec], "probe": tr["probe"][dec],
           "obs_valid": ((ks <= n_eff) & (k < n_eff)).astype(np.float32),
           "v_valid": (ks < n_eff).astype(np.float32),
           "action": np.zeros((H, ACTION_DIM), np.float32), "reward": np.zeros(H, np.float32), "term": np.zeros(H, np.float32),
           "valid": np.zeros(H, np.float32), "a_next": np.zeros((H + 1, ACTION_DIM), np.float32), "a_next_valid": np.zeros(H + 1, np.float32)}
    if first_seated:
        out["done_valid"] = (((ks <= n_eff) | ((ks <= n) & (tr["done"][dec] > 0.5))) & (k < n_eff)).astype(np.float32)
    if "gap_mm" in tr:
        out["gap_mm"] = tr["gap_mm"][dec]
    m = max(0, min(H, n_eff - k))
    if m > 0:
        out["action"][:m], out["reward"][:m], out["term"][:m], out["valid"][:m] = act[k:k + m], tr["reward"][k:k + m], tr["term"][k:k + m], 1.0
    for j in range(H + 1):                       # the data's own action at each observation (the warm-up's next action)
        if k + j < n_eff:
            out["a_next"][j], out["a_next_valid"][j] = act[k + j], 1.0
    return out


def action_scale(trajs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per action dimension: the median magnitude and the 99.9th percentile (at least twice the median)."""
    d = np.abs(np.concatenate([t["delta"] for t in trajs]))
    s0 = np.maximum(np.percentile(d, 50, axis=0), 1e-5)
    return s0.astype(np.float32), np.maximum(np.percentile(d, 99.9, axis=0), 2 * s0).astype(np.float32)
