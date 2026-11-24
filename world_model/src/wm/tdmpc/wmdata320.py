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
  reward   1000 * depth gain + shaping * (gamma * phi(next) - phi) + 10 at the last decision if seated there
           (potential-based shaping: the best behaviour is unchanged; its weight is a required training argument)
  term     the last decision of a trajectory that ended for good, decided from FK only: one that ends SEATED (any
           kind but random motion). Everything else is cut, not ended -- a jammed push too (decided with the
           researcher 2026-11-17: after a jam the robot backs off and tries again, so a jam is not an end; what
           follows it is the model's own value, learned from the online rounds' back-off retries). At a terminal
           state the shaping potential counts as 0, so the shaping leaves the best behaviour unchanged.
           Labels version LABELS.
  rotation the angle term of seated and phi: the full rotation angle arccos((tr(H*_R^T hand_R) - 1) / 2), for
           ManiSkill bit-identical to before. AutoMate labels = ManiSkill labels (full angle; decision 9 of
           2026-11-22: labels and reward unchanged from ManiSkill). tilt_definition(meta) names it "full_angle"; an
           AutoMate meta (or its switches) that names any other angle term is refused (no other label exists).
  probe    (along, lateral x, lateral y, rotation) relative to H* [mm, deg]: FK labels used to find the decision
           before the face (face_items); also the target of the stand-alone policy's training-only offset read-out
           head (policy320.ResnetFMPolicy.off), never read at test time
  offset   the target of the world model's offset read-out head (--offset-weight, 2026-11-18: wm320.WM320._offset),
           per the offset label of the data (offset_label(meta), OFFSET_LABELS; stored in the checkpoint):
             "along_mm" (ManiSkill; every checkpoint without the key): = probe, unchanged
             "gap_mm"   (AutoMate, the user's decision 6 of 2026-11-23): probe with its first component replaced by
                        the pin-tip gap g = s - L [mm], s = -(hand - H*) . axis the hand's distance back from the seated
                        pose along the insertion axis, L = pin_length_m (AutoMate: the tight-fit depth); g < 0: the
                        tip is |g| past the entrance. L is used here only, to build the training label; at test time
                        the agent reads g = o[0] directly (online.common320.gap_estimate_mm), no L, H* or truth
The bonus, the last reward and the seated label of the last observation all refer to the same instant (step 8n).
"""

from __future__ import annotations

import numpy as np

from . import demo320 as D

EXECUTE, H = 8, 4                      # control steps per decision; decisions per training window
SEATED_M, SEATED_DEG, BONUS = 1.5e-3, 4.0, 10.0
LABELS = "wm320-labels-3"              # 3: only a seated end is terminal; a jam is cut (no fixed continuation)
TILT_DEFINITIONS = ("full_angle",)     # the labels' one angle term (ManiSkill and AutoMate, decision 9 of 2026-11-22)
OFFSET_ALONG, OFFSET_GAP = "along_mm", "gap_mm"   # the offset head's first component (module docstring, offset)
OFFSET_LABELS = (OFFSET_ALONG, OFFSET_GAP)

ACTION_DIM = EXECUTE * D.JOINTS


def _basis(axis: np.ndarray) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([a, u, np.cross(a, u)])


def tilt_definition(meta: dict) -> str:
    """The angle term of the labels (module docstring): "full_angle". An AutoMate meta naming another term (top level
    or switches.tilt_definition) raises ValueError."""
    if meta.get("simulator") != "automate":
        return "full_angle"
    t = meta.get("tilt_definition", (meta.get("switches") or {}).get("tilt_definition", "full_angle"))
    if t not in TILT_DEFINITIONS:
        raise ValueError(f"{meta.get('grasp')}: tilt_definition {t!r} has no label here (known: {TILT_DEFINITIONS})")
    return t


def offset_label(meta: dict) -> str:
    """The offset label of a trajectory (module docstring, offset): "gap_mm" for AutoMate (meta simulator
    "automate"), "along_mm" otherwise."""
    return OFFSET_GAP if meta.get("simulator") == "automate" else OFFSET_ALONG


def offset_target(meta: dict, probe: np.ndarray) -> np.ndarray:
    """The world model's offset-head target from the probe labels (T, 4) of the same steps: probe itself for
    "along_mm"; for "gap_mm" a copy whose first component is the pin-tip gap g = -along - L [mm]."""
    if offset_label(meta) == OFFSET_ALONG:
        return probe
    out = probe.copy()
    out[:, 0] = (-probe[:, 0].astype(np.float64) - 1000.0 * float(meta["pin_length_m"])).astype(np.float32)
    return out


def decision_labels(meta: dict, hand: np.ndarray, steps: np.ndarray) -> dict:
    """Labels at the given control steps (hand: (T+1, 4, 4) recorded FK hand poses)."""
    Hs, L = np.asarray(meta["H_star"], np.float64), float(meta["pin_length_m"])
    if meta.get("axis") is None:
        raise ValueError(f"{meta.get('grasp')}: no insertion axis in the meta (the reference push had no axial push); "
                         "such a grasp has no labels")
    B = _basis(np.asarray(meta["axis"], np.float64))
    e = hand[steps, :3, 3] - Hs[:3, 3]
    along, lx, ly = (e @ B.T).T
    tilt_definition(meta)                                # refuses an AutoMate meta naming another angle term
    R = np.einsum("ij,tjk->tik", Hs[:3, :3].T, hand[steps, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    dist = np.linalg.norm(e, axis=1)
    lat = np.hypot(lx, ly)
    return {"depth": np.clip(L - dist, 0, L), "seated": (dist < SEATED_M) & (ang < SEATED_DEG),
            "phi": -(np.abs(along) * 1000 + lat * 1000 + ang),
            "probe": np.stack([along * 1000, lx * 1000, ly * 1000, ang], 1).astype(np.float32)}


def trajectory(z, meta: dict, shaping: float, gamma: float) -> dict | None:
    """Decision-level arrays of one fast-copy trajectory (an opened npz). None if it has no complete decision."""
    act, img, qpos, hand = z["action"], z["img_step"], z["qpos"], z["hand"]
    n = len(act) // EXECUTE
    if n < 1 or len(img) < 2 * n + 1 or not np.array_equal(img[:2 * n + 1], 4 * np.arange(2 * n + 1)):
        return None
    steps = EXECUTE * np.arange(n + 1)
    lab = decision_labels(meta, hand, steps)
    reward = np.diff(lab["depth"]) * 1000.0 + shaping * (gamma * lab["phi"][1:] - lab["phi"][:-1])
    if lab["seated"][n]:
        reward[-1] += BONUS
    term = np.zeros(n, np.float32)
    if meta["kind"] != "random":
        term[-1] = float(lab["seated"][n])
    if term[-1]:
        reward[-1] -= shaping * gamma * lab["phi"][n]                 # potential 0 at a terminal state
    q = qpos[steps, :D.JOINTS]                           # the arm's 7 joints only (qpos[7:9]: fingers, never an input)
    assert q.shape[-1] == D.JOINTS == 7
    delta = (act[:EXECUTE * n].reshape(n, EXECUTE, D.JOINTS) - q[:-1][:, None]).reshape(n, ACTION_DIM)
    return {"n": n, "delta": delta.astype(np.float32), "reward": reward.astype(np.float32), "term": term,
            "q": q.astype(np.float32), "done": lab["seated"].astype(np.float32),
            "depth": lab["depth"].astype(np.float32), "probe": lab["probe"], "offset": offset_target(meta, lab["probe"]),
            "end_seated": bool(lab["seated"][n]),
            "kind": meta["kind"], "outcome": meta.get("outcome")}


def scale_action(delta: np.ndarray, s0: np.ndarray, top: np.ndarray) -> np.ndarray:
    """asinh scaling to [-1, 1] (as wm/tdmpc/data.py): near the board the moves are ~2 % of the largest ones."""
    return np.clip(np.arcsinh(delta / s0) / np.arcsinh(top / s0), -1.0, 1.0).astype(np.float32)


def window(tr: dict, k: int, s0, top, warm_next: bool = True) -> dict:
    """The window of H decisions starting at decision k (0 <= k < n). Picture slots s = 0..H+3 hold the pictures
    2k - 6 + 2s (clamped to [0, 2n]); observation j uses slots j..j+3 and the joints of decision min(k+j, n)."""
    n = tr["n"]
    slots = np.clip(2 * k - 6 + 2 * np.arange(H + 4), 0, 2 * n)
    dec = np.minimum(k + np.arange(H + 1), n)
    act = scale_action(tr["delta"], s0, top)
    out = {"slots": slots, "q": tr["q"][dec], "done": tr["done"][dec], "fk_depth": tr["depth"][dec], "probe": tr["probe"][dec],
           "offset": tr["offset"][dec], "obs_valid": (k + np.arange(H + 1) <= n).astype(np.float32),
           "v_valid": (k + np.arange(H + 1) < n).astype(np.float32),
           "action": np.zeros((H, ACTION_DIM), np.float32), "reward": np.zeros(H, np.float32), "term": np.zeros(H, np.float32),
           "valid": np.zeros(H, np.float32), "a_next": np.zeros((H + 1, ACTION_DIM), np.float32), "a_next_valid": np.zeros(H + 1, np.float32)}
    m = min(H, n - k)
    out["action"][:m], out["reward"][:m], out["term"][:m], out["valid"][:m] = act[k:k + m], tr["reward"][k:k + m], tr["term"][k:k + m], 1.0
    for j in range(H + 1):                       # the data's own action at each observation (the warm-up's next action)
        if k + j < n:
            out["a_next"][j], out["a_next_valid"][j] = act[k + j], 1.0
    return out


def action_scale(trajs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per action dimension: the median magnitude and the 99.9th percentile (at least twice the median)."""
    d = np.abs(np.concatenate([t["delta"] for t in trajs]))
    s0 = np.maximum(np.percentile(d, 50, axis=0), 1e-5)
    return s0.astype(np.float32), np.maximum(np.percentile(d, 99.9, axis=0), 2 * s0).astype(np.float32)
