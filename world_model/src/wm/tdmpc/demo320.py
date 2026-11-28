"""Data logic of the 320 px policy: the fast copy of the demonstrations, the split, the policy's samples.
numpy + PIL + the standard library only (no torch, no OpenCV), so that all of it runs in a test without a GPU.

Fast copy ("wm-fast-1"), written once by scripts/data/cache_wm_v1_fast.py: one uncompressed .npz per trajectory with
the stored JPEG pictures byte for byte, every depth picture as a lossless 16-bit PNG, the joints, hand poses, joint
targets and the meta. Why: the policy is trained on independent samples (as the old policy was), one file read per
sample; the original files inflate the whole depth array (18-40 MB) at every read. The simulator's truth arrays
(`eval_rel`, `eval_tip`) are not copied; trajectories of the test ports A-D are not copied at all. The meta of an
AutoMate trajectory (meta simulator = "automate") keeps only the keys of FAST_META_AUTOMATE (2026-11-22): its raw meta
also holds simulator truth (eval_socket_pose, truth, peak forces, stages, start_error_mm, depth_left_mm, the seeds and
rng that regenerate the socket pose), none of which reaches the fast copy; entry_of refuses an AutoMate fast copy
whose meta holds any other key (a copy written before the allow-list). ManiSkill meta is copied whole, as before.
An AutoMate trajectory of condition "test" is never converted (convert raises; the final-test assemblies stay out).

Imitation set (`bc` of load_entries; lead resolution r3, option A, 2026-11-22): a trajectory of a POLICY_KINDS kind
with outcome "inserted". AutoMate: outcome "inserted" is AutoMate's episode success S_any (sim_automate.collect,
the reported success number), so bc also needs expert_completed (the expert passed Chongyu's final_verify), as
ManiSkill's "inserted" always did:
    bc = kind in POLICY_KINDS  AND  outcome == "inserted"  AND  (ManiSkill  OR  expert_completed)
AutoMate's success itself travels separately as automate_success_any (fast meta, manifest entry), for reports only.

Split (decided 2026-11-17): held out = grasps with pitch in (-42, -38) deg or position in (-21, -19) mm, and the 13
held-out shapes; ports A-D = final test; of the remaining grasps one in twenty (crc32 of the grasp's name, the rule
of the previous model's "dev" split) is the SELECTION split: not trained on, used to choose checkpoints, so that
nothing is chosen on the held-out set.
AutoMate data (entries / grasps with simulator "automate", 2026-11-22): split320's bands and ports mean nothing there.
Its split is the grasp's stored `condition` (written at collection by sim_automate.collect.SPLIT_RULES), mapped by
AUTOMATE_SPLIT_OF_CONDITION; "unassigned", a missing or an unknown condition is refused (SplitUndecided). Every
consumer takes the split from split_of_entry (load_entries, the online pool, the online / evaluation scripts, the
depth statistics, checkpoint selection); automate_condition is the one place a later split rule plugs in.
"""

from __future__ import annotations

import io
import json
import os
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.splits import OOD_VAL_SHAPES, TEST_PORTS

IMAGE_EVERY, JOINTS = 4, 7
CHUNK = 16                           # joint targets per policy output (0.8 s at 20 Hz); 8 are executed
CONTEXT, STRIDE_IMG = 4, 2           # the policy's pictures: 4, every second stored picture (0.4 s apart)
POLICY_KINDS = ("reference", "insert", "perturbed")       # successful expert trajectories, as the old policy
SELECT_ONE_IN = 20
MIN_SELECT_GRASPS = 20               # AutoMate: checkpoints are not chosen on fewer selection grasps (train_wm320.py
                                     # refuses such data; the collector's plan counts against it before collecting)
FORMAT = "wm-fast-1"
SPLIT_MODES = ("split320", "strict_extrap")
SUCCESS = {"depth_left_mm": 1.25, "angle_deg": 4.0, "lateral_mm": 2.0}      # the one success criterion (truth, at the end)


def is_success(truth: dict) -> bool:
    """The success criterion on the simulator's final state: every quantity strictly below its bound."""
    return all(float(truth[k]) < v for k, v in SUCCESS.items())


def is_interior_strip(pitch_deg: float, pos_m: float) -> bool:
    """Interior interpolation strips held out in both split320 and strict_extrap."""
    x = pos_m * 1000.0
    return -42.0 < pitch_deg < -38.0 or -21.0 < x < -19.0


def is_extrap_band(pitch_deg: float, pos_m: float) -> bool:
    """Outer extrapolation bands (pitch <= -55 or >= -25 deg, pos <= -28 or >= -12 mm)."""
    x = pos_m * 1000.0
    return pitch_deg <= -55.0 or pitch_deg >= -25.0 or x <= -28.0 or x >= -12.0


def split320(port: str, pitch_deg: float, pos_m: float, grasp: str, split_mode: str = "split320") -> str:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")
    if port in TEST_PORTS:
        return "test"
    if port in OOD_VAL_SHAPES or is_interior_strip(pitch_deg, pos_m):
        return "heldout"
    if split_mode == "strict_extrap" and is_extrap_band(pitch_deg, pos_m):
        return "heldout"
    return "select" if zlib.crc32(grasp.encode()) % SELECT_ONE_IN == 0 else "train"


def _call_split(port: str, pitch_deg: float, pos_m: float, grasp: str, split_mode: str = "split320") -> str:
    """Call split320 preserving 4-arg backwards compatibility when split_mode == 'split320' (tests may monkeypatch split320)."""
    if split_mode == "split320":
        return split320(port, pitch_deg, pos_m, grasp)
    return split320(port, pitch_deg, pos_m, grasp, split_mode=split_mode)


# ---------------------------------------------------------------- AutoMate's split (from the stored condition)
AUTOMATE_SPLIT_OF_CONDITION = {"train": "train", "select": "select", "heldout": "heldout", "heldout_grasp": "heldout",
                               "heldout_shape": "heldout", "test": "test"}
UNDECIDED = "unassigned"


class SplitUndecided(ValueError):
    """An AutoMate grasp without a decided split. A ValueError (not SystemExit), so that a collection-time check that
    catches Exception still writes its report; the training-side scripts turn it into SystemExit."""


def is_automate(e: dict) -> bool:
    return e.get("simulator") == "automate"


def automate_condition(e: dict) -> str | None:
    """The condition of an AutoMate entry or grasp.json: the stored per-grasp `condition` (collect.SPLIT_RULES at
    collection time). The one plug point for a split rule decided after collection (e.g. a versioned split file keyed
    by grasp name): it must be applied here, so that every consumer sees it; train_files_sha256 hashes this value
    for AutoMate entries, so a new rule changes it. No such rule exists yet; the stored value is returned unchanged."""
    return e.get("condition")


def split_of_entry(e: dict, allow_undecided: bool = False, split_mode: str = "split320") -> str:
    """The split of a manifest entry or grasp.json. ManiSkill (no simulator key): split320 with split_mode
    (_call_split; "split320" unchanged). AutoMate: the
    stored condition through AUTOMATE_SPLIT_OF_CONDITION; "unassigned" or a missing condition raises SplitUndecided
    (allow_undecided: returns "unassigned" instead, for collection-time reports only); an unknown condition always
    raises."""
    if not is_automate(e):
        return _call_split(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"], split_mode)
    c = automate_condition(e)
    if c in AUTOMATE_SPLIT_OF_CONDITION:
        return AUTOMATE_SPLIT_OF_CONDITION[c]
    if c in (None, UNDECIDED):
        if allow_undecided:
            return UNDECIDED
        raise SplitUndecided(f"AutoMate grasp {e.get('grasp')} has no decided split (condition {c!r}): the train / "
                             "held-out split of AutoMate is not decided yet")
    raise SplitUndecided(f"AutoMate grasp {e.get('grasp')} has the unknown condition {c!r} "
                         f"(known: {sorted(AUTOMATE_SPLIT_OF_CONDITION)})")


def group320(port: str, pitch_deg: float, pos_m: float, grasp: str, split_mode: str = "split320") -> str:
    """Reporting group of a grasp: unseen_shape, interior_strip, extrap_outer (strict_extrap), test_port, select, train_outer, train_band."""
    s = _call_split(port, pitch_deg, pos_m, grasp, split_mode)
    if s == "test":
        return "test_port"
    if s == "heldout":
        if port in OOD_VAL_SHAPES:
            return "unseen_shape"
        return "interior_strip" if is_interior_strip(pitch_deg, pos_m) else "extrap_outer"
    if s == "select":
        return "select"
    return "train_outer" if is_extrap_band(pitch_deg, pos_m) else "train_band"


def group_of_entry(e: dict) -> str:
    """group320 of a grasp.json / entry; AutoMate has no reporting groups yet (they follow its split rule: held-out
    shape, the band of each grasp variable), so it is refused rather than reported as ManiSkill's groups."""
    if is_automate(e):
        raise ValueError(f"AutoMate grasp {e.get('grasp')}: no reporting groups for AutoMate yet (group320 is ManiSkill's)")
    return group320(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"])


# ---------------------------------------------------------------- pictures
def decode_rgb(blob: np.ndarray, off: np.ndarray, i: int) -> np.ndarray:
    """Picture i of a stored JPEG blob -> (320, 320, 3) uint8, decoded as wm/data/store.py does (PIL, RGB)."""
    return np.asarray(Image.open(io.BytesIO(blob[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def encode_depth(d: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(d, np.uint16)).save(buf, format="PNG")
    return np.frombuffer(buf.getvalue(), np.uint8)


def decode_depth(blob: np.ndarray, off: np.ndarray, i: int) -> np.ndarray:
    """Depth picture i -> (320, 320) uint16 in 0.1 mm, exactly as recorded."""
    return np.asarray(Image.open(io.BytesIO(blob[off[i]:off[i + 1]].tobytes()))).astype(np.uint16)


def jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """A rendered picture through the store's own codec (quality 95, no chroma subsampling), as every training picture."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95, subsampling=0)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


# ---------------------------------------------------------------- conversion
def convert(src: str | Path, dst: str | Path, pin_length_m: float, rel: str, allow_test: bool = False) -> dict | None:
    """One trajectory of wm_v1 -> its fast copy; returns the manifest entry, None if it has no pictures.
    Every depth picture is decoded again and compared with the source before the file is written. An AutoMate
    trajectory of condition "test" raises ValueError (allow_test: only for the collection-time data check, whose
    copy is a scratch folder, scripts/automate/check_expert_data.py)."""
    with np.load(src) as z:
        meta = json.loads(str(z["meta"]))
        if "img_step" not in z.files or len(z["img_step"]) == 0:
            return None
        depth, img = z["depth"], z["img_step"]
        keep = {k: z[k] for k in ("rgb_jpeg", "rgb_offsets", "img_step", "qpos", "hand", "action")}
    if not len(depth) == len(img) == len(keep["rgb_offsets"]) - 1:
        raise ValueError(f"{src}: {len(depth)} depth pictures, {len(img)} image steps, {len(keep['rgb_offsets']) - 1} colour pictures")
    blobs = [encode_depth(d) for d in depth]
    blob, off = np.concatenate(blobs), np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    for i, d in enumerate(depth):
        if not np.array_equal(decode_depth(blob, off, i), d):
            raise ValueError(f"{src}: depth picture {i} does not survive the PNG round trip")
    meta.update(pin_length_m=float(pin_length_m), format=FORMAT, source=rel)
    if is_automate(meta):
        if automate_condition(meta) == "test" and not allow_test:
            raise ValueError(f"{src}: AutoMate trajectory of condition 'test' (a final-test assembly): never copied")
        meta = fast_meta(meta)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, meta=np.asarray(json.dumps(meta)), depth_png=blob, depth_offsets=off, **keep)
    os.replace(tmp, dst)
    return entry_of(meta, rel, len(img))


# keys an AutoMate fast-copy meta may hold (no simulator truth: no eval_*, truth, peak, stages, start_error_mm,
# depth_left_mm, placement_failures, tip_centre_m, disassembly_dist_m, and no episode_seed / seed_offset / rng, which
# regenerate the socket pose). expert_completed and automate_success_any are scalar outcomes (module docstring,
# imitation set). No tilt_definition key in fast meta: wmdata320 defaults AutoMate to symmetric_angle using symmetry_n.
FAST_META_AUTOMATE = ("grasp", "port", "condition", "kind", "outcome", "expert_completed", "automate_success_any", "steps",
                      "idx", "H_star", "axis", "pin_length_m", "pitch_deg", "grasp_pos_m", "spin_deg", "u", "simulator",
                      "symmetry_n", "switches", "start", "schema", "rgb_codec", "format", "source", "assembly_id",
                      "config_sha", "plan_sha")
AUTOMATE_ENTRY_KEYS = ("expert_completed", "automate_success_any")    # AutoMate's extra manifest-entry keys


def fast_meta(meta: dict) -> dict:
    """An AutoMate trajectory's meta reduced to FAST_META_AUTOMATE."""
    return {k: meta[k] for k in FAST_META_AUTOMATE if k in meta}


def entry_of(meta: dict, rel: str, n_images: int) -> dict:
    """Manifest entry. "simulator", expert_completed and automate_success_any only when the meta has them (AutoMate
    data): ManiSkill entries are unchanged. An AutoMate meta with a key outside FAST_META_AUTOMATE is refused
    (simulator truth in the fast copy)."""
    am = {}
    if is_automate(meta):
        extra = sorted(set(meta) - set(FAST_META_AUTOMATE))
        if extra:
            raise ValueError(f"{rel}: AutoMate fast-copy meta holds keys outside the allow-list {extra}: delete the file "
                             "and convert it again")
        am = {k: meta[k] for k in AUTOMATE_ENTRY_KEYS if k in meta}
    return {"path": rel, "n_images": int(n_images), "steps": int(meta["steps"]),
            **{k: meta[k] for k in ("port", "grasp", "condition", "kind", "outcome", "pitch_deg", "grasp_pos_m", "pin_length_m")},
            **({"simulator": meta["simulator"]} if "simulator" in meta else {}), **am}


def is_bc(e: dict) -> bool:
    """The policy imitates the trajectory (module docstring, imitation set). An AutoMate entry of a policy kind with
    outcome "inserted" but without expert_completed (a fast copy written before 2026-11-23) raises ValueError."""
    if e["kind"] not in POLICY_KINDS or e.get("outcome") != "inserted":
        return False
    if not is_automate(e):
        return True
    if "expert_completed" not in e:
        raise ValueError(f"{e.get('path')}: AutoMate entry without expert_completed (a fast copy written before the "
                         "imitation rule of 2026-11-23): delete the fast copy and convert it again")
    return bool(e["expert_completed"])


def split_identity(collection: str | Path) -> dict | None:
    """The split an AutoMate collection was made with: its plan.json's split name and file, the file's sha256 and the
    plan_sha. None when the folder has no plan.json (ManiSkill data). SystemExit when plan.json names a split file
    that is missing, or whose content is not the plan's (the file must equal plan.json's split "document" plus its
    plan_sha, as collect.freeze writes it: a hand-edited or stale file is never hashed and carried). Carried by the
    fast copy's manifest ("split"), depth_stats.json, train_files_sha256 and the checkpoint provenance, so that a
    model is never evaluated against another collection's test list."""
    import hashlib
    plan_f = Path(collection) / "plan.json"
    if not plan_f.exists():
        return None
    plan = json.loads(plan_f.read_text())
    sp = plan.get("split") or {}
    out = {"name": sp.get("name"), "file": sp.get("file"), "sha256": None, "plan_sha": plan.get("plan_sha")}
    if sp.get("file"):
        f = Path(collection) / sp["file"]
        if not f.exists():
            raise SystemExit(f"{plan_f} names the split file {sp['file']}, which is missing")
        raw = f.read_bytes()
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            raise SystemExit(f"{f}: not json ({exc})")
        want = {**(sp.get("document") or {}), "plan_sha": plan.get("plan_sha")}
        if not isinstance(doc, dict) or doc != want:
            diff = sorted(k for k in set(want) | set(doc if isinstance(doc, dict) else {})
                          if not isinstance(doc, dict) or doc.get(k) != want.get(k))
            raise SystemExit(f"{f} differs from {plan_f} (split document and plan_sha) in {diff[:6]}: the split file must "
                             "be the one collect.freeze wrote for this plan")
        out["sha256"] = hashlib.sha256(raw).hexdigest()
    return out


def train_files_sha256(entries: list[dict], root: str | Path) -> str:
    """Identifies the training split and its files behind depth statistics (depth_stats.py, train_wm320.py): sha256
    of the manifest's source and, per entry sorted by path, (path, n_images, steps, condition, file size in bytes);
    the condition of an AutoMate entry is automate_condition(e) (a later split rule changes the hash). A manifest
    with a split identity (AutoMate, split_identity) also folds it in; ManiSkill manifests have none, so their hash is
    the one of before. A re-collection or rewrite at the same paths, a changed condition or split file changes it."""
    import hashlib
    man = json.loads((Path(root) / "manifest.json").read_text())
    rows = sorted([e["path"], int(e["n_images"]), int(e["steps"]), automate_condition(e) if is_automate(e) else e.get("condition"),
                   os.stat(Path(root) / e["path"]).st_size] for e in entries)
    doc = {"source": man.get("source"), "entries": rows}
    if man.get("split") is not None:
        doc["split"] = man["split"]
    return hashlib.sha256(json.dumps(doc).encode()).hexdigest()


def check_automate_conditions(grasps: list[dict]) -> None:
    """SystemExit unless every AutoMate grasp (simulator "automate") has a decided, known split (split_of_entry).
    Grasps without a simulator key (ManiSkill) are not looked at."""
    bad = []
    for g in grasps:
        if is_automate(g):
            try:
                split_of_entry(g)
            except SplitUndecided as exc:
                bad.append(str(exc))
    if bad:
        raise SystemExit(f"{len(bad)} AutoMate grasps have no usable split, e.g. {bad[:3]}")


def load_entries(root: str | Path, allow_undecided: bool = False, split_mode: str = "split320") -> list[dict]:
    """The fast copy's trajectories with their split (split_of_entry; split_mode for ManiSkill entries) and `bc`
    (is_bc: the policy imitates it, a successful expert trajectory; module docstring). An AutoMate entry without a
    decided split raises SplitUndecided (a ValueError); allow_undecided (collection-time reports only) gives it the
    split "unassigned" instead."""
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")
    out = []
    for e in json.loads((Path(root) / "manifest.json").read_text())["entries"]:
        out.append({**e, "split": split_of_entry(e, allow_undecided, split_mode), "bc": is_bc(e)})
    return out


# ---------------------------------------------------------------- the policy's samples (as the old policy's data set)
def context_indices(j: int) -> list[int]:
    """Pictures of a policy sample at picture j: [j-6, j-4, j-2, j], the first picture repeated at the start."""
    return [max(0, j - STRIDE_IMG * c) for c in range(CONTEXT - 1, -1, -1)]


def policy_chunk(qpos: np.ndarray, action: np.ndarray, s0: int) -> np.ndarray:
    """The next 16 commanded targets relative to the measured joints at s0 (held at the end)."""
    act = action[s0:s0 + CHUNK]
    if len(act) == 0:
        act = qpos[s0:s0 + 1, :JOINTS]
    if len(act) < CHUNK:
        act = np.concatenate([act, np.repeat(act[-1:], CHUNK - len(act), 0)])
    return (act - qpos[s0][:JOINTS]).reshape(-1).astype(np.float32)


def policy_items(entries: list[dict]) -> np.ndarray:
    """(entry index, picture index) of every picture of every trajectory in `entries`, the terminal picture included."""
    return np.array([(i, j) for i, e in enumerate(entries) for j in range(e["n_images"])], np.int64).reshape(-1, 2)


def policy_sample(path: str | Path, j: int) -> dict:
    """rgb (4, 320, 320, 3) uint8, depth (4, 320, 320) int16 [0.1 mm], q (7,) the joints measured at picture j,
    a (112,) the next 16 joint targets minus q."""
    with np.load(path) as z:
        img, qpos, act = z["img_step"], z["qpos"], z["action"]
        jb, jo, pb, po = z["rgb_jpeg"], z["rgb_offsets"], z["depth_png"], z["depth_offsets"]
    idx = context_indices(j)
    rgb = {i: decode_rgb(jb, jo, i) for i in set(idx)}
    dep = {i: np.minimum(decode_depth(pb, po, i), 32767).astype(np.int16) for i in set(idx)}
    s0 = int(img[j])
    q = qpos[s0, :JOINTS].astype(np.float32)          # the arm's 7 joints only: qpos[7:9] (fingers) is an exact
    assert q.shape == (JOINTS,)                        # function of the true AutoMate grasp, never a model input
    return {"rgb": np.stack([rgb[i] for i in idx]), "depth": np.stack([dep[i] for i in idx]),
            "q": q, "a": policy_chunk(qpos, act, s0)}


def action_stats(root: str | Path, entries: list[dict], n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread (+1e-4) of the action chunk per dimension over n samples drawn without replacement
    (the old policy's rule); reads joints and targets only."""
    items = policy_items(entries)
    pick = items[np.sort(np.random.default_rng(seed).choice(len(items), size=min(n, len(items)), replace=False))]
    a = []
    for i in np.unique(pick[:, 0]):
        with np.load(Path(root) / entries[i]["path"]) as z:
            img, qpos, act = z["img_step"], z["qpos"], z["action"]
        a += [policy_chunk(qpos, act, int(img[j])) for j in pick[pick[:, 0] == i, 1]]
    a = np.stack(a)
    return a.mean(0), a.std(0) + 1e-4
