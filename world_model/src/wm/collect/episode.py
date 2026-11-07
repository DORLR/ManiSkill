"""Collect one grasp: reference insertion, then every planned trajectory.

Directory layout:  <out>/<port>/<grasp key>/
    grasp.json                summary (written last; its presence marks the grasp done)
    t00_reference.npz         the one insertion (defines H_star and the axis)
    tNN_<kind>.npz            planned trajectories
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data import schema, store
from ..data.recorder import Recorder
from ..env.expert import Expert
from ..env.robot import Robot
from ..env.sampling import sample_start
from .plan import CollectConfig, GraspSpec

REFERENCE_ATTEMPTS = 3
PLACEMENT_ATTEMPTS = 3


def _axis_from(arrays: dict) -> np.ndarray | None:
    """Insertion direction from the FK hand positions during the axial push (no truth)."""
    push = arrays["phase"] == schema.PHASES.index("axial_push")
    if push.sum() < 2:
        return None
    pos = arrays["hand"][1:][push][:, :3, 3]
    d = pos[-1] - pos[0]
    return d / np.linalg.norm(d)


def collect_grasp(robot: Robot, spec: GraspSpec, out: Path, cfg: CollectConfig) -> dict:
    t0 = time.time()
    gdir = out / spec.port / spec.key
    gdir.mkdir(parents=True, exist_ok=True)
    expert, rec = Expert(robot), Recorder(robot, images=cfg.images, image_every=cfg.image_every)
    base_meta = {"grasp": spec.key, "port": spec.port, "condition": spec.condition,
                 "pitch_deg": spec.pitch, "grasp_pos_m": spec.grasp_pos}
    summary = {**base_meta, "seed": spec.seed, "trajectories": [], "bytes": 0}
    try:
        # 1. reference insertion (retry with a new episode seed, as alignment_field does)
        failures = []
        for attempt in range(REFERENCE_ATTEMPTS):
            seed = spec.seed + 7919 * attempt
            robot.reset(seed, spec.pitch, spec.grasp_pos)
            if not expert.to_safe_from_spawn():
                res = {"completed": False, "failure_reason": "placement_to_safe"}
            else:
                rec.begin()
                res = expert.run("insert", rec.on_step, seed_offset=0)
            if res.get("completed"):
                break
            failures.append(str(res.get("failure_reason"))[:120])
        summary.update(episode_seed=seed, reference_attempts=attempt + 1, reference_ok=bool(res.get("completed")),
                       reference_failure=res.get("failure_reason"), reference_attempt_failures=failures)
        if not res.get("completed"):
            return summary
        arrays = rec.end()
        H_star, axis = robot.hand_pose(), _axis_from(arrays)
        label = {"H_star": H_star.tolist(), "axis": None if axis is None else axis.tolist()}
        summary.update(label)
        meta = {**base_meta, **label, "episode_seed": seed, "kind": "reference", "idx": 0,
                "outcome": "inserted", "steps": int(res["steps"])}
        summary["bytes"] += store.save(gdir / "t00_reference.npz", arrays, meta, cfg.rgb_codec, cfg.jpeg_quality)
        summary["trajectories"].append({"idx": 0, "kind": "reference", "outcome": "inserted", "steps": int(res["steps"])})

        # 2. planned trajectories, each from its own start (placement not recorded)
        for ts in spec.trajectories:
            # a start the planner cannot reach is replaced by a new draw (counted in placement_attempts)
            rng = np.random.default_rng(ts.seed_offset)
            start, placed = ts.start, False
            for k in range(PLACEMENT_ATTEMPTS):
                if k:
                    start = sample_start(rng)
                if expert.move_to_offset(start):
                    placed = True
                    break
            entry = {"idx": ts.idx + 1, "kind": ts.kind, "start": start.as_dict(), "placement_attempts": k + 1}
            if not placed:
                entry["outcome"] = "placement_failed"
                summary["trajectories"].append(entry)
                continue
            rec.begin()
            res = expert.run(ts.kind, rec.on_step, ts.seed_offset, offset=ts.offset, waypoints=ts.waypoints)
            arrays = rec.end()
            if ts.kind in ("insert", "perturbed"):
                outcome = "inserted" if res.get("completed") else "expert_failed"
            elif ts.kind == "contact":
                outcome = res.get("outcome", "expert_failed") if res.get("completed") else "expert_failed"
            else:
                outcome = "done" if res.get("completed") else "expert_failed"
            entry.update(outcome=outcome, steps=int(res.get("steps", 0)), failure=res.get("failure_reason"),
                         depth_left_mm=res.get("depth_left_mm"))
            if ts.offset is not None:
                entry["offset"] = ts.offset.as_dict()
            meta = {**base_meta, **label, "episode_seed": seed, **entry, "seed_offset": ts.seed_offset,
                    "waypoints": None if ts.waypoints is None else [w.as_dict() for w in ts.waypoints]}
            name = f"t{ts.idx + 1:02d}_{ts.kind}.npz"
            summary["bytes"] += store.save(gdir / name, arrays, meta, cfg.rgb_codec, cfg.jpeg_quality)
            summary["trajectories"].append(entry)
    except Exception as exc:  # keep the worker alive; the grasp is marked with the error
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["trace"] = traceback.format_exc()[-2000:]
    finally:
        summary["seconds"] = round(time.time() - t0, 1)
        (gdir / "grasp.json").write_text(json.dumps(summary, default=lambda o: asdict(o)))
    return summary
