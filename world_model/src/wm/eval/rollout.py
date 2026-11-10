"""Closed-loop rollouts of a policy agent (v3 protocol, fixed before any v3 result: 60 s per episode,
success = within 1 mm of full depth, seeded policy sampling, 4 grasps per shape and condition).

Per grasp: the expert inserts once (as in data collection; grasps it cannot insert are reported
and skipped, the same for every method), then for each far start (same start box as the data,
fresh seeds) the hand is placed by planner motion (truth used only for placement) and the agent
acts every 0.4 s. Success = the plug is within 1 mm of full depth (truth, used only for judging).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..data.splits import OOD_VAL_SHAPES, TEST_PORTS
from ..env.sampling import CONDITIONS, Offset, sample_grasp, sample_start

EVAL_SEED = 9_000_000                 # disjoint from collection seeds (4_000_000 + ...)
MAX_DECISIONS = 150                   # 150 x 0.4 s = 60 s per episode (24 s cut off slow but correct insertions)
INSERTED_MARGIN_M = 0.001             # success: within 1 mm of full depth
POS_EXTRAP_USED = (-0.030, -0.028)    # the [-12, -10] mm side is not insertable in this environment


def sample_eval_grasp(rng: np.random.Generator, condition: str) -> tuple[float, float]:
    pitch, pos = sample_grasp(rng, condition)
    if CONDITIONS[condition][1] == "extrap":
        pos = float(rng.uniform(*POS_EXTRAP_USED))
    return pitch, pos


@dataclass
class EvalGrasp:
    key: str
    port: str
    condition: str
    pitch: float
    grasp_pos: float
    seed: int
    starts: list[Offset] = field(default_factory=list)


SEED_STRIDE = 100_000      # evaluation seed k shifts every grasp / start / episode seed by k x this


def plan(split: str, grasps_per_cell: int, starts_per_grasp: int, seed: int = 0) -> list[EvalGrasp]:
    """Evaluation episodes. `seed` 0 is the original plan; other seeds draw new grasps, starts and
    episode seeds (policy sampling), the same for every method evaluated with that seed."""
    ports = list(OOD_VAL_SHAPES) if split == "ood_val" else list(TEST_PORTS)
    base = EVAL_SEED + SEED_STRIDE * seed
    out = []
    for si, port in enumerate(ports):
        rng = np.random.default_rng(base + si)
        for ci, cond in enumerate(CONDITIONS):
            for g in range(grasps_per_cell):
                pitch, pos = sample_eval_grasp(rng, cond)
                out.append(EvalGrasp(f"{port}_{cond}_{g}", port, cond, pitch, pos,
                                     base + 1000 * si + 10 * ci + g,
                                     [sample_start(rng) for _ in range(starts_per_grasp)]))
    return out


def _relative(robot):
    from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted import _copy_pose
    return _copy_pose(robot.base.goal_pose).inv() * _copy_pose(robot.base.charger.pose)


def _depth_left_m(robot) -> float:
    """Distance (m) of the plug from full insertion (simulator truth; judging only)."""
    return -float(_relative(robot).p[0])


def _diagnosis(robot) -> dict:
    """Plug offset from the hole axis at the end of an episode (simulator truth; analysis only)."""
    rel = _relative(robot)
    w = float(np.clip(abs(np.asarray(rel.q)[0]), -1, 1))
    return {"depth_left_mm": -float(rel.p[0]) * 1000, "lateral_mm": float(np.hypot(*np.asarray(rel.p)[1:])) * 1000,
            "angle_deg": float(np.degrees(2 * np.arccos(w)))}


def episode(robot, expert, agent, start: Offset, seed: int, max_decisions: int = MAX_DECISIONS,
            on_step=None) -> dict:
    """on_step(robot): optional hook after every control step (used to record videos).
    The policy's sampling is seeded with the episode seed, so an episode can be replayed exactly."""
    import torch

    from .agent import EXECUTE
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if not expert.move_to_offset(start):
        return {"outcome": "placement_failed"}
    f = robot.frame()
    if not agent.reset(f, seed):
        return {"outcome": "no_board_plane"}
    for k in range(max_decisions):
        for q in agent.act(f)[0][:EXECUTE]:
            robot._command(q, 1)
            if on_step is not None:
                on_step(robot)
        if _depth_left_m(robot) <= INSERTED_MARGIN_M:
            return {"outcome": "inserted", "decisions": k + 1}
        f = robot.frame()
    return {"outcome": "timeout", "decisions": max_decisions, **_diagnosis(robot)}


# ---------------------------------------------------------------- parallel runner
_STATE = {}


def _init(vae_ckpt: str, policy_ckpt: str, max_decisions: int = MAX_DECISIONS, residual_ckpt: str | None = None):
    """Never raise here: an exception in a Pool initializer makes the pool respawn workers forever."""
    _STATE.update(port=None, robot=None, agent=None, init_error=None, max_decisions=max_decisions)
    try:
        import torch
        from ..training.common import limit_gpu_memory
        limit_gpu_memory()                      # WM_GPU_MEM_FRACTION, per worker
        kind = torch.load(policy_ckpt, map_location="cpu", weights_only=False)["cfg"].get("input", "latent")
        if kind == "image":
            from .agent import ImagePolicyAgent
            _STATE["agent"] = ImagePolicyAgent(policy_ckpt)
        elif residual_ckpt:
            from .agent import ResidualAgent
            _STATE["agent"] = ResidualAgent(vae_ckpt, policy_ckpt, residual_ckpt)
        else:
            from .agent import LatentPolicyAgent
            _STATE["agent"] = LatentPolicyAgent(vae_ckpt, policy_ckpt)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def _work(g: EvalGrasp) -> dict:
    from ..env.expert import Expert
    from ..env.robot import Robot
    t0 = time.time()
    res = {**{k: v for k, v in asdict(g).items() if k != "starts"}, "episodes": []}
    if _STATE["init_error"]:
        return {**res, "error": "agent init: " + _STATE["init_error"], "seconds": 0.0}
    try:
        if _STATE["port"] != g.port:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = Robot(g.port), g.port
        robot = _STATE["robot"]
        expert = Expert(robot)
        robot.reset(g.seed, g.pitch, g.grasp_pos)
        ok = expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")
        res["expert_ok"] = bool(ok)
        if ok:
            for i, s in enumerate(g.starts):
                ep = episode(robot, expert, _STATE["agent"], s, g.seed + i, _STATE["max_decisions"])
                res["episodes"].append({"start": s.as_dict(), **ep})
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()[-1500:]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run(vae_ckpt: str, policy_ckpt: str, split: str, out: Path, workers: int,
        grasps_per_cell: int, starts_per_grasp: int, only: list[str] | None = None,
        max_decisions: int = MAX_DECISIONS, residual_ckpt: str | None = None, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    res_path = out / "results.jsonl"
    if res_path.exists():
        done = {json.loads(l)["key"] for l in res_path.read_text().splitlines() if l.strip()}
    grasps = [g for g in plan(split, grasps_per_cell, starts_per_grasp, seed)
              if g.key not in done and (not only or g.port in only)]
    grasps.sort(key=lambda g: g.port)
    print(f"{len(grasps)} grasps x {starts_per_grasp} starts, {workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(vae_ckpt, policy_ckpt, max_decisions, residual_ckpt)) as pool, \
            open(res_path, "a") as f:
        for k, r in enumerate(pool.imap_unordered(_work, grasps)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            outs = [e["outcome"][:8] for e in r["episodes"]]
            print(f"[{k + 1}/{len(grasps)} {time.time() - t0:.0f}s] {r['key']} expert={r.get('expert_ok')} "
                  f"{outs} {r.get('error', '')[:80]}", flush=True)
