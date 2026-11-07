"""Parallel collection over grasps; resumable (grasps with grasp.json are skipped).

Each worker keeps one environment and rebuilds it only when the port changes
(grasps are sorted by port and handed out in per-port chunks).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

from .plan import CollectConfig, GraspSpec, build

_STATE = {}


def _init(out: str, cfg: dict):
    _STATE.update(out=Path(out), cfg=CollectConfig(**cfg), port=None, robot=None)


def _work(spec: GraspSpec) -> dict:
    """One grasp. An environment that cannot be built (some generated shapes cannot realise the
    default grasp) marks this grasp as failed instead of stopping the pool."""
    from ..env.robot import Robot
    from .episode import collect_grasp
    try:
        if _STATE["port"] != spec.port:
            if _STATE["robot"] is not None:
                _STATE["robot"].env.close()
            _STATE["robot"], _STATE["port"] = None, None
            _STATE["robot"], _STATE["port"] = Robot(spec.port), spec.port
    except Exception as exc:
        summary = {"grasp": spec.key, "port": spec.port, "condition": spec.condition, "pitch_deg": spec.pitch,
                   "grasp_pos_m": spec.grasp_pos, "seed": spec.seed, "trajectories": [], "bytes": 0,
                   "reference_ok": False, "error": f"environment: {type(exc).__name__}: {exc}", "seconds": 0.0}
        gdir = _STATE["out"] / spec.port / spec.key
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "grasp.json").write_text(json.dumps(summary))
        return summary
    return collect_grasp(_STATE["robot"], spec, _STATE["out"], _STATE["cfg"])


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(cfg: CollectConfig, out: Path, workers: int, only: list[str] | None = None,
        max_grasps: int | None = None, min_free_gb: float = 30.0, max_gb: float = 100.0) -> None:
    """Stops early if the dataset would exceed max_gb or the disk has less than min_free_gb free."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.dump(), indent=1))
    specs = [s for s in build(cfg, only) if not (out / s.port / s.key / "grasp.json").exists()]
    existing = _du(out)
    if max_grasps:
        specs = specs[:max_grasps]
    specs.sort(key=lambda s: s.port)
    chunk = max(1, cfg.train_grasps_per_shape + cfg.band_grasps_per_shape)
    print(f"{len(specs)} grasps to collect, {workers} workers, chunk {chunk}", flush=True)
    t0, total_bytes = time.time(), existing
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(out), cfg.dump()),
                                      maxtasksperchild=60) as pool:
        for k, s in enumerate(pool.imap_unordered(_work, specs, chunksize=chunk)):
            total_bytes += s.get("bytes", 0)
            kinds = " ".join(f"{t['kind'][:3]}:{t.get('outcome', '?')[:8]}" for t in s["trajectories"])
            print(f"[{k + 1}/{len(specs)} {time.time() - t0:.0f}s {total_bytes / 1e9:.2f}GB] {s['grasp']} "
                  f"{s['condition']} ref={s.get('reference_ok')} {kinds} {s.get('error', '')[:80]} {s['seconds']}s",
                  flush=True)
            if shutil.disk_usage(out).free / 1e9 < min_free_gb or total_bytes / 1e9 > max_gb:
                print(f"STOP: dataset {total_bytes / 1e9:.1f} GB (cap {max_gb}) or < {min_free_gb} GB free", flush=True)
                pool.terminate()
                break
