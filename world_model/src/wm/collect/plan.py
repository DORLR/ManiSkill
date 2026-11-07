"""Collection plan: which shapes, grasps and trajectories.

Everything is collected in one pass; training splits are made later from the
recorded metadata (shape, grasp condition), never at collection time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..env.sampling import BAND_CONDITIONS, CONDITIONS, Offset, StartBox, sample_contact_offset, sample_grasp, sample_start

TEST_PORTS = ("A", "B", "C", "D")
UNBUILDABLE = {"G19", "G42", "G45", "G49"}  # the environment cannot grasp these at the requested pitch


@dataclass
class TrajectorySpec:
    idx: int
    kind: str
    seed_offset: int
    start: Offset
    offset: Offset | None = None
    waypoints: list[Offset] | None = None


@dataclass
class GraspSpec:
    key: str
    port: str
    condition: str
    pitch: float
    grasp_pos: float
    seed: int
    trajectories: list[TrajectorySpec] = field(default_factory=list)


@dataclass
class CollectConfig:
    name: str = "wm_v1"
    seed: int = 4_000_000
    generated_shapes: int = 150
    train_grasps_per_shape: int = 4         # grasps inside the training bands
    band_grasps_per_shape: int = 4          # grasps in interp/extrap bands, cycling over conditions
    grasps_per_test_condition: int = 3      # A-D: this many per condition
    trajectories: dict = field(default_factory=lambda: {"insert": 4, "perturbed": 2, "contact": 5, "random": 1})
    random_waypoints: int = 3
    images: bool = True
    image_every: int = 4
    rgb_codec: str = "jpeg"
    jpeg_quality: int = 95

    @classmethod
    def load(cls, path: str | Path) -> "CollectConfig":
        return cls(**json.loads(Path(path).read_text()))

    def dump(self) -> dict:
        return asdict(self)


def shapes(cfg: CollectConfig) -> list[str]:
    gen = [f"G{i:02d}" for i in range(cfg.generated_shapes) if f"G{i:02d}" not in UNBUILDABLE]
    return list(TEST_PORTS) + gen


def _trajectories(rng: np.random.Generator, cfg: CollectConfig) -> list[TrajectorySpec]:
    out = []
    for kind, n in cfg.trajectories.items():
        for _ in range(n):
            idx = len(out)
            spec = TrajectorySpec(idx=idx, kind=kind, seed_offset=int(rng.integers(1, 2**31)),
                                  start=sample_start(rng))
            if kind == "contact":
                spec.offset = sample_contact_offset(rng)
            if kind == "random":
                spec.waypoints = [sample_start(rng, StartBox(tip_gap_m=(0.004, 0.018)))
                                  for _ in range(cfg.random_waypoints)]
            out.append(spec)
    return out


def build(cfg: CollectConfig, only: list[str] | None = None) -> list[GraspSpec]:
    specs = []
    for si, port in enumerate(shapes(cfg)):
        if only and port not in only:
            continue
        rng = np.random.default_rng(cfg.seed + si)
        if port in TEST_PORTS:
            conds = [c for c in CONDITIONS for _ in range(cfg.grasps_per_test_condition)]
        else:
            band = [BAND_CONDITIONS[(si * cfg.band_grasps_per_shape + k) % len(BAND_CONDITIONS)]
                    for k in range(cfg.band_grasps_per_shape)]
            conds = ["train"] * cfg.train_grasps_per_shape + band
        for gi, cond in enumerate(conds):
            pitch, pos = sample_grasp(rng, cond)
            specs.append(GraspSpec(key=f"{port}_g{gi:02d}", port=port, condition=cond, pitch=pitch,
                                   grasp_pos=pos, seed=cfg.seed + 1000 * si + gi,
                                   trajectories=_trajectories(rng, cfg)))
    return specs
