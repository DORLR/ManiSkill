"""Shared training utilities."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def load_config(path: str | Path, **overrides) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def cycle(loader):
    while True:
        yield from loader


class Logger:
    """Prints and appends JSON lines to <run>/log.jsonl."""

    def __init__(self, run_dir: Path):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(run_dir) / "log.jsonl"
        self.t0 = time.time()

    def __call__(self, **rec):
        rec = {"minutes": round((time.time() - self.t0) / 60, 2), **rec}
        line = json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()})
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


class EMA:
    """Exponential moving average of parameters (DiT / NWM use decay 0.9999)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def save_checkpoint(path: Path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def trainable(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
