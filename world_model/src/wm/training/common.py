"""Shared training utilities."""

from __future__ import annotations

import copy
import json
import os
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
    limit_gpu_memory()


def limit_gpu_memory():
    """Cap this process's GPU memory (env WM_GPU_MEM_FRACTION, e.g. 0.2) on a shared GPU: if the cap
    is reached only this process fails, the other users' jobs are never starved of memory."""
    frac = os.environ.get("WM_GPU_MEM_FRACTION")
    if frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)


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
    """Exponential moving average of parameters (DiT / NWM use decay 0.9999), with warm-up
    decay = min(decay, (1 + n) / (10 + n)) after n updates. Without warm-up, after 5000 steps
    61 % of the average is still the initialisation (measured: diffusion loss 0.46 vs 0.004 raw)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        self.n += 1
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)


def save_checkpoint(path: Path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def trainable(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
