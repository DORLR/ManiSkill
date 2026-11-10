"""Online learners (v4): the world model and the success classifier keep training while the actors
(online/actor.py) add episodes.

MixedBatches: every batch comes from the online episodes with probability `online_fraction` (once at
least `min_online_files` exist), else from the offline data. The online file list is re-scanned every
`refresh_every` steps; new files are added to the online dataset and its loader is rebuilt.
The learners write `current.pt` (EMA weights) every `save_every` steps; the actors reload it.

The classifier learner also trains on imagined latents: world-model predictions 0.4 s ahead from
real online histories, placed as the newest seen slot, labelled with the outcome of the real frame
they predict -- so it judges imagined frames as it judges real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.history import HIST, HistoryWindowDataset, LatentFiles, SuccessDataset
from ..training import common


class MixedBatches:
    def __init__(self, offline_ds, make_online_ds, online_dir: Path, cfg: dict, loader_fn):
        self.cfg, self.online_dir = cfg, Path(online_dir)
        self.offline = common.cycle(loader_fn(offline_ds, cfg["workers"]))
        self.make_online_ds, self.loader_fn = make_online_ds, loader_fn
        self.files, self.online, self.n_online = None, None, 0
        self.rng = np.random.default_rng(cfg["seed"] + 17)
        self.step = 0
        self.refresh()

    def refresh(self):
        paths = sorted(str(p) for p in (self.online_dir / "latents").rglob("*.npz") if not p.name.endswith(".tmp.npz"))
        if len(paths) == self.n_online:
            return
        self.files = LatentFiles(paths)
        self.n_online = len(paths)
        if self.n_online >= self.cfg["min_online_files"]:
            self.online = common.cycle(self.loader_fn(self.make_online_ds(self.files), self.cfg["online_workers"]))

    def __next__(self):
        self.step += 1
        if self.step % self.cfg["refresh_every"] == 0:
            self.refresh()
        if self.online is not None and self.rng.random() < self.cfg["online_fraction"]:
            return next(self.online)
        return next(self.offline)

    def __iter__(self):
        return self


def plain_loader(cfg):
    def make(ds, workers):
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=workers,
                          persistent_workers=workers > 0, pin_memory=True)
    return make


def run_wm(cfg: dict, online: dict, run_dir: Path):
    from ..training import wm4
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    offline = HistoryWindowDataset(wm4.offline_files(cfg, "train"), scale, cfg["mask"], p_step=cfg["p_step"])
    mk_online = lambda files: HistoryWindowDataset(files, scale, cfg["mask"], p_step=cfg["p_step"])
    batches = MixedBatches(offline, mk_online, online["dir"], {**cfg, **online}, plain_loader(cfg))
    c = {**cfg, "steps": online["wm_steps"], "patience": None, "save_every": online["save_every"]}
    return wm4.train(c, run_dir, batches, resume=online["wm_resume"])


def run_success(cfg: dict, online: dict, run_dir: Path):
    from ..data.history import history, to_tensors
    from ..models.history_cdit import create_diffusion, grid_to_tokens
    from ..training import success, wm4
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    offline = SuccessDataset(wm4.offline_files(cfg, "train"), scale, cfg["mask"])
    mk_online = lambda files: SuccessDataset(files, scale, cfg["mask"])
    loader = lambda ds, workers: success.balanced_loader(ds, {**cfg, "workers": workers})
    batches = MixedBatches(offline, mk_online, online["dir"], {**cfg, **online}, loader)
    dev = torch.device("cuda")
    sampler = create_diffusion(timestep_respacing=str(online["sampling_steps"]))
    state = {"wm": None, "mtime": None, "k": 0}

    def imagined_batch(_model):
        """World-model predictions from real online histories as the newest slot, with the real label."""
        state["k"] += 1
        if state["k"] % online["imagined_every"] or batches.files is None or batches.n_online < online["min_online_files"]:
            return None
        live = Path(online["wm_live"])
        if live.exists() and live.stat().st_mtime != state["mtime"]:
            state["wm"] = wm4.load_history_wm(str(live), dev).requires_grad_(False)
            state["mtime"] = live.stat().st_mtime
        if state["wm"] is None:
            return None
        rng = np.random.default_rng(state["k"])
        hs, xs = [], []
        files = batches.files
        while len(hs) < online["imagined_batch"]:
            fi = int(rng.integers(len(files)))
            d = files.load(fi)
            n = len(d["img_step"])
            if n < 3:
                continue
            j = int(rng.integers(0, n - 2))                  # predict image j + 2 (0.4 s ahead)
            R_P = d["R_P"].astype(np.float64)
            h_now = history(d, j, R_P, scale)
            h_tgt = history(d, j + 2, R_P, scale)            # the target's history (proprio, avail) ...
            steps = d["img_step"].astype(int)
            from ..data.conditions import N_WAYPOINTS, pose_feature
            ref = d["hand"][steps[j]].astype(np.float64)
            wp = np.rint(np.linspace(steps[j], steps[j + 2], N_WAYPOINTS + 1)[1:]).astype(int)
            path = pose_feature(d["hand_cmd"][np.clip(wp, 0, len(d["hand_cmd"]) - 1)].astype(np.float64), ref, R_P).reshape(-1)
            hs.append((h_now, h_tgt, path, (steps[j + 2] - steps[j]) / 20.0, bool(d["inserted"][j + 2])))
        T = lambda k: torch.stack([torch.from_numpy(np.ascontiguousarray(h[0][k])) for h in hs]).to(dev)
        with torch.no_grad():
            kw = dict(path=torch.tensor(np.stack([h[2] for h in hs]), dtype=torch.float32, device=dev),
                      rel_t=torch.tensor([h[3] for h in hs], dtype=torch.float32, device=dev),
                      z_hist=T("z"), seen=T("seen"), proprio=T("proprio"), dt=T("dt"), avail=T("avail"))
            C = kw["z_hist"].shape[-1]
            x = sampler.p_sample_loop(state["wm"].forward, (len(hs), C, 8, 8), torch.randn(len(hs), C, 8, 8, device=dev),
                                      clip_denoised=False, model_kwargs=kw, progress=False, device=dev)
            z_pred = grid_to_tokens(x).cpu()
        out = []
        for i, (h_now, h_tgt, _, _, lab) in enumerate(hs):
            h = {k: np.array(v, copy=True) for k, v in h_tgt.items()}
            h["seen"][-1] = True
            h["seen"][-2] = False                            # the 0.2 s slot in between is not imagined
            h["z"][-2] = 0
            h["z"][-1] = z_pred[i].numpy()
            t = to_tensors(h)
            t["label"] = torch.tensor(float(lab))
            out.append(t)
        return {k: torch.stack([o[k] for o in out]) for k in out[0]}

    c = {**cfg, "steps": online["success_steps"], "patience": None, "save_every": online["save_every"]}
    return success.train(c, run_dir, batches, resume=online["success_resume"], extra=imagined_batch)
