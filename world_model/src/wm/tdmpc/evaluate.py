"""Score the tagged decisions of a set of branches with either model -> records for report.build()."""

from __future__ import annotations

import torch

from . import data, original


def _records(sc: dict, tags, ks, ret, branch) -> list[dict]:
    """Each score as the mean over the heads, and `<name>-sd` = mean minus one spread over the heads."""
    flat = {}
    for name, v in sc.items():
        flat[name] = v.mean(0).float().cpu().numpy()
        flat[name + "-sd"] = (v.mean(0) - v.std(0)).float().cpu().numpy()
    return [{"branch": int(branch[i]), "tag": tags[i], "k": int(ks[i]), "ret": float(ret[i]),
             "scores": {n: float(x[i]) for n, x in flat.items()}} for i in range(len(tags))]


@torch.no_grad()
def score_wm(model, branches: list[dict], stats: dict, workers: int, files_per_batch: int = 4, device="cuda") -> list[dict]:
    model.eval()
    ds = data.EvalPoints(branches, stats, model.cfg.gamma, hmax=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=files_per_batch, num_workers=workers, collate_fn=data.collate_points)
    out = []
    for b in loader:
        z = model.encode(b["rgb"].to(device).permute(0, 1, 4, 2, 3), b["depth"].to(device), b["q"].to(device), chunk=24)
        sc = model.scores(z, b["action"].to(device), b["valid"].to(device))
        out += _records(sc, b["tags"], b["k"], b["ret"], b["branch"])
    return out


@torch.no_grad()
def score_original(agent, cache: list[dict], stats: dict, batch: int = 2048, device="cuda:0") -> list[dict]:
    if not any(c is not None for c in cache):
        return []
    p = original.points(cache, stats, float(agent.discount), hmax=3)
    out = []
    for i in range(0, len(p["tags"]), batch):
        s = slice(i, i + batch)
        sc = original.scores(agent, p["obs"][s].to(device), p["action"][s].to(device), p["valid"][s].to(device))
        out += _records(sc, p["tags"][s], p["k"][s], p["ret"][s], p["branch"][s])
    return out
