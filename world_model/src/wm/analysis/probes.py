"""What does the latent keep? Offline probes on the contact trials (the same tests as for v1 / v2, so
the numbers are comparable; docs/2026-11-07_v1.md, docs/2026-11-08_v2.md).

Contact trials: the plug is moved to a small offset (0-3 mm, 0-3 deg) 3 mm in front of the hole,
then pushed straight; outcome inserted or jammed. Probes are fitted on train and scored on ood_val
(held-out shapes). The offset (simulator truth) is used only as a probe target / upper bound.

jam_probes    will this push jam? (balanced accuracy, AUC) from
                  latent  4 context latents + the next 8 joint targets
                  joints  measured joints + the next 8 joint targets (no vision)
                  truth   the sampled offset: the reachable upper bound
              at push start, 0.4 s and 0.8 s into the push; logistic regression and a small MLP
offset_probes how well can the offset (lateral y/z mm, roll/pitch/yaw deg) be regressed from the
              latent (linear, MLP), from the pose head's reading (linear map), from joints only
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.datasets import CONTEXT, WM_STRIDE, context_indices, port_axes_ok

PUSH_PHASE = 7          # schema.PHASES index of the straight push of a contact trial
OFFSET_NAMES = ["lateral_y_mm", "lateral_z_mm", "roll_deg", "pitch_deg", "yaw_deg"]


def contact_trials(data_root: str, latent_root: str, split: str) -> list[dict]:
    """Contact trials of a split with their latent file loaded, the push-start image index j0, the
    outcome y (1 = jammed) and the sampled offset (truth, probe target only)."""
    idx = json.loads((Path(data_root) / "index.json").read_text())
    entries = idx["entries"] if isinstance(idx, dict) else idx
    out = []
    for e in entries:
        if e["split"] != split or e["kind"] != "contact" or e.get("outcome") not in ("inserted", "jammed"):
            continue
        if not port_axes_ok(e):
            continue
        lp = Path(latent_root) / Path(e["path"]).relative_to(data_root)
        if not lp.exists():
            continue
        with np.load(e["path"]) as f:
            phase = f["phase"]
            meta = json.loads(str(f["meta"]))
        push = np.flatnonzero(phase == PUSH_PHASE)
        if len(push) == 0:
            continue
        with np.load(lp) as g:
            d = {k: g[k] for k in g.files}
        j0 = int(np.searchsorted(d["img_step"], push[0], side="right") - 1)
        off = meta["offset"]
        out.append({"lat": d, "j0": j0, "push0": int(push[0]), "y": int(e["outcome"] == "jammed"),
                    "offset": np.array([*off["yz_mm"], *off["rpy_deg"]], np.float32),
                    "R_P": np.asarray(e["R_P"], np.float64), "port": e["port"], "condition": e["condition"]})
    return out


def _jam_features(t: dict, j: int, scale: float) -> dict | None:
    d = t["lat"]
    if j >= len(d["img_step"]):
        return None
    s0 = int(d["img_step"][j])
    act = d["action"][s0:s0 + 8]
    if len(act) < 8:
        act = np.concatenate([act, np.repeat(act[-1:], 8 - len(act), 0)]) if len(act) else np.zeros((8, 7))
    a = (act - d["qpos"][s0][:7]).reshape(-1) * 1000
    off = t["offset"]
    truth = np.array([off[0], off[1], np.hypot(off[0], off[1]), *off[2:], np.linalg.norm(off[3:])], np.float32)
    return {"latent": np.concatenate([(d["mean"][context_indices(j)] * scale).reshape(-1), a]),
            "joints": np.concatenate([d["qpos"][s0][:7], a]), "truth": truth}


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))


def _classifier(xtr, ytr, xte, yte, kind: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    xtr, xte = np.asarray(xtr, np.float32), np.asarray(xte, np.float32)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a, b = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = nn.Linear(a.shape[1], 1) if kind == "logistic" else nn.Sequential(
        nn.Linear(a.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    pos = ytr.mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2 if kind == "logistic" else 1e-3)
    for _ in range(1500):
        i = torch.randint(0, len(a), (min(512, len(a)),))
        loss = loss_fn(model(a[i])[:, 0], yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(b)[:, 0]).numpy()
    pred = p > 0.5
    tpr = (pred & (yte == 1)).sum() / max((yte == 1).sum(), 1)
    tnr = (~pred & (yte == 0)).sum() / max((yte == 0).sum(), 1)
    return {"balanced_acc": round(float((tpr + tnr) / 2), 3), "auc": round(_auc(p, yte), 3),
            "jam_recall": round(float(tpr), 3), "insert_recall": round(float(tnr), 3)}


def jam_probes(tr: list[dict], te: list[dict], scale: float) -> dict:
    out = {}
    for name, dj in (("push_start", 0), ("push+0.4s", WM_STRIDE), ("push+0.8s", 2 * WM_STRIDE)):
        ftr = [(f, t["y"]) for t in tr if (f := _jam_features(t, t["j0"] + dj, scale)) is not None]
        fte = [(f, t["y"]) for t in te if (f := _jam_features(t, t["j0"] + dj, scale)) is not None]
        ytr, yte = np.array([y for _, y in ftr]), np.array([y for _, y in fte])
        for inp in ("latent", "joints", "truth"):
            xtr, xte = np.stack([f[inp] for f, _ in ftr]), np.stack([f[inp] for f, _ in fte])
            for kind in ("logistic", "mlp"):
                out[f"{name}/{inp}/{kind}"] = _classifier(xtr, ytr, xte, yte, kind)
    return out


def _regressor(xtr, ytr, xte, kind: str) -> np.ndarray:
    xtr, ytr, xte = (np.asarray(v, np.float32) for v in (xtr, ytr, xte))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-6
    a, b = torch.tensor((xtr - mu) / sd), torch.tensor((ytr - ym) / ys)
    torch.manual_seed(0)
    m = nn.Linear(a.shape[1], b.shape[1]) if kind == "linear" else nn.Sequential(
        nn.Linear(a.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, b.shape[1]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(2000):
        i = torch.randint(0, len(a), (min(512, len(a)),))
        loss = (m(a[i]) - b[i]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor((xte - mu) / sd)).numpy() * ys + ym


def _r2(pred, y) -> dict:
    out = {}
    for k, n in enumerate(OFFSET_NAMES):
        r2 = 1 - ((pred[:, k] - y[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum()
        out[n] = {"r2": round(float(r2), 3), "median_abs_err": round(float(np.median(np.abs(pred[:, k] - y[:, k]))), 3)}
    return out


def _rotvec_deg(R: torch.Tensor) -> torch.Tensor:
    """(B,3,3) -> axis * angle in degrees (small angles)."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    return torch.rad2deg(v * (ang / (2 * torch.sin(ang).clamp_min(1e-6)))[:, None])


def frame_geometry(trials: list[dict], j_of, dev) -> dict:
    """Batched geometry (K, camera pose, hand pose, port axes) of image j_of(t) of each trial."""
    st = lambda f: torch.tensor(np.stack([f(t) for t in trials]), dtype=torch.float32, device=dev)
    return {"K": st(lambda t: t["lat"]["K"]), "cam_T_world": st(lambda t: t["lat"]["cam_T_world"][j_of(t)]),
            "hand": st(lambda t: t["lat"]["hand"][int(t["lat"]["img_step"][j_of(t)])]), "R_P": st(lambda t: t["R_P"])}


@torch.no_grad()
def _pose_reading(ts, vae, scale, dev) -> np.ndarray:
    """Plug pose solved from the push-start latent (rotation vector deg, translation mm)."""
    out = []
    for b in range(0, len(ts), 32):          # the per-pixel pose solve needs ~100 MB per frame
        ch = ts[b:b + 32]
        z = torch.tensor(np.stack([t["lat"]["mean"][t["j0"]] for t in ch]), dtype=torch.float32, device=dev)
        R, t = vae.decode_pose(z, frame_geometry(ch, lambda t: t["j0"], dev))
        out.append(torch.cat([_rotvec_deg(R.float()), t.float() * 1000], -1).cpu().numpy())
    return np.concatenate(out)


def offset_probes(tr: list[dict], te: list[dict], scale: float, vae=None, dev=None) -> dict:
    ytr, yte = np.stack([t["offset"] for t in tr]), np.stack([t["offset"] for t in te])
    ctx = lambda t: (t["lat"]["mean"][context_indices(t["j0"])] * scale).reshape(-1)
    xtr, xte = np.stack([ctx(t) for t in tr]), np.stack([ctx(t) for t in te])
    rep = {"truth_spread_median_abs": {n: round(float(np.median(np.abs(yte[:, k]))), 3) for k, n in enumerate(OFFSET_NAMES)}}
    for kind in ("linear", "mlp"):
        rep[f"latent/{kind}"] = _r2(_regressor(xtr, ytr, xte, kind), yte)
    q = lambda t: t["lat"]["qpos"][int(t["lat"]["img_step"][t["j0"]])][:7]
    rep["joints/mlp"] = _r2(_regressor(np.stack([q(t) for t in tr]), ytr, np.stack([q(t) for t in te]), "mlp"), yte)
    if vae is not None:
        rep["pose_head/linear"] = _r2(_regressor(_pose_reading(tr, vae, scale, dev), ytr,
                                                 _pose_reading(te, vae, scale, dev), "linear"), yte)
    return rep


def pose_by_condition(te: list[dict], vae, scale, dev) -> dict:
    """Correlation of the solved pose's components with the true offset, per grasp condition (held-out
    shapes). The v3 regressor failed exactly here: pitch under grasp-pitch extrapolation."""
    pred, off = _pose_reading(te, vae, scale, dev), np.stack([t["offset"] for t in te])
    cond = np.array([t["condition"] for t in te])
    pairs = {"roll": (0, 2), "pitch": (1, 3), "yaw": (2, 4), "lateral_y": (4, 0), "lateral_z": (5, 1)}
    out = {}
    for c in sorted(set(cond)):
        m = cond == c
        out[c] = {"trials": int(m.sum()), **{n: round(float(abs(np.corrcoef(pred[m, i], off[m, k])[0, 1])), 3)
                                              for n, (i, k) in pairs.items()},
                  "pitch_reading_std_deg": round(float(pred[m, 1].std()), 3)}
    return out


def angle_gate(offset_report: dict) -> dict:
    """Mean R^2 of roll / pitch / yaw from the latent (MLP) and from the pose head (linear)."""
    ang = ("roll_deg", "pitch_deg", "yaw_deg")
    lat = float(np.mean([offset_report["latent/mlp"][a]["r2"] for a in ang]))
    has = "pose_head/linear" in offset_report
    head = float(np.mean([offset_report["pose_head/linear"][a]["r2"] for a in ang])) if has else float("nan")
    pitch = float(offset_report["pose_head/linear"]["pitch_deg"]["r2"]) if has else float("nan")
    return {"latent_angle_r2": round(lat, 3), "pose_head_angle_r2": round(head, 3), "pose_head_pitch_r2": round(pitch, 3)}


assert CONTEXT == 4
