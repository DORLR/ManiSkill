"""Is "jam vs slide in" in the latent, and does the world model use it?  (offline check, no training of
the VAE / world model)

Contact trials: the plug is moved to a small offset (0-3 mm, 0-3 deg) at a 3 mm gap, then pushed
straight. Outcome per trial: inserted or jammed. Train split -> fit, ood_val (held-out shapes) -> report.

Part 1, probes (classifier: will this push jam?), at three moments: push start (3 mm before the
hole, before contact), 0.4 s and 0.8 s into the push. Inputs compared:
    latent      4 context latents (as the policy / world model see them) + the next 8 joint targets
    joints      measured joints + the next 8 joint targets (no vision)
    truth       sampled offset (lateral mm, angles deg): simulator truth, the reachable upper bound
Each with a logistic regression and a small MLP. Metric: balanced accuracy and AUC on ood_val.

Part 2, world model: for every 0.4 s window in the push phase of ood_val contact trials, compare
the change in "distance to inserted" (read by the VAE pose head) predicted by the world model with
the change read from the real next latent, split by trial outcome. If the world model understood
jamming, its predicted progress would be ~0 on jammed trials, like the real one.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/probe_contact.py --out $Q/runs/probe_contact_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wm.data.datasets import CONTEXT, WM_STRIDE, port_axes_ok

PUSH_PHASE = 7
Q = "/mnt/storage2/qingxin"


def trials(data_root: str, latent_root: str, split: str) -> list[dict]:
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
                    "truth": np.array([*off["yz_mm"], np.hypot(*off["yz_mm"]), *off["rpy_deg"],
                                       np.linalg.norm(off["rpy_deg"][1:])], np.float32),
                    "port": e["port"]})
    return out


def features(t: dict, j: int, scale: float) -> dict | None:
    d = t["lat"]
    if j >= len(d["img_step"]):
        return None
    ctx = [max(0, j - WM_STRIDE * c) for c in range(CONTEXT - 1, -1, -1)]
    s0 = int(d["img_step"][j])
    act = d["action"][s0:s0 + 8]
    if len(act) < 8:
        act = np.concatenate([act, np.repeat(act[-1:], 8 - len(act), 0)]) if len(act) else np.zeros((8, 7))
    a = (act - d["qpos"][s0][:7]).reshape(-1) * 1000
    return {"latent": np.concatenate([(d["mean"][ctx] * scale).reshape(-1), a]),
            "joints": np.concatenate([d["qpos"][s0][:7], a]), "truth": t["truth"]}


def fit_eval(xtr, ytr, xte, yte, kind: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = torch.tensor((xtr - mu) / sd), torch.tensor((xte - mu) / sd)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    dim = xtr.shape[1]
    model = nn.Linear(dim, 1) if kind == "logistic" else nn.Sequential(
        nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    pos = ytr.mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6)))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2 if kind == "logistic" else 1e-3)
    for _ in range(1500):
        idx = torch.randint(0, len(xtr), (min(512, len(xtr)),))
        loss = loss_fn(model(xtr[idx])[:, 0], ytr_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xte)[:, 0]).numpy()
    pred = p > 0.5
    tpr = (pred & (yte == 1)).sum() / max((yte == 1).sum(), 1)
    tnr = (~pred & (yte == 0)).sum() / max((yte == 0).sum(), 1)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = (yte == 1).sum(), (yte == 0).sum()
    auc = (ranks[yte == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)
    return {"balanced_acc": round(float((tpr + tnr) / 2), 3), "auc": round(float(auc), 3),
            "jam_recall": round(float(tpr), 3), "insert_recall": round(float(tnr), 3)}


@torch.no_grad()
def world_model_part(te: list[dict], vae_ckpt: str, wm_ckpt: str, scale: float, max_windows: int) -> dict:
    from wm.geometry import apply_rigid
    from wm.models.cdit import TokenCDiT, create_diffusion, grid_to_tokens, tokens_to_grid
    from wm.data.datasets import action_condition
    from wm.models.vae import load_vae
    dev = torch.device("cuda")
    vae = load_vae(vae_ckpt, dev)
    ck = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
    wm = TokenCDiT(action_dim=ck["cfg"]["action_dim"], latent_ch=vae.latent_ch, context_size=CONTEXT, size=ck["cfg"]["size"])
    wm.load_state_dict(ck["ema"])
    wm.to(dev).eval()
    sampler = create_diffusion(timestep_respacing="20")
    history = ck["cfg"].get("history", False)
    from wm.geometry import plug_region_points
    ref = plug_region_points(dev)

    def remaining(z):
        d = vae.decode(z / scale)
        p = ref[None].expand(z.shape[0], -1, -1)
        return (apply_rigid(p, d["R"].float(), d["t"].float()) - p).norm(dim=-1).mean(1) * 1000

    rows = {0: [], 1: []}
    windows = []
    for t in te:
        d = t["lat"]
        n = len(d["img_step"])
        for j in range(max(t["j0"], (CONTEXT - 1) * WM_STRIDE), n - WM_STRIDE):
            windows.append((t, j))
    rng = np.random.default_rng(0)
    if len(windows) > max_windows:
        windows = [windows[i] for i in rng.choice(len(windows), max_windows, replace=False)]
    for b in range(0, len(windows), 64):
        chunk = windows[b:b + 64]
        ctx, nxt, act = [], [], []
        for t, j in chunk:
            d = t["lat"]
            c = [j - WM_STRIDE * k for k in range(CONTEXT - 1, -1, -1)]
            ctx.append(d["mean"][c] * scale)
            nxt.append(d["mean"][j + WM_STRIDE] * scale)
            act.append(action_condition(d["action"], d["qpos"], int(d["img_step"][j]), history))
        ctx = torch.tensor(np.stack(ctx), dtype=torch.float32, device=dev)
        nxt = torch.tensor(np.stack(nxt), dtype=torch.float32, device=dev)
        act = torch.tensor(np.stack(act), dtype=torch.float32, device=dev)
        B = len(chunk)
        kw = dict(y=act, x_cond=tokens_to_grid(ctx), rel_t=torch.ones(B, device=dev))
        x = sampler.p_sample_loop(wm.forward, (B, vae.latent_ch, 8, 8), torch.randn(B, vae.latent_ch, 8, 8, device=dev),
                                  clip_denoised=False, model_kwargs=kw, progress=False, device=dev)
        pred = grid_to_tokens(x)
        now, real, wmp = remaining(ctx[:, -1]), remaining(nxt), remaining(pred)
        for k, (t, j) in enumerate(chunk):
            rows[t["y"]].append((float(now[k] - real[k]), float(now[k] - wmp[k])))
    out = {}
    for y, name in ((0, "inserted_trials"), (1, "jammed_trials")):
        r = np.array(rows[y])
        if len(r) == 0:
            continue
        out[name] = {"windows": len(r), "real_progress_mm_median": round(float(np.median(r[:, 0])), 3),
                     "wm_progress_mm_median": round(float(np.median(r[:, 1])), 3),
                     "abs_error_mm_median": round(float(np.median(np.abs(r[:, 0] - r[:, 1]))), 3),
                     "real_stalled_frac(<0.2mm)": round(float((r[:, 0] < 0.2).mean()), 3),
                     "wm_stalled_frac(<0.2mm)": round(float((r[:, 1] < 0.2).mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{Q}/data/wm_v1")
    ap.add_argument("--latents", default=f"{Q}/data/wm_v1_latents")
    ap.add_argument("--vae", default=f"{Q}/runs/vae_v1/best.pt")
    ap.add_argument("--wm", default=f"{Q}/runs/wm_v1/last.pt")
    ap.add_argument("--skip-probes", action="store_true", help="only the world-model part")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wm-windows", type=int, default=3000)
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    tr, te = trials(args.data, args.latents, "train"), trials(args.data, args.latents, "ood_val")
    report = {"trials": {"train": [len(tr), int(sum(t["y"] for t in tr))], "ood_val": [len(te), int(sum(t["y"] for t in te))]},
              "note": "[trials, jammed]; probes trained on train, scored on ood_val (held-out shapes)", "probes": {}}
    print(json.dumps(report["trials"]), flush=True)
    for name, dj in () if args.skip_probes else (("push_start", 0), ("push+0.4s", WM_STRIDE), ("push+0.8s", 2 * WM_STRIDE)):
        ftr = [(features(t, t["j0"] + dj, scale), t["y"]) for t in tr]
        fte = [(features(t, t["j0"] + dj, scale), t["y"]) for t in te]
        ftr, fte = [x for x in ftr if x[0] is not None], [x for x in fte if x[0] is not None]
        ytr, yte = np.array([y for _, y in ftr]), np.array([y for _, y in fte])
        for inp in ("latent", "joints", "truth"):
            xtr = np.stack([f[inp] for f, _ in ftr]).astype(np.float32)
            xte = np.stack([f[inp] for f, _ in fte]).astype(np.float32)
            for kind in ("logistic", "mlp"):
                r = fit_eval(xtr, ytr, xte, yte, kind)
                report["probes"][f"{name}/{inp}/{kind}"] = r
                print(f"{name:11s} {inp:7s} {kind:8s} {r}", flush=True)
    if not args.skip_probes:
        (out / "report.json").write_text(json.dumps(report, indent=1))
    if args.wm:
        ck_step = torch.load(args.wm, map_location="cpu", weights_only=False).get("step")
        report["world_model"] = {"checkpoint_step": ck_step, **world_model_part(te, args.vae, args.wm, scale, args.wm_windows)}
        print(json.dumps(report["world_model"], indent=1), flush=True)
    (out / ("report.json" if not args.skip_probes else "report_world_model.json")).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
