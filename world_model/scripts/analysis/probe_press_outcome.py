"""Is "this straight push will go in / will be blocked" in the camera image BEFORE the pins touch the board?
(2026-11-14; the user: "wm怎么知道这个东西能不能插入")

Press set (collect/press.py): every trajectory starts with the same kind of straight push from 2-6 mm in front of the board,
so the outcome of that push depends only on where the plug starts relative to the hole. Labels (simulator truth, label only):
    inserted  at the end of the first press segment the hand is >= 0.8 x pin length past where the pin tips meet the board
    blocked   during the whole first press segment the hand never gets more than 1 mm past that point
    (others -- partly in, jammed -- are left out)
Inputs: the first image and the last image while the hand is still >= 0.3 mm in front of the board (no contact yet).
Arms (trained on the train split, AUC on ood_val = held-out shapes and grasps):
    geometry      the true start offset (lateral y/z, rotation vector, gap) -> a small MLP. Not an image arm: how
                  far the outcome is decided by the start pose at all (port shape not given, so a lower bound of that)
    pixels        ResNet-18 (ImageNet start) trained on the label from the two 320 x 320 RGB images
    nwm_latent    NWM's own input: the frozen SD-VAE latents (centre crop 4:3, 224, as NWM's transform) of the two
                  images -> a small conv net trained on the label
    python world_model/scripts/analysis/probe_press_outcome.py --out /mnt/storage3/qingxin/runs/probe_press_outcome
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def roc_auc_score(y, p):
    """Mann-Whitney AUC with tied ranks (sklearn is not in this environment)."""
    y, p = np.asarray(y), np.asarray(p, np.float64)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/mnt/storage3/qingxin/data/wm_v1p")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--arms", nargs="+", default=["geometry", "nwm_latent", "pixels"])
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
frac = __import__("os").environ.get("WM_GPU_MEM_FRACTION")
if frac and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(frac))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def sample(e):
    with np.load(e["path"]) as f:
        meta = json.loads(str(f["meta"]))
        hand, steps, off, jpeg = f["hand"][:, :3, 3].astype(np.float64), f["img_step"].astype(int), f["rgb_offsets"], f["rgb_jpeg"]
    a = np.asarray(meta["axis"], np.float64)
    a /= np.linalg.norm(a)
    L = meta["pin_length_m"] * 1000
    dh = (hand - np.asarray(meta["H_star"])[:3, 3]) @ a * 1000 + L          # mm past where the pin tips meet the board
    first = meta["log"][0]
    n_press = first["chunks"] * meta["planned"][0]["chunk_steps"]
    if first["kind"] != "press" or n_press < 8 or n_press >= len(dh):
        return None
    if dh[n_press] >= 0.8 * L:
        y = 1
    elif dh[: n_press + 1].max() <= 1.0:
        y = 0
    else:
        return None
    before = [j for j, s in enumerate(steps) if s <= n_press and dh[s] <= -0.3]
    if len(before) < 2:
        return None
    imgs = [np.asarray(Image.open(io.BytesIO(jpeg[off[j]:off[j + 1]].tobytes())).convert("RGB")) for j in (0, before[-1])]
    st = meta["start"]
    geo = np.r_[st["lateral_m"], st["rotvec_deg"], st["gap_m"], np.linalg.norm(st["lateral_m"]), np.linalg.norm(st["rotvec_deg"])]
    return {"y": y, "imgs": np.stack(imgs), "geo": geo, "port": meta["port"], "condition": meta["condition"],
            "gap_left_mm": float(-dh[steps[before[-1]]])}


entries = json.loads((Path(args.data) / "index.json").read_text())
S = {}
for sp in ("train", "ood_val"):
    xs = [s for s in (sample(e) for e in entries if e["split"] == sp) if s is not None]
    S[sp] = xs
    print(sp, "samples", len(xs), "inserted", sum(s["y"] for s in xs), "| gap left at the last image, mm, median",
          round(float(np.median([s["gap_left_mm"] for s in xs])), 2), flush=True)
ytr, yte = np.array([s["y"] for s in S["train"]]), np.array([s["y"] for s in S["ood_val"]])
res = {"n": {k: [len(v), int(sum(s["y"] for s in v))] for k, v in S.items()}}


def by_condition(p):
    c = np.array([s["condition"] for s in S["ood_val"]])
    return {k: round(float(roc_auc_score(yte[c == k], p[c == k])), 3) for k in sorted(set(c)) if len(set(yte[c == k])) == 2}


if "geometry" in args.arms:
    G = np.array([s["geo"] for s in S["train"]]); mu, sd = G.mean(0), G.std(0) + 1e-9
    gtr = torch.as_tensor((G - mu) / sd, dtype=torch.float32)
    gte = torch.as_tensor((np.array([s["geo"] for s in S["ood_val"]]) - mu) / sd, dtype=torch.float32)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(gtr.shape[1], 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(3000):
        l = nn.functional.binary_cross_entropy_with_logits(m(gtr).ravel(), torch.as_tensor(ytr, dtype=torch.float32))
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        p = torch.sigmoid(m(gte)).numpy().ravel()
    res["geometry"] = {"auc": round(float(roc_auc_score(yte, p)), 3), "by_condition": by_condition(p)}
    print("geometry", res["geometry"], flush=True)


def train_eval(name, Xtr, Xte, net):
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    curve = []

    def predict():
        net.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(Xte), 64):
                ps.append(torch.sigmoid(net(Xte[i:i + 64].to(dev))).float().cpu())
        net.train()
        return torch.cat(ps).numpy().ravel()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(Xtr), args.batch)
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[idx].to(dev)).ravel(), torch.as_tensor(ytr[idx], dtype=torch.float32, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            p = predict()
            curve.append([step, round(float(loss), 4), round(float(roc_auc_score(yte, p)), 3)])
            print(name, curve[-1], flush=True)
    p = predict()
    res[name] = {"auc_final": round(float(roc_auc_score(yte, p)), 3), "by_condition": by_condition(p), "curve_step_loss_auc": curve}
    print(name, res[name], flush=True)


if "nwm_latent" in args.arms:
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()

    def latents(xs):
        outl = []
        for s in xs:
            im = torch.as_tensor(s["imgs"]).permute(0, 3, 1, 2).float() / 255
            h, w = im.shape[-2:]
            cw = min(w, int(h * 4 / 3)); ch = int(cw * 3 / 4)                    # NWM: centre crop 4:3, resize 224, [-1, 1]
            t, l = (h - ch) // 2, (w - cw) // 2
            im = nn.functional.interpolate(im[:, :, t:t + ch, l:l + cw], size=(224, 224), mode="bilinear", antialias=True) * 2 - 1
            with torch.no_grad():
                outl.append((vae.encode(im.to(dev)).latent_dist.mean * 0.18215).flatten(0, 1).cpu())   # 2 frames x 4 ch
        return torch.stack(outl)
    Ltr, Lte = latents(S["train"]), latents(S["ood_val"])
    del vae
    net = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
                        nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))
    train_eval("nwm_latent", Ltr, Lte, net)

if "pixels" in args.arms:
    import torchvision

    class Pix(nn.Module):
        def __init__(self):
            super().__init__()
            r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            r.fc = nn.Identity()
            self.r, self.head = r, nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, 1))
            self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):                                                     # x: (B, 2, H, W, 3) uint8
            b = x.shape[0]
            x = (x.permute(0, 1, 4, 2, 3).flatten(0, 1).float() / 255 - self.m) / self.s
            return self.head(self.r(x).view(b, -1))
    Xtr = torch.as_tensor(np.stack([s["imgs"] for s in S["train"]]))
    Xte = torch.as_tensor(np.stack([s["imgs"] for s in S["ood_val"]]))
    train_eval("pixels", Xtr, Xte, Pix())

(out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v.get("auc") or v.get("auc_final")) if isinstance(v, dict) and "by_condition" in v else v for k, v in res.items()}))
