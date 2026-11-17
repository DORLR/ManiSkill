"""Does the world model know INSIDE whether this straight push goes in or is blocked, even though its drawn pictures do not
show it? (2026-11-14; the owner: "停了做检查"). Nothing is trained in the world model; it is frozen.

Same cases and labels as wm_outcome.py (press set; target = the image where the REQUESTED command is 5 mm past the board;
inserted = hand >= 3 mm past the board there, blocked = <= 1 mm and the push ends blocked). For every case, ONE forward pass
of the frozen CDiT: context = 4 real images, action = the commanded displacement from the last context image to the target
(normalised as in training), relative time = images ahead / 128, diffusion step t = 999 with pure noise as the target
input (the start of sampling: nothing about the target is given). The hidden tokens of blocks 4 / 8 / 12 (mean- and
max-pooled over the 196 target tokens) are read by a small classifier trained on the train split; AUC on ood_val.
Two context choices:
    start        the first 4 images (as wm_outcome.py, where the drawn pictures gave AUC 0.57)
    pre_contact  the 4 images ending at the last image with the hand >= 0.3 mm in front of the board
Reference with the SAME 4 images and NWM's own image transform: a ResNet-18 trained on them (what the pictures allow).
    python world_model/nwm_asis/wm_internal_outcome.py --ckpt <ckpt> --out <dir>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NWM = os.environ.get("NWM_DIR", "/mnt/storage2/qingxin/code/nwm_orig")
sys.path.insert(0, HERE)
sys.path.insert(0, NWM)
os.chdir(NWM)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import misc  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402
from models import CDiT_models  # noqa: E402
from plug_dataset import PlugTrainingDataset, command_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weights", default="model", choices=["ema", "model"])
ap.add_argument("--past-mm", type=float, default=5.0)
args = ap.parse_args()
frac = os.environ.get("WM_GPU_MEM_FRACTION")
if frac:
    torch.cuda.set_per_process_memory_fraction(float(frac))
dev = torch.device("cuda")
os.makedirs(args.out, exist_ok=True)
ROOT, CTX, LAT = "/mnt/storage3/qingxin/data/wm_v1p", 4, 28


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p, np.float64)
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cases(split):
    ds = PlugTrainingDataset(ROOT, split, "wm_v1p", 224, -64, 64, 1, 1, CTX, misc.transform, goals_per_obs=1)
    out = []
    for path in ds.paths:
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            hand, steps = f["hand"][:, :3, 3].astype(np.float64), f["img_step"].astype(int)
        req = np.load(path.replace(ROOT, ROOT + "_ee", 1))["hand_cmd"][:, :3, 3].astype(np.float64)
        a = np.asarray(meta["axis"], np.float64)
        a /= np.linalg.norm(a)
        L, Hs = meta["pin_length_m"] * 1000, np.asarray(meta["H_star"])[:3, 3]
        dh = (hand - Hs) @ a * 1000 + L
        dr = (req - Hs) @ a * 1000 + L
        first = meta["log"][0]
        n_press = first["chunks"] * meta["planned"][0]["chunk_steps"]
        if first["kind"] != "press":
            continue
        s_img = np.clip(steps, 0, len(dr) - 1)
        tgt = next((j for j in range(len(steps)) if steps[j] <= n_press and dr[s_img[j]] >= args.past_mm), None)
        if tgt is None or tgt < CTX + 1 or tgt - (CTX - 1) > 64 or dh[steps[CTX - 1]] > -0.3:
            continue
        if dh[steps[tgt]] >= 3.0:
            y = 1
        elif dh[steps[tgt]] <= 1.0 and dh[: n_press + 1].max() <= 1.0:
            y = 0
        else:
            continue
        pre = max(j for j in range(tgt) if dh[steps[j]] <= -0.3)
        if pre < CTX - 1:
            continue
        _, P, R = command_positions(path, ROOT)
        out.append({"path": path, "y": y, "tgt": tgt, "pre": pre, "P": P, "R": R, "condition": meta["condition"], "port": meta["port"]})
    return ds, out


ds_tr, tr = cases("train")
ds_te, te = cases("ood_val")
ytr, yte = np.array([c["y"] for c in tr]), np.array([c["y"] for c in te])
print("train", len(tr), int(ytr.sum()), "| ood_val", len(te), int(yte.sum()), flush=True)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = CDiT_models["CDiT-B/2"](context_size=CTX, input_size=LAT, in_channels=4)
print(model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck[args.weights].items()}, strict=True))
step = int(ck.get("train_steps", -1))
del ck
model.eval().to(dev)
grab = {}
for k in (3, 7, 11):
    model.blocks[k].register_forward_hook(lambda m, i, o, k=k: grab.__setitem__(k, o))


@torch.no_grad()
def features(ds, cs, last):
    """last(c) -> index of the last context image. Returns CDiT features (N, 3 * 2 * D) and the context images (N, 4, 3, 224, 224)."""
    feats, imgs = [], []
    g = torch.Generator(device="cpu").manual_seed(0)
    for b0 in range(0, len(cs), 16):
        bc = cs[b0:b0 + 16]
        x = torch.stack([ds.images(c["path"], list(range(last(c) - CTX + 1, last(c) + 1))) for c in bc])
        imgs.append(x.half())
        y = torch.as_tensor(np.stack([ds.action(c["P"], c["R"], last(c), np.array([c["tgt"]]))[0] for c in bc]), dtype=torch.float32, device=dev)
        rel = torch.as_tensor([(c["tgt"] - last(c)) / 128.0 for c in bc], dtype=torch.float32, device=dev)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            z = vae.encode(x.flatten(0, 1).to(dev)).latent_dist.mean.mul_(0.18215).unflatten(0, (len(bc), CTX))
            noise = torch.randn(len(bc), 4, LAT, LAT, generator=g).to(dev)
            model(noise, torch.full((len(bc),), 999, device=dev), y, z, rel)
        feats.append(torch.cat([torch.cat([grab[k].float().mean(1), grab[k].float().amax(1)], 1) for k in (3, 7, 11)], 1).cpu())
    return torch.cat(feats), torch.cat(imgs)


VAL = np.array([sum(map(ord, c["port"])) % 5 == 0 for c in tr])                  # 1/5 of the training ports: choose weight decay


def mlp_probe(Ftr, Fte):
    """Weight decay chosen on held-out TRAINING ports (VAL), then retrained on all training cases; AUC on ood_val."""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Xtr, Xte = ((Ftr - mu) / sd).to(dev), ((Fte - mu) / sd).to(dev)

    def fit(X, y, wd):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
        t = torch.as_tensor(y, dtype=torch.float32, device=dev)
        for _ in range(1500):
            idx = torch.randint(0, len(X), (128,), device=dev)
            l = nn.functional.binary_cross_entropy_with_logits(m(X[idx]).ravel(), t[idx])
            o.zero_grad(); l.backward(); o.step()
        return m.eval()
    pr = lambda m, X: torch.sigmoid(m(X)).ravel().detach().cpu().numpy()
    vt, vv = torch.as_tensor(~VAL, device=dev), torch.as_tensor(VAL, device=dev)
    sel = {wd: round(auc(ytr[VAL], pr(fit(Xtr[vt], ytr[~VAL], wd), Xtr[vv])), 3) for wd in (1e-3, 1e-2, 1e-1)}
    wd = max(sel, key=sel.get)
    m = fit(Xtr, ytr, wd)
    r = {"val_auc_by_weight_decay": sel, "weight_decay": wd, "train_auc": round(auc(ytr, pr(m, Xtr)), 3), "ood_val_auc": round(auc(yte, pr(m, Xte)), 3)}
    print("  probe", r, flush=True)
    return r


def pixel_probe(Itr, Ite):
    import torchvision
    r = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    r.fc = nn.Identity()
    head = nn.Sequential(nn.Linear(512 * CTX, 256), nn.GELU(), nn.Linear(256, 1))
    net = nn.ModuleDict({"r": r, "h": head}).to(dev)
    M = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    S = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    fwd = lambda x: net["h"](net["r"](((x.flatten(0, 1).to(dev).float() * 0.5 + 0.5) - M) / S).view(x.shape[0], -1)).ravel()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    t = torch.as_tensor(ytr, dtype=torch.float32)
    for _ in range(1000):
        idx = rng.integers(0, len(Itr), 16)
        l = nn.functional.binary_cross_entropy_with_logits(fwd(Itr[idx]), t[idx].to(dev))
        opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(fwd(Ite[i:i + 32])).cpu() for i in range(0, len(Ite), 32)]).numpy()
    return round(auc(yte, p), 3)


res = {"checkpoint": args.ckpt, "train_steps": step, "weights": args.weights, "n_train": [len(tr), int(ytr.sum())], "n_ood_val": [len(te), int(yte.sum())]}
for name, last in (("start", lambda c: CTX - 1), ("pre_contact", lambda c: c["pre"])):
    Ftr, Itr = features(ds_tr, tr, last)
    Fte, Ite = features(ds_te, te, last)
    print(name, "features", tuple(Ftr.shape), flush=True)
    res[name] = {"cdit_internal": mlp_probe(Ftr, Fte)}
    per_block = {}
    D = Ftr.shape[1] // 3
    for i, k in enumerate((3, 7, 11)):
        per_block[f"block_{k + 1}"] = mlp_probe(Ftr[:, i * D:(i + 1) * D], Fte[:, i * D:(i + 1) * D])
    res[name]["cdit_per_block"] = per_block
    res[name]["pixels_same_images_resnet"] = pixel_probe(Itr, Ite)
    print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
print(json.dumps(res), flush=True)
