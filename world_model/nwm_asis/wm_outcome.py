"""Can the world model tell "this straight push goes in" from "it is blocked"? (2026-11-14; the owner: "去测，测完给我结果")

Press set, held-out shapes and grasps (ood_val). Each trajectory begins with a straight push from 2-6 mm in front of the board.
    context   the first 4 images (the pins are still in front of the board; checked per case)
    target    the image at which the REQUESTED command is 5 mm past the point where the pin tips meet the board (the same
              rule for both classes, no truth used). Label at that image (simulator truth, label only): inserted = the hand
              is >= 3 mm past that point, blocked = <= 1 mm past it and the push ends blocked; other cases are left out.
    world model  the trained weights of a checkpoint, NWM's inference (create_diffusion(--sampling), model_forward_wrapper)
              a) rollout: 0.4 s per step with the recorded per-step commands, predictions fed back (as eval_plug.py)
              b) one prediction straight to the target time with the whole commanded displacement
    reader    a ResNet-18 trained to tell inserted from blocked on REAL target images of the TRAIN split, after one pass
              through the frozen VAE (so it sees the same kind of picture the model draws). Its AUC on real ood_val target
              images is the ceiling; its AUC on the imagined target images is the answer. Also: the reader on the last context
              image (nothing predicted).
    python world_model/nwm_asis/wm_outcome.py --ckpt <ckpt> --out <dir>
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
from PIL import Image, ImageDraw  # noqa: E402

import misc  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from isolated_nwm_infer import model_forward_wrapper  # noqa: E402
from models import CDiT_models  # noqa: E402
from plug_dataset import PlugTrainingDataset, command_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weights", default="model", choices=["ema", "model"])
ap.add_argument("--sampling", default="250")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--past-mm", type=float, default=5.0)
args = ap.parse_args()
frac = os.environ.get("WM_GPU_MEM_FRACTION")
if frac:
    torch.cuda.set_per_process_memory_fraction(float(frac))
dev = torch.device("cuda")
torch.manual_seed(0)
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
        dh = (hand - Hs) @ a * 1000 + L                                   # mm past where the pin tips meet the board
        dr = (req - Hs) @ a * 1000 + L
        first = meta["log"][0]
        n_press = first["chunks"] * meta["planned"][0]["chunk_steps"]
        if first["kind"] != "press":
            continue
        s_img = np.clip(steps, 0, len(dr) - 1)
        tgt = next((j for j in range(len(steps)) if steps[j] <= n_press and dr[s_img[j]] >= args.past_mm), None)
        if tgt is None or tgt < CTX + 1 or tgt - (CTX - 1) > 64 or dh[steps[CTX - 1]] > -0.3:
            continue
        ended_blocked = dh[: n_press + 1].max() <= 1.0
        if dh[steps[tgt]] >= 3.0:
            y = 1
        elif dh[steps[tgt]] <= 1.0 and ended_blocked:
            y = 0
        else:
            continue
        _, P, R = command_positions(path, ROOT)
        out.append({"path": path, "y": y, "tgt": tgt, "P": P, "R": R, "port": meta["port"], "condition": meta["condition"]})
    return ds, out


ds_tr, tr = cases("train")
ds_te, te = cases("ood_val")
print("train", len(tr), sum(c["y"] for c in tr), "| ood_val", len(te), sum(c["y"] for c in te),
      "| target horizon images, median", int(np.median([c["tgt"] - (CTX - 1) for c in te])), flush=True)

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev)


@torch.no_grad()
def vae_pass(x):
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        return torch.clip(vae.decode(vae.encode(x.to(dev)).latent_dist.mean).sample, -1.0, 1.0).float()


def real_targets(ds, cs):
    return torch.cat([vae_pass(torch.stack([ds.images(c["path"], [c["tgt"]])[0] for c in cs[i:i + 32]])).cpu() for i in range(0, len(cs), 32)])


Xtr, Xte = real_targets(ds_tr, tr), real_targets(ds_te, te)
ytr, yte = np.array([c["y"] for c in tr]), np.array([c["y"] for c in te])

# ------------------------------------------------------------------ the reader (real target images, train split)
import torchvision  # noqa: E402

reader = torchvision.models.resnet18(weights="IMAGENET1K_V1")
reader.fc = nn.Linear(512, 1)
reader = reader.to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
norm = lambda x: ((x.to(dev) * 0.5 + 0.5) - MEAN) / STD
opt = torch.optim.AdamW(reader.parameters(), lr=1e-4, weight_decay=1e-4)
rng = np.random.default_rng(0)
for step in range(800):
    idx = rng.integers(0, len(Xtr), 32)
    loss = nn.functional.binary_cross_entropy_with_logits(reader(norm(Xtr[idx])).ravel(), torch.as_tensor(ytr[idx], dtype=torch.float32, device=dev))
    opt.zero_grad()
    loss.backward()
    opt.step()
reader.eval()


@torch.no_grad()
def read(x):
    return torch.cat([torch.sigmoid(reader(norm(x[i:i + 64]))).ravel().cpu() for i in range(0, len(x), 64)]).numpy()


res = {"checkpoint": args.ckpt, "weights": args.weights, "sampling": args.sampling, "n_train": [len(tr), int(ytr.sum())], "n_ood_val": [len(te), int(yte.sum())],
       "reader_on_real_targets_auc": round(auc(yte, read(Xte)), 3)}
print("reader on real ood_val target images", res["reader_on_real_targets_auc"], flush=True)
del vae

# ------------------------------------------------------------------ the world model
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = CDiT_models["CDiT-B/2"](context_size=CTX, input_size=LAT, in_channels=4)
print(model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck[args.weights].items()}, strict=True))
res["train_steps"] = int(ck.get("train_steps", -1))
del ck
model.eval().to(dev)
allm = (model, create_diffusion(args.sampling), AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev))


def predict(ctx, y, frames):
    return model_forward_wrapper(allm, ctx, y[:, None], None, LAT, dev, CTX, num_goals=1, rel_t=(frames.float() / 128.0).to(dev)).float()


roll, direct, last_ctx, keep = [], [], [], []
CACHE = os.path.join(args.out, "batches")
os.makedirs(CACHE, exist_ok=True)
for b0 in range(0, len(te), args.batch):
    cs = te[b0:b0 + args.batch]
    cf = os.path.join(CACHE, f"b{b0:04d}_n{len(cs)}.pt")
    if os.path.exists(cf):
        c_ = torch.load(cf)
        roll.append(c_["roll"]), direct.append(c_["direct"]), last_ctx.append(c_["last"])
        print(f"{b0 + len(cs)}/{len(te)} (cached)", flush=True)
        continue
    ctx = torch.stack([ds_te.images(c["path"], list(range(CTX))) for c in cs]).to(dev)
    last_ctx.append(ctx[:, -1].cpu())
    H = np.array([c["tgt"] - (CTX - 1) for c in cs])
    # b) one prediction to the target with the whole commanded displacement
    y1 = torch.as_tensor(np.stack([ds_te.action(c["P"], c["R"], CTX - 1, np.array([c["tgt"]]))[0] for c in cs]), dtype=torch.float32, device=dev)
    direct.append(predict(ctx, y1, torch.as_tensor(H)).cpu())
    # a) rollout, 2 images (0.4 s) per step; the last step of a case may be 1 image
    cur, pos, done = ctx.clone(), np.full(len(cs), CTX - 1), [None] * len(cs)
    while any(d is None for d in done):
        k = np.minimum(2, np.maximum(H + (CTX - 1) - pos, 1))
        # cases already at their target keep rolling with a valid in-range action (their output is not used; the first run
        # of 2026-11-14 crashed here at 384/430 with an index past the trajectory's last image)
        y = torch.as_tensor(np.stack([ds_te.action(c["P"], c["R"], int(min(pos[i], c["tgt"] - 1)), np.array([int(min(pos[i] + k[i], c["tgt"]))]))[0]
                                      for i, c in enumerate(cs)]), dtype=torch.float32, device=dev)
        p = predict(cur, y, torch.as_tensor(k))
        pos = pos + k
        for i in range(len(cs)):
            if done[i] is None and pos[i] >= cs[i]["tgt"]:
                done[i] = p[i].cpu()
        cur = torch.cat([cur[:, 1:], p[:, None]], 1)
    roll.append(torch.stack(done))
    torch.save({"roll": roll[-1], "direct": direct[-1], "last": last_ctx[-1]}, cf)
    print(f"{b0 + len(cs)}/{len(te)}", flush=True)
roll, direct, last_ctx = torch.cat(roll), torch.cat(direct), torch.cat(last_ctx)
p_roll, p_dir, p_last = read(roll), read(direct), read(last_ctx)
cond = np.array([c["condition"] for c in te])
by = lambda p: {k: round(auc(yte[cond == k], p[cond == k]), 3) for k in sorted(set(cond)) if len(set(yte[cond == k])) == 2}
res.update({"wm_rollout_auc": round(auc(yte, p_roll), 3), "wm_rollout_by_condition": by(p_roll),
            "wm_one_prediction_auc": round(auc(yte, p_dir), 3), "wm_one_prediction_by_condition": by(p_dir),
            "reader_on_last_context_image_auc": round(auc(yte, p_last), 3),
            "called_inserted_rate_real_rollout_direct": [round(float(yte.mean()), 3), round(float((p_roll > 0.5).mean()), 3), round(float((p_dir > 0.5).mean()), 3)],
            "rollout_called_inserted_given_inserted_blocked": [round(float((p_roll[yte == 1] > 0.5).mean()), 3), round(float((p_roll[yte == 0] > 0.5).mean()), 3)]})
json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
print(json.dumps(res), flush=True)

# a picture: 4 inserted + 4 blocked cases: last context | real target | rollout | one prediction
u8 = lambda x: ((x.float().clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(1, 2, 0).numpy()
pick = [i for i in range(len(te)) if yte[i] == 1][:4] + [i for i in range(len(te)) if yte[i] == 0][:4]
im = Image.new("RGB", (4 * 224 + 4, 30 + 224 * len(pick)), "white")
d = ImageDraw.Draw(im)
d.text((4, 2), f"step {res['train_steps']}: rows 1-4 really inserted, 5-8 really blocked", fill="black")
for c, s in enumerate(["last context", "real target", "model rollout", "model one prediction"]):
    d.text((4 + c * 224, 16), s, fill="black")
for r, i in enumerate(pick):
    for c, x in enumerate([last_ctx[i], Xte[i], roll[i], direct[i]]):
        im.paste(Image.fromarray(u8(x)), (2 + c * 224, 30 + r * 224))
im.save(os.path.join(args.out, "cases.png"))
