"""Pictures and numbers from a checkpoint of the unchanged NWM trained on our data (README.md). Inference as in NWM's
isolated_nwm_infer.py: the EMA weights, `create_diffusion("250")`, its `model_forward_wrapper`, the frozen VAE.

    python world_model/nwm_asis/eval_plug.py --ckpt $Q/runs/nwm_asis/plug_cdit_b/checkpoints/latest.pth.tar --out $Q/runs/nwm_asis/eval_latest
Held-out shapes and grasps (ood_val) only.
  time.png      one prediction straight to +0.4 s / +2 s / +4 s (their "time" evaluation): last seen frame | real | predicted
  rollout.png   step by step, 0.4 s per step, 8 s, the recorded commands, predictions fed back as context (their "rollout")
  push_*.png    approach-set trajectories, taken over about 1 s before the hand meets the board; the command then keeps
                advancing at the speed it had (a command that does not care whether the hand follows), 8 s. Upper row of
                each pair: what the camera really saw (the real script's hand is stopped by the board, or goes in);
                lower row: what the model draws.
  metrics.json  LPIPS / PSNR of the predictions against the real frames, next to "the last seen frame again";
                for the push: how much the picture still changes between 4 s and 8 s (real: the hand is stopped).
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
from PIL import Image, ImageDraw  # noqa: E402

import misc  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from isolated_nwm_infer import model_forward_wrapper  # noqa: E402
from models import CDiT_models  # noqa: E402
from plug_dataset import PlugTrainingDataset, command_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="CDiT-B/2")
ap.add_argument("--n", type=int, default=6, help="cases per data set and section")
ap.add_argument("--sampling", default="250")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--random-init", action="store_true", help="only to test this script")
ap.add_argument("--weights", default="ema", choices=["ema", "model"],
                help="ema: the averaged weights NWM's inference loads (decay 0.9999: after N steps a share 0.9999^N of them is still the "
                     "random start -- 61 % at 5000 steps, 14 % at 20 000, 2 % at 40 000); model: the trained weights themselves, for "
                     "looking at early checkpoints")
args = ap.parse_args()
frac = os.environ.get("WM_GPU_MEM_FRACTION")
if frac:
    torch.cuda.set_per_process_memory_fraction(float(frac))
dev = torch.device("cuda")
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
Q = "/mnt/storage2/qingxin/data/"
CTX, LAT = 4, 28
model = CDiT_models[args.model](context_size=CTX, input_size=LAT, in_channels=4)
step = -1
if not args.random_init:
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck[args.weights].items()}, strict=True))
    step = int(ck.get("train_steps", -1))
    del ck
model.eval().to(dev)
allm = (model, create_diffusion(args.sampling), AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev))
import lpips  # noqa: E402

lp = lpips.LPIPS(net="alex").to(dev)
os.makedirs(args.out, exist_ok=True)
sets = {n: PlugTrainingDataset(Q + n, "ood_val", n, 224, -64, 64, 1, 1, CTX, misc.transform, goals_per_obs=1) for n in ("wm_v1", "wm_v1x", "wm_v1c", "wm_v1a")}
u8 = lambda x: ((x.float().clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
psnr = lambda a, b: float(-10 * torch.log10(((a.float() - b.float()) * 0.5).pow(2).mean().clamp(min=1e-10)))


def predict(ctx, y, frames):
    """ctx (B, 4, 3, 224, 224), y (B, 3) normalised action, frames (B,) offset in images -> (B, 3, 224, 224)."""
    return model_forward_wrapper(allm, ctx, y[:, None], None, LAT, dev, CTX, num_goals=1, rel_t=(frames.float() / 128.0).to(dev)).float()


def sheet(path, rows, labels, title):
    """rows: list of lists of uint8 images; labels: one per column."""
    h, w = rows[0][0].shape[:2]
    im = Image.new("RGB", (w * len(labels) + 4, 30 + h * len(rows)), "white")
    d = ImageDraw.Draw(im)
    d.text((4, 2), title, fill="black")
    for c, s in enumerate(labels):
        d.text((4 + c * w, 16), s, fill="black")
    for r, row in enumerate(rows):
        for c, a in enumerate(row):
            im.paste(Image.fromarray(a), (2 + c * w, 30 + r * h))
    im.save(path)


rep = {"checkpoint": args.ckpt, "train_steps": step, "sampling_steps": args.sampling, "weights": args.weights}
# ------------------------------------------------------------------ time: one prediction to +0.4 / +2 / +4 s
HOR = (2, 10, 20)
rows, acc = [], {}
for name, ds in sets.items():
    ok = [k for k in rng.permutation(len(ds.index_to_data)) if ds.index_to_data[k][3] >= HOR[-1]][:args.n]
    ctxs, reals, ys = [], [], []
    for k in ok:
        i, curr, _, _ = ds.index_to_data[k]
        _, P, R = command_positions(ds.paths[i], ds.root)
        x = ds.images(ds.paths[i], list(range(curr - CTX + 1, curr + 1)) + [curr + h for h in HOR])
        ctxs.append(x[:CTX]), reals.append(x[CTX:]), ys.append(torch.as_tensor(ds.action(P, R, curr, np.array([curr + h for h in HOR])), dtype=torch.float32))
    ctx, real, y = torch.stack(ctxs).to(dev), torch.stack(reals).to(dev), torch.stack(ys).to(dev)
    preds = [predict(ctx, y[:, j], torch.full((len(ok),), h)) for j, h in enumerate(HOR)]
    for j, h in enumerate(HOR):
        with torch.no_grad():
            a = acc.setdefault(f"{name} +{h * 0.2:g}s", {})
            a["lpips_predicted"], a["lpips_last_frame_again"] = round(float(lp(preds[j], real[:, j]).mean()), 4), round(float(lp(ctx[:, -1], real[:, j]).mean()), 4)
            a["psnr_predicted"], a["psnr_last_frame_again"] = round(psnr(preds[j], real[:, j]), 2), round(psnr(ctx[:, -1], real[:, j]), 2)
    for b in range(min(3, len(ok))):
        rows.append([u8(ctx[b, -1])] + [u8(v) for j in range(len(HOR)) for v in (real[b, j], preds[j][b])])
rep["time"] = acc
sheet(os.path.join(args.out, "time.png"), rows, ["now"] + [s for h in HOR for s in (f"real +{h * 0.2:g}s", f"model +{h * 0.2:g}s")],
      f"step {step}: one prediction straight to the time; rows: 3 held-out cases each of demos / exploration / contact / approach")
print("time", json.dumps(acc), flush=True)

# ------------------------------------------------------------------ rollout: 0.4 s per step, recorded commands
STEPS, SHOW = 20, (0, 2, 4, 8, 12, 16, 20)
rows, acc = [], {}
for name, ds in sets.items():
    ok = [k for k in rng.permutation(len(ds.index_to_data)) if ds.index_to_data[k][3] >= 2 * STEPS][:args.n]
    if not ok:
        continue
    ctxs, reals, ys = [], [], []
    for k in ok:
        i, curr, _, _ = ds.index_to_data[k]
        _, P, R = command_positions(ds.paths[i], ds.root)
        x = ds.images(ds.paths[i], list(range(curr - CTX + 1, curr + 1)) + [curr + 2 * s for s in range(1, STEPS + 1)])
        ctxs.append(x[:CTX]), reals.append(x[CTX:])
        ys.append(torch.as_tensor(np.stack([ds.action(P, R, curr + 2 * s, np.array([curr + 2 * s + 2]))[0] for s in range(STEPS)]), dtype=torch.float32))
    ctx, real, y = torch.stack(ctxs).to(dev), torch.stack(reals).to(dev), torch.stack(ys).to(dev)
    cur, out = ctx.clone(), []
    for s in range(STEPS):
        p = predict(cur, y[:, s], torch.full((len(ok),), 2))
        out.append(p)
        cur = torch.cat([cur[:, 1:], p[:, None]], 1)
    for s in (5, 10, 20):
        with torch.no_grad():
            acc[f"{name} +{s * 0.4:g}s"] = {"lpips_predicted": round(float(lp(out[s - 1], real[:, s - 1]).mean()), 4), "lpips_first_frame_again": round(float(lp(ctx[:, -1], real[:, s - 1]).mean()), 4),
                                         "psnr_predicted": round(psnr(out[s - 1], real[:, s - 1]), 2), "psnr_first_frame_again": round(psnr(ctx[:, -1], real[:, s - 1]), 2)}
    for b in range(min(2, len(ok))):
        rows.append([u8(ctx[b, -1])] + [u8(real[b, s - 1]) for s in SHOW[1:]])
        rows.append([u8(ctx[b, -1])] + [u8(out[s - 1][b]) for s in SHOW[1:]])
rep["rollout_recorded_commands"] = acc
sheet(os.path.join(args.out, "rollout.png"), rows, [f"+{s * 0.4:g}s" for s in SHOW],
      f"step {step}: step-by-step rollout with the recorded commands; pairs of rows = real (upper) / model (lower); 2 cases each of demos / exploration / contact / approach")
print("rollout", json.dumps(acc), flush=True)

# ------------------------------------------------------------------ push: the command keeps advancing into the board
from wm.data.frame_history import pressed_at  # noqa: E402

ds = sets["wm_v1a"]
entries = {e["path"]: e for e in json.load(open(Q + "wm_v1a/index.json")) if e["split"] == "ood_val"}
cases = {"blocked": [], "inserted": []}
for i in rng.permutation(len(ds.paths)):
    path = ds.paths[i]
    with np.load(path) as f:
        steps, hand, meta = f["img_step"].astype(int), f["hand"].astype(np.float64), json.loads(str(f["meta"]))
    kind = "inserted" if meta.get("outcome") == "inserted" else "blocked"
    if len(cases[kind]) >= (args.n if kind == "blocked" else max(2, args.n // 2)):
        if all(len(v) >= (args.n if k == "blocked" else max(2, args.n // 2)) for k, v in cases.items()):
            break
        continue
    with np.load(path.replace(ds.root, ds.root + "_ee", 1)) as g:
        cmd = g["hand_cmd"].astype(np.float64)
    a = np.asarray(entries[path]["R_P"], np.float64)[:, 0]
    stop = next((s for s in range(8, len(cmd)) if pressed_at(hand, cmd, s, a)), None)
    if stop is None:
        continue
    curr = int(np.searchsorted(steps, stop)) - 5                       # about 1 s before the hand is first called pressed
    if curr < CTX + 4 or curr + 2 * STEPS >= len(steps):
        continue
    _, P, R = command_positions(path, ds.root)
    v = ((P[curr] - P[curr - 5]) @ R[curr]) / 5.0                      # the command's own step per image over the last second, its frame now
    if np.linalg.norm(v) < 1e-4:
        continue
    x = ds.images(path, list(range(curr - CTX + 1, curr + 1)) + [curr + 2 * s for s in range(1, STEPS + 1)])
    y = (2 * v / ds.spacing - ds.amin) / (ds.amax - ds.amin) * 2 - 1
    cases[kind].append((x[:CTX], x[CTX:], torch.as_tensor(y, dtype=torch.float32), float(np.linalg.norm(v)) * 2000,
                        float((hand[steps[curr + 2 * STEPS], :3, 3] - hand[steps[curr], :3, 3]) @ a) * 1000))
acc = {}
for kind, cs in cases.items():
    if not cs:
        continue
    ctx, real, y = torch.stack([c[0] for c in cs]).to(dev), torch.stack([c[1] for c in cs]).to(dev), torch.stack([c[2] for c in cs]).to(dev)
    cur, out = ctx.clone(), []
    for s in range(STEPS):
        p = predict(cur, y, torch.full((len(cs),), 2))
        out.append(p)
        cur = torch.cat([cur[:, 1:], p[:, None]], 1)
    with torch.no_grad():
        acc[kind] = {"cases": len(cs), "commanded_advance_per_0.4s_mm": round(float(np.mean([c[3] for c in cs])), 3),
                     "commanded_advance_in_8s_mm": round(float(np.mean([c[3] for c in cs])) * STEPS, 1),
                     "real_hand_travel_in_8s_mm": round(float(np.mean([c[4] for c in cs])), 2),
                     "picture_change_4s_to_8s_lpips": {"real": round(float(lp(real[:, 9], real[:, 19]).mean()), 4), "model": round(float(lp(out[9], out[19]).mean()), 4)},
                     "picture_change_0s_to_8s_lpips": {"real": round(float(lp(ctx[:, -1], real[:, 19]).mean()), 4), "model": round(float(lp(ctx[:, -1], out[19]).mean()), 4)},
                     "model_vs_real_at_8s_lpips": round(float(lp(out[19], real[:, 19]).mean()), 4)}
    rows = []
    for b in range(len(cs)):
        rows.append([u8(ctx[b, -1])] + [u8(real[b, s - 1]) for s in SHOW[1:]])
        rows.append([u8(ctx[b, -1])] + [u8(out[s - 1][b]) for s in SHOW[1:]])
    sheet(os.path.join(args.out, f"push_{kind}.png"), rows, [f"+{s * 0.4:g}s" for s in SHOW],
          f"step {step}: really {kind}; the command keeps advancing {acc[kind]['commanded_advance_per_0.4s_mm']} mm per 0.4 s; pairs of rows = real camera (upper) / model (lower)")
rep["push_command_keeps_advancing"] = acc
print("push", json.dumps(acc), flush=True)
json.dump(rep, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
