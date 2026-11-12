"""Figures of the v4 perception for the project log: what the UNet segments, where the two crops are cut, and what
the crop autoencoder reconstructs from the latent (the only visual information the world model, the success
classifier and the joint adapter get).

    python world_model/scripts/analysis/make_architecture_figures.py --unet $Q/runs/unet_v4/best.pt \
        --ae $Q/runs/crop_ae_v4e/best.pt --demos $Q/data/wm_v1 --explore $Q/data/wm_v1x --out $Q/runs/checks_v4/figures

Trajectories: held-out contact exploration on port shapes that never occur in the training split, one that ends
inserted and one that ends blocked; four frames each (evenly spaced). The pipeline is the deployed one (UNet ->
CropPlacer, stateful over the trajectory -> CropAE.encode -> decode); simulator truth is used only to print the
remaining depth on the tiles. Also: reconstruction errors over 40 held-out trajectories per source.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from wm.data import index, store
from wm.data.outcome import diagnosis
from wm.models.crop_ae import load_crop_ae, unpatch
from wm.models.encoder import DEPTH_SCALE_M
from wm.perception.crops import CROP, CropPlacer
from wm.perception.labels import CLASSES
from wm.training.common import limit_gpu_memory
from wm.perception.unet import load_unet

ap = argparse.ArgumentParser()
for a in ("--unet", "--ae", "--demos", "--explore", "--out"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
limit_gpu_memory()
dev = torch.device("cuda")
unet = load_unet(args.unet, dev)
ae = load_crop_ae(args.ae, dev).requires_grad_(False)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)[None]
COLORS = {"plug": (255, 150, 60), "robot": (200, 60, 60), "pins": (255, 235, 0), "opening": (0, 230, 255)}   # RGB
PALETTE = np.array([{"other": (40, 40, 40), "board": (150, 150, 150), "table": (90, 110, 90)}.get(c, COLORS.get(c)) for c in CLASSES], np.uint8)


@torch.no_grad()
def run(path: str) -> dict:
    """The deployed perception on every image of a trajectory."""
    a, meta = store.load(path)
    R_P = np.asarray(index_entry[path]["R_P"], np.float64)
    placer = CropPlacer()
    frames = []
    for j, s in enumerate(a["img_step"].astype(int)):
        rgb, depth = a["rgb"][j], a["depth"][j]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = unet(T(rgb, torch.uint8)).argmax(1)[0].to(torch.uint8).cpu().numpy()
        c = placer(cls, depth, a["K"], a["cam_T_world"][j])
        b = {"rgb": T(rgb, torch.uint8), "depth": T(depth.astype(np.int32), torch.int32),
             "plug_uv0": T(c["plug_uv0"], torch.long), "hole_uv0": T(c["hole_uv0"], torch.long), "K": T(a["K"]),
             "cam_T_world": T(a["cam_T_world"][j]), "hand": T(a["hand"][s]), "R_P": T(R_P)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, aux = ae.encode(b)
            dec = ae.decode(z)
        f = {"rgb": rgb, "cls": cls, "c": c, "z": z.float()[0].cpu().numpy(), "t": s / 20.0,
             "depth_left_mm": diagnosis(a["eval_rel"][s])["depth_left_m"] * 1000}
        for k in ("plug", "hole"):
            u0, v0 = [int(x) for x in c[f"{k}_uv0"]]
            f[k] = {"rgb_in": unpatch(aux[k]["rgb_patch"].float(), 3)[0].permute(1, 2, 0).cpu().numpy(),
                    "rgb_out": unpatch(dec[k]["rgb_patch"].float(), 3)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() if "rgb_patch" in dec[k] else None,
                    "cls_in": cls[v0:v0 + CROP, u0:u0 + CROP],
                    "cls_out": unpatch(dec[k]["seg_patch"].float(), len(CLASSES))[0].argmax(0).cpu().numpy() if "seg_patch" in dec[k] else None,
                    "d_in": unpatch(aux[k]["depth_patch"].float(), 1)[0, 0].cpu().numpy() * DEPTH_SCALE_M * 1000,
                    "v_in": unpatch(aux[k]["valid_patch"].float(), 1)[0, 0].cpu().numpy() > 0.5,
                    "d_out": unpatch(dec[k]["depth_patch"].float(), 1)[0, 0].cpu().numpy() * DEPTH_SCALE_M * 1000,
                    "v_out": torch.sigmoid(unpatch(dec[k]["valid_logit"].float(), 1))[0, 0].cpu().numpy() > 0.5}
        frames.append(f)
    return {"frames": frames, "meta": meta}


def label(img, text, scale=0.42):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (20, 24, 30), -1)
    cv2.putText(img, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 238, 242), 1, cv2.LINE_AA)
    return img


def overlay(rgb, cls):
    o = rgb.astype(np.float32).copy()
    for name, col in COLORS.items():
        m = cls == CLASSES.index(name)
        a = 0.75 if name in ("pins", "opening") else 0.35
        o[m] = (1 - a) * o[m] + a * np.array(col, np.float32)
    return o.astype(np.uint8)


def boxes(rgb, c):
    o = rgb.copy()
    for key, col in (("plug_uv0", COLORS["pins"]), ("hole_uv0", COLORS["opening"])):
        u, v = [int(x) for x in c[key]]
        cv2.rectangle(o, (u, v), (u + CROP - 1, v + CROP - 1), col, 2)
    return o


def depth_tile(d, valid, lo, hi):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy()
    img[~valid] = 0
    return img


def up(img, s):
    return cv2.resize(img, (img.shape[1] * s // 2, img.shape[0] * s // 2), interpolation=cv2.INTER_NEAREST)


def save(name, img):
    cv2.imwrite(str(out / name), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def sheet_segmentation(r, tag):
    fr = r["frames"]
    idx = np.rint(np.linspace(0, len(fr) - 1, 4)).astype(int)
    rows = [[], [], []]
    for j in idx:
        f = fr[j]
        head = f"t={f['t']:.1f}s  {f['depth_left_mm']:.1f} mm to go"
        rows[0].append(label(f["rgb"].copy(), "camera RGB   " + head))
        rows[1].append(label(overlay(f["rgb"], f["cls"]), "UNet classes"))
        rows[2].append(label(boxes(f["rgb"], f["c"]), f"crops: plug {f['c']['how_plug']}, opening {f['c']['how_hole']}"))
    save(f"seg_{tag}.jpg", np.concatenate([np.concatenate(x, 1) for x in rows], 0))
    return idx


def sheet_reconstruction(r, tag, idx):
    rows = []
    for j in idx:
        f = r["frames"][j]
        tiles = []
        for k, nm in (("plug", "plug crop"), ("hole", "opening crop")):
            c = f[k]
            tiles.append(label(up((c["rgb_in"] * 255).astype(np.uint8), 3), f"{nm}: RGB in"))
            if c["rgb_out"] is not None:
                tiles.append(label(up((c["rgb_out"] * 255).astype(np.uint8), 3), "RGB from latent"))
            if c["cls_out"] is not None:                     # v5b: the decoder gives classes, not colours
                tiles.append(label(up(PALETTE[c["cls_in"]], 3), "UNet classes (target)"))
                tiles.append(label(up(PALETTE[c["cls_out"]], 3), "classes from latent"))
            v = c["v_in"]
            lo, hi = (np.percentile(c["d_in"][v], [2, 98]) if v.sum() > 20 else (60.0, 200.0))
            tiles.append(label(up(depth_tile(c["d_in"], v, lo, hi), 3), f"depth in {lo:.0f}-{hi:.0f} mm"))
            tiles.append(label(up(depth_tile(c["d_out"], c["v_out"], lo, hi), 3), "depth from latent"))
        rows.append(np.concatenate(tiles, 1))
    save(f"recon_{tag}.jpg", np.concatenate(rows, 0))


def errors(paths) -> dict:
    e_rgb, e_d, v_acc, how, iou = [], [], [], {}, {"pins": [0, 0], "opening": [0, 0]}
    for p in paths:
        for f in run(p)["frames"]:
            how[f["c"]["how_hole"]] = how.get(f["c"]["how_hole"], 0) + 1
            for k in ("plug", "hole"):
                c = f[k]
                if c["rgb_out"] is not None:
                    e_rgb.append(np.abs(c["rgb_in"] - c["rgb_out"]).mean() * 255)
                if c["cls_out"] is not None:
                    for n in iou:
                        a, b = c["cls_out"] == CLASSES.index(n), c["cls_in"] == CLASSES.index(n)
                        iou[n][0] += int((a & b).sum())
                        iou[n][1] += int((a | b).sum())
                both = c["v_in"] & c["v_out"]
                if both.sum() > 20:
                    e_d.append(np.abs(c["d_in"] - c["d_out"])[both].mean())
                v_acc.append((c["v_in"] == c["v_out"]).mean())
    return {"trajectories": len(paths), "crops": len(v_acc),
            "rgb_abs_error_levels_of_255_mean": round(float(np.mean(e_rgb)), 2) if e_rgb else None,
            "classes_from_latent_iou": {n: round(v[0] / max(v[1], 1), 3) for n, v in iou.items()} if any(v[1] for v in iou.values()) else None,
            "depth_abs_error_mm_mean": round(float(np.mean(e_d)), 3), "depth_abs_error_mm_median_over_crops": round(float(np.median(e_d)), 3),
            "depth_validity_agreement": round(float(np.mean(v_acc)), 4), "opening_crop_placed_by": how}


demo, expl = index.load(args.demos), index.load(args.explore)
index_entry = {e["path"]: e for e in demo + expl}
train_ports = {e["port"] for e in demo if e["split"] == "train"}
unseen = [e for e in expl if e["split"] == "ood_val" and e["port"] not in train_ports and e["n_images"] >= 20]
pick = {"inserted": next(e for e in unseen if e["outcome"] == "inserted"),
        "blocked": next(e for e in unseen if e["outcome"] != "inserted" and e["port"] != next(x for x in unseen if x["outcome"] == "inserted")["port"])}
info = {}
for tag, e in pick.items():
    r = run(e["path"])
    idx = sheet_segmentation(r, tag)
    sheet_reconstruction(r, tag, idx)
    z = r["frames"][idx[2]]["z"]                                   # the latent of one frame: 64 tokens x 16
    zi = np.clip((z - z.mean()) / (3 * z.std()) * 0.5 + 0.5, 0, 1)
    save(f"latent_{tag}.jpg", cv2.resize(cv2.applyColorMap((255 * zi.T).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1], (64 * 10, 16 * 10), interpolation=cv2.INTER_NEAREST))
    info[tag] = {"port": e["port"], "condition": e["condition"], "outcome": e["outcome"], "images": e["n_images"],
                 "frames_shown": [{"image": int(j), "t_s": r["frames"][j]["t"], "depth_left_mm": round(r["frames"][j]["depth_left_mm"], 2),
                                   "plug_crop_by": r["frames"][j]["c"]["how_plug"], "opening_crop_by": r["frames"][j]["c"]["how_hole"]} for j in idx]}
rng = np.random.default_rng(0)
ev = [e for e in expl if e["split"] == "ood_val"]
dv = [e for e in demo if e["split"] == "ood_val" and e["n_images"] >= 10]
info["reconstruction_errors"] = {
    "exploration_held_out": errors([ev[i]["path"] for i in rng.choice(len(ev), 40, replace=False)]),
    "demonstrations_held_out": errors([dv[i]["path"] for i in rng.choice(len(dv), 40, replace=False)])}
info["unseen_ports"] = sorted({e["port"] for e in unseen})
(out / "figures.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
