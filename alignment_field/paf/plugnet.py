"""Plug-in-hand network: the plug's pitch in the hand, from the plug points alone.

The relative rotation is split exactly (scripts/plug_labels.py):
    r = rotvec(R_P^T  R_H  M(theta)),   M(theta) = M_ref @ Rot(axis, theta)
R_P: port axes (board-plane fit + gravity), R_H: hand rotation (FK), M: the port axes in the
hand frame at the pre-insertion pose. M only turns about one hand axis by theta (it follows the
grasp pitch); theta is constant during an episode and does not depend on the port shape.
PlugNet predicts theta from the plug points expressed in the hand frame (depth, camera
extrinsics, FK); the port only enters through the board plane used to separate plug points.
Episode estimate = median of the per-frame predictions.

python -m paf.plugnet --packed data/packed_v6 data/packed_v7 --out runs_v7/plug
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import ConcatDataset, DataLoader

from .model import mlp
from .relnet import BOARD_BAND, BOX_YZ, PLUG_X, DinoFeatures, SetAbstraction, _sample, backproject_torch, \
    to_port_frame
from .train_rel import Packed, to_dev

THETA_SCALE = 10.0  # deg


FINGER_X = 0.013   # m: hand-frame points closer than this along the hand x axis are the gripper fingers


class PlugNet(nn.Module):
    def __init__(self, d=192, use_dino=False, n_cand=4096, n_centres=192, no_fingers=False):
        super().__init__()
        self.use_dino, self.n_cand, self.n_centres, self.no_fingers = use_dino, n_cand, n_centres, no_fingers
        self.dino = DinoFeatures() if use_dino else None
        fdim = (384 if use_dino else 0) + 3 + 3
        self.point_in = mlp((fdim, 128, 128), last_act=True)
        self.sa = SetAbstraction(128, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, batch_first=True, norm_first=True), 3)
        self.head = mlp((2 * d, 256, 1))

    def forward(self, rgb, depth_mm, K, cam_T_world, H, R_P, aug_axis=None, aug_rad=None):
        """aug_axis (3,), aug_rad (B,): training-only rotation of the plug points about the hand-frame
        axis of M's family; the label then shifts by the same angle (exact for the plug pose)."""
        B = rgb.shape[0]
        with torch.autocast("cuda", enabled=False):
            pts_cam, valid = backproject_torch(depth_mm, K.float())
            p = to_port_frame(pts_cam, cam_T_world.float(), H.float(), R_P.float()).reshape(B, -1, 3)
            valid = valid.reshape(B, -1)
            lateral = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
            far = valid & lateral & (p[..., 0] > 0.0)
            xs = torch.where(far, p[..., 0], torch.full_like(p[..., 0], float("nan")))
            x_b = torch.nanquantile(xs, 0.8, dim=1, keepdim=True).nan_to_num(0.05)
            plug = valid & lateral & (p[..., 0] < x_b - BOARD_BAND) & (p[..., 0] > PLUG_X[0])
            R_hp = H[:, :3, :3].float().transpose(1, 2) @ R_P.float()        # port axes -> hand axes
            q = p @ R_hp.transpose(1, 2)                                      # plug points in the hand frame
            if self.no_fingers:
                plug = plug & (q[..., 0] > FINGER_X)
            if aug_rad is not None:
                piv = torch.where(plug[..., None], q, torch.full_like(q, float("nan"))).nanmedian(1).values.nan_to_num(0.0)
                a = aug_axis.float()[None].expand(B, 3) * aug_rad.float()[:, None]
                Ra = _rotvec_to_matrix(a)
                q = (q - piv[:, None]) @ Ra.transpose(1, 2) + piv[:, None]
        idx = _sample(plug, self.n_cand)
        xyz = torch.gather(q, 1, idx[..., None].expand(-1, -1, 3))
        col = torch.gather(rgb.reshape(B, -1, 3).float() / 255.0, 1, idx[..., None].expand(-1, -1, 3))
        m = torch.gather(plug, 1, idx)
        feats = [col, xyz / 0.05]
        if self.dino is not None:
            import torch.nn.functional as F
            fmap = self.dino(rgb)
            Hh, Ww = rgb.shape[1], rgb.shape[2]
            vv, uu = (idx // Ww).float(), (idx % Ww).float()
            grid = torch.stack([(uu + 0.5) / Ww * 2 - 1, (vv + 0.5) / Hh * 2 - 1], -1)[:, None]
            feats.insert(0, F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)[:, :, 0].transpose(1, 2))
        h = self.point_in(torch.cat(feats, -1))
        tok, _ = self.sa(xyz, h, m, self.n_centres)
        z = self.enc(tok)
        return self.head(torch.cat([z.mean(1), z.max(1).values], -1))[:, 0] * THETA_SCALE   # deg


@torch.no_grad()
def tip_distance(depth_mm, K, cam_T_world, H, R_P):
    """Distance (m) of the plug's front (pin tips) ahead of the hand along the port axis, from depth."""
    with torch.autocast("cuda", enabled=False):
        pts_cam, valid = backproject_torch(depth_mm, K.float())
        p = to_port_frame(pts_cam, cam_T_world.float(), H.float(), R_P.float()).reshape(depth_mm.shape[0], -1, 3)
    valid = valid.reshape(depth_mm.shape[0], -1)
    lateral = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
    out = []
    for i in range(p.shape[0]):
        x = p[i, valid[i] & lateral[i] & (p[i, :, 0] > 0), 0]
        if len(x) < 20:
            out.append(float("nan")); continue
        xb = torch.quantile(x, 0.8)
        pl = p[i, valid[i] & lateral[i] & (p[i, :, 0] < xb - BOARD_BAND) & (p[i, :, 0] > PLUG_X[0]), 0]
        out.append(float(torch.quantile(pl, 0.99)) if len(pl) > 20 else float("nan"))
    return out


def _rotvec_to_matrix(v):
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / th
    Kx = torch.zeros(v.shape[0], 3, 3, device=v.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = th.sin()[..., None], th.cos()[..., None]
    return torch.eye(3, device=v.device)[None] + s * Kx + (1 - c) * Kx @ Kx


class PackedTheta(Packed):
    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        self.theta = np.load(Path(root) / "theta.npy")

    def __getitem__(self, i):
        return (*super().__getitem__(i), np.float32(self.theta[self.idx[i]]), np.int64(self.small["grasp"][self.idx[i]]))


def compose(theta_deg, H, R_P, frame):
    """Relative rotation (rotvec, rad) from the plug angle, FK hand pose and port axes."""
    M = np.asarray(frame["M_ref"]) @ Rot.from_rotvec(np.asarray(frame["axis"]) * np.radians(theta_deg)).as_matrix()
    return Rot.from_matrix(R_P.T @ H[:3, :3] @ M).as_rotvec()


@torch.no_grad()
def predict(model, loader, dev="cuda"):
    out = {"pred": [], "theta": [], "grasp": [], "u": [], "H": [], "R": []}
    model.eval()
    for b in loader:
        rgb, depth, K, T, H, R, u, _, th, g = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out["pred"].append(model(rgb, depth, K.float(), T.float(), H.float(), R.float()).float().cpu())
        for k, v in (("theta", th), ("grasp", g), ("u", u), ("H", H), ("R", R)):
            out[k].append(v.float().cpu() if k != "grasp" else v.cpu())
    model.train()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def report(res, frame=None, tag=""):
    e = res["pred"] - res["theta"]
    gs = np.unique(res["grasp"])
    ge = np.array([np.median(res["pred"][res["grasp"] == g]) - res["theta"][res["grasp"] == g][0] for g in gs])
    r = {f"{tag}view_med_deg": float(np.median(np.abs(e))), f"{tag}view_bias_deg": float(e.mean()),
         f"{tag}grasp_med_deg": float(np.median(np.abs(ge))), f"{tag}grasp_p90_deg": float(np.percentile(np.abs(ge), 90))}
    if frame is not None:  # composed relative rotation with the episode (grasp) median, vs the label
        gm = {g: np.median(res["pred"][res["grasp"] == g]) for g in gs}
        rv = np.array([compose(gm[g], H, R, frame) for g, H, R in zip(res["grasp"], res["H"], res["R"])])
        err = np.degrees(rv - res["u"][:, 3:] * np.radians(1.0))
        r[f"{tag}rot_err_med_deg"] = float(np.median(np.linalg.norm(err, axis=1)))
        r[f"{tag}r_y_bias_deg"] = float(err[:, 1].mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", nargs="+", required=True)
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    ap.add_argument("--monitor-packed", default="data/packed_v6_abcd")
    ap.add_argument("--frame", default="runs_v7/plug_frame.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dino", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=0.0, help="rotation augmentation range (+-deg) of the plug points")
    ap.add_argument("--no-fingers", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)
    frame = json.loads(Path(args.frame).read_text())
    tr = ConcatDataset([PackedTheta(p, drop_ports=args.val_ports) for p in args.packed])
    va = ConcatDataset([v for v in (PackedTheta(p, keep_ports=args.val_ports) for p in args.packed) if len(v)])
    mon = PackedTheta(args.monitor_packed)
    mk = lambda ds, sh: DataLoader(ds, args.bs, shuffle=sh, num_workers=args.workers, drop_last=sh,
                                   persistent_workers=True)
    tl, vl, ml = mk(tr, True), mk(va, False), mk(mon, False)
    print(f"train {len(tr)} val {len(va)} monitor {len(mon)}", flush=True)
    model = PlugNet(use_dino=args.dino, no_fingers=args.no_fingers).to(dev)
    (out / "plugnet_kw.json").write_text(json.dumps({"use_dino": args.dino, "no_fingers": args.no_fingers}))
    axis_h = torch.as_tensor(np.asarray(frame["M_ref"]) @ np.asarray(frame["axis"]), device=dev, dtype=torch.float32)
    (out / "plug_frame.json").write_text(json.dumps(frame))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(k, total) / total)))
    t0, hist = time.time(), []
    for ep in range(args.epochs):
        tot = 0.0
        for b in tl:
            rgb, depth, K, T, H, R, u, _, th, g = to_dev(b, dev)
            delta = None
            if args.aug_deg > 0:
                delta = (torch.rand(len(th), device=dev) * 2 - 1) * args.aug_deg
                delta = delta * (torch.rand(len(th), device=dev) < 0.8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(rgb, depth, K.float(), T.float(), H.float(), R.float(),
                             aug_axis=axis_h if delta is not None else None,
                             aug_rad=torch.deg2rad(delta) if delta is not None else None)
            target = th.float() + (delta if delta is not None else 0.0)
            loss = nn.functional.smooth_l1_loss(pred.float() / THETA_SCALE, target / THETA_SCALE, beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        res = {"epoch": ep, "loss": tot / len(tl), "minutes": (time.time() - t0) / 60}
        res.update(report(predict(model, vl), frame, "val_"))
        if ep == args.epochs - 1 or ep % 3 == 2:
            res.update(report(predict(model, ml), frame, "MONITOR_"))
        hist.append(res)
        torch.save(model.state_dict(), out / "model.pt")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    (out / "summary.json").write_text(json.dumps({"history": hist, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
