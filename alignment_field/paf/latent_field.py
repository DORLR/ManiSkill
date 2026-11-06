"""Latent field: slide into the demonstrations in the IL policy's own feature space.

No hand-designed state. "In distribution" is judged where the IL decides: its
feature vector c = encode(RGB crop, 7 joint angles) (paf.fm_policy, frozen).
  bank     c of the demonstration frames (training shapes, pins not yet in
           the holes), standardised and projected on their top PCA_DIM
           principal components -> z
  energy   E(z) = mean distance to the K_NN nearest bank points
  model    h(z_i, a) -> (z_j, E_j): what the IL will see after the hand moves by
           a (hand frame, sigma units, FK only), learned from pairs of views
           of the same grasp (scripts/collect_interaction.py)
  field    a*(z) = argmin_a E_hat(z, a) over sampled motions; the executed step
           is a* clipped to STEP_MAX; stop when E(z) is below the level of
           held-out demonstration frames (two views in a row) or after
           LOOP_MAX_ITERS, then hand over to the IL.
The method sees only the RGB crop, joint angles and FK hand pose.

python -m paf.latent_field --policy runs_v4/policy --out runs_v4/latent_fm
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from . import config as C
from .model import mlp
from .perception import rgb_crop

PCA_DIM, K_NN = 64, 8
A_T, A_R = 15.0, 8.0          # sampled hand motions: +-15 mm, +-8 deg per axis
N_CAND = 4096


def hand_motion(H_i, H_j):
    """Motion from pose i to pose j in the hand frame of i (sigma units)."""
    t = H_i[:3, :3].T @ (H_j[:3, 3] - H_i[:3, 3])
    r = Rotation.from_matrix(H_i[:3, :3].T @ H_j[:3, :3]).as_rotvec()
    return np.r_[t / C.SIGMA_T, r / C.SIGMA_R]


def apply_hand(H, a):
    T = H.copy()
    T[:3, 3] = H[:3, 3] + H[:3, :3] @ (a[:3] * C.SIGMA_T)
    T[:3, :3] = H[:3, :3] @ Rotation.from_rotvec(a[3:] * C.SIGMA_R).as_matrix()
    return T


@torch.no_grad()
def encode(model, rgb, qpos, bs=1024, dev="cuda"):
    out = []
    for i in range(0, len(rgb), bs):
        r = torch.as_tensor(rgb[i:i + bs], device=dev)
        q = torch.as_tensor(qpos[i:i + bs], device=dev, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.encode(r, q).float())
    return torch.cat(out)


class Bank:
    def __init__(self, c_demo=None, state=None, dev="cuda"):
        if state is None:
            mu, sd = c_demo.mean(0), c_demo.std(0) + 1e-4
            x = (c_demo - mu) / sd
            _, _, V = torch.linalg.svd(x - x.mean(0), full_matrices=False)
            state = {"mu": mu, "sd": sd, "P": V[:PCA_DIM].T.contiguous()}
            state["Z"] = self.project_with(state, c_demo)
        self.s = {k: v.to(dev) for k, v in state.items()}

    @staticmethod
    def project_with(s, c):
        return ((c - s["mu"]) / s["sd"]) @ s["P"]

    def z(self, c):
        return self.project_with(self.s, c)

    def energy(self, z, chunk=4096):
        out = []
        for i in range(0, len(z), chunk):
            out.append(torch.cdist(z[i:i + chunk], self.s["Z"]).topk(K_NN, largest=False).values.mean(1))
        return torch.cat(out)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp((PCA_DIM + 6, 768, 768, 768, PCA_DIM + 1))

    def forward(self, z, a):
        o = self.net(torch.cat([z, a / torch.tensor([A_T] * 3 + [A_R] * 3, device=a.device)], -1))
        return z + o[:, :PCA_DIM], o[:, PCA_DIM]


class LatentField:
    """Closed-loop use: needs the IL policy object (for its encoder)."""

    def __init__(self, run_dir, il_model, dev="cuda", seed=0):
        st = torch.load(f"{run_dir}/latent.pt", map_location=dev)
        self.bank = Bank(state=st["bank"], dev=dev)
        self.dyn = Dynamics().to(dev).eval()
        self.dyn.load_state_dict(st["dyn"])
        self.tau = st["tau"]
        self.il = il_model
        self.dev = dev
        self.g = torch.Generator(device=dev).manual_seed(seed)

    @torch.no_grad()
    def z_of(self, obs):
        c = encode(self.il, rgb_crop(obs)[None], obs.qpos[:7][None].astype(np.float32), dev=self.dev)
        return self.bank.z(c)

    @torch.no_grad()
    def step(self, obs):
        z = self.z_of(obs)
        E = float(self.bank.energy(z)[0])
        a = (torch.rand(N_CAND, 6, device=self.dev, generator=self.g) * 2 - 1) * \
            torch.tensor([A_T] * 3 + [A_R] * 3, device=self.dev)
        a = torch.cat([torch.zeros(1, 6, device=self.dev), a])
        _, e_hat = self.dyn(z.expand(len(a), -1), a)
        k = int(e_hat.argmin())
        best = a[k].cpu().numpy()
        n = np.linalg.norm(best)
        step = best * min(1.0, C.STEP_MAX / max(n, 1e-9))
        return {"E": E, "E_hat_best": float(e_hat[k]), "E_hat_stay": float(e_hat[0]), "a_best": best, "step": step}

    def run(self, robot, max_iters=None):
        traj, inside = [], 0
        for it in range(max_iters or C.LOOP_MAX_ITERS):
            obs = robot.observe()
            s = self.step(obs)
            traj.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()})
            inside = inside + 1 if s["E"] < self.tau else 0
            if inside >= 2 or np.linalg.norm(s["step"]) < 0.5:
                break
            if not robot.move_to(apply_hand(obs.H, s["step"])):
                traj[-1]["ik_fail"] = True
                break
        return traj


# ----------------------------------------------------------------- training
def _pairs(groups, n, rng, max_motion=40.0):
    """Random ordered pairs of views of the same grasp."""
    out = []
    for idx, H in groups:
        m = len(idx)
        i, j = rng.integers(m, size=(2, n))
        for a, b in zip(i, j):
            mot = hand_motion(H[a], H[b])
            if np.abs(mot).max() <= max_motion:
                out.append((idx[a], idx[b], mot))
    return out


def main():
    from .fm_policy import InsertPolicyNet
    from .il_data import load_demos
    from .ports_ext import generated_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos", default="data/v4_demos/*.npz")
    ap.add_argument("--states", default="data/v4_demo_states.npz")
    ap.add_argument("--interact", default="data/v4_interact")
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    cfg = json.loads(Path(f"{args.policy}/summary.json").read_text())["args"]
    il = InsertPolicyNet(head=cfg["head"], pretrained=False).to(dev).eval()
    il.load_state_dict(torch.load(f"{args.policy}/model.pt", map_location=dev))

    # bank: demonstration frames, pins not yet in the holes (same order as demo_states)
    ds = np.load(args.states)
    R, Q, _, _ = load_demos(args.demos, list(generated_specs()))
    assert len(R) == len(ds["u"]), "demo frames and demo states are not aligned"
    pre = ds["u"][:, 0] < C.GAP0 / C.SIGMA_T
    val = np.isin(ds["port"], args.val_ports)
    c_tr = encode(il, R[pre & ~val], Q[pre & ~val])
    c_va = encode(il, R[pre & val], Q[pre & val])
    del R, Q
    bank = Bank(c_tr)
    E_va = bank.energy(bank.z(c_va))
    tau = float(E_va.quantile(0.95))
    print(f"bank {len(c_tr)} frames; held-out demo energy median {float(E_va.median()):.2f}, tau(95%) {tau:.2f} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # interaction views -> z, E, and pairs within each grasp
    files = sorted(glob.glob(f"{args.interact}/*.npz"))
    views = {k: [] for k in ("z", "E", "u", "grasp", "diag")}
    groups_tr, groups_ho, groups_diag, off = [], [], [], 0
    rng = np.random.default_rng(0)
    for gi, f in enumerate(files):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if not meta.get("ok") or "rgb" not in d.files or len(d["rgb"]) < 4:
            continue
        z = bank.z(encode(il, d["rgb"], d["qpos"]))
        views["z"].append(z)
        views["E"].append(bank.energy(z))
        views["u"].append(d["u"])
        views["grasp"].append(np.full(len(z), gi))
        views["diag"].append(np.full(len(z), meta["diagnostic_only"]))
        g = (off + np.arange(len(z)), d["H"].astype(np.float64))
        (groups_diag if meta["diagnostic_only"] else groups_ho if rng.random() < 0.1 else groups_tr).append(g)
        off += len(z)
    Z = torch.cat(views["z"])
    EV = torch.cat(views["E"])
    U = np.concatenate(views["u"])
    print(f"interaction views {len(Z)} ({len(groups_tr)} train / {len(groups_ho)} held-out / "
          f"{len(groups_diag)} A-D diagnostic grasps) [{time.time() - t0:.0f}s]", flush=True)

    def to_t(pairs):
        i = torch.as_tensor([p[0] for p in pairs], device=dev)
        j = torch.as_tensor([p[1] for p in pairs], device=dev)
        a = torch.as_tensor(np.array([p[2] for p in pairs]), device=dev, dtype=torch.float32)
        return i, j, a
    tr = to_t(_pairs(groups_tr, 4000, rng))
    ho = to_t(_pairs(groups_ho, 2000, rng))
    dg = to_t(_pairs(groups_diag, 2000, rng))
    print(f"pairs: train {len(tr[0])}, held-out {len(ho[0])}, A-D {len(dg[0])}", flush=True)

    dyn = Dynamics().to(dev)
    opt = torch.optim.AdamW(dyn.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    zs = float(Z.std())
    for it in range(args.steps):
        b = torch.randint(len(tr[0]), (2048,), device=dev)
        i, j, a = tr[0][b], tr[1][b], tr[2][b]
        zj, ej = dyn(Z[i], a)
        loss = ((zj - Z[j]) ** 2).mean() / zs ** 2 + ((ej - EV[j]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 5000 == 0:
            print(f"step {it} loss {loss.item():.4f}", flush=True)

    @torch.no_grad()
    def evaluate(p):
        i, j, a = p
        _, e = dyn(Z[i], a)
        err = (e - EV[j]).abs()
        base = (EV[j] - EV[j].mean()).abs()
        r = np.corrcoef(e.cpu().numpy(), EV[j].cpu().numpy())[0, 1]
        return {"E_after_mae": float(err.mean()), "E_after_mae_const": float(base.mean()), "E_after_corr": float(r)}
    report = {"tau": tau, "bank": int(len(c_tr)), "E_heldout_demo_median": float(E_va.median()),
              "dyn_heldout": evaluate(ho), "dyn_abcd": evaluate(dg)}
    # does the latent energy see "out of distribution"? (u only for this check)
    Xd = torch.as_tensor(ds["u"][pre & ~val], device=dev)
    du = torch.cat([torch.cdist(torch.as_tensor(U[k:k + 4096], device=dev), Xd).min(1).values
                    for k in range(0, len(U), 4096)]).cpu().numpy()
    ev = EV.cpu().numpy()
    diag = np.concatenate(views["diag"]).astype(bool)
    from scipy.stats import spearmanr
    for name, m in (("train_shapes", ~diag), ("A-D", diag)):
        report[f"spearman_E_vs_u_distance_{name}"] = float(spearmanr(ev[m], du[m]).correlation)
        report[f"E_by_u_distance_{name}"] = {
            f"{lo}-{hi}": float(np.median(ev[m & (du >= lo) & (du < hi)])) if (m & (du >= lo) & (du < hi)).any() else None
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 99))}
    print(json.dumps(report, indent=1), flush=True)
    torch.save({"bank": {k: v.cpu() for k, v in bank.s.items()}, "dyn": dyn.state_dict(), "tau": tau},
               out / "latent.pt")
    (out / "summary.json").write_text(json.dumps({"report": report, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
