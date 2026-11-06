"""SID motion field fitted to the demonstrated approach states (Sec. III-B of SID).

python scripts/fit_sid_field.py --states data/v4_demo_states.npz --rel runs_v4/rel --out runs_v4/fieldrel_sid
X_demo  relative states u of the demonstration frames (scripts/demo_states.py) of
        the training shapes, pins not yet in the holes (u_x < GAP0 / sigma);
        u comes from the logged joint angles only
E(u)    = 1/2 d(u, X_demo)^2, d = min_j ||u - u_j|| (sigma units: 1 mm, 1 deg)
v*(u)   = u*(u) - u, u* = nearest demonstrated state (eta(d) = d; the executed
          step is clipped to STEP_MAX by paf.field.step_from_field)
Training states: uniform over the bounded domain Omega (virtual, no robot
motion) plus states near X_demo for precision where the field vanishes; loss
||f(u) - v*(u)||^2. The output directory holds the field and the relational
network so that paf.policy.FieldPolicy(estimate="rel") can load both.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from paf import config as C
from paf.field import FieldNet

LO = np.array([-20.0, -12, -12, -8, -8, -8])
HI = np.array([3.0, 12, 12, 8, 8, 8])


def nearest_target(u, X, chunk=4096):
    out = torch.empty_like(u)
    for i in range(0, len(u), chunk):
        j = torch.cdist(u[i:i + chunk], X).argmin(1)
        out[i:i + chunk] = X[j] - u[i:i + chunk]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/v4_demo_states.npz")
    ap.add_argument("--rel", default="runs_v4/rel")
    ap.add_argument("--out", default="runs_v4/fieldrel_sid")
    ap.add_argument("--val-ports", nargs="+", default=["G46", "G47", "G48"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--tx-range", type=float, nargs=2, default=None,
                    help="keep demo states with t_x in this range (sigma units), e.g. -2.5 0: the part of the "
                         "demonstrations where the relational net has dense training views")
    ap.add_argument("--core", type=float, default=1.0,
                    help="keep this fraction of the demo states with the highest local density (k-NN radius)")
    ap.add_argument("--handoff-box", action="store_true",
                    help="target = the IL's hand-over region (t_x -2.5..-1.5 mm, lateral +-0.5 mm, rotations "
                         "+-0.5 deg), sampled uniformly, instead of the demonstration states")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.states)
    keep = (d["u"][:, 0] < C.GAP0 / C.SIGMA_T) & ~np.isin(d["port"], args.val_ports)
    if args.tx_range:
        keep &= (d["u"][:, 0] >= args.tx_range[0]) & (d["u"][:, 0] <= args.tx_range[1])
    X = torch.as_tensor(d["u"][keep], device="cuda")
    if args.core < 1.0:  # density core: drop the sparsest states (largest distance to the 16th neighbour)
        r = torch.cat([torch.cdist(X[i:i + 4096], X).topk(17, largest=False).values[:, -1]
                       for i in range(0, len(X), 4096)])
        X = X[r <= r.quantile(args.core)]
    if args.handoff_box:
        gb = torch.Generator(device="cuda").manual_seed(1)
        lo_b = torch.tensor([-2.5, -0.5, -0.5, -0.5, -0.5, -0.5], device="cuda")
        hi_b = torch.tensor([-1.5, 0.5, 0.5, 0.5, 0.5, 0.5], device="cuda")
        X = lo_b + torch.rand(20000, 6, device="cuda", generator=gb) * (hi_b - lo_b)
    print(f"X_demo: {len(X)} states from {len(np.unique(d['port'][keep]))} training shapes", flush=True)
    lo, hi = (torch.as_tensor(a, dtype=torch.float32, device="cuda") for a in (LO, HI))
    g = torch.Generator(device="cuda").manual_seed(0)
    net = FieldNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for it in range(args.steps):
        a = lo + torch.rand(3072, 6, device="cuda", generator=g) * (hi - lo)
        b = X[torch.randint(len(X), (1024,), device="cuda", generator=g)] + \
            torch.randn(1024, 6, device="cuda", generator=g) * 1.0
        u = torch.cat([a, b])
        v = nearest_target(u, X)
        loss = ((net(u) - v) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 2000 == 0:
            print(f"step {it} loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        u = lo + torch.rand(8192, 6, device="cuda", generator=g) * (hi - lo)
        v = nearest_target(u, X)
        err = (net(u) - v).norm(dim=1)
        un = X[torch.randint(len(X), (4096,), device="cuda", generator=g)]
        on_manifold = net(un).norm(dim=1)
    fit = {"rmse_domain": float(err.pow(2).mean().sqrt()), "mean_target_norm": float(v.norm(dim=1).mean()),
           "on_manifold_output_median": float(on_manifold.median()),
           "on_manifold_output_p90": float(on_manifold.quantile(0.9)), "n_demo_states": int(len(X)),
           "domain_lo": LO.tolist(), "domain_hi": HI.tolist()}
    print(json.dumps(fit), flush=True)
    torch.save(net.state_dict(), out / "field.pt")
    shutil.copy(f"{args.rel}/model.pt", out / "model.pt")
    (out / "summary.json").write_text(json.dumps({"fit": fit, "args": vars(args)}, indent=1))


if __name__ == "__main__":
    main()
