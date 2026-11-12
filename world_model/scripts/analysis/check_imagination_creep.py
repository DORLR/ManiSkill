"""How does the hand move when the FM acts inside imagination, compared with the simulator, from the same state?

    python world_model/scripts/analysis/check_imagination_creep.py --config <online.json> --episodes <dir> \
        --wm <wm.pt> --success <success.pt> --out <report.json> [--n 128]

Recorded simulator continuations that ran the full 40 decisions without inserting (timeouts; first repetition of
each branch) are started again in imagination at their branch point and the FM acts for 40 decisions. Per decision:
the measured hand's displacement since the branch point, in port axes (mm), imagined (joint adapter) vs real, and
the classifier's P. For reference, the real displacement of the continuations that inserted (branch point to end).
If imagination inserts because the hand creeps forward a little at every decision while the real hand stays, the
imagined displacement along the insertion axis grows steadily and the real one does not.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wm.analysis import imagination_checks4 as ic
from wm.data.kinematics import build_fk
from wm.imagination import FMSampler
from wm.imagination4 import HistoryImagination
from wm.training.adapter import load_adapter
from wm.training.common import limit_gpu_memory
from wm.training.success import load_success
from wm.training.wm4 import load_history_wm

ap = argparse.ArgumentParser()
for a in ("--config", "--episodes", "--wm", "--success", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--n", type=int, default=128)
args = ap.parse_args()
limit_gpu_memory()
torch.manual_seed(0)
np.random.seed(0)
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
a = cfg["actor"]
scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
fk, robot = build_fk()
fm = FMSampler(a["policy"], dev)
imag = HistoryImagination(load_history_wm(args.wm, dev).requires_grad_(False), load_adapter(a["adapter"], dev).requires_grad_(False),
                          fk, load_success(args.success, dev).requires_grad_(False), scale, dev)
eps = [ic.load(p) for p in sorted(Path(args.episodes).glob("*.npz"))]


def port_disp(ep, hands):
    """(n, 4, 4) hand poses -> displacement from the branch point in port axes, mm."""
    ref = ep["hand"][int(ep["img_step"][ep["dec_img"][ep["k_branch"]]])][:3, 3]
    return (hands[:, :3, 3] - ref) @ ep["R_P"].astype(np.float64) * 1000


stuck = [ep for ep in eps if ep["rep"] == 0 and ep["outcome"] == "timeout" and len(ep["dec_img"]) - ep["k_branch"] >= ic.HORIZON]
stuck = [stuck[i] for i in np.random.default_rng(0).choice(len(stuck), size=min(args.n, len(stuck)), replace=False)]
ins = [ep for ep in eps if ep["outcome"] == "inserted"]
ins_disp = np.stack([port_disp(ep, ep["hand"][[-1]])[0] for ep in ins])
print(len(stuck), "stuck continuations;", len(ins), "inserted for reference", flush=True)

T = ic.HORIZON
imag_d, real_d, p_all = np.zeros((len(stuck), T, 3)), np.zeros((len(stuck), T, 3)), np.zeros((len(stuck), T))
dec = np.full((len(stuck), T), np.nan)            # the outcome decision given to an outcome-conditioned adapter
for s0 in range(0, len(stuck), 64):
    chunk = stuck[s0:s0 + 64]
    for ep in chunk:
        ep["cmd_img"] = ic.commanded_hands(fk, ep)
    imag.reset_from([ic.start_at(ep, ep["k_branch"], ep["cmd_img"], scale) for ep in chunk])
    for t in range(T):
        o = imag.obs()
        info = imag.step(fm(o["z"], o["q"]).cpu().numpy().astype(np.float64) + imag.q[:, None, :7])
        for b, ep in enumerate(chunk):
            imag_d[s0 + b, t] = port_disp(ep, info["hand"][b][None])[0]
            j = min(int(ep["dec_img"][ep["k_branch"] + t]) + 2, len(ep["img_step"]) - 1)        # 0.4 s after decision t
            real_d[s0 + b, t] = port_disp(ep, ep["hand"][[int(ep["img_step"][j])]])[0]
            p_all[s0 + b, t] = info["p_success"][b]
            if getattr(imag.adapter, "takes_outcome", False):
                dec[s0 + b, t] = imag.decision[b]

med = lambda x: [round(float(v), 2) for v in np.median(x, 0)]
rep = {"adapter": a["adapter"], "stuck_continuations": len(stuck),
       "inserted_reference_displacement_mm_port_axes_median": med(ins_disp),
       "inserted_reference_displacement_mm_port_axes_p10_p90": [[round(float(v), 2) for v in np.percentile(ins_disp, q, 0)] for q in (10, 90)],
       "by_decision": {}}
for t in (0, 4, 9, 19, 29, 39):
    rep["by_decision"][f"+{(t + 1) * 0.4:g}s"] = {
        "imagined_displacement_mm_median": med(imag_d[:, t]), "real_displacement_mm_median": med(real_d[:, t]),
        "imagined_called_inserted_so_far_pct": round(100 * float((p_all[:, :t + 1].max(1) > 0.5).mean()), 1),
        "decided_goes_in_pct": None if np.isnan(dec[:, t]).all() else round(100 * float(dec[:, t].mean()), 1)}
Path(args.out).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
robot.env.close()
