"""v4 imagination vs simulator on the FM policy's own trajectories (wm/analysis/imagination_checks4.py).

    # 1. simulator rollouts with branch points (needs only the FM and perception)
    WM_GPU_MEM_FRACTION=0.03 python world_model/scripts/analysis/check_imagination4.py record \
        --config world_model/configs/v4/online.json --out $Q/runs/checks_v4/imagination/episodes --workers 6
    # 2. comparison with a world-model / classifier checkpoint
    python world_model/scripts/analysis/check_imagination4.py compare --config world_model/configs/v4/online.json \
        --episodes $Q/runs/checks_v4/imagination/episodes --wm $Q/runs/wm_v4_online/current.pt \
        --success $Q/runs/success_v4_online/current.pt --out $Q/runs/checks_v4/imagination/report_<tag>.json
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wm.analysis import imagination_checks4 as ic
from wm.eval import rollout


def record(args, cfg):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [g for g in rollout.plan("ood_val", 4, 2, 0) if g.key.endswith("_0") and (not args.only or g.port in args.only)]
    items = [(g, i, str(out)) for g in plan for i in range(len(g.starts))]
    if args.reverse:
        items = items[::-1]
    a = cfg["actor"]
    icfg = {"unet": a["unet"], "ae": a["ae"], "policy": a["policy"], "latents": cfg["latents"]}
    print(f"{len(items)} starts, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=ic._init, initargs=(icfg,)) as pool:
        for n, msgs in enumerate(pool.imap_unordered(ic.record_start, items)):
            print(f"[{n + 1}/{len(items)} {time.time() - t0:.0f}s] " + " | ".join(msgs), flush=True)


@torch.no_grad()
def compare(args, cfg):
    from wm.data.kinematics import build_fk
    from wm.geometry import plug_region_points, pose_loss_terms
    from wm.imagination import FMSampler
    from wm.imagination4 import HistoryImagination
    from wm.models.crop_ae import load_crop_ae
    from wm.training.adapter import load_adapter
    from wm.training.common import limit_gpu_memory
    from wm.training.success import load_success
    from wm.training.wm4 import load_history_wm
    limit_gpu_memory()
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda")
    a = cfg["actor"]
    scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
    fm = FMSampler(a["policy"], dev)
    ae = load_crop_ae(a["ae"], dev).requires_grad_(False)
    fk, robot = build_fk()
    wm = load_history_wm(args.wm, dev).requires_grad_(False)
    succ = load_success(args.success, dev).requires_grad_(False)
    wm_step = torch.load(args.wm, map_location="cpu", weights_only=False).get("step")
    imag = HistoryImagination(wm, load_adapter(a["adapter"], dev).requires_grad_(False), fk, succ, scale, dev,
                              args.sampling_steps)
    ref = plug_region_points(dev)
    eps = [ic.load(p) for p in sorted(Path(args.episodes).glob("*.npz"))]
    for ep in eps:
        ep["cmd_img"] = ic.commanded_hands(fk, ep)
    report = {"world_model": args.wm, "world_model_step": wm_step, "success": args.success, "episodes": len(eps),
              "settings": {"branches": ic.BRANCHES, "horizon": ic.HORIZON, "reps": ic.REPS, "imagined": args.imagined}}
    print(len(eps), "episodes; world model step", wm_step, flush=True)
    B = args.batch
    horizon = int(round(torch.load(args.success, map_location="cpu", weights_only=False)["cfg"].get("label_horizon_s", 0.0) / 0.2))
    report["classifier_label_horizon_images"] = horizon

    # ------------------------------------------------ 1. classifier on real histories
    # every decision image (prefixes once) + the final image of every episode, taken after the outcome
    # (an episode ends at insertion, so "inserted" frames are mostly final images)
    items = [(ep, int(ep["dec_img"][k])) for ep in eps for k in range(0 if ep["rep"] == 0 else ep["k_branch"], len(ep["dec_img"]))]
    items += [(ep, len(ep["img_step"]) - 1) for ep in eps]
    probs, truths = [], []
    for s0 in range(0, len(items), B):
        chunk = items[s0:s0 + B]
        imag.reset_from([ic.start_at_image(ep, j, ep["cmd_img"], scale) for ep, j in chunk])
        probs += imag.p_success.tolist()
        truths += [ic.truth_at(ep, j, horizon) for ep, j in chunk]
    p = np.array(probs)
    ins = np.array([t["inserted_within"] for t in truths])       # the classifier's label: inserted within its horizon
    at_depth_wrong = np.array([(t["depth_left_mm"] <= 1.0) and not t["inserted"] for t in truths])
    near = np.array([t["depth_left_mm"] <= 5.0 for t in truths])
    final = np.array([j == len(ep["img_step"]) - 1 for ep, j in items])
    outcome_final = [ep["outcome"] for ep, j in items if j == len(ep["img_step"]) - 1]
    pf = p[final]
    report["1_classifier_final_frames"] = {o: {"episodes": int(sum(x == o for x in outcome_final)),
                                               "called_inserted_pct": round(100 * float(np.mean([q > 0.5 for q, x in zip(pf, outcome_final) if x == o])), 1)}
                                           for o in sorted(set(outcome_final))}
    report["1_classifier"] = {
        "frames": int(len(p)), "positive_frames_inserted_within_horizon": int(ins.sum()),
        "auc": round(ic.auc(p[ins], p[~ins]), 4),
        "auc_within_5mm": round(ic.auc(p[ins & near], p[~ins & near]), 4),
        "precision_at_0.5": round(float(((p > 0.5) & ins).sum() / max(1, (p > 0.5).sum())), 4),
        "recall_at_0.5": round(float(((p > 0.5) & ins).sum() / max(1, ins.sum())), 4),
        "at_depth_beside_or_tilted_frames": int(at_depth_wrong.sum()),
        "at_depth_beside_or_tilted_called_inserted": round(float((p[at_depth_wrong] > 0.5).mean()), 4) if at_depth_wrong.any() else None}
    print("classifier", report["1_classifier"], flush=True)

    # ------------------------------------------------ 2. open loop with the executed commands
    def read(z, ep, js):
        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        geom = {"K": T(np.repeat(ep["K"][None], len(js), 0)), "cam_T_world": T(ep["cam_T_world"][js]),
                "hand": T(ep["hand"][ep["img_step"][js].astype(int)]), "R_P": T(np.repeat(ep["R_P"][None], len(js), 0)),
                "plug_uv0": T(ep["plug_uv0"][js], torch.long), "hole_uv0": T(ep["hole_uv0"][js], torch.long)}
        return ae.decode_pose(z / scale, geom)

    rep0 = [ep for ep in eps if ep["rep"] == 0]
    ol = {}
    for given in (False, True):
        acc = defaultdict(lambda: defaultdict(list))
        for s0 in range(0, len(rep0), B):
            chunk = rep0[s0:s0 + B]
            imag.reset_from([ic.start_at(ep, ep["k_branch"], ep["cmd_img"], scale) for ep in chunk])
            for t in range(ic.HORIZON):
                targets, qs = [], []
                for ep in chunk:
                    kk = min(ep["k_branch"] + t, len(ep["dec_img"]) - 1)
                    s = int(ep["img_step"][ep["dec_img"][kk]])
                    act = ep["action"][s:s + 9]
                    act = np.concatenate([act, np.repeat(ep["action"][-1:], 16 - len(act), 0)])
                    targets.append(act)
                    q = ep["qpos"][s + 1:s + 9, :7]
                    qs.append(np.concatenate([q, np.repeat(ep["qpos"][-1:, :7], 8 - len(q), 0)]))
                info = imag.step(np.stack(targets), np.stack(qs) if given else None)
                if t + 1 not in ic.PRED_STEPS:
                    continue
                for b, ep in enumerate(chunk):
                    j_now = int(ep["dec_img"][ep["k_branch"]])
                    j = j_now + 2 * (t + 1)
                    if j >= len(ep["img_step"]):
                        continue
                    zr = torch.from_numpy(ep["mean"][[j]] * scale).to(dev)
                    zn = torch.from_numpy(ep["mean"][[j_now]] * scale).to(dev)
                    Rp, tp = read(info["z"][b:b + 1], ep, [j])
                    Rr, tr = read(zr, ep, [j])
                    Rn, tn = read(zn, ep, [j])
                    g, _ = pose_loss_terms(Rp, tp, Rr, tr, ref)
                    gn, _ = pose_loss_terms(Rn, tn, Rr, tr, ref)
                    h = f"+{(t + 1) * 0.4:g}s"
                    acc[h]["pred_vs_real_mm"].append(float(g[0]))
                    acc[h]["nothing_changes_mm"].append(float(gn[0]))
        ol["real_joints" if given else "adapter_joints"] = {h: {k: ic.stats(v) for k, v in d.items()} for h, d in acc.items()}
    report["2_open_loop"] = ol
    print("open loop", json.dumps(ol["adapter_joints"]), flush=True)

    # ------------------------------------------------ 3. closed loop: imagined vs simulated outcome
    groups = defaultdict(list)
    for ep in eps:
        groups[(ep["key"], ep["start"], ep["k_branch"])].append(ep)
    rows, starts = [], []
    for (key, s, k), g in sorted(groups.items()):
        g = sorted(g, key=lambda e: e["rep"])
        base = g[0]
        rows.append({"key": key, "start": s, "k": k, "condition": base["condition"],
                     "sim": [e["outcome"] for e in g],
                     "prefix_joint_spread_rad": float(np.abs(np.stack([e["q_branch"] for e in g]) - g[0]["q_branch"]).max())})
        starts += [ic.start_at(base, k, base["cmd_img"], scale)] * args.imagined
    imagined = []
    for s0 in range(0, len(starts), B):
        chunk = starts[s0:s0 + B]
        imag.reset_from(chunk)
        best = imag.p_success.copy()
        for _ in range(ic.HORIZON):
            o = imag.obs()
            rel = fm(o["z"], o["q"]).cpu().numpy().astype(np.float64)
            info = imag.step(rel + imag.q[:, None, :7])
            best = np.maximum(best, info["p_success"])
        imagined += best.tolist()
    for b, r in enumerate(rows):
        r["imagined_max_p"] = imagined[b * args.imagined:(b + 1) * args.imagined]

    def summary(rs):
        sim = [o for r in rs for o in r["sim"]]
        sim_rate = np.array([np.mean([o == "inserted" for o in r["sim"]]) for r in rs])
        im_rate = np.array([np.mean(np.array(r["imagined_max_p"]) > 0.5) for r in rs])
        pos = [im_rate[i] for i, r in enumerate(rs) for o in r["sim"] if o == "inserted"]
        neg = [im_rate[i] for i, r in enumerate(rs) for o in r["sim"] if o != "inserted"]
        bad = [i for i, r in enumerate(rs) if np.mean([o in ("beside_board", "tilted") for o in r["sim"]]) >= 0.5]
        return {"branches": len(rs), "sim_inserted_pct": round(100 * float(np.mean([o == "inserted" for o in sim])), 1),
                "sim_beside_or_tilted_pct": round(100 * float(np.mean([o in ("beside_board", "tilted") for o in sim])), 1),
                "imagined_inserted_pct": round(100 * float(im_rate.mean()), 1),
                "mean_abs_rate_diff_pct": round(100 * float(np.abs(im_rate - sim_rate).mean()), 1),
                "auc_imagined_rate_for_sim_outcome": round(ic.auc(pos, neg), 3),
                "n_branches_sim_mostly_beside_or_tilted": len(bad),
                "imagined_inserted_pct_there": round(100 * float(im_rate[bad].mean()), 1) if bad else None}

    report["3_closed_loop"] = {"all": summary(rows), **{f"k{k}": summary([r for r in rows if r["k"] == k]) for k in ic.BRANCHES
                                                        if any(r["k"] == k for r in rows)}}
    report["prefix_joint_spread_rad_max"] = max(r["prefix_joint_spread_rad"] for r in rows)
    Path(args.out).write_text(json.dumps(report, indent=1))
    Path(args.out).with_suffix(".branches.json").write_text(json.dumps(rows))
    print(json.dumps(report["3_closed_loop"], indent=1))
    robot.env.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--only", nargs="+", help="smoke tests only: these shapes")
    r.add_argument("--reverse", action="store_true")
    c = sub.add_parser("compare")
    for x in ("--config", "--episodes", "--wm", "--success", "--out"):
        c.add_argument(x, required=True)
    c.add_argument("--imagined", type=int, default=8)
    c.add_argument("--sampling-steps", type=int, default=20)
    c.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    record(args, cfg) if args.cmd == "record" else compare(args, cfg)


if __name__ == "__main__":
    main()
