"""Imagination vs simulator on the FM policy's own trajectories (wm/analysis/imagination_checks.py).

    # 1. simulator rollouts with branch points (CPU workers, each with the autoencoder + FM on GPU)
    WM_GPU_MEM_FRACTION=0.045 python world_model/scripts/analysis/check_imagination.py record \
        --vae $Q/runs/vae_v3g/best.pt --policy $Q/runs/policy_v3g/best.pt \
        --out $Q/runs/checks_v3g/imagination/episodes --workers 10
    # 2. readout / open-loop prediction / closed-loop outcome against those rollouts (GPU)
    python world_model/scripts/analysis/check_imagination.py compare --episodes $Q/runs/checks_v3g/imagination/episodes \
        --vae ... --policy ... --wm $Q/runs/wm_v3g_s_best_final.pt --adapter $Q/runs/adapter_v3/best.pt \
        --latents $Q/data/wm_v3g_latents --out $Q/runs/checks_v3g/imagination/report.json
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wm.analysis import imagination_checks as ic
from wm.eval import rollout


def record(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [g for g in rollout.plan("ood_val", 4, 2, 0) if g.key.endswith("_0") and (not args.only or g.port in args.only)]
    items = [(g, i, str(out)) for g in plan for i in range(len(g.starts))]
    if args.reverse:                       # a second process from the other end (finished episodes are skipped)
        items = items[::-1]
    if args.offset:                        # a further process starting in the middle
        items = items[args.offset:] + items[:args.offset]
    print(f"{len(items)} starts x {len(ic.BRANCHES)} branches x {ic.REPS} continuations, {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=ic._init, initargs=(args.vae, args.policy)) as pool:
        for n, msgs in enumerate(pool.imap_unordered(ic.record_start, items)):
            print(f"[{n + 1}/{len(items)} {time.time() - t0:.0f}s] " + " | ".join(msgs), flush=True)


def compare(args):
    from wm.data.kinematics import build_fk
    from wm.imagination import FMSampler, ImaginedInsertion
    from wm.models.vae import load_vae
    from wm.training.adapter import load_adapter
    from wm.training.common import limit_gpu_memory, seed_all
    from wm.training.wm import load_world_model
    limit_gpu_memory()
    seed_all(0)
    dev = torch.device("cuda")
    scale = json.loads((Path(args.latents) / "stats.json").read_text())["scale"]
    fm = FMSampler(args.policy, dev)
    assert abs(fm.scale - scale) < 1e-6, (fm.scale, scale)
    fk, robot = build_fk()
    env = ImaginedInsertion(load_vae(args.vae, dev).requires_grad_(False), load_world_model(args.wm, dev).requires_grad_(False),
                            load_adapter(args.adapter, dev).requires_grad_(False), fk, scale, dev,
                            sampling_steps=args.sampling_steps, success_mm=ic.SUCCESS_MM, wm_samples=1)
    cmp = ic.Comparison(env, fm, batch=args.batch)

    eps = [{**ic._load(p), "name": p.stem} for p in sorted(Path(args.episodes).glob("*.npz"))]
    if args.max_episodes:
        eps = eps[:args.max_episodes]
    for ep in eps:
        ep["hand_cmd"] = ic._hand_cmd(fk, ep)
        ep["reading"] = cmp.real_reading(ep, Path(args.episodes) / "readings" / (ep["name"] + ".npz"))
        n = len(ep["step"])
        ep["truth"] = {k: np.concatenate([ep[k], [ep["end_" + k]]]) for k in ("depth_left_mm", "lateral_mm", "angle_deg")}
        assert len(ep["reading"]["distance_mm"]) == n + 1
    print(f"{len(eps)} episodes, {sum(len(e['step']) for e in eps)} decisions", flush=True)
    report = {"episodes": len(eps), "settings": {"branches": ic.BRANCHES, "horizon": ic.HORIZON, "reps": ic.REPS,
              "imagined_per_branch": args.imagined, "success_mm": ic.SUCCESS_MM, "sampling_steps": args.sampling_steps}}

    secs = set(args.sections.split(","))
    # ------------------------------------------------------------ 1. readout of real frames vs truth
    rd = defaultdict(lambda: defaultdict(list))
    fin = defaultdict(int)
    for ep in eps:
        first = 0 if ep["rep"] == 0 else ep["k_branch"]          # prefixes are identical across continuations
        for j in range(first, len(ep["step"]) + 1):
            st = ic.true_state(ep["truth"]["depth_left_mm"][j], ep["truth"]["lateral_mm"][j])
            r = {k: v[j] for k, v in ep["reading"].items()}
            rd[st]["depth_err_mm"].append(abs(r["depth_mm"] - ep["truth"]["depth_left_mm"][j]))
            rd[st]["lateral_err_mm"].append(abs(r["lateral_mm"] - ep["truth"]["lateral_mm"][j]))
            rd[st]["true_lateral_mm"].append(ep["truth"]["lateral_mm"][j])
            rd[st]["read_lateral_mm"].append(r["lateral_mm"])
        reads_in = bool(ep["reading"]["distance_mm"][-1] < ic.SUCCESS_MM)
        fin[f"{ep['outcome']}|reading_inserted={reads_in}"] += 1
    if "readout" in secs:
        report["1_readout"] = {st: {k: ic.stats(v) for k, v in d.items()} for st, d in rd.items()}
        report["1_readout_final_frame"] = dict(fin)
    print("readout done", flush=True)

    # ------------------------------------------------------------ 2a. open loop from the branch points
    rep0 = [ep for ep in eps if ep["rep"] == 0]
    items = [(ep, ep["k_branch"], ic.HORIZON) for ep in rep0]
    ol = {}
    for given in ((False, True) if "open" in secs else ()):
        preds = cmp.open_loop(items, [ep["hand_cmd"] for ep, _, _ in items], ic.HORIZON, given)
        acc = defaultdict(lambda: defaultdict(list))
        for (ep, k, _), p in zip(items, preds):
            n = len(ep["step"])
            for t in ic.PRED_STEPS:
                j = k + t
                if j > n:
                    continue
                real = {key: v[j] for key, v in ep["reading"].items()}
                now = {key: v[k] for key, v in ep["reading"].items()}
                h = f"+{t * 0.4:g}s"
                for key in ("distance_mm", "depth_mm", "lateral_mm"):
                    acc[h][key + "_err"].append(abs(p[t - 1][key] - real[key]))
                    acc[h][key + "_err_nothing_changes"].append(abs(now[key] - real[key]))
                tk = {"depth_mm": "depth_left_mm", "lateral_mm": "lateral_mm"}
                for key, tr in tk.items():                         # world model + readout vs simulator truth
                    acc[h][key + "_err_vs_truth"].append(abs(p[t - 1][key] - ep["truth"][tr][j]))
                    acc[h][key + "_err_vs_truth_real_frame"].append(abs(real[key] - ep["truth"][tr][j]))
                hand_j = ep["hand"][int(ep["step"][j])] if j < n else ep["hand"][-1]
                acc[h]["hand_pos_err_mm"].append(float(np.linalg.norm(p[t - 1]["hand_pos"] - hand_j[:3, 3]) * 1000))
        ol["real_joints" if given else "adapter_joints"] = {h: {k: ic.stats(v) for k, v in d.items()} for h, d in acc.items()}
    if ol:
        report["2a_open_loop_from_branch_points"] = ol
    print("open loop (branch points) done", flush=True)

    # ------------------------------------------------------------ 2b. 0.4 s ahead from every decision
    items = [(ep, k, 1) for ep in eps for k in range(0 if ep["rep"] == 0 else ep["k_branch"], len(ep["step"]))]
    one = {}
    for given in ((False, True) if "one" in secs else ()):
        preds = cmp.open_loop(items, [ep["hand_cmd"] for ep, _, _ in items], 1, given)
        acc = defaultdict(lambda: defaultdict(list))
        for (ep, k, _), p in zip(items, preds):
            j = k + 1
            st = ic.true_state(ep["truth"]["depth_left_mm"][j], ep["truth"]["lateral_mm"][j])
            real = {key: v[j] for key, v in ep["reading"].items()}
            now = {key: v[k] for key, v in ep["reading"].items()}
            for key in ("distance_mm", "depth_mm", "lateral_mm"):
                acc[st][key + "_err"].append(abs(p[0][key] - real[key]))
                acc[st][key + "_err_nothing_changes"].append(abs(now[key] - real[key]))
                acc[st][key + "_real_change"].append(abs(real[key] - now[key]))
            hand_j = ep["hand"][int(ep["step"][j])] if j < len(ep["step"]) else ep["hand"][-1]
            acc[st]["hand_pos_err_mm"].append(float(np.linalg.norm(p[0]["hand_pos"] - hand_j[:3, 3]) * 1000))
        one["real_joints" if given else "adapter_joints"] = {st: {k: ic.stats(v) for k, v in d.items()} for st, d in acc.items()}
    if one:
        report["2b_one_step_by_true_state"] = one
    print("open loop (every decision) done", flush=True)

    # ------------------------------------------------------------ 3. closed loop: imagined vs simulated outcome
    if "closed" not in secs:
        Path(args.out).write_text(json.dumps(report, indent=1))
        robot.env.close()
        return
    groups = defaultdict(list)
    for ep in eps:
        if args.branches and ep["k_branch"] not in args.branches:
            continue
        groups[(ep["key"], ep["start"], ep["k_branch"])].append(ep)
    branch_rows, starts = [], []
    for (key, s, k), g in sorted(groups.items()):
        g = sorted(g, key=lambda e: e["rep"])
        base = g[0]
        qb = np.stack([e["q_branch"] for e in g])
        R_P = base["R_P"]
        sim = []
        for e in g:
            n = len(e["step"])
            hb, he = e["hand"][int(e["step"][k])], e["hand"][-1]
            lat = float(np.linalg.norm((R_P.T @ (he[:3, 3] - hb[:3, 3]))[1:]) * 1000)
            sim.append({"outcome": e["outcome"], "reading_inserted": bool((e["reading"]["distance_mm"][k + 1:] < ic.SUCCESS_MM).any()),
                        "lateral_travel_mm": lat})
        branch_rows.append({"key": key, "start": s, "k": k, "condition": base["condition"], "port": base["port"],
                            "prefix_joint_spread_rad": float(np.abs(qb - qb[0]).max()), "sim": sim,
                            "start_reading_mm": float(base["reading"]["distance_mm"][k]),
                            "start_truth": {kk: float(base["truth"][kk][k]) for kk in ("depth_left_mm", "lateral_mm")}})
        starts += [ic.start_state(base, k, base["hand_cmd"])] * args.imagined
    imag = cmp.closed_loop(starts, ic.HORIZON)
    for b, row in enumerate(branch_rows):
        row["imagined"] = imag[b * args.imagined:(b + 1) * args.imagined]
    (Path(args.out).with_suffix(".branches.json")).write_text(json.dumps(branch_rows))

    def summary(rows):
        sim = [x for r in rows for x in r["sim"]]
        im = [x for r in rows for x in r["imagined"]]
        rate = lambda v, f: round(100 * float(np.mean([f(x) for x in v])), 1) if v else float("nan")
        sim_rate = np.array([np.mean([x["outcome"] == "inserted" for x in r["sim"]]) for r in rows])
        im_rate = np.array([np.mean([x["inserted"] for x in r["imagined"]]) for r in rows])
        pos = [im_rate[i] for i, r in enumerate(rows) for x in r["sim"] if x["outcome"] == "inserted"]
        neg = [im_rate[i] for i, r in enumerate(rows) for x in r["sim"] if x["outcome"] != "inserted"]
        bad = [i for i, r in enumerate(rows) if np.mean([x["outcome"] in ("beside_board", "tilted") for x in r["sim"]]) >= 0.5]
        return {"branches": len(rows), "sim_inserted_pct": rate(sim, lambda x: x["outcome"] == "inserted"),
                "sim_beside_or_tilted_pct": rate(sim, lambda x: x["outcome"] in ("beside_board", "tilted")),
                "sim_reading_inserted_pct": rate(sim, lambda x: x["reading_inserted"]),
                "imagined_inserted_pct": rate(im, lambda x: x["inserted"]),
                "auc_imagined_rate_for_sim_outcome": round(ic.auc(pos, neg), 3),
                "corr_imagined_vs_sim_rate": round(float(np.corrcoef(im_rate, sim_rate)[0, 1]), 3) if len(rows) > 2 and im_rate.std() > 0 and sim_rate.std() > 0 else float("nan"),
                "mean_abs_rate_diff_pct": round(100 * float(np.abs(im_rate - sim_rate).mean()), 1),
                "imagined_inserted_pct_where_sim_mostly_beside_or_tilted": round(100 * float(im_rate[bad].mean()), 1) if bad else float("nan"),
                "n_branches_sim_mostly_beside_or_tilted": len(bad),
                "lateral_travel_mm_sim": ic.stats([x["lateral_travel_mm"] for x in sim]),
                "lateral_travel_mm_imagined": ic.stats([x["lateral_travel_mm"] for x in im])}

    report["3_closed_loop"] = {"all": summary(branch_rows),
                               **{f"branch_k{k}": summary([r for r in branch_rows if r["k"] == k]) for k in ic.BRANCHES if any(r["k"] == k for r in branch_rows)},
                               **{f"cond_{c}": summary([r for r in branch_rows if r["condition"] == c])
                                  for c in sorted({r["condition"] for r in branch_rows})}}
    report["prefix_joint_spread_rad_max"] = max(r["prefix_joint_spread_rad"] for r in branch_rows)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["3_closed_loop"]["all"], indent=1))
    robot.env.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--vae", required=True)
    r.add_argument("--policy", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--only", nargs="+", help="smoke tests only: these shapes")
    r.add_argument("--reverse", action="store_true", help="process the starts from the last one")
    r.add_argument("--offset", type=int, default=0, help="process the starts from this one on")
    c = sub.add_parser("compare")
    for a in ("--episodes", "--vae", "--policy", "--wm", "--adapter", "--latents", "--out"):
        c.add_argument(a, required=True)
    c.add_argument("--imagined", type=int, default=8, help="imagined continuations per branch")
    c.add_argument("--sampling-steps", type=int, default=20)
    c.add_argument("--batch", type=int, default=64)
    c.add_argument("--max-episodes", type=int, help="smoke tests only")
    c.add_argument("--sections", default="readout,open,one,closed",
                   help="comma list of readout / open / one / closed, to run parts in parallel processes")
    c.add_argument("--branches", type=int, nargs="+", help="closed loop: only these branch points")
    args = ap.parse_args()
    record(args) if args.cmd == "record" else compare(args)


if __name__ == "__main__":
    main()
