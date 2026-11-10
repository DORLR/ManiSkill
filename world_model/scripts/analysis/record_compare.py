"""Side-by-side videos of the same evaluation episode (same grasp, start and seed): FM alone | FM +
residual. Each half: wrist camera (top) and a side view (bottom), with time, and the simulator's depth
left / lateral offset as a caption (truth, for viewing only).

Episodes are picked from two evaluation results of the same seed:
    gained   FM timed out, FM + residual inserted
    lost     FM inserted, FM + residual timed out (both-extrapolated condition first)
    drift    FM + residual timed out far from the hole (largest lateral offset)

    python world_model/scripts/analysis/record_compare.py --vae ... --policy ... --residual ... \
        --fm-results $Q/runs/eval_policy_v3g_ood/results.jsonl \
        --res-results $Q/runs/eval_residual_v3g_ood/results.jsonl --out $Q/runs/videos_v3g_residual --per-kind 2
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from wm.env.expert import Expert
from wm.env.robot import Robot
from wm.eval import rollout
from wm.eval.agent import LatentPolicyAgent, ResidualAgent

W, H_WRIST, H_SIDE = 360, 360, 270


def load(p):
    return {json.loads(l)["key"]: json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()}


def pick(fm, res, per_kind):
    pairs = [(k, i, fm[k]["episodes"][i], res[k]["episodes"][i], fm[k]["condition"]) for k in fm
             if k in res and fm[k].get("expert_ok") and res[k].get("expert_ok")
             for i in range(min(len(fm[k]["episodes"]), len(res[k]["episodes"])))]
    ins = lambda e: e["outcome"] == "inserted"
    gained = [p for p in pairs if not ins(p[2]) and ins(p[3])]
    lost = sorted([p for p in pairs if ins(p[2]) and not ins(p[3])], key=lambda p: p[4] != "combined")
    drift = sorted([p for p in pairs if not ins(p[3]) and p[3].get("lateral_mm", 0) > 10],
                   key=lambda p: -p[3]["lateral_mm"])
    rng = np.random.default_rng(0)
    take = lambda v, n: [v[i] for i in sorted(rng.choice(len(v), size=min(n, len(v)), replace=False))] if v else []
    return [("gained", p) for p in take(gained, per_kind)] + [("lost", p) for p in lost[:per_kind]] + \
           [("drift", p) for p in drift[:per_kind]]


def side_view(robot):
    try:
        img = robot.base.render_rgb_array()
        img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
        return (img[0] if img.ndim == 4 else img)[..., :3].astype(np.uint8)
    except Exception:
        return np.zeros((H_SIDE, W, 3), np.uint8)


def record(robot, expert, agent, g, i, label, every):
    frames, n = [], [0]

    def on_step(r):
        n[0] += 1
        if n[0] % every:
            return
        d = rollout._diagnosis(r)
        panel = Image.new("RGB", (W, H_WRIST + H_SIDE + 40), "black")
        panel.paste(Image.fromarray(r.frame().rgb).resize((W, H_WRIST)), (0, 40))
        panel.paste(Image.fromarray(side_view(r)).resize((W, H_SIDE)), (0, 40 + H_WRIST))
        ImageDraw.Draw(panel).text((6, 4), f"{label}  t={n[0] / 20:.1f}s\ndepth left {d['depth_left_mm']:.1f} mm  "
                                   f"lateral {d['lateral_mm']:.1f} mm", fill=(255, 255, 0))
        frames.append(np.asarray(panel))

    robot.reset(g.seed, g.pitch, g.grasp_pos)
    if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
        return None, frames
    res = rollout.episode(robot, expert, agent, g.starts[i], g.seed + i, on_step=on_step)
    return res, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--fm-results", required=True)
    ap.add_argument("--res-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--every", type=int, default=2, help="record every n-th control step (20 Hz)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {g.key: g for g in rollout.plan("ood_val", 4, 2, args.seed)}
    items = pick(load(args.fm_results), load(args.res_results), args.per_kind)[args.part::args.parts]
    fm = LatentPolicyAgent(args.vae, args.policy)
    rs = ResidualAgent(args.vae, args.policy, args.residual)
    robot, port = None, None
    for kind, (key, i, efm, eres, cond) in items:
        g = plan[key]
        if port != g.port:
            if robot is not None:
                robot.env.close()
            robot, port = Robot(g.port), g.port
        expert = Expert(robot)
        r1, f1 = record(robot, expert, fm, g, i, "FM", args.every)
        r2, f2 = record(robot, expert, rs, g, i, "FM + residual (trained in WM)", args.every)
        if r1 is None or r2 is None:
            continue
        n = max(len(f1), len(f2))
        pad = lambda f: f + [f[-1]] * (n - len(f))
        f1, f2 = pad(f1), pad(f2)
        frames = [np.concatenate([a, np.zeros((a.shape[0], 8, 3), np.uint8), b], 1) for a, b in zip(f1, f2)]
        name = out / f"{kind}_{key}_s{i}_FM-{r1['outcome']}_RES-{r2['outcome']}.mp4"
        imageio.mimsave(name, frames, fps=10)
        print(name.name, "| recorded in eval:", efm["outcome"], eres["outcome"], "| replay:", r1["outcome"], r2["outcome"], flush=True)


if __name__ == "__main__":
    main()
