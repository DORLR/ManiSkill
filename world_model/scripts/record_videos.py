"""Re-run selected evaluation episodes and save videos (wrist camera | side view), same grasps
and starts as the evaluation. Picks episodes by outcome from an evaluation results file.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/record_videos.py --vae ... --policy ... \
        --results $Q/runs/eval_policy_v1_ood_60s/results.jsonl --out $Q/runs/videos_policy_v1 --per-kind 2
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from wm.eval import rollout
from wm.eval.agent import LatentPolicyAgent
from wm.env.expert import Expert
from wm.env.robot import Robot


def pick(results: Path, per_kind: int) -> list[tuple[str, int, str]]:
    """(grasp key, start index, kind): successes, far misses (> 15 mm left), near misses (< 5 mm left)."""
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    kinds = {"success": [], "far_miss": [], "near_miss": []}
    for r in rows:
        for i, e in enumerate(r["episodes"]):
            if e["outcome"] == "inserted":
                kinds["success"].append((r["key"], i, "success"))
            elif e["outcome"] == "timeout" and e.get("depth_left_mm", 0) > 15:
                kinds["far_miss"].append((r["key"], i, "far_miss"))
            elif e["outcome"] == "timeout" and e.get("depth_left_mm", 99) < 5:
                kinds["near_miss"].append((r["key"], i, "near_miss"))
    rng = np.random.default_rng(0)
    out = []
    for v in kinds.values():
        if v:
            out += [v[j] for j in rng.choice(len(v), size=min(per_kind, len(v)), replace=False)]
    return out


def side_view(robot) -> np.ndarray | None:
    try:
        img = robot.base.render_rgb_array()
        img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
        return img[0] if img.ndim == 4 else img
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="ood_val")
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--every", type=int, default=2, help="record every n-th control step (20 Hz control)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {g.key: g for g in rollout.plan(args.split, 2, 2)}
    agent = LatentPolicyAgent(args.vae, args.policy)
    robot, port = None, None
    for key, i, kind in pick(Path(args.results), args.per_kind):
        g = plan[key]
        if port != g.port:
            if robot is not None:
                robot.env.close()
            robot, port = Robot(g.port), g.port
        expert = Expert(robot)
        robot.reset(g.seed, g.pitch, g.grasp_pos)
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            continue
        frames, n = [], [0]

        def on_step(r):
            n[0] += 1
            if n[0] % args.every:
                return
            wrist = Image.fromarray(r.frame().rgb).resize((360, 360))
            side = side_view(r)
            panel = Image.new("RGB", (360 + (480 if side is not None else 0), 360), "black")
            panel.paste(wrist, (0, 0))
            if side is not None:
                panel.paste(Image.fromarray(side[..., :3].astype(np.uint8)).resize((480, 360)), (360, 0))
            ImageDraw.Draw(panel).text((6, 6), f"{key} start {i}  t={n[0] / 20:.1f}s", fill=(255, 255, 0))
            frames.append(np.asarray(panel))

        res = rollout.episode(robot, expert, agent, g.starts[i], g.seed + i, on_step=on_step)
        name = out / f"{kind}_{key}_s{i}_{res['outcome']}.mp4"
        imageio.mimsave(name, frames, fps=10)
        print(name.name, res, flush=True)


if __name__ == "__main__":
    main()
