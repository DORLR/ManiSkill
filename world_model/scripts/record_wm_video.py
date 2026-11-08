"""Videos of the world-model-guided policy with the world model's predictions next to what happened.

Panel: wrist camera | side view | plot. Every 0.4 s the FM policy proposes N chunks and the world
model predicts, for each, how far the plug will still be from inserted 0.8 s later (read from the
predicted latent by the VAE decoder). The plot shows, over time:
    blue   remaining distance read from the real latent (what the encoder sees now)
    orange the world model's prediction for the executed candidate, drawn at the time it refers to
    band   spread (min-max) of the predictions over all N candidates
    grey   simulator truth (judging only)
If the world model is right, orange lands on blue 0.8 s later.
A fourth panel shows what the world model imagines, decoded to points (hand-centred port axes,
top view: insertion axis x vs lateral y): port points and plug points decoded from the real latent
now (grey / blue), and the plug points the world model predicted 0.8 s ago for now (orange).

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/record_wm_video.py --vae ... --policy ... --wm ... \
        --results $Q/runs/eval_wm_v1_ood_60s/results.jsonl --out $Q/runs/videos_wm_v1 --per-kind 1
"""

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from record_videos import pick, side_view  # noqa: E402
from wm.env.expert import Expert  # noqa: E402
from wm.env.robot import Robot  # noqa: E402
from wm.eval import rollout  # noqa: E402
from wm.eval.planner import WorldModelGuidedAgent  # noqa: E402

DT = 0.4          # seconds per decision
HORIZON = 2       # decisions ahead the cost refers to (0.8 s)


class Logged:
    """Wraps the agent; after each decision stores the world model's predictions and the truth."""

    def __init__(self, agent, robot):
        self.agent, self.robot, self.log = agent, robot, []

    def reset(self, f, seed=0):
        self.log.clear()
        return self.agent.reset(f, seed)

    def act(self, f, n_samples=1):
        a = self.agent.act(f, n_samples)
        d = self.agent.debug
        with torch.no_grad():
            dec = self.agent.vae.decode(torch.stack([d["z_now"], d["z_pred"]]) / self.agent.scale)
        pts = lambda k, i: dec[k][i].float().cpu().numpy() * 1000
        self.log.append({"t": len(self.log) * DT, "now": d["now_mm"], "cost": d["cost_mm"], "chosen": d["chosen"],
                         "truth": rollout._depth_left_m(self.robot) * 1000,
                         "plug_now": pts("plug_points", 0), "port_now": pts("port_points", 0),
                         "plug_pred": pts("plug_points", 1)})
        return a


def plot(log, t_now, width=480, height=360) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    t = np.array([e["t"] for e in log])
    ax.plot(t, [e["truth"] for e in log], color="0.6", ls="--", lw=1, label="truth (sim)")
    ax.plot(t, [e["now"] for e in log], color="tab:blue", lw=1.5, label="read from real latent")
    tp = t + HORIZON * DT
    c = [e["cost"] for e in log]
    ax.fill_between(tp, [x.min() for x in c], [x.max() for x in c], color="tab:orange", alpha=0.2,
                    label="WM, all candidates")
    ax.plot(tp, [x[e["chosen"]] for x, e in zip(c, log)], color="tab:orange", marker=".", lw=1,
            label="WM, executed (+0.8 s)")
    ax.axvline(t_now, color="k", lw=0.5)
    ax.set_xlim(0, max(10.0, t_now + 1))
    ax.set_ylim(0, max(40.0, max(e["truth"] for e in log) + 5))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("plug distance to inserted (mm)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB"))


def imagined(log, width=420, height=360) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    e = log[-1]
    ax.scatter(e["port_now"][:, 0], e["port_now"][:, 1], s=2, c="0.6", label="port (real latent)")
    ax.scatter(e["plug_now"][:, 0], e["plug_now"][:, 1], s=3, c="tab:blue", label="plug now (real latent)")
    if len(log) > HORIZON:
        p = log[-1 - HORIZON]["plug_pred"]
        ax.scatter(p[:, 0], p[:, 1], s=3, c="tab:orange", alpha=0.6, label="plug, WM prediction from 0.8 s ago")
    allp = np.concatenate([log[0]["port_now"], log[0]["plug_now"]])     # fixed view from the first decision
    lo, hi = allp.min(0) - 5, allp.max(0) + 5
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("insertion axis x (mm)")
    ax.set_ylabel("lateral y (mm)")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title("decoded latents, top view", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="ood_val")
    ap.add_argument("--per-kind", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--every", type=int, default=2, help="record every n-th control step (20 Hz control)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {g.key: g for g in rollout.plan(args.split, 2, 2)}
    agent = WorldModelGuidedAgent(args.vae, args.policy, args.wm, args.candidates)
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
        torch.manual_seed(g.seed + i)                      # reproducible sampling (not the evaluation's)
        logged = Logged(agent, robot)
        frames, n = [], [0]

        def on_step(r):
            n[0] += 1
            if n[0] % args.every or not logged.log:
                return
            panel = Image.new("RGB", (360 + 480 + 480 + 420, 360), "black")
            panel.paste(Image.fromarray(r.frame().rgb).resize((360, 360)), (0, 0))
            side = side_view(r)
            if side is not None:
                panel.paste(Image.fromarray(side[..., :3].astype(np.uint8)).resize((480, 360)), (360, 0))
            panel.paste(Image.fromarray(plot(logged.log, n[0] / 20)), (840, 0))
            panel.paste(Image.fromarray(imagined(logged.log)), (1320, 0))
            ImageDraw.Draw(panel).text((6, 6), f"{key} start {i}  t={n[0] / 20:.1f}s", fill=(255, 255, 0))
            frames.append(np.asarray(panel))

        res = rollout.episode(robot, expert, logged, g.starts[i], g.seed + i, on_step=on_step)
        name = out / f"wm_{kind}_{key}_s{i}_{res['outcome']}.mp4"
        imageio.mimsave(name, frames, fps=10)
        err = [abs(logged.log[k + HORIZON]["now"] - logged.log[k]["cost"][logged.log[k]["chosen"]])
               for k in range(len(logged.log) - HORIZON)]
        print(name.name, res, f"WM 0.8 s prediction error vs real latent: median {np.median(err):.2f} mm", flush=True)


if __name__ == "__main__":
    main()
