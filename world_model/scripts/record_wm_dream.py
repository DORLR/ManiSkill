"""Fully imagined rollouts: the world model and the FM policy run on their own, from one real frame.

Start: the hand is placed at an evaluation start and one real wrist frame is encoded. From then on
nothing real is fed back: the policy acts on imagined latents, the world model predicts the next
latent, and so on (0.4 s per step). Each imagined latent is decoded by the VAE decoder into plug and
port points (the world model's latents have no image decoder, so points are what it "renders").
For comparison the same joint targets are replayed open loop in the simulator, and the real frames
are encoded and decoded the same way (never shown to the world model).

Panel: real wrist camera | decoded points, real (top) vs imagined (bottom), two views | distance plot.

    CUDA_VISIBLE_DEVICES=1 python world_model/scripts/record_wm_dream.py --vae ... --policy ... --wm ... \
        --keys G105_combined_0:0 G46_train_0:0 --out $Q/runs/videos_wm_dream
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

from wm.data.datasets import CONTEXT  # noqa: E402
from wm.env.expert import Expert  # noqa: E402
from wm.env.robot import Robot  # noqa: E402
from wm.eval import rollout  # noqa: E402
from wm.eval.agent import EXECUTE  # noqa: E402
from wm.eval.planner import WorldModelGuidedAgent  # noqa: E402
from wm.models.policy import ACTION_DIM, CHUNK  # noqa: E402

DT = 0.4


def fig_to_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB"))


def points_panel(real, dream, lim) -> np.ndarray:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6), dpi=100)
    for r, (name, d, col) in enumerate([("real (encoded, decoded)", real, "tab:blue"),
                                        ("imagined by the world model", dream, "tab:orange")]):
        for c, (j, lab) in enumerate([(1, "lateral y (mm)"), (2, "vertical z (mm)")]):
            ax = axes[r, c]
            ax.scatter(d["port"][:, 0], d["port"][:, j], s=1, c="0.6")
            ax.scatter(d["plug"][:, 0], d["plug"][:, j], s=2, c=col)
            ax.set_xlim(lim[0][0], lim[1][0])
            ax.set_ylim(lim[0][j], lim[1][j])
            ax.set_title(f"{name}: x-{lab[0]}", fontsize=8)
            ax.tick_params(labelsize=6)
            if r == 1:
                ax.set_xlabel("insertion axis x (mm)", fontsize=7)
            ax.set_ylabel(lab, fontsize=7)
    fig.tight_layout()
    return fig_to_array(fig)


def distance_panel(log, k_now) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.8, 5.6), dpi=100)
    t = np.arange(len(log)) * DT
    ax.plot(t, [e["truth"] for e in log], color="0.6", ls="--", label="truth (sim)")
    ax.plot(t, [e["real_mm"] for e in log], color="tab:blue", label="read from real latent")
    ax.plot(t, [e["dream_mm"] for e in log], color="tab:orange", label="read from imagined latent")
    ax.axvline(k_now * DT, color="k", lw=0.5)
    ax.set_xlim(0, len(log) * DT)
    ax.set_ylim(0, max(40, max(e["truth"] for e in log) + 5))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("plug distance to inserted (mm)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig_to_array(fig)


@torch.no_grad()
def dream(agent, robot, expert, g, start_i, steps):
    if not expert.move_to_offset(g.starts[start_i]):
        return None
    f = robot.frame()
    if not agent.reset(f, g.seed + start_i):
        return None
    z = agent.encode(f)                                            # the only real input
    ctx = [z] * CONTEXT
    q_imag = torch.as_tensor(f.qpos[:7], dtype=torch.float32, device=agent.device)
    log = []
    for k in range(steps):
        c = torch.stack(ctx)[None]
        a = (agent.policy.sample(c, steps=agent.flow_steps, n=1)[0] * agent.std + agent.mean).view(CHUNK, ACTION_DIM)
        z_next = agent._predict(c, a[None, :EXECUTE])[0]
        ctx = ctx[1:] + [z_next]
        dec_d = agent.vae.decode(z_next[None] / agent.scale)
        targets = (a[:EXECUTE] + q_imag).cpu().numpy()
        q_imag = dec_d["joints"][0].float()                        # imagined joints after the step
        for q in targets:                                          # same targets replayed in the simulator
            robot._command(q, 1)
        fr = robot.frame()
        saved = agent.plug_points
        z_real = agent.encode(fr)                                  # for comparison only
        agent.plug_points = saved                                  # the distance uses the first frame's points
        dec_r = agent.vae.decode(z_real[None] / agent.scale)
        pts = lambda d, key: d[key][0].float().cpu().numpy() * 1000
        log.append({"rgb": fr.rgb, "truth": rollout._depth_left_m(robot) * 1000,
                    "real": {"plug": pts(dec_r, "plug_points"), "port": pts(dec_r, "port_points")},
                    "dream": {"plug": pts(dec_d, "plug_points"), "port": pts(dec_d, "port_points")},
                    "real_mm": float(agent.remaining_mm(z_real[None])[0]),
                    "dream_mm": float(agent.remaining_mm(z_next[None])[0])})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--keys", nargs="+", required=True, help="grasp_key:start_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="ood_val")
    ap.add_argument("--steps", type=int, default=40, help="imagined steps (0.4 s each)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {g.key: g for g in rollout.plan(args.split, 2, 2)}
    agent = WorldModelGuidedAgent(args.vae, args.policy, args.wm, 1)
    robot, port = None, None
    for spec in args.keys:
        key, i = spec.split(":")
        g, i = plan[key], int(i)
        if port != g.port:
            if robot is not None:
                robot.env.close()
            robot, port = Robot(g.port), g.port
        expert = Expert(robot)
        robot.reset(g.seed, g.pitch, g.grasp_pos)
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            print(key, "expert could not insert; skipped", flush=True)
            continue
        torch.manual_seed(g.seed + i)
        log = dream(agent, robot, expert, g, i, args.steps)
        if not log:
            print(key, "placement or board fit failed", flush=True)
            continue
        allp = np.concatenate([log[0]["real"]["port"], log[0]["real"]["plug"]])
        lim = (allp.min(0) - 5, allp.max(0) + 5)
        frames = []
        for k, e in enumerate(log):
            panel = Image.new("RGB", (360 + 640 + 480, 560), "black")
            panel.paste(Image.fromarray(e["rgb"]).resize((360, 360)), (0, 100))
            panel.paste(Image.fromarray(points_panel(e["real"], e["dream"], lim)), (360, 0))
            panel.paste(Image.fromarray(distance_panel(log, k)), (1000, 0))
            ImageDraw.Draw(panel).text((6, 6), f"{key} start {i}  t={(k + 1) * DT:.1f}s\nimagined from one real "
                                       f"frame at t=0", fill=(255, 255, 0))
            frames.append(np.asarray(panel))
        name = out / f"dream_{key}_s{i}.mp4"
        imageio.mimsave(name, frames, fps=2.5)
        err = [abs(e["real_mm"] - e["dream_mm"]) for e in log]
        print(name.name, f"imagined vs real distance gap: after 2 s {err[min(4, len(err) - 1)]:.1f} mm, "
              f"after 8 s {err[min(19, len(err) - 1)]:.1f} mm, end {err[-1]:.1f} mm", flush=True)


if __name__ == "__main__":
    main()
