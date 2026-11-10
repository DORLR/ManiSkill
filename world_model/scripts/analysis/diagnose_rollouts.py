"""Why does the latent FM stop beside the hole?  Replays evaluation episodes exactly (seeded) and logs,
at every decision, what the autoencoder reads from the frame vs the simulator truth (truth used only
for this diagnosis):

    read   the plug's remaining motion solved from the frame's latent (models/vae.py solve_pose), in
           hand-centred port axes: translation along the insertion axis (depth) and across it
           (lateral, mm), rotation angle (deg)
    truth  the charger pose in the socket-goal frame: depth left, lateral offset, angle (rollout._diagnosis)
    policy the commanded hand motion of the executed chunk (mm), and the hand's actual motion

If, when stuck, `read` says "lateral ~0" while truth says ~5 mm, perception misplaces the hole; if `read`
says ~5 mm and the policy still pushes straight, the policy does not act on what it sees.

    python world_model/scripts/analysis/diagnose_rollouts.py --vae $Q/runs/vae_v3g/best.pt \
        --policy $Q/runs/policy_v3g/best.pt --results $Q/runs/eval_policy_v3g_ood/results.jsonl \
        --out $Q/runs/checks_v3g/diagnose_policy --stuck 12 --success 6 --part 0 --parts 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wm.env.expert import Expert
from wm.env.robot import Robot
from wm.eval import rollout
from wm.eval.agent import LatentPolicyAgent, ResidualAgent, frame_batch


class Logged:
    def __init__(self, agent: LatentPolicyAgent, robot):
        self.agent, self.robot, self.log = agent, robot, []

    def reset(self, f, seed=0):
        self.log.clear()
        return self.agent.reset(f, seed)

    @torch.no_grad()
    def act(self, f, n_samples=1):
        a = self.agent.act(f, n_samples)
        z = self.agent.context[-1] / self.agent.scale
        geom = {k: v for k, v in frame_batch(f, self.agent.R_P, self.agent.device).items()
                if k in ("K", "cam_T_world", "hand", "R_P")}
        R, t = self.agent.vae.decode_pose(z[None], geom)
        t = t[0].cpu().numpy() * 1000
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R[0].cpu().numpy()) - 1) / 2, -1, 1))))
        truth = rollout._diagnosis(self.robot)
        R_P = self.agent.R_P
        cmd = (a[0][7] if len(a[0]) > 7 else a[0][-1])                 # joint target 0.4 s ahead (for scale only)
        self.log.append({"read_depth_mm": float(t[0]), "read_lateral_mm": float(np.hypot(t[1], t[2])),
                         "read_lat_y": float(t[1]), "read_lat_z": float(t[2]), "read_angle_deg": ang,
                         "true_depth_left_mm": truth["depth_left_mm"], "true_lateral_mm": truth["lateral_mm"],
                         "true_angle_deg": truth["angle_deg"],
                         "hand_port_mm": (R_P.T @ f.hand[:3, 3] * 1000).round(2).tolist(),
                         "joint_step_mrad": float(np.abs(cmd - f.qpos[:7]).mean() * 1000)})
        return a


def pick(results: Path, n_stuck: int, n_success: int, beside: int = 0) -> list[tuple[str, int, str]]:
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    if beside:                     # episodes that reached depth beside the board (fixed criterion)
        b = [(r["key"], i, "beside") for r in rows for i, e in enumerate(r["episodes"]) if e["outcome"] == "beside_board"]
        rng = np.random.default_rng(0)
        return [b[i] for i in rng.choice(len(b), size=min(beside, len(b)), replace=False)]
    stuck = [(r["key"], i, "stuck") for r in rows for i, e in enumerate(r["episodes"])
             if e["outcome"] == "timeout" and e["depth_left_mm"] > 15 and e["lateral_mm"] > 3]
    ok = [(r["key"], i, "success") for r in rows for i, e in enumerate(r["episodes"]) if e["outcome"] == "inserted"]
    rng = np.random.default_rng(0)
    take = lambda v, n: [v[i] for i in rng.choice(len(v), size=min(n, len(v)), replace=False)]
    return take(stuck, n_stuck) + take(ok, n_success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stuck", type=int, default=12)
    ap.add_argument("--success", type=int, default=6)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--residual", help="diagnose FM + this residual")
    ap.add_argument("--beside", type=int, default=0, help="pick this many beside-the-board episodes instead")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from wm.training.common import limit_gpu_memory
    limit_gpu_memory()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {g.key: g for g in rollout.plan("ood_val", 4, 2, args.seed)}
    items = sorted(pick(Path(args.results), args.stuck, args.success, args.beside))[args.part::args.parts]
    agent = ResidualAgent(args.vae, args.policy, args.residual) if args.residual else LatentPolicyAgent(args.vae, args.policy)
    robot, port = None, None
    for key, i, kind in items:
        g = plan[key]
        if port != g.port:
            if robot is not None:
                robot.env.close()
            robot, port = Robot(g.port), g.port
        expert = Expert(robot)
        robot.reset(g.seed, g.pitch, g.grasp_pos)
        if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
            continue
        logged = Logged(agent, robot)
        res = rollout.episode(robot, expert, logged, g.starts[i], g.seed + i)
        rec = {"key": key, "start": i, "kind": kind, "port": g.port, "condition": g.condition, "result": res, "log": logged.log}
        (out / f"{key}_s{i}.json").write_text(json.dumps(rec))
        last = logged.log[-1]
        print(key, i, kind, res["outcome"], "| end: read lateral %.2f depth %.2f mm | true lateral %.2f depth left %.2f mm"
              % (last["read_lateral_mm"], last["read_depth_mm"], last["true_lateral_mm"], last["true_depth_left_mm"]), flush=True)


if __name__ == "__main__":
    main()
