"""Does imagination agree with the simulator, on the policy's own trajectories? (ood_val shapes)

The world-model checks so far (wm_checks.py) used recorded data: expert insertions, perturbations
and contact trials. The FM policy visits other states (drift across the board, pushing beside the
hole). Here the FM is rolled out in the simulator and the world model is compared with what really
happened, at three levels.

Recording (simulator; record_start, parallel workers)
    starts      the evaluation plan (rollout.plan, seed 0), first grasp of every (shape, condition)
                cell, both starts
    branches    for each branch point k (decisions of 0.4 s: 20 / 40 / 60 = 8 / 16 / 24 s after placement)
                and continuation r = 0..R-1: reset, expert insertion, placement, the FM runs k
                decisions with the episode seed (identical every time: same physics, same seed; the
                joints at the branch point are saved to confirm it), then the policy seed is changed
                to continuation seed r and the FM runs up to H more decisions or until an outcome
                (fixed criterion of eval/rollout.py)
    saved       every control step: command, measured joints, hand pose; every decision: latent,
                camera pose, the 16 targets, simulator truth (depth left, offset, angle); plus a
                final frame's latent. Truth is used only for judging here.

Comparison (GPU; compare)
    1 readout     the pose read from real frames vs truth: what perception already gets wrong
    2 prediction  open loop with the commands the FM really executed, nothing real fed back:
                  from each branch point 0.4 / 2 / 4 / 10 / 16 s ahead (autoregressive 0.4 s steps), and
                  0.4 s ahead from every recorded decision, grouped by the true state of the target
                  frame (far / near the board / off the hole axis at the board).
                  Error = predicted reading vs the real frame's reading, next to "nothing changes"
                  (the reading of the last real frame). Measured joints from the adapter (as in
                  imagination) and, separately, the real ones (world model alone).
    3 outcome     closed loop: from each branch point the FM acts on imagined latents for H
                  decisions, N imagined continuations, against the R simulator continuations.
                  Imagined "inserted" = the reading falls below SUCCESS_MM. Reported: insertion rate
                  within H in the simulator (truth), in the simulator judged by the reading of the
                  real frames, and in imagination; per-branch agreement (AUC of the imagined rate
                  for the simulator outcome, correlation of the rates); lateral hand travel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.datasets import CONTEXT, HISTORY_STEPS, WM_STEP
from ..eval import rollout

BRANCHES = (20, 40, 60)         # decisions (0.4 s each): 8 / 16 / 24 s; FM insertions happen at 17-31 s
HORIZON = 40                    # decisions after the branch point (16 s, the longest checked shift)
REPS = 4                        # simulator continuations per branch
SUCCESS_MM = 1.5                # imagined "inserted": reading of the distance to inserted below this
PRED_STEPS = (1, 5, 10, 25, 40) # open-loop horizons (0.4, 2, 4, 10, 16 s)


# ====================================================================== recording (simulator)
def branch_episode(robot, expert, agent, g, i: int, k_branch: int, rep: int, horizon: int = HORIZON) -> dict:
    """One episode: k_branch decisions with the episode seed, then up to `horizon` with continuation
    seed `rep`. Returns arrays (see module doc) or {"skip": reason}."""
    from ..eval.agent import EXECUTE
    seed = g.seed + i
    robot.reset(g.seed, g.pitch, g.grasp_pos)
    if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
        return {"skip": "expert"}
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if not expert.move_to_offset(g.starts[i]):
        return {"skip": "placement"}
    f = robot.frame()
    if not agent.reset(f, seed):
        return {"skip": "no_board_plane"}
    qpos, hand, action = [f.qpos.copy()], [f.hand.copy()], []
    dec = {k: [] for k in ("z", "cam", "targets", "step", "depth_left_mm", "lateral_mm", "angle_deg")}
    outcome, q_branch = None, None
    for k in range(k_branch + horizon):
        if k == k_branch:
            q_branch = f.qpos.copy()
            cs = 7_000_000 + 1000 * seed + 10 * k_branch + rep        # continuation seed
            torch.manual_seed(cs)
            np.random.seed(cs % (2 ** 32))
        a = agent.act(f)[0]
        truth = rollout._diagnosis(robot)
        dec["z"].append(agent.context[-1].float().cpu().numpy())
        dec["cam"].append(f.cam_T_world)
        dec["targets"].append(a)
        dec["step"].append(len(action))
        for key in ("depth_left_mm", "lateral_mm", "angle_deg"):
            dec[key].append(truth[key])
        for q in a[:EXECUTE]:
            robot._command(q, 1)
            action.append(q)
            qpos.append(robot.qpos().copy())
            hand.append(robot.hand_pose().copy())
        f = robot.frame()
        j = rollout.judge(robot)
        if j is not None:
            outcome = j["outcome"]
            if k < k_branch:
                return {"skip": "finished_before_branch", "outcome": outcome}
            break
    z_final = agent.encode(f).float().cpu().numpy()
    end = rollout._diagnosis(robot)
    return {"outcome": outcome or "timeout", "k_branch": k_branch, "rep": rep, "q_branch": q_branch,
            "R_P": agent.R_P, "K": f.K, "cam_final": f.cam_T_world, "z_final": z_final,
            "end_depth_left_mm": end["depth_left_mm"], "end_lateral_mm": end["lateral_mm"], "end_angle_deg": end["angle_deg"],
            "qpos": np.stack(qpos), "hand": np.stack(hand), "action": np.stack(action),
            **{k: np.stack(v) for k, v in dec.items()}}


_STATE = {}


def _init(vae_ckpt: str, policy_ckpt: str):
    _STATE.update(port=None, robot=None, agent=None, error=None)
    try:
        from ..training.common import limit_gpu_memory
        limit_gpu_memory()
        from ..eval.agent import LatentPolicyAgent
        _STATE["agent"] = LatentPolicyAgent(vae_ckpt, policy_ckpt)
    except Exception as exc:          # never raise in a Pool initializer
        _STATE["error"] = f"{type(exc).__name__}: {exc}"[:500]


def record_start(args) -> list[str]:
    """Worker: all branches and continuations of one (grasp, start); one .npz per episode."""
    g, i, out = args
    from ..env.expert import Expert
    from ..env.robot import Robot
    if _STATE["error"]:
        return [f"{g.key} s{i}: agent init failed: {_STATE['error']}"]
    if _STATE["port"] != g.port:
        if _STATE["robot"] is not None:
            _STATE["robot"].env.close()
        _STATE["robot"], _STATE["port"] = Robot(g.port), g.port
    robot = _STATE["robot"]
    expert = Expert(robot)
    msgs = []
    for k in BRANCHES:
        for r in range(REPS):
            path = Path(out) / f"{g.key}_s{i}_k{k}_r{r}.npz"
            if path.exists() or path.with_suffix(".skip").exists():
                continue
            try:
                ep = branch_episode(robot, expert, _STATE["agent"], g, i, k, r)
            except Exception as exc:
                ep = {"skip": f"error {type(exc).__name__}: {exc}"[:300]}
            if "skip" in ep:
                path.with_suffix(".skip").write_text(json.dumps(ep))
                msgs.append(f"{path.stem}: skip {ep['skip']}")
                if ep["skip"] in ("expert", "placement", "no_board_plane", "finished_before_branch"):
                    break                                   # the same for every continuation
                continue
            meta = {"key": g.key, "port": g.port, "condition": g.condition, "start": i, "outcome": ep.pop("outcome")}
            np.savez(path, meta=json.dumps(meta), **ep)
            msgs.append(f"{path.stem}: {meta['outcome']} ({len(ep['step'])} decisions)")
    return msgs


# ====================================================================== comparison (GPU)
def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        d = {k: (f[k].item() if f[k].ndim == 0 and k != "meta" else f[k]) for k in f.files}   # scalars back to Python
    d.update(json.loads(str(d.pop("meta"))))
    return d


def _hand_cmd(fk, ep: dict) -> np.ndarray:
    """Commanded hand pose of every control step (FK of the command, fingers as measured)."""
    return fk(np.concatenate([ep["action"][:, :7], ep["qpos"][:len(ep["action"]), 7:9]], 1))


def start_state(ep: dict, k: int, hand_cmd: np.ndarray) -> dict:
    """Imagination start state at decision k of a recorded episode (ImaginedInsertion.reset_from)."""
    ctx = [max(0, k - c) for c in range(CONTEXT - 1, -1, -1)]      # decisions are 0.4 s apart
    steps = ep["step"][ctx]
    s0 = int(ep["step"][k])
    h = np.zeros((HISTORY_STEPS, 7))
    lo = max(0, s0 - HISTORY_STEPS)
    if s0 > lo:
        h[HISTORY_STEPS - (s0 - lo):] = ep["action"][lo:s0] - ep["qpos"][lo:s0, :7]
    return {"z": ep["z"][ctx].astype(np.float32), "q": ep["qpos"][s0], "hist": h, "ctx_meas": ep["hand"][steps],
            "ctx_cmd": hand_cmd[np.clip(steps, 0, len(hand_cmd) - 1)], "K": ep["K"], "R_P": ep["R_P"],
            "cam_T_world": ep["cam"][k]}


def true_state(depth_left_mm: float, lateral_mm: float) -> str:
    """Where the plug really is (for grouping the prediction errors)."""
    if depth_left_mm > 15:
        return "far"
    return "off_axis_at_board" if lateral_mm > 3 else "near_hole"


class Comparison:
    def __init__(self, env, fm, batch: int = 64):
        self.env, self.fm, self.batch = env, fm, batch

    def real_reading(self, ep: dict) -> dict:
        """Reading of every real decision frame + the final frame of an episode."""
        z = torch.tensor(np.concatenate([ep["z"], ep["z_final"][None]]), dtype=torch.float32, device=self.env.dev)
        hands = np.concatenate([ep["hand"][ep["step"]], ep["hand"][-1:]])
        cams = np.concatenate([ep["cam"], ep["cam_final"][None]])
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.env.dev)
        n = len(z)
        geom = {"K": T(np.repeat(ep["K"][None], n, 0)), "cam_T_world": T(cams), "hand": T(hands),
                "R_P": T(np.repeat(ep["R_P"][None], n, 0))}
        return self.env.readout(z, geom)

    # ---------------------------------------------------------------- open loop
    def open_loop(self, items: list[tuple[dict, int, int]], hand_cmds: list[np.ndarray], n_steps: int, given_joints: bool):
        """items: (episode, decision k, number of steps); executed targets of the episode, nothing real
        fed back. Returns per item a list of predicted readings (dict per step)."""
        out = []
        for a in range(0, len(items), self.batch):
            chunk = items[a:a + self.batch]
            self.env.reset_from([start_state(ep, k, hand_cmds[a + b]) for b, (ep, k, _) in enumerate(chunk)])
            preds = [[] for _ in chunk]
            for t in range(n_steps):
                targets, q_steps = [], []
                for ep, k, n in chunk:
                    kk = min(k + t, len(ep["targets"]) - 1)          # past the episode end: repeat (not scored)
                    targets.append(ep["targets"][kk])
                    s = int(ep["step"][kk])
                    q = ep["qpos"][s + 1:s + 1 + WM_STEP, :7]
                    q_steps.append(np.concatenate([q, np.repeat(ep["qpos"][-1:, :7], WM_STEP - len(q), 0)]))  # short only past the end
                q_steps = np.stack(q_steps) if given_joints else None
                _, info = self.env.step(np.stack(targets), q_steps)
                for b in range(len(chunk)):
                    preds[b].append({**{key: float(v[b]) for key, v in info["read"].items()},
                                     "hand_pos": info["hand"][b][:3, 3].copy()})
            out += preds
        return out

    # ---------------------------------------------------------------- closed loop
    def closed_loop(self, starts: list[dict], n_steps: int) -> list[dict]:
        """FM acts on imagined latents. Returns per start: inserted (reading < SUCCESS_MM at some step),
        the step it happened, minimum / final reading, lateral hand travel at the end."""
        res = []
        for a in range(0, len(starts), self.batch):
            chunk = starts[a:a + self.batch]
            obs = self.env.reset_from(chunk)
            first = np.where(self.env.done, 0, -1)
            dmin = self.env.d_now.copy()
            lat = np.zeros(len(chunk))
            for t in range(n_steps):
                rel = self.fm(obs["z"], obs["q"]).cpu().numpy().astype(np.float64)
                obs, info = self.env.step(rel + self.env.q[:, None, :7])
                first = np.where((first < 0) & info["success"], t + 1, first)
                dmin = np.minimum(dmin, info["read"]["distance_mm"])
                lat = np.where(first < 0, info["lateral_travel_mm"], lat)    # drift until insertion
            for b in range(len(chunk)):
                res.append({"inserted": bool(first[b] >= 0), "step": int(first[b]), "min_reading_mm": float(dmin[b]),
                            "final_reading_mm": float(self.env.d_now[b]), "lateral_travel_mm": float(lat[b])})
        return res


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2),
            "mean": round(float(x.mean()), 2)}
