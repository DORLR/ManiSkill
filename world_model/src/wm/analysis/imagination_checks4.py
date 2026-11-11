"""Does v4 imagination agree with the simulator, on the FM policy's own trajectories? (ood_val shapes)

Same protocol as the v3g check (imagination_checks.py, 2026-11-09), with the v4 models:
perception = UNet -> crop placement -> crop autoencoder; world model = history CDiT; judge = success
classifier (not a pose readout).

Recording (simulator; record_start, parallel workers; needs only the FM and perception)
    starts      evaluation plan seed 0, first grasp of every (shape, condition) cell, both starts
    branches    branch points after 20 / 40 / 60 decisions; R continuations with their own policy seed,
                up to H decisions or an outcome (eval/rollout.judge)
    saved       every control step (joints, command, hand, simulator pose), every 0.2 s image: latent,
                crop origins, camera pose; the image index of every decision; truth only for judging

Comparison (GPU; compare)
    1 classifier  P(inserted) on the real histories at every decision vs the simulator label; in
                  particular the frames where the plug is at depth but beside the hole or tilted
    2 prediction  open loop with the commands really executed, from the branch points 0.4-16 s ahead,
                  real joints and adapter joints: plug pose read (crop autoencoder) from the predicted
                  vs the real latent, next to "nothing changes"
    3 outcome     from each branch point the FM acts in imagination (N continuations, H decisions);
                  imagined inserted = the classifier says > 0.5 at some step; vs the R simulator
                  continuations: rates, per-branch agreement, branches the simulator ends beside / tilted
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.outcome import diagnosis, inserted
from ..eval import rollout

BRANCHES = (20, 40, 60)
HORIZON = 40
REPS = 4
EXECUTE = 8
PRED_STEPS = (1, 5, 10, 25, 40)


# ====================================================================== recording
def branch_episode(robot, expert, perception, fm, g, i: int, k_branch: int, rep: int, horizon: int = HORIZON) -> dict:
    from ..data.recorder import Recorder
    from ..models.perception import port_axes
    seed = g.seed + i
    robot.reset(g.seed, g.pitch, g.grasp_pos)
    if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
        return {"skip": "expert"}
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if not expert.move_to_offset(g.starts[i]):
        return {"skip": "placement"}
    f = robot.frame()
    R_P = port_axes(f.rgb, f.depth, f.K, f.cam_T_world, seed)
    if R_P is None:
        return {"skip": "no_board_plane"}
    perception.reset()
    rec = Recorder(robot)
    rec.begin()
    lat, pu, hu, dec_img = [], [], [], []

    def encode_new():
        while len(lat) < len(rec.buf["img_step"]):
            j = len(lat)
            s = rec.buf["img_step"][j]
            z, a, b = perception.encode(rec.buf["rgb"][j], rec.buf["depth"][j], rec.K, rec.buf["cam_T_world"][j],
                                        rec.buf["hand"][s], R_P)
            lat.append(z), pu.append(a), hu.append(b)

    outcome, q_branch = None, None
    for k in range(k_branch + horizon):
        encode_new()
        if k == k_branch:
            q_branch = np.asarray(rec.buf["qpos"][-1]).copy()
            cs = 7_000_000 + 1000 * seed + 10 * k_branch + rep
            torch.manual_seed(cs)
            np.random.seed(cs % (2 ** 32))
        n = len(lat)
        dec_img.append(n - 1)
        z = torch.stack([lat[max(0, n - 1 - 2 * c)] for c in (3, 2, 1, 0)])[None]
        qpos = np.asarray(rec.buf["qpos"], np.float64)
        q = torch.tensor(qpos[-1, :7], dtype=torch.float32, device=perception.dev)[None]
        a = fm(z, q)[0].cpu().numpy().astype(np.float64) + qpos[-1, :7]
        for qq in a[:EXECUTE]:
            robot._command(qq, 1)
            rec.on_step("check", qq, {})
            j = rollout.judge(robot)
            if j is not None:
                outcome = j["outcome"]
                break
        if outcome is not None:
            if k < k_branch:
                return {"skip": "finished_before_branch", "outcome": outcome}
            break
    arrays = rec.end()
    encode_new()
    return {"outcome": outcome or "timeout", "k_branch": k_branch, "rep": rep, "q_branch": q_branch,
            "R_P": np.asarray(R_P, np.float32), "mean": (torch.stack(lat) / perception.scale).cpu().numpy().astype(np.float32),
            "plug_uv0": np.stack(pu), "hole_uv0": np.stack(hu), "dec_img": np.asarray(dec_img, np.int32),
            **{k: arrays[k] for k in ("img_step", "qpos", "hand", "action", "eval_rel", "cam_T_world", "K")}}


_STATE = {}


def _init(cfg: dict):
    _STATE.update(port=None, robot=None, error=None)
    try:
        from ..imagination import FMSampler
        from ..online.actor import Perception4
        from ..training.common import limit_gpu_memory
        limit_gpu_memory()
        scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
        _STATE["perception"] = Perception4(cfg["unet"], cfg["ae"], scale, "cuda")
        _STATE["fm"] = FMSampler(cfg["policy"], torch.device("cuda"))
        assert abs(_STATE["fm"].scale - scale) < 1e-6
    except Exception as exc:
        _STATE["error"] = f"{type(exc).__name__}: {exc}"[:500]


def record_start(args) -> list[str]:
    g, i, out = args
    from ..env.expert import Expert
    from ..env.robot import Robot
    if _STATE["error"]:
        return [f"{g.key} s{i}: init failed: {_STATE['error']}"]
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
                ep = branch_episode(robot, expert, _STATE["perception"], _STATE["fm"], g, i, k, r)
            except Exception as exc:
                ep = {"skip": f"error {type(exc).__name__}: {exc}"[:300]}
            if "skip" in ep:
                path.with_suffix(".skip").write_text(json.dumps(ep))
                msgs.append(f"{path.stem}: skip {ep['skip']}")
                if not ep["skip"].startswith("error"):
                    break
                continue
            meta = {"key": g.key, "port": g.port, "condition": g.condition, "start": i, "outcome": ep.pop("outcome")}
            np.savez(path, meta=json.dumps(meta), **ep)
            msgs.append(f"{path.stem}: {meta['outcome']} ({len(ep['dec_img'])} decisions)")
    return msgs


# ====================================================================== comparison
def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        d = {k: (f[k].item() if f[k].ndim == 0 and k != "meta" else f[k]) for k in f.files}
    d.update(json.loads(str(d.pop("meta"))))
    return d


def commanded_hands(fk, ep: dict) -> np.ndarray:
    """Commanded hand pose at each image's step (the command issued at that step; the last image: the hand)."""
    steps = ep["img_step"].astype(int)
    act = ep["action"]
    q = np.concatenate([act[np.minimum(steps, len(act) - 1)][:, :7], ep["qpos"][steps, 7:9]], 1)
    return fk(q)


def start_at(ep: dict, k: int, cmd_img: np.ndarray, scale: float) -> dict:
    """Imagination start state at decision k (history of real latents up to that image)."""
    return start_at_image(ep, int(ep["dec_img"][k]), cmd_img, scale)


def start_at_image(ep: dict, j: int, cmd_img: np.ndarray, scale: float) -> dict:
    """History of real latents up to image j (also the final image, taken after the outcome)."""
    from ..imagination4 import start_from_record
    s = int(ep["img_step"][j])
    lat = ep["mean"][:j + 1] * scale                     # start_from_record only reads the last HIST images
    return start_from_record(lat, ep["hand"][ep["img_step"][:j + 1].astype(int)], cmd_img[:j + 1], ep["qpos"][:s + 1],
                             ep["action"][:s], s, ep["R_P"])


def truth_at(ep: dict, j: int) -> dict:
    rel = ep["eval_rel"][int(ep["img_step"][j])]
    d = diagnosis(rel)
    return {"inserted": bool(inserted(rel)), "depth_left_mm": float(d["depth_left_m"] * 1000),
            "lateral_mm": float(d["lateral_m"] * 1000), "angle_deg": float(d["angle_deg"])}


def auc(pos, neg) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "p90": round(float(np.percentile(x, 90)), 3)}
