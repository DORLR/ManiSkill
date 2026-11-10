"""Online data collection (v4): the FM policy, choosing with the world model, in the simulator.

Why: the world model was trained on expert data and was optimistic on the states the FM policy
reaches (docs/2026-11-09_imagination_check.md). Here the world model itself picks among the FM's
proposals; where it is wrong, the simulator shows what really happens and that data goes back into
its training (online/learner.py) -- the Dyna / TD-MPC2 loop.

Episode (train-split shapes and training-band grasps only; ood_val / test never):
    reset the grasp, the expert inserts once (as in data collection), planner placement at a random
    far start, port axes from the first image, then every control step is recorded (data/recorder.py,
    images every 0.2 s) and every image is encoded (UNet -> crops -> crop autoencoder).
    Every 0.4 s a decision:
        mode "plan" (p_plan): K FM proposals; each is imagined H decisions ahead (the proposal, then the
              FM acting in imagination); score = max over the steps of the success classifier; the
              best is executed (with p_explore a random proposal instead)
        mode "fm": one FM sample (the policy as it is)
    up to MAX_DECISIONS or an outcome (eval/rollout.judge: inserted / beside_board / tilted).
Saved per episode: the raw trajectory (data/store.py format, kind "online_<mode>") and its v4 latent
file (training/encode4.py format, + R_P, + the success label from the simulator at each image).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data import store
from ..data.outcome import inserted
from ..data.recorder import Recorder
from ..models.crop_ae import load_crop_ae
from ..perception.crops import CropPlacer
from ..perception.unet import load_unet

MAX_DECISIONS = 150
EXECUTE = 8


class Perception4:
    """Frame -> UNet -> crop placement -> crop autoencoder latent (scaled)."""

    def __init__(self, unet_ckpt: str, ae_ckpt: str, scale: float, device):
        self.dev = torch.device(device)
        self.unet = load_unet(unet_ckpt, self.dev)
        self.ae = load_crop_ae(ae_ckpt, self.dev).requires_grad_(False)
        self.scale = scale
        self.placer = CropPlacer()

    def reset(self):
        self.placer.reset()

    @torch.no_grad()
    def encode(self, rgb, depth, K, cam_T_world, hand, R_P) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = self.unet(T(rgb, torch.uint8)).argmax(1)[0].to(torch.uint8).cpu().numpy()
        c = self.placer(cls, depth, K, cam_T_world)
        b = {"rgb": T(rgb, torch.uint8), "depth": T(depth.astype(np.int32), torch.int32),
             "plug_uv0": T(c["plug_uv0"], torch.long), "hole_uv0": T(c["hole_uv0"], torch.long), "K": T(K),
             "cam_T_world": T(cam_T_world), "hand": T(hand), "R_P": T(R_P)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = self.ae.encode(b)
        return z.float()[0] * self.scale, c["plug_uv0"], c["hole_uv0"]


class Models:
    """Everything an actor needs on the GPU; world model and classifier are reloaded from the learners."""

    def __init__(self, cfg: dict, device="cuda"):
        from ..data.kinematics import HandFK  # noqa: F401
        from ..imagination import FMSampler
        from ..training.adapter import load_adapter
        self.cfg, self.dev = cfg, torch.device(device)
        self.scale = json.loads((Path(cfg["latents"]) / "stats.json").read_text())["scale"]
        self.perception = Perception4(cfg["unet"], cfg["ae"], self.scale, self.dev)
        self.fm = FMSampler(cfg["policy"], self.dev)
        assert abs(self.fm.scale - self.scale) < 1e-6, (self.fm.scale, self.scale)
        self.adapter = load_adapter(cfg["adapter"], self.dev).requires_grad_(False)
        self.wm = self.success = None
        self.versions = {}
        self.reload()

    def _load_if_new(self, path: str, loader, attr: str):
        p = Path(path)
        if not p.exists():
            return
        m = p.stat().st_mtime
        if self.versions.get(attr) != m:
            setattr(self, attr, loader(str(p), self.dev).requires_grad_(False))
            self.versions[attr] = m

    def reload(self):
        from ..training.success import load_success
        from ..training.wm4 import load_history_wm
        for key, loader, attr in (("wm", load_history_wm, "wm"), ("success", load_success, "success")):
            live = self.cfg[f"{key}_live"]
            self._load_if_new(live if Path(live).exists() else self.cfg[f"{key}_init"], loader, attr)


def run_episode(robot, expert, models: Models, fk, port: str, condition: str, rng: np.random.Generator,
                mode: str, out_dir: Path, name: str) -> dict:
    from ..env.sampling import sample_grasp, sample_start
    from ..eval.rollout import judge
    from ..imagination4 import HistoryImagination, start_from_record
    from ..models.perception import port_axes
    from ..training.encode4 import save_latent_file
    cfg = models.cfg
    t0 = time.time()
    pitch, pos = sample_grasp(rng, condition)
    seed = int(rng.integers(10 ** 8, 2 * 10 ** 8))
    robot.reset(seed, pitch, pos)
    if not (expert.to_safe_from_spawn() and expert.run("insert", lambda *a: None, seed_offset=0).get("completed")):
        return {"name": name, "outcome": "expert_failed"}
    H_star = robot.hand_pose()
    start = sample_start(rng)
    if not expert.move_to_offset(start):
        return {"name": name, "outcome": "placement_failed"}
    f = robot.frame()
    R_P = port_axes(f.rgb, f.depth, f.K, f.cam_T_world, seed)
    if R_P is None:
        return {"name": name, "outcome": "no_board_plane"}
    torch.manual_seed(seed)
    models.perception.reset()
    rec = Recorder(robot)
    rec.begin()
    lat, plug_uv0, hole_uv0 = [], [], []

    def encode_new():
        while len(lat) < len(rec.buf["img_step"]):
            j = len(lat)
            s = rec.buf["img_step"][j]
            z, pu, hu = models.perception.encode(rec.buf["rgb"][j], rec.buf["depth"][j], rec.K, rec.buf["cam_T_world"][j],
                                                rec.buf["hand"][s], R_P)
            lat.append(z), plug_uv0.append(pu), hole_uv0.append(hu)

    imag = HistoryImagination(models.wm, models.adapter, fk, models.success, models.scale, models.dev,
                              cfg["sampling_steps"]) if mode == "plan" else None
    outcome, chosen_scores = None, []
    for k in range(MAX_DECISIONS):
        encode_new()
        n = len(lat)
        ctx = [lat[max(0, n - 1 - 2 * c)] for c in (3, 2, 1, 0)]
        z = torch.stack(ctx)[None]
        qpos = np.asarray(rec.buf["qpos"], np.float64)
        q = torch.tensor(qpos[-1, :7], dtype=torch.float32, device=models.dev)[None]
        if mode == "plan":
            K = cfg["candidates"]
            rel = models.fm(z.expand(K, -1, -1, -1), q.expand(K, -1)).cpu().numpy().astype(np.float64)
            cand = rel + qpos[-1, :7]
            hand = np.asarray(rec.buf["hand"], np.float64)
            action = np.asarray(rec.buf["action"], np.float64).reshape(-1, 7)
            steps = np.asarray(rec.buf["img_step"])
            cmd_img = fk(np.concatenate([np.concatenate([action, cand[0, :1]])[np.minimum(steps, len(action))],
                                         qpos[steps, 7:9]], 1))
            st = start_from_record(lat, hand[steps], cmd_img, qpos, action, int(steps[-1]), R_P)
            imag.reset_from([st] * K)
            best = np.zeros(K)
            targets = cand
            for h in range(cfg["horizon"]):
                info = imag.step(targets)
                best = np.maximum(best, info["p_success"])
                o = imag.obs()
                targets = models.fm(o["z"], o["q"]).cpu().numpy().astype(np.float64) + imag.q[:, None, :7]
            pick = int(rng.integers(K)) if rng.random() < cfg["p_explore"] else int(np.argmax(best))
            chosen_scores.append(float(best[pick]))
            a = cand[pick]
        else:
            a = models.fm(z, q)[0].cpu().numpy().astype(np.float64) + qpos[-1, :7]
        for qq in a[:EXECUTE]:
            robot._command(qq, 1)
            rec.on_step(f"online_{mode}", qq, {})
            j = judge(robot)
            if j is not None:
                outcome = j["outcome"]
                break
        if outcome is not None:
            break
    arrays = rec.end()
    encode_new()
    kind = f"online_{mode}"
    meta = {"grasp": name, "port": port, "condition": condition, "pitch_deg": pitch, "grasp_pos_m": pos,
            "episode_seed": seed, "kind": kind, "outcome": outcome or "timeout", "steps": int(len(arrays["action"])),
            "H_star": H_star.tolist(), "start": start.as_dict(), "chosen_scores": chosen_scores}
    raw = out_dir / "raw" / port / f"{name}.npz"
    store.save(raw, arrays, meta)
    t = {"img_step": arrays["img_step"], "qpos": arrays["qpos"], "action": arrays["action"], "hand": arrays["hand"],
         "hand_cmd": fk.commanded(arrays["action"], arrays["qpos"]) if len(arrays["action"]) else np.zeros((0, 4, 4)),
         "cam_T_world": arrays["cam_T_world"], "K": arrays["K"], "plug_uv0": np.stack(plug_uv0),
         "hole_uv0": np.stack(hole_uv0), "inserted": inserted(arrays["eval_rel"][arrays["img_step"]]),
         "R_P": np.asarray(R_P, np.float32)}
    mean = (torch.stack(lat) / models.scale).cpu().numpy().astype(np.float16)
    save_latent_file(out_dir / "latents" / port / f"{name}.npz", mean, t)
    return {"name": name, "port": port, "condition": condition, "mode": mode, "outcome": meta["outcome"],
            "decisions": k + 1, "seconds": round(time.time() - t0, 1),
            "mean_chosen_score": float(np.mean(chosen_scores)) if chosen_scores else None}


# ---------------------------------------------------------------------------- worker loop
def worker(args):
    """One actor process: episodes forever (or `episodes`), reloading the learners' checkpoints."""
    wid, cfg, out_dir, episodes = args
    from ..env.expert import Expert
    from ..env.robot import Robot
    from ..data.kinematics import HandFK
    from ..training.common import limit_gpu_memory
    limit_gpu_memory()
    out_dir = Path(out_dir)
    rng = np.random.default_rng(cfg["seed"] * 1000 + wid)
    models = Models(cfg)
    log = open(out_dir / f"actor_{wid}.jsonl", "a")
    robot, port, fk = None, None, None
    n = 0
    while episodes is None or n < episodes:
        new_port = cfg["ports"][int(rng.integers(len(cfg["ports"])))]
        if new_port != port:
            if robot is not None:
                robot.env.close()
            robot, port = Robot(new_port), new_port
            fk = None
        expert = Expert(robot)
        if fk is None:
            robot.reset(0, -45.0, -0.024)
            fk = HandFK(robot)
        mode = "plan" if rng.random() < cfg["p_plan"] else "fm"
        name = f"w{wid:02d}_{n:06d}"
        try:
            res = run_episode(robot, expert, models, fk, port, "train", rng, mode, out_dir, name)
        except Exception as exc:                   # an episode error is logged, the actor goes on
            import traceback
            res = {"name": name, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
        log.write(json.dumps(res) + "\n")
        log.flush()
        n += 1
        if n % cfg["reload_every"] == 0:
            models.reload()
    return n
