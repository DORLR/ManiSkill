"""Milestone-1b diagnosis, simulation side: re-run checks (i) and (ii) of batch_assemblies.py for single assemblies
with per-control-step traces and variants that isolate one cause each. Environment code is used unchanged; the
variants only change what this script commands.

Experiments (--exp, comma list; each starts with Robot.reset):
  i          as batch (i): move_to(H*) from the reset pose (move_to's procedure replicated step by step so that every
             control step is recorded: waypoints <= 2 mm / 1 deg, 2 control steps each, settle <= 40 steps), hold 5
  i_slow     same with waypoints <= 0.5 mm (4x slower approach)
  i_open     same as i, fingers fully opened (target 0.04 m) before the motion: no finger load on the socket unless
             the open finger itself touches it
  ii         as batch (ii): move_to(H* - (L + 10 mm) a), straight line at 10 mm/s, stop when the command leads the
             FK hand by > 1.5 mm, hold 10
  ii_slow    same at 2 mm/s
  ii_open    same as ii with the fingers fully open
  ii_dp<x>   same as ii with the grasp offset dp = x mm (e.g. ii_dp-6.5; dtheta = dpsi = 0)
  ii_shift<dx>_<dy>  same as ii with H* (and the whole line) moved laterally by (dx, dy) mm in the socket frame
  teleport   joints written directly at IK(H*) (plug placed exactly seated, zero velocity, PD target = that pose),
             then hold 20 control steps: shows whether the seated pose is a rest state of the simulator
Recorded per control step: depth_left, lateral, tilt, plug-socket force (norm, axial = along a, lateral), socket
force on panda_hand / leftfinger / rightfinger, command-vs-FK lag. At the end: checks.penetration, poses of the plug,
hand and fingers in the socket frame (for section pictures), finger joint positions.
Also prints the SDF / contact attributes of the plug and socket collision prims (USD).
    python -u sim_diag.py --ids 00271 --exp i,ii,i_open --out /mnt/storage3/qingxin/automate/m1b/diag/sim
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wm.sim_automate.app import launch, shutdown  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--exp", default="i,i_slow,i_open,ii,ii_slow,ii_open,teleport")
ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1b/diag/sim")
args = ap.parse_args()
app = launch(headless=True, cameras=False)

import numpy as np  # noqa: E402
from pxr import Usd  # noqa: E402

from wm.sim.geometry import interpolate_T, pose_distance  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import checks as K  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate.robot import Robot  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def usd_collision_attrs(r) -> dict:
    stage = r.sc.sim.stage
    out = {}
    for root in ("/World/envs/env_0/FixedAsset", "/World/envs/env_0/HeldAsset"):
        for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            attrs = {}
            for a in p.GetAttributes():
                n = a.GetName()
                if any(s in n for s in ("sdf", "Sdf", "Offset", "approximation", "collisionEnabled")):
                    v = a.Get()
                    attrs[n] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
            if attrs:
                out[str(p.GetPath())] = attrs
    return out


class Rec:
    def __init__(self, r):
        self.r, self.rows = r, []
        self.names = list(r.sc.finger_contact.body_names)
        self.a = r.axis()

    def __call__(self, cmd=None):
        r = self.r
        t = K.truth(r)
        f = r.sc.contact_force()
        fa = float(f @ self.a)
        fm = r.sc.finger_contact.data.force_matrix_w
        per = {} if fm is None else {n: float(np.linalg.norm(v)) for n, v in
                                     zip(self.names, fm[0].detach().cpu().numpy().reshape(len(self.names), -1, 3).sum(1))}
        row = {"step": len(self.rows), "depth": t["depth_left_mm"], "lat": t["lateral_mm"], "tilt": t["tilt_deg"],
               "F_plug": float(np.linalg.norm(f)), "F_axial": fa, "F_lat": float(np.linalg.norm(f - fa * self.a)),
               **{"F_" + n.replace("panda_", ""): v for n, v in per.items()}}
        if cmd is not None:
            row["lag_mm"] = float(np.linalg.norm(cmd[:3, 3] - r.hand_pose()[:3, 3]) * 1e3)
        self.rows.append(row)

    def peaks(self) -> dict:
        keys = [k for k in self.rows[0] if k.startswith("F_")] if self.rows else []
        return {k: max(abs(x[k]) for x in self.rows) for k in keys}


_MESH = {}


def penetration(r, n: int = 20000, chunk: int = 1000) -> dict:
    """checks.penetration (same points, same trimesh signed distance), evaluated in chunks of `chunk` points: trimesh's
    closest-point query is per point, so the result is identical, but the peak memory is ~n/chunk times smaller
    (measured: the one-shot call was OOM-killed under a 14 GB cap for 00681 and 01026). Also reports the ray-parity
    inside test (mesh.contains) that does not depend on the closest triangle's normal."""
    import trimesh
    key = r.assembly_id
    if key not in _MESH:
        plug = trimesh.load(A.fetch(f"{key}/plug.obj"), force="mesh")
        sock = trimesh.load(A.fetch(f"{key}/socket.obj"), force="mesh")
        pts, _ = trimesh.sample.sample_surface_even(plug, n, seed=0)
        _MESH.clear()
        _MESH[key] = (np.vstack([pts, plug.vertices]), sock)
    pts, sock = _MESH[key]
    M = np.linalg.inv(r.sc.socket_T()) @ r.sc.plug_T()
    q = pts @ M[:3, :3].T + M[:3, 3]
    sd = np.concatenate([trimesh.proximity.signed_distance(sock, q[k:k + chunk]) for k in range(0, len(q), chunk)])
    inside = np.concatenate([sock.contains(q[k:k + chunk]) for k in range(0, len(q), chunk)])
    d_in = np.abs(sd[inside]) if inside.any() else np.zeros(1)
    return {"max_penetration_mm": float(max(sd.max(), 0.0) * 1000), "n_points_inside_0.05mm": int((sd > 5e-5).sum()),
            "n_points": int(len(q)), "ray_inside_points": int(inside.sum()),
            "ray_inside_max_depth_mm": float(d_in.max() * 1000)}


def final(r, rec) -> dict:
    S_inv = np.linalg.inv(r.sc.socket_T())
    poses = {"plug": S_inv @ r.sc.plug_T(), "hand": S_inv @ r.sc.body_T("panda_hand"),
             "leftfinger": S_inv @ r.sc.body_T("panda_leftfinger"), "rightfinger": S_inv @ r.sc.body_T("panda_rightfinger"),
             "tcp_fk": S_inv @ r.hand_pose()}
    return {**K.truth(r), **penetration(r), "q_fingers_mm": (r.qpos()[7:9] * 1e3).tolist(),
            "last": rec.rows[-1] if rec.rows else None, "peaks": rec.peaks(),
            "poses_socket_frame": {k: v.tolist() for k, v in poses.items()}, "trace": rec.rows}


def open_fingers(r, rec):
    r.q_finger = 0.04
    r._command(r.qpos()[:7], 5)
    rec()


def move_traced(r, rec, T_goal, max_lin=0.002, max_ang=np.radians(1.0), settle_steps=40, tol_m=1e-4,
                tol_rad=np.radians(0.05)) -> bool:
    """Robot.move_to's procedure (robot.py), with a record after every control step. Unlike move_to it returns True
    once IK succeeds (a settle failure is not reported), so its move_ok means "IK succeeded"."""
    start = r.hand_pose()
    dt, dr = pose_distance(start, T_goal)
    n = max(1, int(np.ceil(max(dt / max_lin, dr / max_ang))))
    seed_q, targets = r.qpos(), []
    for k in range(1, n + 1):
        sol = r._ik(interpolate_T(start, T_goal, k / n), seed_q)
        if sol is None:
            return False
        targets.append(sol)
        seed_q = sol
    for k, sol in enumerate(targets):
        cmd = interpolate_T(start, T_goal, (k + 1) / n)
        for _ in range(2):
            r._command(sol, 1)
            rec(cmd)
    prev = r.hand_pose()
    for _ in range(settle_steps):
        r._command(targets[-1], 1)
        rec(T_goal)
        now = r.hand_pose()
        dt, dr = pose_distance(now, T_goal)
        if dt < tol_m and dr < tol_rad and r._still(now, prev):      # same stillness test as Robot.move_to
            break
        prev = now
    return True


def exp_i(r, max_lin=0.002, opened=False) -> dict:
    r.reset(0, clamp_deg=0.0, dp_m=0.0, spin_deg=0.0, socket_index=0)
    rec = Rec(r)
    if opened:
        open_fingers(r, rec)
    H = r.H_star()
    ok = move_traced(r, rec, H, max_lin=max_lin)
    for _ in range(5):
        r._command(r._q_cmd, 1)
        rec(H)
    dt, dr = pose_distance(r.hand_pose(), H)
    return {"move_ok": ok, "hand_to_H*_mm": dt * 1e3, "hand_to_H*_deg": float(np.degrees(dr)), **final(r, rec)}


def exp_ii(r, speed=0.010, opened=False, dp_m=0.0, shift_mm=None) -> dict:
    r.reset(0, clamp_deg=0.0, dp_m=dp_m, spin_deg=0.0, socket_index=0)
    rec = Rec(r)
    if opened:
        open_fingers(r, rec)
    H, L = r.H_star(), r.port_spec.pin_length_m
    if shift_mm is not None:                     # whole line moved by (dx, dy) in the socket frame (lateral only)
        S = r.sc.socket_T()
        H = H.copy()
        H[:3, 3] += S[:3, :3] @ np.array([shift_mm[0], shift_mm[1], 0.0]) / 1e3
    placed = r.move_to(K.along(r, H, L + 0.010))
    start = K.truth(r)
    rec()
    p = K.drive_line(r, r.hand_pose(), H, speed, lambda: rec(None), lag_stop=0.0015)
    for _ in range(10):
        r._command(p["q"], 1)
        rec(H)
    return {"placed": placed, "start": start, "push_end": p["end"], "push_steps": p["steps"],
            "cmd_depth_left_at_end_mm": float((H[:3, 3] - p["cmd"][:3, 3]) @ r.axis() * 1e3),
            "grasp": [0.0, dp_m, 0.0], **final(r, rec)}


def exp_teleport(r) -> dict:
    r.reset(0, clamp_deg=0.0, dp_m=0.0, spin_deg=0.0, socket_index=0)
    rec = Rec(r)
    H = r.H_star()
    q = r.kin.ik(H, r.qpos(), tol=1e-7, n_restarts=19)
    if q is None:
        return {"error": "no IK at H*"}
    q[7:9] = r.q_finger
    r.sc.write_state(0, q, r.kin.link("panda_hand", q) @ C.T_HAND_TCP @ r.T_tcp_plug, np.r_[q[:7], r.q_finger, r.q_finger])
    r._q_cmd = q[:7].copy()
    r.sc.scene.write_data_to_sim()
    r.sc.sim.forward() if hasattr(r.sc.sim, "forward") else None
    r.sc.scene.update(dt=0.0)
    at0 = {**K.truth(r), **penetration(r)}
    for _ in range(20):
        r._command(q[:7], 1)
        rec(H)
    return {"written": at0, **final(r, rec)}


def run(aid: str) -> dict:
    res = {"assembly_id": aid}
    t0 = time.time()
    r = Robot(aid, camera=False)
    res["build_s"] = time.time() - t0
    res["usd"] = usd_collision_attrs(r)
    res["q_finger_nominal_mm"] = r.q_finger * 1e3
    res["finger_contact_bodies"] = list(r.sc.finger_contact.body_names)
    for e in args.exp.split(","):
        t1 = time.time()
        try:
            if e == "i":
                x = exp_i(r)
            elif e == "i_slow":
                x = exp_i(r, max_lin=0.0005)
            elif e == "i_open":
                x = exp_i(r, opened=True)
            elif e == "ii":
                x = exp_ii(r)
            elif e == "ii_slow":
                x = exp_ii(r, speed=0.002)
            elif e == "ii_open":
                x = exp_ii(r, opened=True)
            elif e.startswith("ii_shift"):                                     # ii_shift<dx>_<dy> [mm, socket frame]
                x = exp_ii(r, shift_mm=[float(v) for v in e[8:].split("_")])
            elif e.startswith("ii_dp"):
                x = exp_ii(r, dp_m=float(e[5:]) / 1e3)
            elif e == "teleport":
                x = exp_teleport(r)
            else:
                x = {"error": "unknown experiment"}
        except Exception as exc:                                                  # noqa: BLE001
            x = {"error": f"{type(exc).__name__}: {exc}", "trace_back": traceback.format_exc()[-1500:]}
        x["seconds"] = time.time() - t1
        res[e] = x
        short = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()
                 if k in ("depth_left_mm", "lateral_mm", "tilt_deg", "max_penetration_mm", "push_end", "move_ok", "error")}
        print(aid, e, json.dumps(short), "peaks", json.dumps({k: round(v, 2) for k, v in x.get("peaks", {}).items()}),
              flush=True)
    res["seconds"] = time.time() - t0
    r.sc.close()
    return res


for aid in args.ids.split(","):
    try:
        res = run(aid)
    except Exception as exc:                                                      # noqa: BLE001
        res = {"assembly_id": aid, "error": f"{type(exc).__name__}: {exc}", "trace_back": traceback.format_exc()[-2000:]}
    (OUT / f"{aid}.json").write_text(json.dumps(res, indent=1, default=float))
    print(aid, "written", flush=True)
print("DIAG DONE", flush=True)
shutdown(0)
