"""B1 (2026-11-19, the user's request): does the world model imagine what really happened? (factual fidelity)

For windows of recorded trajectories the model has never trained on, the latent of decision t is encoded from the
pictures and joints, then imagined k = 1..H decisions ahead with the actions that were really executed
(z_hat(t+k) = d(...d(z(t), a_t)..., a_{t+k-1})). The offset read-out head turns latents into (along, lateral x,
lateral y [mm], tilt [deg]) of the hand relative to H*; the FK labels give the same quantities (wmdata320 "probe").

Change over k decisions (lateral: 2-vector, along: scalar), and its error against the truth:
    d_true(k) = o_true(t+k) - o_true(t)                       FK labels
    d_wm(k)   = o(z_hat(t+k)) - o(z(t))                        imagined with the executed actions (the dynamics the
                                                               agent's look-ahead rolls; there the later actions are the ACT head's)
    d_enc(k)  = o(z(t+k)) - o(z(t))                            encoded from the real later pictures (read-out floor)
    d_zero    = 0                                              baseline "nothing changes"
    d_cv(k)   = k * (o_true(t) - o_true(t-1))                  baseline "constant velocity" (0 at t = 0)
    error     = |d_x(k) - d_true(k)|  [mm]
Phases by the true pin-tip gap at t, gap = -o_true_along(t) - L (L = pin length): approach (gap > 2 mm), face
(-1 < gap <= 2), inside (gap <= -1, not seated), seated (FK seated label at t).
Through the board: decisions at the face (agent source: proposal / correction / random decisions only, no back-off or
re-position) whose true along progress is |progress| < 0.2 mm over one decision (blocked, not a retreat), and the
share of them where the imagined progress is > 1 mm; the same over 3 decisions (|true| < 0.5 mm, imagined > 1 mm).
Also: the done head on imagined latents against the FK seated label, AUC over states whose pins are in the hole at
t+k (true gap < 0); the reward head on imagined latents against the true reward (mean absolute error), per phase.
Pass (stated before the run): in the phases "face" and "inside", for k = 1, 2, 3, the lateral error of d_wm is below
that of both baselines: per window x = err_wm - err_baseline, mean over windows, 95 % interval by resampling whole
trajectories (2,000 draws, seeded); a cell passes when the upper bound is < 0 for both baselines, with at least
MIN_WINDOWS windows from MIN_TRAJ trajectories (otherwise the cell fails). Overall pass = all six cells pass. Each cell
also reports the read-out floor (err_enc, the read-out on the real later pictures): a fail where err_enc is already
above a baseline is the read-out's, not the imagination's.

Sources: "demo" = held-out trajectories of the fast copy (split heldout: never trained on); "agent" = the trajectories
recorded by run_wm320_attempts.py --record (the in-hole agent on held-out grasps). Labels from FK and H* only.

    python world_model/scripts/analysis/fidelity_factual320.py --ckpt <model.pt> --fast $Q/data/wm_v1_fast \
        --record $OUT/wm_traj --out $OUT/fidelity_factual.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.wm320 import WindowData, auc, load, window_items  # noqa: E402

PHASES = ("approach", "face", "inside", "seated")
BLOCKED_MM, BLOCKED3_MM, THROUGH_MM = 0.2, 0.5, 1.0
MIN_WINDOWS, MIN_TRAJ, BOOT = 100, 20, 2000
PUSH_KINDS = (0, 1, 2)                                  # attempt320.KINDS proposal, correction, random


def phase_of(gap_mm: np.ndarray, seated: np.ndarray) -> np.ndarray:
    p = np.where(gap_mm > 2.0, 0, np.where(gap_mm > -1.0, 1, 2))
    return np.where(seated > 0.5, 3, p)


def true_gap_from_probe(probe_dim0: np.ndarray, L_mm: np.ndarray | float, offset_mode: str) -> np.ndarray:
    """True pin-tip gap [mm] (positive in front of board, negative in hole) from probe dim 0."""
    x = np.asarray(probe_dim0, np.float64)
    if offset_mode == "gap":
        return x
    if offset_mode == "along":
        return -x - np.asarray(L_mm, np.float64)
    raise ValueError(f"unknown offset_mode {offset_mode!r}")


def along_progress_from_delta(delta_dim0: np.ndarray, offset_mode: str) -> np.ndarray:
    """Signed forward insertion progress [mm] (positive = moving into the hole) from dim-0 change over k steps.
    In 'gap' mode: -(g(t+k) - g(t)) == along(t+k) - along(t) identically since L cancels out."""
    d = np.asarray(delta_dim0, np.float64)
    if offset_mode == "gap":
        return -d
    if offset_mode == "along":
        return d
    raise ValueError(f"unknown offset_mode {offset_mode!r}")


def probes_of(root: Path, entries: list[dict], used: set[int], shaping: float, gamma: float, offset_mode: str = "along") -> dict[int, np.ndarray]:
    out = {}
    for e in sorted(used):
        with np.load(root / entries[e]["path"]) as z:
            tr = WD.trajectory(z, json.loads(str(z["meta"])), shaping, gamma, offset_mode=offset_mode)
        out[e] = None if tr is None else tr["probe"]
    return out


@torch.no_grad()
def run_source(m, ck: dict, root: Path, entries: list[dict], n_windows: int, batch: int, workers: int, seed: int, dev) -> dict:
    items = window_items(entries)
    rng = np.random.default_rng(seed)
    if len(items) > n_windows:
        items = items[np.sort(rng.choice(len(items), n_windows, replace=False))]
    shaping, gamma = float(ck.get("shaping", 0.0)), float(m.cfg.gamma)
    offset_mode = str(ck.get("offset_mode", getattr(m, "offset_mode", "along")))
    probes = probes_of(root, entries, {int(e) for e in items[:, 0]}, shaping, gamma, offset_mode=offset_mode)
    keep = np.array([probes[int(e)] is not None for e in items[:, 0]], bool)
    dropped = int((~keep).sum())
    items = items[keep]
    ds = WindowData(root, entries, items, ck["stats"], shaping, gamma, offset_mode=offset_mode)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, persistent_workers=False)
    H = WD.H
    keys = ("lat_wm", "lat_enc", "lat_zero", "lat_cv", "al_wm", "al_enc", "al_zero", "al_cv", "ok", "phase", "prog_true", "prog_wm",
            "done_lab", "done_img", "gap_j", "rew_true", "rew_img")
    rows = {k: [] for k in keys + ("traj", "push", "grasp")}
    gcode = {g: i for i, g in enumerate(sorted({e.get("grasp", "") for e in entries}))}
    for W in dl:
        tmp = {k: [] for k in keys}
        e_idx, k_idx = W["item"][:, 0].numpy(), W["item"][:, 1].numpy()
        W = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in W.items()}
        z = m.observe(W["rgb"], W["depth"], W["q"])                                     # (B, H+1, L)
        act = W["action"].transpose(0, 1)                                               # (H, B, A)
        zs = [z[:, 0]]
        for t in range(H):
            zs.append(m.next(zs[-1], act[t]))
        zs = torch.stack(zs)                                                            # (H+1, B, L)
        off_enc = (m._offset(z) * m.off_scale).transpose(0, 1).double().cpu().numpy()  # (H+1, B, 4)
        off_img = (m._offset(zs) * m.off_scale).double().cpu().numpy()
        done_img = torch.sigmoid(m._done(zs)[..., 0]).double().cpu().numpy()            # (H+1, B)
        rew_img = m.scalar(m.reward(zs[:-1], act))[..., 0].double().cpu().numpy()      # (H, B)
        pr = W["probe"].transpose(0, 1).double().cpu().numpy()                          # (H+1, B, 4) true
        valid = W["valid"].T.double().cpu().numpy()                                     # (H, B)
        ov = W["obs_valid"].T.double().cpu().numpy()
        done = W["done"].T.double().cpu().numpy()
        rew = W["reward"].T.double().cpu().numpy()
        L_mm = np.array([1000.0 * float(entries[int(e)]["pin_length_m"]) for e in e_idx])
        gap = true_gap_from_probe(pr[0, :, 0], L_mm, offset_mode)
        ph = phase_of(gap, done[0])
        prev = np.stack([probes[int(e)][int(k) - 1] if k > 0 else probes[int(e)][int(k)] for e, k in zip(e_idx, k_idx)])
        vel = pr[0] - prev                                                              # (B, 4) true change of the last decision
        chain = np.cumprod(valid, 0)                                                    # transitions 0..j-1 all real
        dk = [entries[int(e)].get("dec_kind") for e in e_idx]                          # agent source: kind of decision k
        push = np.array([True if d is None else (int(k) < len(d) and int(d[int(k)]) in PUSH_KINDS) for d, k in zip(dk, k_idx)], bool)
        for j in range(1, H + 1):
            ok = (chain[j - 1] > 0) & (ov[j] > 0)
            dt = pr[j] - pr[0]
            dw = off_img[j] - off_enc[0]
            de = off_enc[j] - off_enc[0]
            dc = j * vel
            tmp["lat_wm"].append(np.linalg.norm((dw - dt)[:, 1:3], axis=1))
            tmp["lat_enc"].append(np.linalg.norm((de - dt)[:, 1:3], axis=1))
            tmp["lat_zero"].append(np.linalg.norm(dt[:, 1:3], axis=1))
            tmp["lat_cv"].append(np.linalg.norm((dc - dt)[:, 1:3], axis=1))
            tmp["al_wm"].append(np.abs(dw - dt)[:, 0])
            tmp["al_enc"].append(np.abs(de - dt)[:, 0])
            tmp["al_zero"].append(np.abs(dt)[:, 0])
            tmp["al_cv"].append(np.abs(dc - dt)[:, 0])
            tmp["ok"].append(ok)
            tmp["phase"].append(ph)
            tmp["prog_true"].append(along_progress_from_delta(dt[:, 0], offset_mode))
            tmp["prog_wm"].append(along_progress_from_delta(dw[:, 0], offset_mode))
            tmp["done_lab"].append(done[j])
            tmp["done_img"].append(done_img[j])
            tmp["rew_true"].append(rew[j - 1])
            tmp["rew_img"].append(rew_img[j - 1])
            tmp["gap_j"].append(true_gap_from_probe(pr[j, :, 0], L_mm, offset_mode))
        for key in keys:
            rows[key].append(np.stack(tmp[key], 1))                                   # (B, H)
        rows["traj"].append(e_idx.astype(np.int64))
        rows["grasp"].append(np.array([gcode[entries[int(e)].get("grasp", "")] for e in e_idx], np.int64))
        rows["push"].append(push)
    if not rows["traj"]:
        raise RuntimeError("no windows")
    R = {k: np.concatenate(v, 0) for k, v in rows.items()}                            # (N, H) / (N,)
    out = summarise(R, H, len(items))
    out["windows_dropped_no_labels"] = dropped
    return out


def boot_upper(x: np.ndarray, traj: np.ndarray, seed: int = 0) -> float:
    """Upper end of the 95 % interval of mean(x), resampling whole trajectories."""
    ids, inv = np.unique(traj, return_inverse=True)
    sums, cnt = np.bincount(inv, weights=x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(BOOT, len(ids)))
    means = sums[draws].sum(1) / cnt[draws].sum(1)
    return float(np.percentile(means, 97.5))


def summarise(R: dict, H: int, n_windows: int) -> dict:
    out = {"windows": int(n_windows), "trajectories": int(len(np.unique(R["traj"]))), "by_phase": {}, "pass": {}}
    ok = R["ok"].astype(bool)
    for pi, pname in enumerate(PHASES):
        res = {}
        for j in range(1, H + 1):
            sel = ok[:, j - 1] & (R["phase"][:, j - 1] == pi)
            n = int(sel.sum())
            r = {"n": n, "trajectories": int(len(np.unique(R["traj"][sel])))}
            if n:
                for key in ("lat_wm", "lat_enc", "lat_zero", "lat_cv", "al_wm", "al_enc", "al_zero", "al_cv"):
                    v = R[key][sel, j - 1]
                    r[key + "_mean"], r[key + "_median"] = float(v.mean()), float(np.median(v))
                r["wm_minus_enc_mean"] = r["lat_wm_mean"] - r["lat_enc_mean"]
                r["readout_floor_below_baselines"] = bool(r["lat_enc_mean"] < min(r["lat_zero_mean"], r["lat_cv_mean"]))
                r["rew_mae"] = float(np.abs(R["rew_img"][sel, j - 1] - R["rew_true"][sel, j - 1]).mean())
                inh = sel & (R["gap_j"][:, j - 1] < 0)
                r["done_auc_pins_in"] = auc(R["done_img"][inh, j - 1], R["done_lab"][inh, j - 1])
                r["grasps"] = int(len(np.unique(R["grasp"][sel]))) if "grasp" in R else None
                for base in ("zero", "cv"):
                    x = R["lat_wm"][sel, j - 1] - R[f"lat_{base}"][sel, j - 1]
                    r[f"diff_vs_{base}_mean"] = float(x.mean())
                    r[f"diff_vs_{base}_upper95"] = boot_upper(x, R["traj"][sel])
                    if "grasp" in R:                                     # more conservative: whole grasps resampled
                        r[f"diff_vs_{base}_upper95_by_grasp"] = boot_upper(x, R["grasp"][sel])
                r["pass"] = bool(n >= MIN_WINDOWS and r["trajectories"] >= MIN_TRAJ
                                 and r["diff_vs_zero_upper95"] < 0 and r["diff_vs_cv_upper95"] < 0)
            else:
                r["pass"] = False
            res[f"k{j}"] = r
        if pname == "face":
            sel = ok[:, 0] & (R["phase"][:, 0] == pi) & R["push"]
            b1 = sel & (np.abs(R["prog_true"][:, 0]) < BLOCKED_MM)
            moving = sel & (R["prog_true"][:, 0] >= 1.0)
            res["blocked_decisions"] = int(b1.sum())
            res["through_board_share_k1"] = float((R["prog_wm"][b1, 0] > THROUGH_MM).mean()) if b1.any() else None
            if H >= 3:
                s3 = sel & ok[:, 2]
                b3 = s3 & (np.abs(R["prog_true"][:, 2]) < BLOCKED3_MM)
                res["blocked_3_decisions"] = int(b3.sum())
                res["through_board_share_k3"] = float((R["prog_wm"][b3, 2] > THROUGH_MM).mean()) if b3.any() else None
            res["imagined_progress_blocked_mean_mm"] = float(R["prog_wm"][b1, 0].mean()) if b1.any() else None
            res["imagined_progress_moving_mean_mm"] = float(R["prog_wm"][moving, 0].mean()) if moving.any() else None
            res["true_progress_moving_mean_mm"] = float(R["prog_true"][moving, 0].mean()) if moving.any() else None
        out["by_phase"][pname] = res
        if pname in ("face", "inside"):
            out["pass"][pname] = {f"k{j}": res[f"k{j}"]["pass"] for j in (1, 2, 3)}
    out["pass_overall"] = bool(all(v for ph in out["pass"].values() for v in ph.values()))
    out["done_auc_imagined_pins_in"] = {f"k{j}": auc(R["done_img"][ok[:, j - 1] & (R["gap_j"][:, j - 1] < 0), j - 1],
                                                     R["done_lab"][ok[:, j - 1] & (R["gap_j"][:, j - 1] < 0), j - 1]) for j in range(1, H + 1)}
    return out


def show(res: dict) -> str:
    lines = []
    for src, r in res["sources"].items():
        lines.append(f"== {src}: {r['windows']} windows of {r['trajectories']} trajectories; OVERALL PASS {r['pass_overall']}; per cell {r['pass']}")
        lines.append(f"   {'phase':9s} {'k':>2s} {'n':>6s} | lateral change error mean [mm]: {'WM':>6s} {'enc':>6s} {'zero':>6s} {'cv':>6s}"
                     f" | along: {'WM':>6s} {'zero':>6s} {'cv':>6s} | rew MAE")
        for ph, pr in r["by_phase"].items():
            for j in (1, 2, 3, 4):
                x = pr.get(f"k{j}", {})
                if not x.get("n"):
                    continue
                lines.append(f"   {ph:9s} {j:2d} {x['n']:6d} | {'':31s}{x['lat_wm_mean']:6.2f} {x['lat_enc_mean']:6.2f} {x['lat_zero_mean']:6.2f} "
                             f"{x['lat_cv_mean']:6.2f} | {'':7s}{x['al_wm_mean']:6.2f} {x['al_zero_mean']:6.2f} {x['al_cv_mean']:6.2f} | {x['rew_mae']:.2f}")
        f = r["by_phase"]["face"]
        lines.append(f"   face: blocked decisions {f.get('blocked_decisions')}, imagined > {THROUGH_MM} mm through the board: "
                     f"k=1 {f.get('through_board_share_k1')}, k=3 {f.get('through_board_share_k3')} ({f.get('blocked_3_decisions')}); "
                     f"imagined / true progress when moving: {f.get('imagined_progress_moving_mean_mm')} / {f.get('true_progress_moving_mean_mm')} mm")
        lines.append(f"   done head on imagined latents (pins in the hole), AUC: {r['done_auc_imagined_pins_in']}")
        for ph in ("face", "inside"):
            for j in (1, 2, 3):
                x = r["by_phase"][ph][f"k{j}"]
                if x.get("n"):
                    lines.append(f"   {ph} k{j}: WM - zero {x['diff_vs_zero_mean']:+.3f} (upper {x['diff_vs_zero_upper95']:+.3f}), WM - cv "
                                 f"{x['diff_vs_cv_mean']:+.3f} (upper {x['diff_vs_cv_upper95']:+.3f}); read-out floor below baselines "
                                 f"{x['readout_floor_below_baselines']}, WM - enc {x['wm_minus_enc_mean']:+.3f}; pass {x['pass']}; "
                                 f"upper bounds by grasp: zero {x.get('diff_vs_zero_upper95_by_grasp')}, cv {x.get('diff_vs_cv_upper95_by_grasp')} "
                                 f"({x.get('grasps')} grasps)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fast", required=True, help="the fast copy (its held-out trajectories are the 'demo' source)")
    ap.add_argument("--record", help="a run_wm320_attempts.py --record directory (the 'agent' source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=20000, help="windows per source (random, seeded)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split-mode", choices=list(D.SPLIT_MODES), default="split320")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not ck.get("offset_weight", 0.0) > 0:
        raise SystemExit(f"{args.ckpt} has no offset read-out head")
    m = load(args.ckpt, dev)
    sources = {"demo": (Path(args.fast), [e for e in D.load_entries(args.fast, split_mode=args.split_mode) if e["split"] == "heldout"])}
    res = {"ckpt": args.ckpt, "step": int(ck.get("step", -1)), "offset_mode": m.offset_mode, "split_mode": args.split_mode,
           "H": WD.H, "sources": {}, "errors": {}}
    if args.record:
        try:
            man = json.loads((Path(args.record) / "manifest.json").read_text())
            ents = [e for e in man["entries"] if (Path(args.record) / e["path"]).exists()]
            res["agent_manifest"] = {k: man.get(k) for k in ("ckpt", "mode", "split")} | {"entries": len(man["entries"]), "files_missing": len(man["entries"]) - len(ents)}
            if man.get("ckpt") != args.ckpt:
                print(f"warning: the recording was made with {man.get('ckpt')}, not {args.ckpt}", flush=True)
            sources["agent"] = (Path(args.record), ents)
        except Exception as exc:                                     # the demo source still runs
            res["errors"]["agent"] = f"manifest: {type(exc).__name__}: {exc}"[:500]
    for name, (root, ents) in sources.items():
        if not ents:
            res["errors"][name] = "no trajectories"
            continue
        print(f"{name}: {len(ents)} trajectories", flush=True)
        try:
            res["sources"][name] = run_source(m, ck, root, ents, args.windows, args.batch, args.workers, args.seed, dev)
        except Exception as exc:                                  # keep what the other source gave
            res["errors"][name] = f"{type(exc).__name__}: {exc}"[:500]
        Path(args.out).write_text(json.dumps(res, indent=1))     # after every source
    print(show(res), flush=True)
    if res["errors"]:
        print(f"errors: {res['errors']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
