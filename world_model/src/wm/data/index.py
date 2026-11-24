"""Dataset index: one entry per trajectory with its split and port axes.

Port axes (board-plane RANSAC + gravity) are estimated once per trajectory from its
first image, a real-robot quantity; they are stored in the index so training does
not repeat the fit.

    python world_model/scripts/data/build_index.py $Q/data/wm_v1      -> $Q/data/wm_v1/index.json

AutoMate data (sim_automate.collect output, 2026-11-22): build_automate, no port axes (the board-plane fit is
ManiSkill's), only grasps that have grasp.json and only the trajectories it lists with a saved file: a row with
"file" (collect.py writes it just before saving the .npz) names its file; a row without "file" has none
(label_failed, placement_failed, reset_failed, contact_report_error, error) and is skipped. Older grasp.json files
without any "file" key: every row whose outcome is not one of NO_FILE_OUTCOMES, by the name t<idx>_<kind>.npz. A
listed file that is missing raises FileNotFoundError (a stopped run's unfinished grasp or a stale file is not
indexed). Its entries carry simulator "automate", the stored condition, the plug's pin length, expert_completed and
automate_success_any (for reports; demo320 decides the imitation set from the fast copy) and the split of
demo320.split_of_entry ("unassigned" while the split is undecided; the fast copy refuses such entries and skips "test").

    python <code>/scripts/data/build_index.py --automate <collection out dir> --workers 8   -> <out>/index.json
(the launcher form: this tree's wm, foreign wm modules refused; `PYTHONPATH=<code>/src python -m wm.data.index
--automate <out>` does the same without those checks)

Completeness (r5-data-01, r5-ops-chk-01): before anything is read, build_automate takes the collection's state from
plan.json and the output folder without writing to it (automate_completeness; the same facts as collect.run_summary,
which also writes logs/summary.json, so it is not called here):
    planned               the grasps of plan.json
    with_grasp_json       planned grasps whose grasp.json exists (a finished grasp)
    live_workers          <out>/heartbeat/*.json whose (pid, process start time) is a running process
    held_by_live_worker   undone grasps whose .claim names a running process (pid AND start time, collect._alive)
    undone_not_skipped    undone grasps of assemblies that workers do not skip
    skipped_assemblies    collect.blocked_assemblies: {assembly: reason} (repeated build failures or process deaths)
    undone_of_skipped     undone grasps of those assemblies
and refuses (Incomplete, a ValueError) when (i) live_workers or held_by_live_worker is not empty (the collection is
still running), (ii) undone_not_skipped is not empty, or (iii) undone_of_skipped is not empty and accept_skipped is
False (--accept-skipped: the user accepts the listed assemblies' missing grasps). partial=True (--partial, a
deliberate early look) indexes in every case. The facts, the two switches and partial = [(i) or (ii) or (iii) held,
(iii) counting only without accept_skipped] are written
with the index to <out>/index_completeness.json together with index_sha256 (sha256 of index.json's bytes);
scripts/data/cache_wm_v1_fast.py copies them into the fast copy's manifest ("completeness"), depth_stats.py carries
them, and train_wm320.py refuses an AutoMate fast copy whose completeness is missing or partial unless --allow-partial.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import store
from .splits import split_of


def _entry(path: str) -> dict:
    from ..models.perception import port_axes
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        n_img = int(f["img_step"].shape[0]) if "img_step" in f.files else 0
        R = None
        if n_img:
            off = f["rgb_offsets"]
            first = store._decode_jpeg(f["rgb_jpeg"][: off[1]], off[:2])[0]
            R = port_axes(first, f["depth"][0], f["K"], f["cam_T_world"][0])
    return {"path": path, "port": meta["port"], "grasp": meta["grasp"], "condition": meta["condition"],
            "kind": meta["kind"], "outcome": meta.get("outcome"), "n_images": n_img,
            "split": split_of(meta["port"], meta["condition"]),
            "R_P": None if R is None else np.asarray(R).tolist()}


def build(root: str | Path, workers: int = 16) -> list[dict]:
    paths = [str(p) for p in sorted(Path(root).rglob("t*.npz")) if not p.name.endswith(".tmp.npz")]
    with mp.get_context("spawn").Pool(workers) as pool:
        entries = pool.map(_entry, paths, chunksize=8)
    (Path(root) / "index.json").write_text(json.dumps(entries))
    return entries


def load(root: str | Path) -> list[dict]:
    return json.loads((Path(root) / "index.json").read_text())


NO_FILE_OUTCOMES = ("error", "placement_failed", "label_failed", "reset_failed", "contact_report_error")


def automate_files(g: dict) -> list[str]:
    """The trajectory files a grasp.json lists (module docstring)."""
    rows = g.get("trajectories", [])
    if any("file" in t for t in rows):
        return sorted({t["file"] for t in rows if t.get("file") and t.get("outcome") not in NO_FILE_OUTCOMES})
    return sorted({f"t{int(t['idx']):02d}_{t['kind']}.npz" for t in rows if t.get("outcome") not in NO_FILE_OUTCOMES})


def _automate_entries(grasp_json: str) -> list[dict]:
    from ..tdmpc.demo320 import split_of_entry
    gdir = Path(grasp_json).parent
    g = json.loads(Path(grasp_json).read_text())
    names = automate_files(g)
    out = []
    for name in names:
        p = gdir / name
        if not p.exists():
            raise FileNotFoundError(f"{grasp_json} lists {name}, which is missing")
        with np.load(p) as f:
            meta = json.loads(str(f["meta"]))
            n_img = int(f["img_step"].shape[0]) if "img_step" in f.files else 0
        if meta.get("simulator") != "automate":
            raise ValueError(f"{p}: not an AutoMate trajectory (meta simulator {meta.get('simulator')!r})")
        e = {"path": str(p), "port": meta["port"], "grasp": meta["grasp"], "condition": meta.get("condition"),
             "simulator": "automate", "kind": meta["kind"], "outcome": meta.get("outcome"), "n_images": n_img,
             "pin_length_m": float(meta["pin_length_m"]), "R_P": None,
             **{k: meta[k] for k in ("expert_completed", "automate_success_any") if k in meta}}
        e["split"] = split_of_entry(e, allow_undecided=True)
        out.append(e)
    return out


class Incomplete(ValueError):
    """The collection is still running or unfinished (module docstring, completeness)."""


COMPLETENESS_FILE = "index_completeness.json"


def automate_completeness(root: str | Path) -> dict:
    """The completeness facts of an AutoMate collection (module docstring); reads only, writes nothing."""
    from ..sim_automate import collect as AC         # CPU-importable; its liveness and skip rules are the collector's own
    root = Path(root)
    plan_f = root / "plan.json"
    if not plan_f.exists():
        raise FileNotFoundError(f"{root}: no plan.json (not an AutoMate collection output folder)")
    planned = json.loads(plan_f.read_text()).get("grasps")
    if not isinstance(planned, list):
        raise ValueError(f"{plan_f}: no list of planned grasps ('grasps')")
    blocked = AC.blocked_assemblies(root)
    live = []
    for f in sorted((root / "heartbeat").glob("*.json")):
        try:
            hb = json.loads(f.read_text())
            pid, start = int(hb.get("pid", -1)), hb.get("start")
        except (OSError, ValueError, TypeError):
            pid, start = -1, None
            try:                                         # a file being rewritten: its name is <pid>_<start>.json
                p_, s_ = f.stem.split("_", 1)
                pid, start = int(p_), s_
            except ValueError:
                pass
        if AC._alive(pid, None if start is None else str(start)):
            live.append(f.name)
    held, undone_ok, undone_skip, done = [], [], [], 0
    for g in planned:
        gdir = root / g["port"] / g["grasp"]
        if (gdir / "grasp.json").exists():
            done += 1
            continue
        claim = gdir / ".claim"
        if claim.exists():
            try:
                txt = claim.read_text(errors="ignore").strip()
            except OSError:
                txt = ""
            old = AC._parse_claim(txt) if txt else None
            if old is not None and AC._alive(old["pid"], old["start"]):
                held.append(g["grasp"])
        (undone_skip if g["port"] in blocked else undone_ok).append(g["grasp"])
    return {"planned": len(planned), "with_grasp_json": done, "live_workers": live, "held_by_live_worker": held,
            "undone_not_skipped": undone_ok, "skipped_assemblies": blocked, "undone_of_skipped": undone_skip}


def completeness_problems(c: dict) -> dict:
    """{rule: text} of the refusal rules (i)-(iii) that the facts c break (module docstring)."""
    out = {}
    if c["live_workers"] or c["held_by_live_worker"]:
        out["running"] = (f"the collection is still running: {len(c['live_workers'])} live worker heartbeats "
                          f"{c['live_workers'][:5]}, {len(c['held_by_live_worker'])} grasps held by a live worker "
                          f"{c['held_by_live_worker'][:5]}")
    if c["undone_not_skipped"]:
        out["unfinished"] = (f"{len(c['undone_not_skipped'])} planned grasps of assemblies that are not skipped have no "
                             f"grasp.json (e.g. {c['undone_not_skipped'][:5]})")
    if c["undone_of_skipped"]:
        out["skipped"] = (f"{len(c['undone_of_skipped'])} planned grasps of {len(c['skipped_assemblies'])} skipped "
                          f"assemblies have no grasp.json: {c['skipped_assemblies']}")
    return out


def check_completeness(root: str | Path, accept_skipped: bool = False, partial: bool = False) -> dict:
    """The facts plus accept_skipped, partial_flag and partial (module docstring); Incomplete when refused."""
    c = automate_completeness(root)
    why = completeness_problems(c)
    refuse = [t for k, t in why.items() if not partial and (k != "skipped" or not accept_skipped)]
    if refuse:
        hint = ("; --accept-skipped accepts the skipped assemblies' missing grasps (shown to the user)"
                if "skipped" in why and not accept_skipped else "")
        raise Incomplete(f"{root}: " + "; ".join(refuse) + hint + "; --partial indexes anyway as a deliberate early look "
                         "(train_wm320.py then needs --allow-partial)")
    return {**c, "accept_skipped": bool(accept_skipped), "partial_flag": bool(partial),
            "partial": bool("running" in why or "unfinished" in why or ("skipped" in why and not accept_skipped)),
            "problems": why}


def index_sha256(root: str | Path) -> str:
    return hashlib.sha256((Path(root) / "index.json").read_bytes()).hexdigest()


def build_automate(root: str | Path, workers: int = 8, accept_skipped: bool = False, partial: bool = False) -> list[dict]:
    """index.json and index_completeness.json of an AutoMate collection (module docstring); Incomplete (before
    anything is read or written) when the collection is running or unfinished."""
    comp = check_completeness(root, accept_skipped, partial)
    gjs = [str(p) for p in sorted(Path(root).glob("*/*/grasp.json"))]
    with mp.get_context("spawn").Pool(workers) as pool:
        entries = [e for es in pool.map(_automate_entries, gjs, chunksize=4) for e in es]
    old = Path(root) / COMPLETENESS_FILE
    old.unlink(missing_ok=True)                          # never left beside an index it does not describe
    tmp = Path(root) / "index.json.tmp"
    tmp.write_text(json.dumps(entries))
    tmp.replace(Path(root) / "index.json")
    comp["index_sha256"] = index_sha256(root)
    comp["indexed_grasp_json"] = len(gjs)
    tmp = Path(root) / (COMPLETENESS_FILE + ".tmp")
    tmp.write_text(json.dumps(comp, indent=1))
    tmp.replace(old)
    return entries


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--automate", required=True, help="an AutoMate collection's output folder")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accept-skipped", action="store_true")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    es = build_automate(a.automate, a.workers, a.accept_skipped, a.partial)
    print(json.dumps({"trajectories": len(es), "splits": Counter(e["split"] for e in es)}))
