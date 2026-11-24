"""Write the fast copy of the demonstrations (wm/tdmpc/demo320.py, format "wm-fast-1") and its manifest.
Reads the source's index.json (paths absolute or relative to --src). Ports A-D are skipped (the final test must
stay untouched), trajectories without pictures are skipped and counted. AutoMate data (index of
wm.data.index.build_automate; any entry, or a plan.json in --src, says AutoMate): every entry must have simulator
"automate" (an index of the ManiSkill builder is refused), the split is recomputed from each entry's stored
condition (demo320.split_of_entry, never the index's "split" field): "test" is skipped, an undecided split
("unassigned") refuses the whole copy; the pin length comes from the entry (--pins not needed); the manifest records
the collection's split identity (demo320.split_identity: split name, file, its sha256, plan_sha), which
depth_stats.py and train_wm320.py carry on. ManiSkill data needs --pins. Resumable: an output that opens and has the
source's number of pictures is kept; for AutoMate data it must also hold the source's plan_sha and the entry's
condition and be at least as new as its source (else it is converted again), and the output folder is tied to one
collection and split: <dst>/source.json = {source (resolved), split identity} is written before any copy, and a
non-empty --dst whose source.json is missing or different, or whose manifest.json names another source or split, is
refused (a reused --dst could otherwise keep a file whose old condition is "train" for a grasp now held out). A
source trajectory whose meta plan_sha differs from plan.json's is an error. Completeness (AutoMate): the index's
<src>/index_completeness.json (wm.data.index.build_automate: planned grasps, grasps with grasp.json, live workers,
skipped assemblies, accept_skipped, partial) must exist and carry the sha256 of the index.json read here, else the copy
is refused (build the index again); it is written into the manifest as "completeness", which depth_stats.py carries
and train_wm320.py checks (a partial collection is refused there unless --allow-partial). Converter code (AutoMate,
r6-data-04): the manifest records "code" = {"root": this tree, "md5": {file: md5} of CODE_FILES, "earlier_writers":
the "md5" records of earlier runs into the same --dst that differ from this one}; source.json keeps the list of every
distinct record ("code", appended per run), so a resumed copy that keeps files of an older converter says so.
depth_stats.py carries manifest "code", train_wm320.py stores it in provenance ("fast_copy_code"). Recorded only: a
resume with another converter is not refused. CPU only. The script imports the wm package of its own tree (ROOT/src)
and refuses to run with any other.

    python world_model/scripts/data/cache_wm_v1_fast.py --src $Q/data/wm_v1 --dst /mnt/storage3/qingxin/data/wm_v1_fast \
        --pins /mnt/storage3/qingxin/runs/pin_length_by_port.json --workers 8
AutoMate (after `python <code>/scripts/data/build_index.py --automate <collection out> --workers 8`):
    python <code>/scripts/data/cache_wm_v1_fast.py --src <collection out> --dst <fast dir> --workers 8
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                       # world_model/
sys.path.insert(0, str(ROOT / "src"))
from wm.data.splits import TEST_PORTS  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

CODE_FILES = ("src/wm/tdmpc/demo320.py", "src/wm/data/index.py", "src/wm/data/splits.py",
              "scripts/data/cache_wm_v1_fast.py")    # the converter, the index builder and the split constants


def converter_code() -> dict:
    """{"root": this tree, "md5": {file: md5}} of CODE_FILES (module docstring, converter code)."""
    return {"root": str(ROOT), "md5": {f: hashlib.md5((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}}


def foreign_wm_modules() -> list[str]:
    """Loaded wm.* modules whose file lies outside this code tree (as scripts/automate/collect_expert.py)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT.resolve()):
            bad.append(f"{name} ({f})")
    return bad


def work(job):
    """ManiSkill job (src, dst, pin, rel, n_images); AutoMate job (..., plan_sha, condition): module docstring."""
    src, dst, pin, rel, n_images, *am = job
    try:
        if am:
            plan_sha, condition = am
            with np.load(src) as z:
                sp = json.loads(str(z["meta"])).get("plan_sha")
            if sp != plan_sha:
                raise ValueError(f"source meta plan_sha {sp} differs from plan.json's {plan_sha}")
        if Path(dst).exists():
            with np.load(dst) as z:
                meta, n = json.loads(str(z["meta"])), len(z["img_step"])
            keep = meta.get("format") == D.FORMAT and n == n_images
            if am:
                keep = (keep and meta.get("plan_sha") == plan_sha and D.automate_condition(meta) == condition
                        and os.stat(dst).st_mtime_ns >= os.stat(src).st_mtime_ns)
            if keep:
                return rel, D.entry_of(meta, rel, n), None
        return rel, D.convert(src, dst, pin, rel), None
    except Exception as exc:
        return rel, None, f"{type(exc).__name__}: {exc}"[:300]


def bind_destination(src: Path, dst: Path, split_id: dict, code: dict) -> list[dict]:
    """AutoMate: tie --dst to this collection and split (module docstring); SystemExit for a reused folder. Appends
    code to source.json's "code" list when it differs from the last entry -> the earlier entries whose md5 differs
    from code's (written into this --dst by another converter)."""
    mark = {"source": str(src.resolve()), "split": split_id}
    runs = []
    f = dst / "source.json"
    if dst.exists() and any(dst.iterdir()):
        old = json.loads(f.read_text()) if f.exists() else None
        if old is None or {k: old.get(k) for k in mark} != mark:
            raise SystemExit(f"{dst} is not empty and its source.json is {'missing' if old is None else old}, not {mark}: "
                             "use a new --dst for another collection or split")
        runs = list(old.get("code") or [])
        if (dst / "manifest.json").exists():
            man = json.loads((dst / "manifest.json").read_text())
            if Path(str(man.get("source"))).resolve() != src.resolve() or man.get("split") != split_id:
                raise SystemExit(f"{dst}/manifest.json names the source {man.get('source')} and split {man.get('split')}: "
                                 "use a new --dst")
    if not runs or runs[-1] != code:
        runs.append(code)
    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / "source.json.tmp"
    tmp.write_text(json.dumps({**mark, "code": runs}))
    tmp.replace(f)
    return [r for r in runs if r.get("md5") != code["md5"]]


def automate_completeness(src: Path, index_bytes: bytes) -> dict:
    """The index's completeness record (module docstring); SystemExit if it is missing or not this index's."""
    from wm.data.index import COMPLETENESS_FILE
    f = src / COMPLETENESS_FILE
    if not f.exists():
        raise SystemExit(f"{src}: no {COMPLETENESS_FILE} next to index.json: build the index again with "
                         "scripts/data/build_index.py --automate (it records whether the collection was complete)")
    comp = json.loads(f.read_text())
    sha = hashlib.sha256(index_bytes).hexdigest()
    if comp.get("index_sha256") != sha:
        raise SystemExit(f"{f} describes another index.json (index_sha256 {comp.get('index_sha256')}, this one {sha}): "
                         "build the index again with scripts/data/build_index.py --automate")
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pins", default=None, help="pin length per port (json); ManiSkill data only (AutoMate: from the entry)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    foreign = foreign_wm_modules()
    if foreign:
        raise SystemExit(f"wm modules from outside {ROOT}: {foreign[:3]} (unset PYTHONPATH or run this tree's copy)")
    src, dst = Path(args.src), Path(args.dst)
    jobs, skipped = [], {"test_ports": 0, "no_pictures": 0, "port_without_pin_length": 0}
    index_bytes = (src / "index.json").read_bytes()
    index = json.loads(index_bytes)
    split_id = D.split_identity(src)
    automate = any(D.is_automate(e) for e in index) or split_id is not None
    if automate:
        if not all(D.is_automate(e) for e in index):
            raise SystemExit(f"{src}: AutoMate data, but not every index entry has simulator 'automate': the index was "
                             "not built by wm.data.index.build_automate (scripts/data/build_index.py --automate)")
        if args.pins:
            raise SystemExit("--pins is for ManiSkill data; AutoMate takes the pin length from each entry")
        pins = {}
        splits = [D.split_of_entry(e, allow_undecided=True) for e in index]       # never the index's stored "split"
        undecided = [e["grasp"] for e, s in zip(index, splits) if s == D.UNDECIDED]
        if undecided:
            raise SystemExit(f"{len(undecided)} AutoMate trajectories have no decided split (e.g. {undecided[:3]}): the test "
                             "assemblies are not known, nothing copied")
        if split_id is None or not split_id.get("sha256") or not split_id.get("plan_sha"):
            raise SystemExit(f"{src}: no plan.json with a versioned split file: the split identity of an AutoMate fast "
                             "copy must be known")
        completeness = automate_completeness(src, index_bytes)
    else:
        if not args.pins:
            raise SystemExit("ManiSkill data: --pins is required")
        pins = json.loads(Path(args.pins).read_text())
        splits = [None] * len(index)
    for e, s in zip(index, splits):
        if e.get("pin_length_m") is not None and e["port"] not in pins:
            pins[e["port"]] = e["pin_length_m"]
        p = Path(e["path"])
        rel = str(p.relative_to(src)) if p.is_absolute() else str(p)
        if e["port"] in TEST_PORTS or e.get("split") == "test" or s == "test":
            skipped["test_ports"] += 1
        elif not e["n_images"]:
            skipped["no_pictures"] += 1
        elif e["port"] not in pins:
            skipped["port_without_pin_length"] += 1
        elif automate:
            jobs.append((str(src / rel), str(dst / rel), pins[e["port"]], rel, e["n_images"], split_id["plan_sha"],
                         D.automate_condition(e)))
        else:
            jobs.append((str(src / rel), str(dst / rel), pins[e["port"]], rel, e["n_images"]))
    if automate:
        code = converter_code()
        earlier = bind_destination(src, dst, split_id, code)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} trajectories to copy, skipped {skipped}", flush=True)
    entries, errors, t0 = [], [], time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, (rel, entry, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            if err:
                errors.append((rel, err))
            elif entry is None:
                skipped["no_pictures"] += 1
            else:
                entries.append(entry)
            if (k + 1) % 500 == 0:
                print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] {len(errors)} errors", flush=True)
    entries.sort(key=lambda e: e["path"])
    if errors:
        print(json.dumps({"errors": errors[:20], "count": len(errors)}, indent=1), flush=True)
        raise SystemExit(f"{len(errors)} trajectories failed; no manifest written")
    man = {"format": D.FORMAT, "source": str(src), "skipped": skipped, "entries": entries}
    if automate:                                                      # ManiSkill manifests unchanged (no split key)
        man["split"] = split_id
        man["completeness"] = completeness
        man["code"] = {**code, "earlier_writers": [r.get("md5") for r in earlier]}
    (dst / "manifest.json.tmp").write_text(json.dumps(man))
    (dst / "manifest.json.tmp").replace(dst / "manifest.json")
    print(json.dumps({"trajectories": len(entries), "pictures": sum(e["n_images"] for e in entries), "skipped": skipped,
                      "seconds": round(time.time() - t0),
                      **({"completeness": {k: man["completeness"].get(k) for k in ("planned", "with_grasp_json", "partial")}}
                         if automate else {})}), flush=True)


if __name__ == "__main__":
    main()
