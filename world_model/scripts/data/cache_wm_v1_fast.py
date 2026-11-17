"""Write the fast copy of the demonstrations (wm/tdmpc/demo320.py, format "wm-fast-1") and its manifest.
Reads the source's index.json (paths absolute or relative to --src). Ports A-D are skipped (the final test must
stay untouched), trajectories without pictures are skipped and counted. Resumable: an output that opens and has the
source's number of pictures is kept. CPU only.

    python world_model/scripts/data/cache_wm_v1_fast.py --src $Q/data/wm_v1 --dst /mnt/storage3/qingxin/data/wm_v1_fast \
        --pins /mnt/storage3/qingxin/runs/pin_length_by_port.json --workers 8
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from wm.data.splits import TEST_PORTS
from wm.tdmpc import demo320 as D


def work(job):
    src, dst, pin, rel, n_images = job
    try:
        if Path(dst).exists():
            with np.load(dst) as z:
                meta, n = json.loads(str(z["meta"])), len(z["img_step"])
            if meta.get("format") == D.FORMAT and n == n_images:
                return rel, D.entry_of(meta, rel, n), None
        return rel, D.convert(src, dst, pin, rel), None
    except Exception as exc:
        return rel, None, f"{type(exc).__name__}: {exc}"[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    pins = json.loads(Path(args.pins).read_text())
    jobs, skipped = [], {"test_ports": 0, "no_pictures": 0, "port_without_pin_length": 0}
    for e in json.loads((src / "index.json").read_text()):
        p = Path(e["path"])
        rel = str(p.relative_to(src)) if p.is_absolute() else str(p)
        if e["port"] in TEST_PORTS:
            skipped["test_ports"] += 1
        elif not e["n_images"]:
            skipped["no_pictures"] += 1
        elif e["port"] not in pins:
            skipped["port_without_pin_length"] += 1
        else:
            jobs.append((str(src / rel), str(dst / rel), pins[e["port"]], rel, e["n_images"]))
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
    (dst / "manifest.json.tmp").write_text(json.dumps({"format": D.FORMAT, "source": str(src), "skipped": skipped, "entries": entries}))
    (dst / "manifest.json.tmp").replace(dst / "manifest.json")
    print(json.dumps({"trajectories": len(entries), "pictures": sum(e["n_images"] for e in entries), "skipped": skipped,
                      "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
