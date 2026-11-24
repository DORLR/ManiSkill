"""Depth statistics of a fast copy for the new picture encoders (policy320.PictureTokens resnet34 / segformer_b1,
encoders320.depth_z), from the TRAINING split only, written to <data>/depth_stats.json (read by train_wm320.py).

Pixels: every depth picture of every training trajectory (demo320.load_entries, split == "train" and at least one
decision of WD.EXECUTE steps: the trajectories train_wm320.py trains on; the split of AutoMate data is each grasp's
stored condition, demo320.split_of_entry, and an undecided split is refused); valid = stored depth > 0 (0 = no return or
nearer than the D405's minimum range). With d = min(depth * 1e-4 m, D_max) over the N valid pixels:
    mean = sum(d) / N,   std = sqrt(sum(d^2) / N - mean^2),   invalid = (0 - mean) / std
(invalid: the z of 0 m; it lies below the smallest valid z, (DEPTH_MIN_VALID_M - mean) / std, checked). D_max is
the user's choice (--depth-max-m, no default); it is refused before any picture is read unless
0 < D_max <= policy320.DEPTH_MAX_M = 3.2767 m, the int16 ceiling of the 0.1 mm depth input at which the readers clip
(demo320.policy_sample, wm320.WindowData): with a larger D_max the statistics would cover depths the model never sees.
(scripts/automate/prep_automate_data.sh refuses the same range before its first step, r6-data-01; ops test 17 compares
its literal with DEPTH_MAX_M. This check stays for a direct call.)
The manifest's "completeness" (cache_wm_v1_fast.py: planned grasps, grasps with grasp.json, skipped assemblies,
partial) and "code" (the converter's md5 record, cache_wm_v1_fast.converter_code; None for ManiSkill data) are copied
into the output; train_wm320.py reads both from the manifest. Also written: the pixel counts, the
number of pictures and demo320.train_files_sha256 of the training entries (manifest source; per entry path, pictures,
steps, condition and file size; for an AutoMate fast copy also the manifest's split identity), so train_wm320.py
refuses the file for another
split or for files rewritten since; and the manifest's split identity itself ("split": demo320.split_identity of the
collection, sha256 of split_automate_v1.json; None for ManiSkill data).

Unclipped distribution (r9-data-01; for the user's D_max question, decision 5): over the same training pixels, a
histogram h[k] = number of valid pixels with stored depth k (0.1 mm units, k = 1..65535, before any clipping), c =
cumsum(h), n = c[-1]: p_q = k_q * 1e-4 m with k_q = the smallest k with c[k] >= ceil(q n / 100) (numpy's
percentile method 'inverted_cdf') for q = 50, 90, 95, 99, 99.9; share_above_m[x] = (number of valid pixels with depth
> x m) / n for x = 0.3, 0.5, 1.0, 3.2767. Written to <data>/depth_distribution.json (with the split identity and
train_files_sha256) BEFORE the check_depth_settings refusal, so that a refused smoke still leaves the numbers on disk,
and copied into depth_stats.json as "distribution_unclipped". Only these training-split numbers may be used to choose
D_max: check_expert_data.py's depth_mm_p1_p99 covers every condition (held-out and test pictures too).

    python world_model/scripts/automate/depth_stats.py --data <fast copy> --depth-max-m <D_max m>
(the collector smoke uses 0.5 as a smoke-only value; the production D_max is asked from the user after the smoke)
Distribution-only mode (r10-data-01; for the production collection's numbers before the user has chosen D_max):
    python world_model/scripts/automate/depth_stats.py --data <fast copy> --distribution-only
reads the same training pictures with the same histogram loop, writes ONLY depth_distribution.json (next to --out,
default <data>/depth_distribution.json) and never writes a depth_stats.json, so no statistics file with a
placeholder D_max can exist for train_wm320.py to read (it refuses without one). --depth-max-m is refused in this
mode; check_depth_settings is not called (there are no clipped statistics). prep_automate_data.sh runs this mode
for a D_max of "none" and also deletes an existing <fast>/depth_stats.json first.
A run with --depth-max-m deletes an existing output file (--out, default <data>/depth_stats.json) before its D_max
check, so a refused or failed run (check_depth_max, check_depth_settings, the caps, an out-of-memory kill) leaves no
statistics of an earlier D_max for train_wm320.py to accept (r11-data-01); prep_automate_data.sh deletes it too.
It imports the wm package of its own tree (ROOT/src) and refuses to run with a wm module from anywhere else.
Production order (CPU, from the frozen code copy): scripts/data/build_index.py --automate -> cache_wm_v1_fast.py ->
this script -> train_wm320.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import DEPTH_MAX_M, DEPTH_MIN_VALID_M, check_depth_settings  # noqa: E402


def train_entries(data: str) -> list[dict]:
    """The training trajectories, selected as train_wm320.py selects them. An undecided AutoMate split: SystemExit."""
    try:
        ents = D.load_entries(data)
    except D.SplitUndecided as exc:
        raise SystemExit(f"{exc} (demo320.split_of_entry): no depth statistics without a decided training split")
    return [e for e in ents if e["split"] == "train" and e["steps"] >= WD.EXECUTE]


def check_depth_max(depth_max_m: float) -> None:
    """SystemExit unless 0 < D_max <= DEPTH_MAX_M (module docstring); called before any picture is read."""
    if not 0 < depth_max_m <= DEPTH_MAX_M:
        raise SystemExit(f"refused: --depth-max-m {depth_max_m} must lie in (0, {DEPTH_MAX_M}] m (the int16 ceiling of the "
                         "0.1 mm depth input, where the readers clip)")


DIST_Q = (50.0, 90.0, 95.0, 99.0, 99.9)              # percentiles of the unclipped distribution (module docstring)
DIST_ABOVE_M = (0.3, 0.5, 1.0, 3.2767)               # shares of valid pixels deeper than these depths [m]


def distribution(h: np.ndarray) -> dict:
    """The unclipped distribution of a histogram h over stored depth k (0.1 mm units, h[0] unused): n_valid, p50 ...
    p99_9 [m] (inverted_cdf) and share_above_m (module docstring); exact integer arithmetic on the counts."""
    h = np.asarray(h, np.int64).copy()
    h[0] = 0
    c = np.cumsum(h)
    n = int(c[-1])
    out = {"n_valid": n}
    for q in DIST_Q:
        key = "p" + (f"{q:g}".replace(".", "_")) + "_m"
        if n == 0:
            out[key] = None
            continue
        qt = int(round(q * 10))                                 # q in tenths of a percent
        need = -(-qt * n // 1000)                               # ceil(q n / 100)
        out[key] = int(np.searchsorted(c, max(need, 1), side="left")) * 1e-4
    out["share_above_m"] = {f"{x:g}": (None if n == 0 else int(n - c[int(round(x * 1e4))]) / n) for x in DIST_ABOVE_M}
    return out


def write_atomic(f: Path, text: str) -> None:
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, f)


def compute(data: str, depth_max_m: float | None, dist_out: str | Path | None = None,
            distribution_only: bool = False) -> dict:
    """The statistics (module docstring). The unclipped distribution is written to dist_out (default
    <data>/depth_distribution.json) before check_depth_settings can refuse. distribution_only (depth_max_m None):
    only that file is written and the distribution is returned (no clipped statistics, no check_depth_settings)."""
    if distribution_only:
        if depth_max_m is not None:
            raise SystemExit("refused: --distribution-only takes no D_max (it writes no depth_stats.json)")
    else:
        check_depth_max(depth_max_m)
    ents = train_entries(data)
    if not ents:
        raise SystemExit(f"{data}: no training trajectories")
    hist = np.zeros(65536, np.int64)                       # stored depth in 0.1 mm units, before clipping
    n = n_all = pics = 0
    s = ss = 0.0
    dmin = np.inf
    for e in ents:
        with np.load(Path(data) / e["path"]) as z:
            blob, off = z["depth_png"], z["depth_offsets"]
        for i in range(len(off) - 1):
            dep = D.decode_depth(blob, off, i)
            raw = dep[dep > 0].ravel()
            hist += np.bincount(raw, minlength=65536).astype(np.int64)
            v = raw.astype(np.float64) * 1e-4
            pics += 1
            n_all += dep.size
            if v.size and not distribution_only:
                dmin = min(dmin, float(v.min()))
                v = np.minimum(v, depth_max_m)
                n += v.size
                s += float(v.sum())
                ss += float((v * v).sum())
    man = json.loads((Path(data) / "manifest.json").read_text())
    train_sha = D.train_files_sha256(ents, data)
    dist = {"format": "depth-distribution-1", "data": str(data), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "what": "unclipped depth of the valid pixels of the TRAINING split only (as depth_stats.json); the only "
                    "numbers to choose D_max from (check_expert_data's depth_mm_p1_p99 covers every condition)",
            "n_entries": len(ents), "n_pictures": pics, **distribution(hist), "split": man.get("split"),
            "train_files_sha256": train_sha}
    write_atomic(Path(dist_out) if dist_out else Path(data) / "depth_distribution.json", json.dumps(dist, indent=1))
    if distribution_only:
        return dist
    if n == 0:
        raise SystemExit(f"{data}: no valid depth pixel in the training pictures")
    mean = s / n
    std = float(np.sqrt(max(ss / n - mean * mean, 0.0)))
    invalid = (0.0 - mean) / std
    out = {"format": "depth-stats-1", "data": str(data), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "split_rule": "demo320.load_entries split == 'train' and steps >= wmdata320.EXECUTE (as train_wm320.py)",
           "n_entries": len(ents), "train_files_sha256": train_sha, "n_pictures": pics,
           "n_pixels": n_all, "n_valid_pixels": n, "valid_fraction": n / n_all, "smallest_valid_depth_m": dmin,
           "depth_max_m": float(depth_max_m), "depth_mean_m": mean, "depth_std_m": std, "depth_invalid": invalid,
           "smallest_valid_z": (DEPTH_MIN_VALID_M - mean) / std,
           "distribution_unclipped": {k: v for k, v in dist.items() if k not in ("format", "data", "created", "what")},
           **{k: man.get(k) for k in ("split", "completeness", "code")}}
    why = check_depth_settings(depth_max_m, mean, std, invalid)
    if why:
        raise SystemExit(f"unusable statistics: {why}: {out}")
    return out


def foreign_wm_modules() -> list[str]:
    """Loaded wm.* modules whose file lies outside this code tree (as scripts/data/cache_wm_v1_fast.py)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT.resolve()):
            bad.append(f"{name} ({f})")
    return bad


def main():
    foreign = foreign_wm_modules()
    if foreign:
        raise SystemExit(f"wm modules from outside {ROOT}: {foreign[:3]} (unset PYTHONPATH or run this tree's copy)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="fast copy (manifest.json + trajectories)")
    ap.add_argument("--depth-max-m", type=float, help="D_max: depth clipped to [0, D_max] m (required unless "
                                                      "--distribution-only)")
    ap.add_argument("--distribution-only", action="store_true",
                    help="write only depth_distribution.json, never depth_stats.json (module docstring)")
    ap.add_argument("--out", help="default <data>/depth_stats.json (depth_distribution.json is written next to it)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(args.data) / "depth_stats.json"
    dist_out = out.with_name("depth_distribution.json")
    if args.distribution_only:
        if args.depth_max_m is not None:
            raise SystemExit("refused: --distribution-only takes no --depth-max-m (it writes no depth_stats.json)")
        res = compute(args.data, None, dist_out=dist_out, distribution_only=True)
        print(json.dumps(res, indent=1))
        print(f"distribution only: {dist_out} written, no depth_stats.json; training needs depth_stats.py "
              "--depth-max-m <the user's D_max> first")
        return
    if args.depth_max_m is None:
        raise SystemExit("refused: --depth-max-m is required (the user's D_max; no default) unless --distribution-only")
    out.unlink(missing_ok=True)          # r11-data-01: a refused or failed run leaves no file of an earlier D_max
    check_depth_max(args.depth_max_m)
    res = compute(args.data, args.depth_max_m, dist_out=dist_out)
    write_atomic(out, json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
