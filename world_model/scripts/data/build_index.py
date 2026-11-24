"""Index a collected dataset (splits + per-trajectory port axes).

    python world_model/scripts/data/build_index.py $Q/data/wm_v1 --workers 16
AutoMate collection (wm.data.index.build_automate -> <out>/index.json; no port axes): this tree's wm package
(ROOT/src, put first on sys.path); a wm module loaded from anywhere else is refused, as in cache_wm_v1_fast.py.
    python <code>/scripts/data/build_index.py --automate <collection out> --workers 8 [--accept-skipped] [--partial]
It refuses (exit 2, nothing written) a collection that is still running or unfinished (wm.data.index module docstring,
completeness: live workers or claims, planned grasps without grasp.json); --accept-skipped accepts the missing grasps
of assemblies the workers skip, --partial indexes anyway (a deliberate early look; recorded, and train_wm320.py then
needs --allow-partial). It prints the completeness (planned grasps, grasps with grasp.json, skipped assemblies, partial)
with the index counts.
The ManiSkill form (a positional root) is unchanged: it imports wm from the environment as before.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                       # world_model/


def foreign_wm_modules() -> list[str]:
    """Loaded wm.* modules whose file lies outside this code tree (as scripts/data/cache_wm_v1_fast.py)."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT.resolve()):
            bad.append(f"{name} ({f})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help="a ManiSkill dataset (wm_v1)")
    ap.add_argument("--automate", help="an AutoMate collection's output folder (instead of root)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--accept-skipped", action="store_true",
                    help="AutoMate: accept the missing grasps of assemblies the workers skip (listed in the output)")
    ap.add_argument("--partial", action="store_true",
                    help="AutoMate: index a running or unfinished collection (a deliberate early look; recorded)")
    args = ap.parse_args()
    if (args.root is None) == (args.automate is None):
        raise SystemExit("give either a ManiSkill root or --automate <collection out>")
    if args.automate:
        sys.path.insert(0, str(ROOT / "src"))
        from wm.data import index
        foreign = foreign_wm_modules()
        if foreign:
            raise SystemExit(f"wm modules from outside {ROOT}: {foreign[:3]} (unset PYTHONPATH or run this tree's copy)")
        if not (Path(args.automate) / "plan.json").exists():
            raise SystemExit(f"{args.automate}: no plan.json (not an AutoMate collection output folder)")
        try:
            entries = index.build_automate(args.automate, args.workers, args.accept_skipped, args.partial)
        except index.Incomplete as exc:
            print(f"refused: {exc}", flush=True)
            raise SystemExit(2)
        comp = json.loads((Path(args.automate) / index.COMPLETENESS_FILE).read_text())
        print(json.dumps({"trajectories": len(entries), "splits": dict(Counter(e["split"] for e in entries)),
                          "index": str(Path(args.automate) / "index.json"),
                          "completeness": {k: comp[k] for k in ("planned", "with_grasp_json", "skipped_assemblies",
                                                                "accept_skipped", "partial_flag", "partial")},
                          "problems": comp["problems"]}))
        return
    if args.accept_skipped or args.partial:
        raise SystemExit("--accept-skipped / --partial are for --automate")
    from wm.data import index
    entries = index.build(args.root, args.workers)
    print(len(entries), "trajectories;", dict(Counter(e["split"] for e in entries)),
          "; port axes missing:", sum(e["R_P"] is None for e in entries))


if __name__ == "__main__":
    main()
