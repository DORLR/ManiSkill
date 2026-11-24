"""One collection worker (one Isaac app): expert demonstrations on AutoMate in the wm_v1 format (wm.sim_automate.collect).

    python -u collect_expert.py --out <out> --config <out>/inputs/config.json --index 0 --count 3
    python -u collect_expert.py --out <out> --config <out>/inputs/config.json --preflight      (no Isaac, CPU only)
Started by collect_worker.sh (from the run's frozen code copy <out>/code, one capped scope per start); the launchers
collect_launch.sh / collect_add_worker.sh run the preflight first. The plan is deterministic and frozen in
<out>/plan.json and config.json by the first process; every worker rebuilds it and refuses to run if it differs
(wm.sim_automate.collect.freeze). Workers claim grasps (flock); `index` / `count` only spread their starting points.

Order: the config (production keys required), the worker resources and the plan are checked BEFORE Isaac starts, so a
config or plan error costs no Isaac start and exits EXIT_REFUSED (3), which the restart loop never retries. Every `wm`
module must come from this file's code tree (the frozen copy), else EXIT_REFUSED.
Exit codes: wm.sim_automate.collect.EXIT_* (0 done, 1 crash, 3 refused, 5 stopped, 6 scene build failure,
7 forbidden GPU-0 context: app.assert_off_forbidden_gpu found this process on GPU 0 beyond one graphics context of at
most 50 MiB; the collection's STOP is written and the restart loop ends, 8 nvidia-smi unreadable at the GPU check: a
counted crash). Exits 7 and 8 leave through os._exit inside app.py; importing wm.sim_automate.collect registers its
PRE_EXIT_HOOKS entry, which first releases the held claim (not a death) and drops the heartbeat.
GPU: only physical GPU 1, as CUDA_VISIBLE_DEVICES=1 (so "cuda:0" in the config is GPU 1); anything else is refused.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} {time.time() - T0:.0f}s] {m}", flush=True)


def foreign_wm_modules() -> list[str]:
    """Loaded wm.* modules whose file lies outside this code tree."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "wm" or name.startswith("wm.")) and f and not Path(f).resolve().is_relative_to(ROOT):
            bad.append(f"{name} ({f})")
    return bad


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--config", required=True, help="json with CollectConfig fields, every production key set")
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--count", type=int, default=1)
ap.add_argument("--preflight", action="store_true", help="check config, resources and plan, freeze it; no Isaac")
try:
    args = ap.parse_args()
except SystemExit:
    os._exit(3)

# ---- before Isaac: config, resources, plan (CPU-only imports: numpy, scipy, PIL)
try:
    from wm.sim_automate import collect as CO
    bad = foreign_wm_modules()
    if bad:
        raise SystemExit(f"wm modules outside the code tree {ROOT}: {bad}")
    cfg = CO.CollectConfig.load(args.config, production=True)
    res = CO.worker_resources(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.preflight and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":    # GPU 0 is forbidden
        raise SystemExit(f"CUDA_VISIBLE_DEVICES must be exactly 1 (GPU 1 only), found "
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    specs, frozen = CO.prepare(cfg, out, log)
    if args.preflight:
        print(f"RESOURCES {res['memory_max']} {res['cpu_quota']} {res['gpu_gb']} {res['hang_timeout_s']}", flush=True)
        print(f"PREFLIGHT OK: {len(specs)} grasps, plan_sha {frozen['plan_sha']}, device {cfg.device}", flush=True)
        os._exit(0)
except SystemExit as exc:
    print(f"REFUSED: {exc}", flush=True)
    os._exit(3)
except Exception:
    traceback.print_exc()
    print("REFUSED: error before the Isaac start (config, inputs or plan)", flush=True)
    os._exit(3)

# ---- Isaac
hb = CO.Heartbeat(out, args.index)
CO.PROCESS_INFO.update(pid=os.getpid(), start=hb.info["start"], unit=hb.info["unit"])
print(f"WORKER PID {os.getpid()} START {hb.info['start']} UNIT {hb.info['unit']}", flush=True)
hb("launch")
t_launch = time.time()
from wm.sim_automate.app import launch, shutdown  # noqa: E402

app = launch(headless=True, cameras=True)
CO.PROCESS_INFO["launch_s"] = round(time.time() - t_launch, 1)
log(f"Isaac started in {CO.PROCESS_INFO['launch_s']} s; memory {CO._memory()}")
CO.install_stop_handlers(log)
code = CO.EXIT_CRASH
try:
    bad = foreign_wm_modules()
    if bad:
        raise CO.Refused(f"wm modules outside the code tree {ROOT}: {bad}")
    CO.worker(cfg, out, args.index, args.count, log=log, beat=hb)
    print("WORKER DONE", flush=True)
    code = CO.EXIT_DONE
except CO.Stopped as exc:
    print(f"WORKER STOPPED: {exc}", flush=True)
    code = CO.EXIT_STOPPED
except CO.BuildFailed as exc:
    print(f"WORKER BUILD FAILED: {exc}", flush=True)
    code = CO.EXIT_BUILD
except SystemExit as exc:                                   # Refused: the frozen plan, inputs or code differ
    print(f"REFUSED: {exc}", flush=True)
    CO.alert(out, f"worker {args.index} refused after the Isaac start: {exc}")
    code = CO.EXIT_REFUSED
except BaseException:
    traceback.print_exc()
    code = CO.EXIT_CRASH
finally:
    hb.close()
shutdown(code)
