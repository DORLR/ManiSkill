"""Start / stop the Isaac Sim app. Isaac Lab and pxr modules import only after launch().

Run every script with `python -u`: SimulationApp.close() may exit hard (buffered prints are lost) and, measured on
rs2 2026-11-21, can also hang after "Replicator shutdown finished"; shutdown() therefore flushes and leaves with
os._exit, which also frees the GPU.

GPU selection. CUDA_VISIBLE_DEVICES=1 restricts CUDA only. Kit's renderer enumerates GPUs through Vulkan, which does
not read CUDA_VISIBLE_DEVICES: measured on rs2 2026-11-22, a smoke process held a graphics context (type G, 8 MiB)
on GPU 0 although CUDA_VISIBLE_DEVICES=1. Kit's GPU table lists both GPUs (Vulkan, PCI order: row 0 = bus 16 = GPU 0,
row 1 = bus ac = GPU 1) and skips row 0 ("CUDA in bad state", i.e. not CUDA-visible); --/renderer/activeGpu is a row
of that table: with multi-GPU off, activeGpu 0 gave "activeGpu 0 is not compatible with current foundation settings"
and "No device could be created" (rs2 2026-11-23, no renderer); with multi-GPU on Kit had fallen back to row 1.
launch() passes Kit (appended last on Kit's command line, so they override SimulationApp's own values):
    --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false   one render GPU, no multi-GPU set-up
    --/renderer/activeGpu=KIT_GPU          config.KIT_GPU (env WM_KIT_GPU, default 1 = row 1 of Kit's table = physical
                                           GPU 1; AppLauncher's own value, the CUDA index 0, is overridden)
    --/persistent/physics/numThreads=N --/plugins/carb.tasking.plugin/threadCount=N
    --/plugins/omni.tbb.globalcontrol/maxThreadCount=N      CPU threads (N = config.PHYSX_CPU_THREADS; SimulationApp's
                                           own default for the last two is 32 per process)
and sets SimulationApp.DEFAULT_LAUNCHER_CONFIG["limit_cpu_threads"] = N before the start: SimulationApp writes
min(cpu_count, limit_cpu_threads) into os.environ PXR_WORK_THREAD_LIMIT (USD work pool) and OPENBLAS_NUM_THREADS
(isaacsim simulation_app.py, rs2 Isaac Sim 5.1: default 32), so without it every process had a 32-thread USD pool
next to N PhysX threads. thread_env() returns the values in force; the batch and the smoke record them. OpenBLAS reads
its variable when numpy is first imported, so a script that imports numpy before launch() keeps the value of its own
environment for OpenBLAS (collect_expert.py, batch_assemblies.py: OMP_NUM_THREADS / OPENBLAS_NUM_THREADS of the
launcher).
GPU 0 rule (the user's decision 2026-11-22: "8 MB does not matter much"). On GPU 0 (config.FORBIDDEN_GPU) a process of
ours may hold ONLY graphics contexts (nvidia-smi type G) of at most config.FORBIDDEN_GPU_GRAPHICS_MAX_MIB = 50 MiB in
total, and NO compute context (type C or C+G). assert_off_forbidden_gpu() reads `nvidia-smi -i 0 -q -d PIDS` (type and
used memory per PID, graphics included, PCI order) after the app start, after every scene build and after the first
rendered frame, and
    exit 7 (EXIT_FORBIDDEN_GPU)   this process holds a compute context on GPU 0, or graphics contexts above 50 MiB:
                                  a real violation; every launcher stops the whole run, no restart;
    exit 8 (EXIT_GPU_UNREADABLE)  nvidia-smi gave no answer (error, timeout) or no used-memory value for this
                                  process: handled like a crash (counted, rate-capped), not as a violation.
Before either exit every callable in PRE_EXIT_HOOKS is called with the exit code (each in its own try / except), so
that the owner of a claim or a marker (collect.py, batch_assemblies.py) releases it first: such an exit is never booked
as a crash of the data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable

EXIT_FORBIDDEN_GPU = 7
EXIT_GPU_UNREADABLE = 8
# Called as hook(exit_code) right before an exit of assert_off_forbidden_gpu (module docstring). Registered by the
# owners of claims / markers; a hook must be quick and must not raise (exceptions are printed and ignored).
PRE_EXIT_HOOKS: list[Callable[[int], None]] = []


def kit_args() -> list[str]:
    from . import config as C
    n = int(C.PHYSX_CPU_THREADS)
    return ["--/renderer/multiGpu/enabled=false", "--/renderer/multiGpu/autoEnable=false",
            f"--/renderer/activeGpu={int(C.KIT_GPU)}", f"--/persistent/physics/numThreads={n}",
            f"--/plugins/carb.tasking.plugin/threadCount={n}", f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={n}"]


def parse_pids(text: str) -> dict[int, list[dict]]:
    """`nvidia-smi -i <gpu> -q -d PIDS` text -> {pid: [{"type": "G" | "C" | "C+G" | ..., "mib": float | None}, ...]}
    (one entry per context; "mib" None when nvidia-smi gives no number, e.g. "N/A")."""
    out: dict[int, list[dict]] = {}
    cur = None
    for line in text.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "Process ID":
            try:
                cur = {"pid": int(val), "type": None, "mib": None}
            except ValueError:
                cur = None
                continue
            out.setdefault(cur["pid"], []).append(cur)
        elif cur is not None and key == "Type":
            cur["type"] = val
        elif cur is not None and key == "Used GPU Memory":
            num = val.split()[0] if val else ""
            try:
                cur["mib"] = float(num) * (1024.0 if val.endswith("GiB") else 1.0)
            except ValueError:
                cur["mib"] = None
    return {pid: [{k: e[k] for k in ("type", "mib")} for e in es] for pid, es in out.items()}


def contexts_on_gpu(index: int) -> dict[int, list[dict]] | None:
    """parse_pids of the physical GPU `index` (nvidia-smi, PCI order); None if nvidia-smi fails or times out."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    try:
        res = subprocess.run(["nvidia-smi", "-i", str(index), "-q", "-d", "PIDS"], capture_output=True, text=True,
                             env=env, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return parse_pids(res.stdout)


def pids_on_gpu(index: int) -> set[int] | None:
    """PIDs with any context (compute or graphics) on the physical GPU `index`; None if nvidia-smi fails."""
    c = contexts_on_gpu(index)
    return None if c is None else set(c)


def gpu_rule(contexts: list[dict] | None, max_graphics_mib: float) -> tuple[int, str]:
    """The GPU 0 rule (module docstring) for one process's contexts on that GPU -> (exit code or 0, reason):
        0  no context, or only type-G contexts with a known total <= max_graphics_mib
        7  a context of another type (C, C+G, ...), or type-G contexts above max_graphics_mib in total
        8  a type-G context without a used-memory value (cannot show that the rule holds)."""
    if not contexts:
        return 0, "no context"
    bad = [c["type"] for c in contexts if c["type"] != "G"]
    if bad:
        return EXIT_FORBIDDEN_GPU, f"compute context (type {', '.join(str(b) for b in bad)})"
    if any(c["mib"] is None for c in contexts):
        return EXIT_GPU_UNREADABLE, "graphics context without a used-memory value"
    total = sum(c["mib"] for c in contexts)
    if total > max_graphics_mib:
        return EXIT_FORBIDDEN_GPU, f"graphics contexts {total:.0f} MiB > {max_graphics_mib:.0f} MiB"
    return 0, f"graphics context only, {total:.0f} MiB <= {max_graphics_mib:.0f} MiB (accepted)"


def exit_now(code: int, why: str = ""):
    """Print why, run PRE_EXIT_HOOKS(code) (each in its own try / except), then shutdown(code)."""
    if why:
        print(why, flush=True)
    for hook in list(PRE_EXIT_HOOKS):
        try:
            hook(code)
        except BaseException as exc:                                  # noqa: BLE001 - a hook must never block the exit
            print(f"pre-exit hook {getattr(hook, '__name__', hook)!r} failed: {type(exc).__name__}: {exc}", flush=True)
    shutdown(code)


def assert_off_forbidden_gpu(where: str = "") -> dict:
    """Apply the GPU 0 rule of the module docstring to this process; exits (7 or 8, after PRE_EXIT_HOOKS) when it
    does not hold, else returns {"gpu", "contexts", "reason"}."""
    from . import config as C
    bad = int(C.FORBIDDEN_GPU)
    ctx = contexts_on_gpu(bad)
    pid = os.getpid()
    if ctx is None:
        exit_now(EXIT_GPU_UNREADABLE, f"GPU check {where}: nvidia-smi failed; exit {EXIT_GPU_UNREADABLE} (cannot show "
                                      f"that the GPU {bad} rule holds)")
    mine = ctx.get(pid, [])
    code, reason = gpu_rule(mine, C.FORBIDDEN_GPU_GRAPHICS_MAX_MIB)
    if code:
        exit_now(code, f"GPU check {where}: pid {pid} on GPU {bad}: {reason} {mine}; exit {code}")
    print(f"GPU check {where}: pid {pid} on GPU {bad}: {reason}", flush=True)
    return {"gpu": bad, "contexts": mine, "reason": reason}


THREAD_ENV_KEYS = ("PXR_WORK_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def thread_env() -> dict:
    """The thread-count variables of this process (None when unset); recorded with batch and smoke results."""
    return {k: os.environ.get(k) for k in THREAD_ENV_KEYS}


def launch(headless: bool = True, cameras: bool = True):
    """-> the SimulationApp (headless, with the RTX renderer enabled for cameras when `cameras`), on the render GPU of
    the module docstring, its own thread pools limited to config.PHYSX_CPU_THREADS; the GPU 0 rule checked right after
    the start."""
    from . import config as C
    from isaaclab.app import AppLauncher
    from isaacsim import SimulationApp
    SimulationApp.DEFAULT_LAUNCHER_CONFIG["limit_cpu_threads"] = int(C.PHYSX_CPU_THREADS)
    app = AppLauncher(headless=headless, enable_cameras=cameras, kit_args=" ".join(kit_args())).app
    assert_off_forbidden_gpu("after the app start")
    return app


def shutdown(code: int = 0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
