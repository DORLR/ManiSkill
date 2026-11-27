"""AutoDL watcher: rule 1 (memory, container-wide) and rule 4 of qx_watch.py (heartbeat hang of a collection worker -> mark its claim, write
<pid>_<start>.hung, SIGKILL the worker's process group), reusing qx_watch.check_heartbeats unchanged. On AutoDL a
worker's "unit" is the process group that caps.sh qx_run (autodl_overrides.sh) started with setsid; our processes are
those in $QX_RUN_DIR/units/*.pid groups. Touches $QX_RUN_DIR/watch.alive every period.
    python3 autodl_watch.py <code>"""
import os, signal, sys, time
from pathlib import Path

code = Path(sys.argv[1])
sys.path.insert(0, str(code / "scripts" / "automate"))
import qx_watch as W                                          # noqa: E402

W.RUN_DIR = Path("/root/autodl-tmp/run")
UNITS = W.RUN_DIR / "units"


def our_groups() -> dict:
    g = {}
    for f in UNITS.glob("*.pid"):
        try:
            g[int(f.read_text().split()[0])] = f.stem
        except (OSError, ValueError):
            pass
    return g


def in_our_scope(pid: int) -> bool:
    try:
        return os.getpgid(pid) in our_groups()
    except ProcessLookupError:
        return False


def kill_hung(pid: int, unit):
    try:
        pg = os.getpgid(pid)
        if pg in our_groups():
            os.killpg(pg, signal.SIGKILL)
            return
    except ProcessLookupError:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


SHED_USED_BYTES = 120_000_000_000          # container memory in use without inactive page cache (limit 124 GiB)


def mem_used() -> int:
    cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
    ina = next(int(l.split()[1]) for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
               if l.startswith("inactive_file "))
    return cur - ina


def shed_newest_collect(used: int):
    """Rule 1 on AutoDL: write $QX_RUN_DIR/shed/<unit>.scope for our newest collection unit and SIGTERM its process
    group (the worker releases its claim and exits 5; online_worker.sh waits and retries), as qx_watch.py does."""
    units = sorted((u for u in our_groups().items() if u[1].startswith("qingxin-collect-")),
                   key=lambda u: int(u[1].rsplit("-", 1)[-1]) if u[1].rsplit("-", 1)[-1].isdigit() else 0)
    if not units:
        print(f"{time.strftime('%F %T')} memory {used} B > {SHED_USED_BYTES} B but no collection unit to shed", flush=True)
        return
    pg, unit = units[-1]
    d = W.RUN_DIR / "shed"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{unit}.scope").write_text(f"{time.strftime('%F %T')} container memory in use {used} B > {SHED_USED_BYTES} B\n")
    try:
        os.killpg(pg, signal.SIGTERM)
    except ProcessLookupError:
        pass
    W.alert(f"shed {unit}: container memory in use {used} B > {SHED_USED_BYTES} B")
    print(f"{time.strftime('%F %T')} shed {unit} (memory {used} B)", flush=True)


W.in_our_scope = in_our_scope
W.kill_hung = kill_hung
print(f"{time.strftime('%F %T')} autodl_watch: start (qx_watch rule 4 only), code {code}", flush=True)
while True:
    try:
        (W.RUN_DIR / "watch.alive").touch()
        used = mem_used()
        if used > SHED_USED_BYTES:
            shed_newest_collect(used)
        for out in W.registered_outs():
            if (out / "heartbeat").is_dir():
                for a in W.check_heartbeats(out):
                    print(f"{time.strftime('%F %T')} {a}", flush=True)
    except Exception as exc:                                  # noqa: BLE001 - never end the watcher
        print(f"{time.strftime('%F %T')} autodl_watch: check failed: {exc!r}", flush=True)
    time.sleep(30)
