"""AutoDL watcher: rule 4 of qx_watch.py only (heartbeat hang of a collection worker -> mark its claim, write
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
            g[int(f.read_text().strip())] = f.stem
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


W.in_our_scope = in_our_scope
W.kill_hung = kill_hung
print(f"{time.strftime('%F %T')} autodl_watch: start (qx_watch rule 4 only), code {code}", flush=True)
while True:
    try:
        (W.RUN_DIR / "watch.alive").touch()
        for out in W.registered_outs():
            if (out / "heartbeat").is_dir():
                for a in W.check_heartbeats(out):
                    print(f"{time.strftime('%F %T')} {a}", flush=True)
    except Exception as exc:                                  # noqa: BLE001 - never end the watcher
        print(f"{time.strftime('%F %T')} autodl_watch: check failed: {exc!r}", flush=True)
    time.sleep(30)
