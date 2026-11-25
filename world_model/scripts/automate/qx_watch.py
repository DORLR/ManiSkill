"""Watcher for our jobs on rs2 (standard library only). Started by collect_lib.sh cl_watch_ensure in its own scope
(caps.sh qx_run watch: MemoryMax 256M, CPUQuota 5 %, no GPU), one instance for all our runs. Every PERIOD_S:

  1. memory: full avg10 of user@<uid>.service/memory.pressure (the cgroup systemd-oomd watches) > PSI_FULL_STOP, or
     MemAvailable < MEMAVAIL_STOP_BYTES  ->  stop our newest unit of a stoppable kind (not watch / test / prep)
  2. GPU 1: sum over our units of max(declared GPU GB, measured GPU-1 memory of the unit's PIDs from the full process
     table, nvidia-smi -q -d PIDS) > 72 GB  ->  stop our newest unit that holds GPU-1 memory or declared some;
     GPU-1 memory.free < GPU1_FREE_STOP_BYTES (8e9) while our measured GPU-1 sum > 0  ->  stop our newest unit that
     holds GPU-1 memory (other users keep room)
  3. GPU 0: a PID of one of our units holds on GPU 0 more than graphics contexts (nvidia-smi type G) of at most
     GPU0_G_MAX_MIB (50 MiB) in total, or any compute context (type C / C+G / other, or no readable type)  ->  stop
     that unit (the user's rule of 2026-11-22: an idle 8 MiB renderer context is accepted, nothing beyond it); a
     graphics context without a used-memory value only gives an ALERT line (as app.py's exit 8: unreadable, not a
     violation)
  4. heartbeats of every registered collection (caps.sh QX_RUN_DIR/watch/*.dir -> <out>/heartbeat/<pid>_<start>.json):
     the file is older than its timeout_s while that pid still has that start time and lives in one of our
     qingxin-* scopes  ->  append "hang <time>" to its claim file (if the claim is still that pid's), write
     <pid>_<start>.hung, SIGKILL the worker's scope (systemctl --user kill --signal=SIGKILL <unit>, when the unit is a
     qingxin-collect-*.scope that holds the pid; else the pid). The kernel then releases the claim's flock; the next
     worker counts the hang as a death of that grasp. Files of dead pids are removed.
"Stop" is `systemctl --user stop --no-block <exact unit name>` (the worker releases its claim on SIGTERM and exits 5).
Before a stop by 1 or 2 the watcher writes QX_RUN_DIR/shed/<unit> (the restart loop waits and retries: a shed, not a
failure); before a stop by 3 it writes QX_RUN_DIR/gpu0/<unit> and <out>/STOP of every registered collection whose
heartbeat names that unit (a GPU-0 violation stops the run; no restart).
At most one stop for 1-3 per period. Every action is a line in QX_RUN_DIR/ALERT (and <out>/logs/ALERT for 3 and 4).
At its start the watcher writes QX_RUN_DIR/watch/code.sha256 (sha256 of this file): collect_lib.sh cl_watch_ensure
warns when a launcher's frozen qx_watch.py differs from the live one (one watcher serves all our runs).
A failing reading or check never ends the watcher (ALERT, next period). It touches QX_RUN_DIR/watch.alive every period
(the launchers start no worker while that file is older than 180 s). It exits only when for IDLE_EXIT_S no other unit
of ours ran AND no launcher asked for it (QX_RUN_DIR/watch/wanted, touched by collect_lib.sh cl_watch_ensure, older
than IDLE_EXIT_S).
Processes are identified by unit name + cgroup and by pid + start time, never by a process name.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

RUN_DIR = Path("/mnt/storage3/qingxin/run")
CGROOT = Path("/sys/fs/cgroup")
PROC = Path("/proc")
PERIOD_S = 60
PSI_FULL_STOP = 30.0
MEMAVAIL_STOP_BYTES = 10_000_000_000
GPU_OURS_MAX_MIB = 68664                   # 72e9 bytes (was 48e9; raised 2026-11-25, the user's plan B)
GPU0_G_MAX_MIB = 50                        # per pid on GPU 0: graphics contexts only, at most 50 MiB in total
GPU1_FREE_STOP_BYTES = 8_000_000_000
NOT_STOPPABLE = {"watch", "test", "prep"}
IDLE_EXIT_S = 1800
COLLECT_UNIT = re.compile(r"^qingxin-collect-[A-Za-z0-9_.-]+\.scope$")


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def alert(msg: str, out: Path | None = None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} qx_watch: {msg}\n"
    for f in ([RUN_DIR / "ALERT"] + ([out / "logs" / "ALERT"] if out is not None else [])):
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "a") as fh:
                fh.write(line)
        except OSError:
            pass
    log("ALERT " + msg)


def run(cmd: list[str]) -> str:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env, check=True).stdout


# ---------------------------------------------------------------------------------------------- readings
def start_time(pid: int) -> str | None:
    try:
        stat = (PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()[19]


def parse_psi(text: str) -> dict:
    """memory.pressure text -> {"some": avg10, "full": avg10}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] in ("some", "full"):
            out[parts[0]] = float(dict(p.split("=") for p in parts[1:])["avg10"])
    return out


def mem_available(meminfo: str) -> int:
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("no MemAvailable")


def parse_pid_table(txt: str) -> dict:
    """nvidia-smi -q -d PIDS text -> {pid: MiB} (C and G entries of one pid summed)."""
    out, pid = {}, None
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        k = k.strip()
        if k == "Process ID":
            pid = int(v.strip())
        elif k == "Used GPU Memory" and pid is not None:
            m = re.match(r"\s*(\d+)", v)
            out[pid] = out.get(pid, 0) + (int(m.group(1)) if m else 0)
    return out


def parse_pid_rows(txt: str) -> dict:
    """nvidia-smi -q -d PIDS text -> {pid: {"types": [type per entry, "?" if missing], "mib": summed MiB or None if
    any entry has no number}}."""
    out, pid = {}, None
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "Process ID":
            try:
                pid = int(v)
            except ValueError:
                pid = None
                continue
            row = out.setdefault(pid, {"types": [], "mib": 0, "entries": 0})
            row["entries"] += 1
        elif pid is not None and k == "Type":
            out[pid]["types"].append(v or "?")
        elif pid is not None and k == "Used GPU Memory":
            m = re.match(r"(\d+(?:\.\d+)?)\s*(MiB|GiB)", v)
            if m is None or out[pid]["mib"] is None:
                out[pid]["mib"] = None
            else:
                out[pid]["mib"] += float(m.group(1)) * (1024 if m.group(2) == "GiB" else 1)
    for row in out.values():
        row["types"] += ["?"] * (row.pop("entries") - len(row["types"]))
    return out


def gpu0_rule(row: dict) -> tuple[int, str]:
    """The user's GPU-0 rule (module docstring, 3) for one pid's rows, as app.gpu_rule: (7, reason) a violation
    (another type than G, or graphics above GPU0_G_MAX_MIB), (8, reason) graphics without a used-memory value
    (unreadable: alert only), (0, "") allowed."""
    if any(t != "G" for t in row["types"]) or not row["types"]:
        return 7, f"types {row['types']} (only graphics contexts are allowed)"
    if row["mib"] is None:
        return 8, "graphics context without a used-memory value"
    if row["mib"] > GPU0_G_MAX_MIB:
        return 7, f"graphics {row['mib']:.0f} MiB > {GPU0_G_MAX_MIB} MiB"
    return 0, ""


def units() -> list[dict]:
    """Our active scopes: name, kind, pids, start (monotonic us), declared GPU MiB."""
    names = [ln.split()[0] for ln in run(["systemctl", "--user", "list-units", "--type=scope", "--state=active",
                                          "--no-legend", "--plain", "qingxin-*"]).splitlines() if ln.strip()]
    out = []
    for n in names:
        props = dict(ln.split("=", 1) for ln in run(["systemctl", "--user", "show", "-p", "ControlGroup", "-p",
                                                      "ActiveEnterTimestampMonotonic", n]).splitlines() if "=" in ln)
        procs = CGROOT / props.get("ControlGroup", "").lstrip("/") / "cgroup.procs"
        pids = [int(p) for p in procs.read_text().split()] if props.get("ControlGroup") and procs.exists() else []
        decl = 0
        g = RUN_DIR / "qx" / f"{n}.gpu"
        if g.exists() and g.read_text().strip().isdigit():
            decl = int(g.read_text().strip()) * 1_000_000_000 // 1048576
        out.append({"name": n, "kind": n.split("-")[1] if n.count("-") >= 2 else "", "pids": pids,
                    "since": int(props.get("ActiveEnterTimestampMonotonic", "0") or 0), "declared_mib": decl})
    return out


def stop(unit: str, why: str):
    try:
        run(["systemctl", "--user", "stop", "--no-block", unit])
        alert(f"stopped {unit}: {why}")
    except (OSError, subprocess.SubprocessError) as exc:
        alert(f"could not stop {unit} ({why}): {exc}")


# ---------------------------------------------------------------------------------------------- decisions (pure)
def decide_units(us: list[dict], psi_full: float, avail: int, gpu0: dict, gpu1: dict,
                 gpu1_free_mib: int | None = None) -> list[tuple[str, str, str]]:
    """-> [(unit, reason, rule)] for this period, rule in gpu0 / memory / gpu_sum / gpu_free (stop the unit) or
    gpu0_unreadable (an ALERT line only); pure: tested on a CPU. gpu0: parse_pid_rows of GPU 0; gpu1: parse_pid_table
    of GPU 1; gpu1_free_mib: GPU 1 memory.free."""
    acts, unread = [], []
    for u in us:                                                    # 3. GPU 0
        rules = {p: gpu0_rule(gpu0[p]) for p in u["pids"] if p in gpu0}
        bad = {p: why for p, (code, why) in rules.items() if code == 7}
        if bad:
            acts.append((u["name"], f"GPU 0 (forbidden beyond <= {GPU0_G_MAX_MIB} MiB graphics): {bad}", "gpu0"))
        hid = {p: why for p, (code, why) in rules.items() if code == 8}
        if hid:
            unread.append((u["name"], f"GPU 0 memory of {sorted(hid)} unreadable (not stopped)", "gpu0_unreadable"))
    if acts:
        return acts + unread
    return _decide_rest(us, psi_full, avail, gpu1, gpu1_free_mib) + unread


def _decide_rest(us: list[dict], psi_full: float, avail: int, gpu1: dict, gpu1_free_mib: int | None) -> list:
    newest = sorted((u for u in us if u["kind"] not in NOT_STOPPABLE), key=lambda u: u["since"], reverse=True)
    if psi_full > PSI_FULL_STOP or avail < MEMAVAIL_STOP_BYTES:     # 1. memory
        if newest:
            return [(newest[0]["name"], f"user service memory pressure full avg10 {psi_full} > {PSI_FULL_STOP} or "
                                        f"MemAvailable {avail} B < {MEMAVAIL_STOP_BYTES} B", "memory")]
    meas = {u["name"]: sum(gpu1.get(p, 0) for p in u["pids"]) for u in us}
    use = {u["name"]: max(u["declared_mib"], meas[u["name"]]) for u in us}
    total = sum(use.values())
    if total > GPU_OURS_MAX_MIB:                                    # 2. GPU 1, our sum
        gpu_users = [u for u in newest if use[u["name"]] > 0]
        if gpu_users:
            return [(gpu_users[0]["name"], f"our GPU-1 memory {total} MiB > {GPU_OURS_MAX_MIB} MiB (72 GB)", "gpu_sum")]
    if gpu1_free_mib is not None and gpu1_free_mib * 1048576 < GPU1_FREE_STOP_BYTES and sum(meas.values()) > 0:
        holders = [u for u in newest if meas[u["name"]] > 0]        # 2. GPU 1, room for other users
        if holders:
            return [(holders[0]["name"], f"GPU 1 has {gpu1_free_mib} MiB free < {GPU1_FREE_STOP_BYTES} B while we "
                                         f"hold {sum(meas.values())} MiB", "gpu_free")]
    return []


def in_our_scope(pid: int) -> bool:
    try:
        cg = (PROC / str(pid) / "cgroup").read_text().strip().split("::")[-1]
    except OSError:
        return False
    last = cg.rstrip("/").split("/")[-1]
    return last.startswith("qingxin-") and last.endswith(".scope")


def pid_unit(pid: int) -> str | None:
    try:
        cg = (PROC / str(pid) / "cgroup").read_text().strip().split("::")[-1]
    except OSError:
        return None
    return cg.rstrip("/").split("/")[-1] or None


def kill_hung(pid: int, unit: str | None):
    """SIGKILL the whole scope of a hung worker (children too) when `unit` is one of our collect scopes and holds the
    pid; otherwise the pid alone. A process that is already gone is fine."""
    if unit and COLLECT_UNIT.match(unit) and pid_unit(pid) == unit:
        try:
            run(["systemctl", "--user", "kill", "--signal=SIGKILL", unit])
            return
        except (OSError, subprocess.SubprocessError) as exc:
            alert(f"systemctl kill {unit} failed ({exc}); killing pid {pid}")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def check_heartbeats(out: Path, now: float | None = None) -> list[str]:
    """Rule 4 for one collection folder; -> the actions taken."""
    now = time.time() if now is None else now
    acts = []
    for f in sorted((out / "heartbeat").glob("*.json")):
        try:
            hb = json.loads(f.read_text())
            pid, start, timeout = int(hb["pid"]), str(hb["start"]), int(hb["timeout_s"])
            age = now - f.stat().st_mtime
        except (OSError, ValueError, KeyError):
            continue
        if start_time(pid) != start:                                # the process is gone: stale file
            f.unlink(missing_ok=True)
            continue
        if age <= timeout:
            continue
        if not in_our_scope(pid):
            alert(f"heartbeat {f} is {age:.0f} s old but pid {pid} is not in a qingxin-* scope; not touched", out)
            continue
        marked = False
        cp = hb.get("claim")
        if cp and Path(cp).exists():
            try:
                with open(cp, "r+") as fh:
                    txt = fh.read()
                    parts = txt.split()
                    kv = {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}
                    if kv.get("pid") == str(pid) and kv.get("start") == start and "hang" not in kv:
                        fh.write(f" hang {time.strftime('%Y-%m-%d_%H:%M:%S')}")
                        fh.flush()
                        os.fsync(fh.fileno())
                        marked = True
            except OSError:
                pass
        marker = f.with_suffix(".hung")
        marker.write_text(json.dumps({**hb, "age_s": round(age), "killed": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "claim_marked": marked}))
        if start_time(pid) == start:                                # still the same process
            kill_hung(pid, hb.get("unit"))
        f.unlink(missing_ok=True)
        msg = (f"worker pid {pid} (unit {hb.get('unit')}, stage {hb.get('stage')!r}, grasp {hb.get('grasp')}) sent no "
               f"heartbeat for {age:.0f} s > {timeout} s: killed, claim {'marked hang' if marked else 'not marked'}")
        alert(msg, out)
        acts.append(msg)
    return acts


# ---------------------------------------------------------------------------------------------- loop
def gpu_uuid(i: int) -> str:
    for ln in run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        a, b = [x.strip() for x in ln.split(",")]
        if a == str(i):
            return b
    raise RuntimeError(f"GPU {i} not found")


def registered_outs() -> list[Path]:
    out = []
    for f in sorted((RUN_DIR / "watch").glob("*.dir")):
        try:
            out.append(Path(f.read_text().strip()))
        except OSError as exc:
            alert(f"cannot read {f}: {exc}")
    return out


def mark(kind: str, unit: str, why: str):
    """QX_RUN_DIR/<kind>/<unit> (shed or gpu0): read by the restart loop of that unit's start."""
    try:
        d = RUN_DIR / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {why}\n")
    except OSError as exc:
        alert(f"cannot write the {kind} marker of {unit}: {exc}")


def stop_runs_of(unit: str, why: str):
    """<out>/STOP for every registered collection whose heartbeat names `unit` (rule 3: the run stops)."""
    for out in registered_outs():
        try:
            for f in (out / "heartbeat").glob("*.json"):
                if json.loads(f.read_text()).get("unit") == unit:
                    with open(out / "STOP", "a") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} qx_watch: {unit} {why}\n")
                    alert(f"{unit}: {why}; wrote {out / 'STOP'}", out)
                    break
        except (OSError, ValueError) as exc:
            alert(f"stop_runs_of {unit} in {out}: {exc}")


def gpu1_free_mib() -> int:
    v = run(["nvidia-smi", f"--id={gpu_uuid(1)}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).strip()
    return int(v)


def cycle(uid: int) -> int:
    """One period; -> the number of our units other than watchers."""
    try:
        us = units()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        alert(f"systemctl failed: {exc}")
        return 1
    try:
        psi = parse_psi((CGROOT / f"user.slice/user-{uid}.slice/user@{uid}.service/memory.pressure").read_text())
        avail = mem_available((PROC / "meminfo").read_text())
        gpu0 = parse_pid_rows(run(["nvidia-smi", "-i", gpu_uuid(0), "-q", "-d", "PIDS"]))
        gpu1 = parse_pid_table(run(["nvidia-smi", "-i", gpu_uuid(1), "-q", "-d", "PIDS"]))
        free1 = gpu1_free_mib()
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, KeyError) as exc:
        alert(f"a reading failed: {exc}")
        psi, avail, gpu0, gpu1, free1 = None, None, None, None, None
    if psi is not None:
        for unit, why, rule in decide_units(us, psi.get("full", 0.0), avail, gpu0, gpu1, free1):
            if rule == "gpu0_unreadable":
                alert(f"{unit}: {why}")
                continue
            if rule == "gpu0":
                mark("gpu0", unit, why)
                stop_runs_of(unit, why)
            else:
                mark("shed", unit, why)
            stop(unit, why)
    for out in registered_outs():
        try:
            if (out / "heartbeat").is_dir():
                check_heartbeats(out)
        except Exception as exc:                                    # one folder never ends the watcher
            alert(f"heartbeat check of {out} failed: {exc!r}")
    return sum(u["kind"] != "watch" for u in us)


def wanted_recently(now: float) -> bool:
    try:
        return now - (RUN_DIR / "watch" / "wanted").stat().st_mtime < IDLE_EXIT_S
    except OSError:
        return False


def write_code_identity() -> str:
    """QX_RUN_DIR/watch/code.sha256 = sha256 of this file (module docstring); -> the sha256."""
    h = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        d = RUN_DIR / "watch"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"code.sha256.tmp{os.getpid()}"
        tmp.write_text(h + "\n")
        os.replace(tmp, d / "code.sha256")
    except OSError as exc:
        alert(f"cannot write watch/code.sha256: {exc}")
    return h


def main():
    uid = os.getuid()
    log(f"qx_watch started, pid {os.getpid()}, period {PERIOD_S} s, code sha256 {write_code_identity()} ({__file__})")
    idle_since = None
    while True:
        try:
            (RUN_DIR / "watch.alive").touch()
        except OSError as exc:
            log(f"cannot touch watch.alive: {exc}")
        try:
            n = cycle(uid)
        except Exception as exc:                                    # the watcher never ends on an error
            alert(f"cycle failed: {exc!r}")
            n = 1
        if n == 0 and not wanted_recently(time.time()):
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_EXIT_S:
                log("no other unit of ours and no launcher asking for 30 min: exiting")
                return
        else:
            idle_since = None
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    sys.exit(main())
