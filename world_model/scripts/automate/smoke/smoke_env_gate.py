"""Pre-smoke gate of collect_smoke.sh (r4-expert-06): the collector smoke starts only on a smoke_env.py result that is
evidence for the code of the milestone-1b batch the smoke config gates on. Standard library only (run with python -I).

    python -I smoke_env_gate.py <smoke_env result json> <smoke config json>

The batch = the config's m1b_dirs; its results = every <dir>/*.json holding "assembly_id". Required, else exit 1 with
one "refused: ..." line per reason:
  - the result file is readable and has no "fatal";
  - s1-s6 (result["s"]["s1"] ... ["s6"]) and e (result["e"]) each have "pass" true (smoke_env.py --checks fk,b,g,s,e
    --assembly 00015, smoke_env.py docstring);
  - code_changed_during_run is False (present: the end identity was computed);
  - physics_code_sha256 equals the single physics_code_sha256 of the batch results and code_files_sha256 the single
    code_files_sha256 of the batch results (run from the batch's frozen copy <dir>/code; more than one value in the
    batch is refused as mixed code);
  - the switches (result["switches"]) equal those of the batch results (compared as sorted json, refused on any
    difference, e.g. a CPU-PhysX smoke_env result for a GPU-PhysX batch; more than one switch set in the batch is
    refused as mixed switches; r7-ops-03): run smoke_env.py with QX_SWITCHES=<batch>/switches.env;
  - the run started after the newest batch result had ended (result "started", local time of rs2, against
    end = max(unix_time + seconds, the file's modification time) per result: unix_time is taken when the assembly
    starts; r7-integration-01): a result from before or during the batch is not evidence for its finished code;
  - no batch process is running: no <dir>/heartbeat/<pid>_<start>.json names a live process (/proc/<pid> exists and
    its start time equals the file's "start", read as qx_watch.start_time reads it). collect_smoke.sh also refuses
    while a tmux -L qingxin session qingxin-batch* exists (the gaps between a loop's processes).
INFO lines: the result's and the batch's switches; leftover <dir>/<id>.started markers (run_batch.sh keeps one on
purpose after a hang, exit 124: information, not a refusal). Exit 0 prints "SMOKE_ENV OK ...".
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CHECKS_S = ("s1", "s2", "s3", "s4", "s5", "s6")
PROC = Path("/proc")


def start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks), as qx_watch.start_time and the heartbeat's "start"."""
    try:
        stat = (PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    return stat[stat.rindex(")") + 2:].split()[19]


def live_heartbeats(d: Path) -> list[str]:
    """The heartbeat files of <d>/heartbeat whose process still runs (pid alive with the recorded start time)."""
    out = []
    for f in sorted((d / "heartbeat").glob("*.json")):
        try:
            hb = json.loads(f.read_text())
            pid, start = int(hb["pid"]), str(hb["start"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if start_time(pid) == start:
            out.append(f"{f.name} (stage {str(hb.get('stage'))[:60]})")
    return out


def batch_identity(dirs: list) -> tuple[dict, list]:
    """{physics: set, code: set, newest: unix time when the last result ended, n: results, switches: set, started:
    leftover markers} of the batch results; problems (incl. a running batch process)."""
    ids = {"physics": set(), "code": set(), "newest": None, "n": 0, "switches": set(), "started": []}
    why = []
    for d in dirs:
        if not Path(d).is_dir():
            why.append(f"batch folder {d} missing")
            continue
        live = live_heartbeats(Path(d))
        if live:
            why.append(f"batch still running in {d}: live heartbeat {', '.join(live[:3])}")
        ids["started"] += [str(m) for m in sorted(Path(d).glob("*.started"))]
        for f in sorted(Path(d).glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError) as exc:
                why.append(f"batch result {f} unreadable: {type(exc).__name__}")
                continue
            if not isinstance(r, dict) or "assembly_id" not in r:
                continue
            ids["n"] += 1
            ids["physics"].add(r.get("physics_code_sha256"))
            ids["code"].add(r.get("code_files_sha256"))
            ids["switches"].add(json.dumps(r.get("switches"), sort_keys=True))
            t = r.get("unix_time")
            ends = []
            try:
                ends.append(f.stat().st_mtime)
            except OSError:
                pass
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                sec = r.get("seconds")
                ends.append(t + (float(sec) if isinstance(sec, (int, float)) and not isinstance(sec, bool) else 0.0))
            if ends:
                ids["newest"] = max(ends) if ids["newest"] is None else max(ids["newest"], *ends)
    if not ids["n"]:
        why.append(f"no batch results in {dirs}")
    return ids, why


def gate(result_path: str, config_path: str) -> tuple[list, list]:
    """-> (problems, info lines)."""
    info = []
    try:
        cfg = json.loads(Path(config_path).read_text())
        R = json.loads(Path(result_path).read_text())
    except (OSError, ValueError) as exc:
        return [f"cannot read {result_path} / {config_path}: {type(exc).__name__}: {exc}"], info
    dirs = cfg.get("m1b_dirs") or []
    ids, why = batch_identity(dirs)
    if R.get("fatal"):
        why.append(f"smoke_env ended fatally: {str(R['fatal'])[:200]}")
    s = R.get("s") if isinstance(R.get("s"), dict) else {}
    for k in CHECKS_S:
        if (s.get(k) or {}).get("pass") is not True:
            why.append(f"{k} did not pass ({'missing' if k not in s else (s[k].get('error') or 'pass false')})"[:200])
    e = R.get("e") if isinstance(R.get("e"), dict) else {}
    if e.get("pass") is not True:
        why.append(f"e did not pass ({'missing' if not e else (e.get('error') or 'pass false')})"[:200])
    if R.get("code_changed_during_run") is not False:
        why.append(f"code_changed_during_run {R.get('code_changed_during_run')!r} (must be False)")
    for key, name in (("physics", "physics_code_sha256"), ("code", "code_files_sha256")):
        vals = ids[key]
        if len(vals) > 1:
            why.append(f"the batch results carry {len(vals)} different {name} (mixed code): {sorted(map(str, vals))[:4]}")
        elif vals and R.get(name) not in vals:
            why.append(f"{name} {R.get(name)} differs from the batch's {next(iter(vals))} (run smoke_env.py from "
                       f"<batch>/code)")
    rs = json.dumps(R.get("switches"), sort_keys=True)
    if len(ids["switches"]) > 1:
        why.append(f"the batch results carry {len(ids['switches'])} different switch sets (mixed switches)")
    elif ids["switches"] and rs not in ids["switches"]:
        bs = json.loads(next(iter(ids["switches"])))
        why.append("switches differ from the batch's (physics_device_default "
                   f"{(R.get('switches') or {}).get('physics_device_default')!r} vs the batch's "
                   f"{(bs or {}).get('physics_device_default')!r}; run smoke_env.py with QX_SWITCHES=<batch>/switches.env)")
    try:
        started = time.mktime(time.strptime(str(R.get("started")), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        started = None
        why.append(f"started {R.get('started')!r} unreadable")
    if started is not None and ids["newest"] is not None and started <= ids["newest"]:
        why.append(f"started {R.get('started')} is not after the newest batch result had ended "
                   f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ids['newest']))})")
    info.append(f"smoke_env {result_path}: assembly {R.get('assembly')}, started {R.get('started')}, "
                f"switches {json.dumps(R.get('switches'), sort_keys=True)}")
    info.append(f"batch {dirs}: {ids['n']} results, switches {sorted(ids['switches'])[:2]}")
    if ids["started"]:
        info.append(f"leftover markers (a hang or crash booked by the next process, or still open): {ids['started'][:5]}")
    return why, info


def main(argv: list) -> int:
    if len(argv) != 2:
        print("usage: python -I smoke_env_gate.py <smoke_env result json> <smoke config json>")
        return 1
    why, info = gate(*argv)
    for ln in info:
        print(f"INFO {ln}")
    for w in why:
        print(f"refused: smoke_env result: {w}")
    if not why:
        print(f"SMOKE_ENV OK {argv[0]} (s1-s6 and e pass, code and switches of the batch, run after it)")
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
