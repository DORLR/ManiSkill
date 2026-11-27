"""Port plan T7: the numbers of an online smoke (online_smoke.sh), printed and written to <smoke out>/t7.json.

    python online_smoke_check.py <smoke out> <grasps collection>

Per job (act_done, wm, pool/r1): the worker starts' GPU-0 check lines and exit codes (logs/w*_*.log), per attempt the
reference acceptance, placement (attempts, failures, start error), timing (reset, reference, transfer, decisions),
decisions / max_decisions, stop reason, success (AutoMate's rule), start_success, seated_fk; per grasp, in the
order the worker ran them (started_unix), the worker pid, the scene build's seconds (scene_build_s; None: the scene of
the grasp before was used) and the worker's memory after the grasp; for the collect job also a compact table
"rebuilds" of these per grasp, in order (rebuild cost and memory growth over repeated scene rebuilds), and whether
one worker process ran them all. The worker's
memory (CO._memory: process maxrss, scope memory.peak, GPU MiB on GPU 0 / 1). From budget.log the largest measured
GPU-1 MiB and memory.peak per unit (the learner's unit among them). The pool: every entry's simulator and split_sha256
against the collection's split identity, DONE, and train_wm320.online_entries(pool, sha256) (the learner's reader).
Nothing is judged here beyond the hard checks (entries' split, simulator): the numbers are for the user.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from wm.tdmpc import demo320 as D  # noqa: E402

OUT, GR = Path(sys.argv[1]), Path(sys.argv[2])
sid = D.split_identity(GR)
rep = {"split_identity": sid, "jobs": {}}
for name, W in (("act_done", OUT / "act_done"), ("wm", OUT / "wm"), ("collect", OUT / "pool" / "r1")):
    j = {"job_dir": str(W), "worker_logs": []}
    for f in sorted((W / "logs").glob("w*_*.log")):
        txt = f.read_text(errors="ignore")
        j["worker_logs"].append({"log": f.name, "gpu_checks": re.findall(r"^GPU check .*$", txt, re.M),
                                 "exit": (re.findall(r"^exit (\d+)", txt, re.M) or [None])[-1],
                                 "launch": re.findall(r"Isaac started in [0-9.]+ s", txt)})
    res = Path(json.loads((W / "job.json").read_text())["results"]) if (W / "job.json").exists() else None
    j["grasps"] = []
    for f in sorted(res.glob("*/*.json")) if res else []:
        if f.parent.name in ("heartbeat", "logs", "claims", "deaths", "build_failures"):
            continue
        s = json.loads(f.read_text())
        if "starts" not in s:
            continue
        g = {"grasp": s.get("grasp"), "port": s.get("port"), "started_unix": s.get("started_unix"),
             "pid": (s.get("process") or {}).get("pid"), "error": s.get("error"), "seconds": s.get("seconds"),
             "scene_build_s": s.get("scene_build_s"), "memory": s.get("memory"),
             "max_decisions": s.get("max_decisions"), "band_above_top_mm": s.get("band_above_top_mm"), "attempts": []}
        for st in s["starts"]:
            a = st.get("attempt") or {}
            g["attempts"].append({"idx": st.get("idx"), "outcome": st.get("outcome", "attempt"),
                                  "reference": {k: (st.get("reference") or {}).get(k) for k in ("accepted", "completed", "inserted_end", "failure")},
                                  "placement_attempts": st.get("placement_attempts"),
                                  "placement_failures": [x.get("stage") for x in st.get("placement_failures") or []],
                                  "start_error_mm": st.get("start_error_mm"), "timing": st.get("timing"),
                                  "decisions": a.get("decisions"), "stopped": a.get("stopped"), "success": a.get("success"),
                                  "strict_success": a.get("strict_success"), "truth_info_end": a.get("truth_info_end"),
                                  "start_success": a.get("start_success"), "seated_fk": a.get("seated_fk"),
                                  "first_success_decision": a.get("first_success_decision"),
                                  "latency_ms_median": (sorted(x[1] for x in a["latency_ms"])[len(a["latency_ms"]) // 2]
                                                        if a.get("latency_ms") else None)})
        j["grasps"].append(g)
    j["grasps"].sort(key=lambda g: (g["started_unix"] is None, g["started_unix"] or 0.0))
    if name == "collect":
        j["one_worker_process"] = len({g["pid"] for g in j["grasps"]}) == 1
        j["rebuilds"] = [{"grasp": g["grasp"], "assembly": g["port"], "pid": g["pid"], "scene_build_s": g["scene_build_s"],
                          "grasp_s": g["seconds"],
                          **{k: (g["memory"] or {}).get(k) for k in ("maxrss_gb", "memory_peak_gb", "memory_current_gb",
                                                                    "gpu")}} for g in j["grasps"]]
    for f in ("summary.json", "summary_incomplete.json"):
        p = (res or W) / f
        if p.exists():
            j["summary_file"] = str(p)
    rep["jobs"][name] = j
# budget.log: the largest measured GPU-1 MiB and memory.peak per unit
peak_gpu, peak_mem = {}, {}
bl = OUT / "budget.log"
if bl.exists():
    for line in bl.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) == 5 and p[0].startswith("qingxin-") and p[4].isdigit():
            peak_gpu[p[0]] = max(peak_gpu.get(p[0], 0), int(p[4]))
        if len(p) == 3 and p[0] == "peak" and p[2].isdigit():
            peak_mem[p[1]] = max(peak_mem.get(p[1], 0), int(p[2]))
rep["budget_peaks"] = {"gpu1_measured_mib": peak_gpu, "memory_peak_bytes": peak_mem}
# the pool
pool = OUT / "pool"
man = pool / "r1" / "manifest.json"
pc = {"done": (pool / "r1" / "DONE").exists(), "entries": 0, "bad_entries": []}
if man.exists():
    ents = json.loads(man.read_text())["entries"]
    pc["entries"] = len(ents)
    pc["bad_entries"] = [e["path"] for e in ents if e.get("simulator") != "automate" or e.get("split_sha256") != (sid or {}).get("sha256")]
    try:
        import train_wm320 as T
        got, rounds = T.online_entries(pool, (sid or {}).get("sha256"))
        pc["online_entries"] = {"read": len(got), "rounds": rounds, "bc": sum(e["bc"] for e in got)}
    except SystemExit as exc:
        pc["online_entries"] = {"refused": str(exc)}
    except Exception as exc:
        pc["online_entries"] = {"error": f"{type(exc).__name__}: {exc}"}
rep["pool"] = pc
(OUT / "t7.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
ok = pc["done"] and pc["entries"] > 0 and not pc["bad_entries"] and "read" in pc.get("online_entries", {})
print("T7 HARD CHECKS (pool entries, split, learner reader):", "OK" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
