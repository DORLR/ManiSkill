#!/bin/bash
# One AutoMate job of the world model's agent (an online round or a test; run_wm320_attempts.py automate), on rs2:
#   bash <code>/scripts/automate/online_job.sh <code> <switches.env> <job dir> <N workers> <MemoryMax> <CPUQuota> <GPU GB> \
#        <prepare arguments...>
# <code>: a frozen copy (collect_lib.sh cl_freeze_code; online_automate.sh: the learner's; smoke/online_smoke.sh: its
# own); <switches.env>: the grasps collection's frozen <collection>/inputs/switches.env (prepare refuses switches or a
# physics device other than plan.json's). <job dir> must be the one prepare derives from its arguments (<pool>/r<round>
# for --mode collect, --out for a test). Every step reads the collection's frozen assets: WM_AUTOMATE_ASSETS = plan.json
# input_roots.assets of --grasps, which must be <collection>/inputs/assets (set after cl_clean_env; prepare checks it
# and records it in job.json, the workers export job.json's and refuse another). Resources per worker are the caller's (no
# defaults: measure them in the smoke T7); N workers x each must fit the caps (180 GB / 4800 % / 72 GB since 2026-11-25) alone (cl_static_budget),
# and every start is checked against the live sums of all our units, the learner's included (caps.sh qx_check).
#   1. prepare: CPU, in a capped scope (qx_run_retry prep, 4G, 100 %, no GPU) with the workers' environment; exit 3
#      (refused) ends here.
#   2. N restart loops online_worker.sh (one capped scope per start, GPU 1 only), in the background; waits for all.
#   3. finish: CPU, in a capped scope; its exit code is this script's (0 complete / DONE, 2 incomplete, 4 a
#      start_success stop (run_wm320_automate.EXIT_START_SUCCESS), 99 caps refused for QX_RETRY_MAX_S, 1 an error).
# Stop a job: date > <job dir>/STOP (workers stop before their next attempt or grasp; restarts end). Logs: <job>/logs.
set -u
[ $# -ge 8 ] || { echo "usage: bash online_job.sh <code> <switches.env> <job dir> <N> <MemoryMax> <CPUQuota> <GPU GB> <prepare args...>"; exit 1; }
CODE=$(readlink -f "$1"); SW=$(readlink -f "$2"); JOB=$(readlink -m "$3"); N=$4; MEM=$5; CPU=$6; GPU=$7; shift 7
source "$CODE/scripts/automate/caps.sh"
source "$CODE/scripts/automate/collect_lib.sh"
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "refused: N '$N'"; exit 1; }
[[ "$GPU" =~ ^[1-9][0-9]*$ ]] || { echo "refused: GPU GB '$GPU' must be an integer >= 1"; exit 1; }
[ -f "$CODE/../code.sha256" ] || { echo "refused: $CODE is not a frozen copy (no ../code.sha256)"; exit 1; }
[ "$(cl_code_hash "$CODE")" = "$(cat "$CODE/../code.sha256")" ] || { echo "refused: $CODE changed since it was frozen"; exit 1; }
[ -f "$SW" ] || { echo "refused: $SW missing"; exit 1; }
GRS=""; prev=""
for a in "$@"; do [ "$prev" = --grasps ] && GRS=$a; prev=$a; done
[ -n "$GRS" ] || { echo "refused: no --grasps <collection> among the prepare arguments"; exit 1; }
plan_assets() {   # <collection> -> plan.json input_roots.assets when it is the collection's frozen <collection>/inputs/assets
  "$CL_PY" -I -c 'import json, os, sys
c = sys.argv[1]
try:
    a = (json.load(open(os.path.join(c, "plan.json"))).get("input_roots") or {}).get("assets")
except (OSError, ValueError) as exc:
    sys.exit(f"refused: {c}/plan.json unreadable: {exc}")
f = os.path.join(c, "inputs", "assets")
if not a or not os.path.isdir(f) or os.path.realpath(a) != os.path.realpath(f):
    sys.exit(f"refused: plan.json input_roots.assets {a!r} is not the frozen asset folder {f} of the collection")
print(a)' "$1"
}
ASSETS=$(plan_assets "$GRS") || exit 1
[ -e "$JOB/STOP" ] && { echo "refused: $JOB/STOP exists ($(tail -1 "$JOB/STOP")); remove it after checking the cause"; exit 1; }
cl_static_budget "$N" "$MEM" "$CPU" "$GPU" || exit 1
mkdir -p "$JOB/logs"
L=$JOB/logs/job.log
py_step() {     # <stage> <name> <args...>: a CPU step of run_wm320_attempts.py automate in a capped scope
  local stage=$1 name=$2; shift 2
  ( cl_isaac_env; cl_clean_env "$SW" >/dev/null || exit 1; export WM_AUTOMATE_ASSETS=$ASSETS; cl_train_env "$CODE"
    qx_run_retry prep "online-$stage-$name" 4G 100% 0 "$CL_PY" "$CODE/scripts/online/run_wm320_attempts.py" automate "$stage" "$@" )
}
name=$(basename "$(dirname "$JOB")")-$(basename "$JOB")
echo "$(date '+%F %T') job $JOB: prepare $*" | tee -a "$L"
py_step prepare "$name" "$@" 2>&1 | tee -a "$L"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "$(date '+%F %T') job $JOB: prepare exit $rc" | tee -a "$L"; exit 3; }
if [ ! -e "$JOB/DONE" ]; then
  [ -f "$JOB/job.json" ] || { echo "refused: prepare wrote no $JOB/job.json" | tee -a "$L"; exit 3; }
  cl_watch_ensure "$JOB" "$CODE" | tee -a "$L"
  { echo "budget before the workers of $JOB"; qx_budget; echo "new: $N x (MemoryMax $MEM, CPUQuota $CPU, GPU-1 $GPU GB)"; } >> "$L" 2>&1
  pids=()
  for w in $(seq 0 $((N - 1))); do
    bash "$CODE/scripts/automate/online_worker.sh" "$CODE" "$SW" "$JOB" "$w" "$N" "$MEM" "$CPU" "$GPU" &
    pids+=($!)
    echo "$(date '+%F %T') job $JOB: worker loop $w pid $!" | tee -a "$L"
  done
  for p in "${pids[@]}"; do wait "$p"; done
  for w in $(seq 0 $((N - 1))); do echo "worker $w: $(cat "$JOB/logs/w$w.end" 2>/dev/null)" | tee -a "$L"; done
fi
py_step finish "$name" --job "$JOB" 2>&1 | tee -a "$L"
rc=${PIPESTATUS[0]}
echo "$(date '+%F %T') job $JOB: finish exit $rc" | tee -a "$L"
exit $rc
