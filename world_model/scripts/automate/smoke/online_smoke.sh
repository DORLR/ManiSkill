#!/bin/bash
# Smoke of the AutoMate online attempts on rs2 (port plan T7 and T8), GPU 1 only, everything in capped scopes:
#   bash online_smoke.sh <ckpt> <grasps collection> <smoke out> <training grasp keys> <MemoryMax> <CPUQuota> <GPU GB> <torch MiB>
# <ckpt>: an AutoMate checkpoint of train_wm320.py (offset_mode gap, provenance split = the collection's), e.g. the
# learner's last.pt copied aside; <training grasp keys>: a comma list of 4-6 grasps with condition "train" and
# reference_ok from at least 3 assemblies (e.g. 00015_g00,00015_g01,00117_g02,00863_g00,00470_g01); the first is the
# grasp of the two tests and of T8. Resources: one worker's. Beside the learner (MemoryMax 100G = 100 x 2^30 =
# 107.37e9 bytes) and the watcher (256M = 0.27e9 bytes) the caps' 125e9 bytes leave 17.36e9 bytes = 16.17 GiB, so
# use at most 16G (= 17.18e9 bytes) of RAM and at most 12 GB of GPU 1, e.g. 16G 500% 12 (the smoke measures what is
# needed). If the smoke needs more than 16G, the lead asks the user (the smoke is not shrunk). Switches and assets: the collection's frozen
# <collection>/inputs/switches.env (no new switches file) and <collection>/inputs/assets (WM_AUTOMATE_ASSETS = plan.json
# input_roots.assets, set after cl_clean_env in every step; prepare and the restore check refuse anything else).
# Run it while the learner trains if the learner's numbers are wanted (T7: the learner's GPU-1 and RAM peak during a
# round); `systemctl --user list-units 'qingxin-*'` before the start must show no other job of ours but the learner
# and the watcher.
#   T7  (1) test mode act_done (the arm the rounds compare against), 2 attempts on the first grasp (--smoke-grasp),
#           out <smoke out>/act_done
#       (2) test mode wm, 2 attempts on the first grasp (options SMOKE_WM_EXTRA, e.g. "--inside-mm 3"), out
#           <smoke out>/wm
#       (3) collect, all the grasps x 1 attempt in ONE worker process into the pool <smoke out>/pool round 1
#           (manifest, DONE, split_sha256 on every entry): the claim order of the round makes the worker rebuild the
#           scene at each change of assembly, so the rebuild cost and the memory over repeated rebuilds show
#       each through online_job.sh with 1 worker; a job exit 4 (start_success stop) ends the smoke (exit 1), as it
#       ends the rounds; a sampler writes qx_budget plus memory.peak of every qingxin-* unit
#       to <smoke out>/budget.log every 60 s while they run; then online_smoke_check.py (capped CPU scope) prints and
#       writes <smoke out>/t7.json: the GPU-0 check lines of every worker start, per attempt the reference acceptance,
#       the placement (attempts, failures, start error), wall time (reset, reference, transfer, decisions), the
#       stop reason and success; per grasp, in the order the worker ran them, the worker pid, the scene build's seconds
#       and the worker's memory after the grasp (process peak RSS, scope memory.peak / memory.current, GPU MiB); the
#       peaks of budget.log per unit (the learner's among them); the pool's entries (simulator, split_sha256 = the
#       collection's) and train_wm320.online_entries on the pool with the collection's split sha256.
#       Pictures: the AutoMate recorder stores Robot.frame()'s picture itself (automate320 module docstring), so the
#       pictures equal Robot.frame() at the same steps by construction (CPU test T3 checks the recorder's code path).
#   T8  online_restore_check.py in one capped GPU scope: reset + reference + placement on the grasp, then repeats x
#       (save the state, one scripted decision (8 control steps moving the hand 1 mm along the axis), restore): hand
#       deviation after each restore (also after one held control step), socket pose (Robot.sc.check_socket_pose) and
#       the spread of the decision's end pose between repeats -> <smoke out>/t8.json. Only after the user has seen
#       these numbers may the AutoMate prepare's refusal of --probe-max > 0 be lifted (plan item 1).
# Show t7.json and t8.json to the user before any round runs.
set -u
[ $# -eq 8 ] || { echo "usage: bash online_smoke.sh <ckpt> <grasps> <smoke out> <training grasp keys (4-6, >= 3 assemblies)> <MemoryMax> <CPUQuota> <GPU GB> <torch MiB>"; exit 1; }
CK=$(readlink -f "$1"); GR=$(readlink -f "$2"); OUT=$(readlink -m "$3"); KEYS=$4; MEM=$5; CPU=$6; GPU=$7; TMIB=$8
KEY=${KEYS%%,*}
nk=$(tr ',' '\n' <<< "$KEYS" | grep -c .); na=$(tr ',' '\n' <<< "$KEYS" | grep . | sed 's/_g[0-9]*$//' | sort -u | wc -l)
[ "$nk" -ge 4 ] && [ "$nk" -le 6 ] && [ "$na" -ge 3 ] ||
  { echo "refused: <training grasp keys> must be 4-6 grasps from at least 3 assemblies, got $nk grasps of $na assemblies"; exit 1; }
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
mkdir -p "$OUT/logs"
L=$OUT/logs/smoke.log
cl_freeze_code "$LIVE" "$OUT" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
CODE=$OUT/code
SW=$GR/inputs/switches.env
[ -f "$SW" ] || { echo "refused: $SW missing (the collection's frozen switches)" | tee -a "$L"; exit 1; }
ASSETS=$("$CL_PY" -I -c 'import json, os, sys
c = sys.argv[1]
a = (json.load(open(os.path.join(c, "plan.json"))).get("input_roots") or {}).get("assets")
f = os.path.join(c, "inputs", "assets")
if not a or not os.path.isdir(f) or os.path.realpath(a) != os.path.realpath(f):
    sys.exit(f"refused: plan.json input_roots.assets {a!r} is not the frozen asset folder {f} of the collection")
print(a)' "$GR" 2>&1) || { echo "$ASSETS" | tee -a "$L"; exit 1; }
wm_extra=(${SMOKE_WM_EXTRA:-})

sampler() {   # qx_budget and memory.peak of every qingxin-* unit, every 60 s, until the file STOP_SAMPLER exists
  while [ ! -e "$OUT/STOP_SAMPLER" ]; do
    { echo "=== $(date '+%F %T')"; qx_budget
      for u in $(qx_units); do
        cg=$(systemctl --user show -p ControlGroup --value "$u" 2>/dev/null)
        echo "peak $u $(cat "$QX_CGROOT$cg/memory.peak" 2>/dev/null || echo ?)"
      done; } >> "$OUT/budget.log" 2>&1
    sleep 60
  done
}
rm -f "$OUT/STOP_SAMPLER"
sampler &
spid=$!
run_job() {   # <job dir> <prepare args...>; a job exit 4 (start_success stop) ends the smoke with exit 1
  local rc
  bash "$CODE/scripts/automate/online_job.sh" "$CODE" "$SW" "$1" 1 "$MEM" "$CPU" "$GPU" "${@:2}" >> "$L" 2>&1
  rc=$?
  echo "$(date '+%F %T') $1: exit $rc" | tee -a "$L"
  if [ "$rc" = 4 ]; then
    echo "$(date '+%F %T') $1: start_success stop (a placed start satisfied AutoMate's rule); the smoke stops" | tee -a "$L"
    touch "$OUT/STOP_SAMPLER"; wait "$spid" 2>/dev/null
    exit 1
  fi
}
run_job "$OUT/act_done" --grasps "$GR" --ckpt "$CK" --mode act_done --split train --smoke-grasp "$KEY" --out "$OUT/act_done" --starts 2 --torch-mib "$TMIB"
run_job "$OUT/wm" --grasps "$GR" --ckpt "$CK" --mode wm --split train --smoke-grasp "$KEY" --out "$OUT/wm" --starts 2 --torch-mib "$TMIB" "${wm_extra[@]}"
run_job "$OUT/pool/r1" --grasps "$GR" --ckpt "$CK" --mode collect --split train --smoke-grasp "$KEYS" --pool "$OUT/pool" --round 1 --starts 1 --torch-mib "$TMIB" "${wm_extra[@]}"
# T8: one capped GPU scope (Isaac; no model)
( cl_isaac_env; cl_clean_env "$SW" >/dev/null || exit 1; export WM_AUTOMATE_ASSETS=$ASSETS; cl_train_env "$CODE"
  cl_watch_ensure "$OUT" "$CODE" >/dev/null
  qx_run_retry collect "online-smoke-restore" "$MEM" "$CPU" "$GPU" "$CL_PY" -u "$CODE/scripts/automate/smoke/online_restore_check.py" \
    --grasps "$GR" --grasp "$KEY" --out "$OUT/t8.json" ) >> "$OUT/logs/t8.log" 2>&1
echo "$(date '+%F %T') T8 restore check: exit $? (log $OUT/logs/t8.log)" | tee -a "$L"
touch "$OUT/STOP_SAMPLER"; wait "$spid" 2>/dev/null
( cl_isaac_env; cl_clean_env "$SW" >/dev/null || exit 1; export WM_AUTOMATE_ASSETS=$ASSETS; cl_train_env "$CODE"
  qx_run_retry prep "online-smoke-check" 4G 100% 0 "$CL_PY" "$CODE/scripts/automate/smoke/online_smoke_check.py" "$OUT" "$GR" ) 2>&1 | tee -a "$L"
