#!/bin/bash
# One AutoMate online / test worker with restarts (the loop of collect_worker.sh for run_wm320_attempts.py automate
# worker), started by online_job.sh from a frozen code copy:
#   bash <code>/scripts/automate/online_worker.sh <code> <switches.env> <job dir> <index> <count> <MemoryMax> <CPUQuota> <GPU GB>
# <code> is a frozen copy (collect_lib.sh cl_freeze_code: <code>/../code.sha256 holds its hash). Every start goes
# through caps.sh qx_run (start lock, 90 s between GPU starts, our caps, GPU 1 only, CUDA_VISIBLE_DEVICES=1,
# QX_GPU_GB = <GPU GB>) as scope qingxin-collect-online-<job name>-w<index>-<k>-<time> (kind "collect": the watcher
# qx_watch.py treats it as a collection worker: heartbeat <job>/heartbeat, hang kill of the whole scope, shed under
# memory pressure or GPU-1 overuse). Environment: cl_isaac_env, every inherited WM_* unset and the frozen switches
# sourced (cl_clean_env), then WM_AUTOMATE_ASSETS = job.json assets (the collection's frozen <collection>/inputs/assets,
# checked by prepare; the worker refuses another), PYTHONPATH = only <code>/src, HF / torch caches offline (cl_train_env: the agent's encoder
# weights). Each start has its own log <job>/logs/w<index>_<k>_<unix time>.log and unit file. After an exit
# (run_wm320_automate.py worker; the codes of wm.sim_automate.collect), as collect_worker.sh:
#   99 caps refused        wait 60-120 s, try again (not counted); ALERT after REFUSED_ALERT_S, then every REFUSED_REPEAT_S
#   0 + "WORKER DONE"      loop ends (no grasp left)
#   3 refused              loop ends + ALERT (job.json, the checks, or a start_success stop: <job>/STOP)
#   7 forbidden GPU 0      <job>/STOP is written, loop ends + ALERT; the same when qx_watch.py stopped this start for
#                          GPU 0 (QX_RUN_DIR/gpu0/<unit>)
#   shed by the watcher    (QX_RUN_DIR/shed/<unit>, any exit code) wait SHED_WAIT_S, try again; more than SHED_MAX
#                          within 24 h end the loop + ALERT
#   5 / 130 / 143 stopped  loop ends; ALERT unless <job>/STOP exists
#   6 scene build failed   wait 120 s x n (n consecutive, at most 6), try again (not counted)
#   killed after a hang    (qx_watch.py left heartbeat/<pid>_<start>.hung) with a claim held: wait 30-90 s, try again
#                          (the grasp counts the death); without a claim: counted as a crash
#   anything else (crash, also 8: nvidia-smi unreadable at the GPU check) wait 30-90 s, try again; more than 5
#                          crashes in the last hour end the loop + ALERT
# Before every start: <job>/STOP ends the loop; a changed frozen code tree ends it + ALERT; the watcher must run
# (cl_watch_ensure), otherwise one ALERT line, wait 60 s and check again. Why the loop ended: <job>/logs/w<index>.end.
set -u
CODE=$1; SW=$2; JOB=$3; I=$4; N=$5; MEM=$6; CPU=$7; GPU=$8
SHED_WAIT_S=600
SHED_MAX=6
REFUSED_ALERT_S=1800
REFUSED_REPEAT_S=7200
source "$CODE/scripts/automate/caps.sh"
source "$CODE/scripts/automate/collect_lib.sh"
LOGS=$JOB/logs
mkdir -p "$LOGS"
NAME=online-$(basename "$(dirname "$JOB")")-$(basename "$JOB")

end_loop() {   # end_loop <reason> <alert 0/1>
  echo "$(date '+%F %T') worker $I loop ended: $1" >> "$LOGS/launcher.log"
  [ "$2" = 1 ] && echo "$(date '+%F %T') worker $I loop ended: $1" >> "$LOGS/ALERT"
  echo "$(date '+%F %T') $1" > "$LOGS/w$I.end"
  exit 0
}

stop_run() {   # <reason>: <job>/STOP for every worker of the job, then end this loop with an ALERT
  echo "$(date '+%F %T') worker $I: $1" >> "$JOB/STOP"
  end_loop "$1; STOP written (every worker of this job stops, no restart)" 1
}

cl_isaac_env
cl_clean_env "$SW" >> "$LOGS/launcher.log" 2>&1 || end_loop "$SW missing or malformed" 1
ASSETS=$("$CL_PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["assets"])' "$JOB/job.json" 2>> "$LOGS/launcher.log") &&
  [ -d "$ASSETS" ] || end_loop "$JOB/job.json names no existing asset folder (assets)" 1
export WM_AUTOMATE_ASSETS=$ASSETS
cl_train_env "$CODE"
export QX_REGISTRY=$LOGS/workers.jsonl
export OMP_NUM_THREADS=$(( ${CPU%\%} / 100 > 0 ? ${CPU%\%} / 100 : 1 ))
printf '{"time": "%s", "event": "loop start", "index": %s, "job": "%s", "loop_pid": %s}\n' \
  "$(date '+%F %T')" "$I" "$JOB" "$$" >> "$LOGS/workers.jsonl"

crashes=(); sheds=(); nbuild=0; k=0; walert=0; refused_since=0; refused_alert=0
while true; do
  [ -e "$JOB/STOP" ] && end_loop "STOP file" 0
  h=$(cl_code_hash "$CODE") || end_loop "cannot hash the frozen code $CODE" 1
  [ "$h" = "$(cat "$CODE/../code.sha256")" ] || end_loop "the frozen code $CODE changed (sha256 $h)" 1
  if ! cl_watch_ensure "$JOB" "$CODE" >> "$LOGS/launcher.log" 2>&1; then
    if [ $walert -eq 0 ]; then
      echo "$(date '+%F %T') worker $I: the watcher qx_watch.py is not running (or watch.alive is stale); not starting" \
           "until it is" | tee -a "$LOGS/ALERT" >> "$LOGS/launcher.log"
      walert=1
    fi
    sleep 60; continue
  fi
  walert=0
  kn=$((k + 1))
  log=$LOGS/w${I}_${kn}_$(date +%s).log
  ufile=$LOGS/w${I}_${kn}.unit
  rm -f "$ufile"
  { date; free -g | head -2; env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader; } > "$log"
  QX_UNIT_FILE=$ufile qx_run collect "$NAME-w${I}-$kn" "$MEM" "$CPU" "$GPU" \
    "$CL_PY" -u "$CODE/scripts/online/run_wm320_attempts.py" automate worker --job "$JOB" --index "$I" --count "$N" >> "$log" 2>&1
  code=$?
  if [ $code -eq 99 ]; then
    why=$(grep -m1 '^refused' "$log" 2>/dev/null | head -c 300)
    rm -f "$log" "$ufile"
    now=$(date +%s)
    [ $refused_since -eq 0 ] && refused_since=$now
    echo "$(date '+%F %T') worker $I: start refused by the caps (${why:-no reason printed}); refused for" \
         "$(( now - refused_since )) s" >> "$LOGS/launcher.log"
    if [ $(( now - refused_since )) -ge $REFUSED_ALERT_S ] &&
       { [ $refused_alert -eq 0 ] || [ $(( now - refused_alert )) -ge $REFUSED_REPEAT_S ]; }; then
      echo "$(date '+%F %T') worker $I: every start refused by the caps for $(( now - refused_since )) s (last:" \
           "${why:-no reason printed}); the loop keeps waiting" >> "$LOGS/ALERT"
      refused_alert=$now
    fi
    sleep $(( 60 + RANDOM % 60 )); continue
  fi
  refused_since=0; refused_alert=0
  k=$kn
  echo "exit $code $(date)" >> "$log"
  unit=$(cat "$ufile" 2>/dev/null)
  read -r _ _ wpid _ wstart _ < <(grep -m1 '^WORKER PID ' "$log"; echo)
  hung=0
  [ -n "${wpid:-}" ] && [ -e "$JOB/heartbeat/${wpid}_${wstart}.hung" ] && hung=1
  gpu0=0; shed=0
  [ -n "$unit" ] && [ -e "$QX_RUN_DIR/gpu0/$unit.scope" ] && gpu0=1
  [ -n "$unit" ] && [ -e "$QX_RUN_DIR/shed/$unit.scope" ] && shed=1
  case $code in
    0) grep -q '^WORKER DONE' "$log" && end_loop "done (WORKER DONE)" 0 ;;
    3) end_loop "refused (exit 3: job.json, the checks or a start_success stop), log $log" 1 ;;
    7) stop_run "forbidden GPU-0 context (exit 7, unit ${unit:-?}), log $log" ;;
  esac
  [ $gpu0 -eq 1 ] && stop_run "stopped by qx_watch.py for GPU 0 (exit $code, unit $unit: $(head -c 300 "$QX_RUN_DIR/gpu0/$unit.scope")), log $log"
  if [ $shed -eq 1 ]; then
    why=$(head -c 300 "$QX_RUN_DIR/shed/$unit.scope" 2>/dev/null); rm -f "$QX_RUN_DIR/shed/$unit.scope"
    now=$(date +%s); recent=()
    for t in "${sheds[@]}"; do [ $(( now - t )) -lt 86400 ] && recent+=("$t"); done
    sheds=("${recent[@]}" "$now")
    [ ${#sheds[@]} -gt $SHED_MAX ] && end_loop "shed by qx_watch.py ${#sheds[@]} times within 24 h (last: $why), log $log" 1
    echo "$(date '+%F %T') worker $I: shed by qx_watch.py ($why; ${#sheds[@]} in 24 h), retry in $SHED_WAIT_S s" >> "$LOGS/launcher.log"
    sleep $SHED_WAIT_S; continue
  fi
  case $code in
    5|130|143)
      [ -e "$JOB/STOP" ] && end_loop "stopped (STOP file)" 0
      end_loop "stopped by a signal (exit $code), log $log" 1 ;;
    6) nbuild=$((nbuild + 1)); s=$(( 120 * (nbuild < 6 ? nbuild : 6) ))
       echo "$(date '+%F %T') worker $I: scene build failed ($nbuild in a row), retry in $s s" >> "$LOGS/launcher.log"
       sleep $s; continue ;;
  esac
  [ -e "$JOB/STOP" ] && end_loop "stopped (STOP file; exit $code$([ $hung -eq 1 ] && echo ', killed after a hang')), log $log" 0
  nbuild=0
  what="crash (exit $code"
  if [ $hung -eq 1 ]; then
    hk=$("$CL_PY" -I -c "
import json, sys
try:
    h = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print('pre unreadable'); sys.exit()
s = str(h.get('stage') or '?')
pre = not h.get('claim') or s.split(' ')[0] in ('start', 'launch', 'agent', 'done')
print('pre' if pre else 'held', s.replace(' ', '_'))" "$JOB/heartbeat/${wpid}_${wstart}.hung" 2>/dev/null)
    read -r hkind hstage <<< "${hk:-pre unreadable}"
    if [ "$hkind" = held ]; then
      echo "$(date '+%F %T') worker $I: killed after a hang with a claim held (pid $wpid, stage $hstage), restarting" >> "$LOGS/launcher.log"
      sleep $(( 30 + RANDOM % 60 )); continue
    fi
    what="hang before a claim (stage $hstage), killed by qx_watch.py (exit $code"
  fi
  now=$(date +%s); recent=()
  for t in "${crashes[@]}"; do [ $(( now - t )) -lt 3600 ] && recent+=("$t"); done
  crashes=("${recent[@]}" "$now")
  [ "$code" = 8 ] && what="GPU check could not read nvidia-smi (exit 8, counted as a crash"
  [ ${#crashes[@]} -gt 5 ] && end_loop "restart rate cap: ${#crashes[@]} crashes within one hour (last: $what), log $log)" 1
  echo "$(date '+%F %T') worker $I: $what, ${#crashes[@]} in the last hour), restarting" >> "$LOGS/launcher.log"
  sleep $(( 30 + RANDOM % 60 ))
done
