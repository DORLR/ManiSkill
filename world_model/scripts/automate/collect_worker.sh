#!/bin/bash
# One collection worker with restarts, run from the frozen code <out>/code (collect_launch.sh / collect_add_worker.sh
# start it in tmux -L qingxin):   bash <out>/code/scripts/automate/collect_worker.sh <out> <index> <count> [session]
# Every start goes through caps.sh qx_run (start lock, 90 s between GPU starts, our caps, GPU 1 only, scope
# qingxin-collect-*), with the resources of <out>/logs/resources.txt (from the preflight) and the frozen inputs. The
# environment is clean: every inherited WM_* variable is unset, then the frozen <out>/inputs/switches.env is sourced
# (collect_lib.sh cl_clean_env), whatever the tmux -L qingxin server's environment holds.
# Each start has its own log <out>/logs/w<index>_<k>_<unix time>.log and unit file <out>/logs/w<index>_<k>.unit. What
# happens after an exit (codes of wm.sim_automate.collect):
#   99 caps refused        wait 60-120 s, try again (not counted; the refusal line goes to launcher.log, the start log
#                          is removed and the start number k is not advanced); refusals that last REFUSED_ALERT_S
#                          (30 min) give an ALERT line, then one every REFUSED_REPEAT_S (2 h) while they last
#   0 + "WORKER DONE"      loop ends (done)
#   3 refused              loop ends + ALERT (config, plan, inputs or code; a restart cannot help)
#   7 forbidden GPU 0      (app.assert_off_forbidden_gpu: more than graphics contexts of <= 50 MiB on GPU 0, or a
#                          compute context) <out>/STOP is written (every worker of the run stops), loop ends + ALERT;
#                          the same when qx_watch.py stopped this start for GPU 0 (QX_RUN_DIR/gpu0/<unit>)
#   8 GPU unreadable       (nvidia-smi gave no answer at the GPU check) counted as a crash (below)
#   shed by the watcher    (QX_RUN_DIR/shed/<unit>: memory pressure, GPU-1 sum or GPU-1 free memory; any exit code
#                          other than the 0 + WORKER DONE, 3 and 7 above: Kit answers a SIGTERM during its start, before
#                          the collector's handler is installed, by a carb assertion and a core dump, exit 134, not 143;
#                          r6-ops-02 / r6-ops-chk-01) wait SHED_WAIT_S, try again (not a crash); more than SHED_MAX
#                          sheds within 24 h end the loop + ALERT
#   5 / 130 / 143 stopped  loop ends; ALERT unless <out>/STOP exists (a deliberate stop)
#   any other exit with <out>/STOP present (not 0 + WORKER DONE, 3, 7, a GPU-0 or shed marker): loop ends quietly,
#                          not counted as a crash (a hard stop, collect_stop.sh --hard, during the Isaac start makes Kit
#                          abort with 134; r7-ops-06)
#   6 scene build failed   wait 120 s x n (n consecutive, at most 6) outside any scope, try again (not counted)
#   killed after a hang    (qx_watch.py left heartbeat/<pid>_<start>.hung) with a claim held: wait 30-90 s, try
#                          again (not counted; the grasp counts the hang as a death). Without a claim (its claim null,
#                          or stage start / launch / plan / done, or the marker unreadable): counted as a crash
#                          ("hang before a claim"), so a start that keeps hanging meets the rate cap
#   anything else (crash)  wait 30-90 s, try again; more than 5 crashes in the last hour end the loop + ALERT
# Before every start: <out>/STOP ends the loop; a changed frozen code tree (hash) ends it + ALERT; the watcher must run
# (collect_lib.sh cl_watch_ensure: an active qingxin-watch-* scope and QX_RUN_DIR/watch.alive at most 180 s old),
# otherwise one ALERT line, wait 60 s and check again without starting.
# Why the loop ended is written to <out>/logs/launcher.log and <out>/logs/w<index>.end.
set -u
OUT=$1; I=$2; N=$3; SESS=${4:-}
W=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
SHED_WAIT_S=600
SHED_MAX=6
REFUSED_ALERT_S=1800
REFUSED_REPEAT_S=7200
source "$W/scripts/automate/caps.sh"
source "$W/scripts/automate/collect_lib.sh"
LOGS=$OUT/logs
mkdir -p "$LOGS"

end_loop() {   # end_loop <reason> <alert 0/1>
  echo "$(date '+%F %T') worker $I loop ended: $1" >> "$LOGS/launcher.log"
  [ "$2" = 1 ] && echo "$(date '+%F %T') worker $I loop ended: $1" >> "$LOGS/ALERT"
  echo "$(date '+%F %T') $1" > "$LOGS/w$I.end"
  exit 0
}

stop_run() {   # stop_run <reason>: <out>/STOP for every worker of the run, then end this loop with an ALERT
  echo "$(date '+%F %T') worker $I: $1" >> "$OUT/STOP"
  end_loop "$1; STOP written (every worker of this run stops, no restart)" 1
}

cl_isaac_env
cl_clean_env "$OUT/inputs/switches.env" >> "$LOGS/launcher.log" 2>&1 ||
  end_loop "$OUT/inputs/switches.env missing or malformed" 1
export WM_AUTOMATE_ASSETS=$OUT/inputs/assets WM_AUTOMATE_M1B=$OUT/inputs/m1b QX_REGISTRY=$LOGS/workers.jsonl
read -r MEM CPU GPU TIMEOUT < "$LOGS/resources.txt" || end_loop "no $LOGS/resources.txt (run the preflight)" 1
export OMP_NUM_THREADS=$(( ${CPU%\%} / 100 > 0 ? ${CPU%\%} / 100 : 1 ))
printf '{"time": "%s", "event": "loop start", "index": %s, "tmux_socket": "qingxin", "session": "%s", "loop_pid": %s}\n' \
  "$(date '+%F %T')" "$I" "$SESS" "$$" >> "$LOGS/workers.jsonl"

crashes=()
sheds=()
nbuild=0
k=0
walert=0
refused_since=0
refused_alert=0
while true; do
  [ -e "$OUT/STOP" ] && end_loop "STOP file" 0
  h=$(cl_code_hash "$W") || end_loop "cannot hash the frozen code $W" 1
  [ "$h" = "$(cat "$OUT/code.sha256")" ] || end_loop "the frozen code $W changed (sha256 $h)" 1
  if ! cl_watch_ensure "$OUT" "$W" >> "$LOGS/launcher.log" 2>&1; then
    if [ $walert -eq 0 ]; then
      echo "$(date '+%F %T') worker $I: the watcher qx_watch.py is not running (or watch.alive is stale); not starting" \
           "until it is" | tee -a "$LOGS/ALERT" >> "$LOGS/launcher.log"
      walert=1
    fi
    sleep 60; continue
  fi
  walert=0
  kn=$((k + 1))                                   # advanced only when the start was not refused by the caps
  log=$LOGS/w${I}_${kn}_$(date +%s).log
  ufile=$LOGS/w${I}_${kn}.unit
  rm -f "$ufile"
  { date; free -g | head -2; env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader; } > "$log"
  QX_UNIT_FILE=$ufile qx_run collect "$(basename "$OUT")-w${I}-$kn" "$MEM" "$CPU" "$GPU" \
    "$CL_PY" -u "$W/scripts/automate/collect_expert.py" --out "$OUT" --config "$OUT/inputs/config.json" --index "$I" --count "$N" >> "$log" 2>&1
  code=$?
  if [ $code -eq 99 ]; then                       # caps refused: one line, no start log, k unchanged
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
  [ -n "${wpid:-}" ] && [ -e "$OUT/heartbeat/${wpid}_${wstart}.hung" ] && hung=1
  gpu0=0; shed=0
  [ -n "$unit" ] && [ -e "$QX_RUN_DIR/gpu0/$unit" ] && gpu0=1
  [ -n "$unit" ] && [ -e "$QX_RUN_DIR/shed/$unit" ] && shed=1
  case $code in
    0) grep -q '^WORKER DONE' "$log" && end_loop "done (WORKER DONE; see logs/summary.json)" 0 ;;
    3) end_loop "refused (exit 3: config, plan, inputs or code), log $log" 1 ;;
    7) stop_run "forbidden GPU-0 context (exit 7, unit ${unit:-?}), log $log" ;;
  esac
  [ $gpu0 -eq 1 ] && stop_run "stopped by qx_watch.py for GPU 0 (exit $code, unit $unit: $(head -c 300 "$QX_RUN_DIR/gpu0/$unit")), log $log"
  if [ $shed -eq 1 ]; then                        # any exit code (header: shed by the watcher)
    why=$(head -c 300 "$QX_RUN_DIR/shed/$unit" 2>/dev/null); rm -f "$QX_RUN_DIR/shed/$unit"
    now=$(date +%s); recent=()
    for t in "${sheds[@]}"; do [ $(( now - t )) -lt 86400 ] && recent+=("$t"); done
    sheds=("${recent[@]}" "$now")
    if [ ${#sheds[@]} -gt $SHED_MAX ]; then
      end_loop "shed by qx_watch.py ${#sheds[@]} times within 24 h (last: $why), log $log" 1
    fi
    echo "$(date '+%F %T') worker $I: shed by qx_watch.py ($why; ${#sheds[@]} in 24 h), retry in $SHED_WAIT_S s" >> "$LOGS/launcher.log"
    sleep $SHED_WAIT_S; continue
  fi
  case $code in
    5|130|143)
      [ -e "$OUT/STOP" ] && end_loop "stopped (STOP file)" 0
      end_loop "stopped by a signal (exit $code: systemctl stop by collect_stop.sh or qx_watch.py), log $log" 1 ;;
    6) nbuild=$((nbuild + 1)); s=$(( 120 * (nbuild < 6 ? nbuild : 6) ))
       echo "$(date '+%F %T') worker $I: scene build failed ($nbuild in a row), retry in $s s" >> "$LOGS/launcher.log"
       sleep $s; continue ;;
  esac
  [ -e "$OUT/STOP" ] && end_loop "stopped (STOP file; exit $code$([ $hung -eq 1 ] && echo ', killed after a hang')), log $log" 0
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
pre = not h.get('claim') or s.split(' ')[0] in ('start', 'launch', 'plan', 'done')
print('pre' if pre else 'held', s.replace(' ', '_'))" "$OUT/heartbeat/${wpid}_${wstart}.hung" 2>/dev/null)
    read -r hkind hstage <<< "${hk:-pre unreadable}"
    if [ "$hkind" = held ]; then                  # bounded by the grasp's death accounting
      echo "$(date '+%F %T') worker $I: killed after a hang with a claim held (pid $wpid, stage $hstage), restarting" >> "$LOGS/launcher.log"
      sleep $(( 30 + RANDOM % 60 )); continue
    fi
    what="hang before a claim (stage $hstage), killed by qx_watch.py (exit $code"
  fi
  now=$(date +%s); recent=()
  for t in "${crashes[@]}"; do [ $(( now - t )) -lt 3600 ] && recent+=("$t"); done
  crashes=("${recent[@]}" "$now")
  [ "$code" = 8 ] && what="GPU check could not read nvidia-smi (exit 8, counted as a crash"
  if [ ${#crashes[@]} -gt 5 ]; then
    end_loop "restart rate cap: ${#crashes[@]} crashes within one hour (last: $what), log $log)" 1
  fi
  echo "$(date '+%F %T') worker $I: $what, ${#crashes[@]} in the last hour), restarting" >> "$LOGS/launcher.log"
  sleep $(( 30 + RANDOM % 60 ))
done
