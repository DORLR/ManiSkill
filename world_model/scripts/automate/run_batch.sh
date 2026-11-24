#!/bin/bash
# Run batch_assemblies.py until every assigned assembly has a result (a crashed process is restarted; the assembly
# it died in is recorded as failed by the next process).   bash run_batch.sh <tag> --out <dir> --ids <a,b,...> [args]
# Started by parallel_batch.sh from the frozen copy <out>/code with QX_SWITCHES=<out>/switches.env (run_on_rs2.sh).
# Before every start the watcher must run (collect_lib.sh cl_watch_ensure; <out> is registered with it): otherwise the
# loop waits 60 s without starting (counted with the caps refusals below). Each start has its own log
# $LOGS/<tag>_<k>_<unix time>.log and unit file $LOGS/<tag>_<k>_<unix time>.unit (QX_UNIT_FILE, caps.sh qx_run; the
# name gets _<start number of this loop> appended only if two starts fall in the same second): a start that repeats k
# (after a shed or a refusal) never overwrites the log of the stopped process (r7-ops-04). After
# an exit (codes of run_on_rs2.sh / batch_assemblies.py):
#   99      the caps refused the start: wait 60 s, not counted (at most 360 refusals or watcher waits, about 6 h;
#           one ALERT line when they reach 30, about 30 min, r5-ops-05); that start's log and unit file are removed
#           after the refusal line is counted. While GPU 1's free memory F is below 12 GB + n x 7 GB the loops that do
#           not fit wait here for a whole wave, so their 'reached 30' ALERT lines are expected (r11-ops-chk-01; the
#           run plan in parallel_batch.sh); only 'giving up', crash and rate-cap lines are faults
#   7, or QX_RUN_DIR/gpu0/<unit> (qx_watch.py stopped it for GPU 0)
#           a forbidden GPU-0 context: writes <out>/STOP and ends; every run_batch.sh loop of this batch ends before
#           its next start (no restart); processes already running finish their own assembly list. When the gpu0
#           marker of this start's unit exists (whatever the exit code: the process is dead once systemd-run has
#           returned), the <id>.started markers written by this start are removed first, so the interrupted assembly
#           is not booked as a simulator crash later (r5-ops-03)
#   shed    QX_RUN_DIR/shed/<unit> (qx_watch.py stopped it for memory pressure or GPU-1 memory) with any exit code
#           except 124, and no "BATCH DONE" in the start's log: Kit answers SIGTERM by a carb assertion ("Recursion
#           not allowed") and a core dump, so a stopped batch exits 134 (SIGABRT), not 143 (r6-ops-02 / r6-ops-chk-01).
#           The marker is consumed, the <id>.started markers written by this start are removed (the id lists of the
#           loops are disjoint and this start's process is dead, so the interrupted assembly is not a crash and runs
#           again), wait SHED_WAIT_S, not counted; more than SHED_MAX sheds within 24 h end the loop + ALERT.
#           "Written by this start": modification time >= the start time of this start minus MARK_SLACK_S. A marker
#           an earlier process left on purpose (it hung: exit 124 or killed by the watcher; or it crashed) is older by
#           at least the 60 s wait after that exit and stays, so the next process still books that assembly as an
#           error, also when the shed came during that process's launch(), before it booked the marker
#   3       refused (switches, code or inputs): ends + ALERT (a restart cannot help)
#   124     the process overran TIMEOUT (run_on_rs2.sh `timeout -k 120`): a hang; the <id>.started marker stays, so
#           the next process books that assembly as an error (no endless retries of a hanging assembly); counted
#   8       nvidia-smi unreadable at the GPU check: counted, then 60 s
#   other   counted (at most 30 counted runs), then 60 s
# A start whose log says "BATCH DONE" ends the loop with exit 0. 30 counted runs without it end the loop with an
# ALERT line naming the ids without a result, exit 1 (r5-ops-05).
# <out>/STOP (from an exit 7 or written by hand) ends the loop before the next start. ALERT lines go to
# QX_RUN_DIR/ALERT.
W=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
SHED_WAIT_S=600
SHED_MAX=6
MARK_SLACK_S=5            # file-time slack; markers of earlier processes are older by >= 60 s (header: shed)
LOGS=/mnt/storage3/qingxin/automate/m1/logs
source "$W/scripts/automate/caps.sh"
source "$W/scripts/automate/collect_lib.sh"
tag=$1; shift
OUT=""; IDS=""
prev=""
for a in "$@"; do
  [ "$prev" = "--out" ] && OUT=$a
  [ "$prev" = "--ids" ] && IDS=$a
  prev=$a
done
[ -n "$OUT" ] || { echo "run_batch.sh: --out <dir> is required"; exit 1; }
mkdir -p "$LOGS"
alert() { echo "$(date '+%F %T') run_batch $tag: $*" | tee -a "$QX_RUN_DIR/ALERT"; }
release_own_markers() {   # remove the <id>.started markers written by the start that began at $1 (header: shed)
  local id m
  for id in ${IDS//,/ }; do
    m=$OUT/$id.started
    [ -e "$m" ] && [ "$(stat -c %Y "$m" 2>/dev/null || echo 0)" -ge $(( $1 - MARK_SLACK_S )) ] && rm -f "$m"
  done
}
k=1; refused=0; sheds=(); starts=0
while [ $k -le 30 ]; do
  [ -e "$OUT/STOP" ] && { echo "$(date) $tag: $OUT/STOP exists, not starting"; exit 0; }
  if ! cl_watch_ensure "$OUT" "$W" >> "$LOGS/${tag}_watch.log" 2>&1; then
    refused=$((refused + 1))
    [ $refused -eq 30 ] && alert "no live watcher / caps refusals reached 30 (about 30 min); still waiting (log $LOGS/${tag}_watch.log)"
    [ $refused -ge 360 ] && { alert "no live watcher / caps refused 360 times, giving up"; exit 1; }
    echo "$(date '+%F %T') $tag: the watcher qx_watch.py is not running; not starting ($LOGS/${tag}_watch.log)"
    sleep 60; continue
  fi
  t_start=$(date +%s); starts=$((starts + 1))
  name=${tag}_${k}_$t_start
  [ -e "$LOGS/$name.log" ] && name=${name}_$starts
  log=$LOGS/$name.log; ufile=$LOGS/$name.unit
  rm -f "$ufile"
  QX_UNIT_FILE=$ufile bash "$W/scripts/automate/run_on_rs2.sh" "$name" scripts/automate/batch_assemblies.py "$@"
  rc=$?
  unit=$(cat "$ufile" 2>/dev/null)
  if [ $rc -eq 99 ]; then
    refused=$((refused + 1))
    [ $refused -eq 30 ] && alert "caps refusals / watcher waits reached 30 (about 30 min); still waiting (last refusal:" \
      "$(grep -m1 '^refused' "$log" 2>/dev/null | head -c 300))"
    [ $refused -ge 360 ] && { alert "caps refused 360 starts, giving up (last refusal: $(grep -m1 '^refused' "$log" 2>/dev/null | head -c 300))"; exit 1; }
    rm -f "$log" "$ufile"
    sleep 60; continue
  fi
  if [ $rc -eq 7 ] || { [ -n "$unit" ] && [ -e "$QX_RUN_DIR/gpu0/$unit" ]; }; then
    why=""
    if [ -n "$unit" ] && [ -e "$QX_RUN_DIR/gpu0/$unit" ]; then
      why=": qx_watch.py $(head -c 300 "$QX_RUN_DIR/gpu0/$unit" | tr '\n' ' ')"
      release_own_markers "$t_start"                                 # stopped by the watcher: not a crash
    fi
    echo "$(date '+%F %T') run_batch $tag: forbidden GPU-0 context (exit $rc, unit ${unit:-?}$why, log $log); STOP written" |
      tee -a "$OUT/STOP" >> "$QX_RUN_DIR/ALERT"
    exit 7
  fi
  if [ -n "$unit" ] && [ -e "$QX_RUN_DIR/shed/$unit" ] && [ $rc -ne 124 ] && ! grep -q "BATCH DONE" "$log"; then
    why=$(head -c 300 "$QX_RUN_DIR/shed/$unit" | tr '\n' ' '); rm -f "$QX_RUN_DIR/shed/$unit"
    release_own_markers "$t_start"
    now=$(date +%s); recent=()
    for t in "${sheds[@]}"; do [ $(( now - t )) -lt 86400 ] && recent+=("$t"); done
    sheds=("${recent[@]}" "$now")
    if [ ${#sheds[@]} -gt $SHED_MAX ]; then
      alert "shed by qx_watch.py ${#sheds[@]} times within 24 h (last: $why), log $log; loop ended"; exit 1
    fi
    echo "$(date '+%F %T') $tag: shed by qx_watch.py ($why; ${#sheds[@]} in 24 h); markers of this start removed, retry in $SHED_WAIT_S s"
    sleep $SHED_WAIT_S; continue
  fi
  if [ $rc -eq 3 ]; then
    alert "refused (exit 3: switches, code or inputs; log $log); loop ended"; exit 3
  fi
  grep -q "BATCH DONE" "$log" && { echo "$(date '+%F %T') $tag: BATCH DONE"; exit 0; }
  [ $rc -eq 124 ] && echo "$(date '+%F %T') $tag: exit 124, the process overran TIMEOUT (a hang); its marker stays and the next process books that assembly as an error"
  k=$((k + 1))
  [ $rc -ne 0 ] && sleep 60
done
alert "30 counted runs without BATCH DONE (log $(ls -t "$LOGS/${tag}_30_"*.log 2>/dev/null | head -1)); ids without a result: $(for id in ${IDS//,/ }; do [ -e "$OUT/$id.json" ] || echo -n "$id "; done)"
exit 1
