#!/bin/bash
# Stop a collection:   bash collect_stop.sh <out dir> [--hard]
# Writes <out>/STOP: every worker stops before its next trajectory or grasp, releases its claim with the previous
# text (not counted as a death) and exits 5; the restart loops end. To resume: remove <out>/STOP and run
# collect_launch.sh (same config) or collect_add_worker.sh.
# --hard: also stops every live worker now. Workers are found from <out>/heartbeat/<pid>_<start>.json; a worker is
# stopped only if its pid still has that process start time and its unit is one of our qingxin-collect-* scopes, and
# then by `systemctl --user stop <that exact unit>` (SIGTERM: the worker releases its claim and exits 5).
set -u
[ $# -ge 1 ] || { echo "usage: bash collect_stop.sh <out dir> [--hard]"; exit 1; }
OUT=$(readlink -f "$1"); HARD=${2:-}
[ -f "$OUT/plan.json" ] || { echo "$OUT is not a collection folder (no plan.json)"; exit 1; }
date '+%F %T' > "$OUT/STOP"
echo "$(date '+%F %T') STOP written" >> "$OUT/logs/launcher.log"
echo "wrote $OUT/STOP"
[ "$HARD" = "--hard" ] || exit 0
for f in "$OUT"/heartbeat/*.json; do
  [ -e "$f" ] || continue
  pid=$(grep -o '"pid": [0-9]*' "$f" | grep -o '[0-9]*$')
  start=$(grep -o '"start": "[0-9]*"' "$f" | grep -o '[0-9]*' | tail -1)
  unit=$(grep -o '"unit": "[^"]*"' "$f" | sed 's/.*: "//; s/"$//')
  [ -n "$pid" ] && [ -n "$start" ] || continue
  now_start=$(awk '{sub(/.*\) /, ""); print $20}' "/proc/$pid/stat" 2>/dev/null)
  if [ "$now_start" != "$start" ]; then echo "pid $pid: not running (stale $f)"; continue; fi
  if [[ ! "$unit" =~ ^qingxin-collect-[A-Za-z0-9_.-]+\.scope$ ]]; then echo "pid $pid: unit '$unit' is not ours; skipped"; continue; fi
  systemctl --user stop --no-block "$unit" && echo "stopping $unit (pid $pid)" | tee -a "$OUT/logs/launcher.log"
done
