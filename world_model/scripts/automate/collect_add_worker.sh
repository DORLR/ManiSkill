#!/bin/bash
# Add one worker to a collection started by collect_launch.sh:   bash collect_add_worker.sh <out dir> <index> [count=8]
# Uses the collection's frozen code, inputs and resources (no MemoryMax / CPUQuota arguments: one source, the config
# or the measured defaults, as collect_launch.sh). Re-runs the preflight on the frozen config first. Workers claim
# grasps (wm.sim_automate.collect.claim), so any number can be added at any time; <index> / <count> only set where in
# the plan the worker starts. The start itself waits for caps.sh qx_run (lock, 90 s gap, caps).
set -u
[ $# -ge 2 ] || { echo "usage: bash collect_add_worker.sh <out dir> <index> [count=8]"; exit 1; }
OUT=$(readlink -f "$1"); I=$2; N=${3:-8}
[[ "$I" =~ ^[0-9]+$ ]] && [[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "refused: index '$I' / count '$N'"; exit 1; }
[ -d "$OUT/code" ] && [ -d "$OUT/inputs" ] || { echo "refused: $OUT has no frozen code / inputs (use collect_launch.sh)"; exit 1; }
source "$OUT/code/scripts/automate/caps.sh"
source "$OUT/code/scripts/automate/collect_lib.sh"
[ -e "$OUT/STOP" ] && { echo "refused: $OUT/STOP exists (remove it to resume)"; exit 1; }
L=$OUT/logs/launcher.log
cl_freeze_code "$OUT/code" "$OUT" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1      # verifies the hash only
cl_prep "$OUT" "$OUT/inputs/config.json" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
cl_watch_ensure "$OUT" "$OUT/code" | tee -a "$L"
sess=qingxin-collect-$(basename "$OUT")-$I
if $CL_TMUX has-session -t "=$sess" 2>/dev/null; then
  echo "$(date '+%F %T') add worker $I: tmux session $sess exists -> NOT started" | tee -a "$L"; exit 1
fi
if $CL_TMUX new -d -s "$sess" "bash $OUT/code/scripts/automate/collect_worker.sh $OUT $I $N $sess"; then
  echo "$(date '+%F %T') add worker $I: restart loop in tmux -L qingxin session $sess" | tee -a "$L"
else
  echo "$(date '+%F %T') add worker $I: tmux failed -> NOT started" | tee -a "$L"; exit 1
fi
