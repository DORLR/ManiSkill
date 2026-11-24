#!/bin/bash
# Cheap mid-run check of the row-22 condition (r5-integration-01): every flag of check_expert_data.draw_report goes
# to the user (r6-integration-03), per assembly: reset_failed, label_failed and contact_report_error over the socket
# draws (t00's reset included), t00_refused over the grasps (a refused t00 drops the whole grasp), placement_failed
# over the rows that reached placement, random_truncated over the random-motion rows whose expert ran (stopped at the
# step budget or before a waypoint, r6-expert-01; with about 15 random rows per assembly one such row is 6.7 %), the
# placement redraws by reason over the placement attempts, each > 5 %, and any start that already satisfies
# AutoMate's rule.
#   bash draws_check.sh <collection out>          (the lead, e.g. hourly while the collection runs; seconds of CPU)
# Runs check_expert_data.py --data <out> --work <out>/logs/draws --draws-only, which reads only <out>/*/*/grasp.json
# and <out>/plan.json (no trajectory file, no fast copy), from the collection's frozen code <out>/code (this tree if
# there is none), in a caps.sh qx_run scope (kind prep: 4G, 100 %, GPU 0; never stopped by the watcher), with our
# HOME / caches / tmp on storage3 and no bytecode written (collect_lib.sh cl_isaac_env, as prep_automate_data.sh;
# r6-data-05). Prints each
# line of <out>/logs/draws/flags_for_user.txt; each flag not yet in <out>/logs/ALERT is appended there. Output of
# the check: <out>/logs/draws.log. Exit code: the check's (99 = the caps refused the start for
# QX_RETRY_MAX_S, default 3600 s: caps.sh qx_run_retry tries again every 60 s; r9-ops-01), 1 when refused here.
set -u
[ $# -eq 1 ] || { echo "usage: bash draws_check.sh <collection out>"; exit 1; }
OUT=$(readlink -f "$1")
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
[ -f "$OUT/plan.json" ] || { echo "refused: $OUT has no plan.json"; exit 1; }
W=$LIVE; [ -f "$OUT/code/scripts/automate/check_expert_data.py" ] && W=$OUT/code
mkdir -p "$OUT/logs/draws" || { echo "refused: cannot create $OUT/logs/draws"; exit 1; }
rm -f "$OUT/logs/draws/draws.json" "$OUT/logs/draws/flags_for_user.txt"
# printed before the redirected call, so that a start waiting on the caps does not look hung (r10-ops-chk-02)
echo "draws check in a capped scope (a caps refusal is retried every 60 s for up to ${QX_RETRY_MAX_S:-3600} s); progress: $OUT/logs/draws.log"
( cl_isaac_env && cl_clean_env >/dev/null && export PYTHONPATH="$W/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 &&
  cd "$W" && qx_run_retry prep "draws-$(basename "$OUT")" 4G 100% 0 "$CL_PY" -u "$W/scripts/automate/check_expert_data.py" \
    --data "$OUT" --work "$OUT/logs/draws" --draws-only ) > "$OUT/logs/draws.log" 2>&1
rc=$?
echo "draws check exit $rc (code $W; log $OUT/logs/draws.log)"
n=0
if [ -f "$OUT/logs/draws/flags_for_user.txt" ]; then
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    n=$((n + 1))
    echo "SHOW THE USER: $f"
    grep -qF -- "$f" "$OUT/logs/ALERT" 2>/dev/null ||
      echo "$(date '+%F %T') draws_check: SHOW THE USER: $f" >> "$OUT/logs/ALERT"
  done < "$OUT/logs/draws/flags_for_user.txt"
else
  echo "no $OUT/logs/draws/flags_for_user.txt (log $OUT/logs/draws.log)"; [ $rc -eq 0 ] && rc=1
fi
echo "$n flags (draws.json $OUT/logs/draws/draws.json)"
exit $rc
