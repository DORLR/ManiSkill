#!/bin/bash
# CPU preparation of an AutoMate collection for training on rs2 (r4-integration-04 / r4-int-chk-02), capped like every
# process of ours:
#   bash prep_automate_data.sh <run dir> <collection out> <fast dir> <D_max m | none> [--accept-skipped] [--partial]
# D_max (depth clip in metres for depth_stats.py) is the user's choice and has no default (not decided yet: it is asked
# after the smoke with the depth distribution, decision 5 of 2026-11-23); it must lie in (0, 3.2767] m, refused here
# before any step (depth_stats.py checks it again); the collector smoke uses 0.5 as a smoke-only
# value (collect_smoke_check.py item f).
# D_max "none" (the literal word; r10-data-01): a distribution-only run, for the production collection's training-split
# depth distribution before the user has chosen D_max. The last step then deletes any existing <fast>/depth_stats.json
# (and <fast>/depth_distribution.json) and runs depth_stats.py --data <fast> --distribution-only, which writes only
# <fast>/depth_distribution.json; no depth_stats.json is left, so train_wm320.py refuses (no statistics file) until
# depth_stats.py runs with the user's D_max. P.result is then "ok distribution-only" (collect_smoke_check item f does
# not accept it).
# Long runs: tmux -L qingxin new -d -s qingxin-prep-<name> "bash <world_model>/scripts/automate/prep_automate_data.sh ..."
# 0. Refused (exit 1, the line also appended to <run dir>/logs/ALERT once the run dir exists, r5-ops-04) before
#    anything runs: wrong arguments, no <out>/plan.json, a collection that still runs (an active scope
#    qingxin-collect-<out name>-*, caps.sh qx_units; also when systemctl cannot be read), or no
#    <out>/logs/summary.json (no worker has ended) unless --partial (r5-ops-chk-01). Live heartbeats, claims held by
#    live workers and planned grasps without grasp.json are refused by the index step (build_index.py, r5-data-01).
# 1. Freezes the code as train_automate.sh does (collect_lib.sh cl_freeze_code): this world_model tree is copied
#    read-only to <run dir>/code (hash in <run dir>/code.sha256); a later start in the same run dir must find the copy
#    unchanged. Every step runs from that copy.
# 2. Environment of every step: every inherited WM_* unset, WM_FORBIDDEN_GPU=0 (cl_clean_env), our caches / tmp on
#    storage3 (cl_isaac_env), PYTHONPATH = only <run dir>/code/src, one BLAS / OpenMP thread per process
#    (OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1: 8 workers + the parent stay inside the CPU quota),
#    the isaaclab env's python (CL_PY).
# 3. Runs, in this order, each in its own caps.sh qx_run scope qingxin-prep-<run name>-<step>-<unix time> with
#    MemoryMax PREP_MEM, CPUQuota PREP_CPU and GPU 0 GB (so qx_run sets CUDA_VISIBLE_DEVICES empty), and stops at the
#    first failing step except "draws" (exit 99 = the caps refused the start for QX_RETRY_MAX_S, default 3600 s: every
#    step starts through caps.sh qx_run_retry, which tries again every 60 s while the caps refuse; r9-ops-01):
#      draws        scripts/automate/check_expert_data.py --draws-only --data <out> --work <run dir>/draws_<unix time>
#                   (grasp.json / plan.json only: every flag of check_expert_data.draw_report, the row-22
#                   condition, r5-integration-01 / r6-integration-03: per assembly reset_failed, label_failed and
#                   contact_report_error over the socket draws incl. t00, t00_refused over the grasps, placement_failed
#                   over the rows that reached placement, random_truncated over the random-motion rows whose expert
#                   ran (stopped at the step budget or before a waypoint, r6-expert-01), redraws by reason over the
#                   placement attempts, each > 5 %, and any start already satisfying AutoMate's rule). Never stops
#                   the chain: a non-zero exit is an ALERT line; every line of <work>/flags_for_user.txt is
#                   appended to logs/ALERT and printed (show the user). Then "# grasp.json <n> of <planned> planned"
#                   (plan.json's grasps = the config's expected_grasps) is added to P.steps and printed.
#      index        scripts/data/build_index.py --automate <collection out> --workers 8 [--accept-skipped] [--partial]
#                   (-> <out>/index.json and <out>/index_completeness.json; exit 2 = it refused a running or
#                   unfinished collection, or the skipped assemblies' missing grasps without --accept-skipped; --partial
#                   is a deliberate early look, recorded, and training on it then needs train_wm320.py --allow-partial,
#                   which no launcher adds; wm.data.index).
#      fast         scripts/data/cache_wm_v1_fast.py --src <collection out> --dst <fast dir> --workers 8
#                   (copies index_completeness.json into the manifest as "completeness"; train_wm320.py refuses a
#                   partial collection unless --allow-partial)
#      depth_stats  rm -f <fast>/depth_stats.json <fast>/depth_distribution.json (r11-data-01: a failed step leaves no
#                   file of an earlier D_max; r12-data-chk-01: no distribution of an earlier training set),
#                   then scripts/automate/depth_stats.py --data <fast dir> --depth-max-m <D_max>
#                   (-> <fast>/depth_stats.json and <fast>/depth_distribution.json)
#      or, for D_max none:
#      depth_distribution  rm -f <fast>/depth_stats.json <fast>/depth_distribution.json (r11-data-01,
#                   r12-data-chk-01), then scripts/automate/depth_stats.py --data <fast dir>
#                   --distribution-only  (-> <fast>/depth_distribution.json only); afterwards, with the user's D_max
#                   (from the same frozen copy and caps as the step; the launcher prints it with the paths filled in):
#                   ( R=<run dir>; source $R/code/scripts/automate/caps.sh; source $R/code/scripts/automate/collect_lib.sh;
#                     cl_isaac_env; cl_clean_env >/dev/null; export PYTHONPATH=$R/code/src OMP_NUM_THREADS=1
#                     OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd $R/code; qx_run_retry prep <run name>-depth_stats
#                     24G 800% 0 $CL_PY -u scripts/automate/depth_stats.py --data <fast dir> --depth-max-m <D_max> )
# Mid-run (while the collection runs; the lead, e.g. hourly): bash scripts/automate/draws_check.sh <collection out>
# runs the same draws check in a capped CPU scope and prints flags_for_user.
# Records, with P = <run dir>/logs/prep_<unix time>:
#    P.<step>.log    the step's output (command, code hash, PYTHONPATH, exit code)
#    P.<step>.unit   the scope's unit name (QX_UNIT_FILE)
#    P.<step>.pid    "<pid> <process start time in clock ticks>" of the step's process (written inside the scope
#                    before the exec of python, so the pid is python's)
#    P.steps         one line per step run: "<step> <exit code> <unit> <pid> <start time>", and after "draws" the
#                    line "# grasp.json <n> of <planned> planned"
#    <run dir>/draws_<unix time>/draws.json, flags_for_user.txt   the draws check's report
#    P.result        "ok", "ok distribution-only" (D_max none) or "failed <step> <exit code>"
#    <run dir>/logs/prep_latest   the prefix P of the newest run
# A non-zero exit of a step other than draws appends a line to <run dir>/logs/ALERT. Exit code: 0, or the failing
# step's, or 1 (refused before any step). Resources: PREP_MEM / PREP_CPU are stated, not measured on AutoMate data: 8 CPU workers (800 %), and half
# of the 48 GB given to the 16-worker ManiSkill fast copy (scripts/pipelines/cache_fast320.sh); kind "prep" is never
# stopped by qx_watch.py, MemoryMax is the kernel's hard limit.
set -u
PREP_MEM=24G
PREP_CPU=800%
PREP_WORKERS=8
POS=(); FLAGS=()
for a in "$@"; do
  case $a in
    --accept-skipped|--partial) FLAGS+=("$a") ;;
    *) POS+=("$a") ;;
  esac
done
RUN=""; [ ${#POS[@]} -ge 1 ] && RUN=$(readlink -m "${POS[0]}")
refuse() {      # an early refusal: printed, and appended to <run dir>/logs/ALERT when the run dir can be made
  echo "refused: $*"
  [ -n "$RUN" ] && mkdir -p "$RUN/logs" 2>/dev/null &&
    echo "$(date '+%F %T') prep_automate_data refused: $*" >> "$RUN/logs/ALERT"
  exit 1
}
[ ${#POS[@]} -eq 4 ] || refuse "usage: bash prep_automate_data.sh <run dir> <collection out> <fast dir> <D_max m | none> [--accept-skipped] [--partial] (D_max has no default; got: $*)"
OUT=$(readlink -m "${POS[1]}"); FAST=$(readlink -m "${POS[2]}"); DMAX=${POS[3]}
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
DEPTH_MAX_M=3.2767   # = wm.tdmpc.policy320.DEPTH_MAX_M (not imported here: it pulls in torch; ops test 17 compares)
DIST_ONLY=0
if [ "$DMAX" = none ]; then
  DIST_ONLY=1                       # distribution-only run (header; r10-data-01)
elif ! [[ "$DMAX" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] ||
   ! awk -v d="$DMAX" -v m="$DEPTH_MAX_M" 'BEGIN { exit !(d + 0 > 0 && d + 0 <= m + 0) }'; then
  refuse "D_max '$DMAX' must lie in (0, $DEPTH_MAX_M] m (metres, the user's choice; policy320.DEPTH_MAX_M, the int16 ceiling of the 0.1 mm depth input; r6-data-01), or be the word none (distribution only, no depth_stats.json)"
fi
[ -f "$OUT/plan.json" ] || refuse "$OUT has no plan.json (not an AutoMate collection output folder)"
units=$(qx_units) || refuse "systemctl cannot list our scopes: cannot tell whether the collection $OUT still runs"
cunit="qingxin-collect-$(echo -n "$(basename "$OUT")" | tr -c 'a-zA-Z0-9_.-' '_')-"
busy=$(echo "$units" | grep -F -- "$cunit" | tr '\n' ' ')
[ -z "$busy" ] || refuse "the collection $OUT still runs (active scopes: $busy)"
partial=0; for f in "${FLAGS[@]}"; do [ "$f" = --partial ] && partial=1; done
[ -f "$OUT/logs/summary.json" ] || [ $partial -eq 1 ] ||
  refuse "$OUT/logs/summary.json missing: no collection worker has ended (--partial for a deliberate early look)"
mkdir -p "$RUN/logs" || refuse "cannot create $RUN/logs"
cl_freeze_code "$LIVE" "$RUN" || refuse "the frozen code copy $RUN/code (see the line above)"
W=$RUN/code
cl_isaac_env
cl_clean_env || refuse "cl_clean_env failed"
export PYTHONPATH="$W/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$RUN/logs/prep_$(date +%s)
echo "$P" > "$RUN/logs/prep_latest"
cd "$W" || exit 1

run_step() {    # run_step <name> <script relative to the frozen copy> [args...]; sets rc and log
  local name=$1 script=$2
  log=$P.$1.log
  shift 2
  { date; echo "prep step $name: $script $*"; echo "code $W sha256 $(cat "$RUN/code.sha256")"
    echo "PYTHONPATH=$PYTHONPATH MemoryMax $PREP_MEM CPUQuota $PREP_CPU GPU 0"; } > "$log"
  QX_UNIT_FILE=$P.$name.unit qx_run_retry prep "$(basename "$RUN")-$name" "$PREP_MEM" "$PREP_CPU" 0 \
    bash -c 'echo "$$ $(cut -d" " -f22 /proc/$$/stat)" > "$0" && exec "$@"' "$P.$name.pid" "$CL_PY" -u "$W/$script" "$@" \
    >> "$log" 2>&1
  rc=$?
  echo "exit $rc $(date)" >> "$log"
  echo "$name $rc $(cat "$P.$name.unit" 2>/dev/null || echo -) $(cat "$P.$name.pid" 2>/dev/null || echo - -)" >> "$P.steps"
}

step() {        # step <name> <script> [args...]: stop the chain at a non-zero exit (ALERT)
  local name=$1
  run_step "$@"
  if [ $rc -ne 0 ]; then
    echo "failed $name $rc" > "$P.result"
    echo "$(date '+%F %T') prep_automate_data $name ended with exit $rc$([ $rc -eq 99 ] && echo ' (caps refused the start for QX_RETRY_MAX_S)') (log $log)" >> "$RUN/logs/ALERT"
    echo "PREP FAILED at $name (exit $rc; log $log)"; tail -5 "$log"
    exit $rc
  fi
  echo "prep step $name done (log $log)"
}

# draws: never stops the chain; each line of its flags_for_user.txt goes to ALERT
DRAWS=$RUN/draws_$(date +%s)
run_step draws scripts/automate/check_expert_data.py --draws-only --data "$OUT" --work "$DRAWS"
if [ $rc -ne 0 ]; then
  echo "$(date '+%F %T') prep_automate_data draws (check_expert_data.py --draws-only) ended with exit $rc (log $log); the row-22 shares were not checked" | tee -a "$RUN/logs/ALERT"
fi
nflags=0
if [ -f "$DRAWS/flags_for_user.txt" ]; then
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    nflags=$((nflags + 1))
    echo "$(date '+%F %T') prep_automate_data draws: SHOW THE USER: $f ($DRAWS/draws.json)" | tee -a "$RUN/logs/ALERT"
  done < "$DRAWS/flags_for_user.txt"
else
  echo "$(date '+%F %T') prep_automate_data draws: no $DRAWS/flags_for_user.txt; the row-22 shares were not checked" | tee -a "$RUN/logs/ALERT"
fi
echo "prep step draws done (exit $rc; log $log; $nflags flags)"

# grasp.json files against the plan's grasps (= the config's expected_grasps, collect.prepare)
count=$("$CL_PY" -I -c '
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
try:
    planned = json.loads((out / "plan.json").read_text()).get("grasps") or []
except (OSError, ValueError) as e:
    print("? of ? planned (plan.json unreadable: %s)" % str(e)[:200]); sys.exit(0)
n = sum((out / g["port"] / g["grasp"] / "grasp.json").exists() for g in planned)
print("%d of %d planned" % (n, len(planned)))' "$OUT")
echo "# grasp.json $count" >> "$P.steps"
echo "grasp.json $count (plan.json)"
# exit 2 of build_index.py: the collection was refused as running or unfinished (wm.data.index.Incomplete)
run_step index scripts/data/build_index.py --automate "$OUT" --workers $PREP_WORKERS "${FLAGS[@]}"
if [ $rc -eq 2 ]; then
  echo "$(date '+%F %T') prep_automate_data index: the collection $OUT was refused as running or unfinished (log $log; --accept-skipped / --partial are the user's choice)" >> "$RUN/logs/ALERT"
fi
if [ $rc -ne 0 ]; then
  echo "failed index $rc" > "$P.result"
  echo "$(date '+%F %T') prep_automate_data index ended with exit $rc$([ $rc -eq 99 ] && echo ' (caps refused the start for QX_RETRY_MAX_S)') (log $log)" >> "$RUN/logs/ALERT"
  echo "PREP FAILED at index (exit $rc$([ $rc -eq 2 ] && echo ': collection running or unfinished'); log $log)"; tail -5 "$log"
  exit $rc
fi
echo "prep step index done (log $log)"
step fast scripts/data/cache_wm_v1_fast.py --src "$OUT" --dst "$FAST" --workers $PREP_WORKERS
# no statistics file of an earlier D_max may remain for train_wm320.py, in either mode (r11-data-01): with D_max none
# it refuses without one; with a numeric D_max a failed step must not leave the previous D_max's file behind. The
# previous run's depth_distribution.json goes too (r12-data-chk-01): its numbers belong to the old training set, and
# they are what the user's D_max question reads
dstep=$([ $DIST_ONLY -eq 1 ] && echo depth_distribution || echo depth_stats)
if ! rm -f "$FAST/depth_stats.json" "$FAST/depth_distribution.json" || [ -e "$FAST/depth_stats.json" ] \
    || [ -e "$FAST/depth_distribution.json" ]; then
  echo "failed $dstep 1" > "$P.result"
  echo "$(date '+%F %T') prep_automate_data $dstep: cannot delete $FAST/depth_stats.json / depth_distribution.json" >> "$RUN/logs/ALERT"
  echo "PREP FAILED at $dstep (cannot delete $FAST/depth_stats.json / depth_distribution.json)"
  exit 1
fi
if [ $DIST_ONLY -eq 1 ]; then
  step depth_distribution scripts/automate/depth_stats.py --data "$FAST" --distribution-only
  echo "ok distribution-only" > "$P.result"
  echo "PREP OK (distribution only) index $OUT/index.json, fast copy $FAST, training-split depth distribution $FAST/depth_distribution.json; logs $P.*"
  echo "NO depth_stats.json: train_wm320.py refuses until depth_stats.py has run with the user's D_max, e.g.:"
  echo "  ( R=$RUN; source \$R/code/scripts/automate/caps.sh; source \$R/code/scripts/automate/collect_lib.sh; cl_isaac_env; cl_clean_env >/dev/null; export PYTHONPATH=\$R/code/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd \$R/code; qx_run_retry prep $(basename "$RUN")-depth_stats $PREP_MEM $PREP_CPU 0 \$CL_PY -u scripts/automate/depth_stats.py --data $FAST --depth-max-m <D_max> )"
  exit 0
fi
step depth_stats scripts/automate/depth_stats.py --data "$FAST" --depth-max-m "$DMAX"
echo ok > "$P.result"
echo "PREP OK index $OUT/index.json, fast copy $FAST, depth statistics $FAST/depth_stats.json (D_max $DMAX m); logs $P.*"
