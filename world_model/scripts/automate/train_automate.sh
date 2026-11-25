#!/bin/bash
# Capped training launcher for AutoMate on rs2 (GPU 1 only, the same caps as the collection):
#   bash train_automate.sh <run dir> <MemoryMax e.g. 40G> <CPUQuota e.g. 800%> <GPU GB e.g. 20> <script> [args...]
# <script> is a path relative to world_model (e.g. scripts/train/train_wm320.py). There are no default resources:
# no AutoMate training run has been measured yet, so the caller states them (from a measured run with headroom).
# 1. Freezes the code: this world_model tree is copied read-only to <run dir>/code (hash in <run dir>/code.sha256);
#    a second start in the same run dir must find the copy unchanged. The script runs from that copy.
# 2. Writes the budget table (our active units, MemoryMax / CPUQuota / declared and measured GPU-1 memory, against
#    125 GB / 3200 % / 48 GB) plus this job to <run dir>/logs/budget_<unix time>.txt and prints it.
# 3. Preflight in a capped CPU scope with the job's own environment (collect_lib.sh cl_train_preflight: cl_isaac_env,
#    cl_clean_env and cl_train_env as in step 4; r7-data-05): the encoder weight files exist under CL_TORCH_HOME /
#    CL_HF_HOME, `wm` imports from <run dir>/code/src and scikit-learn, joblib and threadpoolctl import from the
#    isaaclab env, not from a user site (CL_EXTRA_SITE is empty; versions and files printed).
# 4. Starts the watcher if none runs, then, inside tmux -L qingxin session qingxin-train-<run name>, the job through
#    caps.sh qx_run_retry train (retried every 60 s while the caps refuse, up to QX_RETRY_MAX_S; r9-ops-01; start lock, 90 s after the last GPU start, our caps, CUDA_VISIBLE_DEVICES=1,
#    CUDA_DEVICE_ORDER=PCI_BUS_ID; GPU 0 is never visible). The job's environment: every inherited WM_* unset,
#    WM_FORBIDDEN_GPU=0, PYTHONPATH = only <run dir>/code/src (+ CL_EXTRA_SITE when set), HF_HOME / TORCH_HOME = our
#    caches, HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE = 1, PYTHONNOUSERSITE=1 (collect_lib.sh cl_train_env); qx_run gives the job
#    QX_GPU_GB = <GPU GB>, the cap the job sets on its own CUDA allocator. Log <run dir>/logs/train_<unix time>.log;
#    a non-zero exit (also a caps refusal that lasted QX_RETRY_MAX_S, 99) appends a line to <run dir>/logs/ALERT. No restarts. Exit 7 (a forbidden
#    GPU-0 context, app.assert_off_forbidden_gpu or a stop by qx_watch.py: QX_RUN_DIR/gpu0/<unit>) also writes
#    <run dir>/STOP, and a later start in this run dir is refused until it is removed; exit 8 (nvidia-smi unreadable
#    at a GPU check) is reported as such. A stop by qx_watch.py for memory pressure or GPU-1 memory (QX_RUN_DIR/shed/
#    <unit>) is named in the ALERT line with the watcher's reason and the newest checkpoint of <run dir> (no
#    automatic restart; the marker is consumed).
set -u
if [ "${1:-}" = "--inside" ]; then                       # the tmux side, from the frozen copy
  shift
  RUN=$1 MEM=$2 CPU=$3 GPU=$4 SCRIPT=$5; shift 5
  W=$RUN/code
  source "$W/scripts/automate/caps.sh"
  source "$W/scripts/automate/collect_lib.sh"
  cl_isaac_env
  cl_clean_env || exit 1
  cl_train_env "$W"
  export OMP_NUM_THREADS=$(( ${CPU%\%} / 100 > 0 ? ${CPU%\%} / 100 : 1 ))
  log=$RUN/logs/train_$(date +%s).log
  { date; echo "train_automate: $SCRIPT $*"; echo "code $W sha256 $(cat "$RUN/code.sha256")"
    echo "PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME"; } > "$log"
  cd "$W"
  ufile=$RUN/logs/train_unit_$(date +%s)
  QX_UNIT_FILE=$ufile qx_run_retry train "$(basename "$RUN")" "$MEM" "$CPU" "$GPU" "$CL_PY" -u "$SCRIPT" "$@" >> "$log" 2>&1
  rc=$?
  echo "exit $rc $(date)" >> "$log"
  unit=$(cat "$ufile" 2>/dev/null)
  if [ $rc -eq 7 ] || { [ -n "$unit" ] && [ -e "$QX_RUN_DIR/gpu0/$unit" ]; }; then
    msg="training $SCRIPT: forbidden GPU-0 context (exit $rc, unit ${unit:-?}); STOP written, no restart (log $log)"
    echo "$(date '+%F %T') $msg" >> "$RUN/STOP"
    echo "$(date '+%F %T') $msg" >> "$RUN/logs/ALERT"
  elif [ -n "$unit" ] && [ -e "$QX_RUN_DIR/shed/$unit" ]; then
    msg="training $SCRIPT stopped by qx_watch.py (exit $rc): $(head -c 300 "$QX_RUN_DIR/shed/$unit" | tr '\n' ' ')(log $log;"
    msg="$msg newest checkpoint: $(ls -t "$RUN"/*.pt "$RUN"/*/*.pt 2>/dev/null | head -1)); no restart"
    rm -f "$QX_RUN_DIR/shed/$unit"
    echo "$(date '+%F %T') $msg" >> "$RUN/logs/ALERT"
  elif [ $rc -eq 8 ]; then
    echo "$(date '+%F %T') training $SCRIPT: nvidia-smi unreadable at a GPU check (exit 8; log $log)" >> "$RUN/logs/ALERT"
  elif [ $rc -ne 0 ]; then
    echo "$(date '+%F %T') training $SCRIPT ended with exit $rc (log $log)" >> "$RUN/logs/ALERT"
  elif [ "$SCRIPT" = scripts/train/train_wm320.py ] && [ ! -e "$RUN/DONE.json" ]; then   # a signal stop at a step boundary
    echo "$(date '+%F %T') training $SCRIPT exited 0 without $RUN/DONE.json (stopped early; last.pt kept; log $log)" >> "$RUN/logs/ALERT"
  fi
  exit $rc
fi
[ $# -ge 5 ] || { echo "usage: bash train_automate.sh <run dir> <MemoryMax> <CPUQuota> <GPU GB> <script> [args...]"; exit 1; }
RUN=$(readlink -f "$1"); MEM=$2; CPU=$3; GPU=$4; SCRIPT=$5; shift 5
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
[[ "$MEM" =~ ^[0-9]+G$ ]] || { echo "refused: MemoryMax '$MEM' must be <integer>G"; exit 1; }
[[ "$CPU" =~ ^[0-9]+%$ ]] || { echo "refused: CPUQuota '$CPU' must be <integer>%"; exit 1; }
[[ "$GPU" =~ ^[1-9][0-9]*$ ]] || { echo "refused: GPU GB '$GPU' must be an integer >= 1"; exit 1; }
cl_static_budget 1 "$MEM" "$CPU" "$GPU" || exit 1
[ -e "$RUN/STOP" ] && { echo "refused: $RUN/STOP exists ($(tail -1 "$RUN/STOP")); remove it after checking the cause"; exit 1; }
mkdir -p "$RUN/logs"
cl_freeze_code "$LIVE" "$RUN" || exit 1
[ -f "$RUN/code/$SCRIPT" ] || { echo "refused: $SCRIPT not in $RUN/code"; exit 1; }
cl_train_preflight "$RUN/code" || exit 1
sess=qingxin-train-$(basename "$RUN")
$CL_TMUX has-session -t "=$sess" 2>/dev/null && { echo "refused: tmux -L qingxin session $sess exists"; exit 1; }
b=$RUN/logs/budget_$(date +%s).txt
{ echo "budget before starting $SCRIPT in $RUN"; qx_budget; echo "new: train MemoryMax $MEM, CPUQuota $CPU, GPU-1 $GPU GB"; } | tee "$b"
cl_watch_ensure "$RUN" "$RUN/code" || { echo "refused: no live watcher (qx_watch.py)"; exit 1; }
$CL_TMUX new -d -s "$sess" bash "$RUN/code/scripts/automate/train_automate.sh" --inside "$RUN" "$MEM" "$CPU" "$GPU" "$SCRIPT" "$@" ||
  { echo "refused: tmux failed"; exit 1; }
echo "started in tmux -L qingxin session $sess; logs in $RUN/logs"
