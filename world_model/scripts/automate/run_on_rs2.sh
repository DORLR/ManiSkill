#!/bin/bash
# Run one of our AutoMate scripts on rs2 with our Isaac Lab environment (GPU 1 only, memory-capped scope).
#   [MEMMAX=40G] [CPUQUOTA=400%] [GPU_GB=7] [TIMEOUT=7200] [QX_SWITCHES=<switches.env>] bash run_on_rs2.sh <log name> <script.py> [args...]
# The scope is started by caps.sh qx_run (scope qingxin-run-<log name>-*, start lock, our caps, GPU 1 only).
# Code: the world_model tree this script lies in (parallel_batch.sh runs it from a frozen copy); outputs and logs:
# /mnt/storage3/qingxin/automate/m1. Environment: every inherited WM_* variable is unset, WM_FORBIDDEN_GPU=0, then the
# switches file QX_SWITCHES is sourced if given (collect_lib.sh cl_clean_env), so the switches are the code's defaults
# or the named file's, never a stale export of the tmux server.
# TIMEOUT (s): `timeout -k 120` sends SIGTERM after TIMEOUT s and SIGKILL 120 s later (exit 124 = the job overran:
# a hang). QX_UNIT_FILE, when set by the caller, receives the scope's unit name (caps.sh qx_run).
# Exit code: the job's, or 99 when the caps refused the start. Exit 7 (a forbidden GPU-0 context,
# app.assert_off_forbidden_gpu) and 8 (nvidia-smi unreadable at the GPU check) are logged as such; this script never
# restarts anything (run_batch.sh decides).
set -u
W=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$W/scripts/automate/caps.sh"
source "$W/scripts/automate/collect_lib.sh"
cl_isaac_env
export PIP_CACHE_DIR=/mnt/storage3/qingxin/isaac/cache/pip OMP_NUM_THREADS=4
LOGS=/mnt/storage3/qingxin/automate/m1/logs
mkdir -p "$LOGS"
name=$1; shift
script=$1; shift
cl_clean_env "${QX_SWITCHES:-}" > "$LOGS/$name.log" 2>&1 || { cat "$LOGS/$name.log"; exit 3; }
cd "$W"
{ date; echo "code $W, switches ${QX_SWITCHES:-defaults}: $(compgen -e | grep '^WM_' | while read -r v; do echo -n "$v=${!v} "; done)"
  free -g | head -2; env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; } >> "$LOGS/$name.log"
qx_run run "$name" "${MEMMAX:-40G}" "${CPUQUOTA:-400%}" "${GPU_GB:-7}" timeout -k 120 ${TIMEOUT:-7200} "$CL_PY" -u "$script" "$@" >> "$LOGS/$name.log" 2>&1
rc=$?
echo "exit $rc $(date)" >> "$LOGS/$name.log"
[ $rc -eq 7 ] && echo "$(date '+%F %T') run_on_rs2 $name: exit 7, a forbidden GPU-0 context; no restart" | tee -a "$QX_RUN_DIR/ALERT" >> "$LOGS/$name.log"
[ $rc -eq 8 ] && echo "$(date '+%F %T') run_on_rs2 $name: exit 8, nvidia-smi unreadable at the GPU check" >> "$LOGS/$name.log"
exit $rc
