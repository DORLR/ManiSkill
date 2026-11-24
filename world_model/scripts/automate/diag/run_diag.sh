#!/bin/bash
# Run a diagnosis script on rs2: our env, GPU 1, capped scope qingxin-diag-* (caps.sh qx_run).  bash run_diag.sh <log name> <MemoryMax> <script> [args]
# Exit 7 (a forbidden GPU-0 context, app.assert_off_forbidden_gpu) writes $LOGS/STOP_GPU0; while that file exists no
# diagnosis starts (run_groups.sh / run_queue_f.sh run several in a row): remove it after checking the cause.
# Exit 8 (nvidia-smi unreadable at the GPU check) is logged as such.
set -u
export HOME=/mnt/storage3/qingxin/isaac/home XDG_CACHE_HOME=/mnt/storage3/qingxin/isaac/cache
export TMPDIR=/mnt/storage3/qingxin/isaac/tmp OMNI_KIT_ACCEPT_EULA=YES NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4
PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python
W=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
LOGS=/mnt/storage3/qingxin/automate/m1b/diag/logs
mkdir -p $LOGS
name=$1; mem=$2; shift 2
[ -e "$LOGS/STOP_GPU0" ] && { echo "refused: $LOGS/STOP_GPU0 exists ($(tail -1 "$LOGS/STOP_GPU0"))"; exit 7; }
cd $W
{ date; free -g | head -2; env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; } > $LOGS/$name.log
source "$W/scripts/automate/caps.sh"
qx_run diag "$name" "$mem" "${CPUQUOTA:-400%}" "${GPU_GB:-7}" timeout ${TIMEOUT:-5400} $PY -u "$@" >> $LOGS/$name.log 2>&1
rc=$?
echo "exit $rc $(date)" >> $LOGS/$name.log
[ $rc -eq 7 ] && echo "$(date '+%F %T') $name: exit 7, a forbidden GPU-0 context; no further diagnosis starts" | tee -a "$LOGS/STOP_GPU0" >> "$QX_RUN_DIR/ALERT"
[ $rc -eq 8 ] && echo "$(date '+%F %T') $name: exit 8, nvidia-smi unreadable at the GPU check" >> $LOGS/$name.log
exit $rc
