#!/bin/bash
# AutoDL launcher for the AutoMate world model (2x RTX 4080 SUPER 32 GB, the container is ours alone): the rs2 recipe
# of wm320_segformer_b1_s0 unchanged except the GPU split.
#   bash train_autodl.sh <GPUs 1|2> <run dir> [extra train_wm320.py args, e.g. --resume, --online <pool>, --steps 600]
# GPUs 2: torchrun, 2 processes x --accum 2 (train_wm320.py: processes x accum = the single-GPU --accum 4 recipe).
# GPUs 1: one process on GPU 0 with --accum 4 (GPU 1 is left to the online workers; a run may be resumed with any
#         number of processes, train_wm320.py docstring).
# Like train_automate.sh: the world_model tree is frozen to <run dir>/code (cl_freeze_code; a later start must find it
# unchanged) and the run imports only from there; the "unit" (caps.sh / autodl_overrides.sh) is written to
# <run dir>/logs/train_unit_<time> and its pid to $QX_RUN_DIR/units, so online_automate.sh sees the learner.
set -u
NG=$1 RUN=$(readlink -m "$2"); shift 2
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"; source "$LIVE/scripts/automate/collect_lib.sh"
[ "${QX_AUTODL:-}" = 1 ] || { echo "refused: not on AutoDL (no /root/autodl-tmp)"; exit 1; }
case $NG in 1) ACC=4;; 2) ACC=2;; *) echo "refused: GPUs must be 1 or 2"; exit 1;; esac
[ "$NG" = 2 ] && [[ " $* " == *" --online "* ]] && { echo "refused: --online needs GPU 1 for the workers: use 1 GPU"; exit 1; }
LMEM=${QX_LEARNER_MEM:-60G}    # the learner's declared memory, counted by qx_check against the workers (measured in the smoke)
[ -e "$RUN/STOP" ] && { echo "refused: $RUN/STOP exists"; exit 1; }
mkdir -p "$RUN/logs"
cl_freeze_code "$LIVE" "$RUN" || exit 1
W=$RUN/code
for v in $(compgen -e | grep '^WM_'); do unset "$v"; done      # as cl_clean_env
export WM_FORBIDDEN_GPU=0 WM_DEDICATED_GPUS=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cl_isaac_env; cl_train_env "$W"
ts=$(date +%s); LOG=$RUN/logs/train_$ts.log; unit=qingxin-train-$(basename "$RUN")-$ts
echo "$unit" > "$RUN/logs/train_unit_$ts"
{ date; echo "train_autodl: GPUs $NG accum $ACC, code $W sha256 $(cat "$RUN/code.sha256"), args $*"
  /root/miniconda3/envs/wm/bin/python -c "import torch, transformers, sklearn, numpy; print('learner env: torch', torch.__version__, 'transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__)"; } > "$LOG" 2>&1
ARGS=(--data /root/autodl-tmp/automate/fast --out "$RUN" --stem standard --encoder segformer_b1 --imagenet --token-dim 128
  --offset-weight 1.0 --shift-px 4 --shaping 0.3 --steps 100000 --wm-batch 8 --bc-batch 16 --accum $ACC --save-every 500
  --lr 3e-4 --enc-lr-scale 0.3 --td-warmup 5000 --bc-weight 1.0 --gamma 0.9666666666666667 --workers-wm 8 --workers-bc 8
  --eval-every 5000 --eval-windows 1024 --keep-every 10000 --seed 0 --device cuda:0 "$@")
( while true; do echo "$(date +%T) mem_GB $(awk '{printf "%.1f", $1/1e9}' /sys/fs/cgroup/memory.current) anon_GB $(awk '/^anon /{printf "%.1f", $2/1e9}' /sys/fs/cgroup/memory.stat) file_GB $(awk '/^file /{printf "%.1f", $2/1e9}' /sys/fs/cgroup/memory.stat) gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"; sleep 30; done ) >> "$RUN/logs/resources.log" 2>&1 &
MON=$!
cd "$W"
if [ "$NG" = 2 ]; then
  setsid env CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID /root/miniconda3/envs/wm/bin/torchrun --standalone --nproc_per_node 2 scripts/train/train_wm320.py "${ARGS[@]}" >> "$LOG" 2>&1 &
else
  setsid env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID /root/miniconda3/envs/wm/bin/python -u scripts/train/train_wm320.py "${ARGS[@]}" >> "$LOG" 2>&1 &
fi
pid=$!; echo "$pid $(qx_pid_start "$pid")" > "$QX_RUN_DIR/units/$unit.pid"; echo "$NG $LMEM 1600%" > "$QX_RUN_DIR/qx/$unit.gpu"
wait $pid; rc=$?
rm -f "$QX_RUN_DIR/units/$unit.pid" "$QX_RUN_DIR/qx/$unit.gpu"; kill $MON 2>/dev/null
echo "exit $rc $(date)" >> "$LOG"
if [ $rc -ne 0 ]; then echo "$(date '+%F %T') training ended with exit $rc (log $LOG)" >> "$RUN/logs/ALERT"
elif [ ! -e "$RUN/DONE.json" ]; then echo "$(date '+%F %T') training exited 0 without DONE.json (stopped early; last.pt kept; log $LOG)" >> "$RUN/logs/ALERT"; fi
exit $rc
