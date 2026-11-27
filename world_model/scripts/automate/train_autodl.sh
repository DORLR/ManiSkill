#!/bin/bash
# AutoDL launcher for the AutoMate world model (2x RTX 4080 SUPER, dedicated machine): the rs2 recipe of
# wm320_segformer_b1_s0 unchanged except the GPU split: torchrun 2 processes x --accum 2 = the single-GPU --accum 4
# (train_wm320.py: processes x accum keeps the single-GPU recipe). From scratch, offline (no --online pool yet).
#   bash train_autodl.sh <out dir> [extra args, e.g. --steps 50 for a smoke]
set -u
OUT=$1; shift
W=/root/autodl-tmp/code/wm/world_model
for v in $(env | grep -o "^WM_[A-Z0-9_]*"); do unset $v; done   # as cl_clean_env on rs2
export WM_FORBIDDEN_GPU=0 WM_DEDICATED_GPUS=1 PYTHONPATH=$W/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TDMPC2_DIR=/root/autodl-tmp/code/ext/tdmpc2_orig/tdmpc2 TDMPC2_EXTRA=/root/autodl-tmp/code/ext/tdmpc2_extra   # TD-MPC2 332f19e, copied from rs2
export HF_HOME=/root/autodl-tmp/cache/hf TORCH_HOME=/root/autodl-tmp/cache/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
mkdir -p "$OUT"; cd $W
LOG=$OUT/train_$(date +%s).log
# memory/GPU log every 30 s while training (the container limit is 124 GiB; nothing else caps it)
( while true; do echo "$(date +%T) mem_GB $(awk '{printf "%.1f", $1/1e9}' /sys/fs/cgroup/memory.current) anon_GB $(awk '/^anon /{printf "%.1f", $2/1e9}' /sys/fs/cgroup/memory.stat) file_GB $(awk '/^file /{printf "%.1f", $2/1e9}' /sys/fs/cgroup/memory.stat) gpu_MiB $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"; sleep 30; done ) > $OUT/resources.log 2>&1 &
MON=$!
{ date; echo "code $(git -C /root/autodl-tmp/code/wm rev-parse HEAD) dirty: $(git -C /root/autodl-tmp/code/wm status --porcelain | wc -l)"; } > $LOG
/root/miniconda3/envs/wm/bin/torchrun --standalone --nproc_per_node 2 scripts/train/train_wm320.py \
  --data /root/autodl-tmp/automate/fast --out "$OUT" --stem standard --encoder segformer_b1 --imagenet --token-dim 128 \
  --offset-weight 1.0 --shift-px 4 --shaping 0.3 --steps 100000 --wm-batch 8 --bc-batch 16 --accum 2 --save-every 500 \
  --lr 3e-4 --enc-lr-scale 0.3 --td-warmup 5000 --bc-weight 1.0 --gamma 0.9666666666666667 --workers-wm 8 --workers-bc 8 \
  --eval-every 5000 --eval-windows 1024 --keep-every 10000 --seed 0 --device cuda:0 "$@" >> $LOG 2>&1
rc=$?; kill $MON 2>/dev/null; echo "exit $rc $(date)" >> $LOG
