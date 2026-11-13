#!/bin/bash
# The direct-prediction error (check_wm6_horizons.py) on every checkpoint the checks pipeline copies; one line per checkpoint.
#   usage: v6_horizons_at.sh <run> <tmux session of the training> <steps ...>
set -uo pipefail
RUN=$1; SESSION=$2; shift 2
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
for K in "$@"; do
    until [ -f $Q/runs/checks_v6/${RUN}_$K.pt ]; do
        tmux has-session -t $SESSION 2>/dev/null || exit 1
        sleep 180
    done
    sleep 20
    if WM_GPU_MEM_FRACTION=0.10 nice -n 10 python world_model/scripts/analysis/check_wm6_horizons.py --ckpt $Q/runs/checks_v6/${RUN}_$K.pt --shifts 2 10 20 --batches 4 --samplers ddpm20 --out $Q/runs/checks_v6/horizons_${RUN}_$K.json > $Q/logs/v6_horizons_${RUN}_$K.log 2>&1; then
        echo "[$(date +%m-%d\ %H:%M)] v6 direct prediction ($RUN step $K; feature mse predicted | nothing changes; read-out mm predicted | real | nothing changes): $(python -c "
import json; r = json.load(open('$Q/runs/checks_v6/horizons_${RUN}_$K.json'))
print({k: (v['feature_mse_ddpm20'], v['feature_mse_nothing_changes'], v['read_out_ddpm20_mm'], v['read_out_real_mm'], v['read_out_nothing_changes_mm']) for k, v in r.items() if isinstance(v, dict)})")" >> $Q/logs/pipeline_v4.log
    fi
done
