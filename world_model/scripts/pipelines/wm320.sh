#!/bin/bash
# The 320 px world model (scripts/train/train_wm320.py): training on the GPU(s) given, under the budget of lib320.sh.
#   wm320.sh <stem> <gpus> <shift_px> <shaping> <wm_batch> <bc_batch> <accum> <gpu_mib> <ram_gb> <workers_wm> <workers_bc> [new|resume] [seed] [online]
# "online": the learner mixes in the complete rounds of $Q/data/online320_<run name> (online320.sh writes them).
# <gpus>: one GPU, or a comma list (e.g. 3,4,5,6,7) for data parallelism: one process per GPU started by torchrun
# (python -m torch.distributed.run --standalone); <wm_batch>, <bc_batch>, <accum>, <gpu_mib> and the loader workers are
# per process, <ram_gb> is the whole job's cap.
# Run it from a frozen copy of the code (code/frozen/<tag>/world_model/scripts/pipelines/wm320.sh): the training then
# imports and runs that copy, so later edits of the repository never reach a running job.
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
stem=${1:?stem}; gpu=${2:?gpu}; shift_px=${3:?shift px}; shaping=${4:?shaping weight}; wb=${5:?windows per micro-batch}
bb=${6:?imitation samples per micro-batch}; acc=${7:?accum}; mib=${8:?GPU MiB}; ram=${9:?RAM GB}; ww=${10:?WM loader workers}
bw=${11:?imitation loader workers}; mode=${12:-new}; seed=${13:-0}; online=${14:-}
case "$online" in ""|online) ;; *) echo "the 14th argument is 'online' or nothing"; exit 1 ;; esac
q320_gpus_ok "$gpu" || { echo "GPUs must be one of 0-7 or a comma list of distinct ones"; exit 1; }
n=$(tr ',' '\n' <<< "$gpu" | wc -l)
[ "$n" -eq 1 ] || q320_watchdog_lists || { q320_log "wm320 $stem: the watchdog's lib320.sh does not know GPU lists; restart it from this version first"; exit 1; }
for v in "$shift_px" "$wb" "$bb" "$acc" "$mib" "$ram" "$ww" "$bw" "$seed"; do q320_int "$v" || { echo "not a whole number: $v"; exit 1; }; done
[[ "$shaping" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "shaping must be a number"; exit 1; }
case "$mode" in new|resume) ;; *) echo "mode must be new or resume"; exit 1 ;; esac
cd "$CODE" || exit 1
export PYTHONPATH=$CODE/world_model/src${PYTHONPATH:+:$PYTHONPATH}
name=wm320_${stem}_s$seed; O=$Q/runs/$name; extra=()
[ -z "$online" ] || { mkdir -p "$Q/data/online320_$name"; extra+=(--online "$Q/data/online320_$name"); }
if [ "$mode" = resume ]; then extra+=(--resume)
elif [ -e "$O/log.jsonl" ]; then q320_log "$name: $O exists; pass 'resume' to continue it"; exit 1; fi
launch=(python); keep=600                  # MiB outside torch's allocator: the CUDA context (+ NCCL's buffers with several GPUs)
[ "$n" -gt 1 ] && { launch=(python -m torch.distributed.run --standalone --nproc_per_node="$n"); keep=1200; }
frac=$(q320_fraction $(( mib - keep )) "$gpu") || { q320_log "wm320 $stem: no memory fraction for $mib MiB on GPUs $gpu"; exit 1; }
q320_run_when_fits "qx320-train-$name" "$ram" "$gpu" "$mib" train "$frac" "$Q/logs/${name}_train.log" "${launch[@]}" world_model/scripts/train/train_wm320.py \
  --data "$Q/data/wm_v1_fast" --out "$O" --stem "$stem" --shift-px "$shift_px" --shaping "$shaping" --wm-batch "$wb" --bc-batch "$bb" \
  --accum "$acc" --workers-wm "$ww" --workers-bc "$bw" --seed "$seed" "${extra[@]}"
rc=$?
if [ $rc -eq 0 ] && [ -e "$O/DONE.json" ]; then q320_log "$name: training finished"; else q320_log "$name: training did not end regularly (code $rc)"; exit 1; fi
