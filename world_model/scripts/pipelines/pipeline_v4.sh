#!/bin/bash
# v4 (docs/design_v4.md): UNet -> crops -> crop autoencoder -> latents -> world model / success classifier
# / FM offline -> online (actors + learners). Every stage is first run on a small subset (smoke) and the
# pipeline stops at the first failure; finished stages are skipped on a rerun (their outputs exist).
#   tmux new-session -d -s qingxin-v4 "bash world_model/scripts/pipelines/pipeline_v4.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
SM=$Q/cache/smoke_v4
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {   # run <name> <gpu memory cap> <command...>: stop the pipeline on failure
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4_$name.log) -- pipeline stopped"; exit 1; fi
}
smoke_cfg() {  # smoke_cfg <in.json> <out.json> key=value ... (values are JSON)
    python - "$@" <<'PY'
import json, sys
src, dst, *kv = sys.argv[1:]
c = json.load(open(src))
for item in kv:
    k, v = item.split("=", 1)
    d = c
    *path, last = k.split(".")
    for p in path:
        d = d[p]
    d[last] = json.loads(v)
json.dump(c, open(dst, "w"), indent=1)
PY
}
mkdir -p $SM
stamp "pipeline v4 started"

# ---- 0. UNet (running separately in qingxin-unet; wait for it)
while tmux has-session -t qingxin-unet 2>/dev/null; do sleep 60; done
[ -f $Q/runs/unet_v4/best.pt ] || { stamp "FAILED: no UNet checkpoint"; exit 1; }
stamp "UNet finished: $(grep val_score $Q/logs/unet_v4.log | tail -1 | cut -c1-400)"

# ---- 1. crops
run smoke_crops 0.05 python $S/data/place_crops.py --data $Q/data/wm_v1 --unet $Q/runs/unet_v4/best.pt --out $SM/crops --workers 4 --limit 40
[ -f $Q/data/wm_v1_crops/summary.json ] || run crops 0.08 python $S/data/place_crops.py --data $Q/data/wm_v1 --unet $Q/runs/unet_v4/best.pt --out $Q/data/wm_v1_crops --workers 12
stamp "crops: $(cat $Q/data/wm_v1_crops/summary.json | tr -d '\n ')"

# ---- 2. crop autoencoder
smoke_cfg $C/crop_ae.json $SM/crop_ae.json crops=\"$SM/crops\" workers=4 eval_batches=2 batch_size=8
run smoke_crop_ae 0.08 python $S/train/train_crop_ae.py --config $SM/crop_ae.json --run $SM/crop_ae --steps 30
[ -f $Q/runs/crop_ae_v4/done ] || { run crop_ae 0.22 python $S/train/train_crop_ae.py --config $C/crop_ae.json --run $Q/runs/crop_ae_v4 && touch $Q/runs/crop_ae_v4/done; }

# ---- 3. latents
run smoke_encode 0.05 python $S/train/encode_latents_v4.py --ae $SM/crop_ae/last.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $SM/crops --out $SM/latents --workers 4 --limit 40
[ -f $Q/data/wm_v4_latents/stats.json ] || run encode 0.10 python $S/train/encode_latents_v4.py --ae $Q/runs/crop_ae_v4/best.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v4_latents --workers 12

# ---- 4. offline: world model, success classifier, FM (smoke each, then in parallel)
smoke_cfg $C/wm.json $SM/wm.json latents=\"$SM/latents\" ae_checkpoint=\"$SM/crop_ae/last.pt\" workers=2 batch_size=4 accum=1 eval_batches=1 eval_sampling_steps=5 eval_every=20 save_every=10
run smoke_wm 0.10 python $S/train/train_wm4.py --config $SM/wm.json --run $SM/wm --steps 20
smoke_cfg $C/success.json $SM/success.json latents=\"$SM/latents\" workers=2 batch_size=8 eval_batches=2 eval_every=20 save_every=10
run smoke_success 0.05 python $S/train/train_success.py --config $SM/success.json --run $SM/success --steps 20
smoke_cfg $C/policy.json $SM/policy.json latents=\"$SM/latents\" workers=2 batch_size=16 eval_batches=2 eval_every=20
run smoke_policy 0.05 python $S/train/train_policy.py --config $SM/policy.json --run $SM/policy --max-steps 20
stamp "offline smoke tests passed"
( [ -f $Q/runs/wm_v4/done ] || { run wm 0.24 python $S/train/train_wm4.py --config $C/wm.json --run $Q/runs/wm_v4 && touch $Q/runs/wm_v4/done; } ) &
P1=$!
( [ -f $Q/runs/success_v4/done ] || { run success 0.06 python $S/train/train_success.py --config $C/success.json --run $Q/runs/success_v4 && touch $Q/runs/success_v4/done; } ) &
P2=$!
( [ -f $Q/runs/policy_v4/done ] || { run policy 0.06 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v4 && touch $Q/runs/policy_v4/done; } ) &
P3=$!
wait $P1 || { stamp "FAILED offline world model -- stopped"; exit 1; }
wait $P2 || { stamp "FAILED offline success classifier -- stopped"; exit 1; }
wait $P3 || { stamp "FAILED offline FM -- stopped"; exit 1; }
for r in wm_v4 success_v4 policy_v4; do [ -f $Q/runs/$r/done ] || { stamp "FAILED: $r not finished"; exit 1; }; done

# ---- 5. online (smoke: 1 actor x 2 episodes, learners a few steps; then actors + learners for good)
smoke_cfg $C/online.json $SM/online.json dir=\"$SM/online\" learner.min_online_files=1 learner.refresh_every=5 learner.save_every=5 learner.online_workers=1 \
    actor.wm_live=\"$SM/online_wm/current.pt\" actor.success_live=\"$SM/online_success/current.pt\" actor.reload_every=1
run smoke_actors 0.05 python $S/online/run_online.py actors --config $SM/online.json --workers 1 --episodes 2
if grep -q '"error"' $SM/online/actor_0.jsonl; then stamp "FAILED smoke actors: $(grep -m1 error $SM/online/actor_0.jsonl | cut -c1-300)"; exit 1; fi
cp $Q/runs/wm_v4/last.pt $SM/wm_last.pt; cp $Q/runs/success_v4/last.pt $SM/success_last.pt
smoke_cfg $SM/online.json $SM/online.json actor.wm_init_last=\"$SM/wm_last.pt\" actor.success_init_last=\"$SM/success_last.pt\"
WS=$(python -c "import torch;print(torch.load('$SM/wm_last.pt',map_location='cpu',weights_only=False)['step']+10)")
SS=$(python -c "import torch;print(torch.load('$SM/success_last.pt',map_location='cpu',weights_only=False)['step']+10)")
run smoke_learner_wm 0.24 python $S/online/run_online.py wm --config $SM/online.json --steps $WS
run smoke_learner_success 0.06 python $S/online/run_online.py success --config $SM/online.json --steps $SS
stamp "online smoke tests passed; starting the online stage"
mkdir -p $Q/data/wm_v4_online
tmux new-session -d -s qingxin-v4-learner-wm "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.22 nice -n 10 python $S/online/run_online.py wm --config $C/online.json > $Q/logs/v4_online_wm.log 2>&1"
tmux new-session -d -s qingxin-v4-learner-success "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/online/run_online.py success --config $C/online.json > $Q/logs/v4_online_success.log 2>&1"
tmux new-session -d -s qingxin-v4-actors "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.03 nice -n 10 python $S/online/run_online.py actors --config $C/online.json --workers 8 > $Q/logs/v4_online_actors.log 2>&1"
stamp "online stage running (tmux qingxin-v4-actors / -learner-wm / -learner-success)"
