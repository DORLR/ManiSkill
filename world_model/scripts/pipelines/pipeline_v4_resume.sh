#!/bin/bash
# Continuation of pipeline_v4.sh after the success classifier failed on 2026-11-10 13:04 (trajectories
# without commands; fixed in data/history.py). The world model and the FM keep running in the first
# pipeline's tmux; this script reruns the classifier (smoke + full), waits for all three, then starts
# the online stage exactly as pipeline_v4.sh section 5.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
SM=$Q/cache/smoke_v4
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4_$name.log) -- pipeline stopped"; exit 1; fi
}
stamp "resume: success classifier fixed (no-command trajectories)"
rm -rf $SM/success
run smoke_success 0.05 python $S/train/train_success.py --config $SM/success.json --run $SM/success --steps 20
[ -f $Q/runs/success_v4/done ] || { rm -rf $Q/runs/success_v4; run success 0.06 python $S/train/train_success.py --config $C/success.json --run $Q/runs/success_v4 && touch $Q/runs/success_v4/done; }
while [ ! -f $Q/runs/wm_v4/done ] || [ ! -f $Q/runs/policy_v4/done ]; do
    if ! pgrep -u ramy -f "train_wm4.py --config world_model/configs/v4/wm.json" >/dev/null && [ ! -f $Q/runs/wm_v4/done ]; then
        stamp "FAILED: world model stopped without finishing -- pipeline stopped"; exit 1; fi
    if ! pgrep -u ramy -f "train_policy.py --config world_model/configs/v4/policy.json" >/dev/null && [ ! -f $Q/runs/policy_v4/done ]; then
        stamp "FAILED: FM stopped without finishing -- pipeline stopped"; exit 1; fi
    sleep 120
done
stamp "offline stage finished (world model, classifier, FM)"
python - $C/online.json $SM/online.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
sm = "/mnt/storage2/qingxin/cache/smoke_v4"
c["dir"] = f"{sm}/online"
c["learner"].update(min_online_files=1, refresh_every=5, save_every=5, online_workers=1)
c["actor"].update(wm_live=f"{sm}/online_wm/current.pt", success_live=f"{sm}/online_success/current.pt", reload_every=1,
                  wm_init_last=f"{sm}/wm_last.pt", success_init_last=f"{sm}/success_last.pt")
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
rm -rf $SM/online $SM/online_wm $SM/online_success
run smoke_actors 0.05 python $S/online/run_online.py actors --config $SM/online.json --workers 1 --episodes 2
if grep -q '"error"' $SM/online/actor_0.jsonl; then stamp "FAILED smoke actors: $(grep -m1 error $SM/online/actor_0.jsonl | cut -c1-300)"; exit 1; fi
cp $Q/runs/wm_v4/last.pt $SM/wm_last.pt; cp $Q/runs/success_v4/last.pt $SM/success_last.pt
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
