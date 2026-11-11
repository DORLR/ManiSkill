#!/bin/bash
# v4c (decided with the user 2026-11-10 15:50: mixed training from the start, no offline-only stage).
# After the crop autoencoder (pipeline_v4b): encode -> world model + success classifier from scratch on the
# mixture (expert data; half online episodes once >= 50 exist) and the FM -> when the FM is trained, 8 actors
# collect FM-only episodes (p_plan 0; planning is switched on by hand after a consistency check).
# Every stage is smoke-tested first; the pipeline stops at the first failure.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
SM=$Q/cache/smoke_v4c
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4c_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4c_$name.log) -- pipeline stopped"; exit 1; fi
}
cfg() {  # cfg <in.json> <out.json> key=value ... (JSON values, dotted keys)
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
rm -rf $SM; mkdir -p $SM
while tmux has-session -t qingxin-v4b 2>/dev/null; do sleep 60; done
grep -q "done crop_ae" $LOG || { stamp "FAILED: crop autoencoder did not finish -- v4c not started"; exit 1; }
stamp "v4c started (crop autoencoder best: $(python -c "import torch;c=torch.load('$Q/runs/crop_ae_v4b/best.pt',map_location='cpu',weights_only=False);print('step',c['step'])"))"

# ---- latents
run smoke_encode 0.05 python $S/train/encode_latents_v4.py --ae $Q/runs/crop_ae_v4b/best.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $SM/latents --workers 4 --limit 40
[ -f $Q/data/wm_v4_latents/stats.json ] || run encode 0.10 python $S/train/encode_latents_v4.py --ae $Q/runs/crop_ae_v4b/best.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v4_latents --workers 12
python - <<PY >> $LOG
import json, numpy as np, glob
s = json.load(open("$Q/data/wm_v4_latents/stats.json"))
fs = glob.glob("$Q/data/wm_v4_latents/**/*.npz", recursive=True)
bad = 0
for f in fs[::50]:
    m = np.load(f)["mean"].astype(np.float32)
    bad += int(not np.isfinite(m).all())
print("[latents] files %d, scale %.4f, sampled files with non-finite latents: %d" % (len(fs), s["scale"], bad))
PY
grep -q "with non-finite latents: 0" $LOG || { stamp "FAILED: non-finite latents -- stopped"; exit 1; }

# ---- smoke: FM, both learners from scratch on the mixture, one FM-only actor episode
cfg $C/wm.json $SM/wm.json latents=\"$SM/latents\" ae_checkpoint=\"$Q/runs/crop_ae_v4b/best.pt\" workers=2 batch_size=4 accum=1 eval_batches=1 eval_sampling_steps=5 eval_every=20 save_every=10
cfg $C/success.json $SM/success.json latents=\"$SM/latents\" workers=2 batch_size=8 eval_batches=2 eval_every=20 save_every=10
cfg $C/policy.json $SM/policy.json latents=\"$SM/latents\" workers=2 batch_size=16 eval_batches=2 eval_every=20
run smoke_policy 0.05 python $S/train/train_policy.py --config $SM/policy.json --run $SM/policy --max-steps 20
cfg $C/online.json $SM/online.json dir=\"$SM/online\" latents=\"$SM/latents\" wm_config=\"$SM/wm.json\" success_config=\"$SM/success.json\" \
    actor.policy=\"$SM/policy/best.pt\" actor.ae=\"$Q/runs/crop_ae_v4b/best.pt\" actor.wm_init=\"$SM/none.pt\" actor.wm_init_last=\"$SM/none.pt\" \
    actor.success_init=\"$SM/none.pt\" actor.success_init_last=\"$SM/none.pt\" actor.wm_live=\"$SM/online_wm/current.pt\" \
    actor.success_live=\"$SM/online_success/current.pt\" learner.min_online_files=1 learner.refresh_every=5 learner.save_every=5 learner.online_workers=1
run smoke_actor_fm 0.05 python $S/online/run_online.py actors --config $SM/online.json --workers 1 --episodes 1
if grep -q '"error"' $SM/online/actor_0.jsonl; then stamp "FAILED smoke actor: $(grep -m1 error $SM/online/actor_0.jsonl | cut -c1-300)"; exit 1; fi
run smoke_learner_wm 0.24 python $S/online/run_online.py wm --config $SM/online.json --steps 20
run smoke_learner_success 0.06 python $S/online/run_online.py success --config $SM/online.json --steps 20
stamp "v4c smoke tests passed"

# ---- full: learners from scratch on the mixture, FM; then FM-only actors
mkdir -p $Q/data/wm_v4_online
tmux new-session -d -s qingxin-v4-learner-wm "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.22 nice -n 10 python $S/online/run_online.py wm --config $C/online.json > $Q/logs/v4c_learner_wm.log 2>&1"
tmux new-session -d -s qingxin-v4-learner-success "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/online/run_online.py success --config $C/online.json > $Q/logs/v4c_learner_success.log 2>&1"
stamp "learners started (world model, success classifier; from scratch on the mixture)"
[ -f $Q/runs/policy_v4/done ] || { run policy 0.06 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v4 && touch $Q/runs/policy_v4/done; }
tmux new-session -d -s qingxin-v4-actors "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.03 nice -n 10 python $S/online/run_online.py actors --config $C/online.json --workers 8 > $Q/logs/v4c_actors.log 2>&1"
stamp "actors started (8, FM only; planning off until a consistency check passes)"
