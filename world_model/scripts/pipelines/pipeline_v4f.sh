#!/bin/bash
# v4f (2026-11-11 11:00, decided with the user): continue after the crop autoencoder v4e whatever the old
# readout gate of pipeline_v4e says. That gate measured the offset at the hand, which mixes in the tilt error
# through the 4 cm lever arm to the pins; measured at the pin tips the readout resolves ~0.5 mm (correlation with
# the truth 0.93 / 0.81) and reads roll and yaw but not pitch (docs/2026-11-10_v4_runs.md). The real test of the
# perception is the success classifier's foresight on held-out exploration starts, run here after it is trained.
#   wait for pipeline_v4e -> (if it did not start the trainings) tip-level readout report, latents, smoke tests,
#   FM + world model + success classifier from scratch -> when the classifier is trained: foresight check.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
X=$Q/data/wm_v1x
CK=$Q/runs/checks_v4/explore
SM=$Q/cache/smoke_v4f
AE=$Q/runs/crop_ae_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4f_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4f_$name.log) -- pipeline stopped"; exit 1; fi
}
while tmux has-session -t qingxin-v4e 2>/dev/null; do sleep 60; done
[ -f $AE/done ] || { stamp "FAILED: crop autoencoder v4e did not finish -- v4f not started"; exit 1; }
run ae_tips 0.06 python $S/analysis/check_ae_tips.py --ae $AE/best.pt --data $X --crops ${X}_crops --geometry $Q/data/wm_v1_labelgeom --out $CK/ae_tips_v4e.json
stamp "autoencoder v4e at the pin tips (held-out exploration): $(grep 'pins at the face  tip within 3 mm' $Q/logs/v4f_ae_tips.log | cut -c36-330)"
if ! grep -q "training started: world model (runs/wm_v4e)" $LOG; then
    stamp "continuing past the old readout gate (user decision 11:00)"
    rm -rf $SM; mkdir -p $SM
    ( run encode_demos 0.10 python $S/train/encode_latents_v4.py --ae $AE/best.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v4e_latents --workers 12 ) &
    P1=$!
    ( run encode_explore 0.06 python $S/train/encode_latents_v4.py --ae $AE/best.pt --data $X --ee ${X}_ee --crops ${X}_crops --out $Q/data/wm_v4ex_latents --workers 8 ) &
    P2=$!
    wait $P1 || exit 1
    wait $P2 || exit 1
    python - <<PY >> $LOG
import glob, json, numpy as np
for name in ("wm_v4e_latents", "wm_v4ex_latents"):
    fs = glob.glob(f"$Q/data/{name}/**/*.npz", recursive=True)
    bad = sum(int(not np.isfinite(np.load(f)["mean"].astype(np.float32)).all()) for f in fs[::25])
    print("[latents %s] files %d, scale %.4f, sampled files with non-finite latents: %d" % (name, len(fs), json.load(open(f"$Q/data/{name}/stats.json"))["scale"], bad))
PY
    [ "$(tail -2 $LOG | grep -c 'non-finite latents: 0')" = "2" ] || { stamp "FAILED: latents"; exit 1; }
    python - <<PY
import json
for f, extra in (("wm", dict(workers=2, batch_size=8, accum=1, eval_batches=1, eval_sampling_steps=5, eval_every=20, save_every=10)),
                 ("success", dict(workers=2, batch_size=16, eval_batches=2, eval_every=20, save_every=10)),
                 ("policy", dict(workers=2, batch_size=16, eval_batches=2, eval_every=20))):
    c = json.load(open(f"$C/{f}.json")); c.update(extra); json.dump(c, open(f"$SM/{f}.json", "w"), indent=1)
PY
    run smoke_wm 0.10 python $S/train/train_wm4.py --config $SM/wm.json --run $SM/wm --steps 20
    run smoke_success 0.05 python $S/train/train_success.py --config $SM/success.json --run $SM/success --steps 20
    run smoke_policy 0.05 python $S/train/train_policy.py --config $SM/policy.json --run $SM/policy --max-steps 20
    rm -rf $SM
    for r in wm_v4e success_v4e policy_v4e; do [ -e $Q/runs/$r ] && { stamp "FAILED: $Q/runs/$r already exists"; exit 1; }; done
    tmux new-session -d -s qingxin-v4e-wm "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.26 nice -n 10 python $S/train/train_wm4.py --config $C/wm.json --run $Q/runs/wm_v4e > $Q/logs/v4e_wm.log 2>&1"
    tmux new-session -d -s qingxin-v4e-success "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/train/train_success.py --config $C/success.json --run $Q/runs/success_v4e > $Q/logs/v4e_success.log 2>&1"
    tmux new-session -d -s qingxin-v4e-policy "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v4e > $Q/logs/v4e_policy.log 2>&1"
    stamp "training started: world model (runs/wm_v4e), success classifier (runs/success_v4e), FM (runs/policy_v4e); latents of crop_ae_v4e; demonstrations + exploration; from scratch"
fi
sleep 120
while tmux has-session -t qingxin-v4e-success 2>/dev/null; do sleep 120; done
[ -f $Q/runs/success_v4e/best.pt ] || { stamp "FAILED: success classifier has no checkpoint"; exit 1; }
run success_foresight 0.04 python $S/analysis/check_success_on_exploration.py --success $Q/runs/success_v4e/best.pt --data $X --latents $Q/data/wm_v4ex_latents --scale-from $Q/data/wm_v4e_latents --geometry $Q/data/wm_v1_labelgeom --out $CK/success_on_exploration.json
stamp "classifier foresight on held-out exploration starts: $(python -c "
import json; r = json.load(open('$CK/success_on_exploration.json'))
print('AUC first frame', r['auc_first_frame'], '| after 2 s', r['auc_after_2s'], '|', {k: (v['really_inserted_pct'], v['called_inserted_pct']) for k, v in r['by_start_tip_offset'].items()}, '| time to insert', r['time_to_insert_s'])")"
