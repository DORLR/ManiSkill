#!/bin/bash
# v5 plan, gates 1 and 2 (2026-11-12), on intermediate checkpoints of the crop autoencoder v5 (runs/crop_ae_v5, trained
# on the same data and crops as v4e, so that a difference is the autoencoder's):
#   gate 1  at steps 10 000 (recorded only), 20 000 and 30 000: gate1_v5.sh (tokens differ; pin tips read near the hole).
#           The first PASS from 20 000 on goes ahead; FAIL at 30 000 stops here.
#   gate 2  latents of demonstrations + exploration with that checkpoint -> success classifier from scratch (the v4e
#           recipe, 128 tokens) -> can it tell, from the first frame, whether a push-only start will go in, inside the
#           ambiguous band (tip offset 0.3-1.2 mm)? v4e: AUC 0.781. Recorded; the decision is the user's and mine.
# GPU 0 (allowed by the user 2026-11-12; the autoencoder trains on GPU 1). Nothing here is used for the final models:
# the final autoencoder is trained again with the contact set.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v5
AE=$Q/runs/crop_ae_v5
CK=$Q/runs/checks_v5
X=$Q/data/wm_v1x
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap, GPU $CUDA_VISIBLE_DEVICES)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v5g_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v5g_$name.log) -- v5 gate pipeline stopped"; exit 1; fi
}
ae_step() { grep -o '"step": [0-9]*, "val_pose_score"' $AE/log.jsonl 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1; }
mkdir -p $CK
PASSED=""
for K in 10000 20000 30000; do
    while :; do
        s=$(ae_step); [ -n "$s" ] && [ "$s" -ge $K ] && break
        tmux has-session -t qingxin-v5-ae 2>/dev/null || { stamp "FAILED: the autoencoder training ended before step $K"; exit 1; }
        sleep 120
    done
    sleep 60
    cp $AE/last.pt $CK/ae_v5_$K.pt
    if GPU=$CUDA_VISIBLE_DEVICES bash world_model/scripts/pipelines/gate1_v5.sh $CK/ae_v5_$K.pt v5_$((K / 1000))k; then
        [ $K -ge 20000 ] && { PASSED=$K; break; }
    else
        [ $? -eq 2 ] && exit 1
    fi
done
[ -n "$PASSED" ] || { stamp "gate 1 FAILED up to step 30000 -- stopped (the autoencoder training goes on; nothing downstream started)"; exit 1; }
G=$CK/ae_v5_$PASSED.pt
stamp "gate 1 passed at step $PASSED: gate 2 starts (latents with $G, classifier from scratch)"
( run encode_demos 0.10 python $S/train/encode_latents_v4.py --ae $G --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v5g_latents --workers 10 ) &
P1=$!
( run encode_explore 0.06 python $S/train/encode_latents_v4.py --ae $G --data $X --ee ${X}_ee --crops ${X}_crops --out $Q/data/wm_v5gx_latents --workers 6 ) &
P2=$!
wait $P1 || exit 1
wait $P2 || exit 1
[ -e $Q/runs/success_v5g ] && { stamp "FAILED: runs/success_v5g already exists"; exit 1; }
run success 0.06 python $S/train/train_success.py --config $C/success_gate.json --run $Q/runs/success_v5g
run success_check 0.04 python $S/analysis/check_success_on_exploration.py --success $Q/runs/success_v5g/best.pt --data $X --latents $Q/data/wm_v5gx_latents --scale-from $Q/data/wm_v5g_latents --geometry $Q/data/wm_v1_labelgeom --out $CK/success_on_exploration_v5g.json
python - <<PY >> $LOG
import json, time
o, n = json.load(open("$Q/runs/checks_v4/explore/success_on_exploration.json")), json.load(open("$CK/success_on_exploration_v5g.json"))
f = lambda r: {k: r[k] for k in ("auc_first_frame", "auc_after_1s", "auc_after_2s")}
print("[%s] gate 2, classifier foresight on held-out push-only exploration starts: v4e %s | v5 (autoencoder step $PASSED) %s | by offset v5 %s"
      % (time.strftime("%m-%d %H:%M"), f(o), f(n), {k: (v["really_inserted_pct"], v["called_inserted_pct"]) for k, v in n["by_start_tip_offset"].items()}))
PY
stamp "v5 gate pipeline finished"
