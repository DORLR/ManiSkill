#!/bin/bash
# Gates 1 and 2 for a crop-autoencoder run trained on the v4e data and crops (2026-11-12; v5_gate.sh generalised).
#   usage: ae_gate.sh <run, e.g. crop_ae_v5b> <tmux session of its training>
# gate 1  at steps 10 000 (recorded), 20 000, 30 000: the tokens of a frame differ (check_latent_tokens.py), the pin tips
#         across the hole axis are read near the hole (check_ae_tips.py, the autoencoder's own decoder, held-out
#         exploration), and the pictures (make_architecture_figures.py). Measured the same day, pins at the face:
#           tip within 1 mm / 3 mm:  blind 0.41 / 1.16   v4e 0.38 / 0.54   a reader of its own on the pixels 0.22 / 0.27
#         PASS: token-specific share >= 10 % and error <= 0.32 / 0.45 (clearly better than v4e, on the way to the bound).
#         The first PASS from 20 000 on goes to gate 2; FAIL at 30 000 stops here.
# gate 2  latents of demonstrations + exploration with that checkpoint -> success classifier from scratch (v4e recipe)
#         -> foresight on held-out push-only starts inside the band 0.3-1.2 mm (v4e: AUC 0.781). Recorded.
# GPU 0. Nothing of this is a final model.
set -uo pipefail
RUN=$1; SESSION=$2
T=${RUN#crop_ae_}
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
AE=$Q/runs/$RUN
CK=$Q/runs/checks_v5
X=$Q/data/wm_v1x
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start ${T}_$name (cap $cap, GPU $CUDA_VISIBLE_DEVICES)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/${T}g_$name.log" 2>&1; then stamp "done ${T}_$name"
    else stamp "FAILED ${T}_$name (logs/${T}g_$name.log) -- $T gate pipeline stopped"; exit 1; fi
}
ae_step() { grep -o '"step": [0-9]*, "val_pose_score"' $AE/log.jsonl 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1; }
mkdir -p $CK
PASSED=""
for K in 10000 20000 30000; do
    while :; do
        s=$(ae_step); [ -n "$s" ] && [ "$s" -ge $K ] && break
        tmux has-session -t $SESSION 2>/dev/null || { stamp "FAILED: the $T autoencoder training ended before step $K"; exit 1; }
        sleep 120
    done
    sleep 60
    G=$CK/ae_${T}_$K.pt; TAG=${T}_$((K / 1000))k
    cp $AE/last.pt $G
    WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/analysis/check_latent_tokens.py --ae $G --data $Q/data/wm_v1 $Q/data/wm_v1x --crops $Q/data/wm_v1_crops $Q/data/wm_v1x_crops --out $CK/latent_tokens_$TAG.json > $Q/logs/gate1_tokens_$TAG.log 2>&1 || { stamp "FAILED: check_latent_tokens ($TAG)"; exit 1; }
    WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/analysis/check_ae_tips.py --ae $G --data $X --crops ${X}_crops --geometry $Q/data/wm_v1_labelgeom --out $CK/ae_tips_$TAG.json > $Q/logs/gate1_tips_$TAG.log 2>&1 || { stamp "FAILED: check_ae_tips ($TAG)"; exit 1; }
    WM_GPU_MEM_FRACTION=0.08 nice -n 10 python $S/analysis/make_architecture_figures.py --unet $Q/runs/unet_v4/best.pt --ae $G --demos $Q/data/wm_v1 --explore $X --out $CK/figures_$TAG > $Q/logs/gate1_figures_$TAG.log 2>&1 || stamp "figures failed ($TAG; logs/gate1_figures_$TAG.log) -- not a gate"
    LINE=$(python - <<PY
import json, os
t, a = json.load(open("$CK/latent_tokens_$TAG.json")), json.load(open("$CK/ae_tips_$TAG.json"))
share = {k: v["token_specific_pct_of_frame_to_frame_variation"] for k, v in t.items() if isinstance(v, dict)}
f1, f3 = a["pins at the face | tip within 1 mm"], a["pins at the face | tip within 3 mm"]
fig = "$CK/figures_$TAG/figures.json"
rec = json.load(open(fig))["reconstruction_errors"] if os.path.exists(fig) else {}
iou = {k: v.get("classes_from_latent_iou") for k, v in rec.items()}
ok = min(share.values()) >= 10.0 and f1["error_mm"] <= 0.32 and f3["error_mm"] <= 0.45
print("gate 1 ($TAG): token-specific share %% %s | pins at the face, tip across the axis mm: within 1 mm %.2f (blind %.2f), within 3 mm %.2f (blind %.2f); corr within 3 mm y %.2f z %.2f | classes from the latent, IoU %s | GATE %s"
      % (share, f1["error_mm"], f1["blind_error_mm"], f3["error_mm"], f3["blind_error_mm"], f3["corr_horizontal"], f3["corr_vertical"], iou, "PASS" if ok else "FAIL"))
PY
)
    stamp "$LINE"
    if echo "$LINE" | grep -q "GATE PASS" && [ $K -ge 20000 ]; then PASSED=$K; break; fi
done
[ -n "$PASSED" ] || { stamp "gate 1 ($T) FAILED up to step 30000 -- stopped (the training goes on; nothing downstream started)"; exit 1; }
G=$CK/ae_${T}_$PASSED.pt
LD=$Q/data/wm_${T}g_latents; LX=$Q/data/wm_${T}gx_latents
stamp "gate 1 ($T) passed at step $PASSED: gate 2 starts (latents with $G, classifier from scratch)"
( run encode_demos 0.10 python $S/train/encode_latents_v4.py --ae $G --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $LD --workers 10 ) &
P1=$!
( run encode_explore 0.06 python $S/train/encode_latents_v4.py --ae $G --data $X --ee ${X}_ee --crops ${X}_crops --out $LX --workers 6 ) &
P2=$!
wait $P1 || exit 1
wait $P2 || exit 1
[ -e $Q/runs/success_${T}g ] && { stamp "FAILED: runs/success_${T}g already exists"; exit 1; }
python - <<PY
import json
c = json.load(open("world_model/configs/v5/success_gate.json")); c["latents"] = ["$LD", "$LX"]
json.dump(c, open("$CK/success_${T}g.json", "w"), indent=1)
PY
run success 0.06 python $S/train/train_success.py --config $CK/success_${T}g.json --run $Q/runs/success_${T}g
run success_check 0.04 python $S/analysis/check_success_on_exploration.py --success $Q/runs/success_${T}g/best.pt --data $X --latents $LX --scale-from $LD --geometry $Q/data/wm_v1_labelgeom --out $CK/success_on_exploration_${T}g.json
python - <<PY >> $LOG
import json, time
o, n = json.load(open("$Q/runs/checks_v4/explore/success_on_exploration.json")), json.load(open("$CK/success_on_exploration_${T}g.json"))
f = lambda r: {k: r[k] for k in ("auc_first_frame", "auc_after_1s", "auc_after_2s")}
print("[%s] gate 2 ($T, autoencoder step $PASSED), classifier foresight on held-out push-only exploration starts: v4e %s | $T %s | by offset $T %s"
      % (time.strftime("%m-%d %H:%M"), f(o), f(n), {k: (v["really_inserted_pct"], v["called_inserted_pct"]) for k, v in n["by_start_tip_offset"].items()}))
PY
stamp "$T gate pipeline finished"
