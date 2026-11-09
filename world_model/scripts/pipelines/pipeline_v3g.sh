#!/bin/bash
# v3g: v3 with the geometric pose readout (per-pixel motion + weighted rigid fit) instead of v3's pose
# regressor, which failed the latent gate on pitch (docs/2026-11-08_v3_runs.md). From scratch.
# The joint adapter (runs/adapter_v3), the commanded hand poses (data/wm_v1_ee) and the latent-free
# baseline (still training in pipeline_v3.sh) are reused unchanged: they do not depend on this change.
#
#   autoencoder v3g           cap 0.08   runs/vae_v3g
#   encode                               data/wm_v3g_latents
#   latent checks + gate: pose-head pitch R^2 >= 0.5 (v3: -1.98), mean angle R^2 >= 0.54, latent angles
#   >= 0.70, jam AUC >= 0.84; else stop before the world models
#   in parallel: CDiT-S 128x2 (cap 0.19), CDiT-B 64x4 (cap 0.16), latent FM (cap 0.07); effective batch 256 for both -> their checks / evaluation
#
#   tmux new-session -d -s qingxin-v3g "bash world_model/scripts/pipelines/pipeline_v3g.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v3g.log
S=world_model/scripts
C=world_model/configs/v3
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (memory cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/$name.log" 2>&1; then stamp "done $name"; return 0
    else stamp "FAILED $name (see logs/$name.log)"; return 1; fi
}

# finished stages are skipped when the pipeline is restarted (their outputs exist)
[ -f $Q/runs/vae_v3g/best.pt ] && grep -q "early_stop_harm\|\"step\": 60000" $Q/runs/vae_v3g/log.jsonl && stamp "skip train_vae_v3g (done)" || \
run train_vae_v3g 0.08 python $S/train/train_vae.py --config $C/vae.json --run $Q/runs/vae_v3g || { stamp "STOP"; exit 1; }
[ -f $Q/data/wm_v3g_latents/stats.json ] && stamp "skip encode_v3g (done)" || \
run encode_v3g 0.06 python $S/train/encode_latents.py --vae $Q/runs/vae_v3g/best.pt --data $Q/data/wm_v1 \
    --ee $Q/data/wm_v1_ee --out $Q/data/wm_v3g_latents || { stamp "STOP"; exit 1; }
run check_latents_v3g 0.06 python $S/analysis/check_latents.py --latents $Q/data/wm_v3g_latents \
    --vae $Q/runs/vae_v3g/best.pt --out $Q/runs/checks_v3g || { stamp "STOP"; exit 1; }
verdict=$(python - <<PY
import json
r = json.load(open("$Q/runs/checks_v3g/latents.json"))
g, auc = r["angle_gate"], r["jam_probes"]["push_start/latent/mlp"]["auc"]
ok = g["pose_head_pitch_r2"] >= 0.5 and g["pose_head_angle_r2"] >= 0.54 and g["latent_angle_r2"] >= 0.70 and auc >= 0.84
print(f"pose-head pitch R2 {g['pose_head_pitch_r2']} (v3 -1.98), pose-head angle R2 {g['pose_head_angle_r2']} (v2 0.54),",
      f"latent angle R2 {g['latent_angle_r2']}, jam AUC {auc}:", "PASS" if ok else "FAIL")
PY
)
stamp "gate: $verdict"
case "$verdict" in *PASS) ;; *) stamp "STOP: latent gate failed; world models not started"; exit 1;; esac

( run train_wm_v3g_s 0.19 python $S/train/train_wm.py --config $C/wm_s.json --run $Q/runs/wm_v3g_s \
  && run check_wm_v3g_s 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3g_latents \
         --vae $Q/runs/vae_v3g/best.pt --wm $Q/runs/wm_v3g_s/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3g --tag s ) &
( run train_wm_v3g_b 0.16 python $S/train/train_wm.py --config $C/wm_b.json --run $Q/runs/wm_v3g_b \
  && run check_wm_v3g_b 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3g_latents \
         --vae $Q/runs/vae_v3g/best.pt --wm $Q/runs/wm_v3g_b/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3g --tag b ) &
( run train_policy_v3g 0.07 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v3g \
  && run eval_policy_v3g 0.03 python $S/eval/eval_policy.py --vae $Q/runs/vae_v3g/best.pt \
         --policy $Q/runs/policy_v3g/best.pt --split ood_val --out $Q/runs/eval_policy_v3g_ood --workers 3 ) &
wait
stamp "pipeline v3g finished"
