#!/bin/bash
# NOTE: ran at commit 9c662f0 (v3 pose regressor; stopped at the latent gate). The v3 configs now point
# to the v3g runs; only its baseline branch (training + evaluation) is still in use. See pipeline_v3g.sh.
# v3, everything from scratch (docs/design_v3.md). GPU 1 only; memory caps per process
# (WM_GPU_MEM_FRACTION of 98 GB) keep our total under 55 GB:
#
#   phase A, in parallel
#     commanded hand poses (CPU)            data/wm_v1_ee
#     joint adapter              cap 0.02   runs/adapter_v3
#     autoencoder v3             cap 0.08   runs/vae_v3          (~4.5 h)
#     latent-free baseline       cap 0.07   runs/baseline_v3     (~5 days; same budget as the latent policy)
#        -> closed-loop evaluation of the baseline when it finishes
#   phase B, after the autoencoder
#     encode all trajectories               data/wm_v3_latents
#     latent checks + gate: the pose head must read the angles at least as well as v2 (mean R^2 of
#     roll/pitch/yaw >= 0.54), the latent must keep them (>= 0.70) and jam-vs-insert must stay
#     separable from the latent (AUC >= 0.84); otherwise stop before the world models
#   phase C, in parallel
#     world model CDiT-S         cap 0.17   runs/wm_v3_s   -> world-model checks
#     world model CDiT-B         cap 0.18   runs/wm_v3_b   (batch 128 x 2 accumulation = 256) -> checks
#     latent FM policy           cap 0.04   runs/policy_v3 -> closed-loop evaluation (3 workers)
#
#   tmux new-session -d -s qingxin-v3 "bash world_model/scripts/pipelines/pipeline_v3.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v3.log
S=world_model/scripts
C=world_model/configs/v3
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {  # run <name> <cap> <command...>: logs to $Q/logs/<name>.log, stamps start / end / failure
    local name=$1 cap=$2; shift 2
    stamp "start $name (memory cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/$name.log" 2>&1; then stamp "done $name"; return 0
    else stamp "FAILED $name (see logs/$name.log)"; return 1; fi
}

# ---------------------------------------------------------------- phase A
( run compute_ee 0 python $S/data/compute_ee.py $Q/data/wm_v1 --out $Q/data/wm_v1_ee --workers 6 ) &
P_EE=$!
( run train_adapter_v3 0.02 python $S/train/train_adapter.py --config $C/adapter.json --run $Q/runs/adapter_v3 ) &
P_AD=$!
( run train_vae_v3 0.08 python $S/train/train_vae.py --config $C/vae.json --run $Q/runs/vae_v3 ) &
P_VAE=$!
( run train_baseline_v3 0.07 python $S/train/train_policy.py --config $C/baseline.json --run $Q/runs/baseline_v3 \
  && run eval_baseline_v3 0.03 python $S/eval/eval_policy.py --policy $Q/runs/baseline_v3/best.pt \
         --split ood_val --out $Q/runs/eval_baseline_v3_ood --workers 3 ) &
P_BASE=$!

wait $P_EE || { stamp "STOP: commanded hand poses failed"; exit 1; }
wait $P_VAE || { stamp "STOP: autoencoder failed"; exit 1; }

# ---------------------------------------------------------------- phase B
run encode_v3 0.06 python $S/train/encode_latents.py --vae $Q/runs/vae_v3/best.pt --data $Q/data/wm_v1 \
    --ee $Q/data/wm_v1_ee --out $Q/data/wm_v3_latents || { stamp "STOP: encoding failed"; exit 1; }
run check_latents_v3 0.04 python $S/analysis/check_latents.py --latents $Q/data/wm_v3_latents \
    --vae $Q/runs/vae_v3/best.pt --out $Q/runs/checks_v3 || { stamp "STOP: latent checks failed"; exit 1; }
verdict=$(python - <<EOF
import json
r = json.load(open("$Q/runs/checks_v3/latents.json"))
g, auc = r["angle_gate"], r["jam_probes"]["push_start/latent/mlp"]["auc"]
ok = g["pose_head_angle_r2"] >= 0.54 and g["latent_angle_r2"] >= 0.70 and auc >= 0.84
print(f"pose-head angle R2 {g['pose_head_angle_r2']} (v2 0.54), latent angle R2 {g['latent_angle_r2']} (v2 0.78),",
      f"jam AUC {auc} (v2 0.86):", "PASS" if ok else "FAIL")
EOF
)
stamp "gate: $verdict"
case "$verdict" in *PASS) ;; *) stamp "STOP: latent gate failed; world models not started"; wait; exit 1;; esac

# ---------------------------------------------------------------- phase C
wait $P_AD || stamp "WARNING: adapter failed; world-model rollout checks will fail"
( run train_wm_v3_s 0.17 python $S/train/train_wm.py --config $C/wm_s.json --run $Q/runs/wm_v3_s \
  && run check_wm_v3_s 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3_latents \
         --vae $Q/runs/vae_v3/best.pt --wm $Q/runs/wm_v3_s/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3 --tag s ) &
( run train_wm_v3_b 0.18 python $S/train/train_wm.py --config $C/wm_b.json --run $Q/runs/wm_v3_b \
  && run check_wm_v3_b 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3_latents \
         --vae $Q/runs/vae_v3/best.pt --wm $Q/runs/wm_v3_b/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3 --tag b ) &
( run train_policy_v3 0.04 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v3 \
  && run eval_policy_v3 0.03 python $S/eval/eval_policy.py --vae $Q/runs/vae_v3/best.pt \
         --policy $Q/runs/policy_v3/best.pt --split ood_val --out $Q/runs/eval_policy_v3_ood --workers 3 ) &
wait
stamp "pipeline v3 finished"
