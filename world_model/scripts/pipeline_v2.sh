#!/bin/bash
# v2, everything from scratch (docs/worklog.md, 2026-11-07):
#   1. VAE v2 (no KL, + insertion-angle loss), 60k steps
#   2. encode all trajectories with it -> data/wm_v2_latents (new latent scale)
#   3. offline checks on the v2 latents (probe_offset, probe_contact probes)
#   4. gate: the pose head must now read the angles (mean R^2 of roll/pitch/yaw > 0.5; v1: 0.19)
#      and the latent must keep them at least as well as v1 (mean R^2 >= 0.59); else stop here
#   5. CDiT v2 (tracking history + predicted-pose loss), 200k steps, on the v2 latents
#   tmux new-session -d -s qingxin-v2 "bash world_model/scripts/pipeline_v2.sh"
set -euo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v2.log
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }

stamp "VAE v2 training (memory cap 16 %)"
WM_GPU_MEM_FRACTION=0.16 nice -n 10 python world_model/scripts/train_vae.py --config world_model/configs/vae_v2.json \
    --run "$Q/runs/vae_v2" > "$Q/logs/train_vae_v2.log" 2>&1
if grep -q "stop_nonfinite" "$Q/runs/vae_v2/log.jsonl"; then stamp "STOP: non-finite values in VAE v2"; exit 1; fi
stamp "VAE v2 done: $(grep val_ "$Q/runs/vae_v2/log.jsonl" | tail -1 | cut -c1-300)"

stamp "encoding all trajectories -> data/wm_v2_latents"
WM_GPU_MEM_FRACTION=0.1 nice -n 10 python world_model/scripts/encode_latents.py --vae "$Q/runs/vae_v2/best.pt" \
    --data "$Q/data/wm_v1" --out "$Q/data/wm_v2_latents" > "$Q/logs/encode_v2.log" 2>&1
stamp "encoded: $(tail -1 "$Q/logs/encode_v2.log")"

stamp "offline checks on v2 latents"
cd world_model/scripts
WM_GPU_MEM_FRACTION=0.05 python probe_offset.py --latents "$Q/data/wm_v2_latents" --vae "$Q/runs/vae_v2/best.pt" \
    --out "$Q/runs/probe_contact_v2" > "$Q/logs/probe_offset_v2.log" 2>&1
WM_GPU_MEM_FRACTION=0.05 python probe_contact.py --latents "$Q/data/wm_v2_latents" --vae "$Q/runs/vae_v2/best.pt" \
    --wm "" --out "$Q/runs/probe_contact_v2" > "$Q/logs/probe_contact_v2.log" 2>&1
cd "$Q/code/ManiSkill"
verdict=$(python - <<EOF
import json
r = json.load(open("$Q/runs/probe_contact_v2/offset_report.json"))
ang = ("roll_deg", "pitch_deg", "yaw_deg")
head = sum(r["pose_head/linear"][a]["r2"] for a in ang) / 3 if isinstance(r["pose_head/linear"], dict) else -1
lat = sum(r["latent/mlp"][a]["r2"] for a in ang) / 3
print(f"pose-head angle R2 {head:.2f} (v1 0.19), latent angle R2 {lat:.2f} (v1 0.59):",
      "PASS" if head > 0.5 and lat >= 0.59 else "FAIL")
EOF
)
stamp "gate: $verdict"
case "$verdict" in *PASS) ;; *) stamp "STOP: angle check failed; world model not started"; exit 1;; esac

free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[ "$free_mib" -gt 25000 ] || { stamp "STOP: only ${free_mib} MiB free on GPU 1"; exit 1; }
stamp "CDiT v2 training (memory cap 20 %)"
WM_GPU_MEM_FRACTION=0.2 nice -n 10 python world_model/scripts/train_wm.py --config world_model/configs/wm_v2.json \
    --run "$Q/runs/wm_v2" > "$Q/logs/train_wm_v2.log" 2>&1
stamp "CDiT v2 finished"
