#!/bin/bash
# Gate 1 of the v5 plan (2026-11-12): does a crop-autoencoder checkpoint keep the detail the v4e latent lost?
#   usage: gate1_v5.sh <checkpoint> <tag>          e.g. gate1_v5.sh $Q/runs/crop_ae_v5/last.pt v5_10k
# (1) the tokens of a frame must differ (v4e: token-specific part of the frame-to-frame variation under 1 %)
# (2) the pin tips across the hole axis, read by the autoencoder's own decoder on held-out exploration frames with the
#     pins at the board face and the tip within 1 mm of the axis -- where insert / jam is decided -- must be read
#     clearly better than by an image-blind reader (v4e: error 0.38 mm, blind 0.41 mm)
# PASS: (1) >= 10 % on both sources and (2) error <= 0.6 x blind. One line is appended to the pipeline log.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
CKPT=$1; TAG=$2
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts/analysis
CK=$Q/runs/checks_v5
mkdir -p $CK
WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/check_latent_tokens.py --ae $CKPT --data $Q/data/wm_v1 $Q/data/wm_v1x --crops $Q/data/wm_v1_crops $Q/data/wm_v1x_crops --out $CK/latent_tokens_$TAG.json > $Q/logs/gate1_tokens_$TAG.log 2>&1 || { echo "[$(date +%m-%d\ %H:%M)] gate 1 ($TAG): FAILED to run check_latent_tokens" >> $LOG; exit 2; }
WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/check_ae_tips.py --ae $CKPT --data $Q/data/wm_v1x --crops $Q/data/wm_v1x_crops --geometry $Q/data/wm_v1_labelgeom --out $CK/ae_tips_$TAG.json > $Q/logs/gate1_tips_$TAG.log 2>&1 || { echo "[$(date +%m-%d\ %H:%M)] gate 1 ($TAG): FAILED to run check_ae_tips" >> $LOG; exit 2; }
python - <<PY >> $LOG
import json, time, torch
t, a = json.load(open("$CK/latent_tokens_$TAG.json")), json.load(open("$CK/ae_tips_$TAG.json"))
step = torch.load("$CKPT", map_location="cpu", weights_only=False).get("step")
share = {k: v["token_specific_pct_of_frame_to_frame_variation"] for k, v in t.items() if isinstance(v, dict)}
comp = {k: v["components_for_99pct"] for k, v in t.items() if isinstance(v, dict)}
rows = {k: (v["error_mm"], v["blind_error_mm"], v["n"]) for k, v in a.items() if isinstance(v, dict) and "tip within" in k}
face = rows.get("pins at the face | tip within 1 mm")
ok = min(share.values()) >= 10.0 and face is not None and face[0] <= 0.6 * face[1]
print("[%s] gate 1 ($TAG, step %s): token-specific share %% %s, components for 99 %% %s | tip across the axis, error mm (blind, n): %s | GATE %s"
      % (time.strftime("%m-%d %H:%M"), step, share, comp, {k: "%.2f (%.2f, %d)" % v for k, v in rows.items()}, "PASS" if ok else "FAIL"))
PY
tail -1 $LOG
tail -1 $LOG | grep -q "GATE PASS"
