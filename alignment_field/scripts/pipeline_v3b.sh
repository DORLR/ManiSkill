#!/bin/bash
# Replaces the tail of pipeline_v3.sh + pipeline_noise.sh (2026-11-05): v3's
# validation shapes (G46-G48) favour the point-cloud head (0.47 mm) over the
# RGB-refined estimate (0.63 mm), so v3 is evaluated with BOTH estimates.
#   1. wait for the running RGB-only evaluation, summarise it
#   2. v3 closed loop on A-D: field, field_pc, oracle, none
#   3. v3 with strong depth noise: field, field_pc
source /root/autodl-tmp/port_align_field/scripts/env.sh
stamp() { echo "[$(date +%T)] $*"; }
while [ "$(ps -eo cmd | grep -c "[s]cripts/evaluate.py --runs runs --rgb-runs")" -gt 0 ]; do sleep 30; done
python scripts/analyze.py --eval results/eval_rgb --runs runs_rgb --out results/summary_rgb.json --n-perm 100 \
  >> logs/eval_rgb.log 2>&1
stamp "RGB-only evaluation summarised"
nice -n 10 python scripts/evaluate.py --single-run runs_v3/main --out results/eval_v3 \
  --methods field field_pc oracle none --n-unseen 8 --workers 15 > logs/eval_v3.log 2>&1
python scripts/analyze.py --eval results/eval_v3 --single-run runs_v3/main --out results/summary_v3.json \
  --n-perm 100 >> logs/eval_v3.log 2>&1
stamp "v3 evaluation finished"
nice -n 10 python scripts/evaluate.py --single-run runs_v3/main --out results/eval_v3_noise --methods field field_pc \
  --depth-noise strong --n-unseen 8 --workers 15 > logs/eval_v3_noise.log 2>&1
python scripts/analyze.py --eval results/eval_v3_noise --single-run runs_v3/main \
  --out results/summary_v3_noise.json --n-perm 100 >> logs/eval_v3_noise.log 2>&1
stamp "v3 noise evaluation finished"
