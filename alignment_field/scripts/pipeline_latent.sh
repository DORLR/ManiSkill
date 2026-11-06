#!/bin/bash
# (a) SID field results, then (b) latent field: collect interaction views, fit, evaluate.
cd /root/autodl-tmp/port_align_field
source scripts/env.sh >/dev/null 2>&1
export PATH=/root/autodl-tmp/paf_venv/bin:$PATH
stamp() { echo "[$(date +%H:%M:%S)] $*" >> logs/pipeline_latent.log; }
while [ "$(ps -eo cmd | grep -c "[e]val_v4_sid")" -gt 0 ]; do sleep 30; done
python scripts/summarize_v4.py --eval results/eval_v4_il results/eval_v4_sid --out results/summary_v4_sid.json > results/summary_v4_sid.txt 2>&1
stamp "SID field evaluated"
python scripts/collect_interaction.py --out data/v4_interact --per-port 4 --abcd-per-port 3 --views 64 --workers 11 > logs/collect_interact.log 2>&1
stamp "interaction views collected"
for P in fm bc; do
  D=runs_v4/policy; [ $P = bc ] && D=runs_v4/policy_bc
  python -m paf.latent_field --policy $D --out runs_v4/latent_$P > logs/latent_$P.log 2>&1 || { stamp "STOP: latent_$P failed"; exit 1; }
done
stamp "latent fields trained"
python scripts/eval_v4.py --policies fm=runs_v4/policy bc=runs_v4/policy_bc --fieldrels fm=runs_v4/fieldrel_sid bc=runs_v4/fieldrel_sid \
  --latents fm=runs_v4/latent_fm bc=runs_v4/latent_bc --kinds latent --out results/eval_v4_latent --workers 11 > logs/eval_v4_latent.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v4_il results/eval_v4_sid results/eval_v4_latent --out results/summary_v4_all.json > results/summary_v4_all.txt 2>&1
stamp "latent field evaluated"
