#!/bin/bash
# Order requested by the user:
#   1. IL policies (FM, BC) trained on expert demos -> IL-only generalisation test, reported first
#   2. each policy's ID set -> its field (relational net already trained)
#   3. field+IL and oracle+IL on the SAME test grasps and starts (same seeds) -> comparison table
source /root/autodl-tmp/port_align_field/scripts/env.sh
stamp() { echo "[$(date +%T)] $*"; }
while [ "$(ps -eo cmd | grep -c "[c]ollect_expert_demos.py")" -gt 0 ]; do sleep 30; done
nice -n 10 python scripts/collect_expert_demos.py --out data/v4_demos --per-port 6 --val-per-port 4 --n-demo 3 \
  --workers 11 >> logs/collect_v4_demos.log 2>&1   # resumable: only fills gaps
stamp "expert demos: $(ls data/v4_demos/*.npz | wc -l) grasps"
rm -rf runs_v4/policy runs_v4/policy_bc runs_v4/field_fm runs_v4/field_bc runs_v4/fieldrel_fm runs_v4/fieldrel_bc \
  results/eval_v4 results/eval_v4_il
python -m paf.fm_policy --data "data/v4_demos/*.npz" --head bc --out runs_v4/policy_bc --epochs 40 > logs/train_v4_policy_bc.log 2>&1 &
python -m paf.fm_policy --data "data/v4_demos/*.npz" --head fm --out runs_v4/policy --epochs 40 > logs/train_v4_policy.log 2>&1
wait
stamp "IL policies trained"
mkdir -p runs_v4/fr_dummy
nice -n 10 python scripts/eval_v4.py --policies fm=runs_v4/policy bc=runs_v4/policy_bc \
  --fieldrels fm=runs_v4/fr_dummy bc=runs_v4/fr_dummy --kinds id alone --out results/eval_v4_il --workers 11 \
  > logs/eval_v4_il.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v4_il --out results/summary_v4_il.json > results/summary_v4_il.txt 2>&1
stamp "IL-only generalisation evaluated"
for P in fm bc; do
  DIR=runs_v4/policy; [ $P = bc ] && DIR=runs_v4/policy_bc
  nice -n 10 python scripts/measure_id.py --policy $DIR --out runs_v4/field_$P --grasps 60 --starts 12 \
    --workers 11 > logs/measure_id_v4_$P.log 2>&1
  stamp "ID set of $P: $(grep 'policy trials' logs/measure_id_v4_$P.log | cut -c1-140)"
  mkdir -p runs_v4/fieldrel_$P && cp runs_v4/rel/model.pt runs_v4/fieldrel_$P/model.pt
  [ -f runs_v4/field_$P/field.pt ] && cp runs_v4/field_$P/field.pt runs_v4/field_$P/id_set.npy runs_v4/fieldrel_$P/
done
if [ -f runs_v4/fieldrel_fm/field.pt ] && [ -f runs_v4/fieldrel_bc/field.pt ]; then
  nice -n 10 python scripts/eval_v4.py --policies fm=runs_v4/policy bc=runs_v4/policy_bc \
    --fieldrels fm=runs_v4/fieldrel_fm bc=runs_v4/fieldrel_bc --kinds field oracle --out results/eval_v4 \
    --workers 11 > logs/eval_v4.log 2>&1
  python scripts/summarize_v4.py --eval results/eval_v4_il results/eval_v4 --out results/summary_v4.json \
    > results/summary_v4.txt 2>&1
  stamp "v4 evaluation finished"
else
  stamp "STOP: at least one policy is too weak to define an ID set; field comparison not run"
fi
