#!/bin/bash
# v4 with demonstrations from Chongyu's expert (replaces the scripted demonstrator):
#   expert demos -> FM and BC IL policies (RGB crop + joint angles -> joint-target chunks)
#   -> each policy's ID set and field (relational net from runs_v4/rel, already trained)
#   -> closed loop on unseen interfaces and grasps (IL alone, field+IL, oracle+IL, IL from ID starts)
source /root/autodl-tmp/port_align_field/scripts/env.sh
stamp() { echo "[$(date +%T)] $*"; }
nice -n 10 python scripts/collect_expert_demos.py --out data/v4_demos --per-port 6 --val-per-port 4 --n-demo 3 \
  --workers 11 > logs/collect_v4_demos.log 2>&1
stamp "expert demos: $(ls data/v4_demos/*.npz | wc -l) grasps"
rm -rf runs_v4/policy runs_v4/policy_bc runs_v4/field_fm runs_v4/field_bc runs_v4/fieldrel_fm runs_v4/fieldrel_bc
python -m paf.fm_policy --data "data/v4_demos/*.npz" --head bc --out runs_v4/policy_bc --epochs 40 > logs/train_v4_policy_bc.log 2>&1 &
python -m paf.fm_policy --data "data/v4_demos/*.npz" --head fm --out runs_v4/policy --epochs 40 > logs/train_v4_policy.log 2>&1
wait
stamp "IL policies trained"
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
    --fieldrels fm=runs_v4/fieldrel_fm bc=runs_v4/fieldrel_bc --out results/eval_v4 --workers 11 > logs/eval_v4.log 2>&1
  python scripts/summarize_v4.py --eval results/eval_v4 --out results/summary_v4.json > results/summary_v4.txt 2>&1
  stamp "v4 evaluation finished"
else
  stamp "STOP: at least one policy is too weak to define an ID set; closed loop not run"
fi
