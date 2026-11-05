#!/bin/bash
# Hands-off v4 chain after collect_v4.sh:
#   pack raw RGB-D -> train relational net (GPU) and FM insertion policy (GPU) together
#   -> measure the policy's ID set and fit the field -> closed-loop comparison -> table.
source /root/autodl-tmp/port_align_field/scripts/env.sh
stamp() { echo "[$(date +%T)] $*"; }
while [ "$(ps -eo cmd | grep -c "[c]ollect_v4.sh")" -gt 0 ]; do sleep 30; done
while [ "$(ps -eo cmd | grep -c "[p]retrain_partial_v4.sh")" -gt 0 ]; do sleep 30; done
REL_INIT=""; POL_INIT=""; REL_EP=20; POL_EP=40
[ -f runs_v4/rel_partial/model.pt ] && REL_INIT="--init runs_v4/rel_partial/model.pt" && REL_EP=8
[ -f runs_v4/policy_partial/model.pt ] && POL_INIT="--init runs_v4/policy_partial/model.pt" && POL_EP=15
rm -rf data/packed_v4_partial
stamp "collection finished: $(ls data/v4_train/*.npz | wc -l) + $(ls data/v4_abcd/*.npz | wc -l) shards"
python scripts/pack_raw.py --data "data/v4_train/*.npz" --out data/packed_v4 && \
python scripts/pack_raw.py --data "data/v4_abcd/*.npz" --out data/packed_v4_abcd
stamp "packed"
python -m paf.train_rel --packed data/packed_v4 --val-ports G46 G47 G48 --monitor-packed data/packed_v4_abcd \
  --monitor-every 2 --epochs $REL_EP $REL_INIT --workers 6 --out runs_v4/rel > logs/train_v4_rel.log 2>&1 &
REL=$!
python -m paf.fm_policy --data "data/v4_train/*.npz" --out runs_v4/policy --epochs $POL_EP $POL_INIT \
  > logs/train_v4_policy.log 2>&1
stamp "policy trained"
nice -n 10 python scripts/measure_id.py --policy runs_v4/policy --out runs_v4/field --grasps 60 --starts 12 \
  --workers 10 > logs/measure_id_v4.log 2>&1
stamp "policy ID set measured"
wait $REL
stamp "relational net trained"
mkdir -p runs_v4/fieldrel && cp runs_v4/rel/model.pt runs_v4/fieldrel/model.pt && \
  cp runs_v4/field/field.pt runs_v4/field/id_set.npy runs_v4/fieldrel/
nice -n 10 python scripts/eval_v4.py --policy runs_v4/policy --fieldrel runs_v4/fieldrel --out results/eval_v4 \
  --workers 11 > logs/eval_v4.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v4 --out results/summary_v4.json > results/summary_v4.txt 2>&1
stamp "v4 evaluation finished"
