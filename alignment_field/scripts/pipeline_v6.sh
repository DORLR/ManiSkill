#!/bin/bash
# v6: wrist camera = D405 default stream (640x480, 87 deg HFOV, 7 cm minimum depth), Chongyu's mount.
# Everything image-based is redone: expert demos -> IL (fm, bc); RelNet views -> RelNet;
# field to the hand-over region; evaluation of IL alone, IL from the hand-over region, field+IL, oracle+IL.
cd /root/autodl-tmp/port_align_field
source scripts/env.sh >/dev/null 2>&1
export PATH=/root/autodl-tmp/paf_venv/bin:$PATH
stamp() { echo "[$(date +%H:%M:%S)] $*" >> logs/pipeline_v6.log; }
rm -rf data/packed_v5   # old-camera pack (regenerable from data/v4_train + data/v5_handoff)
stamp "start; disk free $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
python scripts/collect_expert_demos.py --out data/v6_demos --per-port 6 --val-per-port 4 --n-demo 3 --workers 11 > logs/collect_v6_demos.log 2>&1
stamp "demos: $(grep -c "'ok': [1-9]" logs/collect_v6_demos.log) grasps with >=1 demo"
python -m paf.fm_policy --data "data/v6_demos/*.npz" --head fm --out runs_v6/policy --epochs 40 > logs/train_v6_policy.log 2>&1 &
PF=$!
python -m paf.fm_policy --data "data/v6_demos/*.npz" --head bc --out runs_v6/policy_bc --epochs 40 > logs/train_v6_policy_bc.log 2>&1 &
PB=$!
PORTS=$(python -c "from paf.ports_ext import generated_specs; print(' '.join(p for p in generated_specs() if p not in ('G19','G42','G45','G49')))" 2>/dev/null)
python scripts/collect.py --out data/v6_views --ports $PORTS --per-port 6 --seed0 1300000 --store-raw \
  --mix handoff=30,approach=50,wide_v6=20 --workers 10 > logs/collect_v6_views.log 2>&1
python scripts/collect.py --out data/v6_abcd --ports A B C D --per-port 8 --seed0 1400000 --store-raw \
  --mix handoff=30,approach=50,wide_v6=20 --workers 10 > logs/collect_v6_abcd.log 2>&1
stamp "views collected: $(ls data/v6_views/*.npz | wc -l) + $(ls data/v6_abcd/*.npz | wc -l) grasps"
python scripts/pack_raw.py --data "data/v6_views/*.npz" --out data/packed_v6 > logs/pack_v6.log 2>&1 || { stamp "STOP: pack failed"; exit 1; }
python scripts/pack_raw.py --data "data/v6_abcd/*.npz" --out data/packed_v6_abcd >> logs/pack_v6.log 2>&1 || { stamp "STOP: pack abcd failed"; exit 1; }
stamp "packed: $(grep packed logs/pack_v6.log | paste -sd' ')"
python -m paf.train_rel --packed data/packed_v6 --val-ports G46 G47 G48 --monitor-packed data/packed_v6_abcd \
  --epochs 10 --out runs_v6/rel > logs/train_v6_rel.log 2>&1 || { stamp "STOP: RelNet training failed"; exit 1; }
stamp "RelNet trained"
wait $PF $PB
stamp "IL trained: fm $(tail -1 logs/train_v6_policy.log | cut -c1-120) | bc $(tail -1 logs/train_v6_policy_bc.log | cut -c1-120)"
[ -f runs_v6/policy/summary.json ] && [ -f runs_v6/policy_bc/summary.json ] || { stamp "STOP: IL training failed"; exit 1; }
python scripts/fit_sid_field.py --handoff-box --rel runs_v6/rel --out runs_v6/fieldrel_handoff > logs/field_v6.log 2>&1 || { stamp "STOP: field failed"; exit 1; }
stamp "field fitted"
python scripts/eval_v4.py --policies fm=runs_v6/policy bc=runs_v6/policy_bc --fieldrels fm=runs_v6/fieldrel_handoff bc=runs_v6/fieldrel_handoff \
  --kinds id alone field oracle --out results/eval_v6 --workers 11 > logs/eval_v6.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v6 --out results/summary_v6.json > results/summary_v6.txt 2>&1
stamp "v6 evaluation finished"
