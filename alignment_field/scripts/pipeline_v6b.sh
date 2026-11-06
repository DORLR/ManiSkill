#!/bin/bash
# v6 continued after the 62 GB container memory limit stalled pipeline_v6.sh (two IL
# trainings held ~64 GB). Demos are done; the RelNet view collection keeps running.
# IL now loads every 2nd frame into preallocated arrays without a pinned copy (~16 GB each).
cd /root/autodl-tmp/port_align_field
source scripts/env.sh >/dev/null 2>&1
export PATH=/root/autodl-tmp/paf_venv/bin:$PATH
stamp() { echo "[$(date +%H:%M:%S)] $*" >> logs/pipeline_v6.log; }
stamp "restart (v6b, chunked IL validation): memory limit $(($(cat /sys/fs/cgroup/memory.max)/1000000000)) GB"
python -m paf.fm_policy --data "data/v6_demos/*.npz" --head fm --out runs_v6/policy --epochs 40 --frame-stride 2 > logs/train_v6_policy.log 2>&1 &
PF=$!
python -m paf.fm_policy --data "data/v6_demos/*.npz" --head bc --out runs_v6/policy_bc --epochs 40 --frame-stride 2 > logs/train_v6_policy_bc.log 2>&1 &
PB=$!
while ps -eo cmd | grep -q "[c]ollect.py --out data/v6_views"; do sleep 30; done
python scripts/collect.py --out data/v6_views --ports $(python -c "from paf.ports_ext import generated_specs; print(' '.join(p for p in generated_specs() if p not in ('G19','G42','G45','G49')))" 2>/dev/null) \
  --per-port 6 --seed0 1300000 --store-raw --mix handoff=30,approach=50,wide_v6=20 --workers 10 >> logs/collect_v6_views.log 2>&1   # resumes missing shards only
python scripts/collect.py --out data/v6_abcd --ports A B C D --per-port 8 --seed0 1400000 --store-raw \
  --mix handoff=30,approach=50,wide_v6=20 --workers 10 > logs/collect_v6_abcd.log 2>&1
stamp "views collected: $(ls data/v6_views/*.npz | wc -l) + $(ls data/v6_abcd/*.npz | wc -l) grasps"
python scripts/pack_raw.py --data "data/v6_views/*.npz" --out data/packed_v6 > logs/pack_v6.log 2>&1 || { stamp "STOP: pack failed"; exit 1; }
python scripts/pack_raw.py --data "data/v6_abcd/*.npz" --out data/packed_v6_abcd >> logs/pack_v6.log 2>&1 || { stamp "STOP: pack abcd failed"; exit 1; }
stamp "packed: $(grep packed logs/pack_v6.log | paste -sd' ')"
python -m paf.train_rel --packed data/packed_v6 --val-ports G46 G47 G48 --monitor-packed data/packed_v6_abcd \
  --epochs 10 --workers 4 --out runs_v6/rel > logs/train_v6_rel.log 2>&1 || { stamp "STOP: RelNet training failed"; exit 1; }
stamp "RelNet trained"
wait $PF $PB
stamp "IL trained"
[ -f runs_v6/policy/summary.json ] && [ -f runs_v6/policy_bc/summary.json ] || { stamp "STOP: IL training failed"; exit 1; }
python scripts/fit_sid_field.py --handoff-box --rel runs_v6/rel --out runs_v6/fieldrel_handoff > logs/field_v6.log 2>&1 || { stamp "STOP: field failed"; exit 1; }
stamp "field fitted"
python scripts/eval_v4.py --policies fm=runs_v6/policy bc=runs_v6/policy_bc --fieldrels fm=runs_v6/fieldrel_handoff bc=runs_v6/fieldrel_handoff \
  --kinds id alone field oracle --out results/eval_v6 --workers 10 > logs/eval_v6.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v6 --out results/summary_v6.json > results/summary_v6.txt 2>&1
stamp "v6 evaluation finished"
