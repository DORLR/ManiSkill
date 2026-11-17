#!/bin/bash
# The 320 px ResNet policy: training, selection of the checkpoint, final test.
#   policy320.sh <stem> <train_gpu> <test_gpu> <shift_px> <batch> <accum> <train_gpu_mib> <train_ram_gb> <loader_workers> [resume] [seed]
# While the training runs and afterwards, every kept checkpoint (step*.pt) is tested in closed loop on the SELECTION
# split (all its grasps x 4 starts). After a regular end the checkpoint with the best selection success (ties: the
# later one) is tested on all held-out grasps x 2 starts and on 50 training grasps x 2 starts. A training that did
# not end regularly gets no final test. Ports A-D are never touched here.
# Run again on a finished training (DONE.json), it only completes the tests that are missing.
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
stem=${1:?stem}; gpu=${2:?training GPU}; tgpu=${3:?test GPU}; shift_px=${4:?shift px}; batch=${5:?batch}; accum=${6:?accum}
tmib=${7:?GPU MiB of the training}; tram=${8:?memory cap of the training in GB}; wk=${9:?loader workers}; mode=${10:-new}; seed=${11:-0}
[[ "$gpu" =~ ^[0-7]$ && "$tgpu" =~ ^[0-7]$ ]] || { echo "GPUs must be 0-7"; exit 1; }
for v in "$shift_px" "$batch" "$accum" "$tmib" "$tram" "$wk" "$seed"; do q320_int "$v" || { echo "not a whole number: $v"; exit 1; }; done
case "$mode" in new|resume) ;; *) echo "mode must be new or resume"; exit 1 ;; esac
cd "$Q/code/ManiSkill" || exit 1
R=$Q/runs; name=fm320_${stem}_s$seed; O=$R/$name; DATA=$Q/data/wm_v1_fast; GRASPS=$Q/data/wm_v1
TLOG=$Q/logs/${name}_train.log; ELOG=$Q/logs/${name}_tests.log
efrac=$(q320_fraction "$EVAL_TORCH_MIB" "$tgpu") || exit 1; emib=$(( EVAL_WORKERS * (EVAL_TORCH_MIB + EVAL_EXTRA_MIB) ))

closed() {    # <checkpoint> <split> <number of grasps|all> <starts> <name>
  local d=$O/$5 narg=() rc
  [ -e "$d/summary.json" ] && return 0
  [ -e "$d/FAILED" ] && return 1
  [ "$3" = all ] || narg=(--grasps-n "$3")
  q320_run_when_fits "qx320-test-$name-$5" "$EVAL_RAM_GB" "$tgpu" "$emib" test "$efrac" "$ELOG" python world_model/scripts/eval/eval_policy320.py \
    --grasps "$GRASPS" --split "$2" --policy "$O/$1" --out "$d" --starts "$4" --workers "$EVAL_WORKERS" "${narg[@]}"
  rc=$?
  if [ $rc -eq 0 ] && [ -e "$d/summary.json" ]; then
    q320_log "$name $5: $(python -c "import json; a = json.load(open('$d/summary.json'))['groups']['all']; print('success %s %% +- %s on %d starts' % (round(100 * a['success'], 1) if a['success'] is not None else '-', round(100 * a['se'], 1) if a['se'] is not None else '-', a['starts']))")"
    return 0
  fi
  mkdir -p "$d"; echo "$rc" >> "$d/TRIES"
  # codes 0-2: the test itself did not complete; other codes: it was stopped (watchdog) or never fitted: try again, up to 3 times
  if [ "$rc" -le 2 ] || [ "$(wc -l < "$d/TRIES")" -ge 3 ]; then
    touch "$d/FAILED"; q320_log "$name $5: the test did not complete (code $rc); given up (delete $d/FAILED and $d/TRIES to try again)"
  else
    q320_log "$name $5: the test was stopped or did not fit (code $rc); it will be tried again"
  fi
  return 1; }
select_tests() { local f; for f in $(ls "$O"/step*.pt 2>/dev/null | sort -V); do closed "$(basename "$f")" select all 4 "sel_$(basename "$f" .pt)"; done; }
pending() {   # a kept checkpoint whose selection test is neither complete nor given up
  local f d
  for f in "$O"/step*.pt; do
    [ -e "$f" ] || continue
    d=$O/sel_$(basename "$f" .pt)
    [ -e "$d/summary.json" ] || [ -e "$d/FAILED" ] || return 0
  done
  return 1; }

if [ -e "$O/DONE.json" ]; then
  q320_log "$name: the training is finished; completing its tests"
else
  [ -e "$DATA/manifest.json" ] || { q320_log "$name: no fast copy at $DATA (run cache_fast320.sh first)"; exit 1; }
  extra=()
  if [ "$mode" = resume ]; then extra=(--resume)
  elif [ -e "$O/log.jsonl" ]; then q320_log "$name: $O exists; pass 'resume' to continue it"; exit 1; fi
  mkdir -p "$O"; rm -f -- "${O:?}/TRAIN_EXIT"
  tfrac=$(q320_fraction $(( tmib - 600 )) "$gpu") || exit 1
  ( q320_run "qx320-train-$name" "$tram" "$gpu" "$tmib" train "$tfrac" "$TLOG" python world_model/scripts/train/train_policy320.py \
      --data "$DATA" --out "$O" --stem "$stem" --shift-px "$shift_px" --batch "$batch" --accum "$accum" --workers "$wk" --seed "$seed" "${extra[@]}"
    echo $? > "$O/TRAIN_EXIT" ) &
  tp=$!
  while kill -0 "$tp" 2>/dev/null; do sleep 120; select_tests; done      # the training claims its budget first
  wait "$tp"
  rc=$(cat "$O/TRAIN_EXIT" 2>/dev/null || echo 1)
  if [ "$rc" != 0 ] || [ ! -e "$O/DONE.json" ]; then q320_log "$name: the training did not end regularly (code $rc); no final test"; exit 1; fi
fi
select_tests
while pending; do sleep 120; select_tests; done
missing=$(for f in "$O"/step*.pt; do [ -e "$f" ] && [ ! -e "$O/sel_$(basename "$f" .pt)/summary.json" ] && basename "$f"; done | tr '\n' ' ')
[ -z "$missing" ] || { q320_log "$name: no selection result for $missing; no final test until every kept checkpoint is tested"; exit 1; }
best=$(python - "$O" <<'PY'
import glob, json, re, sys
rows = [(json.load(open(f))["groups"]["all"]["success"], int(re.search(r"sel_step(\d+)", f).group(1))) for f in glob.glob(sys.argv[1] + "/sel_step*/summary.json")]
rows = [r for r in rows if r[0] is not None]
print("step%d.pt" % max(rows)[1] if rows else "")
PY
)
[ -n "$best" ] || { q320_log "$name: no selection test completed; no final test"; exit 1; }
echo "$best" > "$O/CHOSEN"
q320_log "$name: chosen on the selection split: $best"
closed "$best" heldout all 2 "final_heldout_${best%.pt}"; a=$?
closed "$best" train 50 2 "final_train_${best%.pt}"; b=$?
if [ $a -eq 0 ] && [ $b -eq 0 ]; then q320_log "$name: pipeline finished"
else q320_log "$name: a final test is missing (see FAILED / TRIES in its folder; delete them and run the pipeline again)"; exit 1; fi
