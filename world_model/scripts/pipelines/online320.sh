#!/bin/bash
# Online rounds of the 320 px world model, next to its learner (wm320.sh ... online: train_wm320.py --online <pool>).
#   online320.sh <run> <gpus> <workers> <torch_mib per worker> <gpu_mib per GPU> <ram_gb> <last round> <min_step> [starts]
# <gpus>: the collectors' GPUs; the workers are spread evenly over them; <gpu_mib> must cover ceil(workers / GPUs) x
# (<torch_mib> + 1500 for context and renderer). Beside a learner that declares 18,500 MiB of a 24 GB A5000, one worker
# per GPU fits (<gpu_mib> 3500: free minus the learner's unused declaration must exceed it by the budget's 2,000 margin).
# Start it only after the learner's unit qx320-train-<run> runs (not while the learner waits in the budget queue).
# Round R = 1..<last round>; finished parts of a round are skipped, so a restart resumes (the selection tests of
# collected rounds run whatever the learner's state). Before collecting round R, wait until the learner
#   - R = 1: reads this pool (its newest start/resume record has --online <pool>) and has logged step >= <min_step> + 500
#     (e.g. the end of the value warm-up; +500: last.pt, written every 500 steps, is then past <min_step>);
#   - R > 1: has read round R-1 (the first data record of the newest unbroken run of data records that list it) and
#     logged 1000 more steps (so that last.pt holds at least 500 steps on it).
# Every such wait checks every 5 minutes: the learner finished (DONE.json) -> the rounds end (exit 0); its unit not
# running at two checks in a row -> the rounds end (exit 1; a non-finite stop also ends the unit); the wait breaks only
# while the unit runs. Then the learner's newest last.pt is copied to <pool>/r<R>/model.pt; the back-off threshold of
# that very checkpoint is computed (calibrate_vlow320.py: midway between the medians of the mean value before the
# board face of selection pushes that jammed and that went in) into r<R>/v_low.json (a file without
# "value": "mean_V" is from an older version and is recomputed); when the value does not rank them (v_low
# null) the round runs without back-offs (v_low -1e9, logged); the agent (ACT head + corrections + done stop +
# back-off, with exploration) runs on every training grasp x <starts> (default 3) and writes the round's
# trajectories and manifest (run_wm320_attempts.py --mode collect); the learner mixes in rounds with DONE by itself.
# Then the same model is tested on the selection grasps (all x 4 starts): the ACT head alone and with the world model.
# Two rounds in a row without a better "with the world model" success than the best before end the rounds (exit 0).
# Run it from a frozen copy of the code (as wm320.sh).
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
run=${1:?learner run name}; gpus=${2:?gpus}; workers=${3:?workers}; tmib=${4:?torch MiB per worker}; mib=${5:?GPU MiB per GPU}
ram=${6:?RAM GB}; r1=${7:?last round}; min_step=${8:?learner step before round 1}; starts=${9:-3}
q320_gpus_ok "$gpus" || { echo "GPUs must be one of 0-7 or a comma list of distinct ones"; exit 1; }
for v in "$workers" "$tmib" "$mib" "$ram" "$r1" "$min_step" "$starts"; do q320_int "$v" || { echo "not a whole number: $v"; exit 1; }; done
[ "$r1" -ge 1 ] && [ "$workers" -ge 1 ] || { echo "need at least one round and one worker"; exit 1; }
ng=$(tr ',' '\n' <<< "$gpus" | wc -l); per=$(( (workers + ng - 1) / ng ))
[ "$mib" -ge $(( per * (tmib + 1500) )) ] || { echo "$per workers per GPU need $(( per * (tmib + 1500) )) MiB, $mib declared"; exit 1; }
[[ "$gpus" != *,* ]] || q320_watchdog_lists || { q320_log "online $run: the running watchdog does not know GPU lists; restart it first"; exit 1; }
cd "$CODE" || exit 1
export PYTHONPATH=$CODE/world_model/src${PYTHONPATH:+:$PYTHONPATH}
L=$Q/runs/$run; P=$Q/data/online320_$run; mkdir -p "$P"

learner_state() {    # [round] -> "<reads this pool: 1 | 0> <newest logged step>
  #                      <step of the first data record of the newest unbroken run of data records listing the round, or -1>"
  python - "$L/log.jsonl" "$P" "${1:-}" <<'EOF'
import json, sys
ok, step, seen = 0, 0, -1
try:
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except ValueError:                     # a line being written
            continue
        ev = r.get("event")
        if ev in ("start", "resume"):
            ok = int((r.get("args") or {}).get("online") == sys.argv[2])
        if isinstance(r.get("step"), int):
            step = max(step, r["step"])
        if ev == "data" and sys.argv[3]:
            if sys.argv[3] in (r.get("online_rounds") or []):
                seen = r["step"] if seen < 0 else seen
            else:
                seen = -1                      # the round is not (or no longer) read
except OSError:
    pass
print(ok, step, seen)
EOF
}

wait_learner() {     # <R>: waits until the learner is ready for collecting round R (header); exits when it is gone
  local down=0 said="" ok step seen running n=0
  while true; do
    read -r ok step seen < <(learner_state "r$(( $1 - 1 ))")
    [ -e "$L/DONE.json" ] && { q320_log "online $run: the learner is finished; rounds end before round $1"; exit 0; }
    running=""; systemctl --user is-active --quiet "qx320-train-$run.scope" && running=1
    if [ -n "$running" ]; then down=0; else down=$(( down + 1 )); fi
    [ "$down" -lt 2 ] || { q320_log "online $run: the learner's unit is not running; rounds end before round $1"; exit 1; }
    if [ -n "$running" ] && [ "$ok" = 1 ]; then
      if [ "$1" = 1 ]; then [ "$step" -ge $(( min_step + 500 )) ] && [ -e "$L/last.pt" ] && return 0
      else [ "$seen" -ge 0 ] && [ "$step" -ge $(( seen + 1000 )) ] && return 0; fi
    fi
    if [ -z "$said" ] || [ $(( n % 12 )) = 0 ]; then                   # once, then hourly
      q320_log "online $run: round $1 waits for the learner (unit running: ${running:-no}, reads $P: $ok, step $step, round $(( $1 - 1 )) read at step $seen)"
      said=1
    fi
    n=$(( n + 1 ))
    sleep 300
  done
}

calibrate() {        # sets vl: the threshold of round R's checkpoint (computed once, kept in r<R>/v_low.json); exits when there is none
  local rc
  if [ ! -s "$D/v_low.json" ] || ! grep -q '"value": "mean_V"' "$D/v_low.json"; then     # (older files: recompute)
    [ -e "$D/model.pt" ] || { q320_log "online $run: round $R has no model.pt"; exit 1; }
    [ ! -e "$D/v_low.json" ] || mv "$D/v_low.json" "$D/v_low.old.json"                    # never read a stale threshold
    q320_run_when_fits "qx320-test-${run}-r$R-vlow" 30 "${gpus%%,*}" "$mib" test 1 "$Q/logs/online_${run}_r${R}_vlow.log" \
      python world_model/scripts/online/calibrate_vlow320.py --data "$Q/data/wm_v1_fast" --ckpt "$D/model.pt" --out "$D/v_low.json"
    rc=$?
    [ "$rc" = 0 ] || q320_log "online $run: round $R threshold calibration code $rc"
  fi
  vl=$(python -c "import json,sys
try:
    v = json.load(open(sys.argv[1]))['v_low']
    print(-1e9 if v is None else float(v))
except Exception:
    pass" "$D/v_low.json")
  [[ "$vl" =~ ^-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$ ]] || { q320_log "online $run: round $R has no usable v_low.json"; exit 1; }
}
success_of() {       # <summary.json> -> the success figure of all selection grasps, or nothing
  python -c "import json,sys
try:
    s = json.load(open(sys.argv[1]))['groups']['all']['success']
    print('' if s is None else s)
except Exception:
    pass" "$1"
}

best=-1; worse=0
for R in $(seq 1 "$r1"); do
  D=$P/r$R
  if [ ! -e "$D/DONE" ]; then
    mkdir -p "$D"
    if [ ! -e "$D/model.pt" ]; then
      wait_learner "$R"
      cp "$L/last.pt" "$D/model.pt.tmp" && mv "$D/model.pt.tmp" "$D/model.pt" || exit 1
    fi
    calibrate
    q320_log "online $run: round $R collects with v_low $vl ($(tr -d '\n' < "$D/v_low.json" | cut -c1-400))"
    q320_run_when_fits "qx320-collect-${run}-r$R" "$ram" "$gpus" "$mib" test 1 "$Q/logs/online_${run}_r$R.log" \
      python world_model/scripts/online/run_wm320_attempts.py --grasps "$Q/data/wm_v1" --ckpt "$D/model.pt" --mode collect --split train \
      --pool "$P" --round "$R" --starts "$starts" --workers "$workers" --gpus "$gpus" --torch-mib "$tmib" --v-low="$vl"
    rc=$?
    [ -e "$D/DONE" ] || { q320_log "online $run: round $R has no DONE (collector code $rc); rounds stop"; exit 1; }
  fi
  if [ ! -e "$D/select_act/summary.json" ] || [ ! -e "$D/select_wm/summary.json" ]; then calibrate; fi
  for m in act wm; do
    O=$D/select_$m
    [ -e "$O/summary.json" ] && continue
    q320_run_when_fits "qx320-test-${run}-r$R-$m" "$ram" "$gpus" "$mib" test 1 "$Q/logs/online_${run}_r${R}_select_$m.log" \
      python world_model/scripts/online/run_wm320_attempts.py --grasps "$Q/data/wm_v1" --ckpt "$D/model.pt" --mode "$m" --split select \
      --out "$O" --starts 4 --workers "$workers" --gpus "$gpus" --torch-mib "$tmib" --v-low="$vl" \
      || q320_log "online $run: round $R select test $m code $?"
  done
  s=$(success_of "$D/select_wm/summary.json")
  q320_log "online $run: round $R done; select success with the world model ${s:-unknown}, ACT alone $(success_of "$D/select_act/summary.json")"
  [ -n "$s" ] || continue
  if python -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$s" "$best"; then best=$s; worse=0
  else worse=$(( worse + 1 )); fi
  if [ "$worse" -ge 2 ]; then q320_log "online $run: two rounds without improvement; rounds end after round $R"; break; fi
done
exit 0
