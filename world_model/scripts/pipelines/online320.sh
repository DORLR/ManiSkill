#!/bin/bash
# Online rounds of the 320 px world model, next to its learner (wm320.sh ... online: train_wm320.py --online <pool>).
#   online320.sh <run> <gpus> <workers> <torch_mib per worker> <gpu_mib per GPU> <ram_gb> <last round> <min_step> [starts]
# <gpus>: GPUs WITHOUT the learner (the budget leaves no room for a collector worker beside a learner); the workers are
# spread evenly over them; <gpu_mib> must cover ceil(workers / GPUs) x (<torch_mib> + 1500 for context and renderer).
# Before round 1: waits until the learner's newest start/resume record reads this pool (--online) and its log has
# reached step <min_step> (e.g. the end of the value warm-up). Every wait ends the script (exit 1) when the learner
# is finished (DONE.json), when its unit qx320-train-<run> was seen running and then found inactive at two checks
# 5 minutes apart, or when its newest start/resume record reads another pool (or none).
# Round R = 1..<last round> (finished parts of a round are skipped, so a restart resumes): from round 2 on, first waits
# until the learner has read round R-1 and trained 500 steps on it; the learner's newest last.pt is copied to
# <pool>/r<R>/model.pt; the back-off threshold of that very checkpoint is computed (calibrate_vlow320.py: midway
# between the median value before the board face of selection pushes that jammed and that went in; the value holds
# -shaping * phi, so no fixed number fits) into r<R>/v_low.json; when the value does not rank them (v_low null) the
# round runs without back-offs (v_low -1e9, logged); the agent (ACT head + corrections + done stop + back-off, with exploration) runs
# on every training grasp x <starts> (default 3) and writes the round's trajectories and manifest
# (run_wm320_attempts.py --mode collect); the learner mixes in rounds with DONE by itself. Then the same model is tested
# on the selection grasps (all x 4 starts): the ACT head alone and with the world model. Two rounds in a row without a
# better "with the world model" success than the best before end the rounds (exit 0).
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

learner_state() {    # [round] -> "<reads this pool: 1 | 0 no | -1 no start record yet> <newest logged step> <step at which it first read the round, or -1>"
  python - "$L/log.jsonl" "$P" "${1:-}" <<'EOF'
import json, sys
ok, step, seen = -1, 0, -1
try:
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except ValueError:                     # a line being written
            continue
        if r.get("event") in ("start", "resume"):
            ok = int((r.get("args") or {}).get("online") == sys.argv[2])
        if isinstance(r.get("step"), int):
            step = max(step, r["step"])
        if r.get("event") == "data" and sys.argv[3] and sys.argv[3] in (r.get("online_rounds") or []) and seen < 0:
            seen = r["step"]
except OSError:
    pass
print(ok, step, seen)
EOF
}
idle=0; seen_running=""
learner_alive() {    # <ok>: fails (after logging) when the learner is finished, reads another pool, or its unit stopped
  [ -e "$L/DONE.json" ] && { q320_log "online $run: the learner is finished; rounds end"; return 1; }
  [ "$1" = 0 ] && { q320_log "online $run: the learner's newest start/resume does not read $P; rounds end"; return 1; }
  if systemctl --user is-active --quiet "qx320-train-$run.scope"; then idle=0; seen_running=1
  elif [ -n "$seen_running" ]; then idle=$(( idle + 1 )); fi        # (a learner still queued for its budget is not idle)
  [ "$idle" -lt 2 ] || { q320_log "online $run: the learner's unit is not running; rounds end"; return 1; }
}
said=""; idle=0
while true; do
  read -r ok step seen < <(learner_state)
  [ "$ok" = 1 ] && [ "$step" -ge "$min_step" ] && break
  [ -n "$said" ] || { q320_log "online $run: waiting for the learner (reads $P: $ok, step $step of $min_step)"; said=1; }
  learner_alive "$ok" || exit 1
  sleep 300
done
q320_log "online $run: rounds start (learner at step $step)"

calibrate() {        # sets vl: the threshold of round R's checkpoint (computed once, kept in r<R>/v_low.json); exits when there is none
  local rc
  if [ ! -s "$D/v_low.json" ]; then
    [ -e "$D/model.pt" ] || { q320_log "online $run: round $R has no model.pt"; exit 1; }
    q320_run_when_fits "qx320-test-${run}-r$R-vlow" 30 "${gpus%%,*}" 6000 test 1 "$Q/logs/online_${run}_r${R}_vlow.log" \
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
    if [ ! -e "$D/model.pt" ] && [ "$R" -gt 1 ]; then          # the learner has trained on the previous round
      said=""; idle=0
      while true; do
        read -r ok step seen < <(learner_state "r$(( R - 1 ))")
        [ "$ok" = 1 ] && [ "$seen" -ge 0 ] && [ "$step" -ge $(( seen + 500 )) ] && break
        [ -n "$said" ] || { q320_log "online $run: round $R waits for the learner to train on round $(( R - 1 ))"; said=1; }
        learner_alive "$ok" || exit 1
        sleep 300
      done
    fi
    if [ ! -e "$D/model.pt" ]; then
      [ -e "$L/last.pt" ] || { q320_log "online $run: no $L/last.pt"; exit 1; }
      cp "$L/last.pt" "$D/model.pt.tmp" && mv "$D/model.pt.tmp" "$D/model.pt" || exit 1
    fi
    calibrate
    q320_log "online $run: round $R collects with v_low $vl ($(tr -d '\n' < "$D/v_low.json" | cut -c1-300))"
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
