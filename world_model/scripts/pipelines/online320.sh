#!/bin/bash
# Online rounds of the 320 px world model, next to its learner (wm320.sh ... online: train_wm320.py --online <pool>).
#   online320.sh <run> <gpus> <workers> <torch_mib per worker> <gpu_mib per GPU> <ram_gb> <last round> <min_step> [starts]
# <gpus>: GPUs WITHOUT the learner (the budget leaves no room for a collector worker beside a learner); the workers are
# spread evenly over them; <gpu_mib> must cover ceil(workers / GPUs) x (<torch_mib> + 1500 for context and renderer).
# Before round 1: waits until the learner's newest start/resume record reads this pool (--online) and its log has
# reached step <min_step> (e.g. the end of the value warm-up).
# Round R = 1..<last round> (finished parts of a round are skipped, so a restart resumes): from round 2 on, first waits
# until the learner has read round R-1 and trained 500 steps on it; the learner's newest last.pt is copied to
# <pool>/r<R>/model.pt; the back-off threshold v_low is taken from the learner's newest validation on the selection
# grasps: midway between the median value before the board face of pushes that jammed and of pushes that went in
# (the value holds -shaping * phi, so no fixed number fits); the agent (ACT head + corrections + done stop + back-off, with exploration) runs
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

learner_state() {    # [round] -> "<reads this pool: 1|0> <newest logged step> <step at which it first read the round, or -1> <v_low or ->"
  python - "$L/log.jsonl" "$P" "${1:-}" <<'EOF'
import json, sys
ok, step, seen, vl = 0, 0, -1, "-"
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get("event") in ("start", "resume"):
            ok = int((r.get("args") or {}).get("online") == sys.argv[2])
        if isinstance(r.get("step"), int):
            step = max(step, r["step"])
        if r.get("event") == "data" and sys.argv[3] and sys.argv[3] in (r.get("online_rounds") or []) and seen < 0:
            seen = r["step"]
        if r.get("event") == "val":
            s = r.get("select") or {}
            a, b = s.get("V_before_face_jammed_median"), s.get("V_before_face_inserted_median")
            if isinstance(a, float) and isinstance(b, float) and b > a:
                vl = f"{(a + b) / 2:.4f}"
except OSError:
    pass
print(ok, step, seen, vl)
EOF
}
said=""
while true; do
  read -r ok step seen vl < <(learner_state)
  [ "$ok" = 1 ] && [ "$step" -ge "$min_step" ] && [ "$vl" != - ] && break
  [ -n "$said" ] || { q320_log "online $run: waiting for the learner (reads $P: $ok, step $step of $min_step, v_low $vl)"; said=1; }
  sleep 300
done
q320_log "online $run: rounds start (learner at step $step)"

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
      said=""
      while true; do
        read -r ok step seen vl < <(learner_state "r$(( R - 1 ))")
        [ "$ok" = 1 ] && [ "$seen" -ge 0 ] && [ "$step" -ge $(( seen + 500 )) ] && break
        [ -n "$said" ] || { q320_log "online $run: round $R waits for the learner to train on round $(( R - 1 ))"; said=1; }
        sleep 300
      done
    fi
    if [ ! -e "$D/model.pt" ]; then
      [ -e "$L/last.pt" ] || { q320_log "online $run: no $L/last.pt"; exit 1; }
      cp "$L/last.pt" "$D/model.pt.tmp" && mv "$D/model.pt.tmp" "$D/model.pt" || exit 1
    fi
    read -r ok step seen vl < <(learner_state)
    [ -e "$D/v_low" ] || [ "$vl" != - ] || { q320_log "online $run: no v_low from the learner's validation"; exit 1; }
    [ -e "$D/v_low" ] || echo "$vl" > "$D/v_low"              # one threshold per round (also for its tests)
    q320_log "online $run: round $R collects with v_low $(cat "$D/v_low")"
    q320_run_when_fits "qx320-collect-${run}-r$R" "$ram" "$gpus" "$mib" test 1 "$Q/logs/online_${run}_r$R.log" \
      python world_model/scripts/online/run_wm320_attempts.py --grasps "$Q/data/wm_v1" --ckpt "$D/model.pt" --mode collect --split train \
      --pool "$P" --round "$R" --starts "$starts" --workers "$workers" --gpus "$gpus" --torch-mib "$tmib" --v-low "$(cat "$D/v_low")"
    rc=$?
    [ -e "$D/DONE" ] || { q320_log "online $run: round $R has no DONE (collector code $rc); rounds stop"; exit 1; }
  fi
  for m in act wm; do
    O=$D/select_$m
    [ -e "$O/summary.json" ] && continue
    q320_run_when_fits "qx320-test-${run}-r$R-$m" "$ram" "$gpus" "$mib" test 1 "$Q/logs/online_${run}_r${R}_select_$m.log" \
      python world_model/scripts/online/run_wm320_attempts.py --grasps "$Q/data/wm_v1" --ckpt "$D/model.pt" --mode "$m" --split select \
      --out "$O" --starts 4 --workers "$workers" --gpus "$gpus" --torch-mib "$tmib" --v-low "$(cat "$D/v_low")" \
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
