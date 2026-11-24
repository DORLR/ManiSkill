#!/bin/bash
# Split the assemblies without a result in <out> over N workers (separate Isaac apps, GPU 1, MemoryMax each), each a
# run_batch.sh loop in its own tmux -L qingxin session. The starts are serialised and checked by caps.sh qx_run (lock,
# 90 s between GPU starts, our caps); a refused start waits in run_batch.sh.
#   [BATCH_TIMEOUT_S=5400] bash parallel_batch.sh <out dir> [N=6] [MEMMAX=18G] [--ids <a,b,...>] [--retry-crashes]
# --ids: only these assemblies (each must be one of assets.assembly_ids(), no repeats; refused otherwise); without it
# every assembly. The other assemblies are neither run nor retried nor listed.
# The batch runs from a frozen copy of this world_model tree, <out>/code (collect_lib.sh cl_freeze_code; a second
# call must find it unchanged), with the switches frozen in <out>/switches.env (collect_lib.sh cl_write_switches: the
# caller's environment, the code's defaults when unset; written once; refused unless config.APPROVED). Every result
# records its provenance (batch_assemblies.py); collect.py's gate compares it with the collection's. Refused while
# <out>/STOP exists (an exit 7 of any loop writes it: a forbidden GPU-0 context stops the whole batch).
# The watcher qx_watch.py must run (collect_lib.sh cl_watch_ensure; <out> is registered with it): memory-pressure and
# GPU-1 sheds, the periodic GPU-0 check. Refused otherwise.
# TIMEOUT per process: BATCH_TIMEOUT_S from the environment, default 5400 s (about 2 x the measured 25-40 min per
# process with CPU PhysX; old batch2: median 90 s, max 159 s per assembly); a whole number >= 1, refused otherwise
# (r7-ops-01: gpu_batch_r3.sh sets a larger default for GPU PhysX). `timeout -k 120` in run_on_rs2.sh; the launcher
# log names it on every 'worker <w> started' line. In-assembly hangs are caught earlier by the heartbeat (1020 s).
# Refused (printed and in the launcher log) while a tmux session qingxin-batch* of an earlier launch still exists on
# our tmux server (its loops would share assemblies with the new ones, r5-ops-01). A worker whose tmux session could
# not be started is printed and makes the launcher exit 1 after the others have started.
# Per-process logs and unit files: tag <out name>_<launch stamp>_p<w> (r5-ops-02), so neither a --retry-crashes pass
# nor another batch (the GPU batch) overwrites an earlier pass's logs in /mnt/storage3/qingxin/automate/m1/logs.
# --retry-crashes: an <id>.json whose only content is the crash record ("error" exactly 'the simulator process died
# during this assembly', no "ii") is moved to <out>/errors/<id>.<unix time>.json and the assembly runs again, at most
# once per id (an id with a file in <out>/errors/ is not retried again). The launcher log lists every id that still has
# only an error result.
# Run plan (r11-ops-chk-01): caps.sh charges each of our processes at least 7 GB of GPU 1 against its free memory F
# (rule F - P - G_new >= 12 GB). While F < 12 GB + n x 7 GB, only the loops that fit start; the others wait in
# run_batch.sh (e.g. 2026-11-23: F about 35 GB, 3 of N=6 loops at once, two waves of about 30-50 min). Each waiting
# loop then writes one 'caps refusals / watcher waits reached 30 (about 30 min); still waiting' line to the ALERT file:
# expected, not a fault, and no reason to stop the batch. Faults are 'giving up' (360 refusals, about 6 h), crash and
# rate-cap lines.
set -u
POS=(); RETRY=0; IDS=""; prev=""
for a in "$@"; do
  if [ "$prev" = "--ids" ]; then IDS=$a
  elif [ "$a" = "--retry-crashes" ]; then RETRY=1
  elif [ "$a" != "--ids" ]; then POS+=("$a"); fi
  prev=$a
done
USAGE="usage: [BATCH_TIMEOUT_S=5400] bash parallel_batch.sh <out dir> [N=6] [MEMMAX=18G] [--ids <a,b,...>] [--retry-crashes]"
[ ${#POS[@]} -ge 1 ] || { echo "$USAGE"; exit 1; }
[ "$prev" = "--ids" ] && { echo "refused: --ids needs a list; $USAGE"; exit 1; }
[ -z "$IDS" ] || [[ "$IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "refused: --ids '$IDS' (comma-separated assembly ids)"; exit 1; }
OUT=$(readlink -f "${POS[0]}"); N=${POS[1]:-6}; export MEMMAX=${POS[2]:-18G}
TIMEOUT=${BATCH_TIMEOUT_S:-5400}
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo "refused: BATCH_TIMEOUT_S '$TIMEOUT' (a whole number of seconds >= 1)"; exit 1; }
export TIMEOUT
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "refused: N '$N'"; exit 1; }
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
LOG=/mnt/storage3/qingxin/automate/m1/logs/parallel_launcher.log
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
mkdir -p "$(dirname "$LOG")" "$OUT"
[ -e "$OUT/STOP" ] && { echo "refused: $OUT/STOP exists ($(tail -1 "$OUT/STOP"))"; exit 1; }
busy=$($CL_TMUX ls -F '#S' 2>/dev/null | grep '^qingxin-batch')
[ -z "$busy" ] || { echo "$(date) refused: batch loops still running: $(echo $busy)" | tee -a "$LOG"; exit 1; }
STAMP=$(date +%Y%m%dT%H%M%S)
cl_freeze_code "$LIVE" "$OUT" | tee -a "$LOG"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
W=$OUT/code
if [ ! -e "$OUT/switches.env" ]; then
  cl_write_switches "$OUT/switches.env" "$W" && chmod a-w "$OUT/switches.env" || { echo "refused: cannot write $OUT/switches.env"; exit 1; }
fi
echo "$(date) switches $(tr '\n' ' ' < "$OUT/switches.env")" | tee -a "$LOG"
cl_watch_ensure "$OUT" "$W" | tee -a "$LOG"; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "refused: no live watcher (qx_watch.py)" | tee -a "$LOG"; exit 1; }
export QX_SWITCHES=$OUT/switches.env
cd "$W"
listing=$( ( cl_clean_env "$QX_SWITCHES" >/dev/null && qx_run_retry prep listing 2G 100% 0 "$CL_PY" -c "
import json, os, sys, time
sys.path.insert(0, 'src')
from pathlib import Path
from wm.sim_automate import assets as A
out, retry = Path(sys.argv[1]), sys.argv[2] == '1'
CRASH = 'the simulator process died during this assembly'
every = A.assembly_ids()
if sys.argv[3]:
    want = sys.argv[3].split(',')
    bad = sorted(set(a for a in want if a not in every))
    if bad or len(set(want)) != len(want):
        print('REFUSED --ids: ' + (f'not assemblies {bad}' if bad else 'repeated ids') + f' in {sys.argv[3]}')
        sys.exit(3)
    every = [a for a in every if a in set(want)]

def crash_only(f):
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(r, dict) and r.get('error') == CRASH and 'ii' not in r

retried, left = [], []
for a in every:
    f = out / f'{a}.json'
    if not (f.exists() and crash_only(f)):
        continue
    if retry and not list((out / 'errors').glob(f'{a}.*.json')):
        (out / 'errors').mkdir(exist_ok=True)
        os.replace(f, out / 'errors' / f'{a}.{int(time.time())}.json')
        retried.append(a)
    else:
        left.append(a)
print('RETRIED', ' '.join(retried))
print('ERROR_ONLY', ' '.join(left))
print('IDS', ' '.join(a for a in every if not (out / f'{a}.json').exists()))" "$OUT" "$RETRY" "$IDS" ) )
rc=$?
[ $rc -eq 0 ] && echo "$listing" | grep -q '^IDS' || { echo "refused: the listing failed (exit $rc): $listing" | tee -a "$LOG"; exit 1; }
todo=$(echo "$listing" | awk '/^IDS/ {$1=""; print}')
echo "$(date) retried after a crash (moved to $OUT/errors/): $(echo "$listing" | sed -n 's/^RETRIED *//p')" >> "$LOG"
echo "$(date) still only an error result (not retried): $(echo "$listing" | sed -n 's/^ERROR_ONLY *//p')" >> "$LOG"
ids=($todo)
echo "$(date) assemblies: ${IDS:-all}; timeout ${TIMEOUT} s per process" >> "$LOG"
echo "$(date) remaining ${#ids[@]}: ${ids[*]}" >> "$LOG"
notstarted=0
for w in $(seq 0 $((N-1))); do
  list=$(for i in "${!ids[@]}"; do [ $((i % N)) -eq $w ] && echo -n "${ids[$i]},"; done)
  list=${list%,}
  [ -z "$list" ] && continue
  tag=$(basename "$OUT")_${STAMP}_p${w}
  if $CL_TMUX new -d -s qingxin-batch$w "QX_SWITCHES=$QX_SWITCHES MEMMAX=$MEMMAX TIMEOUT=$TIMEOUT bash $W/scripts/automate/run_batch.sh $tag --out $OUT --ids $list"; then
    echo "$(date) worker $w started (tmux -L qingxin qingxin-batch$w, code $W, log tag $tag, timeout ${TIMEOUT} s), ids $list" >> "$LOG"
  else
    echo "$(date) worker $w NOT started (tmux failed; session exists?), ids $list" | tee -a "$LOG"
    notstarted=$((notstarted + 1))
  fi
done
echo "$(date) launcher done ($notstarted workers not started)" >> "$LOG"
[ $notstarted -eq 0 ] || { echo "refused: $notstarted workers not started (launcher log $LOG)"; exit 1; }
