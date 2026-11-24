#!/bin/bash
# Inputs and preflight of a collection (run by collect_lib.sh cl_prep inside a capped CPU scope, from the frozen code
# <out>/code):   bash collect_prep.sh <out> <config json>
# 1. First run: copies the config to <out>/inputs/config.json, the milestone-1b results of each cfg.m1b_dirs[k] to
#    <out>/inputs/m1b/<k>/ and the asset folder (WM_AUTOMATE_ASSETS or /mnt/storage3/qingxin/automate/assets) to
#    <out>/inputs/assets/, every switch of wm.sim_automate.config (collect_lib.sh CL_SWITCHES: socket draw, spin axis,
#    physics device, socket mount, weld, camera scale, PhysX threads, Kit GPU, start gap from (WM_AUTOMATE_START_GAP_FROM,
#    approved "socket_top" on 2026-11-23); the caller's environment, the code's
#    defaults when unset; WM_FORBIDDEN_GPU always 0; refused unless every switch equals config.APPROVED) to
#    <out>/inputs/switches.env, records the sources in
#    <out>/inputs/SOURCES.txt and makes <out>/inputs read-only. Before anything is copied or frozen (r8-ops-01), the
#    config is loaded as production (CollectConfig.load(production=True): keys, values, pinned values, approved
#    switches in the caller's environment, the same that cl_write_switches freezes), worker_resources and the device
#    check run: a hand-written key, value or resource error refuses with nothing frozen. Later runs: refuse unless
#    <config> is byte-identical to <out>/inputs/config.json; when <out>/plan.json is missing too, nothing was collected
#    there and the refusal says to start in a NEW <out> (frozen inputs are never deleted automatically).
# 2. Runs collect_expert.py --preflight on the frozen inputs with a clean environment (every inherited WM_* unset, then
#    switches.env; collect_lib.sh cl_clean_env): production keys and pinned values (CollectConfig.validate: 15 grasps,
#    the wm_v1 trajectory mix and picture settings, approved switches), worker resources, config device =
#    WM_AUTOMATE_DEVICE, enough selection / held-out grasps for train_wm320, plan == expected grasps, plan frozen in
#    <out>/plan.json; writes <out>/logs/resources.txt.
set -u
OUT=$1; CFG=$2
W=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$W/scripts/automate/collect_lib.sh"
IN=$OUT/inputs
if [ -d "$IN" ]; then
  if ! cmp -s "$CFG" "$IN/config.json"; then
    echo "refused: $CFG differs from the frozen $IN/config.json"
    [ -e "$OUT/plan.json" ] || echo "refused: nothing was collected in $OUT (no plan.json): its inputs were frozen by a" \
      "refused first launch; start again in a NEW <out> folder"
    exit 3
  fi
else
  # the production checks of the preflight that need no inputs, before anything is frozen (r8-ops-01); in the
  # environment cl_write_switches computes the switches in (only the switch variables of the caller kept)
  dirs=$(keep=" $(for kv in $CL_SWITCHES; do echo -n "${kv%%=*} "; done)"
  for v in $(compgen -e | grep '^WM_'); do [[ "$keep" == *" $v "* ]] || unset "$v"; done
  unset WM_FORBIDDEN_GPU
  "$CL_PY" -I -c "
import sys
sys.path.insert(0, sys.argv[2])
from wm.sim_automate import config as C
from wm.sim_automate.collect import CollectConfig, Refused, worker_resources
try:
    cfg = CollectConfig.load(sys.argv[1], production=True)
    worker_resources(cfg)
    if cfg.device != C.PHYSICS_DEVICE:
        raise Refused(f'config device {cfg.device!r} differs from WM_AUTOMATE_DEVICE / config.PHYSICS_DEVICE '
                      f'{C.PHYSICS_DEVICE!r}: one physics device per collection')
except Refused as exc:
    print(f'refused: {exc}', file=sys.stderr)
    sys.exit(3)
print('\n'.join(cfg.m1b_dirs))" "$CFG" "$W/src") || { echo "refused: $CFG fails the production checks (above); nothing was frozen"; exit 3; }
  SRC_ASSETS=${WM_AUTOMATE_ASSETS:-/mnt/storage3/qingxin/automate/assets}
  T=$OUT/inputs.tmp
  [ -d "$T" ] && { chmod -R u+w "$T"; rm -rf "$T"; }
  mkdir -p "$T/m1b" || exit 3
  cp "$CFG" "$T/config.json" || exit 3
  k=0
  { echo "config $(readlink -f "$CFG") sha256 $(sha256sum < "$CFG" | cut -c1-64)"; echo "assets $SRC_ASSETS"; } > "$T/SOURCES.txt"
  for d in $dirs; do
    [ -d "$d" ] || { echo "refused: milestone-1b folder $d missing"; exit 3; }
    mkdir -p "$T/m1b/$k"
    find "$d" -maxdepth 1 -name '*.json' -exec cp -p {} "$T/m1b/$k/" \; || exit 3
    echo "m1b/$k $d" >> "$T/SOURCES.txt"
    k=$((k + 1))
  done
  # the switches of wm.sim_automate.config (environment variables) are fixed for the collection here
  cl_write_switches "$T/switches.env" "$W" || { echo "refused: cannot write $T/switches.env"; exit 3; }
  [ -d "$SRC_ASSETS" ] || { echo "refused: asset folder $SRC_ASSETS missing"; exit 3; }
  rsync -a --exclude '*.part' "$SRC_ASSETS/" "$T/assets/" || { echo "refused: asset copy failed"; exit 3; }
  mv "$T" "$IN" && chmod -R a-w "$IN" || { echo "refused: freezing the inputs failed"; exit 3; }
  echo "froze inputs in $IN"
fi
cl_clean_env "$IN/switches.env" || exit 3      # the frozen switches, whatever the caller exported
cat "$IN/switches.env"
export WM_AUTOMATE_ASSETS=$IN/assets WM_AUTOMATE_M1B=$IN/m1b PYTHONDONTWRITEBYTECODE=1
cd "$W"
res=$("$CL_PY" -u scripts/automate/collect_expert.py --out "$OUT" --config "$IN/config.json" --preflight 2>&1)
rc=$?
echo "$res"
if [ $rc -ne 0 ]; then
  if [ ! -e "$OUT/plan.json" ]; then           # inputs frozen, nothing collected: a corrected config needs a new <out>
    msg="nothing was collected in $OUT (no plan.json): its inputs were frozen by a refused first launch; start again in a NEW <out> folder"
    echo "refused: $msg"
    mkdir -p "$OUT/logs" && echo "$(date '+%F %T') collect_prep: preflight exit $rc; $msg" >> "$OUT/logs/ALERT"
  fi
  exit $rc
fi
echo "$res" | awk '/^RESOURCES / {print $2, $3, $4, $5}' > "$OUT/logs/resources.txt"
[ -s "$OUT/logs/resources.txt" ] || { echo "refused: no RESOURCES line"; exit 3; }
