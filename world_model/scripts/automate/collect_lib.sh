#!/bin/bash
# Functions shared by collect_launch.sh, collect_add_worker.sh, collect_worker.sh and train_automate.sh (source it
# after caps.sh). Every process with real work runs through caps.sh qx_run; tmux sessions live on our own tmux server
# (tmux -L qingxin), never on the shared account's default server.
#
#   cl_code_hash <dir>                 sha256 over every file of <dir> (sorted relative paths and contents)
#   cl_freeze_code <world_model> <run dir>
#       First call: copies <world_model> (without __pycache__ / .git) to <run dir>/code, makes it read-only and writes
#       <run dir>/code.sha256. Later calls: refuse unless <run dir>/code still has that hash. Workers and training
#       import only from <run dir>/code.
#   cl_batch_code_guard <world_model> <out> <config>
#       (r11-ops-02, as gpu_batch_r3.sh's guard) while <out>/code does not exist yet: refuses (message, return 1)
#       unless cl_code_hash <world_model> equals <B>/code.sha256, B = the first folder of the config's m1b_dirs (the
#       milestone-1b batch the collection is gated on); a config without m1b_dirs or a B without code.sha256 is
#       refused too. A later start (<out>/code exists) is left to cl_freeze_code's unchanged-copy check. Used by
#       collect_launch.sh and smoke/collect_smoke.sh, so the smoke and production run the batch's frozen code.
#   cl_prep <out> <config>             (starts through caps.sh qx_run_retry: a caps refusal is retried every 60 s for
#                                      up to QX_RETRY_MAX_S, r9-ops-01; so is cl_train_preflight)
#                                      freezes the inputs and runs the CPU preflight (collect_prep.sh in a capped
#                                      scope "prep", no GPU); writes <out>/logs/resources.txt "MEM CPU GPU_GB TIMEOUT_S"
#   cl_static_budget <N> <MEM> <CPU> <GPU_GB>   refuses N workers that can never fit the caps together
#   cl_watch_ensure <dir> <code>       registers <dir> with the watcher, starts qx_watch.py if none runs and returns 0
#                                      only when a qingxin-watch-* scope is active and QX_RUN_DIR/watch.alive is at
#                                      most 180 s old (waits up to 60 s after a start); else WARNING and 1. A live
#                                      watcher whose code (QX_RUN_DIR/watch/code.sha256, written by qx_watch.py at its
#                                      start) differs from <code>/scripts/automate/qx_watch.py gives a WARNING line
#                                      (still 0): restart it by its exact unit name (systemctl --user stop <unit>)
#   cl_isaac_env                       exports our Isaac environment (HOME, caches, tmp on storage3)
#   cl_write_switches <file> <code>    writes the effective switches of wm.sim_automate.config (CL_SWITCHES) as the
#                                      code tree <code> computes them from the caller's environment (its defaults when
#                                      unset, so the defaults are never duplicated here); WM_FORBIDDEN_GPU is never
#                                      taken from the caller (always the code's 0; the GPU-0 rule is not a switch).
#                                      Refused (nothing written) when config.unapproved_switches() is not empty: a
#                                      value other than config.APPROVED (socket draw, spin axis, weld, PhysX threads,
#                                      Kit GPU, mount, camera scale, forbidden GPU, start gap from: "socket_top" since
#                                      the user's decision of 2026-11-23) exported by the launching shell
#   cl_clean_env [<switches file>]     unsets every inherited WM_* variable (the tmux -L qingxin server's environment
#                                      is whatever shell first started it; also WM_AUTOMATE_DIAGNOSTIC), sets
#                                      WM_FORBIDDEN_GPU=0 and sources the validated switches file
#   cl_train_env <code>                exports the training environment of the frozen copy <code> (PYTHONPATH = only
#                                      <code>/src, plus CL_EXTRA_SITE when set (empty: scikit-learn, joblib and
#                                      threadpoolctl come from the isaaclab env itself), HF / torch caches, offline,
#                                      PYTHONNOUSERSITE=1: no user site-packages of any HOME can shadow the env)
#   cl_train_preflight <code>          in a capped CPU scope with the job's environment (cl_isaac_env, cl_clean_env,
#                                      cl_train_env, as train_automate.sh --inside; r7-data-05): wm imports from
#                                      <code>/src, sklearn / joblib / threadpoolctl import (versions and files printed)
#                                      and none of them from a user site; both encoder weight files exist
CL_PY=/mnt/storage3/qingxin/envs/isaaclab/bin/python
CL_TMUX="tmux -L qingxin"
CL_HF_HOME=/mnt/storage2/qingxin/cache/hf                 # SegFormer-B1 (revision a775b4f5) is cached here
CL_TORCH_HOME=/mnt/storage2/qingxin/cache/torch           # hub/checkpoints/resnet34-b627a593.pth
# scikit-learn (wm320.auc), joblib, threadpoolctl: the isaaclab env's own copy (r4-ops-07 / r4-data-04: one source;
# the older pip --target copy /mnt/storage3/qingxin/envs/sklearn_extra stays on disk, unused)
CL_EXTRA_SITE=
CL_TRAIN_WEIGHTS="$CL_TORCH_HOME/hub/checkpoints/resnet34-b627a593.pth $CL_HF_HOME/hub/models--nvidia--segformer-b1-finetuned-ade-512-512"
# the switches of wm.sim_automate.config: environment variable=module attribute
CL_SWITCHES="WM_AUTOMATE_SOCKET_DRAW=SOCKET_DRAW WM_AUTOMATE_SPIN_AXIS=GRASP_SPIN_AXIS WM_AUTOMATE_DEVICE=PHYSICS_DEVICE
WM_AUTOMATE_MOUNT=SOCKET_MOUNT WM_AUTOMATE_WELD=WELD_MODE WM_CAM_SCALE=CAM_SCALE WM_PHYSX_THREADS=PHYSX_CPU_THREADS
WM_KIT_GPU=KIT_GPU WM_FORBIDDEN_GPU=FORBIDDEN_GPU WM_AUTOMATE_START_GAP_FROM=START_GAP_FROM"
CL_SWITCH_RE='^(WM_AUTOMATE_(SOCKET_DRAW|SPIN_AXIS|DEVICE|MOUNT|WELD|START_GAP_FROM)|WM_CAM_SCALE|WM_PHYSX_THREADS|WM_KIT_GPU|WM_FORBIDDEN_GPU)=[a-z_0-9:]+$'

cl_isaac_env() {
  export HOME=/mnt/storage3/qingxin/isaac/home XDG_CACHE_HOME=/mnt/storage3/qingxin/isaac/cache
  export TMPDIR=/mnt/storage3/qingxin/isaac/tmp OMNI_KIT_ACCEPT_EULA=YES NUMPY_MADVISE_HUGEPAGE=0
  export PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=$CL_TORCH_HOME
}

cl_write_switches() {
  local f=$1 code=$2
  ( set -o pipefail
    keep=" $(for kv in $CL_SWITCHES; do echo -n "${kv%%=*} "; done)"
    for v in $(compgen -e | grep '^WM_'); do [[ "$keep" == *" $v "* ]] || unset "$v"; done
    unset WM_FORBIDDEN_GPU
    "$CL_PY" -I -c "
import sys
sys.path.insert(0, sys.argv[1])
from wm.sim_automate import config as C
bad = C.unapproved_switches()
if bad:
    print('refused: switches not approved (config.APPROVED): ' + ', '.join(f'{k} = {v[0]!r}, approved {v[1]!r}'
          for k, v in sorted(bad.items())) + '; unset them in the launching shell', file=sys.stderr)
    sys.exit(2)
for kv in sys.argv[2].split():
    name, attr = kv.split('=')
    print(f'{name}={getattr(C, attr)}')" "$code/src" "$CL_SWITCHES" > "$f.tmp" ) && mv "$f.tmp" "$f" ||
    { rm -f "$f.tmp"; echo "refused: the code $code could not compute the switches, or a switch is not approved"; return 1; }
}

cl_clean_env() {
  local v f=${1:-}
  for v in $(compgen -e | grep '^WM_'); do unset "$v"; done
  export WM_FORBIDDEN_GPU=0
  [ -n "$f" ] || return 0
  [ -f "$f" ] || { echo "refused: $f missing"; return 1; }
  if grep -qvE "$CL_SWITCH_RE" "$f"; then echo "refused: $f malformed"; return 1; fi
  set -a; source "$f"; set +a
}

cl_train_env() {
  local code=$1
  export PYTHONPATH="$code/src${CL_EXTRA_SITE:+:$CL_EXTRA_SITE}" HF_HOME=$CL_HF_HOME TORCH_HOME=$CL_TORCH_HOME
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
}

cl_train_preflight() {
  local code=$1 f
  for f in $CL_TRAIN_WEIGHTS; do [ -e "$f" ] || { echo "refused: encoder weights $f missing"; return 1; }; done
  ( cl_isaac_env; cl_clean_env >/dev/null || exit 1; cl_train_env "$code"
    qx_run_retry prep "train-preflight" 4G 100% 0 "$CL_PY" -c "
import site, sys, wm, sklearn, joblib, threadpoolctl
from pathlib import Path
src = Path(sys.argv[1]).resolve()
assert Path(wm.__file__).resolve().is_relative_to(src), f'wm from {wm.__file__}, not {src}'
user = site.getusersitepackages()
for m in (sklearn, joblib, threadpoolctl):
    assert not str(Path(m.__file__).resolve()).startswith(user), f'{m.__name__} from the user site {m.__file__}'
print('TRAIN PREFLIGHT OK HOME', Path.home(), 'user site', site.ENABLE_USER_SITE, 'wm', wm.__file__, 'sklearn',
      sklearn.__version__, sklearn.__file__, 'joblib', joblib.__version__, joblib.__file__, 'threadpoolctl',
      threadpoolctl.__version__, threadpoolctl.__file__)" "$code/src" ) ||
    { echo "refused: training preflight failed (wm from the frozen copy; scikit-learn, joblib, threadpoolctl from" \
           "${CL_EXTRA_SITE:-the isaaclab env})"; return 1; }
}

cl_code_hash() {
  ( set -o pipefail
    cd "$1" && find . -type f ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}

cl_freeze_code() {
  local src=$1 run=$2 h
  mkdir -p "$run" || return 1
  if [ -d "$run/code" ]; then
    [ -f "$run/code.sha256" ] || { echo "refused: $run/code exists without code.sha256"; return 1; }
    h=$(cl_code_hash "$run/code") || { echo "refused: cannot hash $run/code"; return 1; }
    [ "$h" = "$(cat "$run/code.sha256")" ] || { echo "refused: $run/code changed since it was frozen"; return 1; }
    echo "frozen code $run/code (sha256 $h)"
    return 0
  fi
  rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .git "$src/" "$run/code.tmp/" || { echo "refused: copy failed"; return 1; }
  mv "$run/code.tmp" "$run/code" && chmod -R a-w "$run/code" || { echo "refused: freezing failed"; return 1; }
  h=$(cl_code_hash "$run/code") || return 1
  echo "$h" > "$run/code.sha256" && chmod a-w "$run/code.sha256"
  echo "froze $src -> $run/code (sha256 $h)"
}

cl_batch_code_guard() {
  local live=$1 out=$2 cfg=$3 b h
  [ -e "$out/code" ] && return 0
  # a config that cannot be read or parsed is refused with the reader's own message (r12-ops-03); valid JSON without
  # m1b_dirs keeps the "names no m1b_dirs" refusal
  b=$("$CL_PY" -I -c 'import json, sys
d = json.load(open(sys.argv[1])).get("m1b_dirs")
print(d[0] if isinstance(d, list) and d and isinstance(d[0], str) else "")' "$cfg" 2>&1) || {
    echo "refused: config $cfg unreadable: $(printf '%s' "$b" | tail -n 1)"; return 1; }
  [ -n "$b" ] || { echo "refused: config $cfg names no m1b_dirs (the batch whose frozen code the collection runs)"; return 1; }
  [ -f "$b/code.sha256" ] || { echo "refused: $b/code.sha256 missing (the batch has not frozen its code)"; return 1; }
  h=$(cl_code_hash "$live") || { echo "refused: cannot hash $live"; return 1; }
  [ "$h" = "$(cat "$b/code.sha256")" ] || { echo "refused: $live (sha256 $h) differs from the batch's frozen code" \
    "($b/code.sha256: $(cat "$b/code.sha256")); start from $b/code/scripts/automate/"; return 1; }
  echo "code $live = the frozen code of $b (sha256 $h)"
}

cl_prep() {
  local out=$1 cfg=$2 rc
  mkdir -p "$out/logs"
  local log=$out/logs/prep_$(date +%s).log
  cl_isaac_env
  # printed before the redirected call, so that a start waiting on the caps does not look hung (r10-ops-chk-02)
  echo "preflight in a capped scope (a caps refusal is retried every 60 s for up to ${QX_RETRY_MAX_S:-3600} s); progress: $log"
  qx_run_retry prep "$(basename "$out")" 4G 100% 0 bash "$out/code/scripts/automate/collect_prep.sh" "$out" "$cfg" > "$log" 2>&1
  rc=$?
  echo "exit $rc $(date)" >> "$log"
  if [ $rc -ne 0 ]; then
    echo "refused: preflight exit $rc (log $log):"; grep -E '^(refused|REFUSED)' "$log" | tail -3
    return 1
  fi
  grep -E '^PREFLIGHT OK' "$log"
}

cl_static_budget() {    # N MEM CPU GPU: N x each <= the caps (otherwise later workers could never start)
  local n=$1 mem=$2 cpu=$3 gpu=$4 b
  b=$(qx_to_bytes "$mem") && [[ "$cpu" =~ ^[0-9]+%$ ]] && [[ "$gpu" =~ ^[0-9]+$ ]] ||
    { echo "refused: resources '$mem $cpu $gpu' unreadable"; return 1; }
  if [ $(( n * b )) -gt $QX_CAP_MEM_BYTES ] || [ $(( n * ${cpu%\%} )) -gt $QX_CAP_CPU_PCT ] ||
     [ $(( n * gpu * 1000000000 / 1048576 )) -gt $QX_CAP_GPU_OURS_MIB ]; then
    echo "refused: $n workers x ($mem, $cpu, $gpu GB) exceed the caps (180 GB / 4800 % / 72 GB since 2026-11-25)"; return 1
  fi
}

cl_watch_ok() {    # an active qingxin-watch-* scope and watch.alive touched within the last 180 s
  local m
  systemctl --user list-units --type=scope --state=active --no-legend --plain 'qingxin-watch-*' | grep -q . || return 1
  m=$(stat -c %Y "$QX_RUN_DIR/watch.alive" 2>/dev/null) || return 1
  [ $(( $(date +%s) - m )) -le 180 ]
}

cl_watch_code() {  # WARNING when the live watcher's code differs from <code>'s qx_watch.py (r4-ops-08); always 0
  local want have
  want=$(sha256sum < "$1/scripts/automate/qx_watch.py" 2>/dev/null | cut -c1-64)
  have=$(cat "$QX_RUN_DIR/watch/code.sha256" 2>/dev/null)
  if [ -z "$want" ] || [ "$want" != "$have" ]; then
    echo "WARNING: the live watcher runs qx_watch.py sha256 ${have:-unknown}, $1 has ${want:-none};" \
         "restart it by its unit name (systemctl --user stop <qingxin-watch-...scope>; the next launcher starts it)"
  fi
  return 0
}

cl_watch_ensure() {
  local dir=$1 code=$2 key i
  mkdir -p "$QX_RUN_DIR/watch"
  key=$(echo -n "$dir" | sha256sum | cut -c1-12)
  echo "$dir" > "$QX_RUN_DIR/watch/$key.dir"
  touch "$QX_RUN_DIR/watch/wanted"                 # the watcher does not exit while launchers ask for it
  cl_watch_ok && { cl_watch_code "$code"; return 0; }
  if ! $CL_TMUX has-session -t =qingxin-watch 2>/dev/null; then
    $CL_TMUX new -d -s qingxin-watch \
      "bash -c 'source $code/scripts/automate/caps.sh; qx_run watch main 256M 5% 0 /usr/bin/python3 -u $code/scripts/automate/qx_watch.py >> $QX_RUN_DIR/watch.log 2>&1'" ||
      { echo "WARNING: the watcher did not start (tmux)"; return 1; }
  fi
  for i in $(seq 1 30); do
    cl_watch_ok && { echo "watcher running (tmux -L qingxin session qingxin-watch, log $QX_RUN_DIR/watch.log)"
                     cl_watch_code "$code"; return 0; }
    sleep 2
  done
  echo "WARNING: no active qingxin-watch-* scope with a fresh $QX_RUN_DIR/watch.alive after 60 s (log $QX_RUN_DIR/watch.log)"
  return 1
}
# AutoDL container (no systemd, ours alone): the AutoDL backend of these functions
[ -d /root/autodl-tmp ] && source "$(dirname "${BASH_SOURCE[0]}")/autodl_overrides.sh"
