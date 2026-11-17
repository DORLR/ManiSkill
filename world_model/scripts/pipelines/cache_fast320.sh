#!/bin/bash
# Write the fast copy of the demonstrations once (scripts/data/cache_wm_v1_fast.py). CPU only; resumable.
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
cd "$Q/code/ManiSkill" || exit 1
# <src>: a folder in wm_v1's layout whose index.json has paths relative to it (the full set, or a view with a subset)
q320_run_when_fits qx320-cache 48 - 0 cpu 0 "$Q/logs/cache_fast320.log" python world_model/scripts/data/cache_wm_v1_fast.py \
  --src "${1:?source folder}" --dst "$Q/data/wm_v1_fast" --pins "$Q/data/pin_length_by_port.json" --workers 16
