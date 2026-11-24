#!/bin/bash
# Let workers retry an assembly they skip:   bash collect_unblock.sh <out dir> <assembly id>
# Moves <out>/build_failures/<id>.json and <out>/deaths/<id>.json (the records that put it on the skip list, see
# logs/summary.json) to <dir>/cleared/<id>.<unix time>.json. Nothing is deleted.
set -u
[ $# -eq 2 ] || { echo "usage: bash collect_unblock.sh <out dir> <assembly id>"; exit 1; }
OUT=$(readlink -f "$1"); A=$2
[[ "$A" =~ ^[0-9]{5}$ ]] || { echo "assembly id '$A' must be 5 digits"; exit 1; }
n=0
for d in build_failures deaths; do
  f=$OUT/$d/$A.json
  [ -e "$f" ] || continue
  mkdir -p "$OUT/$d/cleared"
  mv "$f" "$OUT/$d/cleared/$A.$(date +%s).json" && n=$((n + 1)) && echo "moved $f to $OUT/$d/cleared/"
done
[ $n -gt 0 ] && echo "$(date '+%F %T') unblocked $A" >> "$OUT/logs/launcher.log"
[ $n -gt 0 ] || echo "nothing recorded for $A"
