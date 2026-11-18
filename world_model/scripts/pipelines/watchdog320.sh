#!/bin/bash
# Protects the other users of the server, every 15 s:
#  - every job of ours registered in $Q/run/qx320 is held to its RAM cap and its declared GPU memory (as the guard in
#    lib320.sh does; this covers jobs whose launching shell is gone);
#  - when the machine's memory pressure reaches 20 %, stop OUR newest job, one every 45 s until it is below 20 %.
# Only units that our own launcher registered are ever stopped, by name.
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
echo "$$" > "${Q320_STATE:?}/.watchdog_lists"     # this watchdog's lib knows GPU lists (q320_watchdog_lists)
q320_log "watchdog started"
while true; do
  for f in "${Q320_STATE:?}"/*.state; do
    [ -e "$f" ] || continue
    u=$(basename "$f" .state)
    [ "$(q320_state "$u")" = active ] || continue
    read -r g m r < "$f"
    q320_check "$u" "$r" "$g" "$m"
  done
  p=$(q320_pressure)
  if [ "${p:-0}" -ge 20 ]; then
    newest=""; t=0
    for f in "${Q320_STATE:?}"/*.state; do
      [ -e "$f" ] || continue
      u=$(basename "$f" .state)
      [ "$(q320_state "$u")" = active ] || continue
      s=$(systemctl --user show -p ActiveEnterTimestampMonotonic --value "$u.scope")
      if [ "${s:-0}" -gt "$t" ]; then t=$s; newest=$u; fi
    done
    if [ -n "$newest" ]; then
      q320_log "watchdog: memory pressure $p %, stopping $newest"
      systemctl --user stop "$newest.scope"
      sleep 45
      continue
    fi
  fi
  sleep 15
done
