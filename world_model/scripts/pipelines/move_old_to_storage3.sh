#!/bin/bash
# Move our obsolete results (v1-v5) from the nearly full shared NVMe disk to the archive disk, into /mnt/storage3/qingxin
# (user, 2026-11-13: "挪过去吧，还是qingxin的文件夹"). Copy, verify, only then remove the source. Nothing is deleted.
set -uo pipefail
Q=/mnt/storage2/qingxin; A=/mnt/storage3/qingxin
mkdir -p $A/runs $A/data
note() { echo "$(date +%m-%d\ %H:%M) $*"; }
move() {                     # move <kind: runs|data> <name>
    local src=$Q/$1/$2 dst=$A/$1
    [ -e "$src" ] || return 0
    ionice -c3 nice -n 15 rsync -a "$src" "$dst/" || { note "COPY FAILED $1/$2 -- source kept"; return 1; }
    local left; left=$(rsync -a -n -i "$src" "$dst/" | wc -l)
    local b1 b2; b1=$(du -sb "$src" | cut -f1); b2=$(du -sb "$dst/$2" | cut -f1)
    local n1 n2; n1=$(find "$src" -type f | wc -l); n2=$(find "$dst/$2" -type f | wc -l)
    if [ "$left" -eq 0 ] && [ "$n1" -eq "$n2" ] && [ "$b1" -le "$b2" ]; then
        rm -rf "$src"
        echo "$(date +%Y-%m-%d) $1/$2 -> $dst/$2 ($n1 files, $b1 bytes)" >> $Q/$1/MOVED_TO_STORAGE3.txt
        note "moved $1/$2 ($n1 files, $((b1 / 1048576)) MB)"
    else
        note "VERIFY FAILED $1/$2 (differences $left, files $n1 / $n2, bytes $b1 / $b2) -- source kept"
    fi
}
note "start: storage2 free $(df --output=avail -BG /mnt/storage2 | tail -1), storage3 free $(df --output=avail -BG /mnt/storage3 | tail -1)"
cd $Q/runs
for d in *; do
    case "$d" in
        joint_v6*|checks_v6|nwm_asis|MOVED_TO_STORAGE3.txt) ;;        # v6 and the running NWM baseline stay
        *) move runs "$d" ;;
    esac
done
for d in wm_v4e_latents wm_v4ex_latents; do move data "$d"; done    # outputs of the v4e autoencoder, used by nothing since v5
note "done: storage2 free $(df --output=avail -BG /mnt/storage2 | tail -1), storage3 free $(df --output=avail -BG /mnt/storage3 | tail -1)"
