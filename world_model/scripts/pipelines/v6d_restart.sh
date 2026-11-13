#!/bin/bash
# Restart of joint_v6d (v6c + the approach set, from scratch) on GPU 0, and later checks of joint_v6c.
# The first start (11-13 00:42, GPU 1, by approach_set.sh) died within a minute with CUDA out of memory: GPU 1 was
# filled by other users' jobs next to joint_v6c (our process held 11 GB of its 24 GB cap when the card ran out), and
# the three check sessions of joint_v6d then ended too. Nothing of the design is changed here.
# Each of our sessions runs inside a systemd scope with a hard memory limit (shared machine).
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
P=$Q/code/ManiSkill/world_model/scripts/pipelines
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
used0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
[ "$used0" -lt 40000 ] || { stamp "FAILED: GPU 0 holds $used0 MiB -- joint_v6d not restarted"; exit 1; }
tmux has-session -t qingxin-v6d-joint 2>/dev/null && { stamp "FAILED: qingxin-v6d-joint already runs"; exit 1; }
[ -e $Q/runs/joint_v6d ] && mv $Q/runs/joint_v6d $Q/runs/joint_v6d_oom_1113_0042
[ -e $Q/logs/v6d_joint.log ] && mv $Q/logs/v6d_joint.log $Q/logs/v6d_joint_oom_1113_0042.log
LIM="systemd-run --user --scope -q -p MemoryMax"
tmux new-session -d -s qingxin-v6d-joint "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.25 $LIM=70G nice -n 10 python $S/train/train_joint.py --config world_model/configs/v6/joint_d.json --run $Q/runs/joint_v6d > $Q/logs/v6d_joint.log 2>&1"
tmux new-session -d -s qingxin-v6d-checks "GPU=0 $LIM=30G bash $P/v6_checks2.sh joint_v6d qingxin-v6d-joint 5000 10000 20000 30000 > $Q/logs/v6d_checks.log 2>&1"
tmux new-session -d -s qingxin-v6d-approach-at "GPU=0 $LIM=30G bash $P/v6_approach_at.sh joint_v6d qingxin-v6d-checks 5000 10000 20000 30000 > $Q/logs/v6d_approach_at.log 2>&1"
tmux new-session -d -s qingxin-v6d-horizons "GPU=0 $LIM=30G bash $P/v6_horizons_at.sh joint_v6d qingxin-v6d-checks 5000 10000 20000 30000 > $Q/logs/v6d_horizons_at.log 2>&1"
stamp "joint_v6d: the start of 00:42 died within a minute with CUDA out of memory (GPU 1 was full with other users' jobs; kept as runs/joint_v6d_oom_1113_0042) -- so the night gave no answer on the approach-set data. Restarted from scratch on GPU 0 (empty), same config; checks at 5000 / 10000 / 20000 / 30000: push, replay, approach, direct prediction"
tmux new-session -d -s qingxin-v6c-checks2 "GPU=0 $LIM=30G bash $P/v6_checks2.sh joint_v6c qingxin-v6-joint 50000 60000 > $Q/logs/v6c_checks2.log 2>&1"
tmux new-session -d -s qingxin-v6c-approach2 "GPU=0 $LIM=30G bash $P/v6_approach_at.sh joint_v6c qingxin-v6c-checks2 50000 60000 > $Q/logs/v6c_approach2.log 2>&1"
tmux new-session -d -s qingxin-v6c-horizons2 "GPU=0 $LIM=30G bash $P/v6_horizons_at.sh joint_v6c qingxin-v6c-checks2 50000 60000 > $Q/logs/v6c_horizons2.log 2>&1"
stamp "joint_v6c: checks added at 50000 (taken from current.pt at about step $(python -c "import torch; print(torch.load('$Q/runs/joint_v6c/current.pt', map_location='cpu', weights_only=False)['step'])" 2>/dev/null)) and 60000 (end of training)"
