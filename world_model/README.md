# world_model

Latent world model and flow-matching policy for plug insertion with unseen port shapes and unseen
grasps, on Chongyu's ManiSkill task `WMInsertionChargerRandomized-v2` (used unchanged).

Only real-robot information is used as model input or label: wrist RGB-D (D405 default stream),
camera intrinsics, hand-eye mount, joint angles and forward kinematics, commanded joint targets, and
one insertion per grasp (its FK hand pose). Simulator truth is used only by the demonstration expert
(standing in for a teleoperator) and for evaluation (`eval_*` arrays).

The record of every run and decision is in [docs/](docs/README.md); the current design is
[docs/design_v3.md](docs/design_v3.md).

## Layout

```
world_model/
  pyproject.toml            package `wm` (pip install -e world_model)
  configs/
    data/                   collection
    v3/                     current: vae, adapter, wm_s, wm_b, policy, baseline
    archive/v1, archive/v2  configs of the recorded earlier runs
  scripts/
    data/                   collect, summarise, inspect_trajectory, build_index, compute_ee
    train/                  train_vae, encode_latents, train_adapter, train_wm, train_policy
    eval/                   eval_policy (closed loop in simulation)
    analysis/               check_latents, check_world_model
    pipelines/              pipeline_v3.sh (the whole v3 run, with the gate before the world models)
    smoke_v3.py             end-to-end smoke test on a small slice of the data
  src/wm/
    env/                    robot access (float-depth twin camera), sampling, demonstration expert
    data/                   schema, recorder, store, index/splits, kinematics, conditions, datasets
    collect/                collection plan and parallel runner
    models/                 encoder, vae, cdit, adapter, policy, baseline_policy, perception
    training/               common, losses, vae, encode, adapter, wm, policy
    eval/                   agents and closed-loop rollouts
    analysis/               probes (latent), wm_checks (world model)
    third_party/nwm/        NWM CDiT + diffusion, vendored unchanged (CC BY-NC 4.0)
    geometry.py             SE(3) helpers, pose losses
  tests/                    unit tests (CPU, no simulator)
  docs/                     dated records + results/
```

Depends on `alignment_field/` (sibling folder) for the environment adapter and the port-axis fit;
see `src/wm/_paths.py`.

## Running (rs2)

```bash
source /mnt/storage2/qingxin/env.sh && cd $Q/code/ManiSkill
python -m pytest -q world_model/tests
python world_model/scripts/smoke_v3.py --out $Q/runs/smoke_v3
tmux new-session -d -s qingxin-v3 "bash world_model/scripts/pipelines/pipeline_v3.sh"
```
