# world_model

Latent world model for plug insertion with unseen port shapes and unseen grasps,
on Chongyu's ManiSkill task `WMInsertionChargerRandomized-v2` (used unchanged).

Only real-robot information is used as model input or label: wrist RGB-D (D405
default stream), camera intrinsics, hand-eye mount, joint angles (forward
kinematics), commanded joint targets, and one insertion per grasp (its FK hand pose).
Simulator truth is used only by the demonstration expert (standing in for a
teleoperator) and for evaluation (`eval_*` arrays).

## Layout

```
world_model/
  pyproject.toml          package `wm` (pip install -e world_model)
  configs/                collect_v1.json, vae_v1.json, wm_v1.json
  scripts/                command-line entry points (thin wrappers around wm.*)
    collect.py            collect the dataset (resumable)        summarise.py   dataset report
    inspect_trajectory.py arrays + image strip of one file        build_index.py splits + port axes
    train_vae.py          stage 1                                 encode_latents.py  VAE -> latents
    train_wm.py           stage 2                                 smoke_models.py    end-to-end smoke test
  src/wm/
    env/                  robot access (float depth twin camera), sampling, demonstration expert
    data/                 schema, recorder, store (npz + JPEG), index, splits, torch datasets
    collect/              plan (what to collect), episode (one grasp), runner (parallel)
    models/               perception (reused from alignment_field), tokenizer, vae, cdit
    training/             common, losses, vae, encode, wm
    third_party/nwm/      NWM CDiT + diffusion, vendored unchanged (CC BY-NC 4.0)
    geometry.py           SE(3) helpers
  tests/                  unit tests (no simulator needed)
  docs/plan.md            design, data, splits, evaluation, change log
```

Depends on `alignment_field/` (sibling folder) for the environment adapter and the
perception primitives; see `src/wm/_paths.py`.

## Pipeline

```bash
source /mnt/storage2/qingxin/env.sh && cd $Q/code/ManiSkill
python world_model/scripts/collect.py --config world_model/configs/collect_v1.json --out $Q/data/wm_v1 --workers 40
python world_model/scripts/summarise.py $Q/data/wm_v1
python world_model/scripts/build_index.py $Q/data/wm_v1
python world_model/scripts/train_vae.py --config world_model/configs/vae_v1.json --run $Q/runs/vae_v1
python world_model/scripts/encode_latents.py --vae $Q/runs/vae_v1/best.pt --data $Q/data/wm_v1 --out $Q/data/wm_v1_latents
python world_model/scripts/train_wm.py --config world_model/configs/wm_v1.json --run $Q/runs/wm_v1
python -m pytest -q world_model/tests
```
