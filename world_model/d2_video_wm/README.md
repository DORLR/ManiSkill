# `SegFormer-B1 + ST-DiT` (`world_model/d2_video_wm/`)

Built directly on top of `d1-gap-fix` (Changhao's `world-model` branch + test-time `gap` readout fix, no `self.L` dependency at inference).

This branch provides the **`SegFormer-B1 + ST-DiT`** world model (`--arch d2b_segformer`: pretrained `nvidia/mit-b1` hierarchical FPN + float32 depth stem + 4-frame spatial-token fusion + 4-layer AdaLN-Zero `STDiTDynamics`, `44.9M` total / `41.5M` trainable params), along with ablation presets (`d2b_resnet18`, `d2a_segformer`, `d2c_vjepa2_vitl`, etc.).

## File Map

| File | Purpose |
| :--- | :--- |
| `video_wm320.py` | Core `VideoWM320` model: `SegFormerB1FPNTokenizer` (`L576`), `STDiTBlock` (`L1262`), `STDiTDynamics` (`L1312`), `SpatialTokenReadout` (`L1465`), `SpatialReadoutHead` (`L1526`), and `ARCH_PRESETS`. Preserves the full `WM320` API (`observe`, `enc.tokens`, `enc.fuse`, `next`, `reward`, `Q`, `V`, `act_chunk`, `_done`, `_offset`) by packing `(K, D_z) -> (K * D_z)` at the external boundary. |
| `train_video_wm320.py` | Unified offline + online (`--online <pool> --online-frac 0.5`) trainer with single-GPU and multi-GPU `torchrun` DDP (`model.no_sync()`), gradient accumulation, live `B1` factual validation, and `SIGTERM` atomic checkpointing. |
| `eval_video_wm320_compat.py` | Wrapper that transparently loads `video_wm320-1` checkpoints inside unmodified `run_wm320_attempts.py`, `counterfactual320.py`, `fidelity_factual320.py`, and `speed320.py`. |
| `compat_hook/sitecustomize.py` | Subprocess hook injected by `eval_video_wm320_compat.py` so worker processes automatically dispatch `video_wm320-1` checkpoints to `VideoWM320`. |
| `test_d2_shapes_and_memory.py` | Fast CPU/GPU unit test covering forward/backward shapes, packed latent compatibility, and gradient flow. |

## Where to Modify `SegFormer-B1 + ST-DiT`

Inside `world_model/d2_video_wm/video_wm320.py`:
- **`SegFormerB1FPNTokenizer` (`L576`)**: loads pretrained `nvidia/mit-b1` (`transformers.SegformerModel`), fuses multi-scale FPN features with the `320×320` float32 depth stem, and produces spatial tokens.
- **`STDiTBlock` (`L1262`) & `STDiTDynamics` (`L1312`)**: 4-layer action-conditioned spatial-token Transformer (`Z_t ∈ R^{K×D_z}, a_t -> Z_{t+1}`, default `K=64 [8×8]`, `D_z=64`, AdaLN-Zero action conditioning + per-token SimNorm).
- **`SpatialTokenReadout` (`L1465`) & `SpatialReadoutHead` (`L1526`)**: multi-query attentive pooling from `Z_t` into TD-MPC2 `Q`, `V`, `reward`, `done`, `act_chunk`, and `offset (gap)` heads.
- **`ARCH_PRESETS`**: default hyperparameters for `d2b_segformer` (all overridable via `--latent-grid`, `--token-dim`, `--dit-dim`, `--dit-layers`, `--dit-heads`, `--segformer-model-id`).

## Quick Start (`SegFormer-B1 + ST-DiT`)

### 1. Environment & Pretrained Weights
```bash
export PYTHONPATH="$PWD:$PWD/alignment_field:$PWD/world_model/src:$PWD/world_model/d2_video_wm:/path/to/tdmpc2_orig/tdmpc2:$PYTHONPATH"
export HF_HOME="/path/to/hf_cache"  # caches nvidia/mit-b1 (~55 MB)
```

### 2. Shape & Compatibility Unit Test
```bash
python world_model/d2_video_wm/test_d2_shapes_and_memory.py
```

### 3. Training (`--arch d2b_segformer`, effective batch `128 WM / 256 BC`)
```bash
# Single-GPU 40GB+ offline (e.g., H100 MIG / A100 / PRO 6000)
python world_model/d2_video_wm/train_video_wm320.py \
  --data /path/to/wm_v1_fast \
  --out /path/to/runs/segformer_b1_stdit \
  --arch d2b_segformer --segformer-model-id nvidia/mit-b1 \
  --offset-mode gap --offset-weight 1.0 --shift-px 0 --shaping 0.3 \
  --wm-batch 16 --bc-batch 32 --accum 8 \
  --steps 25000 --save-every 2500 --eval-every 2500 --seed 0 --device cuda

# Single-GPU 24GB online round (e.g., RTX 4090 / 5090 Laptop; ~14.2 GB peak VRAM)
python world_model/d2_video_wm/train_video_wm320.py \
  --data /path/to/wm_v1_fast \
  --out /path/to/runs/segformer_b1_stdit_online \
  --arch d2b_segformer --segformer-model-id nvidia/mit-b1 \
  --online /path/to/online_pool --online-frac 0.5 \
  --offset-mode gap --offset-weight 1.0 --shift-px 0 --shaping 0.3 \
  --wm-batch 8 --bc-batch 16 --accum 16 \
  --steps 29500 --save-every 500 --eval-every 1500 --seed 0 --device cuda --resume

# 2-GPU 24GB DDP online round (exact same effective batch 128/256)
torchrun --standalone --nproc_per_node=2 world_model/d2_video_wm/train_video_wm320.py \
  --data /path/to/wm_v1_fast \
  --out /path/to/runs/segformer_b1_stdit_online \
  --arch d2b_segformer --segformer-model-id nvidia/mit-b1 \
  --online /path/to/online_pool --online-frac 0.5 \
  --offset-mode gap --offset-weight 1.0 --shift-px 0 --shaping 0.3 \
  --wm-batch 8 --bc-batch 16 --accum 8 \
  --steps 29500 --save-every 500 --eval-every 1500 --seed 0 --device cuda --resume
```

### 4. Closed-Loop Evaluation / Online Collection
Prefix `run_wm320_attempts.py` with `eval_video_wm320_compat.py run`:
```bash
python world_model/d2_video_wm/eval_video_wm320_compat.py run \
  world_model/scripts/online/run_wm320_attempts.py \
  --grasps /path/to/wm_v1_grasps \
  --ckpt /path/to/runs/segformer_b1_stdit_online/r1_step29500.pt \
  --mode wm --inside-mm 1.5 --split heldout \
  --starts 2 --workers 6 --gpus 0 --torch-mib 2560 --probe-max 2 \
  --out /path/to/evals/r1_heldout_wm
```

*(Note: ablation encoders `d2b_resnet18`, `d2a_segformer`, `d2a_resnet34`, and `d2c_vjepa2_vitl` remain selectable via `--arch`.)*
