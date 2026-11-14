# NWM as it is, on our data (2026-11-13; the user: "原样跑一次")

Navigation World Models (facebookresearch/nwm, commit 3f6cd8e, cloned unchanged to `$Q/code/nwm_orig`) trained on our
scripted trajectories. Unchanged and run from that clone: `train.py`, `models.py` (CDiT), `diffusion/`, `misc.py`
(image transform), `distributed.py`, the frozen pretrained image VAE it loads (`stabilityai/sd-vae-ft-ema`), the
loss, the optimiser, the sampling of 4 context frames + 4 goals at random offsets within +-64 frames, the relative-time
condition. Nothing of our v1-v6 designs is in it: no ResNet encoder, no joint training, no hand-motion output, no
state bits, no label head, no history masking.

Ours, because their dataset code reads their navigation folders (2-D position + yaw):
- `plug_dataset.py`  a dataset class with the constructor and outputs of their `TrainingDataset`, reading our
  trajectory files. The three action numbers their model embeds are, for us, the displacement of the COMMANDED hand
  position from now to the goal time, in the commanded hand's own frame now (their "local coordinates"), divided by
  the mean step ("metric_waypoint_spacing") and scaled with per-step min / max, as they do. Commanded rotation is not
  among the three numbers. Images: the recorded wrist RGB, through their transform (224 x 224). No depth.
- `plug_stats.py`    computes that spacing and min / max from the training split -> `plug_data_config.yaml`.
- `run_plug.py`      puts the class in place of `datasets.TrainingDataset` and calls their `train.main`.
- `plug_cdit_b.yaml` their `nwm_cdit_xl.yaml` with our data and CDiT-B/2 instead of CDiT-XL/2 (one shared GPU; an
  XL checkpoint with optimiser is about 16 GB and the disk had 11 GB free). Their in-training evaluation needs the
  `dreamsim` package and is switched off (`--eval-every` beyond the run); evaluation is done by our own script.
