# World-model work log

All runs on rs2, GPU 1, under `/mnt/storage2/qingxin` (`$Q`). Splits: train (generated shapes),
ood_val (13 held-out generated shapes: G46 G47 G48 G55 G65 ... G145), test (ports A-D, untouched).
Times are Toronto time.

## 2026-11-07

### Data (wm_v1, collected 2026-11-06)

1240 grasps, 13,701 trajectories, 19 GB (`$Q/data/wm_v1`). Train split, trajectories by kind/outcome:

| kind | outcome | train trajectories |
|---|---|---|
| insert | inserted | 1918 |
| perturbed | inserted | 960 |
| reference | inserted | 499 |
| contact | inserted | 773 |
| contact | jammed | 1645 |
| random | done | 483 |
| any | expert failed | ~200 |

- FM policy trains on the successful expert kinds only (reference / insert / perturbed): 3,377
  trajectories, 393,974 samples.
- World model trains on everything: 6,476 trajectories, 527,800 windows.

### Models (v1)

| model | run | selected | validation (ood_val) |
|---|---|---|---|
| VAE (RelationTokenizer, 64 x 8 latent, KL 0.001) | `runs/vae_v1` | step 54k | plug-point 0.64 mm, joints 25.5 mrad |
| latent FM policy | `runs/policy_v1` | step 95k | action error 0.69 mrad |
| CDiT world model | `runs/wm_v1` (stopped at 88.6k) | `runs/wm_v1/last.pt` = step 85k | see below |

CDiT one-step (0.4 s) prediction vs "copy the last latent", decoded by the VAE:

| step | plug translation | rotation | joints |
|---|---|---|---|
| 70k | 0.45 mm (copy 0.72) | 0.23 deg (copy 0.21) | 4.5 mrad (copy 2.7) |
| 75k | 0.45 (0.72) | 0.20 (0.20) | 3.8 (2.8) |
| 80k | 0.42 (0.72) | 0.21 (0.17) | 4.4 (2.7) |
| 85k | 0.49 (0.74) | 0.36 (0.33) | 3.9 (2.9) |

Translation is better than copying; rotation and joints never beat copying; no progress after 70k.
Training was stopped at 88.6k (user decision) to free the GPU.

### Closed loop, 60 s episodes, ood_val (156 grasps x 2 starts)

- Latent FM alone (`runs/eval_policy_v1_ood_60s`): 98/288 = 34% (12 grasps skipped: the expert
  could not insert). By condition: train 40%, pitch interp 38%, pitch extrap 31%, pos interp
  36%, pos extrap 36%, combined 24%.
- FM + world model, test-time selection (`runs/eval_wm_v1_ood_60s`, WM snapshot step 75k): every
  0.4 s the FM samples 16 chunks, the WM predicts 0.8 s ahead, the chunk with the smallest
  decoded "distance to inserted" is executed. Stopped after 47 grasps (user decision): on the
  41 grasps comparable with FM alone (82 episodes, shapes G105-G145) **27 vs 28 inserted, no
  difference**, in every condition.
- This selection does not change the FM; it can only help if a good candidate exists and the WM
  can recognise it. Videos showed neither holds at the hole face (below).

### Videos (desktop folder `videos_policy_v1`)

- `wm_guided_*`: real episodes with the WM's 0.8 s predictions. 0.8 s predictions track the real
  latent closely (median 0.6 mm), but when the plug is stopped at the hole face all 16 candidates
  are predicted the same (~20 mm left): nothing to choose from.
- `dream_*`: fully imagined rollouts from one real frame (policy + WM only; the same targets
  replayed in the simulator for comparison; the WM never sees the replay). The first ~4 s agree;
  then the real plug stops ~20 mm from inserted while the distance read from the imagined latents
  keeps decreasing to 0. Gap after 2 / 8 / 16 s: 3.9 / 7.1 / 22.5 mm (train grasp), 4.8 / 1.1 /
  17.7 (pos extrap), 3.1 / 6.3 / 14.1 (pitch extrap). The WM has no image decoder; imagined
  latents are shown as decoded plug/port points.
- Caveat: decoded readings of imagined latents are only trustworthy while the latents stay near
  the encoder's distribution; not yet checked.

### Offline checks (`scripts/probe_contact.py`, `scripts/probe_offset.py`, `runs/probe_contact_v1`)

Contact trials: plug moved to a small offset (0-3 mm, 0-3 deg) at a 3 mm gap, then pushed
straight. Fit on train (2418 trials, 1645 jammed), report on ood_val (2429 trials, 1723 jammed).

1. Will this push jam? (balanced accuracy / AUC, MLP probe)

   | input at push start | balanced acc | AUC |
   |---|---|---|
   | simulator truth (offset), upper bound | 0.93 | 0.98 |
   | 4 context latents + next targets | 0.79 | 0.87 |
   | joints + next targets (no vision) | 0.72 | 0.79 |

   0.4 s and 0.8 s into the push the latent stays at AUC ~0.88; joints drop to ~0.6. Joints alone
   cannot regress the offset (item 2), so their 0.79 is a grasp/shape prior, not the offset.

2. Which part of the offset does the latent keep? (R^2 on ood_val; truth median |offset| per
   component: 0.58 mm lateral, 0.40 deg angles)

   | component | latent (MLP) | VAE pose-head reading (linear) | joints only |
   |---|---|---|---|
   | lateral y | 0.80 (0.26 mm err) | 0.70 | < 0 |
   | lateral z | 0.67 (0.38 mm) | 0.58 | < 0 |
   | roll | 0.43 | 0.36 | < 0 |
   | pitch | 0.74 | **0.02** | < 0 |
   | yaw | 0.61 | **0.19** | < 0 |

   The latent keeps lateral offset fairly well and angles partly; the pose head reads almost no
   pitch/yaw. Cause: the plug-point loss is in mm on a ~20 mm plug, so 0.4 deg moves its points by
   only ~0.14 mm and is barely penalised. The planner's cost uses this pose head, so it is blind to
   angle as well.

3. Does the WM predict the stall? One 0.4 s step from real context, 3000 push-phase windows:

   | | progress per 0.4 s (median) | windows with < 0.2 mm progress |
   |---|---|---|
   | inserted trials, real | 1.17 mm | 4.9% |
   | inserted trials, WM | 1.15 mm | 5.5% |
   | jammed trials, real | 0.67 mm | 35% |
   | jammed trials, WM | 0.94 mm | 21% |

   The WM knows jams are slower but predicts only ~60% of the stalls and ~40% too much progress;
   over many imagined steps this optimism compounds into "inserted" (the dream videos).

### Diagnosis

- Perception (VAE): small angle misalignment, which decides jam vs slide, is weakly kept in the
  latent and not readable by the pose head.
- Dynamics (WM): it does not see the signal of being blocked (commands advance, joints do not):
  only the next 8 targets are in its condition. Its loss weights all 64 tokens equally, so the
  few numbers that matter for the stall are a small part of it.
- Policy: trained on successes only; at the hole face its candidates are all "push forward".
  A WM can only improve this by training the policy in imagination, after the WM is right
  about contact.

### Changes (decided with the user) -- v2, everything trained from scratch

VAE v2 (`configs/vae_v2.json`, run `runs/vae_v2`, from scratch, 60k steps like v1):

    v1: Chamfer(plug) + Chamfer(port) + 1.0 plug-point + 0.01 joints + 0.001 KL
    v2: Chamfer(plug) + Chamfer(port) + 1.0 plug-point + 2.0 angle + 0.01 joints

- angle = angle between the predicted remaining rotation and the true one (from the same
  FK-of-one-insertion target as the plug-point loss; no new information). 0.1 deg costs as much
  as 0.2 mm.
- KL removed: the encoder is deterministic (LayerNorm + linear). LayerNorm bounds magnitudes and
  latents are rescaled to unit RMS before the WM, which is what the KL was for. v1 checkpoints
  still load through `models.vae.load_vae` (mean half of the Gaussian head) so the v1 policy and
  the v1 checks can be rerun; nothing is initialised from v1.
- 8 channels kept; widen only if the angle check fails.
- Selection on ood_val: plug-point + 2 x angle.

World model v2 (`configs/wm_v2.json`, run `runs/wm_v2`, from scratch, 200k steps like v1, on the
v2 latents `data/wm_v2_latents`):

1. Tracking history in the action condition (`data.datasets.action_condition`): next 8 targets
   relative to the measured joints (56 values, as before) + for each of the 24 control steps
   before (1.2 s) the commanded target minus the joints measured at that step (168 values; zeros
   before the trajectory starts). 224 inputs; the CDiT backbone is unchanged.
2. Predicted-pose loss: the model's clean-latent estimate and the true next latent are read by
   the frozen v2 pose head; loss = plug-region displacement difference (mm, a fixed 5x5x5 point
   grid where the plug is) + rotation difference (deg), only on diffusion steps t < 200 of 1000.
   Weight 0.02 (smoke test on v1 latents: pose error ~0.2, diffusion loss ~0.002, so the term is
   1-2x the diffusion loss; to be rechecked on v2 latents in the training log).
- Not done (by decision): contact-mode gate, oversampling contact windows. Added only if 1+2 do
  not fix the stall prediction.
- The WM-guided agent (`eval/planner.py`) does not yet feed the history; it refuses a history WM.

Order (`scripts/pipeline_v2.sh`, tmux `qingxin-v2`): VAE v2 -> encode all trajectories -> offline
checks on v2 latents -> gate (pose head must read angles: mean R^2 roll/pitch/yaw > 0.5, v1 0.19;
latent at least as good as v1, >= 0.59) -> CDiT v2. If the gate fails the pipeline stops before
the world model.

Acceptance after training:
- VAE v2: angle R^2 well above v1 (target > 0.85), jam AUC towards 0.98, plug-point not worse than
  0.64 mm.
- WM v2: `probe_contact.py --skip-probes --latents data/wm_v2_latents --vae runs/vae_v2/best.pt
  --wm runs/wm_v2/best.pt`: stalled fraction on jammed trials towards the real one (v1: 21% vs
  35%), progress towards the real one (v1: 0.94 vs 0.67 mm), inserted trials unchanged; plus a
  5-step imagined-rollout error. Compared against v1 with the same checks.

A first attempt at 23:58 fine-tuned both models from the v1 checkpoints. It was stopped and
deleted (user decision): a fine-tuned model keeps the old latent space and habits, so its result
cannot show whether the new design works, and a WM trained on v1 latents is useless once the VAE
changes.

### Latent-free FM baseline (`configs/image_policy_v1.json`, `runs/image_policy_v1`)

Same visual encoder architecture as the VAE (DINOv2 frozen + PointNet + plug/port cross-attention
+ relation queries), same 4-frame context, same transformer + flow head and action chunk as the
latent FM; no latent bottleneck, no plug-point / Chamfer supervision; encoder trained from
scratch by the BC loss. Slow (~0.5 s/step) because 128 images per step go through DINOv2 and
PointNet. Validation action error at 5k steps: 3.13 mrad (latent FM at 5k: 1.18). To be evaluated
with the same 312 episodes as the latent FM when training ends.

### Housekeeping

- Deleted on rs2: smoke/trial runs (`runs/smoke`, `*_trial`, `eval_*smoke`, `image_policy_smoke`),
  the NaN debug batch (`runs/nan_batch.pt`, from the v1 VAE NaN incident, fixed by LayerNorm +
  lr 1e-4), the diverged CDiT checkpoints (`runs/wm_v1_diverged`: config and log kept; bf16
  divergence at steps 4100-5100, fixed by fp32/TF32 + warm-up), temporary frame images, the
  fine-tuning attempt (`runs/vae_v2`, `runs/wm_v1b`, `configs/wm_v1b.json`, warm-start code) and
  the duplicate step-85k checkpoint.
- Kept: all logs (`$Q/logs`), all evaluation results, videos, probe reports, the WM snapshot used
  by the stopped evaluation (`runs/wm_v1_snapshot.pt`, step 75k), `cache/tmp/envtxt.bak` (backup
  of Ramy's `~/.conda/environments.txt` before the one-line fix).
- Not committed since `fecbfad` (branch `world-model`): everything after the policy pipeline
  (planner, rollout changes, EMA warm-up, TF32, image baseline, video and probe scripts, today's
  VAE / WM changes). Waiting for the user's go.

### Expected problems with this round

1. Angle term vs. the other losses: chamfer / joints may get slightly worse; plug-point must not.
2. The angle labels themselves: R* comes from FK of one expert insertion per grasp; if the
   inserted pose has play of a few tenths of a degree, the label noise is of the size we try to
   learn.
3. Without KL the latent space may be less smooth; imagined multi-step rollouts can drift into
   regions the decoder never saw. Check with the dream videos and the 5-step error.
4. History may be used as a shortcut: in jammed trials the gap grows after contact, so the WM may
   learn "stall once already stalled" rather than predicting the jam before it happens. Check the
   push-start window separately from later windows.
5. The pose-loss weight 0.02 was measured on v1 latents; with an angle-aware pose head the term may
   be larger and dominate the diffusion loss. Watch train_pose_err vs train_diffusion_loss.
6. Single-step gains may not carry over to multi-step imagination (compounding errors are not
   trained against).
7. Closed loop: the WM-guided agent needs per-step history from the rollout loop before v2 can be
   evaluated with the policy.
