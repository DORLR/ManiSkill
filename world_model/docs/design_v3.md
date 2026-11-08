# Design v3 (current)

Decided with the user on 2026-11-08 after v1 / v2 (2026-11-07_v1.md, 2026-11-08_v2.md). Everything
is trained from scratch. Inputs and labels are real-robot quantities only.

## Interface (Ctrl-World's: the policy gives actions, the world model gives observations + state)

* Policy: (4 context latents, measured joints) -> 16 joint targets (execute 8, replan every 0.4 s).
* Robot state is never predicted by the world model. Joints come from a joint adapter (Ctrl-World's
  adapter for joint position targets): measured joints + 1.2 s tracking history + next 8 targets ->
  next 8 measured joints; hand poses by forward kinematics.
* World model: (4 context latents, robot-state / commanded-path condition, time shift) -> the latent
  at that time.

## Vision: autoencoder (models/encoder.py, models/vae.py)

* Input per frame, as the camera gives it: RGB and depth (pixels without a return keep depth 0; no
  completion, nothing dropped, no segmentation, no point cloud), intrinsics, camera pose (FK +
  hand-eye mount), hand pose (FK), port axes (board-plane fit of the first frame).
* Tokens: 23 x 23 patches of 14 px (the 320 px crop at DINOv2's native patch size): frozen DINOv2
  ViT-S/14 feature (DINOv3 is gated on Hugging Face for this account) + raw RGB patch + raw depth
  patch + the patch's viewing ray in port axes + position embedding; one geometry token (camera
  orientation in port axes, camera position relative to the hand). Self-attention (6 layers): the
  relations are learned. 64 learned queries read the result -> latent 64 x 8 (no joint token, no KL).
* Decoder: every patch's depth (+ a validity logit: where the camera has no return), RGB and DINO
  feature from the latent; pose head = the rigid motion the plug still needs to be inserted.
* Loss = depth L1 [mm] + validity BCE + 10 x RGB L1 + (1 - cos DINO) + reference-grid pose error [mm]
  + 2 x angle error [deg]. Pose label: FK of this grasp's one insertion (geometry.insertion_motion);
  the grasp parameters cancel in it (plug pose relative to its inserted pose).
  Reference grid: a fixed 5x5x5 grid (2 cm cube where the plug sits in hand-centred port axes), the
  same for all shapes; a ruler that turns a rigid-motion error into mm. Angle error: covers the small
  rotations the 2 cm grid barely sees (0.4 deg moves it by 0.14 mm).

## World model (models/cdit.py, training/wm.py): NWM's CDiT, used as in NWM

* Time shift (NWM's rel_t): each sample predicts a target at a shift drawn per sample, half +0.4 s
  (the step used with the policy), half uniform over +-16 s (NWM's range, including the past).
* Condition (data/conditions.py, 216 values, all relative to the current measured hand pose, in port
  axes): for each of the 4 context frames the measured and the commanded hand pose (blocked = the
  commanded pose moved, the measured one did not); the commanded hand path from now to the target time
  as 16 waypoints (NWM sums actions over the shift; a path keeps how it gets there, which matters for
  contact).
* Loss = diffusion loss + 0.02 x (grid error + 2 x angle error) between the pose read from the model's
  clean-latent estimate and from the true target latent, on diffusion steps t < 200.
* Context-noise augmentation (GameNGen): half the samples get Gaussian noise of level in [0, 0.1] on the
  context latents, the level is an input (0 at use time) -> robustness to its own predictions.
* Two sizes trained in parallel with identical data / losses / effective batch 256 / 300k steps:
  CDiT-S (49 M) and CDiT-B (~4x). The comparison shows whether capacity limits the accuracy.
* Model selection: mean over time shifts (0.4, 2, 4, 8, 16 s ahead, 4 s back) of the pose score of a
  sampled prediction on ood_val.

## Policies

* Latent FM (models/policy.py): latents + joints. Budget: batch 256, lr 3e-4, 100k steps.
* Latent-free baseline (models/baseline_policy.py): the same encoder trained end to end by the policy
  loss + joints; same budget (batch 32 x 8 accumulation).

## Checks and evaluation (fixed before results)

* Latent checks (analysis/probes.py), same tests as v1/v2: jam-vs-insert classifiers, offset
  regressions. Gate before the world models: pose head angle R^2 >= 0.54 (v2), latent >= 0.70, jam
  AUC >= 0.84.
* World-model checks (analysis/wm_checks.py): accuracy per time shift vs "nothing changes"; jam-vs-insert
  separation from the push start at 2 s / 4 s (does it foresee the jam); 16 s imagined rollouts with the
  adapter (pose error, adapter joint error, hand error, latent RMS drift).
* Closed loop (eval/rollout.py): 60 s, success within 1 mm of full depth, seeded sampling, 4 grasps per
  shape and condition x 2 starts, ood_val; test ports A-D once at the end.
* Still to be built and run after the world models: (1) ranking FM candidates with the world model and
  CEM over hand paths initialised from FM samples (NWM's planning), compared with FM alone on the same
  episodes; (2) Ctrl-World's checks: policy evaluation in imagination vs in simulation (do the
  rankings agree), and post-training the FM on successful imagined rollouts (success read by the pose
  head). RL fine-tuning, which the supervisor requires, is done in simulation, separately.

## Deviations from the papers (and why)

| | paper | here | reason |
|---|---|---|---|
| world-model size | NWM main results CDiT-XL (~1 B) | S and B compared | ~0.5 M training windows; XL not tried |
| latent | NWM: image VAE (pixels) | own autoencoder: depth/RGB/DINO patches + pose head | user's design: a task latent shared with the policy |
| action | NWM: navigation action summed over the shift | commanded hand path, 16 waypoints, + measured/commanded context poses | arm with contact; path and blocked state matter |
| context noise | not in NWM | GameNGen-style, level as input | long imagined rollouts |
| views / memory | Ctrl-World: multi-view, pose-conditioned memory retrieval | wrist camera only, 4 context frames + direct long shifts | single wrist camera by design; memory to be added only if the 16 s checks need it |
| judging imagined success | Ctrl-World: humans | pose head reading | automatic, shape-independent |
