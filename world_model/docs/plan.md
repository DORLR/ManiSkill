# World model for plug insertion: plan

## 1. Task
A Panda hand holds a charger plug (rigid, no slip). From a far start in front of the
board (tip 6-18 mm from the hole face, lateral up to 10 mm, each rotation up to 6 deg),
insert it. Generalise to unseen port shapes (test: Chongyu's A, B, C, D; training:
procedurally generated shapes) and unseen grasps (pitch -60..-20 deg, grasp position
-30..-10 mm, with interpolation and extrapolation bands held out). A flow-matching
policy (FM) will later be combined with the world model; this document covers the
data and the world model.

## 2. Data (wm_v1)
Collected once; splits are decided afterwards from metadata.

* Shapes: A-D and the 146 buildable generated shapes G00-G149 (G19, G42, G45, G49 cannot be built).
* Grasps: per generated shape 4 inside the training bands + 4 in interpolation/extrapolation
  bands (cycling over the five band conditions); per test port 3 per condition (18).
* Per grasp: one reference insertion (defines H_star and the insertion axis by FK), then
  4 expert insertions from far starts, 2 with stochastic perturbation and recovery,
  5 contact trials (aligned at 3 mm, offset 0-3 mm / 0-3 deg, straight push until inserted
  or jammed; 15 % start aligned), 1 random motion in front of the board.
* Expert: Chongyu's scripted phases (`_approach`, `_align`, `_insert`) started from the
  current pose; moves between trajectories are planner motions and are not recorded.
* Recorded: wrist RGB (JPEG q95) and depth (uint16, 0.1 mm) 320 x 320 crops at 5 Hz,
  camera intrinsics/extrinsics, joint angles, FK hand pose, commanded joint targets at 20 Hz.
  Simulator truth only as `eval_rel` (charger pose in the goal frame).
* Depth: ManiSkill's camera stores positions as int16 mm; a twin camera with SAPIEN's
  default shader on the same mount gives float depth (measured int16 error: mean 0.49 mm,
  max 1.0 mm). D405 minimum range 7 cm: the plug housing is closer and has no depth; the pins do.

Expert audit before collection (2026-11-07): from the reset spawn Chongyu's expert failed
on 127 attempts by its 800-step limit (lateral transport at 3 mm/s), concentrated in
position extrapolation; a straight kinematic push without his tracking compensation
inserted 24/489 (0.36 mm steady tracking error). With the wrapper: smoke grasps inserted
from far starts 13/13, perturbed 6/6; contact trials inserted at offsets <= 0.5 mm and
jammed at 0.5-2 mm.

## 3. Splits
test = A-D. ood_val = held-out generated shapes G46, G47, G48, G55, G65, ..., G145 (all
grasps) and, on training shapes, every grasp outside the training bands. train = the rest.
Model selection and early stopping use ood_val only; A-D is evaluated once at the end.

## 4. Model
Stage 1, token VAE (`wm.models.vae`):
frame -> points in hand-centred port axes (board-plane RANSAC + gravity, per trajectory)
-> plug / port split by board depth -> per-point xyz, RGB, frozen DINOv2 -> PointNet++
set abstraction (plug tokens, port tokens) -> plug<->port cross-attention -> 63 learned
relation queries + 1 joint token -> per-token Gaussian latent, 8 channels.
Decoder: plug/port points (Chamfer), joints (L1), plug pose head trained with the
plug-point loss: the predicted rigid motion is applied to the observed plug points and
compared with where FK of the one insertion puts them (no CAD, no tip label; points far
from the hand, i.e. the pins, carry the largest rotation errors).

Stage 2, world model (`wm.models.cdit`): NWM's CDiT (vendored unchanged) over the
8 x 8 latent grid (patch size 1), 4 context latents 0.4 s apart, action = the 8
commanded joint targets of the next 0.4 s relative to the measured joints; NWM's
Gaussian diffusion (1000 steps, linear, learned sigma), EMA 0.9999.

## 5. Evaluation (ood_val during development, A-D once)
* VAE: plug-point error (mm), Chamfer (mm), joint error (mrad).
* World model: predicted next latent vs the true next latent, both decoded: plug-pose
  difference (deg, mm) and joint difference (mrad), against "copy the last latent".
* Contact outcome: for contact trials, whether the predicted joints show the plug
  advancing (inserted) or stopping (jammed) as in reality.

## 6. Change log
* 2026-11-07: package created; expert audit; float-depth twin camera; wm_v1 collection started.
