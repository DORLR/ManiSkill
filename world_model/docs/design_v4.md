# Design v4 (decided with the user, 2026-11-10 night)

Why: the imagination check (2026-11-09_imagination_check.md) showed that the v3g world model was
optimistic on the FM policy's own states (it imagined the plug moving in while it was jammed; 54.5 %
imagined vs 12.9 % real insertions), that the pose readout could not tell a tilted plug from an inserted
one, and that sub-millimetre offsets near the hole were lost (the whole 320 x 320 image compressed into
64 x 8 numbers). v4 changes the world model only; the FM policy is unchanged (it is the actor).
Pure vision + proprioception (no force / torque, no socket pose). Control unchanged (the simulated arm is
a joint PD spring, 1-22 N/mm at the hand; the user kept it). Everything trained from scratch.

## 1. Vision: segmentation + two native-resolution crops + their relation

* Labels (perception/labels.py): the recorded classes + plug pins + socket openings, drawn by projecting
  the port geometry (ports.py: pin boxes in the plug frame, opening rectangles = pins + 0.5 mm clearance
  on the socket face) with the goal pose read once per grasp from the simulator (an annotator's outline
  on a real image). Checked: plug rigid in the hand to 0.0007 mm over all 1055 grasps.
* UNet (perception/unet.py): ImageNet ResNet-34 encoder, full-resolution decoder, RGB only, 7 classes.
  Result (ood_val shapes, 40k steps): opening IoU 0.97, centre error median 0.15 px (p90 1.5 px); pins
  IoU 0.84, centre median 2.7 px.
* Crops (perception/crops.py): 128 x 128 px at native resolution around the predicted pins and the
  predicted openings. The openings' 3-D point is remembered (ray x board depth around them); when the plug
  covers them it is projected with the current camera pose (the socket does not move).
* Encoder (models/crop_encoder.py): per crop 16 x 16 tokens of 8 x 8 px: ResNet-18 features (stride 8),
  raw RGB, raw depth + validity, frozen DINOv2 (for unseen shapes), the token's viewing ray in port axes
  with Fourier features (absolute geometry -> where the two crops are relative to each other), + one
  position token per crop (centre ray + pixel position, as the user asked). Per-crop self-attention ->
  cross-attention both ways (plug <-> opening) -> joint attention with the camera geometry -> 64 queries
  -> latent 64 x 16 (was 64 x 8).
* Autoencoder (models/crop_ae.py): reconstructs both crops (RGB, depth, validity, DINO) and the v3g
  geometric pose readout from both crops' pixels. Validation logs the readout near the hole separately.

## 2. Time: 6.2 s history with irregularly seen frames (models/history_cdit.py, data/history.py)

* 32 slots of 0.2 s. Vision of a slot = its latent or nothing; attention sees only the seen slots (key
  padding mask in CDiT's cross-attention), no "not seen" flag (user's decision: irregular sampling as in
  time-series classification -- GRU-D / mTAN / Raindrop / masked transformers; CroSTAta masks a random
  block of visual input for manipulation policies).
* Training masks: p 0.5 the last L slots (the plug covers the opening until the end -- the main case,
  the user's point), p 0.2 a middle block, every slot dropped with p 0.15.
* Proprioception token per slot (measured + commanded hand pose, never masked): blocked = commanded moved,
  measured did not.
* CDiT itself (NWM, vendored) unchanged; the subclass replaces the fixed 4-frame context by the history,
  adds a slot-time embedding, conditions on the commanded hand path (16 waypoints) + rel_t.

## 3. Success classifier (models/success.py)

Binary "inserted now" from the same history (same masking, context noise), label from the simulator
(the evaluation criterion: depth <= 1 mm, axis offset <= 2 mm, tilt <= 2 deg; on the real robot the
connector's contact switch). Replaces the pose readout as the judge of success in imagination. The
online learner also trains it on world-model predictions labelled with the real frame's outcome.

## 4. Online training (online/)

Actors (8 processes, train-split shapes / grasps only): the FM policy in the simulator; in half the
episodes the world model chooses among 8 FM proposals (each imagined 2 s ahead, scored by the success
classifier; 10 % a random proposal), in the other half the FM acts alone. Every episode (raw + v4
latents + labels) goes into the online buffer. Learners: the world model and the classifier continue
from their offline checkpoints on a 50/50 mixture of offline and online data, writing current.pt every
1000 steps; actors reload it every 5 episodes. Where the world model is optimistic, the simulator
shows the real outcome and that data corrects it (Dyna / TD-MPC2 loop).

## Pipeline (scripts/pipelines/pipeline_v4.sh)

UNet -> crops -> crop autoencoder (80k steps) -> latents -> offline world model (60k) + success classifier
(60k) + FM (100k), in parallel -> online. Every stage first runs on a small subset; the pipeline stops at
the first failure. GPU caps keep our total under 55 GB.
