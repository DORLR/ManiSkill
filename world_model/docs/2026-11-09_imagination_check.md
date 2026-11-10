# 2026-11-09: does imagination agree with the simulator? (FM's own trajectories)

## Why

The world-model checks so far (`2026-11-08_v3_runs.md`, `results/v3g_wm_s_step35k*.json`) used recorded
data: expert insertions, perturbations, contact trials. The residual policy trained in imagination looked
good there and failed in the simulator, so these checks do not show whether imagination can be trusted on
the states a policy visits. Before using imagined trajectories to improve the FM by imitation (decided with
the user 23:10, instead of RL), the world model is compared with the simulator on the FM's own rollouts.

## Protocol (`scripts/analysis/check_imagination.py`, `wm/analysis/imagination_checks.py`)

* Starts: evaluation plan seed 0 (ood_val shapes), first grasp of every (shape, condition) cell, both
  starts: 13 x 6 x 2 = 156 starts.
* Branch points after 20 / 40 / 60 decisions (8 / 16 / 24 s; FM insertions happen at 17-31 s, median 26 s).
  The FM runs to the branch point with the episode seed (identical every time; checked by the joints at the
  branch point), then 4 continuations with different policy seeds run up to 40 decisions (16 s) or an
  outcome (fixed criterion).
* 1 readout: the pose read from real frames vs simulator truth.
* 2 prediction: open loop with the commands the FM really executed (nothing real fed back), from the branch
  points 0.4-16 s ahead and 0.4 s ahead from every decision. Error of the predicted reading vs the real
  frame's reading, next to "nothing changes"; joints from the adapter and, separately, the real ones.
* 3 outcome: from each branch point the FM acts in imagination (8 continuations, 16 s) vs the 4 simulator
  continuations. Imagined "inserted" = reading < 1.5 mm. Rates, per-branch agreement (AUC, correlation),
  lateral hand travel.

## Smoke test (G46, 12 episodes, old branch points 0/10/25)

Pipeline works (36 s for 12 episodes). Early signs, to be confirmed on the full run: the depth reading has a
~0.7 mm constant offset against truth; the reading of the lateral offset near the hole is ~0 while the truth
is ~0.5 mm; with the FM acting in imagination the hand drifted sideways more (median 2.7 mm) than in the
simulator (1.0 mm).

## Results

(full run started 23:30, `$Q/runs/checks_v3g/imagination/`)
