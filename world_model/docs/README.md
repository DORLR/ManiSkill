# Records

Every run and decision has a dated document. Results (validation logs, check reports, closed-loop
summaries) are copied into `results/`; run directories are on rs2 under `/mnt/storage2/qingxin/runs`.

| document | what |
|---|---|
| [design_v3.md](design_v3.md) | the current design, its checks, and its deviations from NWM / Ctrl-World |
| [2026-11-06_data_wm_v1.md](2026-11-06_data_wm_v1.md) | task, dataset wm_v1 (1240 grasps, 13 701 trajectories), splits, collection findings, initial plan |
| [2026-11-07_v1.md](2026-11-07_v1.md) | v1: point-cloud VAE, one-step CDiT, latent FM 34%, WM-guided selection (no gain), videos, latent and stall checks |
| [2026-11-08_baseline_v1.md](2026-11-08_baseline_v1.md) | latent-free baseline 6% vs latent FM 34%: what was compared, how to read it, budget caveat |
| [2026-11-08_v2.md](2026-11-08_v2.md) | v2: angle loss, no KL, tracking history, predicted-pose loss; results; why stopped |
| [2026-11-08_v3_runs.md](2026-11-08_v3_runs.md) | v3 / v3g runs: smoke tests, gate failures on pitch and their analysis, decisions, results as they arrive |
| [2026-11-09_imagination_check.md](2026-11-09_imagination_check.md) | imagination vs simulator on the FM's own trajectories: readout, open-loop prediction, outcome agreement |

`results/`

| file | content |
|---|---|
| `closed_loop_v1.json` | closed-loop summaries: latent FM (24 s, 48 s subset, 60 s), baseline v1, FM + WM selection (partial) |
| `vae_v1_val_log.jsonl`, `vae_v2_val_log.jsonl` | autoencoder validation curves |
| `wm_v1_val_log.jsonl`, `wm_v2_val_log.jsonl` | world-model validation curves |
| `policy_v1_val_log.jsonl`, `baseline_v1_val_log.jsonl` | policy validation curves |
| `v1_report.json`, `v2_report.json` | jam-vs-insert probes |
| `v1_offset_report.json`, `v2_offset_report.json` | offset regressions |
| `v1_wm_contact.json`, `v2_wm_contact.json`, `v2_report_world_model.json` | world-model stall checks (one step) |
| `v3_adapter.json` | joint adapter on the contact trials (free vs blocked) |
| `v3_latents.json` | v3 latent checks: jam probes, offset regressions, gate |
| `v3_pose_labels.json` | pose label vs true offset; spread of inserted poses within a grasp |
| `v3g_latents.json` | v3g latent checks incl. per-condition table of the solved pose |
| `v3g_pitch_geometry.json`, `v3g_plug_axis_icp.json` | learning-free pitch checks (ICP) |
| `v3g_contact_signal.json` | can hand-pose signals during pushes reveal the offset (leaky: expert commands) |
| `v1_wm_contact_rollout.json`, `v2_wm_contact_rollout.json` | multi-step checks -- **invalid** (decoded joints exploded), kept as the record of that failure |
