# Port plan: online attempts and test-time evaluation on AutoMate (Isaac Lab)

Scope: run `scripts/online/run_wm320_attempts.py` (online rounds, `--mode collect`; tests `--mode act | wm | servo | random`)
and the ACT-alone test on AutoMate grasps. Plan only; nothing is ported yet.
Code read: `/home/qingxin/projects/ManiSkill-wm/world_model`, branch `automate`, uncommitted tree, 2026-11-23
(items 5-7 and 9 re-read after the user's decisions of 2026-11-23, audit round 5).
Warning on line numbers: `src/wm/sim_automate/{collect,checks,config,app,expert}.py` were being edited while this was
read (expert.py again during round 5: the contact-offset and random moves restored to wm_v1's `_move_outside`).
Lines below for those files are as of the last read; function names are the stable reference: re-grep them when
porting. Since audit round 6, collect.py, config.py, expert.py, demo320.py, app.py and train_wm320.py are cited by name only;
since audit round 8, robot.py and scene.py too (Robot.*, the shims, AutomateScene.*); since audit round 9,
online/attempt320.py (AT) too (it changed for decision 6; its places are named by function / block / constant);
since audit round 10, checks.py (CK) too (round 8 inserted `check_contact_report` into it; r9-data-02).
Abbreviations: RA = run_wm320_attempts.py,
AT = online/attempt320.py, BR = collect/branch.py, EX = collect/explore.py, EP = eval/eval_policy320.py,
D = tdmpc/demo320.py, WD = tdmpc/wmdata320.py, AR = sim_automate/robot.py, AE = sim_automate/expert.py,
AC = sim_automate/collect.py, CK = sim_automate/checks.py, CF = sim_automate/config.py.

## 0. Blocker found first: ManiSkill imports at module level

- Now: AT's module imports (`from ..collect import branch`, `from ..collect.branch import ...`, `from ..collect.explore import
  truth_outcome`) import `collect.branch` and `collect.explore`; BR:53-61 imports `wm.data.recorder` (mani_skill
  `scripted._copy_pose`), `wm.env.expert` (mplib, sapien, mani_skill), `wm.env.robot` (gymnasium, sapien),
  `wm.sim.env_adapter` (mani_skill); EX:59-64 the same. `wm.tdmpc.closed_loop_policy` (EP's worker) imports BR/EX too.
  RA:142 imports `wm.env.expert` / `wm.env.robot` inside the worker. None of this imports in the Isaac Lab python.
- AutoMate offers: `sim_automate/record.py:26-57` (loads `wm.data.recorder` with stand-ins, exposes `Recorder`,
  `truth_rel`); AR `Robot`; AR `Frame`; AE `Expert`. No neutral home for FullRecorder / Brancher / Explorer.
- Work: split AT into a simulator-free attempt loop (KINDS, decision loop, back-off, probes) and two thin adapters
  (ManiSkill: today's imports; AutoMate: AR/AE/CK). Keep ManiSkill behaviour bit-identical (test T1). Do not edit BR/EX
  in place while ManiSkill jobs import them; move the pieces AT needs (FullRecorder, `frame`, `offset_pose`,
  Brancher.snapshot/restore/jacobian/fk) into a new module without simulator imports, or copy them into the AutoMate
  adapter. Which of the two is the user's call (code layout, not model logic).
- Simulator detection (round 9, r8-data-02): `common320.simulator_of_robot` calls a robot AutoMate only when its
  class's module starts with `wm.sim_automate.`; any other robot is "maniskill". So the AutoMate adapter's robot class
  must live under `wm.sim_automate`, or `simulator_of_robot` must be extended to it in the same change. Otherwise
  a "gap_mm" checkpoint is refused loudly by `check_offset_label` (not a silent leak), but only inside each worker,
  after Kit's start (up to 5.5 min). Tested in T2.
- Risk: the Isaac python must also import `wm.online.agent320` (torch, `wm320`, `encoders320`: transformers /
  torchvision for resnet34 / segformer). Not checked; part of T2.

## 1. Robot adapter

- Now: AT uses `r._command` (`OnlineTester.attempt`: the place step and the last command after the loop), `r.qpos`
  (in `attempt`), `r.hand_pose`, `r.base.get_state_dict/set_state_dict` (the probe snapshot / restore in
  `attempt`'s `branch_all`, probes only), `robot.port_spec.pin_length_m` (EX:206). RA uses `Robot(port)` (147, 182),
  `robot.env.close()` (146, 179), `robot.reset(seed, pitch_deg, grasp_pos_m)` (168), `robot.base.get_state_dict` for
  the test base state (204-207), `robot.base.scene.sub_scenes` (side camera, 94). BR.Brancher builds `HandFK(robot)`
  (BR:177) and `Expert(robot)` (178); BR.place falls back to `expert._planned_move` (MPlib, BR:191).
- AutoMate offers (already, the starting list missed this): AR is a ManiSkill-shaped facade: `Robot.qpos`,
  `Robot.get_qvel`, `Robot.hand_pose`, `Robot._ik`, `Robot._command`, `Robot.move_to` (returns False when not settled,
  ManiSkill returned True), `base.get_state_dict/set_state_dict` (`_Base.get_state_dict` / `_Base.set_state_dict`) ->
  `Robot.get_state` / `Robot.set_state` -> `AutomateScene.get_state` / `AutomateScene.set_state`,
  `base.agent.robot.get_qvel` (`_ArticulationShim.get_qvel`), `base._planner()` (`_Base._planner`, a `_PlannerShim`)
  with `PinocchioModelShim` (kinematics.py) so `HandFK` works, `Robot._camera()` / `Robot._twin` (a `_TwinShim`) for
  FullRecorder. Missing: `env.close()` is a no-op (`_Env.close`), `scene.sub_scenes`, `_planned_move`
  (`_PlannerShim.plan_pose` returns an error), `Expert.to_safe_from_spawn`.
- Work: (a) when the assembly changes, close with `robot.sc.close()` (as AC `_close`), never `env.close()`; a second
  `AutomateScene` in the same app without it would build on the old stage. (b) `reset` keyword-only (item 3).
  (c) snapshot/restore only for the probe (`probe_max > 0`, the probe block in `attempt`: `branch_all`). Under per-reset sockets (item 3) the test
  base state of RA:204-207 is not needed: every attempt is its own reset + reference + placement.
- Risks: `AutomateScene.set_state` restores articulation roots incl. the socket (fixed-root articulation) but never
  re-checks the socket pose (`AutomateScene.check_socket_pose` is only called by reset); a changed weld inside
  set_state triggers the restart in `AutomateScene.set_weld`. PhysX contact/warm-start caches are not part of the state, so a
  restored branch is not exactly the same; the probe already measures this (`restore_dev_mm`, `R0_repeat`). Needs a
  measured check (T4) before any probe run.

## 2. Recorder (FullRecorder equivalent)

- Now: BR.FullRecorder 97-171: per control step qpos, `_ea._np(r.base.agent.robot.get_qvel())` (134), FK hand,
  `_truth_rel(r)` (136); every 4 steps `_image` (138-171) reading `wm.env.robot` constants CAM_SCALE/CROP/DEPTH_MIN_M/
  DEPTH_UNIT_M, `r._camera()`, `r.base.scene.update_render()`, `r._twin`. AT keeps `rec.buf`, `rec.t`, `rec.frame`.
- AutoMate offers: the same constant names in CF (CAM_SCALE, DEPTH_MIN_M, DEPTH_UNIT_M, the crop; from `wm.sim.config`, same values; the user approved the same
  camera, mount and crop 2026-11-22); the shims make `_image` produce the crop that `Robot.frame()` and
  the collection Recorder produce (rgb crop, depth `< DEPTH_MIN_M -> 0`, 0.1 mm units, K shifted by the crop, same
  `cam_T_world`). `record.truth_rel` = `_truth_rel`. Truth per step: `CK.truth_from_rel_spec` (eval_tip columns).
- Work: an AutoMate FullRecorder with the same buffers and schedule, constants from CF, `Frame` from AR, `truth_rel`
  from record.py, plus `eval_tip` per step for reports. Nothing from truth enters the written trajectory (write_fast,
  RA:104-125, stores rgb/depth/img_step/qpos/hand/action/dec_kind only).
- Check (T3): one attempt's pictures equal `Robot.frame()` pictures taken at the same steps, byte for byte.

## 3. Reference per attempt and its acceptance rule

- Now: RA:159-189: `robot.reset(ep_seed, pitch, pos)`, `Expert.to_safe_from_spawn()` + `run("insert")`, H* = FK hand,
  axis from FK during `axial_push` (190-196), up to 3 tries with `seed + 7919 t` for grasps without axis; then
  `OnlineTester.remember_standoff` (198, EX:213).
- AutoMate offers: `AR.Robot.reset(seed, *, clamp_deg, dp_m, spin_deg, socket_index)` and `Robot.reset_from_meta`
  mapping grasp.json `pitch_deg/grasp_pos_m/spin_deg`; `AE.Expert.reference_label(seed_offset)` returns H*, FK axis,
  `ok = completed AND success at the end`. Under `SOCKET_DRAW = "per_reset"` every reset has its own socket pose, so
  every attempt needs its own reference (the collector does this per trajectory: the label block of AC collect_grasp,
  the `expert.reference_label(` call).
- Acceptance rule (lead resolution r3, binding; the collector was changed to it in round 3, AC module docstring
  "reference"): a reference counts iff the expert completed (Chongyu's final_verify) AND AutoMate's rule holds at its
  end (s_T), for t00 and for every per-reset label reference alike; both parts are recorded
  (reference_expert_completed, reference_success). The online / test reference must use exactly this rule
  (`lab["ok"]` of `AE.reference_label` = completed AND success at the end, plus a defined FK axis). Check against the
  final collect.py when porting: an earlier draft of this plan found the collector ignoring completion.
- Work: per attempt: `reset(seed_a, **grasp, socket_index=k_a)`; `lab = expert.reference_label(stream)`; accept by the
  chosen rule; failure -> attempt counted as `reference_failed` (not an attempt; reported like placement failures; no
  socket redraw, as the collector, so no selection on easy sockets). Drop the 7919 retry loop (AC REFERENCE_ATTEMPTS = 1;
  r9-data-02: not because a reset is deterministic, but because RA's retry uses a new seed, seed + 7919 k, and a new
  seed redraws the socket pose, so retrying until a reference succeeds favours easy sockets; a same-seed retry would only
  repeat the first: AC's comment at REFERENCE_ATTEMPTS, deviation table row 22). Labels meta for this attempt = this reference's H*, axis.
- Since decision 7 of 2026-11-23 (deviation table row 39) the reference can also fail with
  `<phase>_nominal_command_guard: hand_finger_command_clearance` (expert_failed): a nominal outside command whose hand /
  fingers come within 1 mm of the socket. Count it as `reference_failed` and report the failure text per assembly
  (in the collector probe 00422, 00581, 00186 and 00741 had such failures on their recorded kinds).

## 4. Socket poses of online and test attempts

- Now: ManiSkill has a fixed board; nothing to choose.
- AutoMate offers: the socket pose is a function of (reset seed, socket_index) only: `CF.socket_seed_sequence`,
  `CF.socket_pose`; `reset` cannot take a SeedSequence. Collection used (grasp seed, k) for k = 0..12 per grasp
  (t00 = 0, trajectory i = i; 12 trajectories from the pinned mix {insert 4, perturbed 2, contact 5, random 1},
  AC CollectConfig.trajectories; redraws reuse k).
- Work: per attempt `seed_a = SeedSequence(eval_seed, spawn_key=(int(aid), gi, attempt, P_SOCKET)).generate_state(1)[0]`,
  `socket_index = 0`; `gi` parsed from the grasp key `<aid>_gNN` (grasp.json has no `gi`; plan.json does).
  Same seeds for every mode (act / wm / servo / random) and independent of `--grasps-n` (today: `seed + 100_003 k`,
  RA:419). Before any start: assert `CF.SOCKET_DRAW == "per_reset" ==` plan.json `switches.socket_draw`; assert no
  (seed_a, 0) equals any (grasp seed, k) of plan.json over ALL grasps; record the pose (`robot.episode`) in the result
  file. Online rounds: add `round` to the spawn key so rounds draw new sockets.
- Open: `P_SOCKET` and the other purposes must not reuse AC.STREAMS values with the collection entropy; a distinct
  `eval_seed` default avoids that by construction.

## 5. Success and truth

- Now: `attempt.truth_check` per decision `truth_outcome(eval_rel[-1])` + `D.is_success`; the final truth / summary
  after the loop of `attempt`: final `success`,
  depth/lateral/angle (`D.SUCCESS`, ManiSkill numbers at the charger origin, EX:392-403).
- AutoMate offers: `CK.success_steps`: row 0 (start state) never counted,
  `success_any`, `success_end`, `first_success_step`, `start_success`; `CK.success_any`; `CK.truth`.
  AC.automate_success excludes row 0 as well since round 3 (first_success_step >= 1, success_at_start information);
  verify both agree when porting (one function should be the source).
- Work: AutoMate adapter reports `automate_success_any` = `success_steps(eval_rel)["success_any"]` over the whole
  attempt (eval_rel row 0 = the placed start), plus `success_end`, `first_success_step` (in control steps, convert to
  decisions for `first_success_decision`), `start_success`. Per-decision `truth_ok` from `truth_from_rel_spec` on the
  last row. ManiSkill-style depth/lateral/tilt (tip centre) only as info, never shown as a success figure.
- Start state and AutoMate's rule (settled by the user's decision 1 of 2026-11-23, option B): the gap g0 in [6, 18]
  mm is measured from the socket top, b = L + g_top + g0 with g_top = max(L_top, L) - L (`CK.g_top_of(spec)`),
  `config.START_GAP_FROM = "socket_top"` (any other value refused outside diagnostics).
  Every start is outside the socket, as in AutoMate's evaluation. Recomputed offline with `CK.start_box_success`
  (gap_from socket_top) over the start box of all 91 assemblies with a ranges_v5 box (20,000 draws
  each, 1,820,000 starts, scripts/automate/diag/start_success.py): 0 starts satisfy AutoMate's rule. So
  `start_success` of an online / test attempt is not an open gate any more but a consistency check: a True value means
  a wrong start (g_top not applied, a stale spec, another START_GAP_FROM), and the run must stop on it, as the
  collection checkers do (check_expert_data start_success_rows, exit 1).
- AutoMate's success region can reach above the socket top: min(d_dis, 15 mm) - L_top > 0 for 42 of 91 assemblies (at
  most 1.99 mm, 01041; check_expert_data band_above_top_mm). A plug resting on the socket top within that height counts
  as AutoMate success without insertion. Report this beside every success figure (no rule change: the rule is the
  user's choice).

## 6. Start placement

- Now: `B.sample_start` (BR:81-85, gap 6-18 mm, +-10 mm, +-6 deg rotvec in EX.frame), `Brancher.place` (BR:180-204:
  standoff joint move, `move_to` far/near, MPlib fallback, FK settle), `PLACEMENT_ATTEMPTS` 5.
- AutoMate offers: `wm.env.sampling.sample_start` (same box numbers, static XYZ angles, Offset),
  `AE.start_pose(H*, F, L, off, g_top, gap_from)` (static, a thin call of `frames.start_pose`,
  frames.py:83) with `F = frames.frame(axis, robot.kin.base_T[:3, 0])` (frames.py:38). It moves H* back along a by
  b = L + g_top + g0 for gap_from "socket_top" (the default and the only approved value, the user's decision 1 of
  2026-11-23) and raises when g_top is missing; "entrance" (b = L + g0) is the refused old rule, diagnostics only.
  `AE.start_clearance` (plug / hand / fingers at the start and the plug along the descent must stay
  more than START_CLEARANCE_M = 1 mm from the socket, else the start is redrawn: a filter of the start distribution,
  the "clearance" placement failure), `AE.move_to_start` (guarded retreat, transfer, descend, settle;
  ok if < 0.5 mm / 0.2 deg); after a failed transfer the collector re-homes with `reset(same seed, same index)` and
  redraws (the placement loop of AC collect_grasp, `for k in range(PLACEMENT_ATTEMPTS)`; AC PLACEMENT_ATTEMPTS = 3;
  placement stream).
- Work: replace `remember_standoff` + `place` by the collector's loop, with the attempt's H* / FK axis, and the start
  exactly as the collector makes it: `T_start = Expert.start_pose(H*, F, L, off, checks.g_top_of(spec),
  config.START_GAP_FROM)` (AC collect_grasp's `Expert.start_pose(` call passes `extra["g_top_m"]`, which
  collect_grasp sets to `K.g_top_of(ps)`). The online /
  test runner records `config.switches()` (start_gap_from among them) in every result file and refuses to start when
  `config.START_GAP_FROM` differs from plan.json `switches.start_gap_from` of the grasps collection, so the test
  starts come from the collection's distribution. First start from a separate stream (`spawn_key=(aid, gi, attempt,
  P_START)`), redraws from (`..., P_PLACE`). Keep `start_error_mm`, the clearance and the failures in the result
  file.
- Note: `OnlineTester` keeps `self.F = EX.frame(axis)` for `face_gap_mm`, the exploration shift and `Bt`
  (reporting); only F[:, 0] matters for the first two, so this is harmless, but it must come from the new module.

## 7. Fast-copy meta of online / test trajectories

- Now: RA:251-256 builds meta from g's ManiSkill keys plus `kind, outcome, source, pin_length_m, H_star, axis, round,
  start (int), start_pose, attempt_seed, mode, model_step, episode_seed, episode_seed_used, stopped, interventions,
  retries`; write_fast adds `steps, format, dec_kind` (RA:119) and calls `D.entry_of`.
- AutoMate offers: `D.FAST_META_AUTOMATE` (now incl. expert_completed, automate_success_any; tilt_definition removed);
  `entry_of` refuses any other key for `simulator == automate`; `D.is_bc` (bc needs expert_completed for AutoMate).
- Missing from the starting list: the online meta has no `simulator` key, so today an AutoMate attempt would be
  written as a ManiSkill entry and `split_of_entry` would apply `split320` to an assembly id (silently "train" or
  "select"; the pool's split check would not catch it; since round 8 the simulator check of
  train_wm320.online_entries refuses such an entry under an AutoMate fast copy, but the meta must still carry
  `simulator`). Also `start` means
  Offset.as_dict in AutoMate metas but the start index in online metas.
- Work: AutoMate online meta = {grasp, port, condition, kind, outcome, steps, H_star, axis, pin_length_m, pitch_deg,
  grasp_pos_m, spin_deg, u, simulator, symmetry_n, switches, assembly_id, config_sha, plan_sha, format, source, start
  (Offset dict)} + new allow-list keys {round, start_idx, attempt_seed, mode, model_step, stopped, interventions,
  retries, dec_kind, split_sha256}; no episode_seed / reset seed / socket pose (they regenerate the socket).
- Split identity of the pool (round 4, r4-data-06): every online manifest entry (complete rounds and the finished
  grasps of a round in progress) carries `split_sha256` = `demo320.split_identity(<grasps collection>)["sha256"]`,
  and each round's manifest carries the whole identity under "split". train_wm320 already refuses (since round 4) an
  AutoMate pool entry whose `split_sha256` is missing or differs from the fast copy manifest's `split["sha256"]`
  (`train_wm320.online_entries(pool, split_sha)`), because a stored condition "train" means training only
  under the split it was collected with.
- How the entry gets `split_sha256` (round 5, r5-data-03): today write_fast returns
  `{**D.entry_of(meta, ...), "dec_kind": kinds}` (run_wm320_attempts.py:125), and `D.entry_of`
  copies only a fixed key set (port, grasp, condition, kind, outcome, pitch_deg, grasp_pos_m, pin_length_m, simulator,
  and `D.AUTOMATE_ENTRY_KEYS`) and raises on any AutoMate meta key outside `D.FAST_META_AUTOMATE`. A meta key `split_sha256` would therefore make entry_of raise, and without it the entry has no
  `split_sha256`, so online_entries would refuse every AutoMate pool entry. Two ways, one to be chosen: (a) add
  `split_sha256` to both FAST_META_AUTOMATE and AUTOMATE_ENTRY_KEYS (they must change together; this changes D, a
  PROVENANCE_FILES member of train_wm320: ask the user first, as said above), or (b) keep D unchanged and have the
  AutoMate write_fast set `entry["split_sha256"]` on the returned entry after entry_of. Either way T3 asserts that the
  written manifest entry holds it. Extending FAST_META_AUTOMATE for the other online keys likewise changes D: ask
  before. `outcome` stays
  `seated_fk` / `not_seated` from WD labels; online `bc = outcome == "seated_fk"` (`train_wm320.online_entries`) unchanged.
- Simulator of the pool (round 8, r7-data-01): `train_wm320.online_entries` refuses any pool entry whose simulator
  differs from the fast copy's (an entry is AutoMate iff `D.is_automate`; the fast copy is AutoMate iff its manifest
  has a split identity), so a ManiSkill pool given to an AutoMate run is refused, and the reverse.
- Labels: AT passes `meta_l = {H_star, axis, pin_length_m}` (`AT.OnlineTester.attempt`: `meta_l` and the final labels) to `WD.decision_labels`
  (reporting and the stored outcome only; the agent's gap reads no L, item 11a). Since round 3 the
  labels have one angle term (full angle, ManiSkill's) for both simulators, so this gives the training labels; pass
  `simulator` too so that an AutoMate meta is checked the same way. Since decision 6 (2026-11-23) `simulator` also
  selects the offset-head target of the online windows: `WD.offset_label` = `gap_mm` only when simulator ==
  "automate" (r9-data-02), so online windows without it would train the head on `along_mm` beside the fast copy's
  `gap_mm`. `train_wm320.online_entries` refuses such an entry under an AutoMate fast copy (its simulator check), so
  this cannot leak silently, but the online meta must carry `simulator: "automate"`.

## 8. Split filters

- Now: RA:383-387 refuses AutoMate grasps, then `D.check_automate_conditions`; RA:388-390 filters `reference_ok` and
  `D.split_of_entry(g) == args.split`; collect only with split train (RA:343); test only with `--final-test` (RA:338).
- AutoMate offers: `D.split_of_entry` maps condition train/select/heldout_grasp/heldout_shape/test.
- Work: remove the refusal for AutoMate only after items 0-7; keep the rest. Refuse `--split any` and
  `pair_index` grasps for AutoMate (they use ManiSkill `TEST_PORTS`, RA:341, 389-392). `--split heldout` = both
  heldout kinds; report them apart (item 10). In every mode (collect and the tests), RA refuses to start unless the
  checkpoint's `provenance["split"]` equals `demo320.split_identity(<grasps collection>)` (name, file, sha256,
  plan_sha); a checkpoint without it is refused. Optional: the parent-side offset-label check of item 11a, on the
  same loaded checkpoint, before any worker starts.
- Since round 4, eval_policy320.py, eval_ppo_state.py and scripts/analysis/ablation320.py also refuse AutoMate
  grasps / result files up front (before any worker starts), as RA and train_ppo_state.py do; lift each refusal only
  with its port.

## 9. Workers, GPU and memory

- Now: RA `_init` (53-69) sets `CUDA_VISIBLE_DEVICES` per worker from `--gpus`, caps torch memory, builds `WMAgent`;
  `run_pool` (272-298) opens a fresh ProcessPoolExecutor every `10 x workers` grasps; `side_camera` is SAPIEN.
- AutoMate offers: `app.launch` then `app.assert_off_forbidden_gpu` (after start, scene build, first
  frame); exits 7 (compute context or > 50 MiB graphics on GPU 0: stop the run) and 8 (nvidia-smi unreadable:
  counted crash) both exist now (`app.EXIT_FORBIDDEN_GPU`, `app.EXIT_GPU_UNREADABLE`); `app.PRE_EXIT_HOOKS` for releasing claims; `CF.KIT_GPU`
  (env WM_KIT_GPU, default 1 = row 1 of Kit's Vulkan GPU table = physical GPU 1; Kit's table ignores
  CUDA_VISIBLE_DEVICES, which restricts CUDA only: a port that sets 0 renders on GPU 0, which the GPU-0 rule forbids,
  and with CUDA_VISIBLE_DEVICES=1 builds no render device, as the lead measured); Kit start up to 5.5 min (AC HANG_TIMEOUT_S comment); collection workers:
  CPU PhysX, MemoryMax 16G (min 14G), CPUQuota 500 %, GPU-1 declaration worker_gpu_gb (AC WORKER_GPU_GB, default 7; production: the smoke's worker_gpu_gb_recommended = ceil(1.3 x G1), r11-ops-01); budget 125 GB /
  3200 % / 48 GB GPU-1 (caps.sh QX_CAP_MEM_BYTES / QX_CAP_CPU_PCT / QX_CAP_GPU_OURS_MIB); `collect_worker.sh` / `caps.sh` start one capped systemd scope per
  worker.
- GPU memory of an online worker (round 5, r5-data-03): the worker's declared QX_GPU_GB must cover Kit (the Isaac
  app's own GPU memory, outside torch) and torch together, while `torch.cuda.set_per_process_memory_fraction`
  (run_wm320_attempts.py:65, `--torch-mib`) caps only torch's allocator; the CUDA context (about 1 GB) is outside that
  cap too. So declare QX_GPU_GB >= Kit's measured peak + torch cap + 1 GB, and measure Kit's part in the smoke (T7).
- Work: one Isaac app per worker for the worker's whole life; launch the app before importing torch / the agent; set
  `CUDA_VISIBLE_DEVICES=1` before launch; refuse `--gpus` other than `1`. The pool slicing restarts Kit every slice
  (5.5 min each) and ProcessPoolExecutor children share the parent's cgroup, so a per-worker MemoryMax cannot be set:
  use collect-style workers (own scope, claim per grasp, heartbeat, STOP file) instead; that is an infrastructure
  change, to be named to the user. Register a PRE_EXIT_HOOK that writes the grasp's result file as `crash` before exit
  8 and stops the launcher on 7. Budget per worker = Isaac (worker_gpu_gb declared (see above), RAM peak 3.5-11.8 GB measured) + agent
  (torch context + model + `--torch-mib`, RAM and GPU, not measured): measure in the smoke (T7). Refuse `--side-video`
  on AutoMate (SAPIEN camera).
- The learner shares the budget (round 8, r7-data-02): under the agreed scheme (2026-11-18) `train_wm320 --online`
  trains while a round is collected (it reads the finished grasps of a running round), and on AutoMate both it and
  the online workers run on GPU 1 under our caps (caps.sh: 48 GB GPU-1, 125 GB MemoryMax sum). With N workers:
  GPU 1: G_learner + N * (K_peak + T + 1 GB) <= 48 GB, where G_learner = the learner's declared QX_GPU_GB,
  K_peak = Kit's measured GPU peak per worker, T = `--torch-mib` x 2^20 / 1e9 GB (torch's
  allocator cap), 1 GB = the CUDA context outside that cap;
  RAM: M_learner + N * M_worker <= 125 GB, where M_learner and M_worker are the MemoryMax of the learner's scope and
  of one worker's scope. If the planned N does not fit beside the learner, show the numbers and ask the user: fewer
  workers, or rounds and the learner alternate (the learner then trains only between rounds, a change of the agreed
  2026-11-18 schedule). Never shrink either silently.

## 10. Results, summary, groups, split identity

- Now: EP.summarise (109-133) calls `D.group_of_entry`, which raises for AutoMate; cluster = port for
  `unseen_shape` else grasp (EP:120); criterion text is ManiSkill's (EP:129); RA:422-437 adds seated_fk / interventions.
- AutoMate offers: conditions in grasp.json; band variable only in plan.json `grasps[].grasp_kind` (`band_<var>`), not
  in grasp.json; `split_automate_v1.json` next to plan.json (AC freeze) carries `plan_sha`.
- Work: AutoMate groups: select, heldout_shape (cluster = assembly), heldout_grasp per band variable (cluster = grasp,
  band from plan.json joined by grasp key), test (cluster = assembly); success = `automate_success_any`; also count
  `reference_failed`, placement failures, `start_success`. ManiSkill path unchanged (EP is imported by other runs:
  add, do not change).
- Split identity (added in round 3): `demo320.split_identity(collection)` = {name, file, sha256 of
  split_automate_v1.json, plan_sha}; cache_wm_v1_fast writes it into the fast copy's manifest ("split"),
  depth_stats.json carries it, train_files_sha256 folds it in, and train_wm320 stores it in checkpoint provenance
  (`provenance["split"]`, plus `wm_file`). Work: the evaluation refuses a checkpoint unless `provenance["split"]["sha256"]`
  and `["plan_sha"]` equal `demo320.split_identity(<evaluated collection>)`; a checkpoint without it is refused.

## 11. Agent rules tied to ManiSkill geometry (missing from the starting list; ask before changing)

- `max_decisions`: `OnlineTester.attempt`'s max_decisions default `B.MAX_DECISIONS = 150`; RA never passes it. AutoMate's episode limit is
  `robot.max_decisions = 150 + ceil(max(0, L - 20 mm) / 0.5 mm)` (`Robot.max_decisions` -> `AssemblySpec.max_decisions`
  in assets.py). Pass it; this is the AutoMate rule.
- `LEFT_M = 0.06` (AT module constant `LEFT_M`, reasoned from gap <= 18 mm + pins <= 20 mm + 10 mm lateral): AutoMate L can exceed 20 mm.
- Exploration shift window 0.5-4 mm and eps window 0-6 mm in front of the "board face" (in `attempt`: the collect-mode
  eps window `0 <= face_gap_mm < 6` and shift window `0.5 <= gap <= 4`) use
  `face_gap_mm` = distance to the entrance of the tight fit (L); on AutoMate the socket top / lead-in lies g_top above
  it (AE docstring "top"), so the windows mean something else.
- `inside_mm`, back-off 5 mm, re-position radius 3 mm, stall 0.2 mm / 8 decisions: unchanged unless the user says
  (the gap the in-hole rule compares with `inside_mm` is read without L on AutoMate: item 11a).
  Show the user each of these with numbers per assembly (L, g_top, d_dis) and ask.

## 11a. Pin-tip gap at test time without L (the user's decision 6 of 2026-11-23; implemented)

- Rule: on AutoMate the world model's offset read-out head is trained with its first component = the pin-tip gap
  g = s - L [mm] (s = -(hand - H*) . axis, the hand's distance back from the seated pose along the insertion axis;
  L = `pin_length_m`, the tight-fit depth; g < 0: the tips |g| past the entrance). L is used only to build that
  training label. At test time every gap of the agent reads the prediction directly, g_t = o_t[0]: the in-hole rule
  (g_t < -inside_mm), the counterfactual probe's gap bins and window, the ablation "random"'s bins (`gap_bin`) and
  the logs (`per_decision.g`, `inside_gap_mm`). No L, H* or simulator truth enters. ManiSkill keeps g_t = -o_t[0] - L
  (labels, checkpoints, results unchanged).
- Where: label `WD.offset_label` / `WD.offset_target` (`OFFSET_GAP` = "gap_mm" for meta simulator "automate",
  `OFFSET_ALONG` = "along_mm" otherwise; "offset" key of `WD.trajectory` / `WD.window`; "probe" unchanged);
  `WM320.loss` trains the head on W["offset"]; `train_wm320` writes `offset_label` = "gap_mm" into every AutoMate
  checkpoint (top level and `provenance`) and refuses to resume a checkpoint with another label;
  `common320.gap_estimate_mm(o, label, pin_length_fn)` computes g_t (for "gap_mm" the pin-length function is never
  called); `AT.OnlineTester.attempt` uses it and refuses (`common320.check_offset_label` with
  `common320.simulator_of_robot`) an AutoMate attempt whose checkpoint is not "gap_mm" and a ManiSkill attempt whose
  checkpoint is; `agent320.WMAgent` reads `offset_label` (missing = "along_mm"), refuses an AutoMate checkpoint
  (provenance split set) without "gap_mm", and `WMAgent.gap_mm(z)` takes no pin length for "gap_mm".
  `scripts/analysis/fidelity_factual320.py` refuses a "gap_mm" checkpoint (it compares with the along labels).
- Not changed: the stand-alone policy's offset head (`policy320.ResnetFMPolicy.off`, training-only, target "probe")
  keeps the along label; nothing at test time reads it.
- Still on the environment side, and allowed (no agent decision of "wm" / "act"): start placement (`place`,
  `remember_standoff`: H*, axis, L), the time limit, `LEFT_M`, the stall rule's FK hand, success and truth
  (`truth_check`, `truth_outcome`), labels for reporting (`meta_l`: `fk_lat`, the probe's true returns; final
  `seated_fk`), `Bt` / `move_mm` / `axis_err_deg` logs. Privileged by design and named as such: collect mode's
  exploration (`face_gap_mm`: FK + H* + L, for the eps window 0-6 mm and the shift window 0.5-4 mm, and the shift's
  direction across the true axis), the ablation "servo" with `servo_axis` "true", and the probe's extra servo branch
  (true axis; reporting only).
- Where the check runs: today `OnlineTester.attempt` and `WMAgent.__init__`, both worker-side. Optional (not
  required: item 8's split-identity refusal plus train_wm320 always writing "gap_mm" for AutoMate already refuse an
  old checkpoint): RA calls `common320.check_offset_label(ck.get("offset_label", "along_mm"), "automate")` on the
  checkpoint it already loads in the parent, before any worker starts. The adapter's robot class must be found by
  `common320.simulator_of_robot` (item 0).
- When the RA refusal is lifted (item 8): keep the attempt's check; RA's split-identity check (item 8) and this check
  together refuse every checkpoint trained before this rule.
- Test: `tests/test_offset_gap_automate.py` (CPU: label = s - L; no pin length read for "gap_mm"; ManiSkill
  bit-identical to git 6b37757).

## 12. ACT alone

- "ACT alone" in the main comparison = `--mode act` of RA (the world model's ACT head, same checkpoint and seeds).
  EP's worker (EP:50-71 -> `closed_loop_policy.test_grasp_policy`, a separate policy320) needs the same items 0-6 if
  the stand-alone ACT policy is also in the comparison. Ask which one is meant.

## Ordered work list

1. Ask the user: agent rules (item 11), ACT-alone meaning (item 12), module layout (item 0), worker infrastructure
   (item 9), allow-list (item 7). Decided, not to be asked again: the acceptance rule (item 3) and success over rows
   1..T, the start state excluded (item 5), both by the lead's round-3 resolution and already implemented in the
   collector (AC.automate_success excludes row 0; reference = completed AND s_T).
2. Offline, CPU: start-box success share per assembly with `CK.start_box_success` (T5), gap_from socket_top: done
   2026-11-23 for all 91 assemblies (0 of 1,820,000 starts); re-run only if the start rule, the ranges or the geometry
   file change.
3. Neutral attempt core + ManiSkill adapter; prove ManiSkill unchanged (T1).
4. AutoMate adapter: recorder (item 2), reference per attempt (3), socket seeds and assertions (4), success (5),
   placement (6, starts with g_top and START_GAP_FROM as the collector), meta and labels (7), max_decisions (11).
5. run script: AutoMate branch of `work`, split filters (8), result fields, AutoMate summary and groups (10),
   split identity check (10), refuse side video / split any.
6. Worker launcher: capped scopes, app before torch, GPU-1 only, exits 7 / 8, heartbeat, STOP (9).
7. Multi-agent code review, fix, re-review until clean (standing rule).
8. CPU stub tests (T1-T6), then GPU-1 smoke on one training grasp (T7, T8), show the user, then a round.

## Tests

- T1 (CPU, ManiSkill fixture): the refactored attempt loop gives identical summaries and fast-copy bytes as today's
  AT on stub robot + stub agent with fixed seeds; tests/test_online_cpu.py and test_ablation_cpu.py still pass.
- T2 (CPU, Isaac python, no app): import of the AutoMate adapter and `wm.online.agent320` succeeds; no `mani_skill`,
  `sapien`, `mplib` in `sys.modules` afterwards; `common320.simulator_of_robot(<the real AutoMate adapter robot
  class, instance made without an app>) == "automate"` and `simulator_of_robot(<ManiSkill wm.env.robot.Robot>) ==
  "maniskill"` (item 0, r8-data-02).
- T3 (CPU, stubs): stub AutoMate robot (qpos/hand/eval_rel scripted, shims for `_camera`/`_twin`): the recorder's
  buffers match `record.Recorder` on the same steps; write_fast output passes `entry_of`; the entry has
  `simulator = automate`, `split_sha256` of the grasps collection, and `split_of_entry` gives the grasp's condition
  split; a meta with `episode_seed` is refused; train_wm320.online_entries refuses the entry under another split
  sha256 and refuses a ManiSkill pool entry under an AutoMate fast copy (r7-data-01, already in data test 5); RA refuses
  a checkpoint whose provenance split differs from the grasps collection's.
- T4 (CPU, stubs): socket seeds: equal across modes; assertion fires on a planted collision with plan.json; per_grasp
  switch refused.
- T5 (CPU, offline): `start_box_success` share over the start box per assembly with gap_from socket_top (expected 0,
  as measured 2026-11-23); `success_steps` on synthetic eval_rel (start row successful, later rows not -> success_any
  False, start_success True); the runner refuses a START_GAP_FROM that differs from plan.json switches.start_gap_from
  and stops on a start_success True.
- T6 (CPU): AutoMate summarise: groups, clusters, band join from plan.json, refusal on split-identity mismatch
  (checkpoint provenance vs the evaluated collection, and online pool entries vs the fast copy), ManiSkill summary
  unchanged on the existing fixture (tests/test_eval_stats.py).
- T7 (GPU 1, smoke, one training grasp, 2 attempts per mode act and wm, then collect with 1 attempt): GPU-0 check
  passes; reference acceptance logged; placement ok; pictures equal `Robot.frame()` at the same steps; measured RAM
  and GPU per worker; the learner's measured GPU-1 and RAM peak while it trains during the round (item 9); wall time
  per attempt (reset + reference + transfer + decisions).
- T8 (GPU 1, smoke): probe restore: save, run one decision, restore, repeat; hand deviation and socket pose
  (`check_socket_pose`) after restore; only then allow `--probe-max > 0`.
