# Curriculum attempts (surrogate)

- attempt1 (7dde684a, destination_v10, std 0.15 -> 0.05, lr max 3e-4, 6000 updates): reached 0.075 at ~1440 updates,
  0.1 at ~2100, then stalled at 0.125 with the destination window falling from 0.73 to ~0.44. Did not complete.
  Regressed at 0.05 m/s: median walk end gap 14.5 mm vs v9 0.82 mm. Fewer falls at 0.1 m/s than v9.
  Decision: restart from v9 with v9's exploration (std fixed 0.05, lr max 1e-4); the curriculum stays destination_v10.
- attempt2 (7dde684a, destination_v10, std fixed 0.05, lr max 1e-4, 6000 updates): reached 0.1 at once (window 0.80 at update 54),
  0.125 at ~2930, then the window fell from 0.73 to ~0.42; did not complete. Final checkpoint regressed at 0.05 m/s (median end gap
  18.5 mm vs v9 0.82). Its update-2900 checkpoint (end of the 0.1 stage) did not regress and beat v9: walks 8/8 at 0.05, 0.075 and
  0.1 m/s with 0.56-0.86 mm median end gaps, 0.125 m/s 2/8, stops at 0.05 m/s 7/8 vs 4/8.
  Decision: the damage comes from the 0.125 stage; restart from v9 with destination_v11 (4ff06af7): a quarter of walks at the top
  speed and a back-off below 0.65; v9's exploration settings again.
- attempt3 (4ff06af7, destination_v11, v9 exploration, 6000 updates): reached 0.1 at once, the window stayed ~0.70-0.76, backed off
  to 0.075 once and returned; never reached 0.125. No regression (0.05 m/s median end gap 1.02 mm vs 0.82), 0.1 m/s walks 7/8
  (v9 3/8), stops at 0.05 m/s 7/8 (v9 4/8); 0.15 m/s no better than v9.
  Walking destination pay per speed (from suite traces): v9 0.87/0.84/0.54/0.09/0.00 at 0.05..0.15; attempt2 u2900
  0.88/0.85/0.73/0.38/0.00; attempt3 0.84/0.73/0.44/0.16/0.00. A quarter of the walks at the top speed starved the top speed.
  Decision: destination_v12 (9dc63b0e) = v10's half at the top speed + v11's back-off; 10000 updates; when unfinished, test the
  last checkpoint at the highest level reached whose window held >= 0.78.
- attempt4 (9dc63b0e, destination_v12, v9 exploration, 10000 updates): reached 0.1, then bounced 0.1 <-> 0.125 five times
  (0.125 windows peaked 0.69-0.74, backed off at 0.65); never passed 0.125. Tested u6000: walks 8/8 at 0.05/0.075/0.1 with
  0.46-0.65 mm median end gaps, 0.125 4/8 (v9 0/8); but standing joint creep (max_joint_position_range_rad) failed quiet_20s,
  quiet_32s, forward_0.05_to_stop and 9 stops: regressed (probes 10/13, stops at 0.05 2/8 vs 4/8).
  Finding: the 0.8 gate at 0.125 m/s is out of reach on the surrogate within 10000 updates; the creep appears late.
  Decision: test attempt4 at u4100, the length a 6 h Spark run reaches (same 4608 transitions per update). If it does not regress,
  queue destination_v12 on Spark with these flags for 5000 updates (the allocation ends it near 4100).
- attempt4 u4100: no regression (probes 12/13; walks 8/8 at 0.05/0.075/0.1 with no falls, 0.125 2/8 with no falls; stops 5/5/4/4/3
  at 0.05..0.15 vs v9 4/4/3/3/2). Accepted as the good result: the best no-regression system over the Spark horizon; 0.15 m/s is
  not reached under the 0.8 gate on the surrogate.
- Spark: dest_v12_from_v9_c36_u5000_seed20260917_20261024_001 (pack from 9dc63b0e, binding 902feedd...), queued and launched
  2026-10-24T10:26:15Z; starting-checkpoint checks passed (v9 16048b8f, declaration a916268c, physics equal); curriculum running.
- Spark v12 run ended at update 2134 (host memory guard, 15:35 UTC), level 0.1, window ~0.76; checkpoints to 2100 on Spark.
- attempt5 (668c1054, destination_v13, requested by the user): every command a walk in a uniform direction, a fifth stop after
  4-12 s; ghost pay keeps its shape but 5 mm pays 0.8 (width 15.9 mm); level gate over the last 10 updates (360 controls);
  v12's curriculum otherwise; v9 exploration, 6000 updates.
- attempt5 stopped at update ~880 by request: on the 10-update window the curriculum cycled 0.075 -> 0.15 about every 110 updates
  and backed off after ~10 updates at 0.15 (window ~0.65), usually cascading down to 0.075.
- attempt6 (b877a6b3, destination_v14, requested): v13 whose back-off judges the last 50 updates at the level; the level still rises
  on 10 updates above 0.8. v9 exploration, 6000 updates.
- attempt6 stopped at update 1005 by request. Replays: u100 at 0.125 walked backward only; u700 at 0.125 walked left and backward
  for 20 s (5.8 and 10 mm median from the ghost), fell forward and right; u350/u650 at 0.15 froze forward and fell elsewhere.
- attempt7 (4afcf7b7, destination_v15, requested): v14 that raises the level only after 10 updates without any robot falling.
  v9 exploration, 6000 updates.
- attempt7 stopped at update ~1750 by request: 0.075 -> 0.1 -> 0.125 in the first 20 updates, then cycled 0.1 <-> 0.125;
  held 0.125 for ~800 updates at window 0.66-0.73, backed off, and 0.1 sat at 0.70 (was 0.85-0.95). Gait limits from the
  model (scratchpad gait/limits.py, envelope.py): stance reach 80 mm fwd / 66 mm side within +-0.35 rad, swing slew 2 rad/s;
  the fixed 1.2 s clock caps speed near 0.11 fwd / 0.09 side, so 0.125 is beyond the schedule.
- attempt8 (37070c65 + wrapper fix, destination_v16, requested): v15 + speed_v1 gait clock (60/52/44/40 controls at
  <=0.075/0.1/0.125/0.15 m/s, swing 0.4-0.5) + back-off when any standing robot is >5 cm from its ghost in the 10-update window.
  v9 exploration, 6000 updates.
- attempt8 stopped at update ~650 by request: 0.125 m/s windows at 0.80-0.92 (attempt7 0.66-0.73), but the 5 cm rule
  lowered the level within ~10 updates every time (19 back-offs, all distance; 98% of 0.075 windows had a robot >5 cm,
  mostly walks still running at the old speed).
- attempt9 (28836c92, destination_v17, requested): v16 with no back-off of any kind and a 30-update (1080-control)
  window to rise (mean > 0.8, no falls). v9 exploration, 6000 updates.
- attempt9 stopped at update ~850 by request: reached 0.15 at update 535 (0.125 window 0.800, no falls). u500 at 0.125:
  7 directions at 0.125 m/s within 1-6 mm of the ghost, forward 0.094 m/s; all 8 fail planar error (0.032-0.068 vs 0.031),
  3 also torque demand > 1.6 Nm over 0.5%. Foot contacts confirm the 44-control (0.88 s) cycle at 0.125 (attempt7: 60).
- attempt10 (destination_v18, requested): v17 + every walk finished in the 30-update window must pass the evaluator's
  moving checks (planar/yaw error, tilt/vertical RMS, torque over-rating <= 0.5%, joint speed, no fall).
- attempt10 (destination_v18) ran to 6000 updates; reached 0.15 at update 1035 (0.125 window passed every walk check).
  At 0.15 the destination pay fell 0.76 -> 0.34 from update 1450 to 4400 while speed held (lag grew into the kernel's
  flat tail). u1450 at 0.15 m/s: 8/8 directions pass, 0.3-17 mm from the ghost; u2800: 6/8, 8-56 mm. Its suite runs on u1300.
- attempt11 (destination_v19, requested): v18 + destination pay linear past 20 mm (tangent; 0 at 49 mm, -1 cap from 117 mm)
  + walking checks with twice the evaluator's planar bound. From v9, v9 exploration, 6000 updates.
- attempt11 (destination_v19) stopped at update ~4650 by request: reached 0.15 at update 947; window pay fell 0.79 -> 0.13
  while walking checks passed. u1450 vs u3500 at 0.15: robots fall 10-25 mm behind the ghost in the first 1-5 s at u1450
  but 70-110 mm at u3500 (first-2-s speed 0.10-0.12 vs 0.14-0.15), then hold exactly 0.150 m/s and cannot close the gap.
  Learning rate sat at its 1e-5 floor throughout.
- attempt12 = model v10 (destination_v20, requested): ghost_v10 inputs (sideways gap + velocity relative to the ghost)
  from v9 (first five inputs keep v9's columns, two velocity inputs zero), v19 reward/curriculum, evaluator's planar bound.
- attempt12 (v10, destination_v20) stopped at update 615 by request: never left 0.075 (window pay 0.46-0.50). u600 at
  0.075: 8/8 pass, but speed 4-10% off the command and 7-126 mm from the ghost after 20 s (no along-track input).
- attempt13 (destination_v21, requested): ghost_v11 = v10 inputs + along-track gap. Start: starts/v9_ghost_v11.pt
  (sha 41e0e7c5..., from v9 16048b8f; side and along columns both take v9's gap weights, relative velocity zero, so the
  first layer reproduces v9 to 1e-6). v20 reward and curriculum.
- attempt13 (destination_v21, ghost_v11) stopped by request at ~update 3100: reached 0.15 at update 1065. u1250 at 0.15:
  8/8 pass, 1.7-9 mm from the ghost at 20 s, first-2-s speed 0.144-0.150. u2950: 6/8, 11-58 mm (slow starts again,
  0.12-0.13 m/s in the first 2 s), though the gap now closes slightly after the start. Window pay 0.80 -> 0.48.
- attempt14 (1a40b567, destination_v22, requested): v21 whose curriculum judges each walk's drift (gap movement from the
  walk's halfway control to its end, kernel 0.8 at 5 mm, fall 0) instead of the destination pay. Start starts/v9_ghost_v11.pt.
- attempt14 (destination_v22, drift measure) stopped by request at update ~2250: 0.075/0.1 passed at once; reached 0.15 at
  update 1065 with drift score ~0.87, but 35 of 133 walks failed the planar check (691 planar failures at 0.15). The
  planar error is within-step speed swing (25-40 mm/s), not average speed (~1 mm/s over each step).
- attempt15 (destination_v23, requested): v22 + planar error charge -0.5 (err/bound)^2 + learning_rate_min 2e-5.
  Starts from attempt14/checkpoint_update001100.pt (requested); curriculum restarts at 0.075.
- attempt15 (destination_v23) mastered 0.15 at update 747 (first curriculum completion); its suite runs on u750.
- attempt16 (destination_v24, requested): v23 with the curriculum to 0.175 m/s, from attempt15/checkpoint_update000700.pt
  (requested), learning_rate_min 2e-5. speed_suite.py now also tests 0.175 m/s. The speed_v1 clock holds 40 controls
  above 0.15, so 0.175 needs 70 mm strides (sideways reach ~66 mm).
- attempt16 (destination_v24) stopped by request at ~update 4100; u1850 at 0.175: 7/8 pass (back-left fails turn rate),
  0.173-0.176 m/s, 50-180 mm steady lag, targets at 93-96% of range, slew capped in 86-99% of controls.
- Spark: dest_v23_from_v9_ghost_v11_c36_u5000_seed20260917_20261025_001 (pack from b708fa80, binding a2b7c992...),
  destination_v23 (curriculum to 0.15, drift measure, planar charge, ghost_v11, speed clock), learning_rate_min 2e-5,
  from starts_v9_ghost_v11_20261025/v9_ghost_v11.pt (41e0e7c5, declaration 83c43acb; v9 identity embedded, derivation
  recorded). Queued and launched 2026-10-25T07:13:55Z; ~4.9 s/update, the 6 h allocation ends it near update 4400.
- Spark dest_v23_... stopped by request at update 217 (stop.request; checkpoints to 200): after the 0.1 m/s step at update
  115 the squared planar charge rose to -1.25/control and the value loss to 11000 within each episode; reward 5 -> 1.3.
  Surrogate check: the same within-episode rise follows the 0.1 step from the v9 ghost_v11 start in attempts 13/14 without
  the charge (action-limit charge to -2.4) and recovers; the squared charge made it worse natively.
- Spark: dest_v25_from_v9_ghost_v11_c36_u5000_seed20260917_20261025_001 (pack from 5c193b4e, binding 475cabb9...),
  destination_v25 = v23 with the planar charge linear and capped at -1/control. Queued and launched 2026-10-25T07:39:29Z.
- Spark dest_v25 run ended by its 6 h allocation at update 4381: recovered from the 0.125 turbulence by ~700, mastered
  0.15 at update 1341, steady to ~2600, then drift score 0.87 -> 0.49. Surrogate suite on u1350 (copied to
  native_eval/checkpoints/dest_v25_u1350): walks 7/8, 8, 8, 8, 8 at 0.05..0.15 (v9 8,8,3,0,0), stops 2,3,3,8,8 (v9 4,4,3,3,2),
  probes 9/13 -> regressed (probes, stop 0.05): fidgets after a stop at 0.05-0.1 m/s.
- finetune1 (destination_v26, requested): stop fine-tune from Spark v25 u1350, uniform 0.05-0.15 m/s walks, half stop,
  600 updates at the checkpoint's learning rate.

## 2026-10-25 stop fine-tunes from native dest_v25 u1350 (surrogate, 600 updates each)
- finetune1 (destination_v26, stop_fraction 0.5): stops fixed by u300, but the 0.05 walk end gap grew 2.62 / 3.32 / 3.96 mm at u100 / 300 / 600 (limit 2.82). u100 regressed on probes (11/13, still fidgeting), u300 and u600 on the end gap.
- finetune2 (destination_v27, stop_fraction 0.25, commit 84731256): u150 regressed (probes 11/13, stops 5/5/4 at 0.05-0.1). u300 good (gap 2.53 mm, stops 5/7/7, probes 12/13). u600 good (gap 2.77 mm, stops 8/8/8 at 0.05-0.15, probes 12/13 with quiet_32s failing joint position range only). Best candidate: finetune2 u600.
- Spark: dest_v27_from_v25u1350_ghost_v11_c36_u900_seed20260917_20261025_001 (requested: native quarter-stop fine-tune to
  900 updates, test u600 and u900). Pack from 84731256, binding 46d50790..., destination_v27 from the native dest_v25 u1350
  checkpoint (b0a2ccf3, declaration b6e2efa7), --initialize-rate checkpoint (LR 2e-5), learning_rate_min 2e-5. Queued and
  launched 2026-10-25T17:27:24Z. u600/u900 to be copied to native_eval/checkpoints/spark_v27_u* and tested on the surrogate suite.
- Spark dest_v27 (900 updates) finished 18:41Z, exit 0. Surrogate suite: u600 (9c704de0) walks 8/8 at 0.05-0.15, stops
  8,7,7,8,8, probes 12/13, 0.05 walk end gap 2.87 mm (limit 2.82) -> regressed by 0.05 mm. u900 (2ea8e849) stops 8/8 at
  every speed incl. 0.175, probes 12/13, gap 3.34 mm -> regressed. Lag again grows with training, as on the surrogate.
- finetune3 (destination_v28, requested: v9's 3.7 mm ghost pay ramped 0->1 over each run; runs 10% turns, 10% 20/32 s
  stands, 50% walk-then-stop, 30% walks at 0.05-0.15; 32 s episodes; termination 52; commit 6033be24), 600 updates from
  Spark v27 u900. u600: probes 13/13, stops 8/8 at every speed incl. 0.175, walks 8/8 to 0.15, but 0.05 walk end gap
  5.08 mm (u300 4.54) -> regressed; lag got worse (0.15: 28 -> 36 mm). Gap profile: the along-track lag forms in the
  first second (ghost starts at full speed) and is then held constant, never closed; the narrow kernel pays ~0.07 at 30 mm.
- finetune4 (destination_v29, requested: run-end pay linear, max(1 - d/25 mm, -1), ramped; commit 04832a0a), 600 updates
  from finetune3 u600. Worse: u600 probes 10/13, stops 4,2,3,7,8 at 0.05-0.15 (all fail joint position range while
  standing), 0.05 walk end gap 6.96 mm; u300 probes 11/13 (quiet probes, joint range), stops 8/8, gap 7.16 mm. Lag profile
  unchanged: forms in the first second, then held (0.15: 34 -> 39 mm). Best stopping checkpoint stays finetune3 u600.
- finetune5 (destination_v29 from Spark v27 u900, requested: same as finetune4 but from finetune3's start). u300: probes 12/13
  (quiet_32s joint range), stops 8/8 at 0.05-0.15, 0.05 walk end gap 5.07 mm; u600: probes 12/13, stops 8/8 at every speed
  incl. 0.175, gap 5.38 mm -> both regressed on the gap only. Standing intact (finetune4's standing loss came from stacking
  the fine-tunes), but the lag still forms in the first second and holds (0.15: 29.8 -> 34.6 mm at u600).
- finetune6 (destination_v29 from starts/v9_ghost_v11.pt, requested). Value loss rose to 3300 by u200 then settled (140 at
  u600). u300: not regressed (probes 13/13, walks 8/8 to 0.1 with 1.7-1.9 mm end gaps, 0.125 1/8, 0.15 0/8, stops 5,4,5,6,5)
  but not faster at 0.15 -> not good. u600: walks 8/8 to 0.1, 0.125 6/8, 0.15 0/8, gaps 1.7-2.4 mm, but stops 2,1,4 and
  probes 10/13 (fidgeting) -> regressed. The v9 lineage tracks the ghost to ~1-2 mm and closes early lag; the v25
  curriculum lineage holds a steady lag, so the lag came from the curriculum, not the reward.
- attempt17 (destination_v30, requested: v25 curriculum + v29's linear run-end pay, termination 52, commit 4659c63f), from
  finetune6 u300 (48004285). Levels 0.075 @30, 0.1 @61, 0.125 @669, 0.15 @699, complete 714; suite on u750: walks 8/8 to
  0.125, 0.15 7/8 (one off course, planar + torque), 0.05 walk end gap 1.02 mm (limit 2.82, first lineage to pass at speed),
  stops 7,5,5,8,8, probes 11/13 (both quiet probes: joint range) -> regressed on probes only. Lag at 0.15 steady ~15 mm
  (v25 u1350 13 mm), 0.05 lag 0.3 mm. Stop fidgeting again after a curriculum, as in v25.
- finetune7 (destination_v29 stop fine-tune, requested, from attempt17 u750 de4d3935). u150 GOOD: probes 12/13 (quiet_32s
  joint range), walks 8/8 to 0.125, 0.15 7/8 (0 deg: planar + torque), stops 8,8,7,8,8 and 0.175 8/8, no falls, 0.05 walk
  end gap 0.98 mm. u300 regressed (probes 11/13). u600 GOOD: probes 13/13, stops 8/8 except 0.15 7/8, but one fall in the
  0.15 m/s 0 deg walk and stop, gap 2.65 mm (limit 2.82). Lag again grows with fine-tuning (0.15: 15 -> 17 -> 19 -> 20 mm).
  Best candidate: finetune7 u150 (no falls, 1.8 mm of gap margin).
- finetune8 (destination_v29 from Spark v25 u1350 on the surrogate) stopped by request during its suite; moved to the Spark.
- Spark: dest_v29_from_v25u1350_ghost_v11_c36_u600_seed20260917_20261026_001 failed at launch (exit 1, before training):
  scripts/cli.py still capped training episodes at 20 s. Preserved. Fixed cli.py (c16d1ac8 prepare, f3978d75 cli; both
  allow 5-32 s). _002 (pack from f3978d75, binding 24fbdcb3...) queued 2026-10-26T03:59:19Z, training: destination_v29,
  32 s episodes, from Spark v25 u1350 (b0a2ccf3), LR 2e-5. u600 and u300 to be tested on the surrogate suite.
- Spark dest_v29 _002 finished 04:52Z, exit 0. Surrogate suite: u600 (ed73eb07) GOOD and clean below 0.175: probes 13/13,
  walks 8/8 at 0.05-0.15, stops 8/8 at 0.05-0.175, no falls to 0.15, 0.05 walk end gap 2.47 mm (limit 2.82); fails only
  walks at 0.175 (3/8). u300 (3983a569) GOOD: probes 12/13 (quiet_32s joint range), everything else 8/8 to 0.15, gap 2.49 mm.
  Lag at 0.15 ~18 mm steady (v25 u1350 13 mm). Best candidate overall: Spark v29 u600 (copied to native_eval/checkpoints/spark_v29_u600).
- 96 required cases (runner.case_manifest) on the surrogate for Spark v29 u600 (full96_suite.py; 90 s episode as the native
  EvaluationEnvConfig so the 70 s transition runs): 53/96 vs v9's native 16/96. translate 0.10 16/16 (v9 6), translate 0.20
  0/16, gentle arcs 15/16, combined 5/8, tight arcs 0/16, turns 3/4 (fails -0.40), stops 11/12 (forward_fast falls), quiet 2/2,
  transition 0/1 (only reverse_diagonal_turn_right, 39 vs 30 mm/s), formal 0/4 (0.00 and 0.16 fail tilt; v9 passed 0.00).
- Spark: dest_v31_from_v29u600_ghost_v11_c36_u5000_seed20260917_20261026_001 (requested: the v25-style curriculum on turning,
  +0.05 rad/s per step). destination_v31 (commit a4e1f5b4, local only): every command turns at the top or a slower level,
  half also walking 0.05-0.1 m/s, a fifth stopping; levels 0.25 -> 0.4 rad/s; walk checks now use the evaluator's yaw bound
  max(0.06, |yaw|/4). From Spark v29 u600 (ed73eb07, declaration 967ed0da), LR 1e-4 configured, floor 2e-5, 5000 updates
  (6 h allocation). Binding 56835cee...; queued and launched 2026-10-26T07:09:49Z; running at level 0.25.

## 2026-10-27 terrain1 (destination_v32 from Spark v29 u600, mixed_v1 terrain, joint slope inputs, forward walks only)
- Terrain courses, 6 starts, 0.1 m/s, crossed = walked >= 3.6 m (v29 / u500 / u1000):
  hill 4 deg 6/6/6, hill 7 deg 6/6/6, hill 9.5 deg 0/0/3, steps 1.5 cm 6/6/6,
  steps 2.5 cm 0/2/5, bumps 2-4 cm 0/0/0. No falls anywhere.
- Flat suite u1000: regressed (probes 10/13, walk 0.05 end gap 11.87 mm vs v9 0.82 mm); stops better than v9 (0.05-0.075 8/8).
- Verdict: terrain gains are real on steps and the steep hill; lag on flat ground is the cost. Not adopted.

## 2026-10-30 terrain campaign opened (jg2399/terrain-inputs, surrogate only, requested)
- Julian requested a multi-day autonomous surrogate campaign (prompt: ~/Desktop/claude/prompts/terrain-campaign-2026-10-30.md):
  terrain-aware policy inputs from the ground-truth height field, targets at 80% of computed physics ceilings, flat suite must
  stay GOOD vs v9 under unchanged rules.json, wall-clock cap 4 days from 2026-10-30 15:35 EDT. Branch jg2399/terrain-inputs
  from 7c30e482 (scene-video); local commits only, never push, never the Spark.
- Layout recorded: port-curriculum is a symlink to ~/Desktop/cupi/hexapod-cupi, so it now resolves to the campaign branch.
  port stays frozen at its older commit with local viewer edits that analyze.py imports; git -C port diff | shasum =
  4184a84e4f36f5495165b45818d0c4b6c58f3f49. Disk 827 GB free at start.
- Deviation noted: the campaign puts privileged terrain inputs on the actor; TRAINING.md prefers privileged simulation inputs
  in the critic. Pending the program lead's decision, a critic-only lookahead ablation is a required experiment and the
  closing native proposal names the tension and the onboard elevation mapping actor-side inputs imply.
- site/project.json still describes the pre-campaign roadmap state; the campaign never pushes, so the site is left alone.
- Carried suite_v9 and suite_spark_v29_u600 lack evaluation/batch_* (kept as history, unanalyzable); carried terrain_eval_v29
  and terrain_eval_v32 used an older crossed rule and pruned traces. Baselines are rebuilt fresh at this HEAD:
  suite_v9_rebuilt, suite_v29_base, runs/terrain_campaign/terrain_eval_base_v29 and terrain_eval_base_terrain1.
- Physics ceilings (runs/terrain_campaign/ceilings.py -> ceilings.json, quasi-static tripod, stance optimized inside the
  +-0.35 rad action envelope, toes coplanar, tangential share proportional to normal). Anchor: a measured 2 s standing hold
  (zero actions) gives femur 1.27 N m peak and 12.3 N per tibia; the static 6-leg model gives 1.05 N m and 12.2 N - forces
  match to 1%, torque same scale (PD settle rides above pure gravity load). Tripod on flat at the best stance: 1.05 N m
  (66% of the 1.6 cap). Torque ceiling 11 deg (11.5 first infeasible); friction arctan(1.0) = 45 deg never binds; no
  tip-over to 30 deg. That puts 9.5 deg at 94% of cap at the BEST stance - consistent with v29 0/6 and terrain1 3/6 there.
  Step ceiling is kinematic: swing-toe rise 6.15 cm inside the policy envelope (limiter alone would allow 24 cm, so the
  0.35 rad action scale binds step height; raising it is an action-space change, out of scope, named for the lead).
  Campaign targets: grade 9 deg rung crossed 6/6 (80% of 11 deg, rounded to the 1-deg staircase rung), step 4.9 cm
  (staircase extended to 5.0 cm so the target is measurable; body-climb torque on steps is unchecked statically - the
  staircase measures it), flat suite GOOD vs suite_v9_rebuilt, achieved/commanded speed retention >= 0.9 on crossed grades.
- Step 0 flat baselines rebuilt at 7c30e482: suite_v9_rebuilt (dest_v9 u600) and suite_v29_base (spark v29 u600, ed73eb07);
  analyze.py from port says v29 GOOD vs the rebuilt v9 ({regressed: false, top_speed_better: true, good: true};
  v29: probes 13/13, walks 8/8 at 0.05-0.15 and 3/8 at 0.175, stops 8/8 at every speed, 0.05 walk end gap median 2.31 mm)
  - the verdict pipeline reproduces the record at this HEAD before any code change (v29_base.verdict.json).
- Campaign start artifacts: starts/v29_terrain_v1.pt (+.json note) widens v29 u600 242->272 / 245->275 with zero columns
  and zero Adam moments, reproduction check max difference 0.0; terrains/mixed_v2.json from new make_mixed_v2.py (hills
  3-12 deg and up to 1.4 m climbs, steps 1-3 cm, side slopes 3-9 deg, bumps/flat/noise as mixed_v1; 202 features);
  runs/terrain_campaign/grade_staircase.py (hill rungs 3-15 deg) and step_staircase.py (0.5-5.0 cm) run scene.py's
  POLICY loop with live torque capture (applied_torque_abs_max_400hz per control; scene traces carry no torque).
- Terrain course baselines (terrain_eval.py at 7c30e482, runs/terrain_campaign/terrain_eval_base/, both models in one
  folder - the script's one-call interface; carried terrain_eval_v29/_v32 stay untouched). v29 u600: hills 4/7 deg 6/6,
  9.5 deg 0/6, steps 1.5 cm 6/6, 2.5 cm 0/6, bumps 0/6, no falls. terrain1 u1000: hills 4/7 deg 6/6, 9.5 deg 4/6,
  steps 1.5 and 2.5 cm 6/6, bumps 1/6, no falls. terrain1 reads better than the 2026-10-27 table (3/6 and 5/6): the
  carried folders scored with an older crossed rule, so these fresh numbers are the campaign baselines.
- terrain_v1 implemented (commit 96079076 on jg2399/terrain-inputs, local): omniwalk/mdp/footfall.py computes, per leg,
  the time to its next touchdown from the tripod schedule, advances the body's planar pose along the command for that
  long (one midpoint-heading step), carries the leg's current body-frame toe offset, and samples the terrain there plus
  4 ring points at +-3 cm - heights relative to the support height (mean world z of toes with >1 N floor contact,
  smoothing 0.2, snapped on fresh episodes, advanced once per control), clamped +-0.2 m, scaled over 0.1 m: 30 columns
  after the nine ghost_v11 values (actor 272, critic 275), zeros without a terrain. The wrapper whitelists terrain_v1
  and records it in ppo_config; evaluate.py's DeterministicPolicy gains TerrainV1Tracker so suites and scenes rebuild
  the inputs (a unit test proves training and evaluation agree over 25 controls on a box terrain; the first draft
  failed it - support smoothing advanced on every observation rebuild, now gated by env.total_controls).
  RewardConfig gains walk_forward_bias (v33: half the walks forward, half uniform; forward-only excluded) and
  EXPERIMENT_INPUTS/EXPERIMENT_CLOCKS carry v33 (terrain_v1) and v34 (ghost_v11 + slope joints). Focused tests pass
  (locomotion, surrogate incl. the two new footfall tests, amp); graphify updated; ruff clean.
- Smoke test (tc1_smoke, arm A's exact pipeline at 50 updates): train 48 s wall (0.95 s/update - a 1000-update arm is
  ~16 min, far under the 38-min attempt-era estimate), checkpoint .json records destination_input terrain_v1,
  speed_v1, lr 1e-4/floor 2e-5 and the mixed_v2 spec sha 6c95b0ab. First flat eval CRASHED: the suites evaluate under
  the v9 task, whose reward predates the gait clock, and FootfallInputs read the scaling off the task instead of the
  recorded option - fixed in 4fbd218c (wrapper passes its resolved scaling; flat envs short-circuit to zeros before
  touching the phase); rerun passes all 3 focus probes (translate 0.05 planar error 7.4 mm/s, quiet, stop). One
  grade-staircase rung (4 deg) on the smoke checkpoint: crossed 6/6, no falls - the staircase's live torque capture
  and scene loop work. Pipeline proven end to end; round 1 (A/B/C at 1000 updates, one at a time, then D evaluated)
  starts now. tc1_smoke is kept as the failed-then-fixed smoke record.
- Round 1 trained (each 1000 updates, seed 20260917, mixed_v2, --initialize-rate checkpoint, ~16 min wall each):
  tc1_a (v33 + terrain_v1 from starts/v29_terrain_v1), tc1_b (v34 = slope joints, from v29 u600), tc1_c (v33, no new
  inputs, from v29 u600). All three reward curves drift negative late (A 0.24 -> -0.81, B 1.4 -> -1.2, C 0.31 -> -1.53
  at updates 100 -> 1000).
- Round 1 flat gate (suite_tc1_* vs suite_v9_rebuilt, unchanged rules): ALL THREE REGRESSED - A probes 11/13, walk
  0.05 end gap median 4.32 mm; B 10/13, 4.04 mm; C 11/13, 4.02 mm (v9 0.79 mm); every arm lost walks at 0.15+ (0/8,
  v29 had 8/8). Arm C carries no new inputs, so the flat cost comes from terrain training under this recipe itself,
  not from terrain_v1 or slope inputs - and the v29 run mix (half-forward walks, stands, stops) does not by itself
  prevent v32's flat-lag story. Verdicts recorded in tc1_*.verdict.json; all three rejected as champions. The arms
  saved checkpoints every 50 updates, so earlier checkpoints of these same runs are the first recovery candidates.
- Round 1 terrain courses (runs/terrain_campaign/terrain_eval_tc1, u1000): A (terrain_v1) hills 4/7 deg 6/6, 9.5 deg 0/6
  but walks 1.6-1.8 m (v29 baseline: 0.4-0.8 m - A climbs the whole ramp and stalls at the crest), steps 1.5 cm 6/6,
  2.5 cm 1/6 (one 3.76 m), bumps 0/6 (~1.9 m). B: same hills, 2.5 cm 0/6 (~1.0 m), bumps ~1.9 m. C (no new inputs):
  LOST hill 7 deg (0/6, v29 6/6) and steps 1.5 cm 5/6 - this recipe at u1000 damages terrain too. At equal experience
  terrain_v1 beats both controls: A > B >= C on every terrain axis. The 9.5-deg stall at ~1.7 m is the ramp-to-top
  crest, not the slope: the climb itself is learned.
- Arm A checkpoint ladder (terrain_eval_tc1_a_ladder u200-u800 + flat gates on u400/u800): terrain numbers barely move
  u200-u600; u800 adds single 9.5 deg and 2.5 cm crossings but regresses flat (probes 10/13, gap 3.98 mm). u400 is
  flat GOOD (regressed false, top_speed_better true) - the flat break sits between u400 and u800.
- Round 1 staircases (staircase_grade_r1, staircase_step_r1; 6 starts/rung, live torque): a_u400 max 6/6 grade 7 deg,
  stalls 8 deg at ~1.5-2.0 m; v29 u600 max 6/6 7 deg, crossed-at-all 8 deg (3/6), foot-stall from 10 deg. Steps:
  a_u400 max 1.5 cm; v29 2.0 cm (3/6), 0/6 from 2.5 cm. VERDICT: no champion from round 1 - v29 u600 stays champion
  (a_u400 is flat GOOD but not better than v29 on max grade or max step; all u1000 arms flat-regressed). The campaign
  targets (9 deg 6/6, 4.9 cm) remain 2 deg and 3+ cm away; the binding failure everywhere is the crest transition.
- Round 2 (one change): terrains/mixed_v3.json from new make_mixed_v3.py - mixed_v2 tiles with the flat share raised
  0.2 -> 0.5 against the measured flat regression. tc2_a trains v33 + terrain_v1 on it, same start, seed, updates and
  learner settings as tc1_a; ladder and gates as before. If the recipe then holds flat, round 3 iterates input
  families at the crest (forward height profile along the bearing), plus the required critic-only ablation.
- Round 2 results (tc2_a): the reward curve is healthy (1.6 -> 2.0 to u500, softening late, no round-1 collapse) and
  terrain holds across the whole ladder (u400-u1000: hills 4/7 deg 6/6, 9.5 deg 0/6 at the crest ~1.5-1.8 m, steps
  1.5 cm 6/6, 2.5 cm 0/6 at the first step, bumps 0/6 ~1.9 m) - but terrain gains SATURATE at the crest and step-1
  walls at every rung. Flat gates: u600 and u1000 both regress on exactly one axis, the 0.05 walk end gap (4.75 and
  5.64 mm vs the 2.82 limit; probes 13/13 at both, stops strong, top_speed_better true). The half-flat mix fixed
  stability but not the slow-walk ghost lag, and more updates do not push past the crest. Conclusion: the terrain
  bottleneck is input horizon, not training volume - one footfall cycle sees ~10 cm at 0.1 m/s; a crest spans a body
  length.
- Round 3 (one change vs tc2_a): terrain_v2 = terrain_v1 + a 6-point forward height profile along the command bearing
  (PROFILE_DISTANCES_M 5-40 cm, same support reference, clamp and scale; actor 278, critic 281). Registered as
  destination_v35 (v33's reward unchanged; EXPERIMENT_INPUTS v35 -> terrain_v2), commit 10eb0f3e; training-vs-
  evaluation parity proven in test_footfall for v2 as for v1; starts/v29_terrain_v2.pt widens v29 by 36 zero columns
  (reproduction difference 0.0). tc3_a trains it on mixed_v3, same seed and 1000 updates as tc2_a, so A3 vs A2
  isolates the profile's addition.
- Round 3 terrain (terrain_eval_tc3_a_ladder): THE CREST IS SOLVED - tc3_a crosses hill 9.5 deg 6/6 with no falls at
  u600, u800 and u1000 (v29 0/6 at the ramp foot; terrain1 4/6 while flat-regressed; every terrain_v1 policy stalled
  at the crest ~1.7 m). The reward curve is the healthiest of the three rounds (2.0-2.6 to u600, never negative).
  A3 vs A2 at equal experience isolates the cause: the 6-point forward profile. Hills 4/7 deg and steps 1.5 cm stay
  6/6 at every rung. Remaining walls: steps 2.5 cm 0/6 (stall ~0.9 m at the first step - terrain1 shows 6/6 is
  learnable) and bumps 0/6 (~1.9 m). Flat gates on u600/u400 next; the slow-walk ghost gap is the expected risk.
- Round 3 flat gates: every rung fails on the one gap axis alone - 0.05 walk end gap 3.42 (u400), 3.10 (u600),
  4.83 (u800), 3.17 mm (u1000) vs the 2.82 limit; probes and stops pass, top_speed_better true everywhere. u600 is
  the best candidate (gap 0.28 mm over the line, 9.5 deg 6/6).
- Round 4 (one change, finetune7's recorded pattern): tc4_a fine-tunes tc3_a u600 on the FLAT floor (no --terrain),
  destination_v35 task and recorded inputs unchanged, 200 updates, same seed and learner settings. On flat ground the
  terrain inputs are zero, so the first-layer terrain columns get zero gradient - the fine-tune can move deeper
  layers but not unlearn the input wiring; the terrain ladder re-checks the crest skill afterwards rather than
  assuming it. finetune7 precedent: the gap axis recovers early (its u150 was GOOD) and over-fine-tuning regrows lag,
  so gates run at u100 and u200 first.
- Round 4 result: the flat fine-tune at learning_rate_max 1e-4 REGREW the lag instead of closing it (gap 3.33 mm at
  u100, 5.12 at u200, from 3.10 at the start) - the record's "lag again grows with fine-tuning", reproduced. Round 5
  (one change): the same 200-update flat fine-tune at learning_rate_max 2e-5 (the floor), tc5_a; if small steps also
  fail to close the 0.3 mm, the gap looks like a stable property of the terrain-trained gait, and the campaign
  consolidates: tc3_a u600 stands as the terrain result (9.5 deg 6/6) with its flat verdict honestly recorded as
  regressed-by-0.28-mm, for the lead to weigh - the gate itself stays unchanged.
- Round 5 result: the low-LR flat fine-tune regrows lag the same way (gap 3.87 mm at u100, 5.38 at u200). Both
  fine-tune directions fail identically: the 0.05-walk gap is a stable property of the terrain-trained gait, not a
  polishable blemish. Consolidation per plan: tc3_a u600 stands as the terrain flagship (9.5 deg 6/6, flat regressed
  by 0.28 mm on the one axis); its staircases run now. Next experiments target the remaining step wall (2.5 cm) and
  the required critic-only ablation; fine-tuning the gap axis is closed as a direction (two experiments, both worse).
- FLAGSHIP STAIRCASES (staircase_grade_r3, staircase_step_r3): tc3_a u600 crosses the 9 deg rung 6/6 - THE CAMPAIGN
  GRADE TARGET (80% of the 11 deg stance-optimized torque ceiling) IS MET - plus 10 deg at 3/6 (91% of ceiling), and
  at 11 deg (100% of the static ceiling) it climbs the full ramp (~1.9 m) and fails only the crest; foot-stall starts
  at 12 deg. v29: 7 deg 6/6, 8 deg 3/6. The flagship gains +2 deg on both staircase axes with no falls anywhere.
  Steps: 1.5 cm 6/6, 2.0 cm 3/6 (equal to v29), 2.5 cm 0/6 - the step target (4.9 cm) stands unmet; the first-step
  stall is the remaining wall. Round 6 (one change, winning-lineage extension): mixed_v4 with the step share raised
  (0.12 -> 0.30, flat 0.5 -> 0.35, heights 1-3.5 cm), training continued from tc3_a u600.

## 2026-10-31 terrain campaign, day 2
- Round 6 (tc6_a: tc3_a u600 continued 1000 updates on step-heavy mixed_v4): first 2.5 cm step crossings of the
  lineage (2/6 at u600) and first bump crossings (2/6 at u1000), but 9.5 deg slips to 5/6 at the best rungs and the
  step gains flicker (2/6 -> 0/6 across rungs) - learning sits at the capability edge without settling. The stall
  signature at a 2.5 cm step base: the robot drops >50 mm behind the ghost, where v29's 25 mm run-end ramp is
  clamped at -1, so no gradient reaches the climb that would close the gap. Registered destination_v36 (commit
  49cb4cde): one field, end_destination_distance_m 0.025 -> 0.075 (gradient alive to 150 mm of lag; the softer
  precision signal near the ghost is judged by the unchanged flat gate). tc7_a trains v36 from tc3_a u600 on
  mixed_v4, same settings - one change vs tc6_a.
- Round 7 (tc7_a, v36 from tc3_a u600 on mixed_v4): the wide ramp STABILIZES the crest - hill 9.5 deg 6/6 at every
  rung u400-u1000 (tc6 flickered at 5/6) - but steps 2.5 cm stay 1/6-2/6 and bumps 0/6. The step wall is now neither
  perception (inputs see the step) nor gradient (alive to 150 mm): it reads as a gait barrier at the riser contact.
  tc7_a_ext continues the lineage 1000 more updates on mixed_v4 (seed 20260918 - a continuation redraws commands, not
  a control) to see whether the u1000 flicker consolidates.
- tc7_a_ext (1000 more v36 updates on mixed_v4, seed 20260918): steps do NOT consolidate - 2.5 cm 2/6 -> 1/6 -> 0/6
  across its rungs, 1.5 cm slips to 5/6, 9.5 deg to 4/6 by its u1000, bumps pop to 2/6 then nothing holds. The
  lineage cycles capabilities instead of accumulating them: the step axis has plateaued under input and reward
  changes (tc4-tc7ext, no stable step gain in five experiments). The step wall reads as a gait barrier at the riser
  contact - a gait-level or action-scale lever, out of this campaign's input-only scope, named for the lead. The
  campaign's stable step result stays the flagship's 1.5 cm 6/6 / 2.0 cm 3/6 (equal to v29), with transient 2.5 cm
  crossings on record. tc7_a u600 (9.5 deg 6/6 stable + 2.5 cm 1/6) still gets a flat gate as a possible flagship
  upgrade before the report.
- Critic-only ablation built (commit bad6bfd4): destination_input terrain_v2_critic gives both terrain blocks to the
  critic alone, the actor keeps v29's 242 ghost_v11 inputs (evaluation uses the ghost tracker); destination_v37 =
  v35's reward; starts/v29_terrain_v2_critic.pt widens the critic 245 -> 281 (actor unchanged, reproduction 0.0).
  tc8_crit will train it on mixed_v3 with tc3_a's exact budget and seed - the actor-vs-critic evidence TRAINING.md
  asks the native proposal to carry.
- CRITIC-ONLY ABLATION RESULT (tc8_crit: v37, terrain_v2_critic, mixed_v3, tc3's exact budget and seed): hill
  9.5 deg 0/6 at EVERY rung (u400/u600/u1000), stalling at the crest ~1.7-1.9 m; hills 4/7 deg and steps 1.5 cm 6/6;
  2.5 cm and bumps 0/6. Three-level evidence at equal experience: no terrain inputs stall at the 9.5 deg FOOT
  (tc1_c, 0.5-0.7 m), critic-only climbs TO THE CREST but never over it, actor+critic (tc3_a u600) crosses 6/6.
  Terrain knowledge in the value function shapes the climb; crossing needs the ACTOR to see the terrain. This is
  the actor-vs-critic measurement the native proposal carries against TRAINING.md's privileged-inputs-in-the-critic
  preference: keeping the inputs critic-only forfeits the campaign's headline capability.
- tc7_a u600 flat gate: gap 5.04 mm (worse than the flagship's 3.10 - the wide ramp's softer precision near the
  ghost shows on flat). No flagship upgrade: tc3_a u600 stands. Experiment program closed under stopping rule (b)
  on the step axis; grade target met; ablation measured.
- Bearings on the 7-deg slope (runs/terrain_campaign/bearings_flagship, 20 s commands, 2 starts each): no falls for
  flagship or v29 in any case. Flagship: sideways 0.0498/0.05, diagonal 0.068-0.069/0.07 both axes, turns
  0.2985/0.3 rad/s with 1.5-2.3 cm net drift; v29 equal within noise. One difference: diagonal-climb torque-cap
  fraction 0.11-0.13 vs v29's 0.06-0.07. Omnidirectionality is preserved; forward stays the headline.
- Closing deliverables in flight: full96 on the flagship (95-case run, then the separate transition run, command
  shapes from the carried full96_spark_v29_u600 logs, fresh names); scenes grades_9_10_11 and steps_20_25 (2.0 cm
  crossing beside the 2.5 cm wall) simulated with view files for the Scene page; REPORT.md drafted with the ranked
  table, ceilings vs achieved, the native proposal and its risks.
- Flagship full96 (full96_tc3_u600 + _transition, command shapes from the carried v29 logs, fresh names): 37/96 vs
  v29's 53/96. The deficit is concentrated: gentle arcs 3/16 vs 15/16 (-12), combined 2/8 vs 5/8 (-3), quiet 1/2
  (-1); translations, stops, turns, tight arcs, formal and the transition sub-check all EQUAL to v29. The
  terrain-trained gait specifically lost smooth curved-path tracking - the 0.05-gap axis amplified on curves.
  Straight-line walking, stopping and turning in place are untouched. The native proposal therefore asks for the
  inputs and recipe, not this checkpoint, and names the gentle-arc family as the gate a native run must watch.
- v29 full96 rerun at this HEAD (full96_v29_rerun, requested by Julian): 51/96 vs the carried 53/96 - gentle arcs
  14/16 (was 15) and combined 4/8 (was 5) each drifted one case; every other family identical. The harness is
  stable across the revisions, and the same-revision pair is flagship 37/96 vs v29 51/96 with the deficit still
  gentle arcs (-11), combined (-2), quiet (-1). REPORT.md updated to the same-revision numbers.
- CAMPAIGN CLOSED under its stopping rules (day 2 of 4): grade target met at the flagship (9 deg 6/6; ceiling 11 deg
  reached to the crest), step axis plateaued after five one-change experiments (rule b), ablation and bearings
  measured, full96 bound. REPORT.md in runs/terrain_campaign/ holds the ranked table, ceilings vs achieved, the
  bearings and full96 numbers, the native proposal and its risks. Scenes grades_9_10_11 and steps_20_25 (+ view
  files) sit beside it; WebM recording is one click in the viewer Scene page. Branch jg2399/terrain-inputs: five
  local commits (96079076, 4fbd218c, 10eb0f3e, 49cb4cde, bad6bfd4), never pushed, per standing rules.
- Native terrain support implemented (commit fd95788d, local): Terrain.tiled repeats a spec's height map to cover
  the shared native world; EnvConfig declares terrain_spec_path / terrain_mesh_resolution_m / terrain_spawn (tied to
  the mesh floor, bounded, recorded); IsaacEnv builds the tiled height-field trimesh as /Ground, exposes the grid as
  env.terrain (the duck-typed contract footfall.py reads), and spawns resets on local ground with optional in-cell
  scatter and random heading; cli gains --terrain/--terrain-mesh-resolution/--terrain-spawn with identity recording
  (spec sha); spark.prepare mounts the spec read-only at /terrain with its hash like a checkpoint. The stock flat
  env is untouched (terrain is opt-in; flat tests unchanged). Focused suites pass; smoke checklist in
  runs/terrain_campaign/SPARK_SMOKE_PLAN.md - the decisive item is toe height vs terrain height across envs (frame
  skew would be silent otherwise).
- NATIVE STEP AUTHORIZED: Julian relayed the lead's approval in chat - "Yes i have james' approval, make a second
  versain of the issac sim called terrain issac or something like that with your updates and do the training with
  the new model on it, then run the 96 tests on the result." Plan: terrain_isaac_* named attempts from frozen packs
  of this branch (the stock sim setup stays untouched; packs are the versioning), smoke first per the plan, then
  training, then the native 96-case evaluation recording contact force and motor torque.
- Native stage mechanics (terrain_isaac): the branch reached Spark as a git bundle into /home/julian/src/hexapod
  (worktree /home/julian/src/hexapod-terrain, no push anywhere); two whitelists lagged the campaign and are fixed -
  ppo_config's destination_input (commit d01ad575) and prepare's terrain check, which must validate by allowed root
  and hash shape like a checkpoint, never a local stat (commit 1fc8d7de; that is how packs reference /srv paths from
  the laptop). The established flow reconstructed from the Oct-26 run and shell history: prepare runs on the laptop
  (torch lives here) against remote paths + a fetched copy of the admitted inputs declaration
  (flat_pilot_20260930_001/admission_128), the attempt travels as a python-tarfile tar, locomotion.spark.pack
  approve records the root review and refreezes, ACLs grant uid 1000 (orionh), and locomotion.spark.launch runs
  from the frozen source. Inputs staged under runs/julian/terrain_isaac_inputs_20261031_001 (v29_terrain_v2.pt
  5b32fba4..., its declaration b8feecf8... now carrying checkpoint_sha256 as the native loader requires,
  mixed_v3.json 920b0565...). Smoke attempt terrain_isaac_smoke_20261031_001: destination_v35, 128 envs, 50
  updates, seed 20260917, terrain mounted at /terrain with spawn scatter; preflight PASSED recording the shared
  workloads (two llama containers, two venv pythons; 106 GB free); launched.
- Visualization for Julian: scene grades_9_10_11 (flagship on 9/10/11 deg hills, three lanes) simulated and shown on
  the viewer Scene page; at 11 deg (the static ceiling) it climbs the whole ramp and parks at the crest. The page's
  ?scene= URL loader silently fails (CORS was one cause - runs are now served by runs/terrain_campaign/serve_runs.py
  with Access-Control-Allow-Origin - but the effect still no-ops; loading through the page's own file-open path
  works). Viewer follow-up for later, not a campaign fix.
- Native terrain floor crash isolated and fixed (commit 40141bd0). Smoke _005 (0.08 m mesh) and _006 (0.16 m) died
  in Isaac scene setup; in-container probes (terrain_isaac_probe_001, compose-run pattern) showed Terrain.tiled and
  the geometry fine while isaaclab's create_prim_from_mesh aborts from a few thousand faces up (17k triangles
  already die standalone; the stock floor is 2 triangles, which is why nobody hit it), with the aarch64 container's
  USD imaging libraries failing to load (reloc 0x00020100). Probe-004 separately hung 3 h before Kit on a carb
  plugin-load futex (a flaky circular-dependency startup race; an earlier probe logged the matching error and
  survived) - killed, relaunched with a 25-minute kill timer and line-buffered output as probe-005, which PASSED:
  direct UsdGeom.Mesh quad authoring at 8,649 / 34,969 / 562,500 quads plus app updates, clean. env.py's terrain
  branch now authors /Ground/mesh directly (float32 points, int32 quad indices on the old triangles' vertices and
  diagonal, subdivisionScheme none, CollisionAPI + MeshCollisionAPI none, material spawned at
  /Ground/physicsMaterial and bound); flat branch and the collider/material discovery untouched. Independent diff
  review: no blockers (notes: triangulation diagonal now the cooker's choice, bounded by per-cell nonplanarity;
  cooking headroom above 562k quads unverified - default 0.08 m resolution keeps runs near probe scale; terrain
  renders faceted, collision unaffected). Local suites green; tests/ also caught missing source-inventory entries
  for footfall.py and locomotion/terrain.py from the earlier commits - added, surrogate/terrain.py described as the
  re-export it now is. Physics source changed, so admission round 4 (standing1/128 _004 + admission _004) runs
  before smoke _007; new pack freeze 89616471..., re-frozen on approve.
- Admission round 4 passed (standing1/128 _004 all_pass true, admission _004 candidate written, its
  candidate/inputs.json is the next declaration - fetched as terrain_isaac_inputs_004.json). Smoke _007
  (destination_v35, 128 envs, 50 updates, 0.08 m mesh) then segfaulted right after "Completed setting up the
  environment"; _008, byte-identical, died at the same line - deterministic, not the flaky startup race. Isolation
  so far, all in-container probes against the frozen _007/_009 source: stepping probe with a real SimulationContext
  (box drop, PhysX cooks in ~3 s) passes at smoke and full density for triangles AND quads with correct rest
  height, so the mesh collider is healthy; 128 replicated hexapods (passive, env's exact contact view,
  max_contact_data_count=1024xN, 600 steps, forces read every step) also pass, and a no-view variant hung
  mid-stepping once (same flaky-build family as the startup deadlock). Faulthandler via env var is clobbered by
  Kit's crash handler (no stack, no core - apport drops container cores). Instrumenting env.py directly was
  correctly refused by the standing admission gate ("Standing source or neutral stance differs"), so the probe now
  monkeypatches from scripts/cli.py - outside the pinned physics closure, physics bytes admitted - and pack
  refreeze keeps the source-freeze check green. That run pinned the crash: the reset readback completes
  (write_reset, then a full read_state), then the process dies before the first control (_apply_action never
  runs), while the main thread is in torch/CPU code only - consistent with a background Kit/PhysX thread fault
  timed by the terrain stage being live. Next: same probe with faulthandler re-armed AFTER app start (beats Kit's
  handler) plus a heartbeat thread, to get all-thread stacks at the fault.
- Root cause found and fixed (commit b3402a35). The re-armed faulthandler (probe_002) caught the fault mid-import:
  a bare "import triton" kills the container's python - the GB10 image's bundled triton native library faults on
  load, verified standalone with a two-line container run. The chain into it: rsl_rl's OnPolicyRunner import
  (train.py training_runner) pulls tensordict, whose import wraps functorch through
  torch.compiler.allow_in_graph, which imports torch._dynamo, which imports triton whenever the package is
  present. The flat Oct-26 run predates whatever populated triton/tensordict into the shared container volumes,
  which is why it never tripped. torch probes triton with has_triton_package and treats ImportError as absent
  (the normal aarch64 path), so prepare now writes source/triton/__init__.py raising ImportError into every pack
  ahead of the freeze manifest; unit test asserts it ships. Physics sources untouched, admission _004 stays
  valid. Verified in probe_003: with the shim the terrain standing phase ran through and training reached
  checkpoint loading - where it exposed a second, ordinary bug: the widened start's declaration
  (v29_terrain_v2.json) lacked the identity/updates record the native loader requires
  (start_from_checkpoint reads the sidecar as the checkpoint record; require_embedded_declaration matches it
  against the .pt's infos). The .pt itself carries v29's full identity (verified on the laptop), so the
  declaration was amended in place - checkpoint bytes unchanged, old declaration kept
  (starts/v29_terrain_v2.predeclid.json and inputs_001 on the Spark) - and the builder script now embeds
  identity/updates/checkpoint_sha256 for reproducibility. New inputs dir terrain_isaac_inputs_20261031_002
  (same .pt bytes, amended declaration 543d2e63...); smoke_010 rebuilt at b3402a35 (shim in the pack) and queued
  through the real launcher behind other GPU jobs.

## 2026-10-31 — terrain training speed: smoke_010 stopped at iteration 8, timing probe launched

terrain_isaac_smoke_20261031_010 (source b3402a35, start declaration 543d2e63…) ran clean past
every earlier failure point: standing, direct-authored mesh, widened-checkpoint load, then 8 PPO
iterations. Reward −3432 → −1786 and mean episode length 11 → 134 steps over iterations 0–8, so
the flat-trained start adapts on terrain. I stopped the container at iteration 8 (docker kill
after verifying the mounts were ours): iteration time measured 414–416 s with our process alone
at 95% SM (neighbors idle per nvidia-smi pmon), against 8.6 s/it for the flat u2500 run — a 48×
slowdown. The allocation profiles cap one launch at 21600 s (flat_pilot_v1; standard 6600), so
600 updates at this speed cannot run as one launch; speed, not budget, is the blocker. The mesh
is modest (20×20 m spec at 0.08 m → ~125k triangles), so plain narrowphase is an unlikely sole
cause. terrain_isaac_probe_004 (same pack, --updates 2) carries a timing patch in
scripts/cli.py — outside the pinned physics closure; pack approve refreezes — attributing each
control step to sim.step / contact collection / read_state / observations / wrapper Python,
printed cumulatively every 60 s. The next change follows that attribution; no gates, thresholds
or inputs were changed.

## 2026-10-31 (later) — offsets hypothesis refuted; the cost is a fixed per-substep tax

probe_014 A/B (standalone scene, 128 robots, tiled mesh 562,500 quads): unauthored collider
offsets 1493–2202 ms/substep; authored 0.02/0.0 offsets 1371–1426 ms/substep. The explicit
offsets are physically right (and ~1.4x faster; auto margins also showed larger contact
spikes), but they are not the 50x. The decisive observation: the first 50-substep window has
zero contacts (robots still falling, peak 0 N) at full cost in both variants, so the cost is
contact-independent — a fixed per-substep tax tied to the mesh's presence. The full Kit log
shows no PhysX fallback, capacity or mesh warnings. The env.py offsets change and its test
stay drafted locally, uncommitted: one admission round will carry the complete fix once the
axis is known. probe_015 (three variants, same scaffold, offsets on) scales each axis:
PROBE_ENVS=1 (per-pair midphase), PROBE_EXTENT=9 for one 20 m tile = 62,500 quads
(per-triangle work), PROBE_VIEW=0 (contact view/report machinery).

## 2026-11-01 — cause pinned and fixed: chunk the terrain colliders (commit d50af2d1)

probe_015 scaled the fixed per-substep cost along each axis: it is ~linear in mesh faces
(562k -> 62.5k faces: 1495 -> 114 ms/substep at 128 robots), sublinear in robots (128 -> 1:
114 -> 6 ms at equal faces), and unchanged without the contact view. This image's PhysX scans
every triangle of a paired mesh each substep (~20 ns per face per robot), contacts irrelevant.
probe_016 validated the counter: the same grid authored as ~5 m chunk colliders (169 prims,
seam vertices shared) drops the contact-free substep 1495 -> 7.1 ms (~210x) and settled
stepping ~10x; 10 m chunks are uniformly worse than 5 m. Commit d50af2d1 authors the decimated
grid as one collider per 5 m square via Terrain.chunk_grid (pure, CPU-tested: union is the
grid cell for cell, seams shared), filters the contact view on every chunk, sums per-chunk
forces in collect_substep (seam contacts split across two chunks; the sum is the total), and
keeps the explicit 0.02/0.0 offsets on each chunk. The stock-floor offset declaration stays
None, so flat runs and the 96-case evaluation declare unchanged physics. All required CPU
checks pass (371 locomotion tests; ruff; inventories). Admission round 5 (standing1/128 at
d50af2d1 + candidate) is queued on the Spark; then smoke_011 measures the real iteration time
and, if it fits the allocation, the 600-update training follows.

## 2026-11-01 — Admission round 5 green at d50af2d1; smoke_011 queued

Round 5 (standing1 + standing128 at d50af2d1, flat, unchanged gates) passed:
both runs "completed" with all_pass true, candidate written to
terrain_isaac_admission_20261101_005/candidate/{admission.json, inputs.json}
(admission.json fc87f20c…). Parity note: both rounds' standing_report.json are
byte-identical to round 4's (standing1 c27dcae6…, standing128 879e8956…) — the
chunked-collider commit left flat physics unchanged, as designed. Runner note:
the first round-5 runner died after standing1 because a clean launch log gave
the error-grep nothing to match under set -e; resumed with run_round5b.sh
(full logs to files, no abort-on-empty-grep). standing1 itself passed before
the wrapper died; no rerun needed.

smoke_011 prepared at d50af2d1 (freeze 4d75aa0a…) from candidate inputs_005:
same configuration as smoke_010 (destination_v35, 128 envs, 50 updates, seed
20260917, mixed_v3 @ 0.08 m, --terrain-spawn, v29_terrain_v2 start 5b32fba4…,
declaration 543d2e63…). Purpose: measure the fixed per-update time to size the
600-update run. Queued behind a 75%-GPU workload (guard holds ≤25%).

2026-11-01 — smoke_011 failed at env creation; verifier taught the chunked floor (4a90b7fb)
- terrain_isaac_smoke_20261101_011 (50 terrain updates at d50af2d1) failed in its standing
  phase, 38 s in: `ValueError("Actual floor collider differs from the declared floor")` from
  verify_native_recipe. The chunked-collider commit authors explicit 0.02/0.0 offsets on each
  terrain chunk while the config knobs stay None, and the verifier still applied the flat rule
  "declared None -> actual (None, None)" to the first chunk. Flat round 5 passed because
  chunking never engages on flat; smoke_010 passed because it predated the explicit offsets.
- Fix (commit 4a90b7fb, jg2399/terrain-inputs): the terrain collider recipe
  (TERRAIN_CONTACT_OFFSET_M/TERRAIN_REST_OFFSET_M/TERRAIN_CHUNK_M) moved to
  locomotion.terrain (hash-bound in physics_source_files); verify_native_recipe now checks
  every chunk against it — collision API, both offsets, exact-triangle approximation, bound
  canonical material — and records the chunk count. Flat readback is byte-identical by
  construction; the stand-in tests cover a passing chunked floor and each fault on the
  second chunk. 373/373 locomotion CPU tests, ruff clean.
- Source hashes changed, so round 6 (standing1_006 + standing128_006 + admission recompute,
  freeze fab91e6d) queued on Spark at GPU 0%; bindings diff from round 5 only in paths and
  freeze sha. Expect standing reports byte-identical to rounds 4/5 (c27dcae6/879e8956);
  the runner echoes their sha256s. Then smoke_012 from the round-6 candidate declaration.

2026-11-01 — Round 6 green, smoke_012 green at the fix; u600 launched
- Round 6 (freeze fab91e6d at 4a90b7fb): standing1/standing128 all_pass, reports
  byte-identical to rounds 4/5 (c27dcae6/879e8956) — the chunk-aware verifier left flat
  physics unchanged. Candidate declaration = round 5 plus 11 path/hash swaps
  (admission.json de90bb73).
- terrain_isaac_smoke_20261101_012 (50 terrain updates, binding 4052e0d3): completed,
  errors [], wall 711.8 s ≈ 14.2 s/update amortized (13.2–13.3 s/it steady) vs 414 s/it
  pre-chunking. Reward −1452 → +706 (iter ~40), tail at −168 (iter 50); episode length
  ~530 steps. checkpoint_update000050.pt and force_metrics.json written.
- terrain_isaac_ppo_u600_seed20260917_20261101_001 (binding 2a450804): smoke CLI verbatim
  with --updates 600 from the v29-widened start, same round-6 declaration; expected
  ~8.6 ks, cap 21.6 ks, runner timeout 16 ks. Launched at GPU 0%. Next: native 96-case
  evaluation on its final checkpoint.

2026-11-01 — u600 completed; 96-case judgment moved to the surrogate (native refusal)
- terrain_isaac_ppo_u600_seed20260917_20261101_001 completed: errors [], 600 updates,
  2,764,800 transitions, wall 8,084 s (13.5 s/update). Final checkpoint
  checkpoint_update000600.pt sha d5dd16e3…25dd01, declaration 49535005…928d,
  force_metrics.json 4e965f77…20b6 (ground-contact force + motor torque recorded).
  Intermediate checkpoints every 50 updates. Reward: −1452 (it 0) → +516 (it 40), then
  noisy mean around −50…+35 with excursions (−2735 at it 400, recovered); tail −87…−335;
  episode length grew ~530 (smoke it 50) → ~650–677 steps. Training terrain mixed_v3
  (920b0565…), destination_v35, seed 20260917, freeze fab91e6d, admission round 6.
- Native full-96 on this checkpoint is refused by frozen scripts/play.py
  (evaluation_policy): "Native evaluation does not follow a speed-scaled gait clock yet;
  evaluate this checkpoint with locomotion.surrogate.evaluate" — destination_v35 records
  gait_clock_scaling speed_v1, and dropping it would fail the ppo_config equality check
  instead. Every post-v9 checkpoint (v29 51–53/96, flagship 37/96) was judged on the
  surrogate full-96 for the same reason; the native evaluator's full-96 remains v9-only.
- Launched surrogate full96_suite.py (96 required cases, 90 s timeout) on the staged u600
  checkpoint (runs/native_eval/checkpoints/terrain_isaac_u600/, sha verified) at clean
  4a90b7fb → full96_terrain_isaac_u600; then a fresh v29 control at the same revision →
  full96_v29_ctrl_4a90b7fb (prior v29 numbers came from bad6bfd4-dirty). Gate to watch:
  gentle arcs (the flagship's −11 family); scoring = pass count over the 96 recorded cases.

2026-11-01 — Surrogate full-96 on the native terrain u600: 35/96 vs v29 control 51/96
- full96_terrain_isaac_u600 (checkpoint d5dd16e3…, native terrain_isaac_ppo_u600, seed
  20260917, wrapper terrain_v2 + speed_v1 reconstructed from the record): 35/96.
  full96_v29_ctrl_4a90b7fb (ed73eb07…): 51/96, per-family identical to the bad6bfd4-era
  rerun — the harness is stable at this revision. Both at 4a90b7fb (tree dirty only with
  the uncommitted native-eval draft, src/locomotion/scripts/play.py +
  tests/test_destination.py, which the surrogate never imports).
- Family deltas vs the control: gentle arcs 3/16 vs 14/16 (−11), translations 13/32 vs
  16/32 (−3), combined 1/8 vs 4/8 (−3), formal 1/4 vs 0/4 (+1); turns, stops, quiet,
  static, tight arcs equal. The native terrain training reproduces the surrogate
  flagship's trade-off (tc3_a 37/96, gentle arcs 3/16): terrain capability costs smooth
  curved-path tracking on flat, now confirmed with native physics in the loop. The
  gentle-arc family remains the gate any acceptance run must watch; recovering it is a
  recipe/experiment decision (lead's call), not a bug fix.

2026-11-01 — Review found the native terrain inputs dead; adapter + footfall fixed; round 7 queued
- Adversarial review of the native-eval draft (branch jg2399/native-eval-inputs, 106eaba1 —
  play.py rebuilds post-v9 inputs: speed clock + ghost/terrain trackers; 0 confirmed
  findings) surfaced that OmniEnv never aliases isaac.terrain: FootfallInputs reads the
  terrain off the task's env (the adapter), so terrain_isaac_ppo_u600 (d5dd16e3) and every
  native smoke trained with all 36 terrain input values pinned to zero. Terrain physics and
  spawn were real; the reward reads no terrain; the surrogate campaign is unaffected (its
  env exposes .terrain directly).
- Fixes on jg2399/terrain-inputs: 308a3438 (adapter alias + aliasing-contract test carrying
  a terrain stand-in through a real reset; admission Scene double gains the stock
  terrain=None) and 1f1934e6 (footfall math on detached cpu copies — the visible grid would
  otherwise mix cuda state with the cpu float64 grid; cpu compute keeps native features
  bitwise equal to the surrogate). All eight required suites green at 1f1934e6.
- adapter.py is a physics source → round 7 queued on Spark at GPU 0% (standing1/128_007 +
  admission recompute, freeze 4515388e at 1f1934e6); flat reports expected byte-identical
  to rounds 4/5/6 (c27dcae6/879e8956). Then smoke_013 and the u600_002 retrain with LIVE
  inputs — the native input ablation against u600's zeros — and the surrogate full-96 on
  the result. The native full-96 itself waits on the lead (play.py is frozen).
- Proposal for the lead: runs/curriculum/native_eval_inputs_proposal.md (both changes, the
  evidence, identity consequences, and the run plan).

2026-11-01 — Round 7 green (flat parity held), smoke_013 proves the inputs live; u600_002 launched
- Round 7 (freeze 4515388e at 1f1934e6): standing1/128_007 all_pass with reports
  byte-identical to rounds 4/5/6 (c27dcae6/879e8956) — the adapter alias and cpu-copy
  footfall math changed no flat physics. New admission 49a4e207; declaration diff vs
  round 6 = path/hash swaps only.
- terrain_isaac_smoke_20261101_013 (50 live-input updates, binding 0a67c853): completed,
  errors [], ~14.7–15.8 s/it (cpu copies cost ~1.5–2.5 s/it over smoke_012's 13.2–13.3).
  Canary vs smoke_012 (same seed/CLI, zeroed inputs, physics byte-identical): iteration 0
  matches to 2 dp (near-identical first rollout under tiny input-driven mean shifts), then
  diverges fully from iteration 10 (52.90 vs −220.24; it 40: 555.15 vs 516.53) — the only
  changed signal is the terrain block, so the inputs demonstrably reach the policy.
- terrain_isaac_ppo_u600_seed20260917_20261101_002 launched at GPU 0% (binding diff vs
  smoke_013: --updates 50→600 only; expected ~9.3 ks, cap 21.6 ks, runner timeout 16 ks).
  On green: surrogate full-96 + the native input ablation u600_001 (zeros) vs u600_002
  (live). The native full-96 still waits on the lead's review of the play.py draft.

2026-11-01 — u600_002 (the first live-input native terrain training) completed clean
- terrain_isaac_ppo_u600_seed20260917_20261101_002: launch exit 0, status completed,
  errors []. Binding fe96653c; 600 updates in ~2 h 27 m wall (~14.7 s/it, the live-input
  rate smoke_013 predicted). Checkpoint checkpoint_update000600.pt sha
  e29be0bcc7195e6c9a053e09de8ef5eb32ac990576ad188d11f94689e2508d17, declaration
  6a3994d038acfbd1f73ece5dcde55bdfdb9c8c2789c26ba93127fb37f4cec568; staged locally at
  runs/native_eval/checkpoints/terrain_isaac_u600_002/ with shas verified against Spark.
- Mean reward sampled every 50 updates: −1452.46 (update 0, equal to the smokes' first
  rollout) → −81.66 → −6.90 → −690.70 → −351.84 → 358.21 → −1045.28 → 121.06 → 14.30 →
  −314.69 → −345.85 → −16.49 → 342.59 (final; episode length 609). Noisy as all terrain
  runs are; the judge is the full-96, not the reward curve.
- Surrogate full-96 launched on the checkpoint (output full96_terrain_isaac_u600_002).
  Comparators at this harness: u600_001 35/96 (inputs zeroed, checkpoint d5dd16e3) and the
  v29 control 51/96 (ed73eb07) — u600_001 vs u600_002 is the clean native input ablation
  (same CLI, seed, freeze lineage, terrain; only the input visibility differs plus the
  adapter/footfall fix that provides it).

2026-11-01 — Input ablation, flat leg: u600_002 33/96 — statistically unchanged from u600_001
- full96_terrain_isaac_u600_002 (checkpoint e29be0bc): 33/96 vs u600_001 (d5dd16e3) 35/96
  and the v29 control (ed73eb07) 51/96, all at this harness. Family table (v29 / 001 / 002):
  gentle_arc 14-3-4 of 16, tight_arc 0-0-0 of 16, translate 16-13-12 of 32, turn 5-5-4 of 6,
  stop 9-9-9 of 10, quiet 2-2-2 of 2, combined 4-1-0 of 8, formal 0-1-1 of 4, transition
  0-0-0 of 1, static 1-1-1 of 1.
- Per-case flips 001→002: 5 gained, 7 lost, scattered across families (list in the suite
  outputs) — a ±2 net swing on 96 binary cases, bidirectional, no family signature. On the
  flat suite the terrain features are zeros by contract either way, so this leg of the
  ablation only measures training-dynamics side effects of live inputs: none material.
  The gentle-arc regression (4/16 vs v29's 14/16) remains the terrain-training cost itself,
  unchanged by input visibility — consistent with tc3_a's pattern.
- The decisive leg — whether live inputs improve walking ON terrain — runs now:
  terrain_eval.py (6 courses × 6 starts, crossed ≥3.6 m at 0.1 m/s) on u600_001 vs
  u600_002, output terrain_eval_u600_ablation. Context at this judge: v29 0/6 on every
  course; surrogate v32_u1000 crosses hills 4/7 deg 6/6, 9.5 deg 3/6, steps 6/6 and 5/6,
  bumps 0/6.

2026-11-01 — Input ablation, terrain leg: live inputs nearly triple progress on the crest
- terrain_eval_u600_ablation (terrain_eval.py, 6 courses × 6 starts each, 0.1 m/s, crossed
  ≥3.6 m; u600_001 d5dd16e3 zeros vs u600_002 e29be0bc live; same CLI/seed/terrain
  training, 36 trials each, zero falls for either):
  course         001 crossed  002 crossed  001 med/worst m  002 med/worst m
  hill 4 deg        6/6          6/6         3.79/3.79        3.79/3.79
  hill 7 deg        6/6          6/6         3.79/3.78        3.79/3.78
  hill 9.5 deg      0/6          0/6         0.60/0.49        1.70/1.58
  steps 1.5 cm      5/6          6/6         3.79/0.20        3.79/3.79
  steps 2.5 cm      0/6          0/6         0.89/0.11        0.89/0.12
  bumps 2-4 cm      0/6          0/6         1.61/0.70        1.42/1.32
  TOTAL crossed: 17/36 vs 18/36.
- Reading: the inputs matter exactly where terrain knowledge should — the steepest hill
  (median walked 0.60 → 1.70 m, 2.8×; worst 0.49 → 1.58 m, 3.2×, consistent across all
  six starts) and the marginal step course (the one stalled start now crosses). Bumps:
  spread tightens (worst 0.70 → 1.32 m) at a slightly lower median, no crossings either
  way. Steps 2.5 cm unchanged — 600 updates isn't converged there (surrogate v32 needed
  1000+ updates to cross it 5/6 with the same input family).
- With the flat leg (33 vs 35 of 96 — no material flat cost), the ablation answers the
  campaign's question: the adapter fix is worth keeping, and the input machinery earns
  its place natively. Both legs are in the proposal for the lead.
- Next decisions are the lead's (relayed by Julian): the play.py draft review (unblocks
  the native full-96 at one freeze), and the gentle-arc recovery recipe (arc-rich terrain
  commands vs interleaved flat batches) — longer live-input training naturally pairs with
  that recipe choice rather than preceding it, since terrain-only updates deepen the known
  flat-arc regression (33–35/96 vs v29's 51). Prior evidence on recovery: post-hoc flat
  fine-tuning is a closed direction (tc4_a/tc5_a both regrew the gap at either learning
  rate), so the candidates are the two during-training recipes.
- The post-run eval triple is now one command: eval_checkpoint.py <name> <checkpoint.pt>
  runs both judges (full-96 + terrain courses), scores against the recorded comparators
  (v29_ctrl, u600_001, u600_002), and prints a hash-bound block; recording in this file
  stays deliberate. Scorers verified to reproduce the recorded 33/96 and the 1.70 m
  hill-9.5 median from the raw outputs.

## 2026-11-01 — NEW CAMPAIGN TARGET (Julian, direct instruction)
"Take u600_002 and make it fully consistent on the terrain benchmarks it can't currently
handle and pass all of the 96 tests that do not involve impossible speed or moving while
turning; also pass omnidirectional 0.15 m/s. Surrogate first; queue on Spark when a version
should work; repeat until accomplished." Base checkpoint: u600_002 (e29be0bc). Judges, gates
and the 96 manifest stay untouched; the subset below is an acceptance scope, not a suite change.

SCOPE RULING (from the case definitions in evaluation/scoring.py and runner.py):
- Excluded, moving while turning (command mixes planar velocity with yaw): 16 gentle_arc
  (0.15 + 0.05 yaw), 16 tight_arc (0.20 + 0.40 yaw), 8 combined (0.10 + 0.20 yaw), and
  transition__full_program (its only failing segment, reverse_diagonal_turn_right
  [-0.071,-0.071,-0.20], is a moving turn; 13 of its 14 segments already pass).
- Excluded, infeasible speed (0.20 m/s tier: computed torque demand >1.6 Nm beyond the
  0.5% omni bound for every checkpoint ever measured, v29 included): 16 translate_0.20,
  formal__0.20, formal__0.30, stop__forward_fast (its approach is 0.20 forward).
- IN SCOPE: 36 cases — stand, 16 translate_0.10, turn ±0.20/±0.40, quiet×2,
  formal__0.00/__0.16, 11 stops. u600_002 passes 29/36.

THE SEVEN FLAT FAILURES (u600_002, margins vs bounds):
- translate_0.10 at 112.5/157.5/180 deg: planar error 0.0254/0.0276/0.0270 vs 0.025 —
  2–10% over. At 90 deg: planar 0.0282 AND demand 0.0056 vs 0.005. The failing sector is
  left/rear; right-side mirrors pass. Root cause: walk_forward_bias 0.5 (v33 mix) starves
  rear/side bearings of practice.
- turn_+0.40: yaw error 0.247 vs 0.10, demand 0.0415 vs 0.005, terminated; turn_-0.40:
  yaw 0.357, demand 0.0417 (v29 passes +0.40, fails -0.40). Root cause: the run mix only
  ever commands turns at 0.2 rad/s (fixed_yaw_scale_rad_s 0.2); 0.4 was never trained.
- formal__0.16: tilt_rms 1.099 vs 0.90 (MOVING_RMS) — 22% over; all other formal bounds
  pass. No formal moving case has ever passed (v29 fails too): a new quality level, not a
  regression.
ALSO REQUIRED: omnidirectional 0.15 m/s × 16 bearings — new judge omni15_suite.py (adds
cases only; bounds are the evaluator's omni_static: planar ≤ 0.0375 at 0.15, demand ≤ 0.005).
Baseline on u600_002 running (omni15_u600_002). Walks trained 0.05–0.15, forward-biased.

TERRAIN TARGET (terrain_eval.py, all 6/6 with no falls): hill 9.5° (now 0/6, median 1.70 m,
crest-foot stall), steps 2.5 cm (0/6, first-riser stall ~0.9 m; five input/reward experiments
plateaued — recorded as a gait barrier at riser contact; NO foot-clearance term exists in
RewardConfig — the identified gait-level lever), bumps 2–4 cm (0/6, best transients 2/6),
while hills 4°/7° and steps 1.5 cm stay 6/6.

Method: surrogate rounds from u600_002, one-change discipline where feasible, judged by
subset-36 + omni15 + terrain courses (the full 96 still recorded each round for the ledger);
Spark native retrain when a surrogate round holds all targets. Reward/mix changes are
campaign-normal; model INPUT changes would need the lead's OK (none planned).
- omni15 baseline (u600_002, omni15_u600_002): 4/16 — forward sector only (0/22.5/315/337.5).
  Twelve bearings fail, ten of them on computed demand over rating (0.0056–0.0202 vs 0.005;
  worst at rear bearings 180–225 deg, where planar error also peaks at 0.0544–0.0568 vs
  0.0375). Side/rear walking at 0.15 m/s overworks the motors: a gait-efficiency target,
  not a tracking shave. Forward passes with margin (planar 0.0233, demand 0.0033).

2026-11-01 — Round A designed (adversarial panel), tc9_a launched
- Design panel (3 designers × 2 adversarial judges, scores 6.5-7.5 "revise then execute"):
  convergent amendments adopted — flat share stays 0.5 (make_mixed_v3's recorded lever;
  mixed_v4's 0.35 lineage cycled), no reward changes in round A (terrain1 u1000 crossed
  steps 2.5 cm 6/6 with NO clearance term — the record's only 2.5 cm success — so
  distribution gets its shot first; the clearance term is round B if steps still stall),
  the proposed riser_impact force term rejected (horizontal stance friction makes it
  nonzero on flat — physically wrong), turns train at both drawn rates {0.2, 0.4}
  (v29 passes turn_+0.40 within every bound: feasibility proof), formal tilt decomposition
  to run from the existing trace before choosing that lever. Panel journal:
  workflows wf_c7cf2225-3b1.
- Code (branch jg2399/terrain-rounds, commit 92f145fe, 374 locomotion tests OK, ruff clean,
  graph updated): destination_v38 = v35's reward with walk_forward_bias 0.25, run_mix
  (0.15, 0.10, 0.45, 0.30), walk_speed_max 0.17, and the new turn_rate_choices_rad_s
  (0.2, 0.4) knob (default off; the RNG stream of every experiment without it is
  byte-identical — the draw only widens when the knob is set). EXPERIMENT_INPUTS/CLOCKS
  carry v38 as terrain_v2 + speed_v1.
- Terrain mixed_v5 (terrains/make_mixed_v5.py, sha 1a859fccb7413e597c2d057f6ea7aff9394001a
  27bcfaaf46e785ca0bbc2c8ef, 115 features): flat 0.5 kept; bumps 0.13→0.16 at 2-4 cm (the
  course's own heights); step risers 1.5-3 cm (mass toward 2.5, not step-heavy); side
  slopes 0.09→0.06 pay for it; hills unchanged 3-12° (9.5 already interior).
- tc9_a launched in port-curriculum at 92f145fe: 1000 updates, destination_v38, mixed_v5,
  initialized from u600_002 (e29be0bc, initialize-rate checkpoint), lr 1e-4, std 0.05/0.05,
  episode 32 s, seed 20260917, terrain_v2 + speed_v1 wrapper, checkpoints every 50.
- Judges on completion (eval_checkpoint.py triple + omni15): subset-36 ≥ 33 with the four
  left/rear translations and both ±0.40 turns passing; omni15 materially above 4/16 (the
  demand failures may need a round-B efficiency lever — distribution's reach is the test);
  terrain: 4/7° hills and 1.5 cm steps hold 6/6, hill 9.5° ≥ 3/6 crossed, bumps median
  > 2.0 m; steps 2.5 improvement is a bonus (round B owns that wall). Abort: subset-36
  < 29 or any 6/6 course lost → rollback per the panel's isolation path.
- Formal tilt decomposition (the panel's free diagnostic, from the recorded batch_093 trace
  of full96_terrain_isaac_u600_002): tilt_rms 1.099° = static mean 1.090° + oscillation rms
  0.141° (dominant period 0.26 s, 7% share). The robot holds a steady body lean at 0.16 m/s;
  there is no wobble to smooth. The lever is posture at speed, not tilt smoothing — and a
  global tilt_weight raise is the wrong tool anyway (on a ramp the body SHOULD pitch with
  the slope; the existing tilt term already prices 1.09° at a negligible ~0.024/control).
  Decision deferred to round B/C with tc9_a's evidence; candidates: speed-conditioned
  posture shaping on flat, or reporting formal__0.16 as a residual quality item.

2026-11-01 — Round A result (tc9_a): early gain, late drift; abort floor hit at u1000
- tc9_a trained clean (1000 updates, 949 s, ~0.95 s/update; adaptive LR fell to 1e-5 by
  u100). Final checkpoint 41e8c5a3, declaration 3876d792.
- omni15 across rungs: u600_002 init 4/16 → u200 5/16 (demand median 0.0076→0.0053, the
  best omni state yet) → u500 3/16 → u1000 1/16 (planar med 0.043, demand med 0.011,
  67.5° terminates). Quality peaks early and degrades monotonically at low LR — the
  tc6/tc7ext non-consolidation signature on the flat axis.
- u1000 triple: full-96 26/96 (translate 10/32, quiet 1/2 — a quiet case lost, formal 0/4;
  the subset-36 is well under the 29 abort floor). Terrain: hill 9.5° 2/6 CROSSED (the
  native lineage's first crest crossings; median 0.69 — bimodal), steps 1.5 held 6/6,
  bumps worst 0.70→1.30 m, steps 2.5 unchanged. The flat-vs-terrain trade at its
  sharpest; the abort rule fires for u1000 as a candidate, not for the round.
- Actions per the panel's isolation path: (1) the u200 rung runs the full triple now (the
  omni evidence makes it the round's real candidate; eval_tc9_a_u200); (2) the continuation
  CONTROL launched — tc9_ctrl, destination_v35 unchanged on mixed_v3, same CLI/seed/updates
  from u600_002 — to attribute the drift: if the control drifts the same, prolonged
  surrogate continuation itself is the cause (v38/mixed_v5 exonerated); if the control
  holds flat, the new distribution did it. No comparison stands without this control.

2026-11-01 — Drift attributed: the continuation itself, not the round-A changes; round B launched
- tc9_a u200 triple (checkpoint 87c1e243): full-96 35/96, subset-36 30/36 (+1 over init:
  translate_0.10_112.5 fixed; 90 deg's demand violation gone, its planar 0.0261 and the
  157.5/180 pair 0.0255/0.0251 now miss by <= 1.1 mm/s), quiet 2/2 and formal__0.00 held,
  gentle arcs 5/16 (out of scope, health signal). Terrain at u200: hill 9.5 median
  1.79/worst 1.61 (slightly past init, no crossings), steps 1.5 held 6/6, bumps 1.42/1.29,
  steps 2.5 0.71. The u1000 crest crossings were bought with the flat damage; u200 buys
  flat without giving the crest back.
- CONTROL (tc9_ctrl: v35 unchanged, mixed_v3, same CLI/seed/budget from u600_002), omni15
  by rung vs tc9_a: u200 2/16 vs 5/16, u500 1/16 vs 3/16, u1000 0/16 (demand median
  0.0164, 16/16 bearings over) vs 1/16. The control drifts WORSE at every rung: prolonged
  surrogate continuation from the native checkpoint degrades flat quality by itself, and
  the v38 distribution slows the drift and adds quality at matched experience. Round-A
  changes exonerated and credited; the operating recipe for surrogate legs is SHORT
  continuations (~200 updates) judged at rungs.
- Reframing recorded: surrogate rounds are short lever tests; the native Spark retrain is
  the consolidation pass (native training continues in its own physics, so this cross-sim
  drift does not apply there; u600_001/002 showed no such flat decay over 600 native
  updates vs their init).
- Round B (three one-change arms, 200 updates each from tc9_a u200, registered at commit
  13d7aa9d): b1 = v38 unchanged (does a second short leg keep improving?), b2 =
  destination_v39 (over_rating_weight 10 -> 30; the omni demand lever; crest median is the
  canary), b3 = destination_v40 (turn share 0.35 drawing 0.3/0.4 rad/s; the never-trained
  0.40 stepping stone). Judges: omni15 + subset-36 per arm; terrain courses on winners.
- Operational: port-curriculum is a SYMLINK to the main checkout (git ops through it
  detached HEAD and briefly orphaned 13d7aa9d; recovered by ff-merge from the reflog).
  Training now runs from hexapod-cupi directly; memory updated. A zsh word-splitting
  gotcha also cost one empty launch round (set -- does not split; explicit args now).

2026-11-02 — Round B judged: the turn lever works, the demand lever does not; clearance term built
- Arm results (200 updates each from tc9_a u200, vs its baseline 30/36 flat, 5/16 omni):
  b1 (+200 v38 plain) 28/36, 3/16 — a second plain leg drifts, short legs do not stack;
  b2 (over_rating x3) 28/36, 3/16, demand median WORSE (0.0071 vs 0.0053) — the reward-
  weight demand lever is ineffective at this horizon, dropped; b3 (turn-heavy v40) holds
  30/36 and 5/16, fixes translate 157.5/180, and halves turn_+0.40's yaw error
  (0.278 -> 0.147) with demand near bound (0.0098 vs 0.005) — real progress on a
  never-trained capability; turn_-0.40 unmoved (0.357, no checkpoint has ever passed it).
  Universal drift symptom in every arm: quiet__stage2_long loses
  max_joint_position_range_rad (the long-stand fidget) — a surrogate-leg artifact to
  re-check natively, where u600_002 held quiet 2/2 over 600 updates.
- b3 terrain: every held course stays 6/6 (hills 4/7, steps 1.5), crest median 1.73 intact,
  steps 2.5 and bumps unchanged — the turn-heavy mix is terrain-safe. b3 checkpoint
  0d971231 is the round-B winner.
- Swing-clearance term implemented (commit 64798777, 376 tests green): scheduled-swing
  unloaded toes earn up to weight 0.5 (default 0.0 = byte-identical reward), walking only,
  for toe height above local ground up to 3.5 cm, gated to ground whose 5-point spread
  exceeds 8 mm — flat ground pays nothing by construction (the panel's high-stepping
  objection answered). Telemetry samples the footfall TerrainGrid on detached cpu float64
  copies (bitwise native/surrogate parity by the 1f1934e6 discipline). The frozen-baseline
  config guard now pins the three added fields and their inert defaults explicitly.
- tc9_c1 launched: destination_v41 (= v38 + clearance 0.5), 200 updates from tc9_a u200 —
  one change against arm b1. Primary judge: steps 2.5 and bumps walked distances; guards:
  subset-36 >= 28, held courses 6/6, omni15 not below 3/16.

2026-11-02 — Clearance arm verdict: not the riser lever as tested; consolidation moves native
- tc9_c1 (v41, 200 updates from tc9_a u200, checkpoint 8c687eda): steps 2.5 cm WORSE
  (walked [0.98, 0.89, 0.84, 0.25, 0.18, 0.11]; three starts stall almost immediately),
  bumps unchanged (median 1.42), held courses intact. Flat: subset 27/36 (the universal
  drift pair formal__0.00 + quiet__stage2_long, plus translate 202.5), omni15 5/16 held.
  Strictly dominated by b3. The simple rough-gated clearance at weight 0.5 over 200
  updates does not restructure the swing; the stall reads as horizontal riser contact
  (toe-stubbing), which apex-height pay does not prevent. The term stays in the tree
  (default off, tested); the designed successor if the wall must fall on the surrogate is
  landing-conditioned lift (pay apex only when the predicted touchdown cell sits higher
  than the support — touchdown_points/ring machinery already computes both).
- Campaign state after rounds A-C: proven levers = bearing rebalance (rear translations)
  and turn-heavy practice at 0.3/0.4 (halves +0.40 error, terrain-safe). Disproven =
  demand weight x3, simple clearance, long surrogate legs, stacked short legs. Unmoved
  walls = steps 2.5, bumps crossing, crest crossing (surrogate-side), turn -0.40, formal
  tilt (a steady 1.09 deg lean), omni demand at side/rear bearings.
- destination_v42 (the native consolidation candidate): v38's bearings/speeds with turn
  share 0.20 drawing 0.3 or 0.4 rad/s (b3's lever at consolidation dose: ~200 turn runs
  over 1000 updates vs b3's ~70) — clearance off. To train NATIVELY per the recorded
  reframing (fresh native runs showed no drift: u600_002 held flat families over 600
  updates where every surrogate leg paid the quiet/formal tax). Spark queue next, per the
  campaign instruction, behind the usual etiquette.

2026-11-02 — Native v42 consolidation queued on Spark
- destination_v42 registered (commit b1bb4131, 376 tests green; a ruff line-wrap had
  silently dropped v42 from EXPERIMENT_CLOCKS and the clocks-agreement test caught it —
  fixed and amended before anything shipped).
- Physics check: physics_files() over the full tree confirms every campaign change
  (destination_task.py, mdp/rewards.py, agents/rsl_rl_ppo_cfg.py, tests) is a runtime
  source; adapter.py (physics) is untouched since 1f1934e6 — the round-7 admission
  (49a4e207) remains valid, no new admission round.
- Pack terrain_isaac_ppo_v42_u1000_seed20260917_20261102_001 at freeze 959a39e1: binding
  diff vs u600_002 is exactly {source freeze, experiment v35→v42, updates 600→1000,
  terrain mixed_v3→mixed_v5}; input_files bind mixed_v5 (1a859fcc), the v29_terrain_v2
  init (5b32fba4 / declaration 543d2e63), the round-7 admission and standing evidence.
  Fresh-from-v29_terrain_v2 per the drift evidence; 1000 updates per the tc3_a crest dose.
- Launched detached behind the queue: GPU was at 96% (another user's workload — armvla-mind
  and qwen38-server present as usual, untouched); the runner waits up to 8 h for GPU ≤ 25%
  and ABORTS with status queued_out rather than launching into contention. Runner PIDs
  3590796/3590803; marker /home/julian/v42_u1000.done; local waiter polling.
- On green: scp the checkpoint, run the triple + omni15 (comparators u600_002, tc9 rungs,
  v29 control), record the native-consolidation verdict — the first test of whether native
  training holds flat quality while the proven levers land.

2026-11-02 — v42 queued out behind the long job; landing-conditioned lift built (v43), arm d1 launched
- The v42 Spark runner found the GPU at 96% for the full 8 h window (armvla-mind, another
  user's job, untouched) and aborted queued_out as designed. Relaunched with the queue window
  extended 96 -> 288 checks (24 h), same etiquette: launch only at GPU <= 25%, re-check, abort
  rather than contend. Nothing was extracted or launched on the first pass.
- destination_v43 implemented and registered (commit 4cf36317, all eight CONTRIBUTING suites
  green, locomotion 378): the tc9_c1 successor. A swinging unloaded toe is paid, walking only,
  when the footfall module's predicted touchdown — touchdown_seconds/touchdown_points under
  the speed-scaled clock, sampled at the five RING_XY points, ring MAXIMUM so a prediction
  landing short of a riser still reads the riser top — sits at least landing_lift_rise_m
  (1 cm) above the ground under the toe now. Pay is proportional to the toe's clearance over
  the CURRENT ground (dense from the first lift, reusing the clearance telemetry) and
  saturates at rise + landing_lift_target_m (2 cm above the landing ground): the ceiling
  scales with the obstacle, answering the fixed-target objection (3.5 cm pay saturating under
  a 4 cm bump top). The gate opens on the 2.5 cm risers, the 2-4 cm bumps, the 1.5 cm risers
  (held course — aligned shaping, guards watch it) and the upper 9.5-degree ramp (ring max +
  ~5 cm stride reach about 1.3 cm); flat ground, 5 mm noise, descents and the 4/7-degree
  ramps stay closed. Weight 0.5 (v41's dose), default 0.0 byte-identical; the frozen-baseline
  guard pins the three new fields inert. Telemetry on detached cpu float64 copies (footfall
  parity discipline); the flat floor reports zero rise.
- tc9_d1 launched: destination_v43, 200 updates from tc9_a u200 (87c1e243), seed 20260917,
  mixed_v5 — one change against arm b1 at matched experience, the same protocol as c1.
  Primary judge: steps 2.5 and bumps walked distances vs c1 ([0.98..0.11] / 1.42 median) and
  b1; guards: held courses 6/6, subset-36 >= 28, omni15 >= 3/16, crest median not below init.

2026-11-02 — d1 verdict: landing lift is safe and points right, unmoved at 200 updates; d2 extends the horizon
- tc9_d1 (v43 landing lift 0.5, 200 updates from tc9_a u200; checkpoint f7944b50, declaration
  6fbc5fed, outputs full96_tc9_d1_u200/ terrain_eval_tc9_d1_u200/ omni15_tc9_d1_u200/):
  every guard passes — subset-36 30/36, exactly its init's score (fails: the drift pair
  quiet__stage2_long + formal__0.00, formal__0.16, translate_0.10_90, turn ±0.40), held
  courses 6/6, crest median 1.79 not below init, omni15 4/16 (demand at side/rear as before;
  67.5 deg terminates). Unlike c1 (same weight, wrong incentive), NO damage anywhere.
- Primary walls at the median: unchanged (steps 2.5 walked [1.18, 0.98, 0.88, 0.80, 0.22,
  0.12]; bumps [2.09, 1.84, 1.46, 1.46, 1.44, 1.31]). But the TOP END deepened on both:
  steps best 0.98 -> 1.18 m, bumps best ~1.6 -> 2.09 m — the signature of a correctly
  pointed lever that 200 low-LR updates cannot turn into a restructured swing (tc9_a showed
  gait restructuring, its crest crossings, only by u1000).
- tc9_d2 launched: v43 unchanged, 1000 updates from tc9_a u200 — one change against d1
  (horizon), with tc9_a's own u500/u1000 rungs as the matched-experience control (v38, no
  lift: crest 2/6 and bumps-worst gains at u1000, steps 2.5 unchanged, flat drift). Judge:
  steps 2.5 and bumps walked at rungs u500/u1000 vs those controls; the lever is proven for
  the native recipe (a v44 = v42 + landing lift) only if d2 moves the walls beyond what the
  horizon alone bought tc9_a, with the flat drift attributed to the horizon by that same
  control. Dose escalation (weight 1.0) is the fallback arm if d2 is null.

2026-11-02 — v42 lost the GPU lock race to the reference-physics workload; runner evolved to re-queue
- The GPU freed at queue check 47 (11% after ~4 h of the long job); the runner extracted,
  re-approved the binding (a5a656ca) and launched — and locomotion.spark.launch exited 1 in
  guard.both_locks(): BlockingIOError, the non-blocking flock refused because a NEW workload
  (container hexapod-reference-physics-99028e60, another contributor's, GPU back at 95%) won
  the lock in the one-minute gap a utilization poll cannot see. The lock discipline worked
  exactly as designed: no contention, no run directory, nothing to clean. The workload was
  left untouched per etiquette.
- Runner evolved (run_v42_u1000.sh on Spark): up to six 24 h queue windows; a lost lock race
  (BlockingIOError in the launch log) sends it back to the queue after 5 min instead of
  ending the attempt; the marker now reports finished / launch_failed / queued_out honestly
  (the first version wrote "finished" unconditionally — caught because the runner's log, not
  the marker, is the judge). Relaunched detached, queueing behind the reference-physics job.

2026-11-02 — d2 verdict: the landing lift buys CONSISTENT crest crossing; consolidation moves to v44
- tc9_d2 rungs (v43, 1000 updates from tc9_a u200; u500 8e34aad4/038310ba, u1000
  3d206293/ebcabb53; outputs terrain_eval_tc9_d2_u500/, terrain_eval_tc9_d2_u1000/,
  full96_tc9_d2_u1000/, omni15_tc9_d2_u1000/; control tc9_a_u500 6717423b run for the
  missing rung, terrain_eval_tc9_a_u500/):
  - hill 9.5 deg: d2 u1000 CROSSES 6/6 (3.79/3.77, zero falls) — the campaign's first
    consistent crest crossing. Matched-experience control (tc9_a u1000, v38 alone): 2/6
    bimodal, median 0.69. At u500 the separation already shows in the worst start (d2
    1.65 vs control 0.53). The lift lever is PROVEN: same seed, CLI, horizon; one change.
  - Held courses all 6/6 at every rung; steps 2.5 unmoved (0.83/0.17 — the toe still stubs
    the riser; apex height was never the binding constraint there); bumps median nudges
    1.42 -> 1.67 with best 2.09 at u200 (top-end movement only).
  - Flat at u1000: subset-36 31/36, the BEST in-scope score of any surrogate leg (fails:
    translate_0.10_180, turn_-0.40, quiet long, formal 0.00/0.16) with turn_+0.40 passing
    for the first time; total 32/96 where the control collapsed to 26/96 — the lift
    distribution also slows the flat drift, as v38 did for v35.
  - omni15: 4/16 -> 0/16 through the leg, all on computed demand barely over (0 deg 0.0065
    vs 0.005) — the horizon's drift tax, control-attributed (tc9_a u1000 1/16, tc9_ctrl
    0/16). Per the operating recipe this does NOT disqualify the lever: the native
    consolidation holds flat (u600_002 kept omni 4/16 over 600 native updates).
- Consolidation replaced BEFORE launch: the queued v42 runner was stopped while still in
  its queue loop (no run directory, nothing launched; the v42 pack and tar stay on Spark
  and in the scratchpad as the preserved attempt). destination_v44 = destination_v42 +
  landing_lift_weight 0.5 registered (commit c71b308f, suites green 378); physics_files
  over the full tree re-confirms every campaign change is runtime-only — round-7 admission
  49a4e207 stands. Pack terrain_isaac_ppo_v44_u1000_seed20260917_20261102_002 at freeze
  e3ff97a9: binding diff vs the v42 pack is exactly {source freeze, experiment, run name};
  terrain, init checkpoint, admission and budget identical. Runner (re-queue-on-lock-race,
  six 24 h windows) launched detached, queueing behind the reference-physics workload.
- Remaining walls after d2: steps 2.5 (stub at the vertical face — lift alone insufficient;
  next candidate mechanism is forward-foot placement or stride shortening at a detected
  riser, design TBD on v44 evidence), bumps crossing, turn_-0.40, formal 0.00/0.16 lean,
  omni demand at side/rear (native consolidation's to hold).

2026-11-03 — sim2real phase 1: machinery built and proven; round a (1000 updates) pays the drift tax through the adapter
- Mechanism (commit b0f9f311, suites 384/67/66 green, physics closure 18 files unchanged —
  round-7 admission 49a4e207 stands): CompensatedActor = frozen u600_002 core (e29be0bc,
  bytes sha-verified at construction) behind bounded latent-conditioned gains/biases on the
  proprio+executed-target columns and a bounded pre-tanh action delta; heads zero-initialized
  (composite == core bitwise at u0, unit-proven); PPO trains the compensator alone;
  WarmStartedCritic starts from the core critic. DynamicsRandomization in the surrogate env:
  per-episode friction/mass/motor/kd scales, 0-1 control action delay, in-episode sag drain,
  per-row (seed,row,episode) streams; randomization=None proven bitwise inert
  (test_randomization). Judges: sim2real_robust_eval.py — scored half (evaluator's own
  omni_static bounds; friction/mass/delay draws only, since the 400 Hz capture certifies the
  canonical motor law and motor-side draws rightly fail its recurrence check) + walked half
  (terrain_eval scene machinery, all axes incl. motor/kd/sag); sim2real_readout.py — held-out
  linear R2 from the latent to the true draws. No gate, manifest or scorer touched.
- s2r_a: v35 task+reward (the core's own), mixed_v3, seed 20260917, ranges friction (0.6,1.4)
  mass (0.9,1.1) motor (0.9,1.1) kd (0.8,1.2) delay<=1 sag (0,0.1)/floor 0.85, 1000 updates
  (checkpoint 8365f49d, declaration a36e0b1a; outputs full96_s2r_a_u1000/,
  terrain_eval_s2r_a_u1000/, omni15_s2r_a_u1000/, s2r_robust_a/, s2r_readout_a/).
- VERDICT, round a at u1000: FAILS preservation — full-96 24/96 vs core 33/96, omni15 0/16 vs
  4/16, quiet 0/2; planar errors and demand grew (robust scored: composite 22/48 vs bare core
  31/48 on matched draws — the adapter made perturbed flat cases WORSE). Walked half separates
  nothing (36/36 both). Readout null (held-out R2 <= 0.08 on every axis): reward-only training
  did not put identity information in the latent. Terrain held or deepened (hills 6/6,
  9.5-degree median 1.79 vs 1.70, bumps 1.68 vs 1.42, formal 2/4 vs 1/4) — the compensator
  learned the training distribution, not compensation.
- READING: the drift law extends to adapter-space training — freezing the core does not block
  the quiet/formal/omni drift tax, the bounded compensator channels it at 1000 surrogate
  updates. tc9_d1 located the knee below 200 updates for reward levers; s2r_a's own
  50-update checkpoint ladder materializes the horizon arm with zero retraining.
- Next: judge s2r_a u200 and u500 on omni15 + the scored robust half (core comparator 31/48
  already measured on the same seeds) + readout; pick the knee that preserves nominal and
  beats the core under draws, else arm s2r_b = flat ground (no terrain), short horizon.

2026-11-03 — s2r_a ladder + hill judge: parity at the safe horizon; identification needs supervision (phase 1b built)
- Ladder (s2r_a's own 50-update checkpoints, zero retraining): u200 omni15 3/16 (guard >=3
  holds; core 4/16), scored robust 32/48 vs core 31/48 (parity); u500 1/16, 31/48; u1000 0/16,
  22/48 — the drift tax grows with horizon exactly as tc9_d1/d2 located it, and the readout is
  null at every rung. Outputs omni15_s2r_a_u{200,500}/, s2r_robust_a_u{200,500}/,
  s2r_readout_a_u{200,500}/.
- Sharper walked judge (hill-7 added to sim2real_robust_eval.py, matched draws): u200 30/36 =
  core 30/36. The one harsh draw (draw03, weak-motor+sag) stalls the core 0/6 on the ascent at
  ~1.7-1.9 m; the composite salvages 1/6 — a single-start whisper, not a win. Outputs
  s2r_robust_walk2/.
- READING: at the drift-safe horizon the reward-only compensator is behaviorally the core; the
  reward gradient alone neither identifies the world (readout null) nor buys robustness. This
  is the literature's result (RMA trains its adaptation encoder with explicit supervision).
- Phase 1b (commit 37b26937, all eight suites green): Sim2RealVecEnv buffers (observation,
  draw) pairs; LatentSupervision adds an identification loss on the compensator's encoder +
  a local readout head after each PPO update — task reward, PPO loss and gates untouched;
  --latent-supervision RATE recorded in identity, default 0 keeps the stock learner.
- Arm s2r_b: s2r_a's exact recipe at the safe horizon (200 updates) + the supervision
  (wrapper locomotion.rl.sim2real:Sim2RealVecEnv, --latent-supervision 1e-3). Judges, in
  order: readout held-out R2 (did identification emerge), omni15 nominal guard >= 3/16,
  scored robust + hill-7 walked vs the core's recorded 31/48 and 30/36 on the same seeds.

2026-11-03 — s2r_b verdict: supervision works but moves behavior; arm c decouples the instrument
- s2r_b (s2r_a recipe, 200 updates, Sim2RealVecEnv + --latent-supervision 1e-3; outputs
  s2r_readout_b/, omni15_s2r_b_u200/, s2r_robust_b/, s2r_robust_b_walk/): the first non-null
  readout — held-out R2 mass 0.449, kd 0.189, friction 0.058 (motor/delay/sag still <= 0: the
  0.1 s window sees body dynamics, not a rarely-binding torque ceiling or a 32 s drift).
  Nominal omni15 6/16, ABOVE the core's 4/16 and arm a's 3/16. Scored robust 32/48 (parity
  with core 31/48). But hill-7 walked 23/36 vs core 30/36 — the supervised encoder stretched
  the latent under the heads (z std 0.67 -> 1.64, |delta| max 0.04 -> 0.13 measured on matched
  inputs), shifting behavior exactly where z varies most. Identification and control pulled on
  one latent.
- Arm c (commit 84943803): a dedicated identification encoder on the actor; supervision touches
  it alone, the control encoder stays reward-only, dynamics_latent reads the instrument.
  Expectation: behavior returns to arm-a parity (hill 30/36-ish, omni guard) while the readout
  keeps or beats s2r_b's R2. s2r_c = same recipe, 200 updates, identification=True.

2026-11-03 — s2r_c verdict: decoupling restores behavior and calibrates the noise floor; arm d conditions the heads
- s2r_c (identification encoder separate, supervision off the control path; outputs
  s2r_readout_c/, omni15_s2r_c_u200/, s2r_robust_c/, s2r_robust_c_walk/): hill-7 walked
  restored to 30/36 = core (the s2r_b regression was the coupled latent, hypothesis
  CONFIRMED by head magnitudes: z std 1.64 -> supervised-separate, |delta| max back down);
  omni15 5/16 >= core 4; readout mass 0.272, motor 0.059, friction 0.073 (weaker than b's
  coupled 0.449 mass but clean). Scored robust 27/48 vs arms a/b 32/48 and core 31/48 —
  across arms the scored half swings ±5/48 at identical recipes, so NO arm's heads have
  learned real compensation; they read a blind latent and stay noise.
- Arm d (commit b813c7f3, suites green): condition_heads — the heads read the identification
  latent DETACHED. Supervision owns the latent geometry (no s2r_b warp possible by
  construction, pinned by test), PPO trains the heads on a code that names the world. This is
  the RMA coupling adapted to a frozen core. s2r_d = same recipe, 200 updates,
  identification + condition_heads. Judge order unchanged; win = scored robust clear of the
  ±5 noise band above core 31/48 with hill >= 30/36 and omni >= 4/16.

2026-11-03 — PHASE 1 SURROGATE CONCLUSION: instrument real but narrow; no in-range compensation win over an already-robust core
- Arm d (conditioned heads, commit b813c7f3; outputs s2r_readout_d{,_wide}/, omni15_s2r_d_u200/,
  s2r_robust_d{,_walk}/): scored 28/48 (inside the ±5 noise band around core 31/48), hill-7
  26/36 (mild regression), omni 5/16 >= core. The RMA coupling did not buy compensation at the
  drift-safe horizon either. (s2r_c's wide readout re-measure requires its own commit 84943803:
  the d commit moved the identification module into the trunk, so c's checkpoint loads only
  there — recorded, not re-measured.)
- Instrument, measured properly (256 episodes, 64 held out, s2r_readout_d_wide/): mass_scale
  R2 0.252 — real and stable; friction/kd marginal (<= 0.1); motor_scale, action delay and sag
  NOT identifiable from the 0.1 s proprioception window at 0.1 m/s walking excitation. The
  physics agrees: the torque ceiling rarely binds at walking loads, a one-control delay hides
  under velocity noise, and a 32 s drift needs memory no single window carries. The earlier
  b/c mass spread (0.449/0.272) was the 32-episode estimator's variance around ~0.25.
- FOUR-ARM VERDICT (a: reward-only; b: coupled supervision; c: decoupled instrument; d:
  conditioned heads — all 200-update legs vs the frozen u600_002 core on matched draws):
  within the trained ranges (friction 0.6-1.4, mass/motor ±10%, kd ±20%, <=1 control delay,
  <=10%/min sag) the core is ALREADY robust (scored 31/48, hill-7 30/36, flat 36/36), the
  bounded compensator cannot beat it, and its capacity at drift-safe horizons is behavioral
  noise; beyond the safe horizon it channels the known surrogate drift tax (a@u1000: 24/96,
  omni 0/16). Preservation holds on the c architecture (hill parity, omni >= core, composite
  bitwise core at u0).
- PHASE-1 DELIVERABLES STANDING: the composite machinery (commits b0f9f311, 37b26937,
  84943803, b813c7f3; suites green throughout; physics closure and admission untouched), the
  bitwise-inert DynamicsRandomization, the matched-draw judges, and a mass-identifying
  instrument with preserved behavior.
- PHASE-2 NOTES (hardware, per the recorded plan — adapter as instrument first): identification
  of motor/delay/sag needs excitation or memory, not more reward: a short calibration protocol
  (high-acceleration bench moves, the SPI-Active-style minute) and/or the stack's existing
  proprio_history observation group; both are input-side steps that go to James via Julian
  before any native run. On hardware the identified parameters write back into the simulator
  (real2sim) rather than freezing an opaque network — the s2r legs' evidence says bounded
  output patching has no headroom here anyway.
- Campaign focus returns to the native consolidation: v44 queued on Spark (runner healthy,
  window 1 of 6).

2026-11-03 — Rehearsal 001: the deployment loop runs end to end; the hidden world's binding axis is exactly the unidentifiable one
- s2r_rehearsal_001/ (composite s2r_e 200u, c-architecture at d32f5d84; world seed 777003;
  trust bar 0.10 admits mass alone — the one field positive in every wide measurement).
  W* drew friction 1.148, mass 0.940, motor 0.920, kd 1.024, delay 1, sag 0.085/min.
- DEPLOY: pass — flat 6/6 crossed, zero falls (hill-7 0/6 at ~1.7-2.0 m stalls; the bare core
  stalls identically there, 0/6: W* is a weak-and-sagging-motor world and the hill binds it).
- IDENTIFY (64 episodes of W* walking): mass 0.992 vs truth 0.940 (beats the center guess),
  friction/motor estimates sit at the prior center (uninformative); kd/delay/sag miss. The
  estimator behaves exactly as the phase-1 instrument predicted.
- HARDCODE+VALIDATE: only mass written back -> the calibrated sim is nominal in all but name;
  flat gap 0.010 m (trivially matched), hill gap 2.028 m UNCHANGED — the calibrated sim walks
  the hill the real world cannot, because the binding parameters (motor x0.92 with 8.5%/min
  sag) are invisible to walking-data identification. The plan's loop closes mechanically but
  not physically on torque-side worlds.
- FIX FOR 002 (excitation, as the phase-2 note prescribed): identify from episodes that BIND
  the torque ceiling — calibrate and identify on a 7-degree ramp instead of flat ground. If
  motor/sag become readable there, the hardcoded sim should reproduce W*'s hill stalls and the
  validation gap should close for real.

2026-11-03 — ARC SCOPE RULING and the turn-while-walk campaign opens (v45, arcing walks)
- Julian's instruction extends the campaign: keep every current movement/turning/stopping/
  terrain result and add "all turn while walk commands possible". Feasibility, from the
  recorded evidence: tight_arc (0.20 m/s + 0.4 rad/s) is INFEASIBLE — v29_ctrl, the best flat
  policy, terminates on most tight arcs with demand 0.04-0.13 (8-27x the 0.005 bound); it
  rides the already-excluded 0.20 speed tier. gentle_arc (0.15 + 0.05) and combined (0.10 +
  0.2) are WINNABLE: v29 reached 14/16 and 4/8 with near-threshold misses (planar 0.026-0.031
  vs 0.025, yaw 0.057-0.068 vs 0.06), u600_002 fails them demand-heavy (curves cost its
  terrain gait torque it never practised spending). transition:full_program joins the scope:
  every segment is <= 0.15 m/s and <= 0.25 rad/s, arcs included. TARGET SET = in-scope 36
  + gentle_arc 16 + combined 8 + transition 1 = 61 cases, plus held terrain and omni15.
- Root cause confirmed in code: run_mix draws only pure turns, stands and straight walks
  (destination_task.py _mixed_runs) — arc commands were never in the training distribution.
- destination_v45 (commit 1531e8b4, suites green, physics closure untouched): v44 + arcing
  walks — arc_fraction 0.25 of drawn walks (stopping walks included, so stops from arcs are
  practised) with signed yaw from (0.05, 0.2) rad/s, the two scored tiers; draw columns gated
  on the knob so other experiments' RNG streams stay byte-identical (house pattern).
- Legs tw_a (v45) vs tw_ctrl (v44), 200 updates each from tc9_a u200 on mixed_v5, seed
  20260917 — one change (the arc share), matched experience. Judges: full-96 arc families
  (primary: gentle_arc, combined, transition vs u600_002's 4/16, 0/8, 0/1 and v29's 14/16,
  4/8, 0/1), subset-36 guard >= 28, held courses 6/6, omni15 >= 3/16.

2026-11-03 — Rehearsal 002: deployment-side excitation alone does not rescue identification; the limit is structural
- s2r_rehearsal_002/ (same composite, same hidden world 777003, identification moved to a
  full-field 7-degree ramp where W*'s weak sagging motors demonstrably bind — W* stalls the
  core 0/6 there): calibration R2 DROPPED across the board (mass 0.396 -> 0.115 train; motor
  stays ~0.05), nothing new became trusted, the calibrated sim stayed nominal-in-all-but-mass
  and the hill validation gap stayed 2.028 m. Deployment gate passed again (flat 6/6, no
  falls).
- READING: the instrument's encoder is frozen at readout time and was supervised on mixed_v3
  walking; on the uniform ramp its features extrapolate poorly, so binding the torque ceiling
  at DEPLOYMENT injects signal the encoder was never taught to extract. Excitation must be in
  the INSTRUMENT'S TRAINING distribution, not only in the deployment protocol. The episode-mean
  latent plus linear readout also averages away stance-sliced torque signatures.
- REHEARSAL SERIES CLOSED (001-002): the plan's loop runs end to end — deploy, walk, identify,
  hardcode, validate — and its measured limits are now concrete for the hardware phase:
  (i) mass identifies from plain walking (hardcodable); (ii) motor/delay/sag need a designed
  calibration protocol present in BOTH the instrument's supervision data and the deployment
  routine (torque-binding excitation; the SPI-Active-style minute) or direct measurement
  (current/voltage sensing, flagged in RS05_SPEC_REVIEW as absent), with longer observation
  memory (proprio_history) the input-side candidate — James's call, relayed by Julian, when
  hardware phase opens; (iii) behavior-side compensation has no in-range headroom around an
  already-robust core (the four-arm verdict), so the adapter deploys as instrument + preserved
  gait, exactly as recorded in the phase-1 conclusion.
- Campaign focus: the turn-while-walk legs (tw_a vs tw_ctrl) and the v44 Spark queue.

2026-11-03 — tw round 1 (200u): arc exposure cannot restructure at the short horizon; extend to 1000u
- tw_a (v45, arcs; checkpoint 72559f82, declaration 0c9365f7, outputs full96_tw_a_u200/,
  terrain_eval_tw_a_u200/, omni15_tw_a_u200/) vs tw_ctrl (v44; ab7d471e/38199441, outputs
  full96_tw_ctrl_u200/, terrain_eval_tw_ctrl_u200/, omni15_tw_ctrl_u200/), both 200 updates
  from tc9_a u200 on mixed_v5, one change (the arc share).
- Arcs did not move: gentle 2/16 vs control 4/16, combined 1/8 both; the arc share instead
  diluted straight-walk practice (translate 10/32 vs 15/32, omni15 3/16 vs 7/16, total 30/96
  vs 35/96). Terrain held for both (hills 4/7 6/6, steps 1.5 6/6; 9.5 medians 1.65/1.75).
  Stops 10/10 on tw_a — stops-from-arcs practice did no harm there.
- READING: exactly tc9_d1's lesson — 200 low-LR updates expose but do not restructure; the
  crest crossing needed u1000. The matched control also shows the v44 recipe at a short
  surrogate horizon is healthy (35/96, omni 7/16 > u600_002's 4/16).
- Round 2: tw_a2 (v45) vs tw_ctrl2 (v44), 1000 updates, same start/seed/terrain — judge
  whether arcs climb beyond what the horizon alone buys the control, with the drift pair
  watched as always. The native consolidation (no drift) is where a proven arc lever lands.

2026-11-03 — tw round 2 (1000u): the arc-exposure lever FAILS its target but may have moved the steps wall; seed arm launched
- tw_a2 (v45; 04c43dc9/fd77c5c6; outputs full96_tw_a2_u1000/, terrain_eval_tw_a2_u1000/,
  omni15_tw_a2_u1000/) vs tw_ctrl2 (v44; 50ce768a/452922e3; outputs full96_tw_ctrl2_u1000/,
  terrain_eval_tw_ctrl2_u1000/, omni15_tw_ctrl2_u1000/), 1000 updates, one change (arc share).
- ARCS, judged by error medians against the matched control (both drift equally, so drift is
  controlled): tw_a2 gentle planar 0.0455 / yaw 0.0613 / demand 0.0108 vs control 0.0417 /
  0.0509 / 0.0077 — WORSE on all three; combined flat. Pass counts 0/16, 0/8 both. The
  exposure lever is DISPROVEN for arc tracking at this dose. The reframe the evidence forces:
  v29 reached gentle 14/16 with NO arc practice either — arc capability rode flat PRECISION,
  which terrain training costs (the recorded gentle-arc regression). The arc path is the
  precision axis (interleaved flat batches — the u600_002 entry's candidate — and native
  training), not command exposure. The native v44 full-96, free when Spark clears, is the
  next arc datum.
- TERRAIN, the surprise: tw_a2 crossed steps 2.5 cm 2/6 (walked medians 0.98, worst 0.68,
  two full crossings) — the campaign's first-ever crossings of its hardest wall — with bumps
  median 2.01 (best ever) and hill 9.5 5/6; the control stayed 0/6 steps (median 1.04), 6/6
  crest, bumps 1.42. Mechanism candidate: arcing walks force yawed, asymmetric stances that
  negotiate risers. One seed is one trial: tw_a3 (v45, seed 20260918, 1000u, terrain judge)
  tests reproducibility before the lever is reframed as a STEPS lever and considered for the
  native consolidation beside or in place of queued v44.
- Omni15 0/16 vs 4/16 — the usual u1000 drift tax plus the arc dilution; the native
  consolidation remains where flat quality is held.

2026-11-03 — Steps lever REPRODUCED across seeds; the queued native consolidation moves to v45
- tw_a3 (v45, seed 20260918, 1000u; checkpoint 60abecbf, outputs terrain_eval_tw_a3_u1000/):
  steps 2.5 cm crossed 2/6 AGAIN (two full 3.78 m crossings; walked [1.23, 3.78, 1.13, 3.78,
  1.03, 0.14]), bumps walked best-ever spread (median ~2.1, best 2.52), hill 9.5 4/6, held
  courses 6/6. Two seeds (20260917, 20260918) each cross a wall that every prior leg, control
  and native checkpoint left at 0/6; the matched-experience control sits at 0/6. The arc share
  is a PROVEN TERRAIN lever (yawed asymmetric stances negotiate risers), though disproven for
  its original arc-tracking target.
- Consolidation swap, the v42->v44 precedent: the still-unlaunched v44 runner was stopped in
  its queue loop (first kill matched the ssh shell's own cmdline — the recorded self-match
  trap — second kill by exact PID 4074652 confirmed gone; no run directory ever existed; the
  v44 pack and tar stay on Spark and in the scratchpad as the preserved fallback). Pack
  terrain_isaac_ppo_v45_u1000_seed20260917_20261103_001 built at commit 1531e8b4 (freeze
  4a5c2a22): binding diff vs the v44 pack is EXACTLY {source freeze, experiment v44->v45,
  output/source paths}; terrain (mixed_v5 1a859fcc), init (v29_terrain_v2 5b32fba4/543d2e63),
  admission 49a4e207 and budget identical. Runner run_v45_u1000.sh (same six 24 h windows,
  re-queue on lock race) launched detached, PID 3820851, queueing at GPU 94%; marker
  /home/julian/v45_u1000.done; local 30 h waiter re-armed.
- v45 native carries v44's whole recipe (bearing rebalance, turn lever, landing lift) PLUS the
  steps lever, and its full-96 gives the arc families their no-drift datum. If its flat or
  omni numbers land under u600_002's, the preserved v44 pack is the immediate fallback.

2026-11-03 — tw round 3 verdicts and round 4 arms: the recipe web has interactions; the chirality lever is built
- tw_b (v46 = v43 + arcs, no turn mix; 3b163209/2df0d15e; outputs full96_tw_b_u1000/,
  terrain_eval_tw_b_u1000/, omni15_tw_b_u1000/): the best drifted-horizon totals yet —
  34/96, omni15 4/16 PRESERVED through u1000 (every other u1000 leg: 0/16), gentle 3/16,
  stops 10/10 — but the CREST IS LOST (9.5-degree 0/6, median 1.66, where v43 alone crossed
  6/6) and steps 2.5 did not move (0/6). INTERACTIONS: the turn mix is load-bearing for the
  steps synergy AND for holding the crest under arc practice; the arc share without it
  protects flat/omni instead.
- Turn chirality measured (turn_chirality_001/, probe script turn_chirality.py): tc9_d2
  tracks +0.40 at 0.393 rad/s achieved but -0.40 at only -0.173 with tilt spikes to 5.5
  degrees; u600_002 0.185 vs -0.082; both lean ~0.4-0.7 degrees standing. ±0.20 symmetric.
  The fixed TRIPOD_A first-swing convention is the handedness; round B already showed
  practice cannot close it (turn_-0.40 never passed by ANY checkpoint).
- destination_v47 (commit a54683c6, suites green, physics closure untouched, byte-identical
  default pinned): mirror the tripod schedule — phase_swing/touchdown_seconds swap tripods on
  rows with commanded yaw < 0; reward and footfall predictions move together; evaluation and
  scenes rebuild from the recorded flag. One change against v45.
- destination_v48 (commit 3c70c4dd): turn share halved on v45 (0.20 -> 0.10 into walks) —
  prices the turn-mix flat cost against its terrain synergies. Training now (tw_c) with the
  v46 seed repeat (tw_b2, terrain judge) in the same chain; v47 (tw_d) next.

2026-11-03 — Round 4 part 1: v48 dominated and dropped; v46's crest loss was seed variance
- tw_c (v48, turn share halved; outputs full96_tw_c_u1000/, terrain_eval_tw_c_u1000/,
  omni15_tw_c_u1000/): 25/96, omni15 1/16, quiet 0/2, translate 7/32; terrain crest 2/6,
  steps 1/6, bumps 1.46. Worst of all worlds — the half-share keeps the turn mix's flat cost
  without buying its terrain synergies. DISPROVEN, dropped.
- tw_b2 (v46 seed 20260918, terrain judge; checkpoint 88523931, outputs
  terrain_eval_tw_b2_u1000/): crest 5/6, steps 2.5 2/6 (two full crossings), bumps median
  ~2.2 (best tier), hills 4/7 and steps 1.5 all 6/6. tw_b's crest 0/6 was SEED VARIANCE:
  across seeds v46 = flat-preserving (34/96, omni 4/16 at seed 1) AND terrain-capable
  (seed 2). Terrain outcomes swing hard with seed at u1000 — "fully consistent" must be
  judged across seeds as well as starts.
- Next in chain: tw_d (v47, the mirrored schedule, full judges) and tw_b2's flat judgment
  (full96 + omni15) to decide whether v46 is consolidation-grade across seeds.

2026-11-04 — Round 4 part 2: the mirror does not move turn_-0.40 at this dose; u1000 is a variance regime
- tw_d (v47 mirror; outputs full96_tw_d_u1000/, terrain_eval_tw_d_u1000/, omni15_tw_d_u1000/):
  turn_-0.40 STILL terminates (yaw 0.2345, demand 0.0456); clockwise arc halves no better than
  counter-clockwise (combined cw 0/4 yaw 0.0715 vs ccw 0/4 0.063; gentle cw 0/8 vs ccw 2/8).
  31/96 total, crest 3/6 (median 2.81), steps 0/6, bumps 1.99, omni 0/16. READING: flipping
  the schedule reward mid-stream asks a fixed-schedule init to re-learn mirrored coordination
  in 1000 low-sigma updates — not confirmed at this dose; either fresh training or a deeper
  mechanism (the measured standing lean) owns the chirality. Parked for the native era; the
  mirror plumbing stays available (default None, byte-identical).
- tw_b2 flat (full96_tw_b2_u1000/, omni15_tw_b2_u1000/): 27/96, subset 24/36, omni 0/16 —
  v46 seed 2 is terrain-strong but flat-weak, the inverse of seed 1 (34/96, omni 4/16,
  terrain-weak). VERDICT: at u1000 the surrogate is a HIGH-VARIANCE drift regime — seeds swing
  the crest 0<->5/6 and flat by ~7 points; only the steps lever (2/6 x3 across recipes/seeds)
  and the lift's crest (when turn mix present) replicate. No u1000 arm is consolidation-grade
  alone.
- ROUND 5 (regime, not recipe): (a) sweet-spot scan — judge existing mid-horizon rungs
  (tw_a2 u500) for terrain-lever onset below the drift knee; (b) the anchor lever — a
  learner-side decoupled weight pull of the actor toward its init checkpoint (default 0,
  inert), to hold native-grade precision through long surrogate legs while terrain levers
  land. Judge vs tw_a2 (same recipe, no anchor) on the drift pair, omni and the terrain
  triple.

2026-11-04 — Sweet-spot scan: there is none; the anchor arm is decisive
- tw_a2 u500 (terrain_eval_tw_a2_u500/, omni15_tw_a2_u500/): crest 0/6 (median 1.68), steps
  2.5 0/6 (best 1.32), bumps ~1.85, omni15 2/16. The terrain levers mature only by u1000
  (steps crossings, crest restructuring) while the drift tax is already paying at u500 —
  no mid-horizon rung has both. The u1000-lever/u1000-drift coincidence is structural, so
  holding init precision THROUGH the horizon (the anchor, tw_e running) is the only path the
  evidence leaves for a full surrogate pass with these levers.

2026-11-04 — tw_e: the anchor kills the drift (first-ever quiet 2/2 at u1000) but at 3e-3 it suppresses the levers; dose arm launched
- tw_e (v45 + anchor 3e-3, seed 20260917; outputs full96_tw_e_u1000/, terrain_eval_tw_e_u1000/,
  omni15_tw_e_u1000/) vs tw_a2 (same recipe/seed, no anchor): quiet 2/2 — the FIRST u1000
  surrogate leg to hold the stage2_long fidget — total 34/96 vs 29, translate 13/32 vs 11,
  omni15 2/16 vs 0, gentle 3/16 vs 0. The drift tax is beaten by construction for the first
  time. COST: the terrain levers were pulled back with it — crest 1/6 vs 5/6, steps 0/6 vs
  2/6, bumps 1.65 vs 2.01. The 3e-3 displacement budget (roughly step/weight) is too tight
  for the gait restructuring the levers need.
- tw_f = v45 + anchor 1e-3 (one change: dose), same seed/start/judges: looking for the dose
  that holds quiet/omni near tw_e while the crest and steps gains survive near tw_a2.

2026-11-04 — tw_f: best-ever surrogate flat (translate 16/32, omni 4/16) but an anomalous hill-7 loss; two decisive arms
- tw_f (v45 + anchor 1e-3; outputs full96_tw_f_u1000/, terrain_eval_tw_f_u1000/,
  omni15_tw_f_u1000/): translate 16/32 = v29's own count, omni15 4/16 = the native baseline,
  total 33/96 — the strongest flat any surrogate leg has posted. Terrain: hill-7 0/6 (1.71) —
  a course NOTHING ever lost — crest 0/6, steps 0/6, bumps 1.30. Either the intermediate dose
  is a no-mans-land (enough drift to lose the init's hill gait, too much pull to relearn) or
  one-seed variance; tw_f2 (same arm, seed 20260918, terrain judge) decides.
- READING: precision and terrain levers fight over the same weights at u1000 — but the policy's
  terrain inputs (footfall block, zeros on flat) already SEPARATE the regimes at inference;
  the conflict is training interference, not capacity. tw_g probes the cheap resolution:
  initialize FROM the terrain master (tw_a2 u1000), anchor 3e-3 toward THAT init, train
  FLAT-ONLY (no terrain) — practice restores precision where it is cheap while the anchor
  holds the terrain gait in the weights. If tw_g holds crest/steps near tw_a2 while lifting
  flat toward tw_f, the two-phase recipe (levers free, then anchored flat polish) is the
  consolidation path; distillation of the two masters is the fallback build.

2026-11-04 — MILESTONE: tw_g passes turn_-0.40 (first in project history) and sets subset-36 = 32/36; the two-phase recipe works
- tw_f2 (anchor 1e-3, seed 20260918, terrain judge; terrain_eval_tw_f2_u1000/): hills 4/7 and
  steps 1.5 all 6/6 — tw_f's hill-7 loss was one-seed noise; the dose map stands (1e-3 = flat
  champion, no terrain levers).
- tw_g (two-phase: INIT = tw_a2 u1000 the terrain master, then 400 FLAT-ONLY updates under
  anchor 3e-3 toward that init, v45 task; checkpoint in tw_g/, outputs full96_tw_g_u400/,
  terrain_eval_tw_g_u400/, omni15_tw_g_u400/):
  * turn 6/6 — turn_-0.40 PASSES for the first time on any checkpoint, with margin (yaw
    0.0922 vs 0.1, demand 0.0005, tilt rms 0.53 — the levelest turn ever recorded). The
    anchored flat polish fixed the clockwise fall that round-B practice and the v47 mirror
    could not.
  * quiet 2/2, stops 10/10, total 35/96 (ties u600_001), subset-36 32/36 — NEW RECORD (d2
    31) — misses only formal__0.00/0.16 (tilt rms 0.725/0.945, a whisker over the bound) and
    translate_0.10_157.5/180 (the rear bearings the forward bias under-practises).
  * TERRAIN RETAINED through the flat polish, as the anchor theory predicted: crest 5/6 at
    full distance, steps 2.5 2/6, bumps median 2.08. Phase 1 buys the levers free; phase 2
    buys precision anchored. The interference knot is cut by sequencing, not compromise.
- Phase-2 arms next, one change each from tw_g's exact recipe: tw_h = v49 (roll_pitch_weight
  0.05 -> 0.3, the lean/formal lever), tw_i = v50 (walk_forward_bias dropped, uniform
  bearings, the rear-translate/omni practice lever), tw_g2 = seed repeat (variance).

2026-11-04 — Phase-2 round: v50 fixes its rear bearing (subset-36 = 33/36, new record) but both levers pay real terrain; the seed pair brackets the noise
- All three arms share tw_g's exact recipe (init = tw_a2 u1000, 400 flat-only updates, anchor
  3e-3) at commit 6df9366c, one change each, judged by eval_checkpoint + omni15.
- tw_g2 = seed repeat (20260918; checkpoint 2cd9282a, declaration d7b65be1, outputs
  full96_tw_g2_u400/, terrain_eval_tw_g2_u400/, omni15_tw_g2_u400/): total 33/96, subset-36
  31/36 (turn_+0.40 drops on yaw error), crest 4/6, steps 2.5 3/6, bumps 0/6, omni 2/16.
  Same-recipe seed spread is therefore ±1 case per family, ±2 total: a single-case lever
  effect is noise unless BOTH base seeds agree on that case.
- tw_h = v49 (roll_pitch_weight 0.05 -> 0.3; checkpoint a1805a2e, declaration 1db21721,
  outputs full96_tw_h_u400/, terrain_eval_tw_h_u400/, omni15_tw_h_u400/): total 35/96,
  subset 32/36 — formal UNMOVED (0/4; formal__0.00 tilt rms 0.602 vs the base pair
  0.725/0.632, inside the seed spread) — and the crest is GONE (0/6 vs the pair 5/4, steps
  2.5 1/6). Cost without demonstrated benefit: the lean lever is retired at this dose.
- tw_i = v50 (walk_forward_bias dropped; checkpoint 03465878, declaration f8e5a4ab, outputs
  full96_tw_i_u400/, terrain_eval_tw_i_u400/, omni15_tw_i_u400/): translate_0.10_180 PASSES
  — both base seeds fail it, the first rear-translate pass on any checkpoint — subset-36
  33/36 (new record; misses formal__0.00/0.16 and translate_0.10_157.5), total 36/96, the
  highest any surrogate leg has posted (native u600_001 is 35). COST: crest 1/6 and steps
  2.5 0/6, outside the pair's spread — the forward bias is load-bearing for the terrain
  gait, consistent with the v46/tw_b mix findings. omni 2/16: bearing exposure alone does
  not touch the demand wall.
- Residual walls, with tw_g still the balanced champion (35/96 + crest 5/6): formal tilt,
  translate_0.10_157.5, omni demand, and the ruled turn-while-walk scope (gentle 1/16,
  combined 1/8, transition 0/1). The arc families fail NEAR-threshold — planar 0.024-0.044
  vs 0.03, yaw 0.053-0.061, demand 0.006-0.009 vs 0.005 — with demand co-failing on all 16
  gentle cases (the omni15 wall again), and the transition misses only its three arc
  segments. One mechanism — smoother, lower-torque arcing under the anchored terrain gait —
  covers roughly 25 of the remaining cases.
- Round 7 asks what buys arcs inside the two-phase recipe, one change each from tw_g:
  tw_j = 1200 anchored flat updates (does anchored practice accumulate arc skill past
  400?); tw_k = anchor 1e-3 (displacement budget for gait smoothing — the short flat leg
  from the terrain master is the dose map's untested corner); tw_l = destination_v51
  (arc share 0.25 -> 0.50 at commit d78de735 — strong persistent gradients are what the
  anchor design lets displace).

2026-11-04 — Round 7: arcs cannot be bought inside the anchored polish; surrogate PPO only erodes the native init's arc skill; tw_l posts the first fully consistent crest
- All arms init = tw_a2 u1000, flat-only, judged by eval_checkpoint + omni15 at commit d78de735.
- tw_l = destination_v51, arc share 0.25 -> 0.50 (checkpoint 5eec3cc7, declaration cc0bc5dd,
  outputs full96_tw_l_u400/, terrain_eval_tw_l_u400/, omni15_tw_l_u400/): gentle 1/16,
  combined 0/8, omni 1/16 — doubling arc practice bought NOTHING on the scored arcs — but
  hill 9.5 is 6/6 AT FULL DISTANCE (3.76-3.78 m), the first fully consistent crest on any
  checkpoint, steps 2.5 2/6. The arc share is a terrain lever (third confirmation), not an
  arc lever. Cost: turn_-0.40 fails again (planar). One seed; tw_l2 decides.
- tw_k = anchor 1e-3 (checkpoint 9c354514, declaration dca8fafc, outputs full96_tw_k_u400/,
  terrain_eval_tw_k_u400/, omni15_tw_k_u400/): gentle 2/16, total 34/96, turn 6/6, crest
  0/6, steps 0/6 — the loose dose erodes terrain without buying arcs.
- tw_j = 1200 updates at 3e-3 (u800 checkpoint ea8be5c6, declaration d4a8f692; u1200
  checkpoint 97fb4ce9, declaration 1856912a; outputs full96_tw_j_u800/, full96_tw_j_u1200/,
  terrain triples, omni15 dirs): gentle 2 -> 1, combined 1 -> 2, total 34 -> 35; crest
  decays 2/6 -> 0/6 with length; formal__0.00 PASSES at u1200 — the FIRST formal pass in
  the anchored lineage; length grinds the tilt under the bound — but turn_+0.40 drops.
  tw_j's u400 sha differs from tw_g's (f2de1bb8 vs f884214c): threaded training is not
  bit-deterministic, so same-recipe reruns are seed-class evidence, never replicas.
- STRUCTURAL FINDING (full-96 sweep over all recorded checkpoints): natives arc —
  spark_v29_u600 gentle 15/16 combined 5/8, the v29 pair 14/16 4/8 — while NO surrogate
  leg has ever exceeded gentle 6/16, and that at u200: short legs from the native core
  keep 4-6/16 (tc9_d1_u200 6, tc9_a_u200 5), every u1000 leg decays to 0-1, and the
  terrain master sits at 0/16. Surrogate PPO erodes arc precision monotonically and never
  creates it; the anchor slows the erosion but cannot restore what the init lacks. The
  scored arc families therefore wait on the NATIVE v45 u1000 (queued on Spark, pack
  terrain_isaac_ppo_v45_u1000_seed20260917_20261103_001), whose mix practises exactly the
  scored arcs; this campaign's product is the polish recipe that preserves them (short
  anchored legs, by this round's erosion curve).
- Distillation across lineages is SHELVED: the native arc teachers (v29 family) are
  ghost_v11-input policies, the student is terrain_v2 — cross-spec cloning is real
  plumbing, and the queued v45 native will be an input-compatible teacher.
- Round 8, one change per arm, same init and judges: tw_l2 = v51 seed 20260918 (do the
  crest 6/6 and the turn_-0.40 loss reproduce?); tw_m = destination_v52 (v51 +
  walk_forward_bias None, commit c4b018c2: do the crest lever and the rear-bearing lever
  compose? judged against tw_l and tw_i, its single-change parents); tw_n = v45 at 1200
  updates, anchor 6e-3 (does a tighter leash keep turn and crest while length still buys
  formal__0.00?).

2026-11-04 — Round 8: the anchored-polish space is at its noise floor; the crest was a coin flip and the levers do not compose
- tw_l2 = v51 seed 20260918 (checkpoint a829d3d3, declaration 0ad36c29, outputs
  full96_tw_l2_u400/, terrain_eval_tw_l2_u400/, omni15_tw_l2_u400/): crest 0/6 vs tw_l's
  6/6 — the first consistent crest did NOT reproduce; v51's crest is bimodal across seeds
  (6/6 vs 0/6). steps 3/6, total 34/96, turn 6/6 (tw_l's turn_-0.40 loss was noise).
- tw_m = v52 composition (checkpoint 4c82e5d1, declaration 4bf08175, outputs
  full96_tw_m_u400/, terrain_eval_tw_m_u400/, omni15_tw_m_u400/): NEITHER parent's win
  carried — crest 2/6 (tw_l 6/6), translate 13/32 (tw_i 15/32), and a stop dropped (9/10).
  The arc-share and uniform-bearing levers do not compose.
- tw_n = v45 u1200 anchor 6e-3 (checkpoint fcad6faf, declaration 7136ebbf, outputs
  full96_tw_n_u1200/, terrain_eval_tw_n_u1200/, omni15_tw_n_u1200/): formal 1/4 and turn
  6/6 held together at length for the first time, but quiet 1/2, crest 0/6, total 34/96,
  omni 3/16. No dominance over tw_j.
- VERDICT: ten polish arms (g, g2, h, i, j, k, l, l2, m, n) span 33-36/96 with
  interchangeable ±1-2 case trades; single reward levers flip cases within seed variance
  and do not compose. The anchored-polish space from the terrain master is exhausted short
  of the goal.

2026-11-04 — The input-compatible arc teacher was on disk all along: v29_terrain_v2
- runs/curriculum/starts/v29_terrain_v2.pt (sha 5b32fba4..., derivation record beside it)
  is spark_v29_u600 — gentle 15/16, combined 5/8, total 53/96, the best arc-er ever
  recorded — widened ghost_v11 -> terrain_v2 by zero columns with zero Adam moments
  (reproduction check max abs error 0.0): bit-equal behavior in the campaign's own input
  space. It is the queued Spark v45 pack's init, and NOTHING was ever surrogate-trained
  from it. Lineage check: tc9_a initialized from terrain_isaac_u600_002 (gentle 4/16), so
  tc9_a_u200's 5/16 is noise above ITS init — the erosion ruling stands.
- Smoke (omni15 0deg, --task experiment_task(destination_v45), --ppo-config
  starts/v29_terrain_v2_eval_ppo_config.json = the record's own ppo_config with the
  maker's intended_wrapper_options, an evaluator option, bounds and cases untouched):
  loads and PASSES cleanly (planar 0.0228, velocity 0.1505).
- Round 9, the teacher round: (1) judge v29tv2 on the full-96 and omni15 with that
  task/config (terrain_eval takes no task override and the teacher's terrain is irrelevant
  to its role — skipped, recorded here); (2) tw_o = the tw_g recipe from the OTHER master:
  init v29_terrain_v2, 400 flat-only updates, v45 task, anchor 3e-3 toward that init,
  seed 20260917 — does the anchor hold ARCS through a surrogate leg the way tw_g held
  terrain? If gentle survives near 15/16 while stops and quiet lift, the surrogate has an
  arc-capable flat master; if arcs erode anyway, the teacher stays frozen and labels a
  distillation student instead.

2026-11-04 — MILESTONE round 9: the anchor holds arcs — tw_o ties the best checkpoint ever recorded (51/96) as a surrogate leg
- v29tv2 judged (flat judges; --task experiment_task(destination_v45), --ppo-config
  starts/v29_terrain_v2_eval_ppo_config.json; outputs full96_v29tv2/, omni15_v29tv2/;
  terrain_eval skipped — it takes no task override and the teacher's terrain is irrelevant
  to its role): gentle 14/16, combined 4/8, translate 16/32, turn 5/6 (turn_-0.40 fails
  all three bounds — the chirality stall), stop 9/10 (stop__forward_fast), quiet 2/2,
  formal 0/4, TOTAL 51/96, subset-36 33/36, omni15 15/16 — the widened teacher is ALSO an
  omni master; the historical "native baseline 4/16" belonged to v29_ctrl, not this u600.
- tw_o = tw_g's recipe from the OTHER master (init v29_terrain_v2, 400 flat-only updates,
  v45 task, anchor 3e-3 toward that init, seed 20260917; checkpoint 9349dc46, declaration
  33096864, outputs full96_tw_o_u400/, terrain_eval_tw_o_u400/, omni15_tw_o_u400/):
  * gentle 12/16 after 400 surrogate updates — THE ANCHOR HOLDS ARC SKILL (historical
    free-training erosion: 15 -> ~5 in 200 updates). combined 6/8 — best ever recorded on
    any checkpoint, native included.
  * translate 16/32 with EVERY subset translate passing — the 157.5/180 rear misses are
    gone. omni15 13/16 (three demand-only misses, 112.5/135/157.5). quiet 2/2. TOTAL
    51/96 — ties v29_ctrl: the first surrogate leg at the native ceiling. subset-36 33/36
    (ties the record) missing ONLY turn_-0.40 and formal__0.00/0.16.
  * Terrain: hills 4/7 and steps 1.5 all 6/6 from a never-terrain-trained gait — the
    consistent-three need no terrain levers; crest, steps 2.5 and bumps 0/6 as expected.
  * Residual flat gaps, all near-threshold: 4 gentle (3 demand-only, 1 yaw), 2 combined,
    3 omni demand, turn_-0.40 (tw_g's polish fixed this case on the terrain lineage, so a
    fix exists inside the recipe class), the formal pair, stop__forward_fast (inherited
    from the teacher; the terrain lineage passes it), transition.
- Round 10, one change each: judge tw_o's EXISTING u200 checkpoint (an erosion-curve point,
  no training); tw_o2 = seed repeat (20260918); tw_p = teacher-init u1200 at 3e-3 (tw_j
  ground formal__0.00 under the bound at u1200 on the other lineage — does length buy
  formal here while the anchor holds arcs?).
- Composition endgame: the flat side is nearly closed on the teacher lineage while crest
  and steps live on the terrain lineage. Next after round 10: rerun the two-phase recipe
  from the better base (phase 1 terrain u1000 from tw_o, free; phase 2 anchored flat
  polish) — the surrogate rehearsal of exactly what the queued native v45 run does from
  this same init at native fidelity. Distillation remains the fallback if phase-1 erosion
  from tw_o repeats the old lineage's.

## Round 10 — the erosion curve pays: two 52/96 records and the first perfect omni (2026-11-04)

- tw_o_u200 (judge-only; tw_o's existing checkpoint_update000200.pt, 200 anchored updates
  from the teacher; checkpoint d50fcb0b, declaration e520fec3, outputs full96_tw_o_u200/,
  terrain_eval_tw_o_u200/, omni15_tw_o_u200/): TOTAL 52/96 — NEW all-time record, first
  checkpoint past the native ceiling. omni15 16/16 — FIRST PERFECT OMNI on any checkpoint,
  native included. gentle 15/16, one demand-only whisker from perfect
  (gentle_arc_135deg_+0.05); combined 6/8; consistent-three terrain 6/6 each. Regressions
  vs u400: turn 4/6 (turn_+0.40 yaw joins turn_-0.40), quiet 1/2 (stage2_long joint
  range), subset-36 31/36. Teacher-closeness maxes arcs and omni; 200 updates already
  grind the rear translates the teacher misses.
- tw_o2 = seed repeat of tw_o (seed 20260918; checkpoint b0abaae8, declaration a758ea44,
  outputs full96_tw_o2_u400/, terrain_eval_tw_o2_u400/, omni15_tw_o2_u400/): 49/96,
  gentle 12/16, combined 5/8, omni15 13/16, subset 32/36 — the anchored-teacher recipe
  reproduces at u400 within seed noise (gentle 12 on both seeds). tw_o is a class, not a
  fluke.
- tw_p = teacher-init u1200 at anchor 3e-3 (checkpoint 9fc7a7ce, declaration edf7104c,
  outputs full96_tw_p_u1200/, terrain_eval_tw_p_u1200/, omni15_tw_p_u1200/): TOTAL 52/96
  ties the new record. turn 6/6 — the first teacher-lineage turn_-0.40 pass: chirality
  yields to LENGTH on this lineage too (matches tw_j/tw_g on the terrain lineage).
  transition 1/1 — FIRST TRANSITION PASS ON ANY CHECKPOINT EVER. combined 7/8 — new best
  (only combined_0deg_-0.20 left, planar+yaw; never passed by anything). omni15 16/16.
  Costs of flat-only length: stand fidget (quiet__stand, quiet__stage2_long, stop__stand
  all on max_joint_position_range_rad), gentle eroded to 13/16, hill 7deg broke (2/6) —
  flat-only u1200 overfits flat. formal__0.00/0.16 now fail ONLY tilt_rms — length grinds
  tilt toward the bound but does not pass it on this lineage (the terrain lineage's u1200
  arms passed formal__0.00: tilt looks terrain-taught).
- Decomposition of the goal, by evidence: teacher-closeness gives arcs + omni + quiet
  (u200); length gives chirality + transition + combined rears (u1200); terrain gives
  tilt/formal + the 7deg hill (tw_j/tw_a2 lineage). The conjunction needs all three in
  one checkpoint, plus two never-passed cases (combined_0deg_-0.20; gentle 135+0.05
  demand whisker at u200).
- Round 11, one change each: (a) judge-only u200 replicates — tw_p_u200 (same seed,
  thread-noise twin) and tw_o2_u200 (other seed): is the u200 sweet spot a reliable
  class? (b) tw_r = SELF-ANCHORED POLISH: init AND anchor = tw_p's u1200, 3e-3, 200
  updates — does a short anchored polish restore stand-quiet and gentle while the anchor
  holds length's turn/transition/combined? (c) tw_s = ANCHORED TERRAIN LENGTH: tw_p's
  exact recipe + --terrain mixed_v5 (matched control = tw_p itself) — the recorded
  two-phase design collapsed into one phase by the anchor discovery: anchor holds arcs,
  terrain teaches tilt and keeps hill 7, length buys chirality/transition. Phase-2 polish
  (b's recipe) remains available on its best checkpoint.

## Correction — v45 runner pointed at a nonexistent pack name (2026-11-04)

- A queue audit at check 218 (18 h queued, GPU still 94-95%) found run_v45_u1000.sh's
  N was a half-edit: `terrain_isaac_ppo_v45_u1000_seed20260917_20261102_002` — the v45
  name grafted onto the v44 pack's date suffix. No such tar exists in /home/julian and
  nothing is extracted under /srv, so the launch would have failed at extraction the
  moment the GPU freed (swallowed by `|| true`, ending in a launch_failed marker and a
  lost window).
- Fix: runner PID 3820851 stopped by exact PID (self-match trap avoided), N repointed to
  terrain_isaac_ppo_v45_u1000_seed20260917_20261103_001 (the staged tar; top-level entry
  verified by `tarfile -l`; the pack of the entry above, commit 1531e8b4 / freeze
  4a5c2a22), stale v44 header refs corrected. Relaunched detached as PID 2814046;
  restart noted in v45_u1000.log; polling resumed at queue[1] check 1, GPU 94%. Marker
  path and the local 30 h waiter are unchanged. Nothing launched during the 18 h wait,
  so no result is affected; the queue history stays in the log above the restart line.

## Round 11 — the sweet spot is real, the polish dead-ends, and the formal wall is structural (2026-11-04)

- tw_p_u200 (judge-only; same seed as tw_o, thread-noise twin; checkpoint 9ff0eb79,
  declaration 8aeafdc2, outputs full96_tw_p_u200/, terrain_eval_tw_p_u200/,
  omni15_tw_p_u200/): 52/96, gentle 15/16, combined 6/8, omni15 16/16, consistent-three
  6/6 — family-for-family IDENTICAL to tw_o_u200. The u200 anchored-teacher point
  reproduces exactly under thread noise at seed 20260917.
- tw_o2_u200 (other seed, 20260918; checkpoint c38e41fa, declaration 785eec3f): 47/96,
  gentle 13, combined 3, omni15 14 — the point is seed-sensitive (47-52 across seeds);
  20260917 is the strong draw.
- tw_r = self-anchored polish of tw_p_u1200 (init = anchor = that checkpoint, 3e-3, 200
  updates; checkpoint ccb08f8d, declaration 63106119): 49/96. Turn 6/6 HELD, but quiet
  0/2 (the anchor preserves the fidget it was meant to cure), combined 7->4, transition
  1->0, omni 16->12, hill7 still broken. DEAD END: the anchor holds the init's flaws as
  faithfully as its skills, and 200 free-ish flat updates erode the length gains.
- tw_s = tw_p's exact recipe + --terrain mixed_v5 (matched control tw_p; checkpoint
  c56bad44, declaration bf68f7c9): 49/96. Terrain at length CURES the stand fidget
  (joint range 0.0158/0.0084/0.0058 vs tw_p's 0.1115/0.0400/0.0315, bound 0.02; quiet
  2/2) and keeps transition 1/1 and hill7 6/6 — but erodes gentle to 9/16 and omni to
  11/16 even anchored, and does NOT touch formal.
- MARGINS (metrics vs bounds, from the full-96 JSONs): the remaining in-goal cases split
  into two classes.
  * Whiskers (1-15% over, flip with small gains or noise): gentle_arc_135deg_+0.05
    demand 0.0051 vs 0.005; turn_+0.40 yaw 0.1017 vs 0.10 at u200 (u400 passes it);
    combined_0deg_-0.20 planar 0.025-0.027 and yaw 0.060-0.065 vs 0.06;
    quiet__stage2_long joint range 0.0229 vs 0.02 at u200. turn_-0.40 at u1200 already
    passes (yaw 0.094 vs 0.10).
  * STRUCTURAL: formal__0.00/0.16 tilt RMS. The whole teacher lineage sits at 0.84-0.88
    deg (bound 0.50) at stand and 1.18-1.49 (bound 0.90) at 0.16 — unmoved by length
    (u200 to u1200) and by terrain (tw_s 0.884). The terrain-master lineage holds
    0.33-0.44 and 0.83-0.96 (tw_j passes 0.16's tilt_rms and fails it on
    maximum_tilt/yaw_rate_rmse). Cause found in the reward: v45's tilt term is weight
    0.5 at a 5 deg scale, pricing a 0.84 deg wobble at ~0.014 reward units — no
    gradient. NOT a lineage property: an unpressured term.
- destination_v53 registered (commit d5e5a4b6, tests green): v45 + tilt_weight 15 (one
  field; 0.84 deg now costs ~0.42 units). Round 12, one change each vs tw_o/tw_p (task
  v45->v53): tw_t = v53 u200 anchored-teacher (does tilt drop below 0.50 while the
  anchor holds arcs/omni?); tw_u = v53 u1200 (the full-conjunction attempt: length buys
  turn_-0.40/transition/combined, tilt pressure grinds formal the whole way, and the
  stand-fidget question — does tilt pressure still the stand or fight it?).

## Round 12 — MILESTONE: the tilt lever cracks formal; 53/96 and subset 35/36 (2026-11-04)

- tw_t = v53 (tilt_weight 15, commit d5e5a4b6) under the u200 anchored-teacher recipe
  (checkpoint 7dde4950, declaration 0b43dd2e, outputs full96_tw_t_u200/,
  terrain_eval_tw_t_u200/, omni15_tw_t_u200/): 52/96, formal 2/4 — THE FIRST
  TEACHER-LINEAGE FORMAL PASSES (0.00 tilt 0.402 vs bound .50; 0.16 tilt 0.850 vs .90),
  in 200 updates. quiet 2/2 (tilt pressure also cures the stage2_long fidget), subset
  35/36, hill7 and steps 6/6. Cost at u200: gentle 13 (90deg pair demand), omni 12,
  combined 4.
- tw_u = v53 at u1200 (checkpoint d503dfaa, declaration f3eceefd, outputs
  full96_tw_u_u1200/, terrain_eval_tw_u_u1200/, omni15_tw_u_u1200/): TOTAL 53/96 —
  NEW ALL-TIME RECORD. subset-36 35/36 (record; the ONLY miss is turn_-0.40 at planar
  0.0252 vs .025 and yaw 0.1048 vs .10). formal__0.00 tilt 0.331 and __0.16 tilt 0.626 —
  crushed with 30% margin; the "structural" wall was exactly the unpriced reward term.
  quiet 2/2 AND stop__stand passes at u1200 (under v45 length destroyed them: tilt
  pressure prevents the stand-fidget), transition 1/1, hill7 6/6 (v45's u1200 broke it),
  omni15 15/16, consistent-three all 6/6.
- EVERY remaining in-goal miss on tw_u is a whisker: turn_-0.40 planar +0.8% / yaw
  +4.8%; gentle 135deg demand +4%/+18%, 315deg yaw +1%/+3%; combined 0deg_-0.20 planar
  +1% yaw +11%, 180deg_-0.20 planar +8%, 180deg_+0.20 planar +2%; omni
  translate_0.15_315deg demand +20%. Nothing structural remains anywhere in the goal.
- The v53 trajectory u200->u1200 shows length grinding ripple (omni 12->15, turn_-0.40
  yaw 0.353->0.105, formal deepening) with gentle roughly flat. Round 13, one change
  each: judge-only sweep of tw_u's intermediates (u400/u600/u800/u1000 — whiskers flip
  across checkpoints; one point may hold more of the conjunction); tw_v = v53 u2000
  (length extrapolation at the chirality and demand whiskers); tw_u2 = seed repeat of
  tw_u (20260918 — the whisker set is partly a seed draw).

## Round 13 — the v53 length curve: a noisy 54-plateau, and the gains that reproduce (2026-11-04)

- Judge-only sweep of tw_u's own checkpoints plus two arms, judged as always:
  * tw_u_u400 (acf7381f/c76ce960): 54/96 — ties the record with gentle 15/16, combined 5/8,
    formal 2/4, omni15 15/16, subset 34/36 (turn_-0.40, stage2_long 0.0243).
  * tw_u_u800 (4369157f/518aa8c8): 54/96, combined 7/8, gentle 14/16, omni 15/16, subset
    33/36.
  * tw_u_u1000 (5825fefb/449b33f0): 51/96, omni 13/16 — the plateau dips.
  * tw_v = v53 u2000 (checkpoint at 2000 updates): 50/96, omni 12/16 — length past the
    plateau HURTS; the lever saturates in the u400-u1200 band.
  * tw_u2 = seed repeat of tw_u (20260918; dd3368ac/390092bb): 50/96, gentle 12, combined 3,
    omni 12 — but formal 2/4 AND transition 1/1 REPRODUCE on the second seed: the tilt
    lever's gains are structural, the rest of the point spread (50-54) is seed/thread noise.
- The v53 curve (seed 20260917): u200 52, u400 54, u800 54, u1000 51, u1200 53, u2000 50.
  Case-level, the plateau points differ mostly by which whiskers land: every in-goal miss
  at every plateau point is within ~20% of its bound EXCEPT turn_-0.40 before u1200
  (yaw 0.355/0.218 at u400/u800 — chirality still needs the full length) and the
  right-turn transition segment (passes only at u1200, both seeds).
- Remaining whiskers cluster by mechanism at the structurally best point (tw_u_u1200,
  subset 35/36): YAW tracking (turn_-0.40 at +4.8%, gentle_arc_315 pair at +1-3%,
  combined_0deg_-0.20 yaw at +11%, turn_+0.40 at u800 +6.6%) and DEMAND spikes
  (gentle_135 pair, omni15 135/315deg at +4-20%). The v47 mirror stays parked (disproven
  at dose, round 4 part 2); length already buys chirality here.
- Round 14, one field each against v53 at matched length and seed (control tw_u_u1200):
  tw_w = v54 (yaw_tracking_weight 0.5 -> 1.0) — the yaw-whisker class; tw_x = v55
  (acceleration_weight 0.1 -> 0.2) — the demand-spike class; tw_u3 = v53 seed 20260919 —
  the whisker lottery at n=3 seeds.
- Round 14 results (runner scratchpad/round14.sh, all u1200 from v29_terrain_v2,
  judged by eval_checkpoint.py + omni15_suite.py):
  - tw_w (destination_v54, yaw_tracking_weight 1.0 + termination 58.5, seed 20260917):
    51/96, checkpoint b0c8f4fe, declaration d951b581, outputs full96_tw_w_u1200/,
    terrain_eval_tw_w_u1200/, omni15_tw_w_u1200/ (12/16), consistent-three 6/6 each.
    vs control tw_u_u1200 (53/96): gained forward_fast only; lost stage2_long quiet
    (range 0.0227), combined_270_+0.20 (yaw 0.0612), gentle_225_-0.05 (yaw 0.0621);
    the 315-yaw pair it targeted still fails; omni -3 (new demand spikes at
    112.5/135/157.5). Doubling yaw tracking moves and multiplies the yaw whiskers
    rather than curing them. PARKED.
  - tw_x (destination_v55, acceleration_weight 0.2, seed 20260917): 53/96, checkpoint
    056c7eb0, declaration 07d61a10, omni 12/16, consistent-three 6/6. vs control:
    gained gentle_135_-0.05 (one demand-class cure), lost combined_270_+0.20
    (yaw 0.0606); omni -3 with new demand misses at 135/157.5/292.5. The acceleration
    price trades demand whiskers around rather than removing them. PARKED.
  - tw_u3 (destination_v53, seed 20260919): 55/96 RECORD, checkpoint ebbc3d38,
    declaration be8fb80c, outputs full96_tw_u3_u1200/, terrain_eval_tw_u3_u1200/,
    omni15_tw_u3_u1200/. Gained vs control: combined_180 pair, gentle_135_-0.05,
    gentle_315 pair (the control's yaw pair - passable at u1200 after all). Lost:
    stage2_long quiet (0.0206/0.02), gentle_270_-0.05 (yaw 0.0607), full_program
    (s_curve last-second planar). Family: gentle 14/16, combined 7/8, formal 2/4,
    omni 15/16 (270deg physical-bounds composite, every scalar inside bounds),
    subset-36 34/36 (quiet + turn_-0.40 yaw 0.1028). Terrain: hill4 6/6, steps1.5 6/6,
    hill7 2/6 (best/median 1.85/1.61 m) - the consistent-three BREAKS on this seed.
  - formal 2/4 holds on all three arms: the v53 structural gain is 5-for-5 across
    seeds and levers. Seed spread of v53@u1200 now 53/50/55 (n=3). Goal-scope misses
    of the record are all whiskers (+1.2% to +14%) except the two regressions the
    jackpot seed introduced (transition, hill7): arcs/omni anti-correlate with the
    quiet/transition/terrain axis across draws.
- Round 15 (runner scratchpad/round15.sh): (a) harvest judgings, no training -
  tw_u3_u400, tw_u3_u800, tw_u2_u800, saved intermediates of existing runs: does the
  jackpot seed hold its arc gains at plateau lengths while keeping transition + hill7,
  plus a free seed-2 draw; (b) one training arm tw_y = destination_v53 +
  --terrain terrains/mixed_v5.json, u1200, seed 20260917 - terrain training is
  recorded to cure the stand fidget and hold transition/hill7 while eroding arcs at
  v45's reward; the open question is whether v53's formal survives terrain training
  and what it costs in arcs.
- Round 15 results (runner scratchpad/round15.sh):
  - Harvest of saved intermediates, judged vs control tw_u_u1200 (53/96):
    tw_u3_u400 48/96 (subset 34, omni 15, gentle 11, combined 3, quiet 2/2,
    transition 0, hill7 4/6), checkpoint bb62be95, declaration e875a1b0.
    tw_u3_u800 51/96 (subset 34, omni 14, gentle 12, combined 5, quiet 1/2,
    transition 0, hill7 6/6), checkpoint 216038b9, declaration b2b075c0.
    tw_u2_u800 50/96 (subset 33, omni 11, gentle 10, combined 6, quiet 1/2,
    transition 1/1, hill7 6/6), checkpoint 6b7df8fc, declaration feac8caa.
    Seed 3's curve rises monotonically (48/51/55): the arc gains are a late-length
    effect. turn_-0.40 is only a whisker at u1200 (yaw 0.3533 at u400, 0.2453 at
    u800, 0.1028 at u1200): length grinds it down and u1200 is not yet enough.
    full_program fails a different segment per length (reverse-diagonal at u400 and
    u800, s_curve at u1200); seed 3 never passes it. hill7 is non-monotone
    (4/6, 6/6, 2/6 across the lengths).
  - tw_y (destination_v53 + terrain mixed_v5, u1200, seed 20260917): 47/96,
    subset 33, omni 13, formal 0/4 (tilt 0.6257 standing / 1.0336 moving - the v53
    fix UNDONE), combined 2/8, quiet 2/2, hill7 6/6, training reward ends negative
    (-6.2). Checkpoint be543323, declaration ecd85f59. Terrain training restores
    quiet and holds the hills but destroys formal and combined at v53's tilt price:
    slope tilt is unavoidable, so tilt_weight 15 dominates the return. PARKED;
    terrain training is also unnecessary for the goal, since flat-trained v53
    checkpoints already pass the consistent three.
- Round 16 (runner scratchpad/round16.sh): (a) harvest tw_v_u1400 and tw_v_u1600,
  saved intermediates of the seed-1 u2000 arm: does more length crack turn_-0.40
  before quiet/transition/hill7 collapse (tw_v_u2000 was 50, saturated);
  (b) two fresh draws at the record recipe, tw_u4 = v53 u1200 seed 20260920 and
  tw_u5 = v53 u1200 seed 20260921: the u1200 lottery at n=5 (spread 50/53/55).
- Round 16 results (runner scratchpad/round16.sh):
  - tw_v_u1400 (v53 seed 20260917, the u2000 arm's saved intermediate): 55/96 and the
    FIRST PERFECT SUBSET-36 (36/36), with turn 6/6 - the first turn_-0.40 pass in the
    teacher lineage (yaw 0.3533 at u400 -> 0.2453 at u800 -> 0.1028 at u1200 -> pass
    at u1400) - quiet 2/2, transition 1/1, formal 2/4, consistent-three 6/6.
    Checkpoint 8a4b30d9, declaration 1e0a6f2b, outputs full96_tw_v_u1400/,
    terrain_eval_tw_v_u1400/, omni15_tw_v_u1400/. Remaining goal misses: the gentle
    315 pair (demand 0.0098; yaw 0.0601 + demand 0.0055), four combined (planar
    0.0255-0.0299, yaw 0.0611-0.0683), omni 12/16 (90/247.5 physical composite,
    292.5/315 demand 0.0051/0.0078). The conjunction is now arcs + omni only.
  - tw_v_u1600: 50/96 (quiet 1/2, gentle 12, combined 3, omni 11), checkpoint
    c81a2a0d, declaration 082a0a66. The fidget returns past u1400 and the window
    closes: seed 1's peak sits at u1400.
  - Fresh u1200 draws: tw_u4 (seed 20260920) 54/96, subset 35 (turn_-0.40 yaw
    0.1152), omni 14, quiet 2/2, transition 1/1, hill7 5/6 (consistent-three fails,
    median 1.84 m), checkpoint 7cc51cea, declaration 8d652deb. tw_u5 (seed 20260921)
    52/96, subset 34 (quiet 0.0270), omni 14, hill7 3/6, checkpoint 23e216be,
    declaration 4de31092. The u1200 lottery at n=5 reads 53/50/55/54/52; no draw
    matches tw_v_u1400's alignment. Length placement beat the seed lottery.
- Round 17 (runner scratchpad/round17.sh): tw_z = destination_v56, situation-priced
  tracking (registered and committed, ad38c45a: each tracking weight grows with how
  hard its axis is commanded, both validators count the full factor, zero-default
  byte-identical), trained to u1400 on seed 20260917 and judged at u1200 (vs
  tw_u_u1200, the one-field read) and u1400 (vs tw_v_u1400, the aligned-length
  read). The four combined misses are joint speed+yaw tracking whiskers, exactly the
  class v56 prices at 1.5x per axis; the gentle-315/omni demand spikes stay for a
  demand-side situation price if v56 reads clean. Plus bracket harvest tw_v_u1300
  and tw_v_u1500 to map the width of the u1400 window.
- Round 17 results (runner scratchpad/round17.sh):
  - tw_z = destination_v56, situation-priced tracking, u1400 seed 20260917.
    tw_z_u1400: 56/96 RECORD, subset 35/36, omni 15/16, turn 6/6, gentle 14/16,
    combined 6/8, formal 2/4, transition 1/1, consistent-three 6/6. Checkpoint
    706631f2, declaration 10917eca, outputs full96_tw_z_u1400/ etc. vs tw_v_u1400
    (55, subset 36): gained combined_180 pair, combined_270_+0.20,
    gentle_315_+0.05; lost stage2_long quiet (range 0.0309 vs 0.02) and two
    whiskers that moved (combined_0_+0.20 yaw 0.0607, gentle_135_+0.05 demand
    0.0073). Goal gap now SIX items: the quiet range, combined_0 pair (yaw
    0.0607/0.0697 + planar 0.0250), gentle_135_+0.05 demand, gentle_315_-0.05
    (yaw 0.0604), omni 157.5 (demand 0.0056).
  - tw_z_u1200: 55/96, subset 35/36, omni 14/16, turn 6/6 - the one-field read vs
    tw_u_u1200 (53): +2 total, turn_-0.40 passes 200 updates EARLIER than plain
    v53 ever manages, combined +1, gentle +1, quiet -1 (0.0217), omni -1.
    Checkpoint fbb4bf04, declaration df14e002. VERDICT: situation pricing moves
    the tracking classes it targets without collateral on moving families; its
    cost is the stand fidget.
  - Bracket: tw_v_u1300 54/96 (quiet 1/2, turn 5/6, omni 15), checkpoint da4b748d,
    declaration da74932c; tw_v_u1500 51/96 (quiet 1/2, hill7 5/6, omni 11),
    checkpoint 3856be31, declaration 049e3673. Seed 1's perfect-subset window is
    one point wide: exactly u1400.
- Round 18 (runner scratchpad/round18.sh): (a) harvest tw_z_u1300 - the fidget
  fluctuates along the trajectory (0.0217 -> 0.0309), so a neighbour may dip under
  the gate while keeping the arcs; (b) tw_z2 = destination_v57 (v56 +
  quiet_target_weight 0.5 -> 2.0, committed), u1400 seed 20260917, judged at u1300
  and u1400: price the fidget with the stillness-gated term that cannot touch the
  moving families.
- Round 18 results (runner scratchpad/round18.sh):
  - tw_z_u1300 (harvest): 55/96, quiet 1/2, combined 6/8, gentle 13, omni 15,
    turn 6/6, consistent-three 6/6; checkpoint b1d99385, declaration c45d3615.
    The tw_z trajectory carries the fidget at u1200, u1300 and u1400 alike.
  - tw_z2 = destination_v57 (v56 + quiet_target_weight 2.0), u1400 seed 20260917.
    u1300: 55/96, gentle 15/16, turn 6/6, omni 15/16, transition 1/1 - but quiet
    STILL 1/2, combined 4/8 (-2 vs tw_z) and hill7 5/6; checkpoint 8c69961a,
    declaration 7cddd4b6. u1400: 50/96 (hill7 3/6, omni 12); checkpoint ffcd0a8f,
    declaration 028aeb82. VERDICT: quiet_target x4 does not move the quiet joint
    range at either length - the slow wander evades a per-step target price -
    and it costs combined and the hills. v57 DISPROVEN at dose, PARKED.
  - The quiet 2/2 points on record (tw_u_u1200, tw_v_u1400, tw_u4_u1200) differ
    only in seed and length from quiet 1/2 neighbours: the fidget is a draw, not
    a priced behaviour, on everything tried so far.
- Round 19 (runner scratchpad/round19.sh), two fidget levers that cannot dilute
  v56's situation-priced arcs: (a) tw_z3 = destination_v56, u1400, seed 20260920 -
  the seed whose v53 draw held quiet 2/2 at u1200 (tw_u4); (b) tw_z4 =
  destination_v56, u1400, seed 20260917, anchor-weight 0.003 -> 0.006 - double the
  teacher pull (the teacher stands quiet 2/2); the open question is what it costs
  in arcs. Both judged at u1300 and u1400.
- Round 19 results (runner scratchpad/round19.sh):
  - tw_z3 = v56 seed 20260920 (the seed whose v53 draw held quiet): 51/96 at u1300
    (quiet 1/2, hill7 4/6, omni 13; checkpoint f9e12ae8, declaration 84beac71) and
    51/96 at u1400 (quiet 1/2, gentle 11, omni 12; checkpoint de34ec99, declaration
    cc2058eb). Quiet 2/2 is NOT a seed property; the quiet-seed hypothesis fails.
  - tw_z4 = v56, anchor-weight 0.006, seed 20260917: u1300 TIES THE RECORD at 56/96
    with the FIRST PERFECT OMNI15 (16/16) on a record-class checkpoint - turn 6/6,
    gentle 14/16, combined 6/8, transition 1/1, formal 2/4, consistent-three 6/6,
    subset 35/36; checkpoint 6ced5bb0, declaration 06f39854, outputs
    full96_tw_z4_u1300/ etc. u1400: 54/96, omni 14 (checkpoint 62863284,
    declaration 1dc3d152). Goal gap at u1300 is FIVE whiskers: stage2_long quiet
    range 0.0246, combined_0_-0.20 yaw 0.0649, combined_180_-0.20 planar 0.0262,
    gentle_135_+0.05 demand 0.0053, gentle_315_+0.05 demand 0.0060.
  - The doubled anchor did not cure the fidget but pulled it down (tw_z 0.0309 ->
    tw_z4 0.0246/0.0211) and bought omni for free: the anchor is a live dose ladder,
    not a parked lever.
- Round 20 (runner scratchpad/round20.sh): (a) harvest tw_z4_u1250 and tw_z4_u1350 -
  the fidget fluctuates per checkpoint and sits 5-23% over the gate on this
  trajectory, so a neighbour may dip under while keeping the 56-class axes;
  (b) tw_z5 = v56, anchor-weight 0.012, u1400, seed 20260917, judged at u1300 and
  u1400 - the next anchor step (0.003 -> 0.006 helped omni and halved nothing; the
  recorded risk is arc erosion at high anchor, so the dose curve decides).
- Round 20 results (runner scratchpad/round20.sh):
  - tw_z4 neighbours: u1250 56/96 (quiet range 0.0250, omni 14, combined 6/8,
    checkpoint d4ccc109, declaration 1b306cff); u1350 55/96 (quiet 0.0221, combined
    7/8, turn 5/6, omni 14, checkpoint 3e9db6a7, declaration 30dedc89). No neighbour
    dips under the quiet gate; u1300 stays the co-champion peak (56, omni 16).
  - tw_z5 = v56, anchor 0.012: u1300 52/96 with formal 0/4 and transition 0/1
    (checkpoint 3b5ec351, declaration a761f6c4); u1400 54/96, omni 16/16, combined
    7/8, but transition 0/1 and quiet 0.0218 (checkpoint 54506482, declaration
    6098de4a). VERDICT: anchor 0.012 pulls back toward the teacher, whose formal
    fails and whose transition never passed - the ladder closes at 0.006.
  - The fidget has now survived quiet_target x4, anchor x2 and x4, three seeds and
    every length: stage2_long's joint range sits 0.0211-0.0309 on ALL ELEVEN v56
    points while several plain-v53 draws hold 2/2. Mechanism: the existing quiet
    terms are quadratically blind to slow wander (0.002 rad/s costs ~1e-5 units),
    and situation pricing raised moving-phase returns, shifting the balance toward
    micro-adjustment. No registered term prices position wander itself.
- Round 21: build the missing term - quiet pose hold (v58 = v56 + quiet_hold): while
  still, price the squared deviation of each joint from the position captured at
  quiet onset (wrapper memory, new telemetry field, zero-weight byte-identical like
  swing_clearance). This is the only lever aimed at the metric the gate measures.
- Round 21 launched (runner scratchpad/round21.sh): quiet_hold built and committed
  (ec2c743a; rewards.quiet_hold_weight/scale, wrapper quiet_anchor_joint_rad memory
  captured at command changes and resets, emitted only when the weight is on,
  suite 402 OK). tw_q = destination_v58 (v56 + quiet_hold_weight 3.0 at scale
  0.01), anchor 0.006, u1400, seed 20260917 - one field against tw_z4 at matched
  anchor, length and seed - judged at u1250/u1300/u1350/u1400.
- Round 21 results (runner scratchpad/round21.sh): tw_q = destination_v58 (v56 +
  quiet_hold 3.0 at scale 0.01, onset-anchored), anchor 0.006, u1400, seed
  20260917, judged u1250-u1400: 51/47/49/47, quiet 1/2 then 0/2, formal 0/4 and
  transition 0/1 EVERYWHERE (checkpoints 48872147/d9004714/00e66381/64d9ac21,
  declarations 2b099937/df0b2cb6/653f1a22/154943bc). VERDICT: anchoring the hold
  at quiet ONSET fights the settle - the robot is priced for moving from its
  mid-settle pose toward a good stand, so it freezes awkwardly: tilt collapses
  (formal 0/4) and the quiets get worse, the exact opposite of the intent.
  Onset-anchoring DISPROVEN; the term's capture moved to the first still control
  (commit af change in task.py), so the hold starts at zero deviation and prices
  only post-settle wander. The registration (v58 weights) is unchanged; results
  stay bound to their revisions.
- Round 22 (runner scratchpad/round22.sh): tw_q2 = destination_v58 with the
  settled-pose capture, anchor 0.006, u1400, seed 20260917, judged u1250-u1400 -
  the corrected one-field read against tw_z4.
- Round 22 results (runner scratchpad/round22.sh): tw_q2 = destination_v58 with the
  settled-pose capture (commit ea39ba7d; round 21's correction note meant that
  commit), anchor 0.006, u1400, seed 20260917:
  - u1250 55/96: GENTLE 16/16 (first ever), quiet 2/2, omni 16/16, combined 5/8
    (checkpoint f0ca1c25, declaration 07701d21).
  - u1300 55/96: gentle 15, quiet 2/2, omni 16/16 (checkpoint a13861d9,
    declaration 10f3ec12).
  - u1350 56/96 TIES RECORD: gentle 16/16, quiet 2/2, omni 16/16, combined 6/8,
    consistent-three 6/6 (checkpoint 6eecb9c2, declaration 06393a5f).
  - u1400 55/96: quiet 2/2, transition 1/1, omni 15 (checkpoint 29c3465a,
    declaration 6d2e8bd8).
  VERDICT: the settled-pose hold cures the quiet range at every length (the first
  lever that ever moved it) and unlocks gentle and omni - but the pose frozen
  after one second leans: formal 0.00 tilt 0.57-0.71 vs 0.50, locked in because a
  held 0.02 rad correction costs ~12 hold units against ~0.14 tilt units saved.
  turn_-0.40 (yaw 0.1104-0.1134) and the transition's reverse-diagonal segment
  drop back out: the remaining walls shuffle inside a narrow capacity.
- Round 23 (runner scratchpad/round23.sh): tw_q3 = destination_v59 (v58 +
  quiet_hold_grace_controls 150, committed with suite 403 OK) - the hold captures
  and prices after a three-second settle, handing it the level pose the 15x tilt
  price has had time to find - anchor 0.006, u1400, seed 20260917, judged
  u1250-u1400.
- Round 23 results (runner scratchpad/round23.sh): tw_q3 = destination_v59 (hold
  settles three seconds), anchor 0.006, u1400, seed 20260917:
  - u1250 52/96 (gentle 12, omni 14, formal 1/4, transition 1/1; checkpoint
    424c17cb, declaration df8e5ecf).
  - u1300 56/96 TIES RECORD: formal 2/4 RESTORED, quiet 2/2 (range 0.0076 - the
    hold leaves 2.6x margin), omni 16/16, gentle 15, combined 6/8,
    consistent-three 6/6, subset 35/36 (checkpoint 4a71c3e1, declaration
    dcf34fc2). Goal misses: combined_180_+0.20 planar 0.0258, combined_180_-0.20
    planar 0.0310 + yaw 0.0607, gentle_315_+0.05 demand 0.0073, turn_-0.40 yaw
    0.1176, transition (reverse-diagonal segment).
  - u1350 55 (formal 2/4, quiet 2/2, omni 15; checkpoint 0f6fa004, declaration
    18d4429b); u1400 55 (combined 7/8, transition 1/1, omni 11; checkpoint
    672530d6, declaration 756a4c20).
  VERDICT: the three-second settle hands the hold a level pose - formal returns
  at no cost to the quiet cure. Every goal axis has now passed inside the
  v56/v58/v59 family; the conjunction needs one draw to align five whiskers.
- Round 24 (runner scratchpad/round24.sh): (a) tw_q4 = v59, seed 20260918 - the
  lottery at the proven recipe; (b) tw_q5 = destination_v60 (quiet_hold halved to
  1.5 on its 2.6x margin, committed b351cc13) seed 20260917 - hand the freed
  capacity back to turn/combined. Both anchor 0.006, u1400, judged at u1300 and
  u1350.
- Round 24 results (runner scratchpad/round24.sh), judged at u1300/u1350:
  - tw_q4 = v59 seed 20260918: 51/51 - a weak draw (gentle 10-11, omni 12-13,
    formal 1/4) that still proves turn 6/6 + transition 1/1 + quiet 2/2 coexist
    under the hold (u1300; checkpoints 04d0c223/92a0ab45, declarations
    decd2ef6/8aee4101).
  - tw_q5 = destination_v60 (hold halved): 51/54 - formal falls to 1/4 and gentle
    to 10-12 while turn stays missed; the freed capacity bought nothing back
    (checkpoints a3527914/517b3310, declarations 670aeb05/8f92e4f9; u1350 did post
    stop 10/10 + transition + quiet + combined 6/8). The FULL hold is the right
    dose: v60 DISPROVEN at half, PARKED.
  - tw_q3_u1300 (56: quiet+formal+omni+three, gentle 15) remains champion; the
    five-whisker alignment did not land in two more draws.
- Round 25 (runner scratchpad/round25.sh): (a) tw_q6 = v59 warm-started FROM
  tw_q3_u1400 (the anchor then pulls toward the pose-held, quiet-cured policy
  itself - the old self-anchor dead end preserved the fidget, but the fidget is
  now cured in the anchor target), +600 updates, seed 20260922, judged at +400
  and +600; (b) tw_q7 = v59 seed 20260919, judged u1300/u1350 - the lottery at
  n=4 on the proven recipe.

## Round 25 — the cured-policy warm-start sets a record; the seed lottery confirms it is the move (2026-11-06)

Design recap: tw_q6 = destination_v59 warm-started FROM the champion lineage
(init $C/tw_q3/checkpoint_update001400.pt, +600 updates, anchor 0.006 now pulling
toward the pose-held quiet-cured policy, seed 20260922), judged u400/u600.
tw_q7 = v59 fourth teacher-seed draw (init v29_terrain_v2.pt, u1400, seed
20260919), judged u1300/u1350. Control: tw_q3_u1300 (56/96, subset 35/36).

- **tw_q6_u600: 58/96, subset 36/36, omni 15/16 — NEW CHAMPION.** Checkpoint
  7fc3cd0c0ec06b068298aedc7e5e305ceafe5a01e205e45928d85bae26278b35, declaration
  ebfb7740fd7d2d82379029c0dbb6a968a6b484e35eb5de072b2f841767b485ac, outputs
  full96_tw_q6_u600/ terrain_eval_tw_q6_u600/ omni15_tw_q6_u600/.
  vs control: +combined_180_+0.20, +turn_-0.40, +full_program (transition 1/1);
  -combined_0deg_-0.20 (yaw 0.0633). gentle 15/16, turn 6/6, stop 9/10, quiet
  2/2, combined 6/8, formal 2/4, consistent-three 6/6 all (hill4, hill7,
  steps1.5). First subset-36 36/36 of the campaign.
  **Goal gap now four whiskers:** combined_0deg_-0.20 [yaw 0.0633],
  combined_180deg_-0.20 [planar 0.0291], gentle_arc_315deg_+0.05 [yaw 0.0606,
  demand 0.0075], omni _translate_0.15_315deg [demand 0.0062].
- tw_q6_u400: 56/96, subset 35/36, omni 15/16 (checkpoint 27efcb31, declaration
  955c69a2) — ties the old champion at +400; the chain was still improving at
  its +600 endpoint.
- tw_q7_u1300: 52/96, omni 13/16 (e575d914/ddb8e7ff); tw_q7_u1350: 54/96, omni
  16/16 (947d82a6/d27c859c). Fourth teacher draw, weak like 20260918/20260919 —
  seed 20260917 was the lucky one. The lottery is not the move; the warm-start
  chain is. (tw_q7_u1350's omni 16/16 shows the demand whisker is reachable.)

Verdicts: (1) Warm-starting v59 from the cured champion with the anchor pulling
toward it gains +2 and closes subset-36 — the old "self-anchor dead end" failed
because the anchor target carried the stand fidget; with the fidget cured in
the target, self-anchored polish works. (2) Teacher-seed lotteries at n=4 are
parked: 20260917 >> 20260918/19/22-from-teacher. (3) The chain ended while
still improving — iterate it.

Next (round 26, runner scratchpad/round26.sh): iterate the winning move one
rung up — (a) tw_q8 = v59 warm-started FROM tw_q6/checkpoint_update000600.pt
(+600, anchor 0.006, seed 20260923), judged u400/u600; (b) tw_q9 = same init
and recipe, second draw (seed 20260924), judged u400/u600 — two draws from the
winner instead of one draw from the winner and one wasted on the teacher.
Target: the four whiskers (two yaw at ~0.061-0.063 vs 0.06, two demand at
0.0062-0.0075 vs 0.005, one planar 0.0291).

## Round 26 — the chain holds 58 and shuffles whiskers; the s-curve starts to drift (2026-11-06)

Design recap: two v59 draws warm-started from the champion tw_q6_u600
(anchor 0.006, +600 updates): tw_q8 seed 20260923, tw_q9 seed 20260924,
each judged u400/u600. Control: tw_q6_u600 (58/96, subset 36/36, omni 15/16).

- tw_q8_u400: 56/96, subset 36/36, omni 14/16 (checkpoint f132e15f,
  declaration 56408025).
- tw_q8_u600: 58/96, subset 36/36, omni 15/16 (checkpoint 01901e3a,
  declaration dfb1c55d). +combined_0deg_-0.20, +forward_fast;
  -gentle_315_-0.05 [demand 0.0051], -full_program [s curve last-second planar].
- tw_q9_u400: 57/96, subset 36/36, **omni 16/16** (checkpoint 33e5a985,
  declaration 0dc1f9a3). +gentle_arc_315deg_+0.05; -gentle_135_+0.05 [0.0054],
  -full_program.
- tw_q9_u600: 58/96, subset 36/36, **omni 16/16** (checkpoint fb24e8ec,
  declaration 1c379a13). +forward_fast; -full_program. Gate gap:
  combined_0deg_-0.20 [planar 0.0265, yaw 0.0667], combined_180deg_-0.20
  [planar 0.0261], gentle_315_+0.05 [demand 0.0074], full_program [s curve].
  Terrain 6/6 all three on every arm.

Verdicts: (1) The chain is stable at 56-58 with subset 36/36 on every point —
warm-start polish does not collapse. (2) The frontier is whisker-shuffle:
combined_0_-0.20, gentle_315_+0.05, omni translate_315 and forward_fast each
passed on some point this round; the union of chain points now covers every
gate case, no single point covers all. (3) NEW DRIFT: full_program's s-curve
(last-second planar) fails on all four round-26 points while tw_q6_u600 passes
it — polishing whiskers is trading away the transition. (4) combined_180_-0.20
remains never-passed (planar 0.0261-0.0326 vs 0.025 across the whole chain).
Champion bookkeeping: tw_q6_u600 (58, transition pass) and tw_q9_u600 (58,
omni 16/16) are co-leaders; tw_q6_u600 stays the recorded control.

Next (round 27, runner scratchpad/round27.sh): A/B on how to keep what the
chain has while whiskers align — (a) tw_r1 = v59 from tw_q9_u600 (the
omni-perfect co-leader), anchor 0.006, +600, seed 20260925, judged u400/u600:
pure iteration; (b) tw_r2 = v59 from tw_q6_u600 (the transition-passing
champion), **anchor 0.012 toward that champion**, +600, seed 20260926, judged
u400/u600: tests whether a stronger pull to an already-good policy holds
full_program during polish. Distinct from the parked teacher-anchored 0.012
(that dose pulled toward the fidgety teacher; this pulls toward the cured
champion).

## Round 27 superseded; v61/v62 registered — the bound-margin round (2026-11-06)

Round 27 (tw_r1 iterate / tw_r2 chain-anchored 0.012) was stopped on the user's
direction before judging: tw_r1 trained to u600 (checkpoints preserved at
$C/tw_r1/, unjudged), tw_r2 never started. Superseded by a user-directed reward
design: "make the penalty for going over the limit for each of these start
slightly earlier."

Registered at revision 5ce55cf2 (tests 90/90 green): RewardConfig gains
bound_margin_weight (default 0, inert) and bound_margin_fraction (default 0.8).
When on, each control pays for three scorer-mirrored quantities once they cross
the fraction of their own case bound: planar velocity error vs max(0.025,
0.25*speed), yaw rate error vs max(0.06, 0.25*|yaw_cmd|), mean over-rating
torque fraction vs 0.005. Excess is normalized by the margin band, capped at
two band-widths, squared, averaged over the three families. Rationale: the
whisker misses sit at 104-130% of their bounds where the tracking kernels
(normalized by the command itself) are nearly flat; this term puts gradient
exactly in that band, starting before the limit. destination_v61 = v59 +
weight 1.0; destination_v62 = v59 + weight 3.0 (dose A/B).

Next (round 28, runner scratchpad/round28.sh): tw_s1 = v61 and tw_s2 = v62,
both warm-started from the co-leader tw_q9_u600 (58/96, omni 16/16; its gap is
exactly the whisker set the term targets), anchor 0.006, +600 updates, seeds
20260927/20260928, judged u400/u600 vs control tw_q9_u600.

User directive for the bound-margin series (2026-11-06): if the 80%-start term
harms the other benchmarks in round 28 (cases lost outside the targeted
whiskers — gentle/formal/transition/terrain/omni regressions vs tw_q9_u600),
the next round registers a 95% start (bound_margin_fraction 0.95, already
inside the validated range — a v63 on the less harmful dose) instead of
iterating the 80% recipe.

## Round 28 — the 80% bound margin aligns the never-passed case, costs the sprint; triple dose disproven (2026-11-06)

Design recap: tw_s1 = destination_v61 (bound margin weight 1.0, start 80%) and
tw_s2 = destination_v62 (weight 3.0), both warm-started from tw_q9_u600,
anchor 0.006, +600, seeds 20260927/20260928, judged u400/u600. Control:
tw_q9_u600 (58/96, subset 36/36, omni 16/16). Code at revision 5ce55cf2.

- **tw_s1_u600 (v61): 58/96, subset 36/36, omni 16/16** (checkpoint 69ff88fd,
  declaration 54834d0d). **+combined_180deg_-0.20 — the first pass of that case
  in the entire campaign** (planar had sat 0.026-0.033 vs 0.025 on every prior
  checkpoint). -forward_fast: the 0.30 m/s sprint now terminates (a fall).
  Goal gap three cases: combined_0deg_-0.20 [yaw 0.0654], gentle_315_+0.05
  [demand 0.0052 vs 0.005 — the narrowest whisker yet], full_program (s curve +
  now fixed-heading bend last-second). Terrain 6/6 all three.
- tw_s1_u400: 56/96 (3334dd10/965d9ebb), transitional.
- tw_s2_u400 (v62): 57/96, omni 16/16 (35a7f9c8/5c712105): +gentle_315_+0.05
  but -combined_90_-0.20, -forward_fast (fall).
- tw_s2_u600 (v62): 54/96, omni 14/16 (27b56423/5e2395bc): planar misses pile
  up across combined (0.0253-0.0282) — the triple price makes the gait
  over-cautious and the caution itself breaks the bounds. **v62 parked.**

Verdicts: (1) The bound-margin mechanism works at weight 1.0: it put gradient
in the whisker band and aligned the one structurally stuck case. (2) It harms
one non-gate benchmark — forward_fast pass -> fall on 3 of 4 arms (the demand
family presses the sprint's over-rating torque). The user's standing directive
triggers: next start at 95%. (3) Weight 3.0 inverts the effect (new planar
misses from over-caution) — parked on evidence.

v63 registered at revision e199b166 (tests 91/91): weight 1.0, fraction 0.95 —
the active zone hugs 95-105% of each bound, so behavior a full margin inside
the limit (the sprint included) pays nothing.

Next (round 29, runner scratchpad/round29.sh): (a) tw_s3 = v63 from tw_q9_u600
(seed 20260929) — the clean fraction A/B against round 28's v61 arm;
(b) tw_s4 = v63 from tw_s1_u600 (seed 20260930) — carries the aligned
combined_180 forward under the gentler band, judged vs tw_s1_u600. Both
anchor 0.006, +600, judged u400/u600.

## Round 29 — v63 (bound margin from 95 percent): fraction A/B and carry-forward (2026-11-06)

Runner scratchpad/round29.sh; destination_v63 (bound_margin_weight 1.0, bound_margin_fraction 0.95)
at e199b166; anchor 0.006, u600, judged u400/u600.

- tw_s3 (from tw_q9_u600, seed 20260929) — the clean fraction A/B against round 28's v61 arm:
  - tw_s3_u400: 57/96, subset 36/36, omni 16/16 (ckpt ec8150ce, decl cf375c5c). Gains
    combined_180deg_-0.20; loses combined_270deg_+0.20 (yaw 0.0603) and combined_90deg_-0.20
    (planar 0.0250) — both hairs.
  - tw_s3_u600: 59/96 — campaign best (prior best 58), subset 36/36, omni 15/16
    (ckpt af61b74f, decl abef7450). Gains gentle_arc_315deg_+0.05 (round 28's demand miss) with
    nothing lost vs tw_q9_u600, so forward_fast passes: the 95 percent start avoids the sprint
    fall the 80 percent start caused. Goal-scope misses: combined_0deg_-0.20 (yaw 0.0631),
    combined_180deg_-0.20 (planar 0.0276), full_program (s curve + fixed heading bend last
    second). Omni miss: translate_0.15_270deg (computed 0.0053).
- tw_s4 (from tw_s1_u600, seed 20260930) — the carry-forward under the gentler band, judged vs
  tw_s1_u600: regressed at both points. u400 55/96 omni 15/16 (ckpt 14ccf797, decl 08f62e2c;
  loses combined_180/90/270). u600 54/96 omni 13/16 (ckpt 5dc8619d, decl 485963a6; two
  native-capture omni failures). No forward_fast gain either, so the v61-shaped policy keeps the
  fall and relaxes back out of bounds when the margin pressure narrows — parked.

Findings: (a) the 95 percent fallback did what was asked — sprint intact, demand case gained,
new best; (b) combined_0deg_-0.20 sits at 105–111 percent of its 0.06 yaw bound, past the clamp
plateau of the 95 percent band ((err/bound − 0.95)/0.05 clamps at 2 above 105 percent), so the
margin term adds no gradient there and only the base yaw kernel pulls; (c) whiskers still
shuffle by seed (combined_270_+0.20 and combined_90_-0.20 flicker at hairs).

Next (round 30, runner scratchpad/round30.sh): chain v63 from both tw_s3 checkpoints —
(a) tw_s5 = v63 from tw_s3_u600 (seed 20261001), control tw_s3_u600: ride the campaign-best
basin; (b) tw_s6 = v63 from tw_s3_u400 (seed 20261002), control tw_s3_u400: the omni-16/16,
combined_180-passing basin. Judged u400/u600.

## Round 30 — the v63 chain is past its peak (2026-11-06)

Runner scratchpad/round30.sh; destination_v63 at e199b166; anchor 0.006, u600, judged u400/u600.

- tw_s5 (from tw_s3_u600, seed 20261001, control tw_s3_u600 at 59): u400 56/96 omni 15/16
  (ckpt e2a6a558, decl 138c8bd0), u600 55/96 omni 15/16 (ckpt 1316bfe9, decl a65c75c6).
  tw_s5_u600 loses forward_fast to a fall again.
- tw_s6 (from tw_s3_u400, seed 20261002, control tw_s3_u400 at 57): u400 56/96 omni 15/16
  (ckpt 5de63123, decl 1cc250a9), u600 55/96 omni 14/16 (ckpt b3b16048, decl f530dc19). Both
  points lose forward_fast.

Findings: (a) continued training under the margin term decays the sprint even from the 95
percent start — three of four arms fell on forward_fast — and shuffles the combined whiskers
without gaining; the v63 chain is parked at tw_s3_u600 (59/96, campaign best). (b) Rescoring
the full_program traces (score_transitions on the stored control_trace.npz, env 0) shows every
lead checkpoint fails the same two segments by hairs: s_curve last-second planar 0.0307-0.0338
and fixed_heading_bend 0.0278-0.0313 against 0.03, while every constant-command segment settles
to 0.008-0.024. Those are the only two segments whose command VARIES within the segment
(s_curve: yaw 0.15*sin(2*pi*t/10s) while planar holds [0.12, 0]; fixed_heading_bend: the 0.10
m/s translation direction rotates 90 degrees over 8 s, about 0.196 rad/s). Training commands
are piecewise constant everywhere (task.py _resample, destination_task.py _mixed_runs: walks,
turns, stands and arcs all hold a fixed vector 2-20 s), so the policy has never trained on a
rotating command vector — the same exposure gap the recorded v45 arc ruling named ("exposure
is the binding lever"), one step further. full_program blocks the goal whatever the whiskers
do, so it gets a designed fix, not a lottery.

Next (round 31): varying-command runs. A drawn share of the run mix's walks evolves per
control — half bend (the planar command rotates at a signed rate from (0.1, 0.2) rad/s),
half sweep (the yaw command follows one 0.15 rad/s sinusoid over 500 controls) — mirroring
the two failing shapes. The bootstrap wrapper detects switches by comparing command values
(rl/wrapper.py), so the task publishes the true resample flag; without that, every varying
row would bootstrap every control. destination_v64 = v63 + vary (margin kept),
destination_v65 = v59 + vary (the margin ablation). Arms: tw_s7 = v64 and tw_s8 = v65, both
from tw_s3_u600 (seeds 20261003/20261004), control tw_s3_u600, judged u400/u600; the check
that matters is full_program's two segments.

## Round 31 — vary exposure alone does not move the curving segments (2026-11-06)

Arms from tw_s3_u600 (control, 59/96): tw_s7 = destination_v64 (vary + margin, seed 20261003),
tw_s8 = destination_v65 (vary alone, seed 20261004), u600, anchor 0.006.

- tw_s7_u400: 57/96, subset 36/36, omni 15/16 — checkpoint d252f332…, declaration e020f7e3…
- tw_s7_u600: 54/96, subset 36/36, omni 16/16 — dcee48a8…, 37ab0b10…
- tw_s8_u400: 55/96, subset 36/36, omni 14/16 — d8ee4e8c…, 3e0f81aa…
- tw_s8_u600: 55/96, subset 36/36, omni 16/16 — 765ae3cf…, 751f61b7…
- Terrain consistent-three pass on all four; forward_fast intact on all four (one vary round
  harms no sprint, margin or not).

full_program rescore (score_transitions on the stored traces): control s_curve lp 0.0330 /
bend lp 0.0308; arms 0.0321–0.0358 / 0.0336–0.0373 — no arm improved either segment; three of
four worsened the bend. Decomposition of the last-second error (control and tw_s7_u600): the
error is a speed deficit along the commanded direction (along −0.029 to −0.035, lateral ≤ 0.010,
bend direction lag ≤ 2.8°, bend speed 0.066–0.072 against the commanded 0.100). The robot
tracks the rotating direction and under-speeds.

Why training pressure missed it: tracking_errors normalizes by commanded speed, so a 29 %
deficit on a 0.10 walk keeps 92 % of the kernel; the 95 % margin band there is 0.00125 wide,
so lp 0.030 sits ~5 band-widths past it, clamped flat with no gradient; the planar-error charge
(0.5, cap 2.0) is live but carries base weight on ~17 % of rows. Verdict: exposure is necessary
but not sufficient — the lever is price, not more data. tw_s7/tw_s8 parked.

Round 32: vary_tracking_scale — on rows whose command is currently varying, linear_tracking and
planar_error multiply by the scale (situation pricing per the v56 pattern; the ceiling validator
counts the scale, terminations 100/125 keep the tenfold dominance). Arms from tw_s3_u600:
tw_s9 = destination_v66 (scale 2.0, seed 20261005), tw_s10 = destination_v67 (scale 3.0, seed
20261006), u600, judged u400/u600 against control tw_s3_u600. The check that matters stays
full_program's two segments by trace rescore, with forward_fast and the subset watched.

## Round 32 — situation prices do not reach the behavior; the deficit is a command-in-motion mode (2026-11-06)

Arms from tw_s3_u600: tw_s9 = destination_v66 (2x planar prices on varying rows, seed 20261005),
tw_s10 = destination_v67 (3x, seed 20261006), u600, anchor 0.006.

- tw_s9_u400: 56/96, subset 36/36, omni 16/16 — checkpoint 5c956526…, declaration b1076065…
- tw_s9_u600: 53/96, subset 36/36, omni 15/16, forward_fast FELL — 060580bb…, d1eb1bef…
- tw_s10_u400: 55/96, subset 36/36, omni 16/16, forward_fast FELL — 0ed6b487…, 35af000c…
- tw_s10_u600: 55/96, subset 36/36, omni 16/16, forward_fast FELL — 5df91e95…, 3d3c85c4…
- Segments (rescored): s_curve 0.0311–0.0373, bend 0.0337–0.0370 vs control 0.0330/0.0308 —
  unmoved to worse; the sprint broke in three of four arms. Price escalation is dead: it harms
  globally and buys no curve speed. tw_s9/tw_s10 parked; tw_s9_u400 noted as the only arm that
  kept every gate family it entered with.

Mechanism findings (full_program trace, tw_s3_u600; per-second decomposition + per-segment
limiter/saturation + obs checks):
- Every constant-command segment tracks at command (forward/left/reverse/right 0.097–0.101 on
  0.10; diagonal_turn 0.100; gentle_arc 0.149 on 0.15; strafe_arc 0.101). Only s_curve (0.092
  on 0.12) and the bend (decaying 0.082→0.065 on 0.10) under-speed.
- Not the actuators: the bend runs 6.9 % of joint steps at the 0.040 rad / 20 ms cap vs 9.4 %
  (left) and 18.2 % (gentle_arc) in passing segments; 400 Hz saturation lower than passing arcs.
- Not the inputs: obs 210:213 == command == requested everywhere; the gait-clock features
  advance normally through both failing segments (task phase, not command age).
- Scene probe (scratchpad bend_probe.py, diagnostic): warmup at 0.10 → full speed; the control
  drops to ~0.083 the instant the command starts rotating, decays with accumulated rotation,
  and recovers to 0.100 within ~1.5 s of the rotation stopping, at a full 90° crab. Same for
  the s-curve at 0.12. The robot can crab at speed; it holds a learned ramp mode while the
  command vector is in motion.
- tw_s9_u400 (2x-priced vary training) is identical to the control on all four probe
  protocols, the training geometry included, while its whisker set did shift — the targeted
  pressure never reached this behavior.

Round 33 — anchor bracket: the remaining binder candidate is the 0.006 anchor to init. Arms
from tw_s3_u600, both destination_v66 (2x price, the dose that kept the sprint at u400):
tw_s11 = anchor 0.003 (seed 20261007), tw_s12 = anchor 0.0015 (seed 20261008), u600, judged
u400/u600 against control tw_s3_u600, plus the bend probe on each u400. If neither moves the
segments, PPO from this init cannot reach the mode under any tried pressure, and the designed
escalation is a command-derivative observation — an input change that needs the program lead's
approval, relayed by Julian.

## Round 33 — the anchor is not the binder; ROOT CAUSE: evaluation pins the ghost through varying commands (2026-11-06)

Arms from tw_s3_u600, both destination_v66 (2x price): tw_s11 = anchor 0.003 (seed 20261007),
tw_s12 = anchor 0.0015 (seed 20261008), u600.

- tw_s11_u400: 53/96, subset 36/36, omni 15/16, forward_fast FELL — 6b9902f5…, 60a36f09…
- tw_s11_u600: 54/96, subset 36/36, omni 14/16 — 75418b73…, 35e8443a…
- tw_s12_u400: 55/96, subset 36/36, omni 15/16 — 87b613b6…, 087c98ea…
- tw_s12_u600: 53/96, subset 36/36, omni 14/16 — 8d333db0…, f3e0255e…
- Segments: s_curve 0.0377/0.0378 (s11) and 0.0399/0.0448 (s12); bend 0.0362/0.0430 and
  0.0358/0.0362 — strictly worse, monotonically with anchor freedom. Bend probe on both u400s:
  slightly slower through rotation than the control, unchanged elsewhere. The anchor hypothesis
  is dead; every pressure mix on the vary motif (exposure, 2x/3x prices, anchors to 0.0015)
  leaves the segments unmoved or worse. tw_s11/tw_s12 parked.

ROOT CAUSE (code-level): evaluation rebuilds the policy's destination inputs with
DestinationTracker (src/locomotion/omniwalk/mdp/destination.py:70) and its ghost/terrain
subclasses, whose freshness rule is `episode start OR any command value change` — the
destination restarts at the body and the command age resets to zero. During s_curve and
fixed_heading_bend the command changes every control, so the policy runs both segments with the
ghost pinned at the body and age zero: permanently the "command just began" inputs, in which the
learned behavior is a gentle ramp. Training's task-side destination advances smoothly through a
varying run (no restart), so rounds 31–33 trained a situation evaluation never presents, and
strengthening ghost-chasing on curves degraded the pinned-ghost case — exactly the monotonic
worsening observed. The scene probe's command staircase pins the same way, and constant-command
segments (diagonal_turn, arcs) are untouched: every observation fits.

Round 34 — train on the evaluator's inputs: destination_v68 (= v65 + vary_ghost_matches_eval)
presents varying rows the pinned ghost (destination read at the body, age zero — the wrapper's
existing pending path, now driven by a task-published mask) while the reward keeps the advancing
destination, so curve speed must come from the command inputs. Verified: identical RNG stream to
v65, observation differences confined to the nine ghost columns on varying rows (gap and age
exactly zero there), steady rows untouched, 99/99 tests. Arms from tw_s3_u600: tw_s13 (seed
20261009), tw_s14 (seed 20261010), u600, anchor 0.006, judged u400/u600 vs control tw_s3_u600;
decisive readout stays the two segments' rescore, sprint and subset watched.

Escalation if v68 moves nothing: the clean fix is eval-side — a registered input variant whose
tracker treats a continuous evolution as one command. That is an input change: it needs the
program lead's approval, relayed by Julian.

Round 34 HELD before launch: Julian paused the loop (2026-11-06). destination_v68 is
registered, tested (99/99) and committed (see Git); no tw_s13/tw_s14 training was started.

## Native dispatch — the v59 spine replayed on Spark, queued in sequence (2026-11-07, Julian's instruction)

The queued v45_u1000 native run was removed first at Julian's instruction (runner PID 2814046
killed before any launch; GPU never freed during its whole queue; noted in v45_u1000.log; the
staged v45 pack remains on disk as the preserved fallback).

Queued in its place, one after the other (runner /home/julian/run_v59_chain.sh, detached PID
3270020, marker /home/julian/v59_chain.done, log v59_chain.log; GPU-etiquette queue loops per
stage, launch only at <= 25 percent, flock race re-queues):

- chain1 = destination_v59, u1400, seed 20260917, anchor 0.006, flat, init = the widened
  v29_terrain_v2 start (5b32fba4/543d2e63) — pack dest_v59_chain1_u1400_seed20260917_20261107_001.
- chain2 = destination_v59, u600, seed 20260922, anchor 0.006, init = chain1's
  run/standing/checkpoint_update001400.pt — pack dest_v59_chain2_u600_seed20260922_20261107_001.
- chain3 = destination_v59, u600, seed 20260924, anchor 0.006, init = chain2's
  run/standing/checkpoint_update000600.pt — pack dest_v59_chain3_u600_seed20260924_20261107_001.

These mirror the surrogate spine tw_q3 -> tw_q6 -> tw_q9 (the v59 champion's exact ancestry)
natively. All three packs share source freeze 0c021250… at commit 6caef53e, which adds
--anchor-weight to the native trainer and prepare (ActorAnchor from rl/ppo.py, installed after
the starting checkpoint loads, recorded in the declaration identity; spark/launch and the gates
untouched; 57 spark-side tests green). Recipe args otherwise match the surrogate spine exactly
(c36 recipe, lr max 1e-4, no lr floor, std 0.05, episode 32 s, 128 robots); the v45 binding's
input roots and admission mounts are reused unchanged, minus its terrain.

Chain2/chain3 shipped with placeholder init hashes; the runner fills the real ones from the
previous stage's finished checkpoint before pack approval (which refreezes the binding), so no
judged quantity is ever unpinned at launch. Local 30 h waiter armed on the chain marker.

## Ghost-blind round — review item 1 on the surrogate: the actor loses the ghost (2026-11-07, Julian's instruction)

The v29 PR review asks the ghost inputs off the actor (the nine ghost_v11 values derive from
the simulator's world pose; the robot's GPS is three orders of magnitude short of the judged
millimetre scale). destination_v69 implements the critic-only option: v59's reward unchanged,
new input terrain_v2_ghost_critic — the actor keeps proprioception, command, clock and the
terrain blocks (269 inputs), the critic keeps the full privileged 281. Wrapper, ppo-config,
evaluator tracker (TerrainV2GhostCriticTracker) and tests registered; 103 CPU tests green.

Narrowed start: starts/tw_s3_ghost_critic.pt from tw_s3_u600 (af61b74f) by
make_v59_ghost_critic_start.py — actor first-layer columns 233..241 and their Adam moments
dropped (278 -> 269), critic untouched, reproduction on ghost-zero inputs exact (0.00e+00);
derivation in starts/tw_s3_ghost_critic.json.

Arm gb1 = destination_v69 from that start, u2000, anchor 0.006, seed 20261009, 2-update smoke
green before launch. Judged at u600/u1200/u2000 flat-only (eval_checkpoint --skip-terrain +
omni15; terrain courses skipped at Julian's instruction — the terrain blocks stay in the
actor and read zero on flat ground). Control: tw_s3_u600 (59/96 with the ghost). The question
this round answers: how much of the tracking does the policy keep when it must infer its slip
from proprioception alone.

## Ghost-blind verdict and the native spine chain's completion (2026-11-07)

Ghost-blind gb1 (destination_v69 from starts/tw_s3_ghost_critic.pt, u2000, seed 20261009),
judged flat-only against tw_s3_u600 (59/96, omni 15/16 with the ghost):
- gb1_u600: 29/96, omni 4/16 — 79a42f32…, 2443f48a…
- gb1_u1200: 35/96, omni 7/16 — 27242a65…, b73ab9f1…
- gb1_u2000: 39/96, omni 8/16 — c5500f7f…, d34899a7… — still climbing at the end
  (+6 then +4 per segment). Families at u2000: stop 10/10 and quiet 2/2 HELD (run-end
  precision survives without the ghost); formal 2/4 (better than every comparator);
  translate 15/32, gentle_arc 7/16, turn 2/6, combined 0/8 — the walk-precision families
  carry the loss. Losing the world-pose gap costs ~20 cases at this depth; the curve says
  more updates buy more back. gb1 is the first robot-honest candidate of the lineage.

Native v59 spine chain COMPLETE on Spark (runner run_v59_chain.sh, all three stages exit 0,
hash patching worked at both seams):
- chain1 u1400 (seed 20260917): run/standing/checkpoint_update001400.pt 7e49b9eb…, declaration abfa8fc1…
- chain2 u600 from chain1 (seed 20260922): checkpoint 5c3f71d4…, declaration e5f47d78…
- chain3 u600 from chain2 (seed 20260924): checkpoint f62e4ce3… — final mean reward 2357,
  episode length 741, iteration ~6.7 s.
The native tw_q3 -> tw_q6 -> tw_q9 replication Julian queued is finished; the chain3
checkpoint is ready to copy down and judge on the surrogate suites against tw_q9_u600.

## Native chain judged, stage 1 (2026-11-07)

Checkpoints fetched with hashes verified against the runner's recorded values
(native_chain/: nc1_u1400 7e49b9eb…/abfa8fc1…, nc2 5c3f71d4…/e5f47d78…, nc3
f62e4ce3…/4fbb250c…).

- nc1_u1400 (native chain1 = tw_q3's recipe trained on Isaac): 55/96, terrain hill 4° 6/6,
  hill 7° 6/6, steps 1.5 cm 6/6 all f0, omni15 13/16 — outputs full96_nc1_u1400/,
  terrain_eval_nc1_u1400/. The surrogate twin tw_q3 recorded 56/96 (u1300, subset 35/36):
  the recipe reproduces natively to within one case, and the native checkpoint crosses all
  three terrain courses clean.
- nc2/nc3 judging re-ran after a runner path slip (zsh ${U#00} left u0600 folder names; the
  verified checkpoints were untouched); results recorded when they land.

## Native chain judged, stages 2 and 3 — full_program passes for the first time (2026-11-07)

- nc2_u600: 54/96, terrain 6/6 all three f0, omni 15/16 — 5c3f71d4…, e5f47d78…,
  outputs full96_nc2_u600/, terrain_eval_nc2_u600/. Surrogate twin tw_q6_u600: 58.
- nc3_u600: 56/96, subset 35/36, terrain 6/6 all three f0, omni 15/16 — f62e4ce3…,
  4fbb250c…, outputs full96_nc3_u600/, terrain_eval_nc3_u600/. Surrogate twin tw_q9_u600: 58.
  NATIVE BEST (prior native best 52, round 10), and **nc3 PASSES full_program** — the case no
  checkpoint, surrogate or native, ever passed, the target of rounds 31–33. The Isaac-trained
  policy clears both curving segments under the same pinned-ghost evaluation inputs. Costs vs
  the surrogate twin: forward_fast fell (the recurring native sprint weakness), turn_-0.40 at
  yaw 0.1012 (a hair from the never-passed chirality case), gentle_arc_0deg_-0.05.
- Chain summary: native 55 -> 54 -> 56 against surrogate 56 -> 58 -> 58 — the spine recipe,
  anchor and warm-start pattern all carry to Isaac within two cases per stage, and the native
  line brings full_program where the surrogate line never did.

## Odometry-ghost round — review item 1's replacement option built as v70 (2026-11-07, Julian's instruction)

destination_v70: v59's reward with the actor's nine ghost values computed from LEG ODOMETRY
(mdp/odometry.py) — stance-toe dead reckoning (body-frame toes are pure joint forward
kinematics; a planted toe is world-fixed) plus the gyro's yaw increment, with the ghost as the
integrated command restarting at the odometry pose, exactly the true destination's convention.
Same nine features, same scales, same layout as ghost_v11, so the actor stays 278 wide and
tw_s3_u600 warm-starts directly; the critic keeps the true privileged ghost. Input name
terrain_v2_odom; evaluation rebuilds the same inputs through OdomTerrainTracker. Measured
odometry drift on recorded runs: 0.3-6 mm/s (leg_odometry.py diagnostic), ~10x coarser than
the privileged gap, ~10x better than blind. This is the reviewer's own sketch for item 1; the
registration awaits James's sign-off, relayed by Julian; surrogate-first per policy.

105 CPU tests green (unit: translation and rotation odometry; wrapper: widths unchanged and
odometry features equal to the privileged ones at zero drift); 2-update train smoke green.
Arm od1 = destination_v70 from tw_s3_u600, u2000, anchor 0.006, seed 20261011, judged
u600/u1200/u2000 flat-only against tw_s3_u600 (59/96) and gb1 (39/96 blind).

## Odometry-ghost verdict — the robot-honest candidate recovers most of the ghost (2026-11-07)

A first judging crashed on an eval-only gap (toe telemetry empty before the first control;
fixed at cf9f7318 by sourcing odometry from current['toe_body'] through one shared observed()
accessor; the crashed outputs are preserved as *_failed_toe_keyerror). Clean verdicts, flat
judging against tw_s3_u600 (59/96, omni 15/16 privileged) and gb1 (39/96, omni 8/16 blind):

- od1_u600: 52/96, omni15 16/16 — aed5192a…, 7306f4ce…
- od1_u1200: 53/96, subset 36/36, omni15 16/16 PERFECT — 3bdfb005…, 7d01da6f…
- od1_u2000: 52/96 — d0c1880b…, b47eb712…

The leg-odometry ghost recovers 14 of the ~20 cases the blind ablation lost (53 vs 39 toward
59) and the subset holds 36/36 with a perfect omni — better than the privileged control's
15/16. The cost concentrates exactly where theory says: six combined whiskers at planar
0.025-0.031 (odometry drift noise eats the tightest margins), the gentle_arc_315 demand hair,
forward_fast's sprint, and full_program. destination_v70 stands as the robot-honest candidate
for review item 1, pending James's sign-off relayed by Julian.

## Diagnostic — foot contact removed from the odometry (2026-11-07)

Mechanism (leg_odometry.py with contact forced always-true): drift explodes from 6-118 mm to
0.7-2.9 m per 20 s run — the swinging legs vote against the stance and drag the estimate at
roughly walking speed; only turn-in-place survives (9 mm, symmetric swings cancel).

Judged (od1_u1200 re-evaluated with the eval tracker's observed() patched to the no-sensor
fallback; outputs full96_od1_u1200_nocontact/, omni15_od1_u1200_nocontact/ — diagnostic only,
no registered input changed): 8/96, omni15 0/16 — far below even the blind ablation's 39/96.
A confidently wrong ghost is much worse than none: the policy trusts a gap signal that is
fiction everywhere except turning in place. Foot contact is the keystone sensor of the
odometry input; even a crude current-spike contact estimate beats its absence by an order of
magnitude.

## Diagnostic — foot contact inferred from the gait schedule (2026-11-07)

No sensor at all: the eval odometry labels a toe stance unless its tripod's swing window says
it is swinging (phase fraction < 0.4 for tripod A, 0.5-0.9 for tripod B, speed_v1 phase; all
stance at zero command). Accuracy vs real contact 91-97% per case; the misses are the
touch-down/lift-off edges, so odometry drift grows to 91-265 mm (2-5x sensed, ~10x better
than no contact). Judged (od1_u1200, outputs full96_od1_u1200_gaitcontact/,
omni15_od1_u1200_gaitcontact/ — diagnostic only, no registered input changed): 47/96,
omni15 16/16. Contact-source ladder at matched checkpoint: privileged ghost 59 > sensed 53
(omni 16) > schedule-inferred 47 (omni 16) > blind 39 (omni 8) > always-stance 8 (omni 0).
Losses vs sensed are exactly six curving walks (five gentle_arc + combined_0deg_+0.20);
every straight translate still passes. A contact estimate with no hardware already recovers
most of the odometry's value; real sensing buys back the arcs.

## Dispatch — native v69/v70 arms from the chain3 model (2026-11-07)

At Julian's instruction, the two sim2real input arms replayed natively on Spark from the final
native chain model (nc3 u600, checkpoint f62e4ce330eb..., declaration 4fbb250ce8fb...), freeze
b669acdae5c3 at HEAD cf9f7318, sequential queue runner /home/julian/run_v6970.sh (GPU <= 25%
gate, never cancels others):
- dest_v69_ghost_critic_u2000_seed20261009_20261107_001 — destination_v69, init = narrowed
  nc3 start starts_v6970_20261107/nc3_ghost_critic.pt (036a64327ba2..., actor 278->269,
  ghost-zero reproduction 0.00e+00, declaration a852c0d69baa...), anchor 0.006, u2000.
- dest_v70_odom_u2000_seed20261011_20261107_001 — destination_v70 (actor ghost from leg
  odometry; native env supplies current.toe_body + tibia_floor_force_world_n), init = nc3
  verbatim, anchor 0.006, u2000.
Native mirror of the local gb1/od1 arms (seeds 20261009/20261011 kept). Experiment inputs are
the locally registered terrain_v2_ghost_critic / terrain_v2_odom; lineage sign-off by the
program lead is still pending and these runs add no acceptance claims. Judging on completion:
full-96 (flat) + omni15 at u600/u1200/u2000, unchanged gates.
